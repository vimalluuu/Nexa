"""
nexa/inference/sampler.py
==========================
Pure sampling functions and SamplingConfig for text generation.

Each function operates on a 1-D logit tensor [vocab_size] and returns
a modified logit tensor. They are pure (no side effects, no I/O) and
fully testable in isolation.

The sampling pipeline (applied in this order):
    logits → repetition_penalty → temperature → top_k → top_p → softmax → sample

Independence
------------
Pure PyTorch + Python math. No pretrained models, no external AI APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ===========================================================================
# SamplingConfig
# ===========================================================================

@dataclass
class SamplingConfig:
    """
    All parameters controlling how the next token is selected.

    Attributes
    ----------
    temperature : float
        Scales logits before softmax.
        1.0 = unchanged. < 1 = sharper. > 1 = flatter.
        Ignored when greedy=True.
    greedy : bool
        If True, always pick argmax. Deterministic. Overrides everything else.
    top_k : int
        0 = disabled. k > 0: only the top-k tokens by logit are kept;
        the rest are set to -∞ before softmax.
    top_p : float
        1.0 = disabled. p < 1: nucleus sampling — keep the smallest set of
        tokens whose cumulative probability sums to at least p.
    repetition_penalty : float
        1.0 = disabled. r > 1: tokens already in the context are penalised —
        their logits are divided by r (positive logits) or multiplied by r
        (negative logits), making them less likely to repeat.
    max_new_tokens : int
        Stop after generating this many new tokens (hard cap).
    eos_token_id : int, optional
        Stop as soon as this token is generated (e.g., <eos>).

    Common Presets
    --------------
    Greedy:       SamplingConfig(greedy=True)
    Conservative: SamplingConfig(temperature=0.7, top_k=40)
    Balanced:     SamplingConfig(temperature=0.8, top_p=0.9, repetition_penalty=1.2)
    Creative:     SamplingConfig(temperature=1.2, top_p=0.95)
    """

    temperature:        float          = 1.0
    greedy:             bool           = False
    top_k:              int            = 0
    top_p:              float          = 1.0
    repetition_penalty: float          = 1.0
    max_new_tokens:     int            = 100
    eos_token_id:       Optional[int]  = None

    def __post_init__(self) -> None:
        if not self.greedy and self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0.0 (got {self.temperature}). "
                "Use greedy=True for argmax decoding."
            )
        if self.top_k < 0:
            raise ValueError(f"top_k must be ≥ 0 (0 = disabled), got {self.top_k}.")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(
                f"top_p must be in (0, 1] (got {self.top_p}). "
                "Use 1.0 to disable nucleus filtering."
            )
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                f"repetition_penalty must be > 0.0 (got {self.repetition_penalty}). "
                "Use 1.0 to disable."
            )
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be ≥ 1 (got {self.max_new_tokens})."
            )

    @classmethod
    def greedy_config(cls, max_new_tokens: int = 100) -> "SamplingConfig":
        """Create a greedy (deterministic) config."""
        return cls(greedy=True, max_new_tokens=max_new_tokens)

    @classmethod
    def default_sampling(cls, max_new_tokens: int = 100) -> "SamplingConfig":
        """Balanced sampling: temperature=0.8, top_p=0.9, repetition_penalty=1.2."""
        return cls(
            temperature=0.8, top_p=0.9,
            repetition_penalty=1.2, max_new_tokens=max_new_tokens
        )


# ===========================================================================
# Pure sampling functions (each operates on a 1-D logit tensor)
# ===========================================================================

def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """
    Divide all logits by temperature before softmax.

    Effect on the resulting probability distribution:
        temperature → 0+   : approaches greedy (one-hot distribution)
        temperature = 1.0  : unchanged (identity transform)
        temperature → ∞    : approaches uniform distribution

    Why divide? softmax(x/T) is equivalent to raising each probability to
    the power 1/T and renormalising. Values < 1 sharpen; values > 1 flatten.

    Parameters
    ----------
    logits : Tensor  shape [vocab_size]
        Raw logits from the LM head.
    temperature : float
        Must be > 0.0.

    Returns
    -------
    Tensor  shape [vocab_size]
        Scaled logits.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0.0, got {temperature}.")
    return logits / temperature


def apply_top_k(logits: Tensor, k: int) -> Tensor:
    """
    Set all logits outside the top-K to -∞.

    Only the K tokens with the highest logits are kept in the sampling pool.
    All others become -∞ so that softmax assigns them probability 0.

    Intuition: if the model has 32,000 tokens, we're probably not interested
    in more than ~50 of them at any given step. Top-K hard-limits the range.

    Parameters
    ----------
    logits : Tensor  shape [vocab_size]
        Input logits (possibly temperature-scaled).
    k : int
        Number of top tokens to keep. 0 = disabled (return unchanged).

    Returns
    -------
    Tensor  shape [vocab_size]
        Logits with non-top-K values set to -∞.

    Notes
    -----
    If multiple tokens tie for the K-th position, all ties are kept
    (so the actual number of surviving tokens may be slightly > K).
    """
    if k <= 0 or k >= logits.size(-1):
        return logits   # disabled or k covers entire vocabulary

    # Find the minimum logit value among the top-K
    top_k_vals, _ = torch.topk(logits, k=k)
    min_top_k = top_k_vals[..., -1]    # scalar: the K-th largest value

    # Zero out (set to -∞) everything below the threshold
    return logits.masked_fill(logits < min_top_k, float("-inf"))


def apply_top_p(logits: Tensor, p: float) -> Tensor:
    """
    Nucleus (top-p) sampling filter: keep the smallest set of tokens
    whose cumulative probability mass sums to at least p.

    Unlike top-K, the nucleus size adapts dynamically:
    - When the model is confident (peaked distribution), only a few tokens
      are in the nucleus.
    - When the model is uncertain (flat distribution), many tokens are included.

    Algorithm
    ---------
    1. Sort logits descending → sorted_logits, sorted_indices.
    2. Convert sorted logits → sorted probabilities via softmax.
    3. Compute cumulative sum of sorted probabilities.
    4. Shift cumsum right by 1 (= cumsum minus current prob).
       This ensures the token that pushes cumsum over p is KEPT.
    5. Mask all tokens where shifted_cumsum ≥ p → set their logits to -∞.
    6. Restore original vocabulary ordering.

    Parameters
    ----------
    logits : Tensor  shape [vocab_size]
        Input logits (possibly temperature-scaled).
    p : float
        Cumulative probability threshold ∈ (0, 1].
        1.0 = disabled (return unchanged).

    Returns
    -------
    Tensor  shape [vocab_size]
        Logits with non-nucleus tokens set to -∞.

    Examples
    --------
    probs sorted desc: [0.50, 0.30, 0.15, 0.04, 0.01]
    cumsum:            [0.50, 0.80, 0.95, 0.99, 1.00]
    shifted cumsum:    [0.00, 0.50, 0.80, 0.95, 0.99]
    with p = 0.85:     [keep, keep, keep, -∞,   -∞  ]  ← nucleus sums to 0.95
    """
    if p >= 1.0:
        return logits   # disabled

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Convert to probabilities in sorted order (for cumsum)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where (cumsum - current_prob) >= p
    # The shift ensures at least 1 token is always kept
    remove_mask = (cumulative_probs - sorted_probs) >= p

    # Apply -∞ to filtered positions (in sorted order)
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

    # Restore original vocabulary ordering
    result = torch.empty_like(logits)
    result.scatter_(0, sorted_indices, sorted_logits)
    return result


def apply_repetition_penalty(
    logits:        Tensor,
    generated_ids: Tensor,
    penalty:       float,
) -> Tensor:
    """
    Penalise tokens that have already appeared in the generated sequence.

    Based on: "CTRL: A Conditional Transformer Language Model for
    Controllable Generation" (Keskar et al., 2019).

    For each unique token ID in `generated_ids`:
        logit > 0  →  logit ← logit / penalty   (reduce positive logit)
        logit < 0  →  logit ← logit × penalty   (push negative logit lower)

    Net effect: the probability of repeating a previously seen token is
    reduced by a factor of approximately `penalty`. With penalty=1.3, a
    previously generated token becomes ~30% less likely.

    Parameters
    ----------
    logits : Tensor  shape [vocab_size]
        Raw logits from the LM head (before temperature scaling).
    generated_ids : Tensor  shape [S]
        All token IDs generated so far (including the prompt).
        Unique IDs are extracted automatically.
    penalty : float
        Must be > 0.0. 1.0 = disabled (no penalty).
        Typical values: 1.1 to 1.5.

    Returns
    -------
    Tensor  shape [vocab_size]
        Modified logits.

    Notes
    -----
    Repetition penalty is applied BEFORE temperature so that the penalty
    operates in the raw logit space, not the temperature-scaled space.
    """
    if penalty == 1.0:
        return logits   # disabled

    # Clamp penalty to valid range (must be > 0)
    penalty = max(penalty, 1e-8)

    # Get the unique token IDs that have been seen so far
    unique_ids = generated_ids.unique()

    # Extract their current logits
    scores = logits[unique_ids]

    # Apply penalty:
    #   positive logit → divide (become less positive → less probable)
    #   negative logit → multiply (become more negative → even less probable)
    penalised = torch.where(
        scores > 0,
        scores / penalty,
        scores * penalty,
    )

    # Write penalised scores back
    result = logits.clone()
    result[unique_ids] = penalised
    return result


# ===========================================================================
# Unified sampling entry point
# ===========================================================================

def sample_next_token(
    logits:        Tensor,
    generated_ids: Tensor,
    config:        SamplingConfig,
) -> int:
    """
    Select the next token ID from logits using the specified sampling strategy.

    This is the central decision point of the generation loop. It applies
    all configured transformations in the correct order and returns one
    integer token ID.

    Pipeline (in order):
        1. greedy shortcut (if config.greedy)
        2. repetition_penalty
        3. temperature
        4. top_k
        5. top_p
        6. softmax → multinomial sample

    Parameters
    ----------
    logits : Tensor  shape [vocab_size]
        Raw logits from the LM head at the last position.
    generated_ids : Tensor  shape [S]
        All token IDs generated so far (prompt + previously generated).
        Used for repetition penalty.
    config : SamplingConfig
        The sampling hyperparameters.

    Returns
    -------
    int
        The selected next token ID.
    """
    # ----- Greedy shortcut (no sampling at all) -----
    if config.greedy:
        return int(logits.argmax().item())

    # ----- Step 1: Repetition penalty (in raw logit space) -----
    if config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_ids, config.repetition_penalty)

    # ----- Step 2: Temperature scaling -----
    logits = apply_temperature(logits, config.temperature)

    # ----- Step 3: Top-K filter -----
    if config.top_k > 0:
        logits = apply_top_k(logits, config.top_k)

    # ----- Step 4: Top-P (nucleus) filter -----
    if config.top_p < 1.0:
        logits = apply_top_p(logits, config.top_p)

    # ----- Step 5: Sample from the final distribution -----
    probs = F.softmax(logits, dim=-1)

    # Safety: if all filters collapsed to zero probability, fall back to greedy
    if not torch.isfinite(probs).any() or probs.sum() <= 0:
        return int(logits.nan_to_num(neginf=float("-inf")).argmax().item())

    return int(torch.multinomial(probs, num_samples=1).item())
