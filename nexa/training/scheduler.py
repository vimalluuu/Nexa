"""
nexa/training/scheduler.py
===========================
Cosine learning rate schedule with linear warmup — implemented from scratch.

Concept: Why Schedule the Learning Rate?
-----------------------------------------
A fixed learning rate is suboptimal for two reasons:

1. **Early instability (too-high LR at step 0)**:
   At initialization, random weights produce large, chaotic gradients.
   A large learning rate at step 0 can destroy the careful initialisation
   and send weights to bad regions of the loss landscape.

   Fix: **Linear warmup** — ramp LR from 0 to max_lr over the first
   `warmup_steps`, letting the model find a stable region first.

2. **Late oscillation (too-high LR at convergence)**:
   As the model approaches a good minimum, a large LR causes the weights
   to "bounce" around the minimum rather than settle into it.

   Fix: **Cosine decay** — smoothly reduce LR from max_lr to min_lr.
   The cosine shape spends more time near max_lr (fast progress early),
   then transitions slowly toward min_lr (fine-grained convergence).

Schedule Formula
-----------------
Given step t, warmup_steps W, max_steps T, max_lr α, min_lr α_min:

    if t < W:
        lr(t) = α × (t+1) / W                             ← linear warmup

    elif t >= T:
        lr(t) = α_min                                     ← plateau

    else:
        progress = (t − W) / (T − W)                      ← in [0, 1]
        lr(t) = α_min + 0.5(α − α_min)(1 + cos(π × progress))  ← cosine

Visual:
    LR
    ↑
    α  |         ╱╲
       |        ╱  ╲___
    α_min|╱          ───
         +--+---------+--→ step
            W         T

Independence
------------
Pure Python + math stdlib. No external dependencies.
"""

from __future__ import annotations

import math

import torch.optim as optim


# ---------------------------------------------------------------------------
# Stateless schedule function (pure, testable)
# ---------------------------------------------------------------------------

def cosine_warmup_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:
    """
    Compute the learning rate at a given training step.

    This is a pure function with no side effects — it simply maps an integer
    step index to a float learning rate. The Trainer calls this every step
    to set `optimizer.param_groups[*]['lr']`.

    Parameters
    ----------
    step : int
        Current training step (0-indexed).
    warmup_steps : int
        Number of warmup steps. LR grows linearly from 0 → max_lr.
    max_steps : int
        Total training steps. LR reaches min_lr at or after this point.
    max_lr : float
        Peak learning rate, reached at the end of warmup.
    min_lr : float
        Minimum learning rate, held after max_steps.
        Typically 1/10 of max_lr (e.g., 3e-5 when max_lr=3e-4).

    Returns
    -------
    float
        The learning rate to use at this step.

    Examples
    --------
    >>> # Warmup phase: growing from 0 to max_lr
    >>> cosine_warmup_lr(0, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
    0.0001
    >>> cosine_warmup_lr(10, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
    0.001
    >>> # Midpoint of cosine decay: roughly halfway between max_lr and min_lr
    >>> cosine_warmup_lr(55, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)  # approx
    0.00055...
    """
    if warmup_steps > 0 and step < warmup_steps:
        # Linear warmup: step 0 → lr = max_lr/warmup_steps (not 0, to avoid zero LR)
        return max_lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    # Cosine decay phase
    # progress ∈ [0, 1]: how far we are through the decay phase
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    # Cosine annealing: 1 → 0 as progress goes 0 → 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Stateful scheduler class (manages optimizer.param_groups automatically)
# ---------------------------------------------------------------------------

class CosineWarmupScheduler:
    """
    Stateful wrapper around cosine_warmup_lr.

    Manages the internal step counter and applies the computed LR to
    ALL param groups of the optimizer on each call to `.step()`.

    We implement this manually (rather than using torch.optim.lr_scheduler)
    so the schedule logic is completely transparent and testable.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose param groups will be updated.
    warmup_steps : int
        Number of linear warmup steps.
    max_steps : int
        Total training steps (schedule ends here).
    max_lr : float
        Peak learning rate.
    min_lr : float
        Minimum learning rate (floor of cosine decay).

    Usage
    -----
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=100, max_steps=1000,
                                       max_lr=3e-4, min_lr=3e-5)
    # In training loop:
    scheduler.step()      # update LR for this step, then increment counter
    print(scheduler.current_lr)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be ≥ 1, got {max_steps}.")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be ≥ 0, got {warmup_steps}.")
        if warmup_steps >= max_steps and max_steps > 1:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be < max_steps ({max_steps})."
            )
        if min_lr < 0 or max_lr <= 0:
            raise ValueError("LR values must be positive (max_lr > 0, min_lr ≥ 0).")

        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self._step        = 0   # steps completed so far

        # Apply initial LR immediately (step 0)
        self._apply_lr(self._compute_lr())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self) -> float:
        """
        Compute LR for the current step, apply it, advance the counter.

        Call this AFTER optimizer.step() in the training loop.

        Returns
        -------
        float
            The learning rate that was applied this step.
        """
        lr = self._compute_lr()
        self._apply_lr(lr)
        self._step += 1
        return lr

    @property
    def current_lr(self) -> float:
        """LR that will be applied on the next call to step()."""
        return self._compute_lr()

    @property
    def steps_completed(self) -> int:
        """Number of optimizer steps taken so far."""
        return self._step

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_lr(self) -> float:
        return cosine_warmup_lr(
            self._step,
            self.warmup_steps,
            self.max_steps,
            self.max_lr,
            self.min_lr,
        )

    def _apply_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def __repr__(self) -> str:
        return (
            f"CosineWarmupScheduler("
            f"step={self._step}/{self.max_steps}, "
            f"lr={self.current_lr:.2e}, "
            f"warmup={self.warmup_steps})"
        )
