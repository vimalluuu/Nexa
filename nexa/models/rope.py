"""
nexa/models/rope.py
====================
Rotary Positional Embedding (RoPE) — implemented from scratch.

Concept
--------
The Transformer's attention mechanism is *permutation invariant*: it assigns
the same score to "cat sat" and "sat cat". We must inject position information.

Classical approach (sinusoidal / learned):
    Add a position vector to the token embedding.
    x_position = x_token + p_m

Problem: The attention score QK^T then mixes token content and position
in a way that doesn't cleanly separate. The model must learn to disentangle them.

RoPE approach (Su et al., 2022 — implemented here from scratch):
    Multiply Q and K by a position-dependent *rotation matrix* before
    computing their dot product.

The magical property:
    Q_m · K_n = f(q, m) · f(k, n) depends only on (q, k, m − n)
    — the content vectors AND their RELATIVE DISTANCE, not absolute positions.

This is exactly what we want: "how similar are token at position m and token
at position n, given they are (m−n) apart?"

Implementation
--------------
For each pair of dimensions (2i, 2i+1) in a d_head-dimensional head:

    θᵢ = 10000^(−2i / d_head)     (decreasing frequency for each dimension pair)

    At position m:
    [x₂ᵢ  ]  →  [cos(m·θᵢ) · x₂ᵢ  − sin(m·θᵢ) · x₂ᵢ₊₁]
    [x₂ᵢ₊₁]     [sin(m·θᵢ) · x₂ᵢ  + cos(m·θᵢ) · x₂ᵢ₊₁]

This is just a 2D rotation of each consecutive pair of dimensions.
Low-index dimensions rotate fast (small θ → frequent), high-index rotate slowly.
This is analogous to how binary numbers encode integers: low bits toggle fast.

No pretrained embeddings are loaded. We compute these values analytically.
"""

from __future__ import annotations

import torch
from torch import Tensor


def precompute_rope_freqs(
    d_head: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> tuple[Tensor, Tensor]:
    """
    Pre-compute the cosine and sine rotation tables for RoPE.

    These tables are computed ONCE at model construction and stored as
    non-parameter buffers (they move to the correct device automatically,
    but are not updated by the optimizer).

    Parameters
    ----------
    d_head : int
        Dimension per attention head. Must be even.
    max_seq_len : int
        Maximum sequence length the model will handle.
    base : float
        The base for the geometric progression of frequencies.
        Standard value is 10000 (from the original Transformer paper).
        Larger values slow down the rotation (good for long sequences).

    Returns
    -------
    cos : Tensor  shape [max_seq_len, d_head // 2]
        cos(m · θᵢ) for each position m and dimension pair i.
    sin : Tensor  shape [max_seq_len, d_head // 2]
        sin(m · θᵢ) for each position m and dimension pair i.

    Notes
    -----
    We store half the dimension (d_head // 2) because each rotation involves
    pairs of dimensions. The cos and sin are broadcast over all heads.
    """
    assert d_head % 2 == 0, f"d_head must be even for RoPE, got {d_head}."

    half = d_head // 2

    # θᵢ = base^(−2i / d_head)  for i = 0, 1, ..., half−1
    # Computed as: 1 / (base^(exponent)) where exponent ∈ [0, 1)
    exponents = torch.arange(0, d_head, 2, dtype=torch.float32) / d_head  # [half]
    inv_freqs = 1.0 / (base ** exponents)                                   # [half]

    # Position indices: 0, 1, 2, ..., max_seq_len − 1
    positions = torch.arange(max_seq_len, dtype=torch.float32)              # [S]

    # Outer product: freqs[m, i] = m · θᵢ
    freqs = torch.outer(positions, inv_freqs)    # [max_seq_len, half]

    return freqs.cos(), freqs.sin()              # [max_seq_len, half] each


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply Rotary Positional Embedding to a query or key tensor.

    This rotates each consecutive pair of dimensions (2i, 2i+1) by an angle
    that depends on the token's position in the sequence.

    Parameters
    ----------
    x : Tensor  shape [B, n_heads, S, d_head]
        The query or key tensor to rotate.
    cos : Tensor  shape [S, d_head // 2]
        Cosine table (first S rows of the precomputed buffer).
    sin : Tensor  shape [S, d_head // 2]
        Sine table (first S rows of the precomputed buffer).

    Returns
    -------
    Tensor  shape [B, n_heads, S, d_head]
        Rotated tensor with identical shape but position-encoded values.

    How the interleaving works
    --------------------------
    x[..., ::2]  selects dimensions 0, 2, 4, ...  → x₁ (first of each pair)
    x[..., 1::2] selects dimensions 1, 3, 5, ...  → x₂ (second of each pair)

    After rotation:
        new x₁ = x₁·cos − x₂·sin
        new x₂ = x₁·sin + x₂·cos

    torch.stack(..., dim=-1) then .flatten(-2) interleaves them back:
    [x₁[0], x₂[0], x₁[1], x₂[1], ...]  → original index order preserved.
    """
    # Split into even/odd-indexed dimensions
    x_even = x[..., ::2]    # [B, H, S, d_head // 2]
    x_odd  = x[..., 1::2]   # [B, H, S, d_head // 2]

    # Reshape cos/sin for broadcasting: [1, 1, S, d_head//2]
    cos = cos.unsqueeze(0).unsqueeze(0)    # [1, 1, S, half]
    sin = sin.unsqueeze(0).unsqueeze(0)    # [1, 1, S, half]

    # Apply 2D rotation to each pair
    rotated_even = x_even * cos - x_odd * sin   # new even dims
    rotated_odd  = x_even * sin + x_odd * cos   # new odd dims

    # Interleave: stack along last dim → [..., half, 2], then flatten
    # Result: [..., d_head] with (rotated_even[i], rotated_odd[i]) at indices (2i, 2i+1)
    out = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)

    return out
