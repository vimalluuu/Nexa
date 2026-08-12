"""
nexa/models/attention.py
=========================
Multi-Head Causal Self-Attention with Rotary Positional Embedding.

Concept
--------
Self-attention answers: "Given every token in this sequence, how should I
update the representation of each token by mixing information from others?"

Multi-head attention runs this operation H times in parallel, with independent
weight matrices (W_q, W_k, W_v, W_out) for each head. Each head learns to
attend to different types of relationships:
  - Head 0: might learn syntactic dependencies
  - Head 1: might track coreference (pronouns ↔ nouns)
  - Head 2: might capture local n-gram patterns
  … etc.

The "causal" constraint: position i may ONLY attend to positions j ≤ i.
This is enforced by adding −∞ to attention scores for j > i before softmax,
making those weights vanish (softmax(−∞) → 0).

Implementation Details
-----------------------
Q, K, V projections: Linear(d_model, d_model, bias=False)  — modern models omit bias
RoPE: Applied to Q and K (NOT V) — rotates token representations by position
Scaling: divide scores by √d_head to prevent the dot product from growing too large
Weight tying with output: out_proj recombines the heads back to d_model

Parameters (per layer)
----------------------
W_q:  d_model × d_model      (= n_heads × d_head × d_model, packed)
W_k:  d_model × d_model
W_v:  d_model × d_model
W_o:  d_model × d_model
Total: 4 × d_model²
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from nexa.models.config import ModelConfig
from nexa.models.rope import precompute_rope_freqs, apply_rope


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with Rotary Positional Embedding.

    Architecture:
        1. Project input to Q, K, V (each d_model-dimensional).
        2. Reshape to [B, n_heads, S, d_head].
        3. Apply RoPE to Q and K.
        4. Compute scaled dot-product attention with causal mask.
        5. Concatenate heads and project back to d_model.

    All linear projections have NO bias — this matches the convention of modern
    large language models and slightly reduces parameter count.

    Parameters
    ----------
    config : ModelConfig
        The architecture configuration. Uses: d_model, n_heads, max_seq_len, dropout.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head  = config.d_head    # = d_model // n_heads

        # Q, K, V projections — all the same shape, all no-bias
        self.q_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)

        # Pre-compute RoPE tables and register as non-parameter buffers.
        # register_buffer: moves to device with .to(), saved in state_dict,
        # but NOT updated by the optimizer.
        cos, sin = precompute_rope_freqs(self.d_head, config.max_seq_len)
        self.register_buffer("rope_cos", cos)    # [max_seq_len, d_head // 2]
        self.register_buffer("rope_sin", sin)    # [max_seq_len, d_head // 2]

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute multi-head causal self-attention.

        Parameters
        ----------
        x : Tensor  shape [B, S, d_model]
            Input sequence. B = batch, S = sequence length, D = d_model.

        Returns
        -------
        Tensor  shape [B, S, d_model]
            Attention output — each position's representation updated by
            attending to all PREVIOUS positions (causal) and itself.
        """
        B, S, D = x.shape

        # ------------------------------------------------------------------
        # Step 1: Project to Q, K, V
        # ------------------------------------------------------------------
        q = self.q_proj(x)    # [B, S, D]
        k = self.k_proj(x)    # [B, S, D]
        v = self.v_proj(x)    # [B, S, D]

        # ------------------------------------------------------------------
        # Step 2: Split into heads → [B, n_heads, S, d_head]
        # view() splits D into (n_heads, d_head), transpose moves H before S
        # ------------------------------------------------------------------
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)    # [B, H, S, d_head]
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # ------------------------------------------------------------------
        # Step 3: Apply RoPE to Q and K (position encoding via rotation)
        # V is NOT rotated — it carries content, not positional queries/keys.
        # ------------------------------------------------------------------
        cos = self.rope_cos[:S]    # [S, d_head // 2]  — slice to actual seq len
        sin = self.rope_sin[:S]    # [S, d_head // 2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ------------------------------------------------------------------
        # Step 4: Scaled dot-product attention with causal mask
        #
        # scores[b, h, i, j] = Q[b,h,i] · K[b,h,j] / √d_head
        #                     = "how much should position i attend to position j?"
        # ------------------------------------------------------------------
        scale = self.d_head ** -0.5                               # 1 / √d_head
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale    # [B, H, S, S]

        # Causal mask: positions j > i get −∞, which softmax maps to 0.
        # triu(diagonal=1) keeps the strict upper triangle (j > i positions).
        causal_mask = torch.full(
            (S, S), float("-inf"), device=x.device, dtype=x.dtype
        ).triu(diagonal=1)                                         # [S, S]
        scores = scores + causal_mask                              # [B, H, S, S]

        # Softmax over the key dimension (dim=-1 = last S)
        attn_weights = torch.softmax(scores, dim=-1)               # [B, H, S, S]
        attn_weights = self.attn_dropout(attn_weights)

        # ------------------------------------------------------------------
        # Step 5: Weighted combination of value vectors
        # ------------------------------------------------------------------
        out = torch.matmul(attn_weights, v)    # [B, H, S, d_head]

        # ------------------------------------------------------------------
        # Step 6: Re-combine heads → [B, S, d_model]
        # transpose(1,2) swaps H and S back, .contiguous() makes memory layout
        # compatible with .view(), then we merge (H, d_head) → D.
        # ------------------------------------------------------------------
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        # Final linear projection mixes information across heads
        return self.out_proj(out)    # [B, S, d_model]

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention("
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"d_head={self.d_head})"
        )
