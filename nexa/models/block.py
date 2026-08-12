"""
nexa/models/block.py
=====================
TransformerBlock — one decoder layer of the Nexa Transformer.

Concept: Pre-Norm Residual Block
---------------------------------
A TransformerBlock applies two sublayers in sequence:
    1. Multi-Head Causal Self-Attention
    2. SwiGLU Feed-Forward Network

Each sublayer uses:
    a. Pre-normalization: normalize BEFORE the sublayer (more stable than post-norm)
    b. Residual connection: add the sublayer input back to its output

Pre-norm (what we use):         Post-norm (original Transformer):
    x = x + F(Norm(x))              x = Norm(x + F(x))

Pre-norm makes gradient flow cleaner and allows larger learning rates.
It's the standard in all modern large language models.

The full block computation:
    x ← x + Attention(RMSNorm(x))     # attention with residual
    x ← x + MLP(RMSNorm(x))           # feed-forward with residual
    return x
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from nexa.models.config import ModelConfig
from nexa.models.norm import build_norm
from nexa.models.attention import MultiHeadAttention
from nexa.models.mlp import SwiGLU


class TransformerBlock(nn.Module):
    """
    One decoder layer: Pre-RMSNorm + MultiHeadAttention + Pre-RMSNorm + SwiGLU.

    Input and output are both [B, S, d_model]. The residual stream passes
    through unchanged in dimensionality — this is intentional and critical.
    The block can only ADD to the residual stream, not replace it.

    This additive structure (residual connections) means that:
    - At initialization, each block is close to the identity function.
    - Gradient can flow straight through to early layers (no vanishing).
    - The model can "skip" layers by learning near-zero weights.

    Parameters
    ----------
    config : ModelConfig
        The architecture configuration. Passed to attention, mlp, and norm.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Pre-normalization before attention
        self.norm1 = build_norm(config.norm_type, config.d_model, config.norm_eps)

        # Multi-head causal self-attention (with RoPE)
        self.attn = MultiHeadAttention(config)

        # Pre-normalization before feed-forward
        self.norm2 = build_norm(config.norm_type, config.d_model, config.norm_eps)

        # SwiGLU feed-forward network
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply one transformer block to the residual stream.

        Parameters
        ----------
        x : Tensor  shape [B, S, d_model]
            The residual stream entering this block.

        Returns
        -------
        Tensor  shape [B, S, d_model]
            Updated residual stream after attention and MLP.
        """
        # Attention sublayer (pre-norm + residual)
        x = x + self.attn(self.norm1(x))

        # MLP sublayer (pre-norm + residual)
        x = x + self.mlp(self.norm2(x))

        return x

    def __repr__(self) -> str:
        return (
            f"TransformerBlock(\n"
            f"  norm1={self.norm1},\n"
            f"  attn={self.attn},\n"
            f"  norm2={self.norm2},\n"
            f"  mlp={self.mlp}\n)"
        )
