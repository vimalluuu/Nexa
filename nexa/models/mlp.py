"""
nexa/models/mlp.py
===================
SwiGLU Feed-Forward Network — the MLP component of each Transformer block.

Concept
--------
Attention mixes information ACROSS positions (token i looks at token j).
The MLP processes each position INDEPENDENTLY (token i only talks to itself).
This is where the model "thinks" — applying nonlinear transformations to
the attended representations.

Why SwiGLU?
-----------
Classic FFN: x → Linear → ReLU → Linear
SwiGLU FFN: x → [Linear_gate → SiLU] ⊙ [Linear_value] → Linear

The gating mechanism (⊙ = element-wise multiplication) allows the network
to SELECTIVELY amplify or suppress features. The gate decides which parts
of the value vector to pass through, learned from data.

SiLU (Sigmoid Linear Unit), also called Swish:
    SiLU(x) = x · sigmoid(x)
    — Smooth, non-monotonic, allows negative values (unlike ReLU)
    — Self-gated: the magnitude of x controls its own passage

Three linear projections (all bias=False, following modern convention):
    gate_proj : d_model → d_ff      (the gate path)
    up_proj   : d_model → d_ff      (the value path)
    down_proj : d_ff   → d_model    (projects back to residual stream)

Parameters per layer: 3 × d_model × d_ff
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nexa.models.config import ModelConfig


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Forward pass:
        gate   = SiLU(gate_proj(x))     # what to amplify
        value  = up_proj(x)             # what to pass through
        hidden = gate ⊙ value           # gated activation
        output = down_proj(hidden)      # project back to d_model

    Parameters
    ----------
    config : ModelConfig
        The architecture configuration. Uses: d_model, d_ff, dropout.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU transformation to each position independently.

        Parameters
        ----------
        x : Tensor  shape [B, S, d_model]
            Input from the attention residual stream.

        Returns
        -------
        Tensor  shape [B, S, d_model]
            Nonlinearly transformed output, same shape as input.
        """
        # Gate: SiLU activation on the gate projection
        # SiLU(x) = x · σ(x)  — smooth, non-zero-gradient everywhere
        gate = F.silu(self.gate_proj(x))    # [B, S, d_ff]

        # Value: linear projection (no activation)
        value = self.up_proj(x)             # [B, S, d_ff]

        # Element-wise product: gate decides which values survive
        hidden = gate * value               # [B, S, d_ff]
        hidden = self.dropout(hidden)

        # Project back to model dimension
        return self.down_proj(hidden)       # [B, S, d_model]

    def __repr__(self) -> str:
        d_model = self.gate_proj.in_features
        d_ff    = self.gate_proj.out_features
        return f"SwiGLU(d_model={d_model}, d_ff={d_ff})"
