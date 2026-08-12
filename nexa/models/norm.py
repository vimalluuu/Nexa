"""
nexa/models/norm.py
====================
Normalization layers for the Nexa Transformer.

Two implementations are provided:
    RMSNorm   — Root Mean Square normalization (default, modern)
    LayerNorm — Standard layer normalization (reference implementation)

Concept: Why normalize?
-----------------------
During a forward pass, activations can grow or shrink as they pass through
many layers. Normalization keeps the scale of activations stable, which:
  1. Prevents vanishing / exploding gradients.
  2. Allows larger learning rates.
  3. Makes the model less sensitive to weight initialization.

RMSNorm vs LayerNorm
--------------------
LayerNorm computes: x̂ = (x - μ) / σ × γ + β      (subtracts mean, normalizes)
RMSNorm computes:   x̂ = x / RMS(x) × γ             (just scales by RMS)

Empirically, the mean subtraction in LayerNorm is largely unnecessary.
RMSNorm is ~10% faster and simpler while achieving the same training stability.
We use Pre-RMSNorm (norm BEFORE the sublayer) for training stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    For an input tensor x of shape [..., d_model]:

        RMS(x) = sqrt(mean(x²) + ε)
        output = (x / RMS(x)) × γ

    where γ (weight) is a learned per-dimension scale parameter, initialized
    to ones (i.e., the identity transformation at the start of training).

    Note: Unlike LayerNorm, RMSNorm has NO bias parameter (β). The mean
    subtraction is omitted — the model learns to center activations implicitly.

    Parameters
    ----------
    d_model : int
        The last dimension of the input tensor to normalize.
    eps : float
        Numerical stability constant added before sqrt. Default: 1e-6.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # γ — learned scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMSNorm to the last dimension of x.

        Parameters
        ----------
        x : Tensor
            Any shape, with the d_model dimension last: e.g. [B, S, d_model].

        Returns
        -------
        Tensor
            Same shape as x, with unit RMS along the last dimension.
        """
        # Compute RMS along the last (d_model) dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        # Normalize and apply learned scale
        return (x / rms) * self.weight

    def __repr__(self) -> str:
        return f"RMSNorm(d_model={self.weight.shape[0]}, eps={self.eps})"


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (for reference and ablation experiments).

    Uses PyTorch's built-in F.layer_norm under the hood, which is
    heavily optimized (CUDA kernel, fused operations).

    Parameters
    ----------
    d_model : int
        The last dimension of the input tensor to normalize.
    eps : float
        Numerical stability constant. Default: 1e-5 (PyTorch default).
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.ln(x)

    def __repr__(self) -> str:
        return f"LayerNorm(d_model={self.ln.normalized_shape[0]}, eps={self.ln.eps})"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_norm(norm_type: str, d_model: int, eps: float = 1e-6) -> nn.Module:
    """
    Instantiate the appropriate normalization layer by name.

    Parameters
    ----------
    norm_type : str
        "rmsnorm" or "layernorm".
    d_model : int
        Dimension of the input's last axis.
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    nn.Module
        An RMSNorm or LayerNorm instance.

    Raises
    ------
    ValueError
        If norm_type is not recognized.
    """
    match norm_type.lower():
        case "rmsnorm":
            return RMSNorm(d_model, eps=eps)
        case "layernorm":
            return LayerNorm(d_model, eps=eps)
        case _:
            raise ValueError(
                f"Unknown norm_type '{norm_type}'. "
                f"Choose 'rmsnorm' or 'layernorm'."
            )
