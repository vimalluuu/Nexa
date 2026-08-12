"""
nexa/models/config.py
======================
ModelConfig — the single source of truth for Nexa's Transformer architecture.

Every structural hyperparameter lives here. Nothing is hardcoded in the model
modules — they all receive a ModelConfig and query it.

Independence
------------
This module uses only Python stdlib (dataclasses, pathlib) and our own
load_config/OmegaConf utilities. No pretrained weights, no external AI libs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Nexa Transformer architecture configuration.

    All model modules (attention, MLP, transformer) receive a ModelConfig
    instance and use its fields exclusively. No magic numbers in model code.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary (must match the trained tokenizer).
    max_seq_len : int
        Maximum sequence length the model can process in one forward pass.
    d_model : int
        Embedding / hidden dimension. Must be divisible by n_heads.
    n_heads : int
        Number of attention heads. d_model must be divisible by n_heads.
    n_layers : int
        Number of stacked TransformerBlocks.
    d_ff : int
        Inner dimension of the feed-forward / SwiGLU network.
        Typically 4 × d_model for standard FFN, but used directly with SwiGLU.
    dropout : float
        Dropout probability applied in attention weights and after embedding.
        Set to 0.0 for inference.
    norm_type : str
        Which normalization to apply. Options: "rmsnorm" | "layernorm".
    norm_eps : float
        Small constant for numerical stability in normalization.
    pos_encoding : str
        Positional encoding type. Currently only "rotary" (RoPE) is implemented.
        "sinusoidal" and "learned" are planned for future phases.
    tie_embeddings : bool
        If True (default), the LM head weight is tied to the embedding weight.
        Reduces parameters and improves performance.
    pad_token_id : int
        Token ID used for padding (ignored in loss computation).
    bos_token_id : int
        Beginning-of-sequence token ID.
    eos_token_id : int
        End-of-sequence token ID.
    """

    # Vocabulary
    vocab_size: int = 32_000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Sequence
    max_seq_len: int = 1024

    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048

    # Regularization
    dropout: float = 0.1

    # Normalization
    norm_type: str = "rmsnorm"
    norm_eps: float = 1.0e-6

    # Positional encoding
    pos_encoding: str = "rotary"

    # Weight tying
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Validate all field values after construction."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}). "
                f"Current d_model / n_heads = {self.d_model / self.n_heads:.2f} (not an integer)."
            )
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {self.n_layers}.")
        if self.d_ff < 1:
            raise ValueError(f"d_ff must be ≥ 1, got {self.d_ff}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError(
                f"norm_type must be 'rmsnorm' or 'layernorm', got '{self.norm_type}'."
            )
        if self.pos_encoding not in ("rotary", "sinusoidal", "learned"):
            raise ValueError(
                f"pos_encoding must be 'rotary', 'sinusoidal', or 'learned', "
                f"got '{self.pos_encoding}'."
            )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def d_head(self) -> int:
        """
        Dimension per attention head.
        d_head = d_model / n_heads.
        Each head independently performs attention on d_head-dimensional vectors.
        """
        return self.d_model // self.n_heads

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """
        Load a ModelConfig from a YAML file (e.g., configs/model_config.yaml).

        The YAML file must have a top-level "model:" key whose children
        match the ModelConfig field names.

        Parameters
        ----------
        path : str or Path
            Path to the YAML config file.

        Returns
        -------
        ModelConfig

        Examples
        --------
        >>> cfg = ModelConfig.from_yaml("configs/model_config.yaml")
        >>> cfg.d_model
        512
        """
        from nexa.utils import load_config  # local import avoids circular deps

        raw = load_config(path)
        model_dict: dict = OmegaConf.to_container(raw.model, resolve=True)  # type: ignore[arg-type]
        return cls(**{k: v for k, v in model_dict.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Create a ModelConfig from a plain Python dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return (
            f"ModelConfig("
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, "
            f"d_ff={self.d_ff}, "
            f"vocab_size={self.vocab_size}, "
            f"max_seq_len={self.max_seq_len})"
        )
