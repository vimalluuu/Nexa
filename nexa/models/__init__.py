"""
nexa/models/__init__.py
========================
Public API for Nexa's model sub-package.

Primary entry point:

    from nexa.models import NexaTransformer, ModelConfig

    config = ModelConfig.from_yaml("configs/model_config.yaml")
    model  = NexaTransformer(config)
    logits = model(input_ids)            # [B, S, vocab_size]

    # Or build a tiny model for testing/experiments:
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=256,
                         vocab_size=1000, max_seq_len=128)
    model = NexaTransformer(config)

Independence guarantee
----------------------
NexaTransformer is initialized from scratch. There is NO from_pretrained()
method. All weights are random at construction time (see transformer.py for
the initialization scheme) and must be trained using the Nexa training loop.
"""

from nexa.models.config import ModelConfig
from nexa.models.norm import RMSNorm, LayerNorm, build_norm
from nexa.models.rope import precompute_rope_freqs, apply_rope
from nexa.models.attention import MultiHeadAttention
from nexa.models.mlp import SwiGLU
from nexa.models.block import TransformerBlock
from nexa.models.transformer import NexaTransformer

__all__ = [
    # Primary API
    "NexaTransformer",
    "ModelConfig",
    # Components (exposed for testing, ablations, custom architectures)
    "RMSNorm",
    "LayerNorm",
    "build_norm",
    "precompute_rope_freqs",
    "apply_rope",
    "MultiHeadAttention",
    "SwiGLU",
    "TransformerBlock",
]
