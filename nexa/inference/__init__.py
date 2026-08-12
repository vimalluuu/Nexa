"""
nexa/inference/__init__.py
===========================
Public API for Nexa's inference sub-package.

    from nexa.inference import Generator, SamplingConfig, GenerationResult
    from nexa.inference import (
        apply_temperature, apply_top_k, apply_top_p,
        apply_repetition_penalty, sample_next_token,
    )

Independence
------------
All inference uses only Nexa's own trained weights (Phase 4).
No pretrained AI models, no external APIs.
"""

from nexa.inference.sampler import (
    SamplingConfig,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
    sample_next_token,
)
from nexa.inference.generator import Generator, GenerationResult

__all__ = [
    # Primary API
    "Generator",
    "SamplingConfig",
    "GenerationResult",
    # Pure sampling functions (exposed for testing and ablations)
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "apply_repetition_penalty",
    "sample_next_token",
]
