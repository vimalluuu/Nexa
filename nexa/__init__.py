"""
nexa/__init__.py
=================
Root package for Nexa — an independent AI system built from scratch.

Independence Rule
------------------
Nexa does NOT use any pretrained AI model weights (OpenAI, Anthropic,
Google, Meta LLaMA, Mistral, Whisper, or similar). All neural network
weights are trained from scratch by us on legally permitted datasets.

Ordinary software libraries (PyTorch, NumPy, FastAPI, etc.) are allowed.

Version History
----------------
0.1.0  — Phase 1: Project setup (configs, logging, utilities)
0.2.0  — Phase 2: BPE Tokenizer (planned)
0.3.0  — Phase 3: Transformer model (planned)
0.4.0  — Phase 4: Training loop (planned)
0.5.0  — Phase 5: Text generation (planned)
0.6.0  — Phase 6: Chat UI + REST API (planned)
0.7.0  — Phase 7: Memory system (planned)
0.8.0  — Phase 8: Speech I/O (planned)
"""

__version__ = "0.5.0"
__author__ = "Nexa Team"

# Phase 1: utilities are immediately available
from nexa.utils import get_logger, load_config  # noqa: F401

__all__ = [
    "__version__",
    "__author__",
    "get_logger",
    "load_config",
]
