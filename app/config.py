"""
app/config.py
==============
AppConfig — all configurable settings for the Nexa Chat server.

Everything that could be a CLI flag or environment variable lives here.
No hardcoded paths or magic numbers in app/ code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    """
    Configuration for the Nexa Chat server.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing nexa_final.pt and model_config.json.
        Created by scripts/train.py.
    tokenizer_dir : Path
        Directory containing the trained BPE tokenizer.
        Created by scripts/train.py or scripts/train_tokenizer.py.
    device : str
        Device for model inference. "auto" → cuda > mps > cpu.
    host : str
        Network interface to bind the server to.
        "127.0.0.1" = localhost only (default, safe).
        "0.0.0.0"   = all interfaces (expose on network).
    port : int
        Port number for the HTTP server.
    reload : bool
        Enable uvicorn hot-reload (development mode). Disable in production.

    Default Sampling Parameters
    ---------------------------
    These become the defaults when the client doesn't send sampling params.
    """

    # Model locations
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    tokenizer_dir:  Path = field(default_factory=lambda: Path("data/processed/tokenizer"))
    device:         str  = "auto"

    # Server
    host:   str  = "127.0.0.1"
    port:   int  = 8000
    reload: bool = False

    # Memory system
    memory_persist_path: Optional[Path] = field(default_factory=lambda: Path("data/memories.json"))
    max_memories:        int            = 200

    # Default sampling config
    temperature:        float = 0.8
    top_k:              int   = 0
    top_p:              float = 0.9
    repetition_penalty: float = 1.2
    max_new_tokens:     int   = 80

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.tokenizer_dir  = Path(self.tokenizer_dir)
        if self.memory_persist_path is not None:
            self.memory_persist_path = Path(self.memory_persist_path)

    @property
    def model_weights_path(self) -> Path:
        return self.checkpoint_dir / "nexa_final.pt"

    @property
    def model_config_path(self) -> Path:
        return self.checkpoint_dir / "model_config.json"

    @property
    def model_available(self) -> bool:
        """True if both weight file and config exist on disk."""
        return self.model_weights_path.exists() and self.model_config_path.exists()

    def as_sampling_dict(self) -> dict:
        """Return default sampling parameters as a plain dict (for API responses)."""
        return {
            "temperature":        self.temperature,
            "top_k":              self.top_k,
            "top_p":              self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens":     self.max_new_tokens,
        }


# Module-level singleton (created once, shared across app)
_default_config: AppConfig | None = None


def get_app_config() -> AppConfig:
    """Return the module-level AppConfig singleton."""
    global _default_config
    if _default_config is None:
        _default_config = AppConfig()
    return _default_config


def set_app_config(cfg: AppConfig) -> None:
    """Override the module-level config (used in tests and serve.py)."""
    global _default_config
    _default_config = cfg
