"""
nexa/utils/config_loader.py
============================
YAML-based configuration loader for Nexa.

Concept — Why a Config System?
--------------------------------
Hardcoding values like ``hidden_dim = 512`` inside source files is fragile:
- Changing a hyperparameter means editing code (risky, pollutes git history)
- Running an experiment with different settings requires code changes
- You can't easily compare two runs that used different settings

Instead, we load configuration from **YAML files** at startup.
YAML is human-readable, easy to edit, and supports comments — ideal for ML.

Architecture
-------------
We use a three-layer approach:

    1. **Load YAML → Python dict**   (via PyYAML)
    2. **Merge dicts into OmegaConf DictConfig**  (supports dot-access + CLI overrides)
    3. **Deserialize into a typed Python dataclass**  (via dacite — gives type safety)

This means:

    cfg = load_config("configs/model_config.yaml")
    print(cfg.model.d_model)   # 512 — dot-access, not cfg["model"]["d_model"]

And from the CLI you can override any value:

    python train.py model.learning_rate=1e-3 model.n_layers=12

Usage
------
    from nexa.utils import load_config
    cfg = load_config("configs/model_config.yaml")
    print(cfg.model.d_model)   # → 512
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import OmegaConf, DictConfig

from nexa.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    path: Union[str, Path],
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """
    Load a YAML config file and return an OmegaConf DictConfig.

    Supports dot-notation access (``cfg.model.d_model``), type checking,
    and CLI-style overrides.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file (absolute or relative to CWD).
    overrides : list of str, optional
        List of OmegaConf-style override strings, e.g.
        ``["model.d_model=256", "training.learning_rate=1e-4"]``.
        These are applied AFTER the file is loaded, overriding file values.

    Returns
    -------
    DictConfig
        An OmegaConf config object with dot-access and merge support.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    yaml.YAMLError
        If the YAML is malformed.

    Examples
    --------
    >>> cfg = load_config("configs/model_config.yaml")
    >>> cfg.model.d_model
    512
    >>> cfg = load_config("configs/model_config.yaml", overrides=["model.d_model=256"])
    >>> cfg.model.d_model
    256
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}\n"
            f"Make sure you are running from the project root (where configs/ lives)."
        )

    log.debug("Loading config from: %s", path)

    # Step 1: Parse raw YAML into a Python dict
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Step 2: Convert to OmegaConf DictConfig (enables dot-access + validation)
    cfg: DictConfig = OmegaConf.create(raw)

    # Step 3: Apply CLI-style overrides if provided
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
        log.debug("Applied %d config override(s): %s", len(overrides), overrides)

    log.debug("Config loaded: %s", OmegaConf.to_yaml(cfg, resolve=True))
    return cfg


def merge_configs(*paths: Union[str, Path]) -> DictConfig:
    """
    Load and merge multiple YAML config files left-to-right.

    Later files override earlier ones. This is useful for combining
    a base config with a run-specific override file.

    Parameters
    ----------
    *paths : str or Path
        One or more YAML file paths to merge in order.

    Returns
    -------
    DictConfig
        The merged configuration.

    Examples
    --------
    >>> cfg = merge_configs(
    ...     "configs/model_config.yaml",
    ...     "configs/train_config.yaml",
    ... )
    >>> cfg.model.d_model   # from model_config.yaml
    512
    >>> cfg.training.learning_rate   # from train_config.yaml
    0.0003
    """
    if not paths:
        raise ValueError("merge_configs() requires at least one path argument.")

    base = OmegaConf.create({})
    for path in paths:
        part = load_config(path)
        base = OmegaConf.merge(base, part)

    return base


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """
    Convert an OmegaConf DictConfig back to a plain Python dict.

    Useful for serialization (e.g., saving config alongside a checkpoint).

    Parameters
    ----------
    cfg : DictConfig
        OmegaConf config object.

    Returns
    -------
    dict
        Plain Python dictionary with all values resolved.
    """
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
