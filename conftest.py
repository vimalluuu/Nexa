"""
conftest.py — pytest configuration and shared fixtures
=======================================================
This file is automatically loaded by pytest before any test run.

It does two important things:
1. Adds the project root to sys.path so `import nexa` works even if the
   package isn't installed (belt-and-suspenders alongside pip install -e .)
2. Provides shared fixtures available to ALL test files without importing.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
# pytest is typically run from the project root (where conftest.py lives).
# This line makes `import nexa` work reliably regardless of how pytest
# was invoked.
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Shared fixtures (available to all test files automatically)
# ---------------------------------------------------------------------------
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """Return the path to the configs/ directory."""
    return project_root / "configs"


@pytest.fixture(scope="session")
def model_config_path(configs_dir: Path) -> Path:
    """Path to configs/model_config.yaml."""
    path = configs_dir / "model_config.yaml"
    assert path.exists(), f"model_config.yaml not found at {path}"
    return path


@pytest.fixture(scope="session")
def train_config_path(configs_dir: Path) -> Path:
    """Path to configs/train_config.yaml."""
    path = configs_dir / "train_config.yaml"
    assert path.exists(), f"train_config.yaml not found at {path}"
    return path


@pytest.fixture(scope="session")
def inference_config_path(configs_dir: Path) -> Path:
    """Path to configs/inference_config.yaml."""
    path = configs_dir / "inference_config.yaml"
    assert path.exists(), f"inference_config.yaml not found at {path}"
    return path
