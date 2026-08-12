"""
tests/test_utils.py
====================
Phase 1 verification tests — logger and config loader.

These tests confirm that the foundational utilities installed by Phase 1
are importable and behave correctly, without touching any ML components.

Run with:
    pytest tests/test_utils.py -v
"""

import logging
import pytest
from pathlib import Path


# =============================================================================
# Logger tests
# =============================================================================

class TestGetLogger:
    """Tests for nexa.utils.logger.get_logger()"""

    def test_import(self):
        """get_logger is importable from the nexa.utils public API."""
        from nexa.utils import get_logger  # noqa: F401
        assert callable(get_logger)

    def test_returns_logger_instance(self):
        """get_logger() returns a standard logging.Logger."""
        from nexa.utils import get_logger
        log = get_logger("test.basic", log_to_file=False)
        assert isinstance(log, logging.Logger)

    def test_named_logger(self):
        """Logger carries the name it was created with."""
        from nexa.utils import get_logger
        log = get_logger("test.named", log_to_file=False)
        assert log.name == "test.named"

    def test_idempotent(self):
        """Calling get_logger() twice with the same name returns the same object."""
        from nexa.utils import get_logger
        log1 = get_logger("test.idem", log_to_file=False)
        log2 = get_logger("test.idem", log_to_file=False)
        assert log1 is log2

    def test_has_handlers(self):
        """Logger has at least one handler attached (the Rich console handler)."""
        from nexa.utils import get_logger
        log = get_logger("test.handlers", log_to_file=False)
        assert len(log.handlers) >= 1

    def test_level_is_info_by_default(self):
        """Default log level is INFO."""
        from nexa.utils import get_logger
        log = get_logger("test.level", log_to_file=False)
        assert log.level == logging.INFO

    def test_set_global_level(self):
        """set_global_level() changes level on all configured loggers."""
        from nexa.utils import get_logger, set_global_level
        log = get_logger("test.global_level", log_to_file=False)
        set_global_level(logging.DEBUG)
        assert log.level == logging.DEBUG
        # Reset to INFO so other tests aren't affected
        set_global_level(logging.INFO)

    def test_log_methods_exist(self):
        """Logger exposes the standard log-level methods."""
        from nexa.utils import get_logger
        log = get_logger("test.methods", log_to_file=False)
        for method_name in ("debug", "info", "warning", "error", "critical"):
            assert hasattr(log, method_name), f"Missing method: {method_name}"

    def test_does_not_propagate(self):
        """Logger does not propagate to the root logger (avoids duplicate output)."""
        from nexa.utils import get_logger
        log = get_logger("test.propagate", log_to_file=False)
        assert log.propagate is False


# =============================================================================
# Config loader tests
# =============================================================================

class TestLoadConfig:
    """Tests for nexa.utils.config_loader.load_config()"""

    @pytest.fixture
    def model_config_path(self) -> Path:
        """Return the path to the model config file."""
        # The test is run from the project root, so this path is correct.
        return Path("configs/model_config.yaml")

    @pytest.fixture
    def train_config_path(self) -> Path:
        return Path("configs/train_config.yaml")

    @pytest.fixture
    def inference_config_path(self) -> Path:
        return Path("configs/inference_config.yaml")

    def test_import(self):
        """load_config is importable from the nexa.utils public API."""
        from nexa.utils import load_config  # noqa: F401
        assert callable(load_config)

    def test_loads_model_config(self, model_config_path):
        """model_config.yaml loads without errors."""
        from nexa.utils import load_config
        cfg = load_config(model_config_path)
        assert cfg is not None

    def test_model_config_dot_access(self, model_config_path):
        """Loaded config supports dot-notation access."""
        from nexa.utils import load_config
        cfg = load_config(model_config_path)
        # Top-level key
        assert hasattr(cfg, "model")
        # Nested keys
        assert cfg.model.d_model == 512
        assert cfg.model.n_heads == 8
        assert cfg.model.n_layers == 6

    def test_model_config_required_keys(self, model_config_path):
        """model_config.yaml contains all required architecture keys."""
        from nexa.utils import load_config
        cfg = load_config(model_config_path)
        required_keys = [
            "vocab_size", "max_seq_len", "d_model",
            "n_heads", "n_layers", "d_ff", "dropout",
        ]
        for key in required_keys:
            assert hasattr(cfg.model, key), f"Missing key in model config: {key}"

    def test_train_config_loads(self, train_config_path):
        """train_config.yaml loads without errors."""
        from nexa.utils import load_config
        cfg = load_config(train_config_path)
        assert cfg.training.learning_rate == pytest.approx(3e-4)
        assert cfg.training.batch_size == 16

    def test_inference_config_loads(self, inference_config_path):
        """inference_config.yaml loads without errors."""
        from nexa.utils import load_config
        cfg = load_config(inference_config_path)
        assert cfg.inference.temperature == pytest.approx(0.8)
        assert cfg.inference.top_k == 50

    def test_override_applies(self, model_config_path):
        """CLI-style overrides correctly override YAML values."""
        from nexa.utils import load_config
        cfg = load_config(model_config_path, overrides=["model.d_model=256"])
        assert cfg.model.d_model == 256

    def test_missing_file_raises(self):
        """load_config() raises FileNotFoundError for nonexistent files."""
        from nexa.utils import load_config
        with pytest.raises(FileNotFoundError):
            load_config("configs/this_file_does_not_exist.yaml")

    def test_merge_configs(self, model_config_path, train_config_path):
        """merge_configs() merges multiple YAML files into one DictConfig."""
        from nexa.utils import merge_configs
        cfg = merge_configs(model_config_path, train_config_path)
        # Both namespaces should be present
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "training")

    def test_config_to_dict(self, model_config_path):
        """config_to_dict() returns a plain Python dict."""
        from nexa.utils import load_config, config_to_dict
        cfg = load_config(model_config_path)
        result = config_to_dict(cfg)
        assert isinstance(result, dict)
        assert "model" in result
        assert isinstance(result["model"]["d_model"], int)


# =============================================================================
# Package-level smoke test
# =============================================================================

class TestNexaPackage:
    """Verify the nexa root package is installed and importable."""

    def test_package_importable(self):
        """import nexa succeeds."""
        import nexa  # noqa: F401

    def test_version_exists(self):
        """nexa.__version__ is defined."""
        import nexa
        assert hasattr(nexa, "__version__")
        assert isinstance(nexa.__version__, str)

    def test_sub_packages_importable(self):
        """All sub-package stubs are importable (even if not yet implemented)."""
        import nexa.tokenizer  # noqa: F401
        import nexa.models     # noqa: F401
        import nexa.training   # noqa: F401
        import nexa.inference  # noqa: F401
        import nexa.memory     # noqa: F401
        import nexa.tools      # noqa: F401
        import nexa.speech     # noqa: F401
        import nexa.app        # noqa: F401
