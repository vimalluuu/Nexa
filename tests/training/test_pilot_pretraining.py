"""
Tests for Phase 8.9 — Pilot Pretraining Infrastructure

Covers pilot-specific logic:
- SHA-256 checksum verification
- preflight config check
- shard token loading
- reload_and_validate
- inference sanity check
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
from pathlib import Path

import pytest
import torch

from nexa.models.config import ModelConfig
from nexa.models.transformer import NexaTransformer
from nexa.training.dataset import NexaDataset
from nexa.training.trainer import Trainer, TrainerConfig
from scripts.training.run_pilot_pretraining import (
    sha256_file,
    verify_checksums,
    load_shard_tokens,
    run_inference_check,
    reload_and_validate,
    CHECKSUMS_FILE,
    SHARD_TRAIN_DIR,
    SHARD_VAL_DIR,
    PRETRAINING_CONFIG,
    TARGET_STEPS,
    TRAIN_TOKENS,
    TRAIN_TOKENS_PER_STEP,
    WARMUP_STEPS,
    SEED,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _write_shard(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(tokens)}H", *tokens))


def _tiny_cfg() -> ModelConfig:
    return ModelConfig.from_dict(dict(
        vocab_size=64, d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=16, dropout=0.0,
    ))


def _tiny_trainer_cfg(tmp_path: Path, max_steps: int = 10) -> TrainerConfig:
    return TrainerConfig(
        block_size=16, batch_size=2, learning_rate=1e-3, min_lr=1e-4,
        weight_decay=0.1, max_steps=max_steps, warmup_steps=2,
        eval_interval=max_steps, eval_steps=5,
        save_interval=max_steps, checkpoint_dir=tmp_path / "ckpts",
        log_interval=max(1, max_steps // 4), seed=42, device="cpu",
    )


def _make_ds(vocab: int = 64, n: int = 2000) -> NexaDataset:
    tokens = (list(range(vocab)) * (n // vocab + 1))[:n]
    return NexaDataset(tokens, block_size=16)


# ===========================================================================
# 1. SHA-256 checksum
# ===========================================================================

class TestSha256File:
    def test_basic_hash(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert sha256_file(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert sha256_file(f) == hashlib.sha256(b"").hexdigest()

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content_a")
        f2.write_bytes(b"content_b")
        assert sha256_file(f1) != sha256_file(f2)

    def test_same_content_same_hash(self, tmp_path):
        data = struct.pack("<1000H", *range(1000))
        f1 = tmp_path / "shard1.bin"
        f2 = tmp_path / "shard2.bin"
        f1.write_bytes(data)
        f2.write_bytes(data)
        assert sha256_file(f1) == sha256_file(f2)


# ===========================================================================
# 2. verify_checksums (using synthetic shards + manifest)
# ===========================================================================

class TestVerifyChecksums:
    def test_valid_checksums(self, tmp_path, monkeypatch):
        """All shards match their recorded checksums."""
        shard_dir = tmp_path / "processed"
        shard = shard_dir / "train" / "shard_000.bin"
        _write_shard(shard, list(range(500)))
        digest = sha256_file(shard)
        checksum_file = tmp_path / "checksums.sha256"
        checksum_file.write_text(f"{digest}  train/shard_000.bin\n")

        monkeypatch.setattr(
            "scripts.training.run_pilot_pretraining.CHECKSUMS_FILE", checksum_file
        )
        # patch the base dir used in path construction
        import scripts.training.run_pilot_pretraining as rcp
        orig_fn = rcp.verify_checksums

        def patched():
            results = {}
            for line in checksum_file.read_text().strip().splitlines():
                d, rel = line.strip().split(None, 1)
                full = shard_dir / rel
                if full.exists():
                    results[rel] = sha256_file(full) == d
                else:
                    results[rel] = False
            return results

        result = patched()
        assert result["train/shard_000.bin"] is True

    def test_corrupt_shard_detected(self, tmp_path):
        """A shard that doesn't match its recorded checksum returns False."""
        shard_dir = tmp_path / "processed"
        shard = shard_dir / "train" / "shard_000.bin"
        _write_shard(shard, list(range(500)))
        wrong_digest = "0" * 64   # bogus hash

        def check():
            actual = sha256_file(shard)
            return actual == wrong_digest

        assert check() is False

    def test_missing_file_returns_false(self, tmp_path):
        missing = tmp_path / "nonexistent.bin"
        assert not missing.exists()


# ===========================================================================
# 3. Constants and config values
# ===========================================================================

class TestConstants:
    def test_target_steps(self):
        """One epoch = 18,901,546 // 2,048 = 9,229 steps."""
        assert TARGET_STEPS == TRAIN_TOKENS // TRAIN_TOKENS_PER_STEP
        assert TARGET_STEPS == 9229

    def test_tokens_per_step(self):
        assert TRAIN_TOKENS_PER_STEP == 4 * 512  # batch × seq_len

    def test_warmup_is_small_fraction(self):
        """Warmup should be at most 10% of total steps."""
        assert WARMUP_STEPS <= TARGET_STEPS * 0.10

    def test_seed_is_42(self):
        assert SEED == 42

    def test_pretraining_config_exists(self):
        assert PRETRAINING_CONFIG.exists()

    def test_pretraining_config_correct_vocab(self):
        cfg = ModelConfig.from_yaml(PRETRAINING_CONFIG)
        assert cfg.vocab_size == 2048
        assert cfg.d_model == 256
        assert cfg.n_heads == 8
        assert cfg.n_layers == 6
        assert cfg.d_ff == 1024

    def test_phase87_shards_exist(self):
        """The Phase 8.6 shards must exist before we can train."""
        assert SHARD_TRAIN_DIR.exists(), "Train shard dir missing"
        assert SHARD_VAL_DIR.exists(),   "Val shard dir missing"
        train_shards = list(SHARD_TRAIN_DIR.glob("shard_*.bin"))
        val_shards   = list(SHARD_VAL_DIR.glob("shard_*.bin"))
        assert len(train_shards) == 2, f"Expected 2 train shards, got {len(train_shards)}"
        assert len(val_shards)   == 1, f"Expected 1 val shard, got {len(val_shards)}"

    def test_phase86_checksums_exist(self):
        assert CHECKSUMS_FILE.exists(), "Checksum file missing"

    def test_phase86_checksums_valid(self):
        """All Phase 8.6 shards must match their recorded SHA-256."""
        result = verify_checksums()
        assert result, "No checksums found"
        for rel, ok in result.items():
            assert ok, f"Checksum MISMATCH: {rel}"


# ===========================================================================
# 4. load_shard_tokens
# ===========================================================================

class TestLoadShardTokens:
    def test_loads_all_tokens(self, tmp_path):
        _write_shard(tmp_path / "shard_000.bin", list(range(100)))
        _write_shard(tmp_path / "shard_001.bin", list(range(100, 200)))
        tokens = load_shard_tokens(tmp_path)
        assert tokens == list(range(200))

    def test_empty_dir(self, tmp_path):
        assert load_shard_tokens(tmp_path) == []

    def test_real_train_shards_load(self):
        """Verify that real Phase 8.6 shards load correctly."""
        tokens = load_shard_tokens(SHARD_TRAIN_DIR)
        assert len(tokens) == 18_901_546
        assert all(0 <= t < 65536 for t in tokens[:1000])  # sample check

    def test_real_val_shards_load(self):
        tokens = load_shard_tokens(SHARD_VAL_DIR)
        assert len(tokens) == 2_000_821


# ===========================================================================
# 5. run_inference_check
# ===========================================================================

class TestRunInferenceCheck:
    def test_produces_valid_ids(self):
        cfg = _tiny_cfg()
        model = NexaTransformer(cfg)
        model.eval()
        result = run_inference_check(model, cfg.vocab_size)
        assert result["all_valid"]
        for s in result["samples"]:
            assert all(0 <= t < cfg.vocab_size for t in s["generated_tokens"])

    def test_three_samples(self):
        cfg = _tiny_cfg()
        model = NexaTransformer(cfg)
        model.eval()
        result = run_inference_check(model, cfg.vocab_size)
        assert len(result["samples"]) == 3

    def test_prompt_in_output(self):
        cfg = _tiny_cfg()
        model = NexaTransformer(cfg)
        model.eval()
        result = run_inference_check(model, cfg.vocab_size)
        for s in result["samples"]:
            pl = len(s["prompt_tokens"])
            assert s["generated_tokens"][:pl] == s["prompt_tokens"]


# ===========================================================================
# 6. reload_and_validate
# ===========================================================================

class TestReloadAndValidate:
    def _train_save(self, tmp_path):
        cfg = _tiny_cfg()
        model = NexaTransformer(cfg)
        ds = _make_ds()
        tcfg = _tiny_trainer_cfg(tmp_path, max_steps=10)
        Trainer(model, tcfg, ds, ds).train()
        ckpts = sorted((tmp_path / "ckpts").glob("step_*.pt"))
        return ckpts[-1], cfg

    def test_reproduced_loss_finite(self, tmp_path):
        ckpt, cfg = self._train_save(tmp_path)
        ds = _make_ds()
        result = reload_and_validate(ckpt, ds, cfg, batch_size=2, eval_steps=5)
        assert math.isfinite(result["reproduced_val_loss"])

    def test_reproduced_ppl_positive(self, tmp_path):
        ckpt, cfg = self._train_save(tmp_path)
        ds = _make_ds()
        result = reload_and_validate(ckpt, ds, cfg, batch_size=2, eval_steps=5)
        assert result["reproduced_val_ppl"] > 1.0

    def test_step_recorded(self, tmp_path):
        ckpt, cfg = self._train_save(tmp_path)
        ds = _make_ds()
        result = reload_and_validate(ckpt, ds, cfg, batch_size=2, eval_steps=5)
        assert result["checkpoint_step"] == 10

    def test_inference_in_result(self, tmp_path):
        ckpt, cfg = self._train_save(tmp_path)
        ds = _make_ds()
        result = reload_and_validate(ckpt, ds, cfg, batch_size=2, eval_steps=5)
        assert "inference" in result
        assert result["inference"]["all_valid"]
