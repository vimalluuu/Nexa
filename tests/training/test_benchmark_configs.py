"""
Tests for Phase 8.7 pretraining readiness benchmark.

Covers:
- Parameter counting accuracy
- Memory estimation
- Token window iterator (from synthetic shards)
- Recommendation engine logic
- Config validation (all 3 candidate YAML files)
- Benchmark config initialization
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nexa.models.config import ModelConfig
from nexa.models.transformer import NexaTransformer
from scripts.training.benchmark_configs import (
    count_parameters,
    estimate_memory_mb,
    recommend_config,
    token_window_iterator,
    CANDIDATE_CONFIGS,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_small_config(**overrides) -> ModelConfig:
    """Small config for fast in-test model creation."""
    base = dict(
        vocab_size=2048, d_model=64, n_heads=4, n_layers=2, d_ff=128,
        max_seq_len=128, dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig.from_dict(base)


def _write_shard(path: Path, tokens: list[int]) -> None:
    """Write a synthetic uint16 binary shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(tokens)}H", *tokens))


# ===========================================================================
# 1. Parameter counting
# ===========================================================================

class TestCountParameters:
    def test_total_is_positive(self):
        model = NexaTransformer(_make_small_config())
        counts = count_parameters(model)
        assert counts["total"] > 0

    def test_embedding_plus_rest_equals_total(self):
        model = NexaTransformer(_make_small_config())
        counts = count_parameters(model)
        # embedding + attention + mlp + norms_and_other = total
        assert (counts["embedding"] + counts["attention"] +
                counts["mlp"] + counts["norms_and_other"]) == counts["total"]

    def test_non_embedding_less_than_total(self):
        model = NexaTransformer(_make_small_config())
        counts = count_parameters(model)
        assert counts["non_embedding"] < counts["total"]
        assert counts["non_embedding"] == counts["total"] - counts["embedding"]

    def test_tied_embedding_not_double_counted(self):
        """With tie_embeddings=True, the LM head shares embedding weights."""
        cfg = _make_small_config(tie_embeddings=True)
        model = NexaTransformer(cfg)
        counts = count_parameters(model)
        embed_params = cfg.vocab_size * cfg.d_model  # 2048 × 64
        assert counts["embedding"] == embed_params
        # Total should NOT have embedding counted twice
        assert counts["total"] < embed_params * 2 + counts["attention"] + counts["mlp"]

    def test_scaling_with_layers(self):
        """More layers → more parameters."""
        model_2l = NexaTransformer(_make_small_config(n_layers=2))
        model_4l = NexaTransformer(_make_small_config(n_layers=4))
        c2 = count_parameters(model_2l)
        c4 = count_parameters(model_4l)
        assert c4["total"] > c2["total"]
        assert c4["attention"] == 2 * c2["attention"]
        assert c4["mlp"]       == 2 * c2["mlp"]

    def test_scaling_with_d_model(self):
        """Larger d_model → more parameters (quadratic in attention, linear in embedding)."""
        model_small = NexaTransformer(_make_small_config(d_model=64,  n_heads=4, d_ff=128))
        model_large = NexaTransformer(_make_small_config(d_model=128, n_heads=4, d_ff=256))
        c_s = count_parameters(model_small)
        c_l = count_parameters(model_large)
        assert c_l["total"] > c_s["total"]


# ===========================================================================
# 2. Memory estimation
# ===========================================================================

class TestEstimateMemoryMB:
    def _counts(self):
        model = NexaTransformer(_make_small_config())
        return count_parameters(model)

    def test_returns_positive_values(self):
        counts = self._counts()
        mem = estimate_memory_mb(counts, batch_size=4, seq_len=128, d_model=64, n_layers=2)
        assert mem["model_weights_mb"] > 0
        assert mem["optimizer_states_mb"] > 0
        assert mem["gradients_mb"] > 0
        assert mem["activations_mb"] > 0
        assert mem["total_estimated_mb"] > 0

    def test_optimizer_states_twice_model(self):
        """AdamW keeps 2 state tensors per param."""
        counts = self._counts()
        mem = estimate_memory_mb(counts, batch_size=4, seq_len=128, d_model=64, n_layers=2)
        # optimizer ≈ 2 × model weights
        assert abs(mem["optimizer_states_mb"] - 2 * mem["model_weights_mb"]) < 0.1

    def test_larger_batch_means_more_activation(self):
        counts = self._counts()
        m1 = estimate_memory_mb(counts, batch_size=1,  seq_len=128, d_model=64, n_layers=2)
        m8 = estimate_memory_mb(counts, batch_size=8,  seq_len=128, d_model=64, n_layers=2)
        # Larger batch should produce more activation memory
        assert m8["activations_mb"] > m1["activations_mb"]
        # Should scale roughly linearly (within 30% after rounding)
        ratio = m8["activations_mb"] / max(m1["activations_mb"], 0.001)
        assert 4 <= ratio <= 12  # 8× expected; rounding may push 0.1→1.0

    def test_larger_seq_means_more_activation(self):
        counts = self._counts()
        m1 = estimate_memory_mb(counts, batch_size=4, seq_len=64,  d_model=64, n_layers=2)
        m2 = estimate_memory_mb(counts, batch_size=4, seq_len=128, d_model=64, n_layers=2)
        # Larger sequence should produce more activation memory
        assert m2["activations_mb"] > m1["activations_mb"]


# ===========================================================================
# 3. Token window iterator
# ===========================================================================

class TestTokenWindowIterator:
    def test_yields_correct_shape(self, tmp_path):
        tokens = list(range(1000))
        shard = tmp_path / "shard_000.bin"
        _write_shard(shard, tokens)

        batches = list(token_window_iterator(tmp_path, seq_len=10, batch_size=4, max_tokens=1000))
        assert len(batches) > 0
        x, y = batches[0]
        assert x.shape == (4, 10)
        assert y.shape == (4, 10)

    def test_y_is_x_shifted_by_one(self, tmp_path):
        tokens = list(range(200))
        _write_shard(tmp_path / "shard_000.bin", tokens)

        (x, y), *_ = list(token_window_iterator(tmp_path, seq_len=10, batch_size=1, max_tokens=200))
        assert y[0, 0] == x[0, 1]
        assert y[0, -1] == x[0, -1] + 1  # shifted sequence

    def test_max_tokens_limits_output(self, tmp_path):
        tokens = list(range(10000))
        _write_shard(tmp_path / "shard_000.bin", tokens)

        batches = list(token_window_iterator(tmp_path, seq_len=10, batch_size=2, max_tokens=100))
        total = sum(x.shape[0] * x.shape[1] for x, _ in batches)
        assert total <= 200  # approximate; at most 2× the limit

    def test_multiple_shards_streamed(self, tmp_path):
        _write_shard(tmp_path / "shard_000.bin", list(range(500)))
        _write_shard(tmp_path / "shard_001.bin", list(range(500, 1000)))

        batches = list(token_window_iterator(tmp_path, seq_len=10, batch_size=4, max_tokens=900))
        assert len(batches) > 0


# ===========================================================================
# 4. Recommendation engine
# ===========================================================================

class TestRecommendConfig:
    def _make_result(self, name: str, tokens_per_sec: int, total_params: int, total_mb: float) -> dict:
        return {
            "name": name,
            "config": {"d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 256,
                       "vocab_size": 2048, "max_seq_len": 128},
            "benchmark_settings": {"batch_size": 4, "seq_len": 128,
                                   "warmup_steps": 5, "benchmark_steps": 20},
            "parameters": {"total": total_params, "non_embedding": total_params - 50000,
                           "embedding": 50000, "attention": 100000, "mlp": 200000,
                           "norms_and_other": total_params - 350000},
            "memory_estimate_mb": {"total_estimated_mb": total_mb, "model_weights_mb": 10,
                                   "optimizer_states_mb": 20, "gradients_mb": 10,
                                   "activations_mb": total_mb - 40},
            "throughput": {"tokens_per_sec": tokens_per_sec, "tokens_processed": 50000},
            "timing_ms": {"fwd_avg": 10, "fwd_p95": 15, "bwd_avg": 20, "bwd_p95": 25,
                          "opt_avg": 2, "total_step_avg": 32},
            "loss": {"avg_train_loss": 7.0, "avg_val_loss": 7.1,
                     "train_perplexity": math.exp(7.0), "val_perplexity": math.exp(7.1)},
        }

    def test_returns_dict_with_recommended(self):
        results = [
            self._make_result("small",  5000,  1_000_000, 50),
            self._make_result("medium", 1000,  6_000_000, 200),
            self._make_result("large",  200,  25_000_000, 400),
        ]
        rec = recommend_config(results)
        assert "recommended" in rec
        assert rec["recommended"] in ["small", "medium", "large"]

    def test_prefers_largest_eligible(self):
        """Given all are above threshold, picks the largest by params."""
        results = [
            self._make_result("small",  5000,  1_000_000, 50),
            self._make_result("medium", 1000,  6_000_000, 200),
            self._make_result("large",  500,  25_000_000, 400),
        ]
        rec = recommend_config(results)
        assert rec["recommended"] == "large"

    def test_excludes_too_slow(self):
        """Configs below 100 tokens/sec should not be recommended unless all are slow."""
        results = [
            self._make_result("small",  5000,  1_000_000, 50),
            self._make_result("medium", 50,    6_000_000, 200),  # too slow
        ]
        rec = recommend_config(results)
        assert rec["recommended"] == "small"

    def test_rationale_is_string(self):
        results = [self._make_result("small", 2000, 1_000_000, 50)]
        rec = recommend_config(results)
        assert isinstance(rec["rationale"], str)
        assert len(rec["rationale"]) > 30

    def test_steps_per_epoch_positive(self):
        results = [self._make_result("medium", 1000, 6_000_000, 200)]
        rec = recommend_config(results)
        assert rec["steps_per_epoch_estimate"] > 0


# ===========================================================================
# 5. Candidate YAML configs
# ===========================================================================

class TestCandidateConfigs:
    @pytest.mark.parametrize("name, config_path, bench_settings", CANDIDATE_CONFIGS)
    def test_config_loads(self, name, config_path, bench_settings):
        assert config_path.exists(), f"{config_path} does not exist"
        cfg = ModelConfig.from_yaml(config_path)
        assert cfg.vocab_size == 2048
        assert cfg.d_model % cfg.n_heads == 0

    @pytest.mark.parametrize("name, config_path, bench_settings", CANDIDATE_CONFIGS)
    def test_model_instantiates(self, name, config_path, bench_settings):
        cfg = ModelConfig.from_yaml(config_path)
        model = NexaTransformer(cfg)
        assert model.num_parameters > 0

    @pytest.mark.parametrize("name, config_path, bench_settings", CANDIDATE_CONFIGS)
    def test_forward_pass_correct_shape(self, name, config_path, bench_settings):
        cfg = ModelConfig.from_yaml(config_path)
        model = NexaTransformer(cfg)
        model.eval()
        seq = bench_settings["seq_len"]
        b   = bench_settings["batch_size"]
        x = torch.randint(0, 2048, (b, seq))
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (b, seq, 2048)

    def test_small_config_smallest_params(self):
        _, small_path, _ = CANDIDATE_CONFIGS[0]
        _, large_path, _ = CANDIDATE_CONFIGS[2]
        cfg_s = ModelConfig.from_yaml(small_path)
        cfg_l = ModelConfig.from_yaml(large_path)
        m_s = NexaTransformer(cfg_s)
        m_l = NexaTransformer(cfg_l)
        assert m_s.num_parameters < m_l.num_parameters

    def test_all_configs_have_correct_vocab_size(self):
        for name, path, _ in CANDIDATE_CONFIGS:
            cfg = ModelConfig.from_yaml(path)
            assert cfg.vocab_size == 2048, f"{name} has wrong vocab_size: {cfg.vocab_size}"

    def test_all_configs_have_bos_eos(self):
        for name, path, _ in CANDIDATE_CONFIGS:
            cfg = ModelConfig.from_yaml(path)
            assert cfg.bos_token_id == 1
            assert cfg.eos_token_id == 2
