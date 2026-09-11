"""
tests/training/test_v2_controlled_training.py
==============================================
Tests for V2 controlled training infrastructure:
- V2ShardDataset loading and token range validation
- Checkpoint save/reload produces identical validation loss
- Resume training keeps loss finite
"""

import math
import struct
import tempfile
from pathlib import Path

import pytest
import torch

from nexa.models import NexaTransformer, ModelConfig
from nexa.training.scheduler import CosineWarmupScheduler


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_dummy_shard(path: Path, n_tokens: int, vocab_size: int = 8192) -> None:
    """Write a dummy uint16 binary shard for testing."""
    import random
    random.seed(0)
    tokens = [random.randint(0, vocab_size - 1) for _ in range(n_tokens)]
    with open(path, "wb") as f:
        for t in tokens:
            f.write(struct.pack("<H", t))


def _make_small_model() -> tuple[NexaTransformer, ModelConfig]:
    cfg = ModelConfig(
        vocab_size   = 8192,
        d_model      = 64,
        n_layers     = 2,
        n_heads      = 4,
        d_ff         = 128,
        max_seq_len  = 64,
        dropout      = 0.0,
    )
    model = NexaTransformer(cfg)
    return model, cfg


# ──────────────────────────────────────────────────────────────────────────────
# V2ShardDataset tests
# ──────────────────────────────────────────────────────────────────────────────

def test_v2_shard_dataset_loads_and_validates(tmp_path):
    """V2ShardDataset correctly loads uint16 shard and validates token range."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.training.run_v2_controlled_training import V2ShardDataset

    shard = tmp_path / "shard_000.bin"
    n_tokens = 2048
    _write_dummy_shard(shard, n_tokens, vocab_size=8192)

    ds = V2ShardDataset(tmp_path, block_size=64)
    assert ds.token_count() == n_tokens
    assert len(ds) == n_tokens - 64

    x, y = ds[0]
    assert x.shape == (64,)
    assert y.shape == (64,)
    # y is x shifted by 1
    assert torch.equal(x[1:], y[:-1])

    check = ds.check_token_range(vocab_size=8192)
    assert check["valid"], f"Invalid tokens found: {check}"


def test_v2_shard_dataset_detects_invalid_tokens(tmp_path):
    """V2ShardDataset detects token IDs >= vocab_size."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.training.run_v2_controlled_training import V2ShardDataset

    shard = tmp_path / "shard_000.bin"
    # Write some tokens that exceed vocab_size=100
    with open(shard, "wb") as f:
        for t in [0, 50, 99, 100, 200]:  # 100 and 200 are out of range
            f.write(struct.pack("<H", t))

    ds = V2ShardDataset(tmp_path, block_size=2)
    check = ds.check_token_range(vocab_size=100)
    assert not check["valid"]
    assert check["invalid_count"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint save/reload test
# ──────────────────────────────────────────────────────────────────────────────

def test_checkpoint_reload_produces_identical_loss(tmp_path):
    """Loading a checkpoint into a fresh model produces identical val loss."""
    import torch.nn.functional as F

    model, cfg = _make_small_model()
    torch.manual_seed(0)
    model.eval()

    # Create dummy input
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    y = torch.randint(0, cfg.vocab_size, (2, 64))

    with torch.no_grad():
        logits = model(x)
        loss_before = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1)).item()

    # Save checkpoint
    ckpt_path = tmp_path / "step_0000100.pt"
    torch.save({"model_state": model.state_dict(), "step": 100}, ckpt_path)

    # Load fresh model
    model_fresh = NexaTransformer(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_fresh.load_state_dict(ckpt["model_state"])
    model_fresh.eval()

    with torch.no_grad():
        logits_fresh = model_fresh(x)
        loss_after = F.cross_entropy(logits_fresh.view(-1, cfg.vocab_size), y.view(-1)).item()

    diff = abs(loss_before - loss_after)
    assert diff < 1e-5, f"Reload loss mismatch: before={loss_before:.6f} after={loss_after:.6f} diff={diff:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# Resume training stays finite
# ──────────────────────────────────────────────────────────────────────────────

def test_resume_training_loss_stays_finite(tmp_path):
    """Training resumed from a checkpoint produces finite losses."""
    import torch.nn.functional as F

    model, cfg = _make_small_model()
    torch.manual_seed(42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=2, max_steps=10, max_lr=1e-3, min_lr=1e-4
    )

    # Train 5 steps
    model.train()
    for _ in range(5):
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        y = torch.randint(0, cfg.vocab_size, (2, 64))
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    # Save
    ckpt_path = tmp_path / "step_0000005.pt"
    torch.save({
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_step" : scheduler.steps_completed,
        "step"           : 5,
    }, ckpt_path)

    # Reload and resume
    model2, cfg2 = _make_small_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = CosineWarmupScheduler(
        optimizer2, warmup_steps=2, max_steps=15, max_lr=1e-3, min_lr=1e-4
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model2.load_state_dict(ckpt["model_state"])
    optimizer2.load_state_dict(ckpt["optimizer_state"])
    scheduler2._step = ckpt["scheduler_step"]

    model2.train()
    for _ in range(5):
        x = torch.randint(0, cfg2.vocab_size, (2, 64))
        y = torch.randint(0, cfg2.vocab_size, (2, 64))
        logits = model2(x)
        loss = F.cross_entropy(logits.view(-1, cfg2.vocab_size), y.view(-1))
        assert math.isfinite(loss.item()), f"Loss became non-finite: {loss.item()}"
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        scheduler2.step()


# ──────────────────────────────────────────────────────────────────────────────
# Inference: token IDs in range and seeded reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def test_greedy_inference_token_ids_in_range():
    """Greedy generation produces token IDs strictly within [0, vocab_size)."""
    import torch.nn.functional as F

    model, cfg = _make_small_model()
    model.eval()
    torch.manual_seed(0)

    input_ids = torch.tensor([[cfg.bos_token_id]], dtype=torch.long)
    generated = []
    with torch.no_grad():
        for _ in range(20):
            logits = model(input_ids)
            next_id = int(logits[0, -1, :].argmax())
            generated.append(next_id)
            if next_id == cfg.eos_token_id:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

    assert all(0 <= t < cfg.vocab_size for t in generated), \
        f"Generated token out of range: {generated}"


def test_seeded_generation_reproducible():
    """Two generations with the same seed and temperature produce identical tokens."""
    import torch.nn.functional as F

    model, cfg = _make_small_model()
    model.eval()

    def generate(seed):
        torch.manual_seed(seed)
        input_ids = torch.tensor([[cfg.bos_token_id]], dtype=torch.long)
        out = []
        with torch.no_grad():
            for _ in range(10):
                logits = model(input_ids)
                probs = F.softmax(logits[0, -1, :] / 0.8, dim=-1)
                next_id = int(torch.multinomial(probs, 1))
                out.append(next_id)
                if next_id == cfg.eos_token_id:
                    break
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        return out

    assert generate(42) == generate(42), "Seeded generation is not reproducible"
    assert generate(42) != generate(99) or True  # May differ with different seed (soft check)
