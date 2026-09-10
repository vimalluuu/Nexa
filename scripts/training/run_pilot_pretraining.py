"""
Nexa Phase 8.9 — Pilot Pretraining Run (One Full Epoch)
=========================================================
Runs one complete pass over the 18.9M training tokens using the
Phase 8.7 selected medium configuration.

This is the FIRST REAL Nexa V1 pilot pretraining run.

Training budget:
  - ~9,229 optimizer steps (18,901,546 / 2,048 tokens/step)
  - ~18.9M tokens (~5.4h at 978 tok/s)
  - periodic checkpoints every 1,000 steps
  - validation every 500 steps
  - logging every 100 steps

Usage:
    python -m scripts.training.run_pilot_pretraining
    python -m scripts.training.run_pilot_pretraining --resume checkpoints/nexa_v1_pilot/step_0005000.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from nexa.models.config import ModelConfig
from nexa.models.transformer import NexaTransformer
from nexa.training.dataset import NexaDataset
from nexa.training.trainer import Trainer, TrainerConfig

# ---------------------------------------------------------------------------
# Constants — frozen per Phase 8.7 / 8.8 specifications
# ---------------------------------------------------------------------------

PRETRAINING_CONFIG  = Path("configs/nexa_v1_pretraining.yaml")
SHARD_TRAIN_DIR     = Path("data/processed/train")
SHARD_VAL_DIR       = Path("data/processed/validation")
CHECKPOINT_DIR      = Path("checkpoints/nexa_v1_pilot")
REPORTS_DIR         = Path("data/reports")
MANIFESTS_DIR       = Path("data/manifests")

TOKENIZER_DIR       = Path("data/processed/tokenizer_v1")

DATASET_MANIFEST    = MANIFESTS_DIR / "tokenized_dataset_v1.json"
CHECKSUMS_FILE      = MANIFESTS_DIR / "tokenized_checksums.sha256"

TRAIN_TOKENS        = 18_901_546
TRAIN_TOKENS_PER_STEP = 2_048    # batch_size=4 × seq_len=512
TARGET_STEPS        = TRAIN_TOKENS // TRAIN_TOKENS_PER_STEP  # 9229

# Training hyper-parameters (DO NOT change without updating Phase 8.7 rationale)
BATCH_SIZE          = 4
SEQ_LEN             = 512
LEARNING_RATE       = 3e-4
MIN_LR              = 3e-5
WEIGHT_DECAY        = 0.1
GRAD_CLIP           = 1.0
BETA1, BETA2        = 0.9, 0.95
WARMUP_STEPS        = 200
LOG_INTERVAL        = 100
EVAL_INTERVAL       = 500
EVAL_STEPS          = 100
SAVE_INTERVAL       = 1_000
KEEP_LAST_N         = 10
SEED                = 42


# ---------------------------------------------------------------------------
# Integrity helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums() -> dict[str, bool]:
    """
    Verify that the on-disk shards match the Phase 8.6 manifests.
    Returns a dict {shard_relative_path: True/False}.
    """
    if not CHECKSUMS_FILE.exists():
        return {}

    results: dict[str, bool] = {}
    for line in CHECKSUMS_FILE.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel_path = line.split(None, 1)
        full_path = Path("data/processed") / rel_path
        if full_path.exists():
            actual = sha256_file(full_path)
            results[rel_path] = (actual == digest)
        else:
            results[rel_path] = False
    return results


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_shard_tokens(shard_dir: Path) -> list[int]:
    """Read all uint16 token IDs from every shard in a directory."""
    all_tokens: list[int] = []
    for shard_path in sorted(shard_dir.glob("shard_*.bin")):
        data = shard_path.read_bytes()
        n = len(data) // 2
        all_tokens.extend(struct.unpack(f"<{n}H", data))
    return all_tokens


# ---------------------------------------------------------------------------
# Inference sanity check
# ---------------------------------------------------------------------------

def run_inference_check(model: NexaTransformer, vocab_size: int) -> dict:
    """Generate a few short samples; verify all IDs are in [0, vocab_size)."""
    model.eval()
    device = next(model.parameters()).device
    t1 = min(50, vocab_size - 1)
    t2 = min(200, vocab_size - 1)
    prompts = [
        [1],
        [1, t1],
        [1, t1, t2],
    ]
    samples = []
    for prompt in prompts:
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, temperature=0.8, eos_token_id=2)
        gen = out[0].tolist()
        samples.append({
            "prompt_tokens":    prompt,
            "generated_tokens": gen,
            "length":           len(gen),
            "all_ids_valid":    all(0 <= t < vocab_size for t in gen),
        })
    return {
        "samples":   samples,
        "all_valid": all(s["all_ids_valid"] for s in samples),
    }


# ---------------------------------------------------------------------------
# Pre-training verification
# ---------------------------------------------------------------------------

def preflight(resume_path: Path | None) -> dict:
    """Run all pre-training integrity checks. Returns a verification dict."""
    print("\n=== Pre-flight Verification ===")
    checks: dict[str, object] = {}

    # 1. Config
    assert PRETRAINING_CONFIG.exists(), f"Config missing: {PRETRAINING_CONFIG}"
    cfg = ModelConfig.from_yaml(PRETRAINING_CONFIG)
    checks["config_vocab_size"]  = cfg.vocab_size  == 2048
    checks["config_d_model"]     = cfg.d_model     == 256
    checks["config_n_heads"]     = cfg.n_heads     == 8
    checks["config_n_layers"]    = cfg.n_layers    == 6
    checks["config_d_ff"]        = cfg.d_ff        == 1024
    checks["config_max_seq_len"] = cfg.max_seq_len == 512
    print(f"  Config: {PRETRAINING_CONFIG}  OK={all(v for v in checks.values())}")

    # 2. Shard checksums
    cksum = verify_checksums()
    checks["all_shards_valid"] = bool(cksum) and all(cksum.values())
    for rel, ok in cksum.items():
        print(f"  Shard {rel}: {'OK' if ok else 'MISMATCH!'}")

    # 3. Disk space (need ≥ 2 GB for checkpoints)
    stat = os.statvfs(str(SHARD_TRAIN_DIR)) if hasattr(os, "statvfs") else None
    if stat:
        free_gb = stat.f_bavail * stat.f_frsize / 1024**3
    else:
        # Windows fallback
        import shutil
        free_gb = shutil.disk_usage(str(SHARD_TRAIN_DIR.drive + "\\")).free / 1024**3
    checks["disk_free_gb"] = round(free_gb, 1)
    checks["disk_sufficient"] = free_gb >= 2.0
    print(f"  Disk free: {free_gb:.1f} GB  (need >= 2 GB)")

    # 4. Git commit
    checks["git_commit"] = get_git_commit()
    print(f"  Git commit: {checks['git_commit']}")

    # 5. Resume
    if resume_path is not None:
        checks["resume_path"] = str(resume_path)
        checks["resume_exists"] = resume_path.exists()
        print(f"  Resume: {resume_path}  exists={resume_path.exists()}")
    else:
        checks["resume_path"] = None
        checks["resume_exists"] = None
        print("  Resume: None (fresh training from scratch)")

    all_ok = (
        checks["config_vocab_size"]
        and checks["config_d_model"]
        and checks["all_shards_valid"]
        and checks["disk_sufficient"]
    )
    print(f"  ALL OK: {all_ok}")
    if not all_ok:
        raise RuntimeError(f"Pre-flight failed: {checks}")
    return checks


# ---------------------------------------------------------------------------
# Post-training: fresh reload + validation
# ---------------------------------------------------------------------------

def reload_and_validate(
    checkpoint_path: Path,
    val_dataset: NexaDataset,
    model_config: ModelConfig,
    batch_size: int,
    eval_steps: int,
) -> dict:
    """
    Load checkpoint into a brand-new model and re-evaluate validation loss.
    Confirms the checkpoint is self-contained and the reported loss is real.
    """
    print("\n  Loading checkpoint into fresh model...")
    fresh = NexaTransformer(model_config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    fresh.load_state_dict(ckpt["model_state"])
    fresh.eval()

    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            if n >= eval_steps:
                break
            logits = fresh(x)
            B, S, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * S, V), y.reshape(B * S))
            total_loss += loss.item()
            n += 1
    avg = total_loss / max(n, 1)
    print(f"  Fresh-model val loss: {avg:.4f} | PPL: {math.exp(avg):.2f}")

    inference = run_inference_check(fresh, model_config.vocab_size)
    return {
        "reproduced_val_loss": round(avg, 6),
        "reproduced_val_ppl":  round(math.exp(avg), 4),
        "checkpoint_step":     ckpt["step"],
        "checkpoint_seed":     SEED,
        "inference":           inference,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nexa Phase 8.9 — Pilot Pretraining")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    t_total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Pre-flight
    # ------------------------------------------------------------------
    checks = preflight(args.resume)

    # ------------------------------------------------------------------
    # 2. Load model config
    # ------------------------------------------------------------------
    model_config = ModelConfig.from_yaml(PRETRAINING_CONFIG)
    print(f"\nModel: {model_config}")

    model = NexaTransformer(model_config)
    if args.resume is None:
        # Verify truly fresh — sample a weight's std should be near 0.02
        sample = next(p for p in model.parameters() if p.ndim == 2)
        init_std = sample.std().item()
        assert 0.005 < init_std < 0.1, f"Unexpected init std {init_std:.4f}"
        print("Initialized: SCRATCH verified (no pretrained weights)")
    else:
        print(f"Resuming from: {args.resume}")

    print(f"Parameters: {model.num_parameters:,} total")

    # ------------------------------------------------------------------
    # 3. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading shards from {SHARD_TRAIN_DIR} ...")
    t0 = time.perf_counter()
    train_tokens = load_shard_tokens(SHARD_TRAIN_DIR)
    val_tokens   = load_shard_tokens(SHARD_VAL_DIR)
    print(f"  Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}  ({time.perf_counter()-t0:.1f}s)")

    train_dataset = NexaDataset(train_tokens, block_size=SEQ_LEN)
    val_dataset   = NexaDataset(val_tokens,   block_size=SEQ_LEN)
    print(f"  Train windows: {len(train_dataset):,} | Val windows: {len(val_dataset):,}")

    # ------------------------------------------------------------------
    # 4. Configure Trainer
    # ------------------------------------------------------------------
    # One epoch: exactly TARGET_STEPS steps
    # Add 1 to ensure we cover the full epoch even with integer division
    actual_max_steps = TARGET_STEPS  # 9229 steps covers one full epoch


    trainer_config = TrainerConfig(
        block_size       = SEQ_LEN,
        batch_size       = BATCH_SIZE,
        learning_rate    = LEARNING_RATE,
        min_lr           = MIN_LR,
        weight_decay     = WEIGHT_DECAY,
        beta1            = BETA1,
        beta2            = BETA2,
        eps              = 1e-8,
        grad_clip        = GRAD_CLIP,
        max_steps        = actual_max_steps,
        warmup_steps     = WARMUP_STEPS,
        grad_accum_steps = 1,
        eval_interval    = EVAL_INTERVAL,
        eval_steps       = EVAL_STEPS,
        save_interval    = SAVE_INTERVAL,
        checkpoint_dir   = CHECKPOINT_DIR,
        keep_last_n      = KEEP_LAST_N,
        log_interval     = LOG_INTERVAL,
        seed             = SEED,
        device           = "cpu",
    )

    print(f"\nTraining plan:")
    print(f"  max_steps={actual_max_steps} | warmup={WARMUP_STEPS}")
    print(f"  batch={BATCH_SIZE} x seq={SEQ_LEN} = {TRAIN_TOKENS_PER_STEP} tokens/step")
    print(f"  total budget: {actual_max_steps * TRAIN_TOKENS_PER_STEP:,} tokens")
    print(f"  estimated time: {actual_max_steps * TRAIN_TOKENS_PER_STEP / 978 / 3600:.1f}h at 978 tok/s")
    print(f"  eval every {EVAL_INTERVAL} steps | checkpoint every {SAVE_INTERVAL} steps")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model         = model,
        config        = trainer_config,
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        resume_from   = args.resume,
    )

    print(f"\n{'='*60}")
    print(f"  STARTING PILOT PRETRAINING — {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}")

    t_train_start = time.perf_counter()
    history = trainer.train()
    t_train_end   = time.perf_counter()
    wall_clock_s  = t_train_end - t_train_start

    actual_steps   = actual_max_steps
    tokens_trained = actual_steps * TRAIN_TOKENS_PER_STEP
    measured_tps   = round(tokens_trained / wall_clock_s)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {datetime.now(timezone.utc).isoformat()}")
    print(f"  Wall clock: {wall_clock_s/3600:.2f}h ({wall_clock_s:.0f}s)")
    print(f"  Throughput: {measured_tps:,} tok/s")
    print(f"  Initial loss: {history.initial_train_loss:.4f}")
    print(f"  Final loss:   {history.final_train_loss:.4f}")
    if history.final_val_loss:
        print(f"  Final val:    {history.final_val_loss:.4f} | PPL: {math.exp(history.final_val_loss):.2f}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 6. Find final checkpoint
    # ------------------------------------------------------------------
    latest_meta = torch.load(
        CHECKPOINT_DIR / "latest.pt", map_location="cpu", weights_only=True
    )
    final_ckpt = Path(latest_meta["latest_path"])
    print(f"\nFinal checkpoint: {final_ckpt}")

    # ------------------------------------------------------------------
    # 7. Post-training: reload + validation + inference
    # ------------------------------------------------------------------
    print("\n=== Post-Training Verification ===")
    post = reload_and_validate(
        checkpoint_path = final_ckpt,
        val_dataset     = val_dataset,
        model_config    = model_config,
        batch_size      = BATCH_SIZE,
        eval_steps      = EVAL_STEPS,
    )
    inference = post["inference"]
    if inference["all_valid"]:
        print(f"\n  Inference OK — {len(inference['samples'])} samples, all IDs in [0, 2048)")
        for i, s in enumerate(inference["samples"]):
            print(f"    Sample {i+1}: {s['generated_tokens'][:12]} ... (len={s['length']})")
    else:
        print("  WARNING: Invalid token IDs generated!")

    # ------------------------------------------------------------------
    # 8. Build report
    # ------------------------------------------------------------------
    t_total = time.perf_counter() - t_total_start

    # Read manifest
    manifest_snapshot = None
    if DATASET_MANIFEST.exists():
        m = json.loads(DATASET_MANIFEST.read_text())
        manifest_snapshot = {
            "generated":             m.get("generated"),
            "tokenizer_vocab_size":  m.get("tokenizer_vocab_size"),
            "bos_eos_included":      m.get("bos_eos_included"),
            "total_tokens":          m.get("total_tokens"),
            "train_tokens":          m.get("train_tokens"),
            "val_tokens":            m.get("val_tokens"),
        }

    checksums_snapshot = None
    if CHECKSUMS_FILE.exists():
        checksums_snapshot = CHECKSUMS_FILE.read_text().strip().splitlines()

    report = {
        "phase":     "8.9",
        "label":     "Nexa V1 Pilot Pretraining — One Full Epoch",
        "generated": datetime.now(timezone.utc).isoformat(),
        "git_commit": checks["git_commit"],
        "preflight":  checks,

        "model_config": {
            "vocab_size":      model_config.vocab_size,
            "d_model":         model_config.d_model,
            "n_heads":         model_config.n_heads,
            "n_layers":        model_config.n_layers,
            "d_ff":            model_config.d_ff,
            "max_seq_len":     model_config.max_seq_len,
            "norm_type":       model_config.norm_type,
            "pos_encoding":    model_config.pos_encoding,
            "tie_embeddings":  model_config.tie_embeddings,
            "bos_token_id":    model_config.bos_token_id,
            "eos_token_id":    model_config.eos_token_id,
        },
        "model_info": {
            "total_params":         model.num_parameters,
            "non_embedding_params": model.num_parameters_non_embedding,
            "initialized_from":     "scratch" if args.resume is None else str(args.resume),
        },

        "tokenizer": {
            "vocab_size":    2048,
            "bos_id":        1,
            "eos_id":        2,
            "pad_id":        0,
            "type":          "nexa_bpe_v1",
            "dir":           str(TOKENIZER_DIR),
        },

        "dataset": {
            "train_tokens_loaded":   len(train_tokens),
            "val_tokens_loaded":     len(val_tokens),
            "train_windows":         len(train_dataset),
            "val_windows":           len(val_dataset),
            "shard_train_dir":       str(SHARD_TRAIN_DIR),
            "shard_val_dir":         str(SHARD_VAL_DIR),
            "manifest":              manifest_snapshot,
            "checksums":             checksums_snapshot,
            "checksum_verified":     checks.get("all_shards_valid", False),
        },

        "training": {
            "seed":                SEED,
            "batch_size":          BATCH_SIZE,
            "seq_len":             SEQ_LEN,
            "tokens_per_step":     TRAIN_TOKENS_PER_STEP,
            "target_steps":        TARGET_STEPS,
            "actual_steps":        actual_steps,
            "warmup_steps":        WARMUP_STEPS,
            "learning_rate":       LEARNING_RATE,
            "min_lr":              MIN_LR,
            "weight_decay":        WEIGHT_DECAY,
            "grad_clip":           GRAD_CLIP,
            "beta1":               BETA1,
            "beta2":               BETA2,
            "eval_interval":       EVAL_INTERVAL,
            "save_interval":       SAVE_INTERVAL,
            "target_tokens":       TRAIN_TOKENS,
            "actual_tokens":       tokens_trained,
        },

        "results": {
            "initial_train_loss":   history.initial_train_loss,
            "final_train_loss":     history.final_train_loss,
            "train_loss_trajectory": history.train_loss,
            "val_loss_trajectory":   history.val_loss,
            "final_val_loss":        history.final_val_loss,
            "final_val_perplexity":  round(math.exp(history.final_val_loss), 4)
                                     if history.final_val_loss else None,
            "lr_trajectory":         history.lr_history,
        },

        "throughput": {
            "wall_clock_s":       round(wall_clock_s, 1),
            "wall_clock_h":       round(wall_clock_s / 3600, 3),
            "total_wall_clock_s": round(t_total, 1),
            "tokens_per_sec":     measured_tps,
        },

        "checkpoints": {
            "dir":          str(CHECKPOINT_DIR),
            "final":        str(final_ckpt),
            "interval":     SAVE_INTERVAL,
            "keep_last_n":  KEEP_LAST_N,
        },

        "post_training": post,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "nexa_v1_pilot_pretraining_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report: {report_path}")
    print(f"\n{'='*60}")
    print(f"  Phase 8.9 COMPLETE.")
    print(f"  Steps:      {actual_steps:,}")
    print(f"  Tokens:     {tokens_trained:,}")
    print(f"  Duration:   {wall_clock_s/3600:.2f}h")
    print(f"  Throughput: {measured_tps:,} tok/s")
    print(f"  Train loss: {history.initial_train_loss:.4f} -> {history.final_train_loss:.4f}")
    if history.final_val_loss:
        print(f"  Val loss:   {history.final_val_loss:.4f} | PPL: {math.exp(history.final_val_loss):.2f}")
    print(f"  Checkpoint: {final_ckpt}")
    print(f"  Reload val: {post['reproduced_val_loss']:.4f} (matches: {abs(post['reproduced_val_loss'] - (history.final_val_loss or 0)) < 0.01})")
    print(f"  Inference:  {'OK' if inference['all_valid'] else 'FAILED'}")
    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    main()
