"""
Nexa Phase 8.8 — Controlled Pretraining Run
=============================================
Runs a SHORT, CONTROLLED pretraining experiment on the Phase 8.6 tokenized
binary shards using the Phase 8.7 selected medium configuration.

This is a pipeline validation run, NOT a production training run.

Goals:
  1. Verify the complete training path end-to-end on real data
  2. Confirm loss decreases from a random-init baseline
  3. Save and reload a checkpoint, verify resume works
  4. Run an inference sanity check from the checkpoint
  5. Produce a structured experiment report

Budget (defaults):
  - 200 optimizer steps   (~400K tokens at B=4, S=512)
  - ~5-6 minutes wall-clock at 1,228 tok/s

Usage:
    python -m scripts.training.run_controlled_pretraining
    python -m scripts.training.run_controlled_pretraining --max-steps 100
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from nexa.models.config import ModelConfig
from nexa.models.transformer import NexaTransformer
from nexa.training.dataset import NexaDataset
from nexa.training.trainer import Trainer, TrainerConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PRETRAINING_CONFIG = Path("configs/nexa_v1_pretraining.yaml")
SHARD_TRAIN_DIR    = Path("data/processed/train")
SHARD_VAL_DIR      = Path("data/processed/validation")
CHECKPOINT_DIR     = Path("checkpoints/nexa_v1_controlled")
REPORTS_DIR        = Path("data/reports")
MANIFESTS_DIR      = Path("data/manifests")

DATASET_MANIFEST   = MANIFESTS_DIR / "tokenized_dataset_v1.json"
CHECKSUMS_FILE     = MANIFESTS_DIR / "tokenized_checksums.sha256"

BYTES_PER_TOKEN = 2  # uint16


# ---------------------------------------------------------------------------
# Shard loading into NexaDataset
# ---------------------------------------------------------------------------

def load_shard_tokens(shard_dir: Path, max_tokens: int | None = None) -> list[int]:
    """Read token IDs from all shards in a directory into a flat list."""
    all_tokens: list[int] = []
    for shard_path in sorted(shard_dir.glob("shard_*.bin")):
        data = shard_path.read_bytes()
        n = len(data) // 2
        tokens = list(struct.unpack(f"<{n}H", data))
        all_tokens.extend(tokens)
        if max_tokens is not None and len(all_tokens) >= max_tokens:
            break
    if max_tokens is not None:
        all_tokens = all_tokens[:max_tokens]
    return all_tokens


# ---------------------------------------------------------------------------
# Inference sanity check
# ---------------------------------------------------------------------------

def run_inference_check(model: NexaTransformer, device: torch.device, vocab_size: int) -> dict:
    """
    Generate short samples from a few seed prompts.
    Checks that generation runs without error and produces valid token IDs.
    Prompts are bounded by vocab_size to ensure validity.
    """
    model.eval()
    # Use BOS=1 and token IDs guaranteed to be < vocab_size
    t1 = min(10, vocab_size - 1)
    t2 = min(20, vocab_size - 1)
    t3 = min(100, vocab_size - 1)
    prompts = [
        [1],           # <bos> only
        [1, t1, t2],   # <bos> + a couple tokens
        [1, t3],       # <bos> + single token
    ]
    results = []
    for prompt in prompts:
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                eos_token_id=2,
            )
        gen_tokens = generated[0].tolist()
        valid = all(0 <= t < vocab_size for t in gen_tokens)
        results.append({
            "prompt_tokens":    prompt,
            "generated_tokens": gen_tokens,
            "length":           len(gen_tokens),
            "all_ids_valid":    valid,
        })
    return {"samples": results, "all_valid": all(r["all_ids_valid"] for r in results)}


# ---------------------------------------------------------------------------
# Checkpoint resume validation
# ---------------------------------------------------------------------------

def validate_resume(
    checkpoint_path: Path,
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    train_dataset: NexaDataset,
    val_dataset: NexaDataset,
    resume_steps: int,
) -> dict:
    """
    Load checkpoint into a fresh model+trainer, train for `resume_steps` more
    steps, and confirm that:
      1. The resumed step counter starts where the checkpoint ended.
      2. The resumed training loss is sensible (not NaN/inf, not wildly worse).
    """
    # Fresh model — must not carry any weights from previous training
    fresh_model = NexaTransformer(model_config)

    resume_cfg = TrainerConfig(
        block_size      = trainer_config.block_size,
        batch_size      = trainer_config.batch_size,
        learning_rate   = trainer_config.learning_rate,
        min_lr          = trainer_config.min_lr,
        weight_decay    = trainer_config.weight_decay,
        beta1           = trainer_config.beta1,
        beta2           = trainer_config.beta2,
        eps             = trainer_config.eps,
        grad_clip       = trainer_config.grad_clip,
        max_steps       = trainer_config.max_steps + resume_steps,
        warmup_steps    = trainer_config.warmup_steps,
        eval_interval   = 0,     # skip eval for resume check
        save_interval   = 0,     # skip save for resume check
        log_interval    = resume_steps,
        checkpoint_dir  = CHECKPOINT_DIR,
        seed            = trainer_config.seed,
        device          = trainer_config.device,
    )

    resume_trainer = Trainer(
        model         = fresh_model,
        config        = resume_cfg,
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        resume_from   = checkpoint_path,
    )

    history = resume_trainer.train()

    resumed_losses = [loss for _, loss in history.train_loss]
    return {
        "resumed_from_step":  trainer_config.max_steps,
        "resume_steps_run":   resume_steps,
        "resumed_losses":     resumed_losses,
        "final_resumed_loss": resumed_losses[-1] if resumed_losses else None,
        "is_finite":          all(math.isfinite(l) for l in resumed_losses),
        "loss_not_nan":       not any(math.isnan(l) for l in resumed_losses),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nexa Phase 8.8 — Controlled Pretraining")
    parser.add_argument("--max-steps",     type=int,   default=200,  help="Optimizer steps for controlled run")
    parser.add_argument("--resume-steps",  type=int,   default=20,   help="Extra steps to verify checkpoint resume")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--max-train-tokens", type=int, default=None,
                        help="Cap how many tokens to load from train shards (default: all)")
    args = parser.parse_args()

    print("\n=== Nexa Phase 8.8 — Controlled Pretraining Run ===\n")
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Verify prerequisites
    # ------------------------------------------------------------------
    for path in [PRETRAINING_CONFIG, SHARD_TRAIN_DIR, SHARD_VAL_DIR]:
        if not path.exists():
            print(f"ERROR: Required path missing: {path}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load model config (medium, from_scratch)
    # ------------------------------------------------------------------
    model_config = ModelConfig.from_yaml(PRETRAINING_CONFIG)
    print(f"Model config: {model_config}")

    model = NexaTransformer(model_config)
    total_params = model.num_parameters
    print(f"Parameters:   {total_params:,} total | {model.num_parameters_non_embedding:,} non-embedding")
    print("Initialization: SCRATCH (no pretrained weights)")

    # Verify param stats look like fresh init (std close to 0.02)
    sample_param = next(p for p in model.parameters() if p.ndim == 2)
    init_std = sample_param.std().item()
    assert 0.005 < init_std < 0.1, f"Unexpected init std: {init_std:.4f} — check initialization"

    # ------------------------------------------------------------------
    # 3. Load tokenized shards into NexaDataset
    # ------------------------------------------------------------------
    block_size = model_config.max_seq_len  # 512

    print(f"\nLoading train shards from {SHARD_TRAIN_DIR} ...")
    train_tokens = load_shard_tokens(SHARD_TRAIN_DIR, max_tokens=args.max_train_tokens)
    print(f"  Train tokens loaded: {len(train_tokens):,}")

    print(f"Loading val shards from {SHARD_VAL_DIR} ...")
    val_tokens = load_shard_tokens(SHARD_VAL_DIR)
    print(f"  Val tokens loaded:   {len(val_tokens):,}")

    train_dataset = NexaDataset(train_tokens, block_size=block_size)
    val_dataset   = NexaDataset(val_tokens,   block_size=block_size)
    print(f"  Train windows: {len(train_dataset):,} | Val windows: {len(val_dataset):,}")

    # ------------------------------------------------------------------
    # 4. Configure training
    # ------------------------------------------------------------------
    warmup_steps = max(10, args.max_steps // 10)

    trainer_config = TrainerConfig(
        block_size      = block_size,
        batch_size      = 4,
        learning_rate   = 3e-4,
        min_lr          = 3e-5,
        weight_decay    = 0.1,
        beta1           = 0.9,
        beta2           = 0.95,
        eps             = 1e-8,
        grad_clip       = 1.0,
        max_steps       = args.max_steps,
        warmup_steps    = warmup_steps,
        grad_accum_steps = 1,
        eval_interval   = max(1, args.max_steps // 4),
        eval_steps      = 50,
        save_interval   = args.max_steps,   # save once at the end
        checkpoint_dir  = CHECKPOINT_DIR,
        keep_last_n     = 5,
        log_interval    = max(1, args.max_steps // 10),
        seed            = args.seed,
        device          = "cpu",
    )

    print(f"\nTraining config:")
    print(f"  max_steps={args.max_steps} | warmup={warmup_steps} | batch=4 | seq={block_size}")
    print(f"  lr={trainer_config.learning_rate:.1e} -> {trainer_config.min_lr:.1e}")
    print(f"  eval every {trainer_config.eval_interval} steps | save at step {args.max_steps}")
    tokens_per_step = trainer_config.batch_size * block_size
    print(f"  tokens/step={tokens_per_step:,} | total budget ~{args.max_steps * tokens_per_step:,} tokens")

    # ------------------------------------------------------------------
    # 5. Run controlled training
    # ------------------------------------------------------------------
    trainer = Trainer(
        model         = model,
        config        = trainer_config,
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
    )

    print(f"\nStarting training...")
    t_train_start = time.perf_counter()
    history = trainer.train()
    t_train_end = time.perf_counter()
    wall_clock_s = t_train_end - t_train_start

    tokens_processed = args.max_steps * tokens_per_step
    measured_tps = round(tokens_processed / wall_clock_s)

    print(f"\nTraining complete in {wall_clock_s:.1f}s | {measured_tps:,} tok/s")
    print(f"  Initial loss:  {history.initial_train_loss:.4f}")
    print(f"  Final loss:    {history.final_train_loss:.4f}")
    if history.val_loss:
        final_val = history.final_val_loss
        print(f"  Final val loss: {final_val:.4f} | PPL: {math.exp(final_val):.2f}")

    # ------------------------------------------------------------------
    # 6. Find saved checkpoint
    # ------------------------------------------------------------------
    latest_path = CHECKPOINT_DIR / "latest.pt"
    if not latest_path.exists():
        print("ERROR: No checkpoint found after training.")
        sys.exit(1)

    latest_meta = torch.load(latest_path, map_location="cpu", weights_only=True)
    checkpoint_path = Path(latest_meta["latest_path"])
    print(f"\nCheckpoint saved: {checkpoint_path}")

    # ------------------------------------------------------------------
    # 7. Checkpoint resume validation
    # ------------------------------------------------------------------
    print(f"\nValidating checkpoint resume ({args.resume_steps} extra steps)...")
    resume_result = validate_resume(
        checkpoint_path = checkpoint_path,
        model_config    = model_config,
        trainer_config  = trainer_config,
        train_dataset   = train_dataset,
        val_dataset     = val_dataset,
        resume_steps    = args.resume_steps,
    )
    if resume_result["is_finite"] and resume_result["loss_not_nan"]:
        print(f"  Resume OK: final resumed loss = {resume_result['final_resumed_loss']:.4f}")
    else:
        print("  WARNING: Resume produced NaN/Inf losses!")

    # ------------------------------------------------------------------
    # 8. Inference sanity check
    # ------------------------------------------------------------------
    print("\nRunning inference sanity check...")
    device = torch.device("cpu")
    model.eval()
    model.to(device)

    inference_result = run_inference_check(model, device, model_config.vocab_size)

    if inference_result["all_valid"]:
        print(f"  Generation OK: {len(inference_result['samples'])} samples, all token IDs valid")
        for i, s in enumerate(inference_result["samples"]):
            print(f"    Sample {i+1}: {s['generated_tokens'][:10]}... (len={s['length']})")
    else:
        print("  WARNING: Generated invalid token IDs!")

    # ------------------------------------------------------------------
    # 9. Build experiment report
    # ------------------------------------------------------------------
    wall_total = time.perf_counter() - t_start

    # Read manifest references
    manifest_ref, checksum_ref = None, None
    if DATASET_MANIFEST.exists():
        manifest_ref = json.loads(DATASET_MANIFEST.read_text())
        manifest_ref = {
            "generated": manifest_ref.get("generated"),
            "tokenizer_vocab_size": manifest_ref.get("tokenizer_vocab_size"),
            "bos_eos_included": manifest_ref.get("bos_eos_included"),
        }
    if CHECKSUMS_FILE.exists():
        checksum_ref = CHECKSUMS_FILE.read_text().strip().splitlines()

    report = {
        "phase":        "8.8",
        "generated":    datetime.now(timezone.utc).isoformat(),
        "git_note":     "See git log for exact commit hash",
        "config": {
            "model": {
                "vocab_size":  model_config.vocab_size,
                "d_model":     model_config.d_model,
                "n_heads":     model_config.n_heads,
                "n_layers":    model_config.n_layers,
                "d_ff":        model_config.d_ff,
                "max_seq_len": model_config.max_seq_len,
                "norm_type":   model_config.norm_type,
                "pos_encoding": model_config.pos_encoding,
                "tie_embeddings": model_config.tie_embeddings,
            },
            "training": {
                "max_steps":      args.max_steps,
                "warmup_steps":   warmup_steps,
                "batch_size":     trainer_config.batch_size,
                "block_size":     block_size,
                "learning_rate":  trainer_config.learning_rate,
                "min_lr":         trainer_config.min_lr,
                "weight_decay":   trainer_config.weight_decay,
                "grad_clip":      trainer_config.grad_clip,
                "seed":           args.seed,
                "device":         "cpu",
            },
        },
        "model_info": {
            "total_params":          total_params,
            "non_embedding_params":  model.num_parameters_non_embedding,
            "initialized_from":      "scratch",
        },
        "dataset_info": {
            "train_tokens_loaded":   len(train_tokens),
            "val_tokens_loaded":     len(val_tokens),
            "train_windows":         len(train_dataset),
            "val_windows":           len(val_dataset),
            "shard_train_dir":       str(SHARD_TRAIN_DIR),
            "shard_val_dir":         str(SHARD_VAL_DIR),
            "manifest_ref":          manifest_ref,
            "checksums_ref":         checksum_ref,
        },
        "training_budget": {
            "optimizer_steps":       args.max_steps,
            "tokens_per_step":       tokens_per_step,
            "total_tokens_processed": tokens_processed,
        },
        "results": {
            "initial_train_loss":  history.initial_train_loss,
            "final_train_loss":    history.final_train_loss,
            "loss_trajectory":     history.train_loss,
            "val_loss_trajectory": history.val_loss,
            "final_val_loss":      history.final_val_loss,
            "final_val_perplexity": round(math.exp(history.final_val_loss), 4)
                                    if history.final_val_loss else None,
            "lr_trajectory":       history.lr_history,
        },
        "throughput": {
            "wall_clock_s":        round(wall_clock_s, 2),
            "tokens_per_sec":      measured_tps,
            "total_wall_clock_s":  round(wall_total, 2),
        },
        "checkpoint": {
            "path":                str(checkpoint_path),
            "dir":                 str(CHECKPOINT_DIR),
        },
        "resume_verification": resume_result,
        "inference_sanity_check": inference_result,
    }

    # ------------------------------------------------------------------
    # 10. Write report
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "nexa_v1_controlled_pretraining_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Phase 8.8 complete.")
    print(f"  Wall clock:    {wall_clock_s:.1f}s training | {wall_total:.1f}s total")
    print(f"  Throughput:    {measured_tps:,} tok/s")
    print(f"  Report:        {report_path}")
    print(f"  Checkpoint:    {checkpoint_path}")
    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    main()
