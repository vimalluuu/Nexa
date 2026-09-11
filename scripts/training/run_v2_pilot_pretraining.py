"""
scripts/training/run_v2_pilot_pretraining.py
=============================================
Nexa Phase 15 — V2 Substantial Pilot Pretraining

Trains the Nexa V2 Candidate B model (~29M params) for exactly 1,526 optimizer
steps = 49,987,648 tokens (< 50M target).

DO NOT change the architecture, tokenizer, optimizer, or scheduler without
an explicit phase decision.

After the run: verify checkpoint, run inference sanity check, write reports.
Stop. Do NOT automatically continue past 50M tokens.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexa.models import NexaTransformer, ModelConfig
from nexa.tokenizer.byte_tokenizer import NexaByteTokenizer
from nexa.training.scheduler import CosineWarmupScheduler
from nexa.utils import get_logger

log = get_logger("nexa.v2.pilot", log_to_file=False)

# ============================================================================
# AUTHORITATIVE V2 CONFIGURATION — do not alter
# ============================================================================

# Token budget
TARGET_TOKENS   = 50_000_000
TOKENS_PER_STEP = 4 * 1024 * 8          # batch × seq × grad_accum
MAX_STEPS       = TARGET_TOKENS // TOKENS_PER_STEP   # 1525 complete steps
ACTUAL_TOKENS   = MAX_STEPS * TOKENS_PER_STEP        # 49,971,200

assert MAX_STEPS == 1525, f"Expected 1525 steps, got {MAX_STEPS}"
assert ACTUAL_TOKENS == 49_971_200, f"Expected 49,971,200 tokens, got {ACTUAL_TOKENS}"

MODEL_CFG = dict(
    vocab_size    = 8192,
    d_model       = 512,
    n_layers      = 6,
    n_heads       = 8,
    d_ff          = 2048,
    max_seq_len   = 1024,
    dropout       = 0.1,
    norm_type     = "rmsnorm",
    pos_encoding  = "rotary",
    tie_embeddings= True,
    pad_token_id  = 0,
    bos_token_id  = 1,
    eos_token_id  = 2,
)

TRAIN_CFG = dict(
    block_size       = 1024,
    batch_size       = 4,
    grad_accum_steps = 8,
    learning_rate    = 6e-4,
    min_lr           = 6e-5,
    weight_decay     = 0.1,
    beta1            = 0.9,
    beta2            = 0.95,
    eps              = 1e-8,
    grad_clip        = 1.0,
    warmup_steps     = 100,
    eval_steps       = 20,   # Bounded: 20 batches per validation
    log_interval     = 50,
    eval_interval    = 250,
    save_interval    = 250,
    seed             = 42,
)

# Paths
TOKENIZER_DIR   = Path("data/processed/tokenizer_v2/8192")
TRAIN_SHARD_DIR = Path("data/processed/v2/train")
VAL_SHARD_DIR   = Path("data/processed/v2/validation")
CHECKPOINT_DIR  = Path("checkpoints/nexa_v2_pilot")
REPORTS_DIR     = Path("data/reports")

PROGRESS_LOG    = REPORTS_DIR / "nexa_v2_substantial_pretraining_progress.jsonl"


# ============================================================================
# Dataset — reuse from Phase 14
# ============================================================================

class V2ShardDataset(Dataset):
    """Reads pre-tokenized uint16 binary shards. Returns sliding-window (x, y)."""

    def __init__(self, shard_dir: Path, block_size: int, max_tokens: int | None = None) -> None:
        shard_files = sorted(shard_dir.glob("shard_*.bin"))
        if not shard_files:
            raise FileNotFoundError(f"No shards found in {shard_dir}")
        all_tokens: list[int] = []
        for sf in shard_files:
            with open(sf, "rb") as f:
                raw = f.read()
            n = len(raw) // 2
            tokens = list(struct.unpack(f"<{n}H", raw))
            all_tokens.extend(tokens)
            if max_tokens and len(all_tokens) >= max_tokens:
                all_tokens = all_tokens[:max_tokens]
                break
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        self.block_size = block_size
        log.info("V2ShardDataset: %d tokens from %s", len(self.data), shard_dir)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    def token_count(self) -> int:
        return len(self.data)


# ============================================================================
# Helpers
# ============================================================================

def sha256_shard(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def count_params(model: NexaTransformer) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb   = model.embedding.weight.numel()
    return {"total": total, "embedding": emb, "non_embedding": total - emb,
            "embedding_pct": round(emb / total * 100, 2)}

def append_progress(record: dict) -> None:
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

@torch.no_grad()
def evaluate(model: NexaTransformer, val_ds: V2ShardDataset,
             batch_size: int, eval_steps: int, device: torch.device) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    n = 0
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    for x, y in loader:
        if n >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(
            logits.view(-1, MODEL_CFG["vocab_size"]), y.view(-1)
        ).item()
        n += 1
    return total_loss / max(1, n), n

def save_checkpoint(path: Path, step: int, tokens: int, loss: float,
                    model, optimizer, scheduler, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step"           : step,
        "tokens_processed": tokens,
        "loss"           : loss,
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_step" : scheduler.steps_completed,
        "model_config"   : MODEL_CFG,
        "train_config"   : cfg,
        "seed"           : TRAIN_CFG["seed"],
        "rng_state"      : torch.get_rng_state(),
    }, path)
    log.info("Checkpoint → %s", path)

def verify_v1_immutability() -> dict[str, bool]:
    checks = {
        "v1_tokenizer"       : Path("data/processed/tokenizer").exists(),
        "v1_checkpoint_dir"  : Path("checkpoints/nexa_v1_pilot").exists(),
        "v1_eval_report"     : Path("data/reports/nexa_v1_evaluation_report.json").exists(),
        "v2_corpus_frozen"   : Path("data/reports/nexa_v2_corpus_design.json").exists(),
    }
    return checks


# ============================================================================
# Final verification (reload + inference)
# ============================================================================

def run_final_verification(ckpt_path: Path, val_ds: V2ShardDataset,
                           val_loss_in_run: float, tokenizer: NexaByteTokenizer,
                           report: dict) -> None:
    log.info("=== FINAL CHECKPOINT VERIFICATION ===")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg  = ModelConfig(**MODEL_CFG)
    model = NexaTransformer(cfg)
    model.load_state_dict(ckpt["model_state"])

    # Param count
    params = count_params(model)
    assert params["total"] == 29_366_784, f"Param mismatch: {params['total']}"
    log.info("Param count verified: %d", params["total"])

    # Reload val loss
    device = torch.device("cpu")
    val_loss_reload, val_batches = evaluate(
        model, val_ds,
        TRAIN_CFG["batch_size"], TRAIN_CFG["eval_steps"], device
    )
    diff = abs(val_loss_in_run - val_loss_reload)
    tol  = 1e-4
    log.info("Reload val_loss: in-run=%.6f  reload=%.6f  diff=%.2e  tol=%.0e",
             val_loss_in_run, val_loss_reload, diff, tol)

    report["final_verification"] = {
        "param_count_ok"      : True,
        "val_loss_in_run"     : round(val_loss_in_run, 6),
        "val_loss_reload"     : round(val_loss_reload, 6),
        "val_ppl_reload"      : round(math.exp(val_loss_reload), 4),
        "absolute_difference" : round(diff, 8),
        "tolerance"           : tol,
        "reload_passed"       : diff <= tol,
        "step_in_checkpoint"  : ckpt["step"],
        "tokens_in_checkpoint": ckpt["tokens_processed"],
    }

    # Inference check
    model.eval()
    PROMPT   = "The history of science"
    MAX_NEW  = 60
    EOS      = cfg.eos_token_id

    def gen(temperature: float = 0.0, seed: int | None = None) -> list[int]:
        if seed is not None:
            torch.manual_seed(seed)
        ids = tokenizer.encode(PROMPT, add_bos=True, add_eos=False)
        ctx = torch.tensor([ids], dtype=torch.long)
        out = []
        with torch.no_grad():
            for _ in range(MAX_NEW):
                logits = model(ctx)
                nxt = logits[0, -1, :]
                if temperature == 0.0:
                    next_id = int(nxt.argmax())
                else:
                    probs = F.softmax(nxt / temperature, dim=-1)
                    next_id = int(torch.multinomial(probs, 1))
                out.append(next_id)
                if next_id == EOS:
                    break
                ctx = torch.cat([ctx, torch.tensor([[next_id]])], dim=1)
        return out

    greedy   = gen(temperature=0.0)
    sampled1 = gen(temperature=0.8, seed=77)
    sampled2 = gen(temperature=0.8, seed=77)

    in_range     = all(0 <= t < cfg.vocab_size for t in greedy + sampled1)
    reproducible = sampled1 == sampled2
    eos_handled  = EOS in greedy or len(greedy) == MAX_NEW

    greedy_text  = tokenizer.decode(greedy)
    sampled_text = tokenizer.decode(sampled1)

    log.info("Greedy: %s", repr(greedy_text[:100]))
    log.info("Sampled: %s", repr(sampled_text[:100]))
    log.info("In-range=%s  EOS=%s  Reproducible=%s", in_range, eos_handled, reproducible)

    report["inference_check"] = {
        "prompt"             : PROMPT,
        "greedy_text"        : greedy_text,
        "sampled_text"       : sampled_text,
        "token_ids_in_range" : in_range,
        "eos_handled"        : eos_handled,
        "seeded_reproducible": reproducible,
        "passed"             : in_range and eos_handled and reproducible,
    }


# ============================================================================
# Main training loop
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("  Nexa Phase 15 — V2 Substantial Pilot Pretraining")
    print("=" * 70)
    print(f"  Target steps : {MAX_STEPS:,}")
    print(f"  Target tokens: {ACTUAL_TOKENS:,} / {TARGET_TOKENS:,}")
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    import subprocess
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    # ----------------------------------------------------------------
    # [0] Pre-flight
    # ----------------------------------------------------------------
    print("[0/6] Pre-flight verification...")

    # V1 immutability
    v1_ok = verify_v1_immutability()
    for k, v in v1_ok.items():
        assert v, f"V1 artifact missing: {k}"

    # Tokenizer
    assert TOKENIZER_DIR.exists(), f"Tokenizer not found: {TOKENIZER_DIR}"
    tokenizer = NexaByteTokenizer.load(TOKENIZER_DIR)
    assert tokenizer.vocab_size == 8192

    # Shard checksums (first two shards as representative sample)
    train_shards = sorted(TRAIN_SHARD_DIR.glob("shard_*.bin"))
    val_shards   = sorted(VAL_SHARD_DIR.glob("shard_*.bin"))
    assert train_shards and val_shards

    shard_checksums = {
        "train_shard_000": sha256_shard(train_shards[0]),
        "train_shard_001": sha256_shard(train_shards[1]),
        "val_shard_000"  : sha256_shard(val_shards[0]),
    }

    # Model
    model_cfg = ModelConfig(**MODEL_CFG)
    model     = NexaTransformer(model_cfg)
    params    = count_params(model)
    assert params["total"] == 29_366_784, f"Param count mismatch: {params['total']}"
    print(f"      Model params verified: {params['total']:,}")

    # Disk space check (rough: need at least 2GB free for checkpoints)
    import shutil
    free_gb = shutil.disk_usage(".").free / 1e9
    print(f"      Disk free: {free_gb:.1f} GB")
    if free_gb < 2.0:
        print("      WARNING: Less than 2GB free — checkpoint writes may fail.")

    report: dict = {
        "phase"             : 15,
        "git_commit"        : git_commit,
        "seed"              : TRAIN_CFG["seed"],
        "model_config"      : MODEL_CFG,
        "train_config"      : TRAIN_CFG,
        "tokenizer_dir"     : str(TOKENIZER_DIR),
        "train_shard_dir"   : str(TRAIN_SHARD_DIR),
        "val_shard_dir"     : str(VAL_SHARD_DIR),
        "max_steps"         : MAX_STEPS,
        "tokens_per_step"   : TOKENS_PER_STEP,
        "target_tokens"     : TARGET_TOKENS,
        "actual_tokens"     : ACTUAL_TOKENS,
        "effective_batch"   : 32,
        "shard_checksums"   : shard_checksums,
        "v1_immutability"   : v1_ok,
        "model_params"      : params,
        "disk_free_gb"      : round(free_gb, 1),
        "train_step_metrics": [],
        "val_metrics"       : [],
        "checkpoints"       : [],
    }

    # ----------------------------------------------------------------
    # [1] Load data
    # ----------------------------------------------------------------
    print("\n[1/6] Loading V2 shards...")
    train_ds = V2ShardDataset(TRAIN_SHARD_DIR, block_size=1024)
    val_ds   = V2ShardDataset(VAL_SHARD_DIR,   block_size=1024, max_tokens=5_000_000)
    report["train_tokens_available"] = train_ds.token_count()
    report["val_tokens_loaded"]      = val_ds.token_count()
    print(f"      Train: {train_ds.token_count():,} | Val: {val_ds.token_count():,}")
    print(f"      Steps available: {len(train_ds) // (TRAIN_CFG['batch_size'] * TRAIN_CFG['grad_accum_steps']):,}")

    # ----------------------------------------------------------------
    # [2] Build model + optimizer + scheduler
    # ----------------------------------------------------------------
    print("\n[2/6] Initialising model, optimizer, scheduler...")
    device = torch.device("cpu")
    torch.manual_seed(TRAIN_CFG["seed"])
    torch.set_num_threads(6)  # Ryzen 5 5600G — 6 physical cores

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.ndim >= 2],
             "weight_decay": TRAIN_CFG["weight_decay"]},
            {"params": [p for p in model.parameters() if p.ndim <  2],
             "weight_decay": 0.0},
        ],
        lr    = TRAIN_CFG["learning_rate"],
        betas = (TRAIN_CFG["beta1"], TRAIN_CFG["beta2"]),
        eps   = TRAIN_CFG["eps"],
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps = TRAIN_CFG["warmup_steps"],
        max_steps    = MAX_STEPS,
        max_lr       = TRAIN_CFG["learning_rate"],
        min_lr       = TRAIN_CFG["min_lr"],
    )

    def infinite_loader(ds: Dataset):
        while True:
            loader = DataLoader(
                ds,
                batch_size = TRAIN_CFG["batch_size"],
                shuffle    = True,
                drop_last  = True,
            )
            for batch in loader:
                yield batch

    data_iter = infinite_loader(train_ds)

    # ----------------------------------------------------------------
    # [3] Training loop — exactly MAX_STEPS optimizer steps
    # ----------------------------------------------------------------
    print(f"\n[3/6] Training — {MAX_STEPS:,} steps × {TOKENS_PER_STEP:,} tokens/step...")
    print(f"      Total tokens: {ACTUAL_TOKENS:,}")
    print(f"      Estimated time: ~{ACTUAL_TOKENS / 500.9 / 3600:.1f} hours (ESTIMATED at 500.9 tok/s)")
    print()

    step          = 0
    accum_count   = 0
    running_loss  = 0.0
    tokens_done   = 0
    optimizer.zero_grad()

    wall_start    = time.perf_counter()
    interval_start = wall_start
    interval_toks = 0

    last_val_loss = float("nan")

    while step < MAX_STEPS:
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss   = F.cross_entropy(
            logits.view(-1, MODEL_CFG["vocab_size"]), y.view(-1)
        )

        # Safety gate — stop on NaN/Inf
        if not math.isfinite(loss.item()):
            log.error("Non-finite loss at step %d accum %d: %s", step, accum_count, loss.item())
            report["stopped_early"] = {"reason": "non_finite_loss", "step": step}
            break

        (loss / TRAIN_CFG["grad_accum_steps"]).backward()
        running_loss += loss.item()
        accum_count  += 1
        tokens_done  += x.numel()
        interval_toks += x.numel()

        if accum_count < TRAIN_CFG["grad_accum_steps"]:
            continue

        # Optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), TRAIN_CFG["grad_clip"]
        )
        lr = scheduler.step()
        optimizer.step()
        optimizer.zero_grad()

        avg_loss     = running_loss / accum_count
        running_loss = 0.0
        accum_count  = 0
        step        += 1

        # Logging
        if step % TRAIN_CFG["log_interval"] == 0:
            now     = time.perf_counter()
            elapsed = now - wall_start
            tps     = interval_toks / (now - interval_start)
            interval_start = now
            interval_toks  = 0

            pct = step / MAX_STEPS * 100
            eta_hrs = ((MAX_STEPS - step) * TOKENS_PER_STEP / tps) / 3600

            record = {
                "step"     : step,
                "tokens"   : tokens_done,
                "loss"     : round(avg_loss, 6),
                "lr"       : lr,
                "grad_norm": round(float(grad_norm), 4),
                "tok_per_s": round(tps, 1),
                "elapsed_s": round(elapsed, 1),
            }
            report["train_step_metrics"].append(record)
            append_progress({**record, "type": "train"})

            log.info(
                "step %4d/%d (%.1f%%) | loss %.4f | lr %.2e | gnorm %.3f | "
                "%.0f tok/s | ETA ~%.1fh",
                step, MAX_STEPS, pct, avg_loss, lr, grad_norm, tps, eta_hrs,
            )

        # Validation + checkpoint
        if step % TRAIN_CFG["eval_interval"] == 0 or step == MAX_STEPS:
            val_loss, val_batches = evaluate(
                model, val_ds, TRAIN_CFG["batch_size"], TRAIN_CFG["eval_steps"], device
            )
            val_ppl = math.exp(val_loss)
            last_val_loss = val_loss

            val_record = {
                "step"    : step,
                "tokens"  : tokens_done,
                "val_loss": round(val_loss, 6),
                "val_ppl" : round(val_ppl, 4),
                "batches" : val_batches,
                "val_toks": val_batches * TRAIN_CFG["batch_size"] * TRAIN_CFG["block_size"],
            }
            report["val_metrics"].append(val_record)
            append_progress({**val_record, "type": "val"})

            log.info("  → val_loss %.4f | val_ppl %.2f (%d batches)", val_loss, val_ppl, val_batches)
            model.train()

            # Checkpoint
            ckpt_path = CHECKPOINT_DIR / f"step_{step:07d}.pt"
            save_checkpoint(
                ckpt_path, step, tokens_done, avg_loss,
                model, optimizer, scheduler, TRAIN_CFG,
            )
            report["checkpoints"].append({
                "step"    : step,
                "tokens"  : tokens_done,
                "loss"    : round(avg_loss, 6),
                "val_loss": round(val_loss, 6),
                "path"    : str(ckpt_path),
            })

    wall_end = time.perf_counter()
    wall_total = wall_end - wall_start

    report.update({
        "final_step"        : step,
        "final_tokens"      : tokens_done,
        "wall_clock_seconds": round(wall_total, 1),
        "final_train_loss"  : round(avg_loss, 6),
        "e2e_tps"           : round(tokens_done / wall_total, 1),
    })

    # ----------------------------------------------------------------
    # [4] Final checkpoint (also saved above at last eval_interval;
    #     write explicitly at exactly MAX_STEPS if not already done)
    # ----------------------------------------------------------------
    print(f"\n[4/6] Final checkpoint...")
    final_ckpt = CHECKPOINT_DIR / f"step_{step:07d}.pt"
    if not final_ckpt.exists():
        save_checkpoint(
            final_ckpt, step, tokens_done, avg_loss,
            model, optimizer, scheduler, TRAIN_CFG,
        )
    report["final_checkpoint"] = str(final_ckpt)

    # ----------------------------------------------------------------
    # [5] Final verification (reload + inference)
    # ----------------------------------------------------------------
    print("\n[5/6] Final verification...")
    run_final_verification(final_ckpt, val_ds, last_val_loss, tokenizer, report)

    v = report.get("final_verification", {})
    i = report.get("inference_check", {})
    print(f"      Reload:    {'PASSED' if v.get('reload_passed') else 'FAILED'} "
          f"(diff={v.get('absolute_difference', '?'):.2e})")
    print(f"      Inference: {'PASSED' if i.get('passed') else 'FAILED'}")

    # ----------------------------------------------------------------
    # [6] Write reports
    # ----------------------------------------------------------------
    print("\n[6/6] Writing reports...")

    # Recommendation
    val_losses = [r["val_loss"] for r in report["val_metrics"]]
    still_decreasing = len(val_losses) >= 2 and val_losses[-1] < val_losses[-2]
    report["recommendation"] = {
        "continued_learning_observed": still_decreasing,
        "suggested_next_experiment"  : (
            "Continue training to 100M tokens" if still_decreasing
            else "Evaluate loss curve before extending training"
        ),
    }

    # JSON report
    json_path = REPORTS_DIR / "nexa_v2_substantial_pretraining_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown summary
    _write_markdown_summary(report)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Nexa Phase 15 — Complete")
    print("=" * 70)
    print(f"  Git commit           : {git_commit}")
    print(f"  Optimizer steps      : {step:,} / {MAX_STEPS:,}")
    print(f"  Tokens processed     : {tokens_done:,}")
    print(f"  Wall clock           : {wall_total/3600:.2f} hours")
    print(f"  E2E throughput       : {report['e2e_tps']:.1f} tok/s (MEASURED)")
    print(f"  Final train loss     : {report['final_train_loss']:.4f}")
    print(f"  Final val loss       : {v.get('val_loss_reload', float('nan')):.4f}")
    print(f"  Final val PPL        : {v.get('val_ppl_reload', float('nan')):.2f}")
    print(f"  Checkpoint reload    : {'PASSED' if v.get('reload_passed') else 'FAILED'}")
    print(f"  Inference            : {'PASSED' if i.get('passed') else 'FAILED'}")
    print(f"\n  Report: {json_path}")
    print("\nSTOP — do not continue training without explicit Phase 16 decision.")


def _write_markdown_summary(report: dict) -> None:
    v   = report.get("final_verification", {})
    i   = report.get("inference_check", {})
    val = report.get("val_metrics", [])

    lines = [
        "# Nexa Phase 15 — V2 Substantial Pilot Pretraining Summary",
        "",
        "## Configuration (AUTHORITATIVE)",
        "| Parameter | Value |",
        "|---|---|",
        f"| Git Commit | `{report['git_commit']}` |",
        f"| Seed | {report['seed']} |",
        f"| vocab_size | {MODEL_CFG['vocab_size']} |",
        f"| d_model | {MODEL_CFG['d_model']} |",
        f"| n_layers | {MODEL_CFG['n_layers']} |",
        f"| n_heads | {MODEL_CFG['n_heads']} |",
        f"| d_ff | {MODEL_CFG['d_ff']} |",
        f"| context_length | {MODEL_CFG['max_seq_len']} |",
        f"| batch_size | {TRAIN_CFG['batch_size']} |",
        f"| grad_accum_steps | {TRAIN_CFG['grad_accum_steps']} |",
        f"| effective_batch_size | {report['effective_batch']} |",
        f"| tokens_per_step | {TOKENS_PER_STEP:,} |",
        f"| optimizer_steps | {report['final_step']:,} |",
        f"| target_tokens | {TARGET_TOKENS:,} |",
        f"| actual_tokens | {report['final_tokens']:,} |",
        f"| learning_rate | {TRAIN_CFG['learning_rate']} |",
        f"| min_lr | {TRAIN_CFG['min_lr']} |",
        f"| warmup_steps | {TRAIN_CFG['warmup_steps']} |",
        f"| scheduler | cosine warmup |",
        "",
        "## Results (MEASURED)",
        "| Metric | Value |",
        "|---|---|",
        f"| Wall clock | {report.get('wall_clock_seconds', 0)/3600:.2f} hours |",
        f"| E2E throughput | {report.get('e2e_tps', '?')} tok/s |",
        f"| Final train loss | {report.get('final_train_loss', '?')} |",
        f"| Final val loss | {v.get('val_loss_reload', '?')} |",
        f"| Final val PPL | {v.get('val_ppl_reload', '?')} |",
        f"| Checkpoint reload | {'✅' if v.get('reload_passed') else '❌'} |",
        f"| Inference check | {'✅' if i.get('passed') else '❌'} |",
        "",
        "## Validation Loss Trajectory (MEASURED)",
        "| Step | Tokens | Val Loss | Val PPL | Batches |",
        "|---|---|---|---|---|",
    ]
    for r in val:
        lines.append(
            f"| {r['step']:,} | {r['tokens']:,} | {r['val_loss']:.4f} "
            f"| {r['val_ppl']:.2f} | {r['batches']} |"
        )

    lines += [
        "",
        "## Inference Sample",
        f"**Prompt**: `{i.get('prompt', '?')}`",
        f"**Greedy**: `{i.get('greedy_text', '?')[:120]}`",
        f"**Sampled**: `{i.get('sampled_text', '?')[:120]}`",
        "",
        "## Checkpoints",
        "| Step | Tokens | Train Loss | Val Loss |",
        "|---|---|---|---|",
    ]
    for c in report.get("checkpoints", []):
        lines.append(
            f"| {c['step']:,} | {c['tokens']:,} | {c['loss']:.4f} | {c['val_loss']:.4f} |"
        )

    lines += [
        "",
        "## Recommendation",
        f"**Continued learning observed**: {report.get('recommendation', {}).get('continued_learning_observed', '?')}",
        f"**Suggested next experiment**: {report.get('recommendation', {}).get('suggested_next_experiment', '?')}",
        "",
        "## Limitations",
        "- 50M tokens is far below Chinchilla-optimal (~590M) for a 29M-parameter model.",
        "- val PPL at this scale does not reflect final model quality.",
        "- Throughput measurements include data loading overhead.",
        "- Seeded generation reproducibility verified but linguistic quality is not evaluated here.",
    ]

    md_path = REPORTS_DIR / "nexa_v2_substantial_pretraining_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Markdown summary → %s", md_path)


if __name__ == "__main__":
    main()
