"""
Nexa Phase 8.7 — Pretraining Readiness Benchmark
==================================================
Benchmarks 3 candidate model configurations against the real Phase 8.6 binary
token shards. Measures:
  - Exact parameter counts (total and non-embedding)
  - Memory estimates (model, optimizer, activation)
  - Forward time per step
  - Backward time per step
  - Optimizer step time
  - Tokens per second
  - Validation loss (short run)

No pretraining is started. Each config gets a fixed small number of steps
(default: 20 warmup + 50 benchmark steps) to produce stable throughput estimates.

Usage:
    python -m scripts.training.benchmark_configs
    python -m scripts.training.benchmark_configs --steps 30
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexa.models.config import ModelConfig
from nexa.models.transformer import NexaTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARD_TRAIN_DIR = Path("data/processed/train")
SHARD_VAL_DIR   = Path("data/processed/validation")
CONFIGS_DIR     = Path("configs")
REPORTS_DIR     = Path("data/reports")

CANDIDATE_CONFIGS = [
    ("small",  Path("configs/nexa_v1_small.yaml"),  {"batch_size": 4, "seq_len": 256}),
    ("medium", Path("configs/nexa_v1_medium.yaml"), {"batch_size": 4, "seq_len": 512}),
    ("large",  Path("configs/nexa_v1_large.yaml"),  {"batch_size": 2, "seq_len": 512}),
]

BYTES_PER_TOKEN   = 2   # uint16
FLOAT32_BYTES     = 4
ADAMW_STATE_MULT  = 2   # m and v vectors


# ---------------------------------------------------------------------------
# Shard reader (streaming, no full load)
# ---------------------------------------------------------------------------

def read_shard_tokens(shard_path: Path) -> list[int]:
    """Read all uint16 token IDs from a binary shard file."""
    data = shard_path.read_bytes()
    n = len(data) // 2
    return list(struct.unpack(f"<{n}H", data))


def token_window_iterator(
    shard_dir: Path,
    seq_len: int,
    batch_size: int,
    max_tokens: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Yield (x, y) pairs of shape [batch_size, seq_len] from binary shards.
    Streams sequentially; never loads more than one shard at a time.
    """
    buffer: list[int] = []
    tokens_yielded = 0
    window = seq_len + 1  # x needs seq_len tokens, y is x shifted by 1

    for shard_path in sorted(shard_dir.glob("shard_*.bin")):
        buffer.extend(read_shard_tokens(shard_path))

        while len(buffer) >= window * batch_size:
            batch_x, batch_y = [], []
            for b in range(batch_size):
                start = b * window
                chunk = buffer[start : start + window]
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])

            x = torch.tensor(batch_x, dtype=torch.long)
            y = torch.tensor(batch_y, dtype=torch.long)
            yield x, y

            # Advance buffer by one full batch-worth of windows
            buffer = buffer[window * batch_size :]
            tokens_yielded += seq_len * batch_size
            if max_tokens is not None and tokens_yielded >= max_tokens:
                return


# ---------------------------------------------------------------------------
# Parameter count utilities
# ---------------------------------------------------------------------------

def count_parameters(model: NexaTransformer) -> dict[str, int]:
    """Break down exact parameter counts by component."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Embedding (vocab_size × d_model)
    embed = sum(p.numel() for p in model.embedding.parameters())

    # All attention layers
    attn = sum(
        p.numel()
        for block in model.blocks
        for p in block.attn.parameters()
        if p.requires_grad
    )

    # All MLP layers
    mlp = sum(
        p.numel()
        for block in model.blocks
        for p in block.mlp.parameters()
        if p.requires_grad
    )

    # Norms (tiny but exact)
    norms = total - embed - attn - mlp

    return {
        "total":              total,
        "embedding":          embed,
        "attention":          attn,
        "mlp":                mlp,
        "norms_and_other":    norms,
        "non_embedding":      total - embed,
    }


def estimate_memory_mb(params: dict[str, int], batch_size: int, seq_len: int, d_model: int, n_layers: int) -> dict[str, float]:
    """Estimate peak RAM usage in MB for a single training step (float32, CPU)."""
    total_params = params["total"]

    # Weights
    model_mb = total_params * FLOAT32_BYTES / 1024**2

    # AdamW keeps m (momentum) and v (variance) per parameter
    optimizer_mb = total_params * ADAMW_STATE_MULT * FLOAT32_BYTES / 1024**2

    # Gradients (same size as weights)
    grad_mb = total_params * FLOAT32_BYTES / 1024**2

    # Activations: rough estimate = B × S × d_model × n_layers × 2 (fwd+bwd) × 4 bytes
    # Factor 2 accounts for intermediate activations saved for backward
    activation_mb = (batch_size * seq_len * d_model * n_layers * 2 * FLOAT32_BYTES) / 1024**2

    total_mb = model_mb + optimizer_mb + grad_mb + activation_mb

    return {
        "model_weights_mb":     round(model_mb, 1),
        "optimizer_states_mb":  round(optimizer_mb, 1),
        "gradients_mb":         round(grad_mb, 1),
        "activations_mb":       round(activation_mb, 1),
        "total_estimated_mb":   round(total_mb, 1),
    }


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------

def benchmark_config(
    name: str,
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    warmup_steps: int,
    benchmark_steps: int,
) -> dict:
    """
    Benchmark one model configuration.
    Returns a result dict with timing, throughput, and loss measurements.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name} | d={config.d_model} L={config.n_layers} "
          f"h={config.n_heads} ff={config.d_ff} | B={batch_size} S={seq_len}")
    print(f"{'='*60}")

    model = NexaTransformer(config)
    model.train()

    param_counts = count_parameters(model)
    memory = estimate_memory_mb(
        param_counts, batch_size, seq_len,
        config.d_model, config.n_layers
    )

    print(f"  Parameters: {param_counts['total']:,} total | "
          f"{param_counts['non_embedding']:,} non-embedding")
    print(f"  Est. memory: {memory['total_estimated_mb']:.0f} MB total")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # --- Warmup (not timed) ---
    print(f"  Warming up ({warmup_steps} steps)...", end="", flush=True)

    data_iter = token_window_iterator(
        SHARD_TRAIN_DIR, seq_len, batch_size,
        max_tokens=(warmup_steps + benchmark_steps) * seq_len * batch_size * 2
    )

    for step, (x, y) in enumerate(data_iter):
        if step >= warmup_steps:
            break
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(" done")

    # --- Timed benchmark ---
    fwd_times, bwd_times, opt_times = [], [], []
    losses = []
    tokens_processed = 0

    print(f"  Running {benchmark_steps} timed steps...", end="", flush=True)
    t_total_start = time.perf_counter()

    for step, (x, y) in enumerate(data_iter):
        if step >= benchmark_steps:
            break

        # Forward
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)

        # Backward
        loss.backward()
        t2 = time.perf_counter()
        bwd_times.append(t2 - t1)

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        t3 = time.perf_counter()
        opt_times.append(t3 - t2)

        losses.append(loss.item())
        tokens_processed += batch_size * seq_len

    t_total = time.perf_counter() - t_total_start
    print(" done")

    # --- Validation loss (first val shard, max 200 windows) ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y in token_window_iterator(SHARD_VAL_DIR, seq_len, batch_size, max_tokens=200 * seq_len * batch_size):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            val_losses.append(loss.item())
            if len(val_losses) >= 200:
                break

    avg_val_loss = sum(val_losses) / max(len(val_losses), 1)
    avg_train_loss = sum(losses) / max(len(losses), 1)

    def _avg(lst):  return round(sum(lst) / max(len(lst), 1) * 1000, 1)  # ms
    def _p95(lst):
        s = sorted(lst)
        return round(s[int(len(s) * 0.95)] * 1000, 1)

    tokens_per_sec = round(tokens_processed / t_total)

    result = {
        "name":                  name,
        "config": {
            "d_model":           config.d_model,
            "n_heads":           config.n_heads,
            "n_layers":          config.n_layers,
            "d_ff":              config.d_ff,
            "vocab_size":        config.vocab_size,
            "max_seq_len":       config.max_seq_len,
        },
        "benchmark_settings": {
            "batch_size":        batch_size,
            "seq_len":           seq_len,
            "warmup_steps":      warmup_steps,
            "benchmark_steps":   benchmark_steps,
        },
        "parameters":            param_counts,
        "memory_estimate_mb":    memory,
        "timing_ms": {
            "fwd_avg":           _avg(fwd_times),
            "fwd_p95":           _p95(fwd_times),
            "bwd_avg":           _avg(bwd_times),
            "bwd_p95":           _p95(bwd_times),
            "opt_avg":           _avg(opt_times),
            "total_step_avg":    round(t_total / max(len(fwd_times), 1) * 1000, 1),
        },
        "throughput": {
            "tokens_per_sec":    tokens_per_sec,
            "tokens_processed":  tokens_processed,
        },
        "loss": {
            "avg_train_loss":    round(avg_train_loss, 4),
            "avg_val_loss":      round(avg_val_loss, 4),
            "train_perplexity":  round(math.exp(avg_train_loss), 2),
            "val_perplexity":    round(math.exp(avg_val_loss), 2),
        },
    }

    print(f"  tokens/sec: {tokens_per_sec:,}")
    print(f"  fwd avg: {result['timing_ms']['fwd_avg']} ms | "
          f"bwd avg: {result['timing_ms']['bwd_avg']} ms")
    print(f"  val loss: {avg_val_loss:.4f} | val PPL: {result['loss']['val_perplexity']:.2f}")

    del model, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def recommend_config(results: list[dict]) -> dict:
    """
    Select the recommended config based on:
    1. Must achieve >= 100 tokens/sec (practical for iterative training)
    2. Memory must fit comfortably (< 8 GB)
    3. Among those passing the above, prefer the largest (best capacity)
    """
    MIN_TOKENS_PER_SEC = 100
    MAX_MEMORY_MB = 8 * 1024  # 8 GB

    eligible = [
        r for r in results
        if r["throughput"]["tokens_per_sec"] >= MIN_TOKENS_PER_SEC
        and r["memory_estimate_mb"]["total_estimated_mb"] <= MAX_MEMORY_MB
    ]

    if not eligible:
        # Fall back to fastest
        eligible = sorted(results, key=lambda r: r["throughput"]["tokens_per_sec"], reverse=True)

    # Prefer largest eligible model by parameter count
    chosen = max(eligible, key=lambda r: r["parameters"]["total"])

    # Estimate steps to train through corpus once
    train_tokens = 18_901_546
    tokens_per_step = (chosen["benchmark_settings"]["batch_size"] *
                       chosen["benchmark_settings"]["seq_len"])
    steps_per_epoch = train_tokens // tokens_per_step

    # Estimate total time for 10 epochs at measured throughput
    tps = chosen["throughput"]["tokens_per_sec"]
    hours_10_epochs = (train_tokens * 10) / (tps * 3600) if tps > 0 else 999

    return {
        "recommended": chosen["name"],
        "recommended_config": chosen["config"],
        "recommended_params": chosen["parameters"]["total"],
        "rationale": (
            f"Config '{chosen['name']}' ({chosen['parameters']['total']:,} params) "
            f"achieves {chosen['throughput']['tokens_per_sec']:,} tok/s with "
            f"~{chosen['memory_estimate_mb']['total_estimated_mb']:.0f} MB memory. "
            f"Steps per epoch: {steps_per_epoch:,}. "
            f"Est. time for 10 epochs: {hours_10_epochs:.1f} hours."
        ),
        "steps_per_epoch_estimate":  steps_per_epoch,
        "hours_for_10_epochs":       round(hours_10_epochs, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nexa Phase 8.7 — Configuration Benchmark")
    parser.add_argument("--warmup-steps",    type=int, default=5,  help="Warmup steps (not timed)")
    parser.add_argument("--benchmark-steps", type=int, default=20, help="Timed benchmark steps")
    args = parser.parse_args()

    if not SHARD_TRAIN_DIR.exists():
        print(f"ERROR: Training shards not found at {SHARD_TRAIN_DIR}. Run Phase 8.6 first.")
        sys.exit(1)

    print("\n=== Nexa Phase 8.7 — Pretraining Readiness Benchmark ===")
    print(f"CPU benchmark | warmup={args.warmup_steps} | timed_steps={args.benchmark_steps}\n")

    results = []
    for name, config_path, bench_settings in CANDIDATE_CONFIGS:
        cfg = ModelConfig.from_yaml(config_path)
        result = benchmark_config(
            name=name,
            config=cfg,
            batch_size=bench_settings["batch_size"],
            seq_len=bench_settings["seq_len"],
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
        )
        results.append(result)

    recommendation = recommend_config(results)

    report = {
        "phase":           "8.7",
        "hardware": {
            "cpu":         "AMD Ryzen 5 5600G",
            "cores":       "6 physical / 12 logical",
            "ram_gb":      16,
            "accelerator": "CPU only (no CUDA)",
        },
        "benchmark_results": results,
        "recommendation":    recommendation,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "pretraining_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Recommendation: {recommendation['recommended']}")
    print(f"  {recommendation['rationale']}")
    print(f"  Report written: {report_path}")
    print(f"{'='*60}")

    # Write recommended config YAML for pretraining
    chosen_name = recommendation["recommended"]
    src = {c[0]: c[1] for c in CANDIDATE_CONFIGS}[chosen_name]
    import shutil
    dest = Path("configs/nexa_v1_pretraining.yaml")
    shutil.copy2(src, dest)
    print(f"\n  Selected config copied to: {dest}")


if __name__ == "__main__":
    main()
