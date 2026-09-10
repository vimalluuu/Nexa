# Nexa V1 Pretraining Configuration — Phase 8.7

Generated: 2026-09-10  
Phase: 8.7  
Status: **Readiness complete. Configuration selected.**

---

## Hardware

| Property | Value |
|---|---|
| CPU | AMD Ryzen 5 5600G |
| Physical cores | 6 (12 logical) |
| RAM | 16 GB |
| Accelerator | CPU only (no CUDA) |
| Training framework | PyTorch (CPU tensors) |

---

## Benchmark Results

Three candidate configurations were benchmarked against the real Phase 8.6
binary shards (`data/processed/train/`, `data/processed/validation/`).

Each config ran **5 warmup steps** (excluded from timing) followed by
**20 timed steps** using real tokenized data.

| Config | Params | d_model | n_layers | d_ff | Batch | Seq | tok/s | Fwd ms | Bwd ms | Val Loss | Val PPL | Est. RAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **small**  | 1,311,872  | 128 | 4 | 512  | 4 | 256 | 3,801 | 116.9 | 57.9  | 7.0439 | 1145.9 | ~24 MB  |
| **medium** | 6,819,072  | 256 | 6 | 1024 | 4 | 512 | 1,228 | 1050.6 | 510.8 | 6.7024 | 814.4  | ~128 MB |
| **large**  | 26,221,056 | 512 | 6 | 2048 | 2 | 512 | 696   | 737.2  | 569.3 | 6.6039 | 738.0  | ~424 MB |

All three configurations pass the feasibility threshold (>100 tok/s, <8 GB RAM).

---

## Recommendation: **medium**

> **Recommended first pretraining config: `configs/nexa_v1_pretraining.yaml`**
> (copied from `configs/nexa_v1_medium.yaml`)

### Why not `large`?

The automatic recommender selected `large` (26M params) because it met all
mechanical thresholds. However, for this pilot pretraining experiment the
dataset size determines the correct model size.

**Chinchilla scaling analysis** (approximate):

| Config | Params | Tokens needed (optimal ~20×) | Available (pilot) | Ratio (actual) |
|---|---|---|---|---|
| small  | 1.3M  | ~26M   | 20.9M | 16.1× |
| **medium** | **6.8M** | **~136M** | **20.9M** | **3.1×** |
| large  | 26.2M | ~524M  | 20.9M | 0.8× |

The **large** model would be severely undertrained on a 20.9M-token pilot
corpus (less than 1 token per parameter). This does not mean large cannot
be run at all — it can — but it will show poor loss curves and be a poor use
of the 75+ hours it requires for 10 epochs.

The **medium** model at 6.8M params achieves:
- 3.1× token-to-parameter ratio — still below Chinchilla optimal but far more sensible for a pilot
- **1,228 tok/s throughput** — about 4.3 hours per epoch on 18.9M train tokens
- **128 MB estimated peak RAM** — minimal system pressure on 16 GB machine
- Better val loss than small (6.70 vs 7.04) confirming greater model capacity is needed
- **~43 hours for 10 epochs** — a long but feasible pilot experiment

### Why not `small`?

Small at 1.3M params is very fast (3,801 tok/s) but achieves higher loss than
medium (7.04 vs 6.70), even on freshly initialized random weights. For a real
pretraining experiment we want enough capacity to observe meaningful loss
improvements over time.

---

## Selected Configuration

```yaml
model:
  name: "nexa-v1-medium"
  vocab_size: 2048
  max_seq_len: 512
  d_model: 256
  n_heads: 8
  n_layers: 6
  d_ff: 1024
  dropout: 0.1
  norm_type: "rmsnorm"
  pos_encoding: "rotary"
  tie_embeddings: true
```

| Metric | Value |
|---|---|
| Total parameters | 6,819,072 |
| Non-embedding parameters | 6,294,784 |
| Embedding parameters | 524,288 |
| Throughput | 1,228 tok/s |
| Estimated peak RAM | ~128 MB |
| Estimated time / epoch | ~4.3 hours |
| Estimated time for 10 epochs | ~43 hours |

---

## Training Estimates

| Metric | Value |
|---|---|
| Train tokens | 18,901,546 |
| Val tokens | 2,000,821 |
| Steps per epoch (B=4, S=512) | 9,228 |
| Train shards | 2 × ~19 MB |
| Val shards | 1 × ~3.8 MB |
| Vocab size | 2048 |
| BOS/EOS | included in token count |

---

## Future Upgrade Path

When the corpus grows:

| Corpus size | Recommended config |
|---|---|
| ~26M tokens | small (1.3M params) |
| ~136M tokens | medium (6.8M params) — current |
| ~524M tokens | large (26.2M params) |
| 1B+ tokens | Nexa V2 architecture (future) |

---

## Files

| File | Purpose |
|---|---|
| `configs/nexa_v1_small.yaml`         | Candidate A — 1.3M params |
| `configs/nexa_v1_medium.yaml`        | Candidate B — 6.8M params |
| `configs/nexa_v1_large.yaml`         | Candidate C — 26.2M params |
| `configs/nexa_v1_pretraining.yaml`   | **Selected config** (copy of medium) |
| `scripts/training/benchmark_configs.py` | Benchmark runner |
| `tests/training/test_benchmark_configs.py` | 31 tests |
| `data/reports/pretraining_benchmark_report.json` | Full JSON results |
