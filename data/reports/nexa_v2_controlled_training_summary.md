# Nexa Phase 14 — V2 Controlled Training Summary

## Status
**Overall: ✅ ALL VERIFICATIONS PASSED**

## Configuration (AUTHORITATIVE)
| Parameter | Value |
|---|---|
| Git Commit | `5a9561c29fde0c716f53d69e030955ef250ae644` |
| Seed | 42 |
| vocab_size | 8192 |
| d_model | 512 |
| n_layers | 6 |
| n_heads | 8 |
| d_ff | 2048 |
| context_length | 1024 |
| batch_size | 4 |
| grad_accum_steps | 8 |
| effective_batch_size | 32 |
| tokens_per_step | 32,768 |
| total_steps | 100 |
| total_tokens | 3,276,800 |
| learning_rate | 0.0006 |
| scheduler | cosine warmup |
| warmup_steps | 10 |

## Parameter Count
| Metric | Value |
|---|---|
| Total | 29,366,784 |
| Embedding | 4,194,304 (14.28%) |
| Non-Embedding | 25,172,480 |

## Training Metrics (MEASURED)
| Metric | Value |
|---|---|
| Initial train loss | 9.067006 |
| Final train loss | 5.753936 |
| Final val loss | 4.550239 |
| Final val PPL | 94.655 |
| Val batches | 20 |
| Val tokens | 81,920 |
| E2E throughput | 500.9 tok/s |
| Wall clock | 6541.53 s |

## Pipeline Verifications
| Check | Result |
|---|---|
| Shard token ID range [0,8192) | ✅ |
| V2 tokenizer vocab_size=8192 | ✅ |
| Model params = 29,366,784 | ✅ |
| Checkpoint save | ✅ |
| Checkpoint reload | ✅ |
| Resume training | ✅ |
| Inference (greedy + seeded) | ✅ |
| V1 immutability | ✅ |

## Updated Throughput (MEASURED — supersedes Phase 13 estimate)
Phase 13 planning estimate: ~470 tok/s (benchmark, BS=4, ctx=1024)
Phase 14 measured E2E: **500.9 tok/s** (including data loading and checkpointing)

## Training Time Projections (ESTIMATED using Phase 14 measured throughput)

- 10M tokens → ~5.5 hours (ESTIMATED)
- 25M tokens → ~13.9 hours (ESTIMATED)
- 50M tokens → ~27.7 hours (ESTIMATED)
- 100M tokens → ~55.5 hours (ESTIMATED)
- 283M tokens → ~156.9 hours (ESTIMATED)

## Limitations
- 100-step controlled run is too short to draw meaningful conclusions about model quality.
- val_loss at this scale reflects random initialization, not meaningful learned representations.
- Seeded generation reproducibility is verified but generated text has no linguistic quality.
- Measured throughput may differ slightly during full training due to OS scheduling noise.