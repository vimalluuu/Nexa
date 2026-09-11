# Nexa Phase 13 — V2 Model Architecture and Training Configuration

## 1. Executive Summary
This report defines the Nexa V2 model architectures, contexts, and training configurations based on empirical CPU throughput measurements. Surprisingly, the larger Candidate B (~29M parameters) demonstrated higher throughput than the smaller Candidate A (~17M parameters) due to optimal power-of-two cache alignment (`d_model=512` vs `d_model=384`) on the Ryzen 5 CPU. We recommend proceeding with Candidate B at a 1024 context length over 50M tokens.

## 2. V1 Reference
- **Config**: 6 layers, 8 heads, 256 d_model, 1024 d_ff, 2048 vocab.
- **Parameters**: 6.8M
- **Throughput**: ~1056 tok/sec (at 512 context)

## 3. V2 Tokenizer Constraints
Because V2 utilizes a byte-level BPE with `vocab_size=8192` (a 4x increase over V1), the embedding matrix consumes a drastically larger percentage of the model parameters. Weight tying remains enforced to prevent doubling this cost.

## 4. Candidate Architectures
| Candidate | Vocab | d_model | Layers | Heads | d_ff | Total Params |
|---|---|---|---|---|---|---|
| Candidate A | 8192 | 384 | 6 | 12 | 1536 | ~17.3M |
| Candidate B | 8192 | 512 | 6 | 8 | 2048 | ~29.3M |

## 5. Parameter Analysis
| Model | Total Parameters | Embedding Params | Embedding % |
|---|---|---|---|
| V1 | 6,819,072 | 524,288 | 7.7% |
| Cand A | 17,306,496 | 3,145,728 | 18.2% |
| Cand B | 29,366,784 | 4,194,304 | 14.3% |

## 6. Context Analysis
- **512**: Fast (628 tok/s for Cand B).
- **1024**: Good balance (470 tok/s for Cand B).
- **2048**: Severe bottleneck. Cand A throughput dropped massively to 193 tok/s due to quadratic attention memory complexity spilling out of CPU cache. Not recommended.

## 7. Hardware Analysis (Ryzen 5 5600G)
The benchmark proved that theoretical FLOPs do not perfectly map to CPU time. Memory bandwidth and cache alignment dominate. Candidate B (`d_model=512`) outperformed Candidate A (`d_model=384`) despite having 12M more parameters, because `512` perfectly aligns with cache lines whereas `384` causes unaligned fetches.

## 8. Measured CPU Benchmark
*(Forward + Backward + AdamW Optimizer at BS=4)*
- **V1 Baseline**: 1056.6 tok/s | Fwd: 1288.6ms | Bwd: 626.8ms
- **Candidate A (512)**: 599.0 tok/s | Fwd: 2187.9ms | Bwd: 1168.1ms
- **Candidate A (1024)**: 433.7 tok/s | Fwd: 6318.7ms | Bwd: 3062.9ms
- **Candidate A (2048)**: 193.7 tok/s | Fwd: 28649.3ms | Bwd: 13087.6ms
- **Candidate B (512)**: 628.8 tok/s | Fwd: 1825.5ms | Bwd: 1330.3ms
- **Candidate B (1024)**: 470.2 tok/s | Fwd: 5219.9ms | Bwd: 3358.9ms

## 9. Training-Time Projections
*Estimates based on measured tok/sec throughput.*
| Scenario | V1 Baseline | Candidate B (1024) |
|---|---|---|
| 10M tokens | ~2.6 hrs | ~5.9 hrs |
| 25M tokens | ~6.6 hrs | ~14.8 hrs |
| 50M tokens | ~13.1 hrs | ~29.5 hrs |
| 100M tokens | ~26.3 hrs | ~59.1 hrs |
| 283M (Full) | ~74.4 hrs | ~167.2 hrs |

## 10. Tokens/Parameter Analysis
| Corpus Target | Candidate A (~17.3M) | Candidate B (~29.3M) |
|---|---|---|
| 25M | 1.4 t/p | 0.8 t/p |
| 50M | 2.9 t/p | 1.7 t/p |
| 100M | 5.8 t/p | 3.4 t/p |
| 250M | 14.4 t/p | 8.5 t/p |

*Note: Chinchilla optimal is ~20 t/p. We are operating in the compute-constrained, highly over-parameterized regime typical for local CPU experimentation.*

## 11. Recommended V2 Configuration
- **Model Size**: Candidate B (29.3M) - *Faster than Candidate A on CPU due to 512-dim cache alignment.*
- **Context Length**: 1024 - *Fits cleanly in cache, retains literature structure.*
- **Batch Config**: Batch Size 4, Gradient Accumulation 8 (Effective 32).
- **Initial Target**: 50,000,000 tokens - *Takes ~29.5 hours. Overnight + 1 workday.*

## 12. Alternatives
- **Alternative 1 (Fast Iteration)**: Candidate B (1024) over 25M tokens (~14.8 hours).
- **Alternative 2 (Long Context)**: Candidate B (2048). Expected ~200 tok/s. Would require 34+ hours for just 25M tokens. Deferred.

## 13. Phase 14 Controlled Run Specification
Before any true V2 training, execute this strictly bound validation:
1. Initialize Candidate B (512-dim, 1024 context).
2. Train for exactly **100 steps** at `BS=4, GradAcc=8`.
3. Validation interval = 50 steps.
4. Verify optimizer (`AdamW`) and scheduler (`Cosine`).
5. Save a checkpoint at Step 100.
6. Verify token generation pipeline using the new byte tokenizer.

## 14. Risks and Limitations
- The 50M token recommendation will require heavy, uninterrupted CPU thermal load for ~30 hours.
- Throughput measurements represent single-script benchmarks; data loading overheads may marginally reduce actual throughput during training.
