# Nexa Phase 10 — V1 Improvement Analysis and Scaling Plan

**Generated:** 2026-09-11T04:22:17.980107+00:00
**Git Commit:** `c63ebb15948e159711f1ea563ec30dd72f4c3ed3`

## Executive Summary
The Phase 10 analysis deconstructs the frozen Phase 9 evaluation baseline to identify exact bottlenecks in the Nexa V1 architecture. The analysis separates hard observations from inferences and theoretical hypotheses to dictate a concrete, CPU-safe roadmap for Nexa V2.

## Frozen V1 Baseline
- **Checkpoint:** `checkpoints/nexa_v1_pilot/step_0009229.pt`
- **Validation Loss:** 3.3635
- **Status:** FROZEN

## Dataset Analysis
**OBSERVED**
- The current corpus is ~18.9M training tokens.
- The pilot training pipeline successfully tokenizes and deduplicates the corpus.
**INFERRED**
- The dataset size is the primary limiter of factual generalization.
**HYPOTHESIZED**
- Chinchilla scaling laws suggest ~136M tokens as a theoretical compute-optimal target for a 6.8M parameter model, meaning we could safely scale the dataset by 5-10x.
- Scaling actual training data needs will depend heavily on data quality, filtering, and model capacity, not just raw token volume.

## Tokenizer Analysis
**OBSERVED**
- The tokenizer (nexa/tokenizer/tokenizer.py) implements Character-level BPE, not true byte-level BPE.
- End-of-word is marked with `</w>`, which is replaced by a space during decoding.
- Unseen characters are explicitly mapped to `<unk>`.
- Tokenization efficiency is ~2.27 tokens per word on the Wikimedia dataset.
- The vocabulary size is 2048.
**INFERRED**
- The strict character-level constraint (without byte-fallback) means any Unicode character not seen during BPE training immediately becomes `<unk>`.
- The small vocabulary size forces frequent word fragmentation, leading to higher token-per-word counts.
**HYPOTHESIZED**
- Malformed text outputs like 'expeese' and 'Autoese' may stem from the model struggling to correctly predict long sequences of short token fragments.
- Using a larger vocab size (4096, 8192, or 16384) or migrating to true Byte-Level BPE (which prevents <unk> tokens entirely) would likely improve semantic boundaries and generation coherence.

## Model Capacity Analysis
| Model | Params | CPU Feasibility | Est. Memory (Weights) | Relative Cost |
|-------|--------|-----------------|-----------------------|---------------|
| Current V1 | 6.8M | High (~1,145 tok/s) | ~27 MB | 1x |
| Moderate   | ~15M | Moderate (~400-600 tok/s) | ~60 MB | ~2.2x |
| Larger     | ~30M | Low (~150-250 tok/s) | ~120 MB | ~4.5x |

**OBSERVED**
- The 6.8M model executes comfortably on a Ryzen 5 5600G CPU at ~1,145 tok/s.
- Memory usage is negligible (<100 MB total).
**INFERRED**
- Scaling to ~15M parameters is well within the 16GB RAM budget but will halve throughput.
- Scaling to ~30M parameters is memory-safe but will make single-epoch training on >100M tokens extremely slow on CPU.
**HYPOTHESIZED**
- A ~15M parameter model hits the optimal balance of representation capacity and CPU-bound turnaround times.

## Context Analysis
**OBSERVED**
- Current context length is 512 tokens.
**INFERRED**
- Attention memory and compute cost scales quadratically. Moving to 1024 would 4x the attention compute bottleneck per forward pass.
- The current dataset consists of encyclopedic extracts where short-range factual recall is common.
**HYPOTHESIZED**
- Given the small parameter count, the model likely lacks the capacity to effectively attend to 2048 tokens of context anyway.
- Context length 512 or 1024 is sufficient; increasing to 2048 does not yield enough benefit for the massive CPU cost.

## Training Regime Analysis
**OBSERVED**
- Training ran for exactly 1 pass (1 epoch) over 18.9M tokens.
- Final train loss was 2.4716.
- Final val loss was 3.3635.
**INFERRED**
- The model is not suffering from optimization limitation (loss decreased smoothly).
- The model is not catastrophically overfitting, as validation loss did not aggressively diverge, though a ~0.89 gap exists between train and val.
- The model is data-limited.
**HYPOTHESIZED**
- Multiple epochs or a larger dataset will continue to yield loss improvements.
- The model is currently undertrained.

## V2 Recommendation
The following is a concrete V2 proposal based on the bottleneck analysis.

- **Dataset Target**: ~100M - 150M tokens `[RECOMMENDED]`
- **Tokenizer Design**: True Byte-Level BPE (No <unk>) `[RECOMMENDED]`
- **Vocab Size**: 8192 `[RECOMMENDED]`
- **Vocab Size Alternative**: 4096 `[ALTERNATIVE]`
- **D Model**: 384 `[RECOMMENDED]`
- **Layers**: 8 `[RECOMMENDED]`
- **Heads**: 12 `[RECOMMENDED]`
- **D Ff**: 1536 `[RECOMMENDED]`
- **Larger Model Architecture**: d_model=512, layers=10 `[DEFER]`
- **Context Length**: 512 `[RECOMMENDED]`
- **Context Length Alternative**: 1024 `[ALTERNATIVE]`
- **Batch Size**: 8 `[RECOMMENDED]`
- **Learning Rate**: 3e-4 `[RECOMMENDED]`
- **Warmup Steps**: 1000 `[RECOMMENDED]`
- **Training Token Target**: ~150M `[RECOMMENDED]`
- **Checkpoint Interval**: 5000 `[RECOMMENDED]`
- **Validation Interval**: 500 `[RECOMMENDED]`

## Experiment Priority Matrix
Ranked strictly by Information Value per Unit of Compute:

### 1. Tokenizer Rewrite (Byte-Level BPE)
- **Compute Cost:** Very Low (No training)
- **Value:** Critical. Determines vocabulary and tokenization efficiency before any data expansion.

### 2. Dataset Expansion & Quality Filter
- **Compute Cost:** Low (Data processing only)
- **Value:** High. Prepares the pipeline for theoretical scaling-law limits.

### 3. Mini Architecture Test (Same Model vs Moderate Model on 10M tokens)
- **Compute Cost:** Moderate (A few hours)
- **Value:** High. Quantifies exact CPU throughput drop and loss trajectory curve before committing to full 150M run.

### 4. Full Nexa V2 Training Run
- **Compute Cost:** Very High (Days)
- **Value:** Final Validation.
