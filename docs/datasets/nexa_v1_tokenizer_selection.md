# Nexa V1 Tokenizer Selection

Generated: 2026-09-09T22:46:09.592406+00:00  
Phase: 8.5

> [!IMPORTANT]
> Nexa's tokenizer is trained **entirely from scratch** using Nexa's own
> BPE implementation (`nexa.tokenizer`). No pretrained tokenizer, no
> SentencePiece, no Hugging Face tokenizers, no GPT/Llama/Mistral vocabulary.

---

## Training Setup

- **Training corpus**: 2M-word sample from Phase 8 pilot corpus
- **Min-frequency filter**: 3 (pairs appearing < 3× ignored)
- **Measurement corpus**: Full pilot — 2,257 docs, 9,182,235 words
- **Special tokens**: `<pad>` (0), `<bos>` (1), `<eos>` (2), `<unk>` (3)

---

## Candidate Metrics

| Vocab Size | Merges | Tokens | Tok/Word | Chars/Tok | UNK% | Coverage% | Train(s) | Est. MB (int32) |
|---|---|---|---|---|---|---|---|---|
| **2,048 ✓** | 1,245 | 20,897,853 | 2.276 | 2.66 | 0.005 | 100.0 | 782 | 84 |
| **4,096** | 3,293 | 17,509,859 | 1.907 | 3.17 | 0.006 | 100.0 | 1494 | 70 |
| **8,192** | 7,389 | 15,131,476 | 1.648 | 3.67 | 0.007 | 100.0 | 2980 | 60 |
| **16,384** | 15,581 | 13,375,736 | 1.457 | 4.15 | 0.008 | 100.0 | 5583 | 54 |
| **32,768** | 31,965 | 12,134,112 | 1.321 | 4.58 | 0.009 | 100.0 | 10960 | 48 |

---

## Sequence-Length Distribution

| Vocab Size | p50 | p90 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| **2,048 ✓** | 4,264 | 15,274 | 20,730 | 38,676 | 122 | 1,855,589 |
| **4,096** | 3,509 | 12,463 | 17,118 | 32,588 | 102 | 1,616,493 |
| **8,192** | 3,003 | 10,654 | 14,570 | 28,107 | 81 | 1,447,994 |
| **16,384** | 2,639 | 9,308 | 12,644 | 24,789 | 72 | 1,318,245 |
| **32,768** | 2,375 | 8,343 | 11,508 | 22,247 | 59 | 1,209,919 |

---

## Embedding Parameter Cost (d_model=128)

| Vocab Size | Embed + LM-head params | % of 803K total model |
|---|---|---|
| **2,048 ✓** | 524,288 | 65% |
| **4,096** | 1,048,576 | 131% |
| **8,192** | 2,097,152 | 261% |
| **16,384** | 4,194,304 | 522% |
| **32,768** | 8,388,608 | 1045% |

---

## Selected Vocabulary Size

**Recommended: `vocab_size = 2048`**

Vocabulary size 2048 achieves the best balance of token compression, low UNK rate, word-type coverage, and low embedding parameter count for CPU-only training. At vocab_size=2048: tokens/word=2.276, chars/token=2.66, UNK rate=0.005%, coverage=100.0%, embedding params=524,288 (d_model=128). Larger vocabularies improve compression but add significant embedding parameter cost for this small model.

---

## Exact Pilot Token Count

Using the selected tokenizer (vocab_size=2048):

| Metric | Value |
|---|---|
| Total documents tokenized | 2,257 |
| Total tokens | **20,897,853** |
| Total words | 9,182,235 |
| Tokens per word | 2.2759 |
| Chars per token | 2.6591 |
| UNK token count | 1,029 |
| UNK rate | 0.0049% |
| Est. storage (uint16) | 42 MB |
| Est. storage (int32)  | 84 MB |

---

## Files

- Selected tokenizer: `data/processed/tokenizer_v1/`
- Phase 2 toy tokenizer (v0, preserved): `data/processed/tokenizer/`
- Candidate tokenizers: `data/processed/tokenizer_v1_{vocab_size}/`
- Experiment report: `data/reports/tokenizer_experiment_report.json`
- Candidates manifest: `data/manifests/tokenizer_v1_candidates.yaml`

---

## Independence Rule

This tokenizer was trained entirely from scratch on Nexa's own pilot corpus.
No pretrained model weights, no external tokenizer files, no Hugging Face
tokenizers, no SentencePiece, no GPT/Llama/Mistral vocabulary.
