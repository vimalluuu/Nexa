# Nexa V1 Corpus Analysis

Generated: 2026-09-09T16:18:56.172416+00:00  
Phase: 8.4

> [!IMPORTANT]
> **Exact Nexa BPE token counts are UNKNOWN until Nexa's own BPE tokenizer is
> trained and applied.** All token estimates below are rough approximations
> (English BPE ratio: 1.2–1.5 tokens per word) and must not be treated as
> authoritative counts.

---

## 1. Cleaned Corpus

| Metric | Wikimedia | PG-19 | Combined |
|---|---|---|---|
| Documents | 19,536 | 25 | 19,561 |
| Characters | 316,170,902 | 17,461,260 | 333,632,162 |
| Words | 49,728,938 | 3,181,869 | 52,910,807 |
| Avg chars/doc | 16,184 | 698,450 | 17,056 |
| Median chars/doc | 10,295 | 296,567 | — |

---

## 2. Deduplication

- **Input documents**: 19,561
- **Kept (deduplicated)**: 19,561
- **Duplicates removed**: 0 (0.00%)

  - wikimedia_english: 19,536 in → 19,536 kept (0 dupes, 0.00%)
  - pg19: 25 in → 25 kept (0 dupes, 0.00%)

---

## 3. Deduplicated Corpus

| Metric | Wikimedia | PG-19 | Combined |
|---|---|---|---|
| Documents | 19,536 | 25 | 19,561 |
| Characters | 316,170,902 | 17,461,260 | 333,632,162 |
| Words | 49,728,938 | 3,181,869 | 52,910,807 |
| Avg chars/doc | 16,184 | 698,450 | 17,056 |
| Median chars | 10,295 | 296,567 | — |
| Min chars | 201 | 9,902 | — |
| Max chars | 197,884 | 4,565,810 | — |
| Potentially repetitive docs (TTR < 0.10) | 0 | 4 | — |

**Source proportions (deduplicated):**

| Source | Doc % | Word % |
|---|---|---|
| wikimedia_english_dedup | 99.9% | 94.0% |
| pg19_dedup | 0.1% | 6.0% |

---

## 4. Source Balance Scenarios

| Ratio (Wiki/PG-19) | Wiki words | PG-19 words | Total words | Est. docs |
|---|---|---|---|---|
| 90/10 wiki/pg19 | 47,619,726 | 3,181,869 | 50,801,595 | 18,732 |
| 80/20 wiki/pg19 | 42,328,645 | 3,181,869 | 45,510,514 | 16,653 |
| 70/30 wiki/pg19 | 37,037,564 | 3,181,869 | 40,219,433 | 14,575 |

---

## 5. Pilot Candidate Selection

- **Strategy**: Sort by source_id ascending (deterministic); take first N Wikimedia articles until word target reached; include all PG-19 books
- **Random seed**: 42
- **Target word count**: 7,500,000 (planning figure)
- **Wikimedia selected**: 2,232 docs, 6,000,366 words
- **PG-19 selected**: 25 docs, 3,181,869 words
- **Total selected**: 2,257 docs, 9,182,235 words
- **Estimated token range**: 11,018,682–13,773,352 (English BPE 1.2–1.5×)

> [!CAUTION]
> **Exact Nexa token count = UNKNOWN**
> The token count will be determined when Nexa's own BPE tokenizer is
> trained on this corpus and applied to the selected documents.

---

## 6. Document-Length Distribution (Wikimedia, deduplicated)

| Percentile | Characters | Words |
|---|---|---|
| p10 | 1,614 | 250 |
| p25 | 3,798 | 598 |
| p50 | 10,295 | 1,622 |
| p75 | 22,902 | 3,591 |
| p90 | 39,311 | 6,177 |
| p99 | 78,258 | 12,352 |

---

## Notes

- Raw files remain **untouched** (`data/raw/`)
- Cleaned data is at `data/cleaned/`
- Deduplicated data is at `data/deduplicated/`
- Pilot index is at `data/deduplicated/pilot_candidate_index.jsonl`
- No deduplication, tokenization, or training was performed on any AI model
- No pretrained model or external AI/LLM API was used at any stage
