"""
Nexa Phase 8.4 - Corpus Statistics
=====================================
Streaming computation of corpus statistics from JSONL files.

Computes per-source and combined statistics:
  - document count
  - character count / distribution
  - word count / distribution
  - median document length (using reservoir sampling for memory efficiency)
  - source proportions
  - document-length percentile distribution
  - repetitiveness flag (type-token ratio)

Never loads the full corpus into RAM.
Uses reservoir-based or histogram-based approaches for percentiles.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Maximum samples for reservoir-based median/percentile estimation
_RESERVOIR_SIZE = 50_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _type_token_ratio(text: str) -> float:
    """Unique words / total words (lexical diversity). Low = repetitive."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _reservoir_sample(stream: Iterator, k: int) -> list:
    """Reservoir sampling to get k items from an iterator in O(n) memory."""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from a sorted list."""
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * pct / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


# ---------------------------------------------------------------------------
# Per-document record
# ---------------------------------------------------------------------------

@dataclass
class DocRecord:
    chars: int
    words: int
    ttr:   float   # type-token ratio


# ---------------------------------------------------------------------------
# Statistics accumulator
# ---------------------------------------------------------------------------

@dataclass
class CorpusStats:
    dataset:         str   = ""
    doc_count:       int   = 0
    char_total:      int   = 0
    word_total:      int   = 0
    char_min:        int   = 0
    char_max:        int   = 0
    word_min:        int   = 0
    word_max:        int   = 0
    low_ttr_count:   int   = 0      # docs with TTR < 0.1 (potentially repetitive)
    _char_reservoir: list  = field(default_factory=list, repr=False)
    _word_reservoir: list  = field(default_factory=list, repr=False)
    _reservoir_n:    int   = 0      # total items seen (for reservoir sampling)

    # TTR threshold for flagging repetitive docs
    TTR_THRESHOLD: float = 0.10

    def update(self, doc: DocRecord) -> None:
        self.doc_count  += 1
        self.char_total += doc.chars
        self.word_total += doc.words

        if self.doc_count == 1:
            self.char_min = self.char_max = doc.chars
            self.word_min = self.word_max = doc.words
        else:
            self.char_min = min(self.char_min, doc.chars)
            self.char_max = max(self.char_max, doc.chars)
            self.word_min = min(self.word_min, doc.words)
            self.word_max = max(self.word_max, doc.words)

        if doc.ttr < self.TTR_THRESHOLD and doc.words >= 50:
            self.low_ttr_count += 1

        # Reservoir sampling for char and word counts
        self._reservoir_n += 1
        n = self._reservoir_n
        if len(self._char_reservoir) < _RESERVOIR_SIZE:
            self._char_reservoir.append(doc.chars)
            self._word_reservoir.append(doc.words)
        else:
            j = random.randint(0, n - 1)
            if j < _RESERVOIR_SIZE:
                self._char_reservoir[j] = doc.chars
                self._word_reservoir[j] = doc.words

    @property
    def avg_chars(self) -> float:
        return self.char_total / self.doc_count if self.doc_count else 0.0

    @property
    def avg_words(self) -> float:
        return self.word_total / self.doc_count if self.doc_count else 0.0

    def _sorted_chars(self) -> list[float]:
        return sorted(self._char_reservoir)

    def _sorted_words(self) -> list[float]:
        return sorted(self._word_reservoir)

    def char_percentiles(self) -> dict[str, float]:
        s = self._sorted_chars()
        return {
            "p10": round(_percentile(s, 10)),
            "p25": round(_percentile(s, 25)),
            "p50": round(_percentile(s, 50)),   # median
            "p75": round(_percentile(s, 75)),
            "p90": round(_percentile(s, 90)),
            "p99": round(_percentile(s, 99)),
        }

    def word_percentiles(self) -> dict[str, float]:
        s = self._sorted_words()
        return {
            "p10": round(_percentile(s, 10)),
            "p25": round(_percentile(s, 25)),
            "p50": round(_percentile(s, 50)),
            "p75": round(_percentile(s, 75)),
            "p90": round(_percentile(s, 90)),
            "p99": round(_percentile(s, 99)),
        }

    def to_dict(self) -> dict:
        char_pct = self.char_percentiles()
        word_pct = self.word_percentiles()
        return {
            "dataset":              self.dataset,
            "doc_count":            self.doc_count,
            "char_total":           self.char_total,
            "word_total":           self.word_total,
            "avg_chars":            round(self.avg_chars, 1),
            "avg_words":            round(self.avg_words, 1),
            "char_min":             self.char_min,
            "char_max":             self.char_max,
            "word_min":             self.word_min,
            "word_max":             self.word_max,
            "median_chars":         char_pct["p50"],
            "median_words":         word_pct["p50"],
            "char_percentiles":     char_pct,
            "word_percentiles":     word_pct,
            "low_ttr_docs":         self.low_ttr_count,
            "low_ttr_pct":          round(self.low_ttr_count / self.doc_count * 100, 2)
                                    if self.doc_count else 0.0,
        }


# ---------------------------------------------------------------------------
# Compute statistics from a JSONL file
# ---------------------------------------------------------------------------

def compute_stats(jsonl_path: Path, dataset_name: str, seed: int = 42) -> CorpusStats:
    """
    Stream through a JSONL file and compute corpus statistics.

    Uses reservoir sampling for percentile estimation — no full list in RAM.
    """
    random.seed(seed)
    stats = CorpusStats(dataset=dataset_name)

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            chars = len(text)
            words = _word_count(text)
            ttr   = _type_token_ratio(text)
            stats.update(DocRecord(chars=chars, words=words, ttr=ttr))

    return stats


def compute_combined_stats(per_source: list[CorpusStats]) -> dict:
    """Aggregate statistics across multiple sources."""
    total_docs  = sum(s.doc_count   for s in per_source)
    total_chars = sum(s.char_total  for s in per_source)
    total_words = sum(s.word_total  for s in per_source)

    source_proportions = {}
    for s in per_source:
        source_proportions[s.dataset] = {
            "doc_count":   s.doc_count,
            "doc_pct":     round(s.doc_count  / total_docs  * 100, 2) if total_docs  else 0,
            "char_total":  s.char_total,
            "char_pct":    round(s.char_total / total_chars * 100, 2) if total_chars else 0,
            "word_total":  s.word_total,
            "word_pct":    round(s.word_total / total_words * 100, 2) if total_words else 0,
        }

    return {
        "total_docs":          total_docs,
        "total_chars":         total_chars,
        "total_words":         total_words,
        "avg_chars":           round(total_chars / total_docs, 1) if total_docs else 0,
        "avg_words":           round(total_words / total_docs, 1) if total_docs else 0,
        "source_proportions":  source_proportions,
        "per_source":          [s.to_dict() for s in per_source],
    }
