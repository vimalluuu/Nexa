"""
Nexa Phase 8.3 - Document Filtering
=====================================
Configurable quality filtering with full statistics tracking.

Filters are applied after cleaning and normalization.
Every removal reason is counted and reported.
Thresholds are configurable via config.py or direct arguments.

No silent discards — every rejected document is counted and categorized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class FilterStats:
    """Per-dataset filtering statistics."""
    dataset:           str = ""
    total_input:       int = 0
    kept:              int = 0
    removed_empty:     int = 0
    removed_too_short: int = 0
    removed_too_long:  int = 0
    removed_binary:    int = 0
    removed_other:     int = 0
    total_chars:       int = 0
    total_words:       int = 0
    min_chars:         int = -1
    max_chars:         int = 0
    extraction_failures: int = 0

    def record_kept(self, text: str) -> None:
        chars = len(text)
        words = len(text.split())
        self.kept += 1
        self.total_chars += chars
        self.total_words += words
        if self.min_chars < 0 or chars < self.min_chars:
            self.min_chars = chars
        if chars > self.max_chars:
            self.max_chars = chars

    @property
    def removed_total(self) -> int:
        return (self.removed_empty + self.removed_too_short +
                self.removed_too_long + self.removed_binary + self.removed_other)

    @property
    def avg_chars(self) -> float:
        return self.total_chars / self.kept if self.kept > 0 else 0.0

    @property
    def avg_words(self) -> float:
        return self.total_words / self.kept if self.kept > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset":             self.dataset,
            "total_input":         self.total_input,
            "kept":                self.kept,
            "removed_total":       self.removed_total,
            "removed_empty":       self.removed_empty,
            "removed_too_short":   self.removed_too_short,
            "removed_too_long":    self.removed_too_long,
            "removed_binary":      self.removed_binary,
            "removed_other":       self.removed_other,
            "extraction_failures": self.extraction_failures,
            "total_chars":         self.total_chars,
            "total_words":         self.total_words,
            "avg_chars":           round(self.avg_chars, 1),
            "avg_words":           round(self.avg_words, 1),
            "min_chars":           max(self.min_chars, 0),
            "max_chars":           self.max_chars,
        }


def _is_binary_or_garbled(text: str) -> bool:
    """
    Heuristic: text is binary/garbled if it has a high ratio of
    non-printable or replacement characters.
    """
    if not text:
        return False
    non_print = sum(1 for c in text if ord(c) > 127 and not c.isprintable())
    replacement = text.count("\ufffd")
    ratio = (non_print + replacement) / len(text)
    return ratio > 0.15


def filter_document(
    text: str,
    stats: FilterStats,
    min_chars: int = 200,
    min_words: int = 30,
    max_chars: int = 10_000_000,
) -> str | None:
    """
    Apply quality filters to a cleaned, normalized document.

    Returns the text if it passes all filters, None if rejected.
    Updates stats in-place for every decision.

    Parameters
    ----------
    text : str
        Cleaned, normalized text.
    stats : FilterStats
        Statistics object to update.
    min_chars : int
        Minimum character count to keep.
    min_words : int
        Minimum word count to keep.
    max_chars : int
        Maximum character count to keep.
    """
    stats.total_input += 1

    stripped = text.strip()

    if not stripped:
        stats.removed_empty += 1
        return None

    char_count = len(stripped)
    word_count = len(stripped.split())

    if char_count < min_chars or word_count < min_words:
        stats.removed_too_short += 1
        return None

    if char_count > max_chars:
        stats.removed_too_long += 1
        return None

    if _is_binary_or_garbled(stripped):
        stats.removed_binary += 1
        return None

    stats.record_kept(stripped)
    return stripped
