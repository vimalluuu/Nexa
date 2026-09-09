"""
Nexa Phase 8.4 - Exact Document Deduplication
================================================
Deterministic, AI-free exact deduplication using SHA-256 content hashes.

Strategy:
  - Hash the normalized text field of each document (not raw wikitext)
  - A document is a duplicate if its text hash was seen in ANY earlier source
  - Processing order determines which copy is kept: first occurrence wins
  - Default processing order: wikimedia_english → pg19
  - Fully streaming: never loads the full corpus into memory at once

Cross-source deduplication:
  - A single global hash set is maintained across all sources
  - If a PG-19 book has text that also appeared in Wikimedia, it's a duplicate
  - In practice this is very unlikely but handled correctly

Provenance:
  - Every retained document preserves all original fields unchanged
  - Duplicate records are counted and logged by source and document ID

Output format:
  Same JSONL structure as cleaned data; no fields added or removed.
  (Provenance fields were already embedded by the extraction pipeline.)

Raw and cleaned data are NEVER modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

# Text is hashed after stripping leading/trailing whitespace for robustness
_HASH_ENCODING = "utf-8"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """
    Compute a deterministic SHA-256 hash of the document text.

    The text is NFC-normalised and stripped before hashing, so that two
    documents differing only in surrounding whitespace are considered equal.
    This is consistent with the normalization already applied in Phase 8.3.
    """
    import unicodedata
    normalized = unicodedata.normalize("NFC", text.strip())
    return hashlib.sha256(normalized.encode(_HASH_ENCODING)).hexdigest()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class DeduplicationStats:
    """Deduplication statistics for a single source."""
    dataset:          str = ""
    input_docs:       int = 0
    kept_docs:        int = 0
    duplicate_docs:   int = 0
    duplicate_sources: list[dict] = field(default_factory=list)

    @property
    def duplicate_pct(self) -> float:
        return (self.duplicate_docs / self.input_docs * 100.0
                if self.input_docs > 0 else 0.0)

    def to_dict(self) -> dict:
        return {
            "dataset":          self.dataset,
            "input_docs":       self.input_docs,
            "kept_docs":        self.kept_docs,
            "duplicate_docs":   self.duplicate_docs,
            "duplicate_pct":    round(self.duplicate_pct, 4),
            "duplicate_sources": self.duplicate_sources,
        }


# ---------------------------------------------------------------------------
# Streaming deduplication
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream JSON objects from a JSONL file one line at a time."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dedup_source(
    input_path: Path,
    output_path: Path,
    seen_hashes: set[str],
    dataset_name: str,
    max_dup_log: int = 100,
) -> DeduplicationStats:
    """
    Deduplicate one source JSONL file against a global seen_hashes set.

    Streams through input_path line by line.
    Writes non-duplicate documents to output_path.
    Updates seen_hashes in-place (so cross-source dedup works when called
    sequentially for multiple sources sharing the same set).

    Parameters
    ----------
    input_path   : cleaned JSONL to read from
    output_path  : deduplicated JSONL to write to
    seen_hashes  : shared set of hashes already encountered (modified in-place)
    dataset_name : label for statistics
    max_dup_log  : maximum number of duplicate source records to store in stats

    Returns
    -------
    DeduplicationStats for this source
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = DeduplicationStats(dataset=dataset_name)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in iter_jsonl(input_path):
            stats.input_docs += 1
            text = record.get("text", "")
            h = content_hash(text)

            if h in seen_hashes:
                stats.duplicate_docs += 1
                if len(stats.duplicate_sources) < max_dup_log:
                    stats.duplicate_sources.append({
                        "source_id": record.get("source_id", ""),
                        "title":     record.get("title", ""),
                        "dataset":   record.get("dataset", ""),
                    })
                log.debug("Duplicate: %s / %s",
                          record.get("dataset"), record.get("source_id"))
            else:
                seen_hashes.add(h)
                stats.kept_docs += 1
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if stats.input_docs % 5000 == 0:
                log.info("  [%s] Processed %d | kept %d | dupes %d",
                         dataset_name, stats.input_docs,
                         stats.kept_docs, stats.duplicate_docs)

    log.info("Dedup complete [%s]: %d in -> %d kept (%d dupes, %.2f%%)",
             dataset_name, stats.input_docs, stats.kept_docs,
             stats.duplicate_docs, stats.duplicate_pct)
    return stats


def run_deduplication(
    sources: list[tuple[Path, Path, str]],
) -> tuple[list[DeduplicationStats], dict]:
    """
    Run deduplication over an ordered list of sources using one shared hash set.

    Parameters
    ----------
    sources : list of (input_path, output_path, dataset_name) tuples.
              Order determines which occurrence is kept (first wins).

    Returns
    -------
    (per_source_stats, combined_stats_dict)
    """
    seen_hashes: set[str] = set()
    per_source: list[DeduplicationStats] = []

    for input_path, output_path, name in sources:
        if not input_path.exists():
            log.warning("Input not found: %s — skipping", input_path)
            continue
        stats = dedup_source(input_path, output_path, seen_hashes, name)
        per_source.append(stats)

    total_in   = sum(s.input_docs     for s in per_source)
    total_kept = sum(s.kept_docs      for s in per_source)
    total_dupe = sum(s.duplicate_docs for s in per_source)

    combined = {
        "total_input_docs":     total_in,
        "total_kept_docs":      total_kept,
        "total_duplicate_docs": total_dupe,
        "total_duplicate_pct":  round(total_dupe / total_in * 100.0, 4)
                                if total_in > 0 else 0.0,
        "unique_hashes":        len(seen_hashes),
        "per_source":           [s.to_dict() for s in per_source],
    }
    return per_source, combined
