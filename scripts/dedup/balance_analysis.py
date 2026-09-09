"""
Nexa Phase 8.4 - Source Balance Analysis + Pilot Selection
============================================================
Analyzes how different source-mix ratios affect the corpus composition,
and defines a deterministic pilot-candidate selection strategy.

SOURCE BALANCE ANALYSIS
------------------------
For three candidate mix ratios (90/10, 80/20, 70/30 Wikimedia/PG-19),
reports the resulting document and word counts from the deduplicated corpus.

PILOT CANDIDATE SELECTION
--------------------------
Defines a REPRODUCIBLE selection of documents targeting approximately
10M eventual Nexa tokens.

IMPORTANT DISCLAIMER:
  The final Nexa token count is UNKNOWN until Nexa's own BPE tokenizer
  is trained and applied. We select in terms of WORD COUNT, not token count.

  Rough BPE token estimate for English: ~1.2–1.5 tokens per word
  (varies by vocabulary size, text type, and tokenizer configuration).
  For 10M tokens: expect to need ~6.7M–8.3M words.

  This module selects approximately 7.5M words (planning figure) split
  across sources, which is a PLANNING ESTIMATE only.

SELECTION METHOD:
  - Deterministic: sort documents by source_id (integer, ascending)
  - Take the first N documents from each source that together contribute
    the target word count
  - Reproducible given the same deduplicated JSONL and same random seed

OUTPUT:
  - data/deduplicated/pilot_candidate_index.jsonl
    (one record per selected document: {dataset, source_id, title, words})
  - Selection is NOT tokenized here — that happens in a future phase
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planning constants (NOT actual token counts)
# ---------------------------------------------------------------------------
# Target word count for pilot (approximate; real token count unknown)
PILOT_TARGET_WORDS = 7_500_000
# Minimum words per document to include in pilot
PILOT_MIN_WORDS = 50
# Seed for any random operations
PILOT_SEED = 42


# ---------------------------------------------------------------------------
# Source balance analysis
# ---------------------------------------------------------------------------

def analyze_balance(
    source_stats: dict[str, dict],
    ratios: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """
    For each proposed Wikimedia/PG-19 mix ratio, compute the resulting
    document and word counts.

    Parameters
    ----------
    source_stats : dict mapping dataset name → {doc_count, word_total}
    ratios       : list of (wiki_fraction, pg19_fraction) pairs (must sum to 1.0)

    Returns
    -------
    List of scenario dicts
    """
    if ratios is None:
        ratios = [(0.90, 0.10), (0.80, 0.20), (0.70, 0.30)]

    wiki_docs  = source_stats.get("wikimedia_english", {}).get("doc_count", 0)
    wiki_words = source_stats.get("wikimedia_english", {}).get("word_total", 0)
    pg19_docs  = source_stats.get("pg19", {}).get("doc_count", 0)
    pg19_words = source_stats.get("pg19", {}).get("word_total", 0)

    total_words = wiki_words + pg19_words
    total_docs  = wiki_docs  + pg19_docs

    scenarios = []
    for wiki_frac, pg19_frac in ratios:
        assert abs(wiki_frac + pg19_frac - 1.0) < 1e-9, "Fractions must sum to 1"

        # Target words from each source for this ratio
        target_wiki_words = total_words * wiki_frac
        target_pg19_words = total_words * pg19_frac

        # Actual available
        avail_wiki_words = min(wiki_words, int(target_wiki_words))
        avail_pg19_words = min(pg19_words, int(target_pg19_words))

        # Proportional doc estimate
        wiki_doc_fraction = avail_wiki_words / wiki_words if wiki_words else 0
        pg19_doc_fraction = avail_pg19_words / pg19_words if pg19_words else 0
        est_wiki_docs = int(wiki_docs * wiki_doc_fraction)
        est_pg19_docs = int(pg19_docs * pg19_doc_fraction)

        scenarios.append({
            "ratio":          f"{int(wiki_frac*100)}/{int(pg19_frac*100)} wiki/pg19",
            "wiki_fraction":  wiki_frac,
            "pg19_fraction":  pg19_frac,
            "wikimedia_english": {
                "target_words":    int(target_wiki_words),
                "available_words": avail_wiki_words,
                "est_docs":        est_wiki_docs,
                "word_pct":        round(avail_wiki_words / total_words * 100, 2)
                                   if total_words else 0,
            },
            "pg19": {
                "target_words":    int(target_pg19_words),
                "available_words": avail_pg19_words,
                "est_docs":        est_pg19_docs,
                "word_pct":        round(avail_pg19_words / total_words * 100, 2)
                                   if total_words else 0,
            },
            "combined": {
                "total_words":     avail_wiki_words + avail_pg19_words,
                "total_est_docs":  est_wiki_docs + est_pg19_docs,
            },
        })

    return scenarios


# ---------------------------------------------------------------------------
# Pilot candidate selection
# ---------------------------------------------------------------------------

@dataclass
class PilotSelection:
    """Records for the pilot candidate selection."""
    seed:               int
    target_words:       int
    wiki_target_words:  int
    pg19_target_words:  int
    wiki_selected_docs: int = 0
    wiki_actual_words:  int = 0
    pg19_selected_docs: int = 0
    pg19_actual_words:  int = 0
    total_selected_docs: int = 0
    total_actual_words:  int = 0
    selection_records:  list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "seed":                self.seed,
            "disclaimer":          (
                "Word-count-based selection only. Exact Nexa BPE token count is "
                "UNKNOWN until Nexa's own tokenizer is trained and applied. "
                "Estimated BPE token count: approx. 1.2-1.5 * word count."
            ),
            "target_words":        self.target_words,
            "estimated_tokens_low":  int(self.total_actual_words * 1.2),
            "estimated_tokens_high": int(self.total_actual_words * 1.5),
            "exact_nexa_token_count": "UNKNOWN — pending Nexa BPE training",
            "wiki_target_words":   self.wiki_target_words,
            "pg19_target_words":   self.pg19_target_words,
            "wiki_selected_docs":  self.wiki_selected_docs,
            "wiki_actual_words":   self.wiki_actual_words,
            "pg19_selected_docs":  self.pg19_selected_docs,
            "pg19_actual_words":   self.pg19_actual_words,
            "total_selected_docs": self.total_selected_docs,
            "total_actual_words":  self.total_actual_words,
        }


def select_pilot_candidates(
    wiki_jsonl: Path,
    pg19_jsonl: Path,
    output_index: Path,
    target_words: int = PILOT_TARGET_WORDS,
    wiki_fraction: float = 0.80,
    pg19_fraction: float = 0.20,
    seed: int = PILOT_SEED,
    min_words: int = PILOT_MIN_WORDS,
) -> PilotSelection:
    """
    Select a reproducible pilot candidate set from the deduplicated corpus.

    Strategy:
      1. Sort documents by source_id (integer ascending) for determinism
      2. Take documents from Wikimedia until wiki_target_words is reached
      3. Take all PG-19 documents (25 books, always included in full)
      4. Write a lightweight index JSONL (no text) for later retrieval

    Parameters
    ----------
    wiki_jsonl      : deduplicated Wikimedia JSONL
    pg19_jsonl      : deduplicated PG-19 JSONL
    output_index    : where to write the selection index JSONL
    target_words    : total target word count (planning figure)
    wiki_fraction   : fraction of target_words for Wikimedia
    pg19_fraction   : fraction of target_words for PG-19
    seed            : random seed (used for documentation; selection is sort-based)
    min_words       : minimum words per document to include

    Returns
    -------
    PilotSelection with counts and word totals
    """
    random.seed(seed)

    wiki_target = int(target_words * wiki_fraction)
    pg19_target = int(target_words * pg19_fraction)

    sel = PilotSelection(
        seed=seed,
        target_words=target_words,
        wiki_target_words=wiki_target,
        pg19_target_words=pg19_target,
    )

    output_index.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # --- Wikimedia: sort by source_id, take until word target met ---
    log.info("Selecting Wikimedia pilot candidates (target: %d words)...", wiki_target)
    wiki_docs = []
    if wiki_jsonl.exists():
        with open(wiki_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                w = len(rec.get("text", "").split())
                if w >= min_words:
                    try:
                        sort_key = int(rec.get("source_id", 0))
                    except (ValueError, TypeError):
                        sort_key = 0
                    wiki_docs.append((sort_key, w, rec))

    wiki_docs.sort(key=lambda x: x[0])   # deterministic: ascending source_id

    wiki_words_acc = 0
    for _, w, rec in wiki_docs:
        if wiki_words_acc >= wiki_target:
            break
        entry = {
            "dataset":   rec.get("dataset", "wikimedia_english"),
            "source_id": rec.get("source_id", ""),
            "title":     rec.get("title", ""),
            "source_url": rec.get("source_url", ""),
            "snapshot":  rec.get("snapshot", ""),
            "words":     w,
        }
        records.append(entry)
        wiki_words_acc += w
        sel.wiki_selected_docs += 1

    sel.wiki_actual_words = wiki_words_acc
    log.info("Wiki pilot: %d docs, %d words", sel.wiki_selected_docs, wiki_words_acc)

    # --- PG-19: always include all (small corpus, ~3.2M words) ---
    log.info("Selecting PG-19 pilot candidates...")
    pg19_words_acc = 0
    if pg19_jsonl.exists():
        with open(pg19_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                w = len(rec.get("text", "").split())
                if w >= min_words:
                    entry = {
                        "dataset":          rec.get("dataset", "pg19"),
                        "source_id":        rec.get("source_id", ""),
                        "title":            rec.get("title", ""),
                        "publication_year": rec.get("publication_year", ""),
                        "gutenberg_url":    rec.get("gutenberg_url", ""),
                        "source_url":       rec.get("source_url", ""),
                        "words":            w,
                    }
                    records.append(entry)
                    pg19_words_acc += w
                    sel.pg19_selected_docs += 1

    sel.pg19_actual_words = pg19_words_acc
    log.info("PG-19 pilot: %d docs, %d words", sel.pg19_selected_docs, pg19_words_acc)

    sel.total_selected_docs = sel.wiki_selected_docs + sel.pg19_selected_docs
    sel.total_actual_words  = sel.wiki_actual_words  + sel.pg19_actual_words
    sel.selection_records   = records

    # Write lightweight index (no text field)
    with open(output_index, "w", encoding="utf-8") as f:
        for entry in records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info("Pilot index written: %d records, %d total words -> %s",
             len(records), sel.total_actual_words, output_index)
    return sel
