"""
Nexa Phase 8.5 - Corpus Text Sampler
======================================
Streams text from the pilot candidate corpus for tokenizer training.

Pilot index: data/deduplicated/pilot_candidate_index.jsonl
  - Contains {dataset, source_id, words, ...} — no text field
  - Actual text must be read from the deduplicated JSONL files

Approach:
  - Build a source_id lookup from the pilot index
  - Stream the deduplicated JSONL, yielding text only for pilot documents
  - Support word-budget-limited sampling for training efficiency

Memory safety: never loads the full corpus into RAM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

DEDUP_WIKI = Path("data/deduplicated/wikimedia_english/docs.jsonl")
DEDUP_PG19 = Path("data/deduplicated/pg19/docs.jsonl")
PILOT_INDEX = Path("data/deduplicated/pilot_candidate_index.jsonl")


def load_pilot_index(index_path: Path = PILOT_INDEX) -> dict[str, set[str]]:
    """
    Load the pilot candidate index and return a mapping:
        dataset_name -> set of source_ids

    Parameters
    ----------
    index_path : Path to pilot_candidate_index.jsonl

    Returns
    -------
    dict mapping dataset name -> set of source_id strings
    """
    pilot: dict[str, set[str]] = {}
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ds = rec.get("dataset", "")
            sid = str(rec.get("source_id", ""))
            if ds not in pilot:
                pilot[ds] = set()
            pilot[ds].add(sid)
    log.info("Pilot index loaded: %s",
             {k: len(v) for k, v in pilot.items()})
    return pilot


def stream_pilot_texts(
    pilot_index: dict[str, set[str]],
    max_words: int | None = None,
) -> Iterator[str]:
    """
    Yield document texts for all pilot-selected documents.

    Streams from the deduplicated JSONL files in the fixed order:
        wikimedia_english -> pg19

    Parameters
    ----------
    pilot_index : output of load_pilot_index()
    max_words   : if set, stop after yielding this many total words

    Yields
    ------
    str : text of each pilot document (no truncation within a document)
    """
    total_words = 0
    sources = [
        ("wikimedia_english", DEDUP_WIKI),
        ("pg19",              DEDUP_PG19),
    ]

    for dataset, jsonl_path in sources:
        pilot_ids = pilot_index.get(dataset, set())
        if not pilot_ids or not jsonl_path.exists():
            continue

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = str(rec.get("source_id", ""))
                if sid not in pilot_ids:
                    continue
                text = rec.get("text", "")
                if not text:
                    continue
                yield text
                total_words += len(text.split())
                if max_words is not None and total_words >= max_words:
                    log.info("Reached word limit %d — stopping stream", max_words)
                    return

    log.info("Pilot stream complete: ~%d words yielded", total_words)


def collect_training_sample(
    pilot_index: dict[str, set[str]],
    max_words: int = 2_000_000,
) -> list[str]:
    """
    Collect up to max_words worth of text into a list for BPE training.

    Using a word-budget limit keeps training fast while still covering
    diverse vocabulary. 2M words (~12MB) is typically sufficient for
    high-quality BPE merge rules.

    Parameters
    ----------
    pilot_index : output of load_pilot_index()
    max_words   : word budget for training sample

    Returns
    -------
    list[str] : training texts (one per document)
    """
    texts = list(stream_pilot_texts(pilot_index, max_words=max_words))
    total = sum(len(t.split()) for t in texts)
    log.info("Training sample: %d texts, ~%d words", len(texts), total)
    return texts
