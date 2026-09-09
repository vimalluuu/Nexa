"""
Nexa Phase 8.3 - PG-19 Plain Text Extractor
=============================================
Processes plain .txt files from the PG-19 dataset.

PG-19 README states:
  "The only processing of the text that has been applied is the removal
   of boilerplate license text, and the mapping of offensive discriminatory
   words as specified by Ofcom to placeholder <DW> tokens."

So PG-19 files are already substantially clean. This extractor:
  1. Reads each .txt file
  2. Loads corresponding metadata from _metadata.yaml
  3. Applies normalization (unicode, whitespace)
  4. Applies quality filtering
  5. Writes to JSONL with full provenance

PG-19 boilerplate removal:
  The boilerplate was already removed by PG-19's own processing.
  We do minimal additional cleanup: strip leading/trailing whitespace
  and remove any remaining Project Gutenberg header/footer artifacts
  that PG-19 might have left (heuristic patterns only).

Raw files are NEVER modified.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import yaml

from scripts.cleaning.config import (
    MIN_CHARS, MIN_WORDS, MAX_CHARS,
    PG19_GCS_BASE,
)
from scripts.cleaning.filter_docs import FilterStats, filter_document
from scripts.cleaning.normalize import normalize_text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Residual PG boilerplate patterns (heuristic, conservative)
# ---------------------------------------------------------------------------
# Match lines that are clearly Project Gutenberg header/footer artifacts.
# PG-19 already removed most; these catch any remainders.
_PG_HEADER_RE = re.compile(
    r"^\s*(?:The Project Gutenberg|START OF (?:THE |THIS )?PROJECT GUTENBERG|"
    r"Produced by|This eBook is for the use of anyone|"
    r"GUTENBERG[- ]LITERARY ARCHIVE|TERMS OF USE|Release Date:|Posted:|"
    r"Language:\s*English|Character set encoding:|"
    r"\*\*\*\s*START OF|END OF THE PROJECT)",
    re.IGNORECASE | re.MULTILINE,
)

_PG_FOOTER_START_RE = re.compile(
    r"^\s*(?:END OF (?:THE |THIS )?PROJECT GUTENBERG|\*\*\*\s*END OF)",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_pg_boilerplate(text: str) -> str:
    """
    Remove any residual Project Gutenberg header/footer blocks.
    PG-19 has already done this; we add a conservative guard only.
    """
    # If a footer marker is found, truncate there
    footer = _PG_FOOTER_START_RE.search(text)
    if footer:
        text = text[:footer.start()]

    # Remove header lines (first 30 lines that match boilerplate patterns)
    lines = text.splitlines()
    start_line = 0
    for i, line in enumerate(lines[:30]):
        if _PG_HEADER_RE.match(line):
            start_line = i + 1
    if start_line > 0:
        text = "\n".join(lines[start_line:])

    return text


def extract_pg19(
    source_dir: Path,
    metadata_yaml: Path,
    output_path: Path,
    min_chars: int = MIN_CHARS,
    min_words: int = MIN_WORDS,
    max_chars: int = MAX_CHARS,
) -> dict:
    """
    Extract and clean all PG-19 books in source_dir.

    Parameters
    ----------
    source_dir    : Directory containing <book_id>.txt files
    metadata_yaml : Path to _metadata.yaml with per-book records
    output_path   : JSONL output file
    min_chars     : Minimum characters
    min_words     : Minimum words
    max_chars     : Maximum characters

    Returns
    -------
    dict with extraction and filtering statistics
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata
    book_meta: dict[str, dict] = {}
    if metadata_yaml.exists():
        with open(metadata_yaml, encoding="utf-8") as f:
            raw_meta = yaml.safe_load(f) or {}
        for entry in raw_meta.get("files", []):
            bid = str(entry.get("book_id", ""))
            book_meta[bid] = entry
    else:
        log.warning("No _metadata.yaml found at %s", metadata_yaml)

    stats = FilterStats(dataset="pg19")
    extraction_errors = 0

    txt_files = sorted(source_dir.glob("*.txt"))
    log.info("Found %d .txt files in %s", len(txt_files), source_dir)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for txt_path in txt_files:
            book_id = txt_path.stem  # e.g. "11" from "11.txt"

            try:
                raw_text = txt_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                log.warning("Failed to read %s: %s", txt_path, exc)
                extraction_errors += 1
                stats.extraction_failures += 1
                continue

            # Strip residual PG boilerplate (conservative)
            text = _strip_pg_boilerplate(raw_text)

            # Normalize
            try:
                text = normalize_text(text)
            except Exception as exc:
                log.warning("Normalization failed for %s: %s", txt_path, exc)
                extraction_errors += 1
                stats.extraction_failures += 1
                continue

            # Filter
            kept = filter_document(
                text, stats,
                min_chars=min_chars,
                min_words=min_words,
                max_chars=max_chars,
            )
            if kept is None:
                log.debug("Filtered out book_id=%s", book_id)
                continue

            # Build provenance record
            meta_entry = book_meta.get(book_id, {})
            record = {
                "dataset":          "pg19",
                "source_id":        book_id,
                "title":            meta_entry.get("title", ""),
                "publication_year": meta_entry.get("publication_date", ""),
                "gutenberg_url":    meta_entry.get("gutenberg_url", ""),
                "source_url":       f"{PG19_GCS_BASE}/{book_id}.txt",
                "license_note":     meta_entry.get("license_note", ""),
                "text":             kept,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.debug("Kept book_id=%s '%s' (%d chars)",
                      book_id, meta_entry.get("title", "?")[:40], len(kept))

    result = {
        "txt_files_found":   len(txt_files),
        "extraction_errors": extraction_errors,
        **stats.to_dict(),
    }
    log.info("PG-19 extraction complete: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    from scripts.cleaning.config import (
        RAW_PG19_DIR, PG19_OUTPUT_FILE,
        CLEANED_PG19_DIR,
    )

    meta_yaml = RAW_PG19_DIR.parent / "_metadata.yaml"
    result = extract_pg19(RAW_PG19_DIR, meta_yaml, PG19_OUTPUT_FILE)
    print(json.dumps(result, indent=2))
