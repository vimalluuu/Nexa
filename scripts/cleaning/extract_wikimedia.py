"""
Nexa Phase 8.3 - Wikimedia XML Streaming Extractor
====================================================
Streams through the bz2-compressed MediaWiki XML dump using
xml.etree.ElementTree.iterparse — never loads the full XML into RAM.

For each page:
  - Checks namespace (only ns=0 main articles accepted)
  - Skips redirects
  - Extracts: page ID, title, revision text, timestamp
  - Applies wikitext cleaning
  - Applies normalization
  - Applies quality filtering
  - Writes kept documents to JSONL (one JSON object per line)

Provenance fields preserved:
  dataset, source_id, title, source_url, snapshot, namespace, text

Raw file is NEVER modified.
"""

from __future__ import annotations

import bz2
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from scripts.cleaning.clean_wikitext import clean_wikitext
from scripts.cleaning.config import (
    MIN_CHARS, MIN_WORDS, MAX_CHARS,
    WIKI_BASE_URL, WIKI_NAMESPACE_MAIN, WIKI_SNAPSHOT,
)
from scripts.cleaning.filter_docs import FilterStats, filter_document
from scripts.cleaning.normalize import normalize_text

log = logging.getLogger(__name__)

# MediaWiki XML namespace URI
MW_NS = "http://www.mediawiki.org/xml/export-0.11/"


def _tag(local: str) -> str:
    """Expand a local tag name to the full qualified name."""
    return f"{{{MW_NS}}}{local}"


# Precompute qualified tag names for speed
_PAGE_TAG     = _tag("page")
_TITLE_TAG    = _tag("title")
_NS_TAG       = _tag("ns")
_ID_TAG       = _tag("id")
_REDIRECT_TAG = _tag("redirect")
_REVISION_TAG = _tag("revision")
_TEXT_TAG     = _tag("text")
_TIMESTAMP_TAG = _tag("timestamp")


def iter_wiki_pages(bz2_path: Path) -> Iterator[dict]:
    """
    Stream through a bz2 MediaWiki XML dump and yield raw page dicts.

    Yields
    ------
    dict with keys:
        page_id, title, namespace, is_redirect, wikitext, timestamp
    """
    with bz2.open(bz2_path, "rb") as raw_stream:
        for event, elem in ET.iterparse(raw_stream, events=("end",)):
            if elem.tag != _PAGE_TAG:
                continue

            ns_val    = elem.findtext(_NS_TAG, default="")
            title     = elem.findtext(_TITLE_TAG, default="")
            page_id   = elem.findtext(_ID_TAG, default="")
            redirect  = elem.find(_REDIRECT_TAG)

            revision  = elem.find(_REVISION_TAG)
            wikitext  = ""
            timestamp = ""
            if revision is not None:
                text_el = revision.find(_TEXT_TAG)
                wikitext  = (text_el.text or "") if text_el is not None else ""
                timestamp = revision.findtext(_TIMESTAMP_TAG, default="")

            yield {
                "page_id":     page_id,
                "title":       title,
                "namespace":   ns_val,
                "is_redirect": redirect is not None,
                "wikitext":    wikitext,
                "timestamp":   timestamp,
            }

            # CRITICAL: free memory after processing each page
            elem.clear()


def extract_wikimedia(
    bz2_path: Path,
    output_path: Path,
    snapshot: str = WIKI_SNAPSHOT,
    min_chars: int = MIN_CHARS,
    min_words: int = MIN_WORDS,
    max_chars: int = MAX_CHARS,
) -> dict:
    """
    Full extraction pipeline for one Wikimedia dump shard.

    1. Stream pages from bz2 XML
    2. Filter to namespace 0, skip redirects
    3. Clean wikitext → plain text
    4. Normalize
    5. Filter by quality thresholds
    6. Write to JSONL

    Parameters
    ----------
    bz2_path    : Path to the .bz2 dump file
    output_path : Path to write JSONL output
    snapshot    : Snapshot identifier (e.g. '20260901')
    min_chars   : Minimum characters to keep a document
    min_words   : Minimum words to keep a document
    max_chars   : Maximum characters (safety ceiling)

    Returns
    -------
    dict with extraction and filtering statistics
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = FilterStats(dataset="wikimedia_english")
    raw_total     = 0
    skipped_ns    = 0
    skipped_redir = 0
    extraction_errors = 0

    log.info("Starting Wikimedia extraction: %s", bz2_path)
    log.info("Output: %s", output_path)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for page in iter_wiki_pages(bz2_path):
            raw_total += 1

            if raw_total % 5000 == 0:
                log.info(
                    "  Processed %d pages | kept %d | skipped_ns %d | redir %d",
                    raw_total, stats.kept, skipped_ns, skipped_redir,
                )

            # Namespace filter: only main articles (ns=0)
            if page["namespace"] != WIKI_NAMESPACE_MAIN:
                skipped_ns += 1
                continue

            # Skip redirects
            if page["is_redirect"]:
                skipped_redir += 1
                continue

            # Clean wikitext
            try:
                plain = clean_wikitext(page["wikitext"])
            except Exception as exc:
                log.warning("Wikitext cleaning error page_id=%s: %s", page["page_id"], exc)
                extraction_errors += 1
                stats.extraction_failures += 1
                continue

            # Normalize
            try:
                plain = normalize_text(plain)
            except Exception as exc:
                log.warning("Normalization error page_id=%s: %s", page["page_id"], exc)
                extraction_errors += 1
                stats.extraction_failures += 1
                continue

            # Filter
            kept = filter_document(
                plain, stats,
                min_chars=min_chars,
                min_words=min_words,
                max_chars=max_chars,
            )
            if kept is None:
                continue

            # Build output record with full provenance
            record = {
                "dataset":    "wikimedia_english",
                "source_id":  page["page_id"],
                "title":      page["title"],
                "source_url": WIKI_BASE_URL + page["title"].replace(" ", "_"),
                "snapshot":   snapshot,
                "namespace":  page["namespace"],
                "timestamp":  page["timestamp"],
                "text":       kept,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {
        "raw_total_pages":       raw_total,
        "skipped_wrong_ns":      skipped_ns,
        "skipped_redirects":     skipped_redir,
        "extraction_errors":     extraction_errors,
        **stats.to_dict(),
    }
    log.info("Extraction complete: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    from scripts.cleaning.config import RAW_WIKI_DIR, WIKI_OUTPUT_FILE

    bz2_files = sorted(RAW_WIKI_DIR.glob("*.bz2"))
    if not bz2_files:
        log.error("No .bz2 files found in %s", RAW_WIKI_DIR)
        sys.exit(1)

    for bz2_file in bz2_files:
        result = extract_wikimedia(bz2_file, WIKI_OUTPUT_FILE)
        print(json.dumps(result, indent=2))
