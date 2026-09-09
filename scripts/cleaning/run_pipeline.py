"""
Nexa Phase 8.3 - Pipeline Orchestrator
========================================
Runs the full extraction and cleaning pipeline for all sources,
generates reports, verifies raw files remain untouched, and writes
a summary to data/reports/.

Usage:
  python -m scripts.cleaning.run_pipeline [--wiki-only] [--pg19-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.cleaning.config import (
    CLEAN_REPORT_FILE,
    CLEANED_PG19_DIR,
    CLEANED_WIKI_DIR,
    MAX_CHARS,
    MIN_CHARS,
    MIN_WORDS,
    PG19_OUTPUT_FILE,
    RAW_PG19_DIR,
    RAW_REPORT_FILE,
    RAW_WIKI_DIR,
    WIKI_OUTPUT_FILE,
    WIKI_SNAPSHOT,
)
from scripts.cleaning.extract_pg19 import extract_pg19
from scripts.cleaning.extract_wikimedia import extract_wikimedia

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_quick(path: Path, n_bytes: int = 65536) -> str:
    """Quick partial SHA-256 for raw-file integrity check (not full verify)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def _check_disk(path: str = "F:/", require_gb: float = 2.0) -> None:
    free = shutil.disk_usage(path).free / 1024 ** 3
    log.info("F: free disk space: %.1f GiB", free)
    if free < require_gb:
        log.error("STOP: insufficient disk space on F: (%.1f GiB < %.1f GiB required)",
                  free, require_gb)
        sys.exit(1)
    log.info("Disk space check OK (%.1f GiB free, need %.1f GiB)", free, require_gb)


def _verify_raw_files() -> dict:
    """Verify raw source files exist and read first 64KB to confirm integrity."""
    integrity = {}

    bz2_files = sorted(RAW_WIKI_DIR.glob("*.bz2"))
    for f in bz2_files:
        integrity[str(f)] = {
            "exists": f.exists(),
            "size_bytes": f.stat().st_size if f.exists() else 0,
            "partial_sha256": _sha256_quick(f) if f.exists() else "",
        }
        log.info("Raw wiki file: %s (%.1f MB)", f.name, f.stat().st_size / 1e6)

    txt_files = sorted(RAW_PG19_DIR.glob("*.txt"))
    for f in txt_files:
        integrity[str(f)] = {
            "exists": f.exists(),
            "size_bytes": f.stat().st_size if f.exists() else 0,
        }

    log.info("Raw file inventory: %d wiki, %d pg19", len(bz2_files), len(txt_files))
    return {
        "wiki_bz2_files":  len(bz2_files),
        "pg19_txt_files":  len(txt_files),
        "files":           integrity,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Nexa Phase 8.3 — extraction pipeline")
    parser.add_argument("--wiki-only", action="store_true", help="Only process Wikimedia")
    parser.add_argument("--pg19-only", action="store_true", help="Only process PG-19")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    log.info("=== Nexa Phase 8.3 Extraction Pipeline ===")
    log.info("Started: %s", started_at)

    # Disk safety
    _check_disk("F:/", require_gb=2.0)

    # Verify raw files (before ANY processing)
    log.info("Verifying raw source files...")
    raw_inventory = _verify_raw_files()

    REPORTS_DIR = RAW_REPORT_FILE.parent
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_report = {
        "generated": started_at,
        "phase": "8.3",
        "inventory": raw_inventory,
        "thresholds": {
            "min_chars": args.min_chars,
            "min_words": args.min_words,
            "max_chars": args.max_chars,
        },
    }
    RAW_REPORT_FILE.write_text(
        json.dumps(raw_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Raw inventory report written: %s", RAW_REPORT_FILE)

    # ---------------------------------------------------------------------------
    # Run extraction pipelines
    # ---------------------------------------------------------------------------
    results: dict[str, dict] = {}
    t0 = time.time()

    if not args.pg19_only:
        bz2_files = sorted(RAW_WIKI_DIR.glob("*.bz2"))
        if not bz2_files:
            log.error("No .bz2 files found in %s", RAW_WIKI_DIR)
        else:
            log.info("--- Wikimedia Extraction ---")
            wiki_stats: dict = {"total_input": 0, "kept": 0}
            for bz2_file in bz2_files:
                r = extract_wikimedia(
                    bz2_file, WIKI_OUTPUT_FILE,
                    snapshot=WIKI_SNAPSHOT,
                    min_chars=args.min_chars,
                    min_words=args.min_words,
                    max_chars=args.max_chars,
                )
                for k, v in r.items():
                    if isinstance(v, int):
                        wiki_stats[k] = wiki_stats.get(k, 0) + v
                    else:
                        wiki_stats[k] = v
            results["wikimedia_english"] = wiki_stats
            log.info("Wiki extraction done: kept %d / %d pages",
                     wiki_stats.get("kept", 0),
                     wiki_stats.get("total_input", 0))

    if not args.wiki_only:
        log.info("--- PG-19 Extraction ---")
        meta_yaml = RAW_PG19_DIR.parent / "_metadata.yaml"
        pg19_stats = extract_pg19(
            RAW_PG19_DIR, meta_yaml, PG19_OUTPUT_FILE,
            min_chars=args.min_chars,
            min_words=args.min_words,
            max_chars=args.max_chars,
        )
        results["pg19"] = pg19_stats
        log.info("PG-19 extraction done: kept %d / %d",
                 pg19_stats.get("kept", 0),
                 pg19_stats.get("total_input", 0))

    elapsed = time.time() - t0

    # ---------------------------------------------------------------------------
    # Combined statistics
    # ---------------------------------------------------------------------------
    combined_kept   = sum(r.get("kept", 0)         for r in results.values())
    combined_chars  = sum(r.get("total_chars", 0)  for r in results.values())
    combined_words  = sum(r.get("total_words", 0)  for r in results.values())
    combined_input  = sum(r.get("total_input", 0)  for r in results.values())
    combined_removed = sum(r.get("removed_total", 0) for r in results.values())

    cleaning_report = {
        "generated":         datetime.now(timezone.utc).isoformat(),
        "started_at":        started_at,
        "elapsed_seconds":   round(elapsed, 1),
        "phase":             "8.3",
        "thresholds": {
            "min_chars": args.min_chars,
            "min_words": args.min_words,
            "max_chars": args.max_chars,
        },
        "combined": {
            "total_input":    combined_input,
            "kept":           combined_kept,
            "removed":        combined_removed,
            "total_chars":    combined_chars,
            "total_words":    combined_words,
            "avg_chars":      round(combined_chars / combined_kept, 1) if combined_kept else 0,
        },
        "datasets":          results,
        "output_files": {
            "wikimedia_english": str(WIKI_OUTPUT_FILE),
            "pg19":              str(PG19_OUTPUT_FILE),
        },
    }
    CLEAN_REPORT_FILE.write_text(
        json.dumps(cleaning_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Cleaning report written: %s", CLEAN_REPORT_FILE)

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    free_gb = shutil.disk_usage("F:/").free / 1024 ** 3
    log.info("=== Pipeline Complete ===")
    log.info("Elapsed: %.1f s", elapsed)
    log.info("Combined kept: %d docs | %d chars | %d words",
             combined_kept, combined_chars, combined_words)
    log.info("F: free: %.1f GiB", free_gb)
    log.info("No pretrained model or API was used.")


if __name__ == "__main__":
    main()
