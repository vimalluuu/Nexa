"""
Nexa Phase 8.4 - Dedup + Analysis Pipeline Orchestrator
=========================================================
Runs deduplication, corpus analysis, balance analysis, and pilot selection.
Writes all reports to data/reports/.

Usage:
  python -m scripts.dedup.run_dedup
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLEANED_WIKI    = Path("data/cleaned/wikimedia_english/docs.jsonl")
CLEANED_PG19    = Path("data/cleaned/pg19/docs.jsonl")
DEDUP_WIKI      = Path("data/deduplicated/wikimedia_english/docs.jsonl")
DEDUP_PG19      = Path("data/deduplicated/pg19/docs.jsonl")
PILOT_INDEX     = Path("data/deduplicated/pilot_candidate_index.jsonl")

REPORTS_DIR     = Path("data/reports")
DEDUP_REPORT    = REPORTS_DIR / "deduplication_report.json"
ANALYSIS_REPORT = REPORTS_DIR / "corpus_analysis_report.json"
MD_REPORT       = Path("docs/datasets/nexa_v1_corpus_analysis.md")


def main() -> None:
    from scripts.dedup.exact_dedup import run_deduplication
    from scripts.dedup.corpus_stats import compute_stats, compute_combined_stats
    from scripts.dedup.balance_analysis import analyze_balance, select_pilot_candidates

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # Disk check
    free_gb = shutil.disk_usage("F:/").free / 1024 ** 3
    log.info("F: free: %.1f GiB", free_gb)
    if free_gb < 2.0:
        log.error("Insufficient disk space"); sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MD_REPORT.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. DEDUPLICATION
    # =========================================================================
    log.info("=== Phase 8.4a: Deduplication ===")
    sources = [
        (CLEANED_WIKI, DEDUP_WIKI, "wikimedia_english"),
        (CLEANED_PG19, DEDUP_PG19, "pg19"),
    ]
    _, dedup_combined = run_deduplication(sources)

    dedup_report = {
        "generated":  datetime.now(timezone.utc).isoformat(),
        "phase":      "8.4",
        "note":       "Exact SHA-256 deduplication across all sources. First occurrence kept.",
        **dedup_combined,
    }
    DEDUP_REPORT.write_text(
        json.dumps(dedup_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Dedup report written: %s", DEDUP_REPORT)

    # =========================================================================
    # 2. CORPUS STATISTICS — cleaned
    # =========================================================================
    log.info("=== Phase 8.4b: Corpus Statistics (cleaned) ===")
    cleaned_wiki_stats = compute_stats(CLEANED_WIKI, "wikimedia_english_cleaned")
    cleaned_pg19_stats = compute_stats(CLEANED_PG19, "pg19_cleaned")
    cleaned_combined   = compute_combined_stats([cleaned_wiki_stats, cleaned_pg19_stats])

    # =========================================================================
    # 3. CORPUS STATISTICS — deduplicated
    # =========================================================================
    log.info("=== Phase 8.4c: Corpus Statistics (deduplicated) ===")
    dedup_wiki_stats = compute_stats(DEDUP_WIKI, "wikimedia_english_dedup")
    dedup_pg19_stats = compute_stats(DEDUP_PG19, "pg19_dedup")
    dedup_combined_stats = compute_combined_stats([dedup_wiki_stats, dedup_pg19_stats])

    # =========================================================================
    # 4. BALANCE ANALYSIS
    # =========================================================================
    log.info("=== Phase 8.4d: Balance Analysis ===")
    source_summary = {
        "wikimedia_english": {
            "doc_count":  dedup_wiki_stats.doc_count,
            "word_total": dedup_wiki_stats.word_total,
        },
        "pg19": {
            "doc_count":  dedup_pg19_stats.doc_count,
            "word_total": dedup_pg19_stats.word_total,
        },
    }
    balance_scenarios = analyze_balance(source_summary)

    # =========================================================================
    # 5. PILOT CANDIDATE SELECTION
    # =========================================================================
    log.info("=== Phase 8.4e: Pilot Candidate Selection ===")
    pilot = select_pilot_candidates(DEDUP_WIKI, DEDUP_PG19, PILOT_INDEX)

    # =========================================================================
    # 6. WRITE ANALYSIS REPORT
    # =========================================================================
    analysis_report = {
        "generated":     datetime.now(timezone.utc).isoformat(),
        "phase":         "8.4",
        "elapsed_s":     round(time.time() - t0, 1),
        "cleaned_corpus":     cleaned_combined,
        "deduplicated_corpus": dedup_combined_stats,
        "deduplication_summary": dedup_combined,
        "balance_analysis":  balance_scenarios,
        "pilot_selection":   pilot.to_dict(),
        "token_count_note": (
            "Exact Nexa BPE token count is UNKNOWN until Nexa's own BPE tokenizer "
            "is trained on this corpus and applied. All 'token' estimates in this "
            "report are rough approximations based on English BPE ratios (1.2–1.5 "
            "tokens per word). Do not treat these as authoritative token counts."
        ),
    }
    ANALYSIS_REPORT.write_text(
        json.dumps(analysis_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Analysis report written: %s", ANALYSIS_REPORT)

    # =========================================================================
    # 7. WRITE MARKDOWN REPORT
    # =========================================================================
    _write_md_report(analysis_report, dedup_report, pilot)

    free_after = shutil.disk_usage("F:/").free / 1024 ** 3
    log.info("=== Pipeline Complete ===")
    log.info("Elapsed: %.1f s", time.time() - t0)
    log.info("F: free: %.1f GiB", free_after)


def _write_md_report(analysis: dict, dedup: dict, pilot) -> None:
    """Write the human-readable Markdown corpus analysis report."""
    cleaned   = analysis["cleaned_corpus"]
    deduped   = analysis["deduplicated_corpus"]
    d         = dedup
    bal       = analysis["balance_analysis"]
    ps        = analysis["pilot_selection"]

    lines = [
        "# Nexa V1 Corpus Analysis",
        "",
        f"Generated: {analysis['generated']}  ",
        f"Phase: 8.4",
        "",
        "> [!IMPORTANT]",
        "> **Exact Nexa BPE token counts are UNKNOWN until Nexa's own BPE tokenizer is",
        "> trained and applied.** All token estimates below are rough approximations",
        "> (English BPE ratio: 1.2–1.5 tokens per word) and must not be treated as",
        "> authoritative counts.",
        "",
        "---",
        "",
        "## 1. Cleaned Corpus",
        "",
        "| Metric | Wikimedia | PG-19 | Combined |",
        "|---|---|---|---|",
    ]

    wiki_c = next(s for s in cleaned["per_source"] if "wikimedia" in s["dataset"])
    pg19_c = next(s for s in cleaned["per_source"] if "pg19"      in s["dataset"])

    def fmt(n): return f"{n:,}"
    lines += [
        f"| Documents | {fmt(wiki_c['doc_count'])} | {fmt(pg19_c['doc_count'])} | {fmt(cleaned['total_docs'])} |",
        f"| Characters | {fmt(wiki_c['char_total'])} | {fmt(pg19_c['char_total'])} | {fmt(cleaned['total_chars'])} |",
        f"| Words | {fmt(wiki_c['word_total'])} | {fmt(pg19_c['word_total'])} | {fmt(cleaned['total_words'])} |",
        f"| Avg chars/doc | {fmt(int(wiki_c['avg_chars']))} | {fmt(int(pg19_c['avg_chars']))} | {fmt(int(cleaned['avg_chars']))} |",
        f"| Median chars/doc | {fmt(wiki_c['median_chars'])} | {fmt(pg19_c['median_chars'])} | — |",
        "",
        "---",
        "",
        "## 2. Deduplication",
        "",
        f"- **Input documents**: {fmt(d['total_input_docs'])}",
        f"- **Kept (deduplicated)**: {fmt(d['total_kept_docs'])}",
        f"- **Duplicates removed**: {fmt(d['total_duplicate_docs'])} ({d['total_duplicate_pct']:.2f}%)",
        "",
    ]

    for src in d.get("per_source", []):
        lines.append(
            f"  - {src['dataset']}: {fmt(src['input_docs'])} in → "
            f"{fmt(src['kept_docs'])} kept "
            f"({fmt(src['duplicate_docs'])} dupes, {src['duplicate_pct']:.2f}%)"
        )

    wiki_d = next(s for s in deduped["per_source"] if "wikimedia" in s["dataset"])
    pg19_d = next(s for s in deduped["per_source"] if "pg19"      in s["dataset"])

    lines += [
        "",
        "---",
        "",
        "## 3. Deduplicated Corpus",
        "",
        "| Metric | Wikimedia | PG-19 | Combined |",
        "|---|---|---|---|",
        f"| Documents | {fmt(wiki_d['doc_count'])} | {fmt(pg19_d['doc_count'])} | {fmt(deduped['total_docs'])} |",
        f"| Characters | {fmt(wiki_d['char_total'])} | {fmt(pg19_d['char_total'])} | {fmt(deduped['total_chars'])} |",
        f"| Words | {fmt(wiki_d['word_total'])} | {fmt(pg19_d['word_total'])} | {fmt(deduped['total_words'])} |",
        f"| Avg chars/doc | {fmt(int(wiki_d['avg_chars']))} | {fmt(int(pg19_d['avg_chars']))} | {fmt(int(deduped['avg_chars']))} |",
        f"| Median chars | {fmt(wiki_d['median_chars'])} | {fmt(pg19_d['median_chars'])} | — |",
        f"| Min chars | {fmt(wiki_d['char_min'])} | {fmt(pg19_d['char_min'])} | — |",
        f"| Max chars | {fmt(wiki_d['char_max'])} | {fmt(pg19_d['char_max'])} | — |",
        f"| Potentially repetitive docs (TTR < 0.10) | {wiki_d['low_ttr_docs']} | {pg19_d['low_ttr_docs']} | — |",
        "",
        "**Source proportions (deduplicated):**",
        "",
        "| Source | Doc % | Word % |",
        "|---|---|---|",
    ]
    for ds, prop in deduped.get("source_proportions", {}).items():
        lines.append(f"| {ds} | {prop['doc_pct']:.1f}% | {prop['word_pct']:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## 4. Source Balance Scenarios",
        "",
        "| Ratio (Wiki/PG-19) | Wiki words | PG-19 words | Total words | Est. docs |",
        "|---|---|---|---|---|",
    ]
    for sc in bal:
        w = sc["wikimedia_english"]
        p = sc["pg19"]
        c = sc["combined"]
        lines.append(
            f"| {sc['ratio']} | {fmt(w['available_words'])} | {fmt(p['available_words'])} | "
            f"{fmt(c['total_words'])} | {fmt(c['total_est_docs'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Pilot Candidate Selection",
        "",
        f"- **Strategy**: Sort by source_id ascending (deterministic); take first N Wikimedia articles until word target reached; include all PG-19 books",
        f"- **Random seed**: {ps['seed']}",
        f"- **Target word count**: {fmt(ps['target_words'])} (planning figure)",
        f"- **Wikimedia selected**: {fmt(ps['wiki_selected_docs'])} docs, {fmt(ps['wiki_actual_words'])} words",
        f"- **PG-19 selected**: {fmt(ps['pg19_selected_docs'])} docs, {fmt(ps['pg19_actual_words'])} words",
        f"- **Total selected**: {fmt(ps['total_selected_docs'])} docs, {fmt(ps['total_actual_words'])} words",
        f"- **Estimated token range**: {fmt(ps['estimated_tokens_low'])}–{fmt(ps['estimated_tokens_high'])} (English BPE 1.2–1.5×)",
        "",
        "> [!CAUTION]",
        "> **Exact Nexa token count = UNKNOWN**",
        "> The token count will be determined when Nexa's own BPE tokenizer is",
        "> trained on this corpus and applied to the selected documents.",
        "",
        "---",
        "",
        "## 6. Document-Length Distribution (Wikimedia, deduplicated)",
        "",
        "| Percentile | Characters | Words |",
        "|---|---|---|",
    ]
    char_pct = wiki_d.get("char_percentiles", {})
    word_pct = wiki_d.get("word_percentiles", {})
    for p in ["p10", "p25", "p50", "p75", "p90", "p99"]:
        lines.append(f"| {p} | {fmt(int(char_pct.get(p, 0)))} | {fmt(int(word_pct.get(p, 0)))} |")

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- Raw files remain **untouched** (`data/raw/`)",
        "- Cleaned data is at `data/cleaned/`",
        "- Deduplicated data is at `data/deduplicated/`",
        "- Pilot index is at `data/deduplicated/pilot_candidate_index.jsonl`",
        "- No deduplication, tokenization, or training was performed on any AI model",
        "- No pretrained model or external AI/LLM API was used at any stage",
        "",
    ]

    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report written: %s", MD_REPORT)


if __name__ == "__main__":
    main()
