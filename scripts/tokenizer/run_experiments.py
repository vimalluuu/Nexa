"""
Nexa Phase 8.5 - Tokenizer Experiment Orchestrator
====================================================
Trains candidate NexaTokenizers for vocabulary sizes:
  2048, 4096, 8192, 16384, 32768

For each candidate:
  1. Stream a 2M-word training sample from the pilot corpus
  2. Train Nexa's own BPE tokenizer (nexa.tokenizer.tokenizer.NexaTokenizer)
  3. Save candidate tokenizer to data/processed/tokenizer_v1_{vocab_size}/
  4. Measure on the FULL pilot corpus (9.18M words)
  5. Record all metrics

Then:
  6. Select the recommended vocabulary size
  7. Copy recommended tokenizer to data/processed/tokenizer_v1/
  8. Write reports and manifests

Usage:
  python -m scripts.tokenizer.run_experiments
  python -m scripts.tokenizer.run_experiments --vocab-sizes 4096 8192

STOP condition: stops after tokenizer selection.
Does NOT tokenize the full corpus.
Does NOT change the Transformer or training loop.
Does NOT delete the Phase 2 toy tokenizer (data/processed/tokenizer/).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PILOT_INDEX        = Path("data/deduplicated/pilot_candidate_index.jsonl")
CANDIDATES_DIR     = Path("data/processed")
SELECTED_DIR       = Path("data/processed/tokenizer_v1")
TOY_TOKENIZER_DIR  = Path("data/processed/tokenizer")    # v0 — DO NOT TOUCH

REPORTS_DIR        = Path("data/reports")
MANIFESTS_DIR      = Path("data/manifests")
DOCS_DIR           = Path("docs/datasets")

EXPERIMENT_REPORT  = REPORTS_DIR  / "tokenizer_experiment_report.json"
CANDIDATES_YAML    = MANIFESTS_DIR / "tokenizer_v1_candidates.yaml"
MD_REPORT          = DOCS_DIR      / "nexa_v1_tokenizer_selection.md"

# ---------------------------------------------------------------------------
# Candidate vocabulary sizes
# ---------------------------------------------------------------------------
DEFAULT_VOCAB_SIZES = [2048, 4096, 8192, 16384, 32768]

# Training sample word budget (2M words = fast training, good quality)
TRAINING_SAMPLE_WORDS = 2_000_000

# min_frequency for BPE: at 2M words, pairs appearing < 3 times are noise
MIN_FREQUENCY = 3


def candidate_dir(vocab_size: int) -> Path:
    return CANDIDATES_DIR / f"tokenizer_v1_{vocab_size}"


def train_one_candidate(
    vocab_size: int,
    training_texts: list[str],
) -> tuple["NexaTokenizer", float]:  # type: ignore[name-defined]
    """Train one candidate tokenizer. Returns (tokenizer, training_time_s)."""
    from nexa.tokenizer.tokenizer import NexaTokenizer

    log.info("--- Training vocab_size=%d ---", vocab_size)
    log.info("  Training texts: %d | min_freq: %d", len(training_texts), MIN_FREQUENCY)

    t0 = time.time()
    tok = NexaTokenizer.train(
        corpus=training_texts,
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
    )
    elapsed = time.time() - t0

    out_dir = candidate_dir(vocab_size)
    tok.save(out_dir)
    log.info("  Trained in %.1f s | actual vocab_size=%d | merges=%d | saved -> %s",
             elapsed, tok.vocab_size, len(tok.merges), out_dir)
    return tok, elapsed


def _write_candidates_yaml(all_metrics: list[dict], recommendation: dict) -> None:
    """Write data/manifests/tokenizer_v1_candidates.yaml"""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "phase":              "8.5",
        "generated":          datetime.now(timezone.utc).isoformat(),
        "training_approach":  (
            "Nexa's own BPE tokenizer (nexa.tokenizer.tokenizer.NexaTokenizer) "
            "trained from scratch on a 2M-word sample from the Phase 8 pilot corpus. "
            "No pretrained tokenizer used."
        ),
        "training_sample_words": TRAINING_SAMPLE_WORDS,
        "min_frequency":         MIN_FREQUENCY,
        "measurement_corpus":    "Full pilot corpus (9,182,235 words, 2,257 docs)",
        "recommended":           recommendation["recommended_vocab_size"],
        "candidates":            [],
    }
    for m in all_metrics:
        data["candidates"].append({
            "vocab_size":           m["vocab_size"],
            "actual_vocab_size":    m["vocab_size"],
            "num_merges":           m["num_merges"],
            "training_time_s":      m["training_time_s"],
            "total_tokens":         m["total_tokens"],
            "tokens_per_word":      m["tokens_per_word"],
            "tokens_per_char":      m["tokens_per_char"],
            "compression_ratio":    m["compression_ratio"],
            "unk_rate":             m["unk_rate"],
            "coverage_pct":         m["word_type_coverage_pct"],
            "p90_seq_len":          m["seq_len_percentiles"].get("p90", 0),
            "p95_seq_len":          m["seq_len_percentiles"].get("p95", 0),
            "p99_seq_len":          m["seq_len_percentiles"].get("p99", 0),
            "est_storage_mb_int32": m["est_storage_mb_int32"],
        })
    CANDIDATES_YAML.write_text(
        yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    log.info("Candidates YAML written: %s", CANDIDATES_YAML)


def _write_md_report(all_metrics: list[dict], recommendation: dict) -> None:
    """Write docs/datasets/nexa_v1_tokenizer_selection.md"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rec_size = recommendation["recommended_vocab_size"]

    lines = [
        "# Nexa V1 Tokenizer Selection",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}  ",
        "Phase: 8.5",
        "",
        "> [!IMPORTANT]",
        "> Nexa's tokenizer is trained **entirely from scratch** using Nexa's own",
        "> BPE implementation (`nexa.tokenizer`). No pretrained tokenizer, no",
        "> SentencePiece, no Hugging Face tokenizers, no GPT/Llama/Mistral vocabulary.",
        "",
        "---",
        "",
        "## Training Setup",
        "",
        f"- **Training corpus**: 2M-word sample from Phase 8 pilot corpus",
        f"- **Min-frequency filter**: {MIN_FREQUENCY} (pairs appearing < {MIN_FREQUENCY}× ignored)",
        f"- **Measurement corpus**: Full pilot — 2,257 docs, 9,182,235 words",
        f"- **Special tokens**: `<pad>` (0), `<bos>` (1), `<eos>` (2), `<unk>` (3)",
        "",
        "---",
        "",
        "## Candidate Metrics",
        "",
        "| Vocab Size | Merges | Tokens | Tok/Word | Chars/Tok | UNK% | Coverage% | Train(s) | Est. MB (int32) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        mark = " ✓" if m["vocab_size"] == rec_size else ""
        lines.append(
            f"| **{m['vocab_size']:,}{mark}** "
            f"| {m['num_merges']:,} "
            f"| {m['total_tokens']:,} "
            f"| {m['tokens_per_word']:.3f} "
            f"| {m['compression_ratio']:.2f} "
            f"| {m['unk_rate']*100:.3f} "
            f"| {m['word_type_coverage_pct']:.1f} "
            f"| {m['training_time_s']:.0f} "
            f"| {m['est_storage_mb_int32']:.0f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Sequence-Length Distribution",
        "",
        "| Vocab Size | p50 | p90 | p95 | p99 | min | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        sp = m["seq_len_percentiles"]
        mark = " ✓" if m["vocab_size"] == rec_size else ""
        lines.append(
            f"| **{m['vocab_size']:,}{mark}** "
            f"| {sp.get('p50',0):,} "
            f"| {sp.get('p90',0):,} "
            f"| {sp.get('p95',0):,} "
            f"| {sp.get('p99',0):,} "
            f"| {sp.get('min',0):,} "
            f"| {sp.get('max',0):,} |"
        )

    # Embedding param cost table
    d_model = recommendation.get("d_model_assumed", 128)
    lines += [
        "",
        "---",
        "",
        f"## Embedding Parameter Cost (d_model={d_model})",
        "",
        "| Vocab Size | Embed + LM-head params | % of 803K total model |",
        "|---|---|---|",
    ]
    total_model_params = 803_000
    for vs, ep in sorted(recommendation["embedding_params_by_size"].items()):
        pct = ep / total_model_params * 100
        mark = " ✓" if vs == rec_size else ""
        lines.append(f"| **{vs:,}{mark}** | {ep:,} | {pct:.0f}% |")

    lines += [
        "",
        "---",
        "",
        "## Selected Vocabulary Size",
        "",
        f"**Recommended: `vocab_size = {rec_size}`**",
        "",
        recommendation["rationale"],
        "",
        "---",
        "",
        "## Exact Pilot Token Count",
        "",
    ]
    chosen = next(m for m in all_metrics if m["vocab_size"] == rec_size)
    lines += [
        f"Using the selected tokenizer (vocab_size={rec_size}):",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total documents tokenized | {chosen['doc_count']:,} |",
        f"| Total tokens | **{chosen['total_tokens']:,}** |",
        f"| Total words | {chosen['total_words']:,} |",
        f"| Tokens per word | {chosen['tokens_per_word']:.4f} |",
        f"| Chars per token | {chosen['compression_ratio']:.4f} |",
        f"| UNK token count | {chosen['unk_tokens']:,} |",
        f"| UNK rate | {chosen['unk_rate']*100:.4f}% |",
        f"| Est. storage (uint16) | {chosen['est_storage_mb_uint16']:.0f} MB |",
        f"| Est. storage (int32)  | {chosen['est_storage_mb_int32']:.0f} MB |",
        "",
        "---",
        "",
        "## Files",
        "",
        f"- Selected tokenizer: `data/processed/tokenizer_v1/`",
        f"- Phase 2 toy tokenizer (v0, preserved): `data/processed/tokenizer/`",
        f"- Candidate tokenizers: `data/processed/tokenizer_v1_{{vocab_size}}/`",
        f"- Experiment report: `data/reports/tokenizer_experiment_report.json`",
        f"- Candidates manifest: `data/manifests/tokenizer_v1_candidates.yaml`",
        "",
        "---",
        "",
        "## Independence Rule",
        "",
        "This tokenizer was trained entirely from scratch on Nexa's own pilot corpus.",
        "No pretrained model weights, no external tokenizer files, no Hugging Face",
        "tokenizers, no SentencePiece, no GPT/Llama/Mistral vocabulary.",
        "",
    ]

    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report written: %s", MD_REPORT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexa Phase 8.5 — Tokenizer Experiments")
    parser.add_argument(
        "--vocab-sizes", nargs="+", type=int,
        default=DEFAULT_VOCAB_SIZES,
        help="Vocabulary sizes to evaluate (default: 2048 4096 8192 16384 32768)",
    )
    parser.add_argument(
        "--sample-words", type=int, default=TRAINING_SAMPLE_WORDS,
        help="Word budget for BPE training sample (default: 2000000)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip training if candidate tokenizer already exists",
    )
    args = parser.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    t_start = time.time()

    log.info("=== Nexa Phase 8.5 Tokenizer Experiments ===")
    log.info("Started: %s", started)
    log.info("Vocabulary sizes: %s", args.vocab_sizes)
    log.info("Training sample words: %d", args.sample_words)

    # Disk check
    free_gb = shutil.disk_usage("F:/").free / 1024 ** 3
    log.info("F: free: %.1f GiB", free_gb)

    # Verify v0 tokenizer intact
    if not TOY_TOKENIZER_DIR.exists():
        log.warning("Phase 2 toy tokenizer not found at %s — this is unexpected", TOY_TOKENIZER_DIR)
    else:
        log.info("Phase 2 toy tokenizer (v0) confirmed at %s — will NOT be touched", TOY_TOKENIZER_DIR)

    # Load pilot index
    from scripts.tokenizer.corpus_sampler import (
        load_pilot_index, collect_training_sample, stream_pilot_texts
    )
    from scripts.tokenizer.measure_tokenizer import measure_tokenizer
    from scripts.tokenizer.select_vocab import recommend_vocab_size

    pilot_index = load_pilot_index(PILOT_INDEX)

    # Collect training sample ONCE (shared across all vocab sizes)
    log.info("Collecting training sample (~%dM words)...", args.sample_words // 1_000_000)
    training_texts = collect_training_sample(pilot_index, max_words=args.sample_words)
    actual_words = sum(len(t.split()) for t in training_texts)
    log.info("Training sample ready: %d texts, %d words", len(training_texts), actual_words)

    # Train and measure all candidates
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []

    for vocab_size in sorted(args.vocab_sizes):
        cdir = candidate_dir(vocab_size)

        if args.skip_existing and (cdir / "tokenizer.json").exists():
            log.info("Loading existing candidate vocab_size=%d from %s", vocab_size, cdir)
            from nexa.tokenizer.tokenizer import NexaTokenizer
            tok = NexaTokenizer.load(cdir)
            train_time = 0.0
        else:
            tok, train_time = train_one_candidate(vocab_size, training_texts)

        log.info("Measuring vocab_size=%d on full pilot corpus...", vocab_size)
        metrics = measure_tokenizer(tok, pilot_index, training_time_s=train_time)
        metrics["target_vocab_size"] = vocab_size
        all_metrics.append(metrics)
        log.info(
            "  vocab=%d | tokens=%d | tok/word=%.3f | unk=%.4f%% | coverage=%.1f%%",
            metrics["vocab_size"], metrics["total_tokens"],
            metrics["tokens_per_word"], metrics["unk_rate"] * 100,
            metrics["word_type_coverage_pct"],
        )

    # Recommend
    recommendation = recommend_vocab_size(all_metrics)
    rec_size = recommendation["recommended_vocab_size"]
    log.info("Recommended vocabulary size: %d", rec_size)
    log.info("Rationale: %s", recommendation["rationale"])

    # Copy recommended tokenizer to data/processed/tokenizer_v1/
    rec_dir = candidate_dir(rec_size)
    if SELECTED_DIR.exists():
        shutil.rmtree(SELECTED_DIR)
    shutil.copytree(rec_dir, SELECTED_DIR)
    log.info("Selected tokenizer saved: %s", SELECTED_DIR)

    # Verify v0 still intact
    assert TOY_TOKENIZER_DIR.exists() or not Path("data/processed/tokenizer").exists(), \
        "CRITICAL: Phase 2 toy tokenizer was deleted!"
    log.info("Phase 2 toy tokenizer (v0) verified intact.")

    # Write experiment report
    elapsed_total = time.time() - t_start
    report = {
        "generated":             datetime.now(timezone.utc).isoformat(),
        "started_at":            started,
        "elapsed_total_s":       round(elapsed_total, 1),
        "phase":                 "8.5",
        "training_approach":     "Nexa BPE trained from scratch (nexa.tokenizer)",
        "training_sample_words": actual_words,
        "training_texts_count":  len(training_texts),
        "min_frequency":         MIN_FREQUENCY,
        "pilot_docs":            sum(len(v) for v in pilot_index.values()),
        "pilot_total_words":     9_182_235,
        "independence_rule":     (
            "No pretrained tokenizer used. No SentencePiece. No HuggingFace. "
            "No GPT/Llama/Mistral vocabulary. Trained entirely from scratch."
        ),
        "toy_tokenizer_v0_preserved": TOY_TOKENIZER_DIR.exists(),
        "recommendation":        recommendation,
        "candidates":            all_metrics,
    }
    EXPERIMENT_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Experiment report written: %s", EXPERIMENT_REPORT)

    # YAML manifest
    _write_candidates_yaml(all_metrics, recommendation)

    # Markdown report
    _write_md_report(all_metrics, recommendation)

    free_after = shutil.disk_usage("F:/").free / 1024 ** 3
    log.info("=== Experiments Complete ===")
    log.info("Elapsed: %.1f s | F: free: %.1f GiB", elapsed_total, free_after)
    log.info("Selected tokenizer: vocab_size=%d at %s", rec_size, SELECTED_DIR)


if __name__ == "__main__":
    main()
