"""
Nexa Phase 8.5 - Tokenizer Metrics Measurement
===============================================
Measures a trained NexaTokenizer against the full pilot corpus.

Metrics computed:
  - Final vocabulary size
  - Number of BPE merges
  - Exact pilot token count
  - Tokens per word
  - Tokens per character
  - Compression ratio (chars / tokens)
  - Unknown-token rate
  - Coverage (% of word types in vocab)
  - Sequence-length statistics (per-document):
      median, p90, p95, p99, min, max, mean
  - Tokenizer training time (passed in)
  - Estimated tokenized storage (bytes)

Memory safety: streams the corpus, never accumulates all token sequences.
Uses reservoir sampling for percentile estimation.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
from pathlib import Path
from typing import Iterator

from nexa.tokenizer.tokenizer import NexaTokenizer
from nexa.tokenizer.vocab import UNK_TOKEN

log = logging.getLogger(__name__)

_RESERVOIR_SIZE = 20_000


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def measure_tokenizer(
    tokenizer: NexaTokenizer,
    pilot_index: dict[str, set[str]],
    training_time_s: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Stream through the full pilot corpus, encode every document, and
    compute comprehensive metrics.

    Parameters
    ----------
    tokenizer       : a trained NexaTokenizer
    pilot_index     : output of load_pilot_index()
    training_time_s : wall-clock seconds taken to train this tokenizer
    seed            : random seed for reservoir sampling

    Returns
    -------
    dict with all metrics
    """
    from scripts.tokenizer.corpus_sampler import stream_pilot_texts

    random.seed(seed)

    # Accumulators
    total_tokens     = 0
    total_words      = 0
    total_chars      = 0
    total_unk        = 0
    doc_count        = 0
    total_seq_len    = 0

    # Reservoir for sequence-length percentiles
    reservoir_n      = 0
    seq_len_reservoir: list[int] = []

    # Vocabulary coverage tracking (word types)
    word_type_sample: set[str] = set()
    word_type_in_vocab: set[str] = set()
    _MAX_WORD_TYPES = 100_000

    # Stream and encode every pilot document
    for text in stream_pilot_texts(pilot_index):
        doc_count += 1
        words = text.split()
        chars = len(text)

        # Encode
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        seq_len = len(ids)

        # Accumulate
        total_tokens  += seq_len
        total_words   += len(words)
        total_chars   += chars
        total_seq_len += seq_len

        # Count UNK tokens
        for tid in ids:
            if tokenizer.vocab.id_to_token(tid) == UNK_TOKEN:
                total_unk += 1

        # Reservoir sampling for seq len percentiles
        reservoir_n += 1
        if len(seq_len_reservoir) < _RESERVOIR_SIZE:
            seq_len_reservoir.append(seq_len)
        else:
            j = random.randint(0, reservoir_n - 1)
            if j < _RESERVOIR_SIZE:
                seq_len_reservoir[j] = seq_len

        # Word-type coverage (sample first 100k unique types)
        if len(word_type_sample) < _MAX_WORD_TYPES:
            for w in words:
                if len(word_type_sample) >= _MAX_WORD_TYPES:
                    break
                word_type_sample.add(w)

        if doc_count % 500 == 0:
            log.info("  Measured %d docs | tokens so far: %d", doc_count, total_tokens)

    # Coverage: check which sampled word types have all their BPE tokens in vocab
    for w in word_type_sample:
        word_tokens = tokenizer._tokenize_word(w)
        if all(t in tokenizer.vocab or t == UNK_TOKEN for t in word_tokens):
            if UNK_TOKEN not in word_tokens:
                word_type_in_vocab.add(w)
    coverage_pct = (len(word_type_in_vocab) / len(word_type_sample) * 100.0
                    if word_type_sample else 0.0)

    # Percentiles
    sorted_seqs = sorted(seq_len_reservoir)
    seq_pct = {
        "p50":  round(_percentile(sorted_seqs, 50)),
        "p90":  round(_percentile(sorted_seqs, 90)),
        "p95":  round(_percentile(sorted_seqs, 95)),
        "p99":  round(_percentile(sorted_seqs, 99)),
        "min":  min(seq_len_reservoir) if seq_len_reservoir else 0,
        "max":  max(seq_len_reservoir) if seq_len_reservoir else 0,
    }

    # Storage estimate: 2 bytes per token ID (uint16), 4 bytes per token (int32)
    est_storage_uint16 = total_tokens * 2
    est_storage_int32  = total_tokens * 4

    return {
        "vocab_size":               tokenizer.vocab_size,
        "num_merges":               len(tokenizer.merges),
        "training_time_s":          round(training_time_s, 1),
        "doc_count":                doc_count,
        "total_tokens":             total_tokens,
        "total_words":              total_words,
        "total_chars":              total_chars,
        "tokens_per_word":          round(total_tokens / total_words, 4) if total_words else 0,
        "tokens_per_char":          round(total_tokens / total_chars, 4) if total_chars else 0,
        "compression_ratio":        round(total_chars / total_tokens, 4) if total_tokens else 0,
        "avg_tokens_per_doc":       round(total_seq_len / doc_count, 1) if doc_count else 0,
        "seq_len_percentiles":      seq_pct,
        "unk_tokens":               total_unk,
        "unk_rate":                 round(total_unk / total_tokens, 6) if total_tokens else 0,
        "word_type_coverage_pct":   round(coverage_pct, 2),
        "est_storage_uint16_bytes": est_storage_uint16,
        "est_storage_int32_bytes":  est_storage_int32,
        "est_storage_mb_uint16":    round(est_storage_uint16 / 1e6, 1),
        "est_storage_mb_int32":     round(est_storage_int32  / 1e6, 1),
    }
