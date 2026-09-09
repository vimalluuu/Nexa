"""
Nexa Phase 8.5 - Vocabulary Size Recommendation
=================================================
Analyzes metrics across all candidate tokenizers and recommends ONE
vocabulary size for Nexa V1 pilot training.

Decision criteria (in priority order for a CPU-only small model):
  1. Context-length efficiency (tokens/doc): lower = can fit more text in context
  2. Embedding + LM-head parameter cost: vocab_size * 2 * d_model params
  3. Training efficiency: smaller vocab = faster softmax on CPU
  4. Token compression: chars/token >= ~3.5 needed for reasonable efficiency
  5. UNK rate: must be < 0.5% (basically no unknown tokens for English)
  6. Word type coverage: > 85% preferred

We DO NOT automatically select the largest vocabulary — larger is not
always better, especially for CPU-only training with a small model.

Returns a recommendation dict with rationale.
"""

from __future__ import annotations


# Typical Nexa pilot model size (from model_config.yaml)
# Used to estimate embedding parameter counts
ASSUMED_D_MODEL = 128  # conservative estimate for current small model
ASSUMED_D_MODEL_V1 = 256  # potential V1 upgrade


def compute_embedding_params(vocab_size: int, d_model: int) -> int:
    """Embedding table + LM head combined parameters."""
    return vocab_size * d_model * 2   # input embed + output LM head


def recommend_vocab_size(
    candidates: list[dict],
    d_model: int = ASSUMED_D_MODEL,
) -> dict:
    """
    Analyze candidate tokenizer metrics and return a recommendation.

    Parameters
    ----------
    candidates : list of metric dicts (output of measure_tokenizer)
                 each dict must have 'vocab_size', 'tokens_per_word',
                 'compression_ratio', 'unk_rate', 'word_type_coverage_pct',
                 'seq_len_percentiles'
    d_model    : assumed embedding dimension for parameter counting

    Returns
    -------
    dict with:
        recommended_vocab_size   : int
        rationale                : str
        candidate_summary        : list of per-candidate analysis dicts
        embedding_params         : dict of vocab_size -> param count
    """
    summary = []
    for m in candidates:
        vs          = m["vocab_size"]
        tpw         = m["tokens_per_word"]
        cr          = m["compression_ratio"]
        unk_rate    = m["unk_rate"]
        coverage    = m["word_type_coverage_pct"]
        p90_seq     = m["seq_len_percentiles"].get("p90", 0)
        embed_params = compute_embedding_params(vs, d_model)

        # Score: lower is better (we want efficient, small, low-unk)
        # Composite score weights:
        #   - tokens per word (compression): lower = better, weight 3
        #   - unk rate (penalty for high OOV): weight 5 (critical)
        #   - embed params relative to total model: weight 2
        #   - coverage bonus: higher = better, weight 2
        unk_penalty      = unk_rate * 5_000_000        # unk in [0,1], scale up
        compression_score = tpw * 3                    # lower tpw → better
        coverage_bonus   = (100 - coverage) * 0.02    # penalty for low coverage
        param_penalty    = embed_params / 10_000_000  # penalty for large embed

        score = compression_score + unk_penalty + coverage_bonus + param_penalty

        summary.append({
            "vocab_size":             vs,
            "tokens_per_word":        tpw,
            "compression_ratio_cpc":  cr,
            "unk_rate":               unk_rate,
            "word_type_coverage_pct": coverage,
            "p90_seq_len":            p90_seq,
            "embedding_params":       embed_params,
            "composite_score":        round(score, 4),
            "viable":                 unk_rate < 0.005 and coverage > 70.0,
        })

    # Sort by composite score ascending (lower = better)
    summary_sorted = sorted(summary, key=lambda x: x["composite_score"])

    # Among viable candidates, pick the lowest score
    viable = [s for s in summary_sorted if s["viable"]]
    if viable:
        recommended = viable[0]["vocab_size"]
        rationale_base = (
            f"Vocabulary size {recommended} achieves the best balance of "
            f"token compression, low UNK rate, word-type coverage, and "
            f"low embedding parameter count for CPU-only training."
        )
    else:
        # Fall back to lowest UNK rate
        recommended = min(candidates, key=lambda m: m["unk_rate"])["vocab_size"]
        rationale_base = (
            f"No candidate met all viability thresholds. Selected "
            f"vocab_size={recommended} as having the lowest UNK rate."
        )

    # Detailed rationale
    chosen = next(s for s in summary if s["vocab_size"] == recommended)
    embed_params_all = {
        s["vocab_size"]: compute_embedding_params(s["vocab_size"], d_model)
        for s in summary
    }

    rationale = (
        f"{rationale_base} "
        f"At vocab_size={recommended}: "
        f"tokens/word={chosen['tokens_per_word']:.3f}, "
        f"chars/token={chosen['compression_ratio_cpc']:.2f}, "
        f"UNK rate={chosen['unk_rate']*100:.3f}%, "
        f"coverage={chosen['word_type_coverage_pct']:.1f}%, "
        f"embedding params={chosen['embedding_params']:,} "
        f"(d_model={d_model}). "
        f"Larger vocabularies improve compression but add "
        f"significant embedding parameter cost for this small model."
    )

    return {
        "recommended_vocab_size":  recommended,
        "rationale":               rationale,
        "d_model_assumed":         d_model,
        "candidate_summary":       summary,
        "embedding_params_by_size": embed_params_all,
        "viability_thresholds": {
            "max_unk_rate":       0.005,
            "min_coverage_pct":   70.0,
        },
    }
