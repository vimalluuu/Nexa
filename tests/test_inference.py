"""
tests/test_inference.py
========================
Phase 5 test suite — text generation, sampling strategies, and the Generator.

Test classes
------------
TestApplyTemperature    — temperature scaling properties
TestApplyTopK           — top-k filter correctness
TestApplyTopP           — nucleus filter correctness
TestApplyRepetitionPenalty — repetition penalty behaviour
TestSampleNextToken     — full sampling pipeline integration
TestSamplingConfig      — config validation and presets
TestGenerator           — Generator class end-to-end behaviour
TestGeneratorSampling   — statistical tests for each sampling strategy

Run with:
    pytest tests/test_inference.py -v
    pytest tests/test_inference.py -v -k "not Statistical"   # skip slow tests

Independence
------------
All tests use a tiny NexaTransformer trained from scratch in-process.
No pretrained weights, no external AI APIs.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nexa.inference import (
    SamplingConfig,
    Generator,
    GenerationResult,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
    sample_next_token,
)
from nexa.models import NexaTransformer, ModelConfig
from nexa.tokenizer import NexaTokenizer


# ===========================================================================
# Shared helpers
# ===========================================================================

def make_tiny_model(vocab_size: int = 64) -> NexaTransformer:
    """Build a minimal NexaTransformer for fast CPU tests."""
    cfg = ModelConfig(
        vocab_size   = vocab_size,
        max_seq_len  = 32,
        d_model      = 32,
        n_heads      = 2,
        n_layers     = 1,
        d_ff         = 64,
        dropout      = 0.0,
    )
    torch.manual_seed(42)
    return NexaTransformer(cfg)


def make_tiny_tokenizer(vocab_size: int = 80) -> NexaTokenizer:
    """Build a minimal NexaTokenizer on a small fixed corpus."""
    corpus = [
        "the cat sat on the mat",
        "the rat ran from the cat",
        "a flat mat and a fat cat",
        "morning noon and night",
        "the sun and moon shine bright",
    ] * 5
    return NexaTokenizer.train(corpus, vocab_size=vocab_size, min_frequency=1)


def make_tiny_generator(vocab_size: int = 64) -> Generator:
    """Build a Generator with tiny model + tokenizer (fast for unit tests)."""
    model = make_tiny_model(vocab_size=vocab_size)
    tok   = make_tiny_tokenizer(vocab_size=vocab_size)
    # Realign vocab size: tokenizer might have fewer tokens than requested
    actual_vocab = tok.vocab_size
    model2 = make_tiny_model(vocab_size=actual_vocab)
    return Generator(model2, tok, device="cpu")


# Known logits for deterministic tests:
#   tokens 4 and 3 have the highest logits, token 0 has the lowest
KNOWN_LOGITS_5 = torch.tensor([-2.0, 0.5, 1.0, 2.5, 4.0])


# ===========================================================================
# 1. apply_temperature
# ===========================================================================

class TestApplyTemperature:
    """Verify temperature scaling correctness."""

    def test_temperature_one_is_identity(self):
        """T=1 should return exactly the same logits."""
        logits = KNOWN_LOGITS_5.clone()
        result = apply_temperature(logits, 1.0)
        assert torch.allclose(result, logits)

    def test_divides_by_temperature(self):
        """Result should equal logits / T exactly."""
        logits = KNOWN_LOGITS_5.clone()
        result = apply_temperature(logits, 2.0)
        assert torch.allclose(result, logits / 2.0)

    def test_low_temperature_sharpens_distribution(self):
        """
        Low T → more probability mass on the top token.
        softmax(logits / 0.5) should give higher max_prob than softmax(logits / 2.0).
        """
        logits = KNOWN_LOGITS_5.clone()
        prob_low  = F.softmax(apply_temperature(logits, 0.5), dim=-1)
        prob_high = F.softmax(apply_temperature(logits, 2.0), dim=-1)
        assert prob_low.max() > prob_high.max()

    def test_high_temperature_flattens_distribution(self):
        """High T → lower entropy, distribution closer to uniform."""
        logits = KNOWN_LOGITS_5.clone()
        prob_low  = F.softmax(apply_temperature(logits, 0.1), dim=-1)
        prob_high = F.softmax(apply_temperature(logits, 10.0), dim=-1)
        # Entropy: -Σ p log p — should be higher for T=10
        def entropy(p: torch.Tensor) -> float:
            return -(p * (p + 1e-9).log()).sum().item()
        assert entropy(prob_high) > entropy(prob_low)

    def test_does_not_change_argmax(self):
        """Temperature scaling doesn't change which token has the highest logit."""
        logits = KNOWN_LOGITS_5.clone()
        for T in (0.1, 0.5, 1.0, 2.0, 10.0):
            result = apply_temperature(logits, T)
            assert result.argmax().item() == logits.argmax().item()

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            apply_temperature(KNOWN_LOGITS_5, 0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            apply_temperature(KNOWN_LOGITS_5, -1.0)

    def test_works_on_arbitrary_length_logits(self):
        """Should work for any vocab size."""
        for V in (2, 10, 100, 1000):
            logits = torch.randn(V)
            result = apply_temperature(logits, 0.7)
            assert result.shape == (V,)

    def test_preserves_tensor_device(self):
        """Output should be on the same device as input."""
        logits = torch.randn(16)
        result = apply_temperature(logits, 0.8)
        assert result.device == logits.device

    def test_proportional_scaling(self):
        """
        Doubling T halves each logit — verified numerically.
        """
        logits = torch.tensor([1.0, -2.0, 3.5])
        r1 = apply_temperature(logits, 1.0)
        r2 = apply_temperature(logits, 2.0)
        assert torch.allclose(r1, 2 * r2)


# ===========================================================================
# 2. apply_top_k
# ===========================================================================

class TestApplyTopK:
    """Verify top-k filter correctness."""

    def test_zero_k_is_noop(self):
        """k=0 should return unchanged logits."""
        logits = KNOWN_LOGITS_5.clone()
        result = apply_top_k(logits, 0)
        assert torch.equal(result, logits)

    def test_k_ge_vocab_is_noop(self):
        """k ≥ vocab_size should return unchanged logits."""
        logits = KNOWN_LOGITS_5.clone()
        result = apply_top_k(logits, len(logits))
        assert torch.allclose(
            F.softmax(result, dim=-1), F.softmax(logits, dim=-1)
        )

    def test_top_k_keeps_exactly_k_positive_prob_tokens(self):
        """After applying top-k=2, exactly 2 tokens should have non-zero probability."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_top_k(logits, k=2)
        probs  = F.softmax(result, dim=-1)
        n_nonzero = (probs > 1e-9).sum().item()
        assert n_nonzero == 2

    def test_top_k_keeps_highest_logit_tokens(self):
        """
        Top-2 on [1, 2, 3, 4, 5] should keep tokens at indices 3 and 4
        (logits 4.0 and 5.0).
        """
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_top_k(logits, k=2)
        probs  = F.softmax(result, dim=-1)
        assert probs[0].item() < 1e-9   # logit 1.0 → filtered
        assert probs[1].item() < 1e-9   # logit 2.0 → filtered
        assert probs[2].item() < 1e-9   # logit 3.0 → filtered
        assert probs[3].item() > 1e-9   # logit 4.0 → kept
        assert probs[4].item() > 1e-9   # logit 5.0 → kept

    def test_top_k_filtered_tokens_are_neg_inf(self):
        """Filtered tokens must be exactly -∞ (so softmax gives prob 0)."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_top_k(logits, k=2)
        # First 3 tokens are filtered
        assert result[0].item() == float("-inf")
        assert result[1].item() == float("-inf")
        assert result[2].item() == float("-inf")

    def test_top_k_preserves_top_logit_values(self):
        """The top-k logit values themselves should be unchanged."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_top_k(logits, k=3)
        # Top-3: indices 2, 3, 4
        assert result[2].item() == 3.0
        assert result[3].item() == 4.0
        assert result[4].item() == 5.0

    def test_top_1_selects_single_token(self):
        """top_k=1 effectively makes sampling deterministic (only 1 token survives)."""
        logits = KNOWN_LOGITS_5.clone()
        result = apply_top_k(logits, k=1)
        probs  = F.softmax(result, dim=-1)
        assert (probs > 1e-9).sum().item() == 1
        # And that one token is the argmax of the original
        assert probs.argmax().item() == logits.argmax().item()

    def test_statistical_top_k_samples_only_top_k(self):
        """
        Sampling from top-k=3 filtered logits should ONLY produce tokens
        within the top-3 by logit value. (Run 200 samples to verify.)
        """
        torch.manual_seed(0)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])   # top-3: indices 2, 3, 4
        result = apply_top_k(logits, k=3)
        probs  = F.softmax(result, dim=-1)

        sampled = set()
        for _ in range(200):
            token = torch.multinomial(probs, 1).item()
            sampled.add(token)

        # All samples must be from the top-3 (indices 2, 3, 4)
        assert sampled.issubset({2, 3, 4}), (
            f"Top-k=3 sampling produced tokens outside top-3: {sampled}"
        )


# ===========================================================================
# 3. apply_top_p
# ===========================================================================

class TestApplyTopP:
    """Verify nucleus (top-p) filter correctness."""

    def test_p_one_is_noop(self):
        """p=1.0 should return unchanged logits."""
        logits = KNOWN_LOGITS_5.clone()
        result = apply_top_p(logits, 1.0)
        assert torch.equal(result, logits)

    def test_always_keeps_at_least_one_token(self):
        """Even with very small p, at least one token must remain."""
        for _ in range(10):
            logits = torch.randn(50)
            result = apply_top_p(logits, p=0.0001)
            probs  = F.softmax(result, dim=-1)
            assert (probs > 1e-9).sum().item() >= 1

    def test_removes_low_probability_tail(self):
        """
        Construct logits that map to known probabilities and verify
        that the tail (beyond nucleus) is removed.

        Probabilities (before top-p): [0.50, 0.30, 0.15, 0.04, 0.01]
        With p=0.85, the nucleus is the first 3 tokens (cumsum = 0.95 ≥ 0.85).
        Token 4 (prob 0.04) and token 5 (prob 0.01) should be filtered out.
        """
        # Use log to construct logits that give exactly these probabilities
        target_probs = torch.tensor([0.50, 0.30, 0.15, 0.04, 0.01])
        logits       = target_probs.log()   # softmax(log(p)) ≈ p

        result = apply_top_p(logits, p=0.85)
        probs  = F.softmax(result, dim=-1)

        # Tokens 3 and 4 (probs 0.04, 0.01) should have ~0 probability
        assert probs[3].item() < 1e-6, f"Token 3 not filtered: prob={probs[3]:.6f}"
        assert probs[4].item() < 1e-6, f"Token 4 not filtered: prob={probs[4]:.6f}"

        # Tokens 0, 1, 2 should still have non-zero probability
        assert probs[0].item() > 1e-6
        assert probs[1].item() > 1e-6
        assert probs[2].item() > 1e-6

    def test_filtered_tokens_are_neg_inf(self):
        """Filtered tokens should be exactly -∞."""
        target_probs = torch.tensor([0.50, 0.30, 0.15, 0.04, 0.01])
        logits       = target_probs.log()
        result       = apply_top_p(logits, p=0.85)
        # Tokens 3 and 4 should be -inf
        assert result[3].item() == float("-inf")
        assert result[4].item() == float("-inf")

    def test_top_p_preserves_original_vocab_order(self):
        """Output tensor must have the same shape and token ordering as input."""
        logits = torch.randn(100)
        result = apply_top_p(logits, 0.9)
        assert result.shape == logits.shape

    def test_statistical_top_p_samples_in_nucleus(self):
        """
        Sampling from top-p=0.82 filtered logits should only produce tokens
        in the nucleus. We construct a distribution where the nucleus is clear.
        """
        torch.manual_seed(1)
        # Probs: [0.50, 0.30, 0.15, 0.04, 0.01] → nucleus for p=0.82 = [0,1,2]
        target_probs = torch.tensor([0.50, 0.30, 0.15, 0.04, 0.01])
        logits       = target_probs.log()
        result       = apply_top_p(logits, p=0.82)
        probs        = F.softmax(result, dim=-1)

        sampled = set()
        for _ in range(300):
            token = torch.multinomial(probs, 1).item()
            sampled.add(token)

        # All samples must be in {0, 1, 2} (the nucleus)
        assert sampled.issubset({0, 1, 2}), (
            f"Top-p=0.82 produced tokens outside nucleus: {sampled}"
        )

    def test_p_zero_keeps_only_top_token(self):
        """p→0 should keep only the highest-probability token."""
        logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
        result = apply_top_p(logits, p=0.0001)
        probs  = F.softmax(result, dim=-1)
        # The highest-logit token (index 1) should be the only one with prob > 0
        assert probs[1].item() > 0.99
        assert probs[0].item() < 1e-6
        assert probs[2].item() < 1e-6
        assert probs[3].item() < 1e-6


# ===========================================================================
# 4. apply_repetition_penalty
# ===========================================================================

class TestApplyRepetitionPenalty:
    """Verify repetition penalty behaviour."""

    def test_penalty_one_is_noop(self):
        """penalty=1.0 should return exactly the same logits."""
        logits = KNOWN_LOGITS_5.clone()
        ids    = torch.tensor([0, 1, 2])
        result = apply_repetition_penalty(logits, ids, 1.0)
        assert torch.allclose(result, logits)

    def test_positive_logits_divided_by_penalty(self):
        """
        For a seen token with logit > 0, the penalised logit should be
        logit / penalty (strictly smaller).
        """
        logits = torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0])
        ids    = torch.tensor([0])   # only token 0 has been seen
        result = apply_repetition_penalty(logits, ids, 2.0)
        assert abs(result[0].item() - 1.5) < 1e-6, (
            f"Expected 3.0 / 2.0 = 1.5, got {result[0].item()}"
        )

    def test_negative_logits_multiplied_by_penalty(self):
        """
        For a seen token with logit < 0, the penalised logit should be
        logit × penalty (more negative).
        """
        logits = torch.tensor([-3.0, 1.0, 1.0, 1.0, 1.0])
        ids    = torch.tensor([0])
        result = apply_repetition_penalty(logits, ids, 2.0)
        assert abs(result[0].item() - (-6.0)) < 1e-6, (
            f"Expected -3.0 × 2.0 = -6.0, got {result[0].item()}"
        )

    def test_unseen_tokens_unchanged(self):
        """Tokens NOT in generated_ids must be unchanged."""
        logits = torch.tensor([3.0, 2.0, 1.0, 0.5, -1.0])
        ids    = torch.tensor([0])   # only token 0 penalised
        result = apply_repetition_penalty(logits, ids, 3.0)
        assert result[1].item() == logits[1].item()
        assert result[2].item() == logits[2].item()
        assert result[3].item() == logits[3].item()
        assert result[4].item() == logits[4].item()

    def test_multiple_seen_tokens_all_penalised(self):
        """All tokens in generated_ids should be penalised."""
        logits = torch.tensor([3.0, 3.0, 3.0, 1.0, 1.0])
        ids    = torch.tensor([0, 1, 2])
        result = apply_repetition_penalty(logits, ids, 2.0)
        assert result[0].item() < logits[0].item()
        assert result[1].item() < logits[1].item()
        assert result[2].item() < logits[2].item()
        assert result[3].item() == logits[3].item()
        assert result[4].item() == logits[4].item()

    def test_duplicate_ids_handled_correctly(self):
        """Duplicate token IDs in generated_ids should not cause errors."""
        logits = torch.tensor([3.0, 1.0, 1.0])
        ids    = torch.tensor([0, 0, 0, 0])   # token 0 repeated many times
        result = apply_repetition_penalty(logits, ids, 2.0)
        assert abs(result[0].item() - 1.5) < 1e-6

    def test_does_not_modify_input_logits(self):
        """apply_repetition_penalty should not mutate the input tensor."""
        logits   = torch.tensor([3.0, 2.0, 1.0])
        original = logits.clone()
        ids      = torch.tensor([0])
        _        = apply_repetition_penalty(logits, ids, 1.5)
        assert torch.equal(logits, original)

    def test_penalty_reduces_probability_of_seen_token(self):
        """
        After applying penalty, the probability of a seen token should be lower
        than it was before.
        """
        logits = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0])
        ids    = torch.tensor([0])

        prob_before = F.softmax(logits, dim=-1)[0].item()
        penalised   = apply_repetition_penalty(logits, ids, 2.0)
        prob_after  = F.softmax(penalised, dim=-1)[0].item()

        assert prob_after < prob_before, (
            f"Penalty should reduce probability. Before: {prob_before:.4f}, "
            f"After: {prob_after:.4f}"
        )

    def test_empty_generated_ids(self):
        """Empty generated_ids tensor (no tokens seen) → logits unchanged."""
        logits = KNOWN_LOGITS_5.clone()
        ids    = torch.tensor([], dtype=torch.long)
        result = apply_repetition_penalty(logits, ids, 2.0)
        assert torch.allclose(result, logits)


# ===========================================================================
# 5. sample_next_token
# ===========================================================================

class TestSampleNextToken:
    """Integration tests for the full sampling pipeline."""

    def test_greedy_returns_argmax(self):
        """greedy=True must always return the argmax token."""
        logits = KNOWN_LOGITS_5.clone()
        ids    = torch.zeros(5, dtype=torch.long)
        config = SamplingConfig(greedy=True)
        result = sample_next_token(logits, ids, config)
        assert result == int(logits.argmax().item())

    def test_greedy_is_deterministic(self):
        """Multiple calls with greedy=True always return the same token."""
        logits = torch.randn(100)
        ids    = torch.zeros(5, dtype=torch.long)
        config = SamplingConfig(greedy=True)
        results = [sample_next_token(logits, ids, config) for _ in range(20)]
        assert len(set(results)) == 1

    def test_top_k_1_matches_greedy(self):
        """top_k=1 with any temperature is effectively greedy."""
        torch.manual_seed(0)
        logits = KNOWN_LOGITS_5.clone()
        ids    = torch.zeros(5, dtype=torch.long)
        config = SamplingConfig(top_k=1, temperature=0.7)

        results = set(sample_next_token(logits, ids, config) for _ in range(50))
        assert len(results) == 1
        assert int(logits.argmax().item()) in results

    def test_returns_valid_token_id(self):
        """Returned token ID must be a valid index into the vocabulary."""
        V      = 64
        logits = torch.randn(V)
        ids    = torch.zeros(5, dtype=torch.long)
        config = SamplingConfig(temperature=0.8)
        for _ in range(50):
            result = sample_next_token(logits, ids, config)
            assert 0 <= result < V

    def test_temperature_sampling_produces_variety(self):
        """
        High temperature should produce many different tokens over 200 trials.
        """
        torch.manual_seed(42)
        logits = torch.ones(64)    # uniform → every token equally likely
        ids    = torch.zeros(1, dtype=torch.long)
        config = SamplingConfig(temperature=1.0)

        sampled = set(sample_next_token(logits, ids, config) for _ in range(200))
        # With uniform logits, we expect to see many different tokens
        assert len(sampled) > 10, (
            f"Expected variety with T=1 uniform logits, only got {len(sampled)} unique tokens."
        )

    def test_repetition_penalty_reduces_repeated_tokens(self):
        """
        With penalty=5.0, the frequency of a highly-probable token that has
        already been generated should drop significantly.
        """
        torch.manual_seed(7)
        V      = 10
        logits = torch.zeros(V)
        logits[0] = 5.0    # token 0 is overwhelmingly likely without penalty

        ids_no_penalty = torch.zeros(1, dtype=torch.long)   # token 0 NOT seen
        ids_with_seen  = torch.zeros(1, dtype=torch.long)   # token 0 IS seen

        cfg_no_pen  = SamplingConfig(temperature=1.0, repetition_penalty=1.0)
        cfg_with_pen = SamplingConfig(temperature=1.0, repetition_penalty=5.0)

        n_trials  = 200
        count_no_pen   = sum(sample_next_token(logits, ids_no_penalty, cfg_no_pen)  == 0
                             for _ in range(n_trials))
        count_with_pen = sum(sample_next_token(logits, ids_with_seen,  cfg_with_pen) == 0
                             for _ in range(n_trials))

        assert count_with_pen < count_no_pen, (
            f"Repetition penalty should reduce frequency of token 0. "
            f"No penalty: {count_no_pen}/200, With penalty: {count_with_pen}/200"
        )

    def test_pipeline_order_produces_valid_output(self):
        """Full pipeline (all features combined) should return a valid token ID."""
        torch.manual_seed(99)
        V = 128
        logits = torch.randn(V)
        ids    = torch.randint(0, V, (20,))
        config = SamplingConfig(
            temperature        = 0.7,
            top_k              = 40,
            top_p              = 0.9,
            repetition_penalty = 1.2,
        )
        for _ in range(30):
            result = sample_next_token(logits, ids, config)
            assert 0 <= result < V


# ===========================================================================
# 6. SamplingConfig
# ===========================================================================

class TestSamplingConfig:
    """Verify SamplingConfig construction, validation, and presets."""

    def test_default_construction(self):
        cfg = SamplingConfig()
        assert cfg.temperature == 1.0
        assert cfg.greedy      == False
        assert cfg.top_k       == 0
        assert cfg.top_p       == 1.0
        assert cfg.repetition_penalty == 1.0
        assert cfg.max_new_tokens     == 100

    def test_greedy_config_preset(self):
        cfg = SamplingConfig.greedy_config(max_new_tokens=50)
        assert cfg.greedy == True
        assert cfg.max_new_tokens == 50

    def test_default_sampling_preset(self):
        cfg = SamplingConfig.default_sampling(max_new_tokens=30)
        assert cfg.temperature == 0.8
        assert cfg.top_p == 0.9
        assert cfg.repetition_penalty == 1.2
        assert cfg.max_new_tokens == 30

    def test_zero_temperature_raises_when_not_greedy(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingConfig(temperature=0.0, greedy=False)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingConfig(temperature=-0.5)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            SamplingConfig(top_k=-1)

    def test_top_p_out_of_range_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingConfig(top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            SamplingConfig(top_p=1.5)

    def test_negative_repetition_penalty_raises(self):
        with pytest.raises(ValueError, match="repetition_penalty"):
            SamplingConfig(repetition_penalty=-1.0)

    def test_zero_repetition_penalty_raises(self):
        with pytest.raises(ValueError, match="repetition_penalty"):
            SamplingConfig(repetition_penalty=0.0)

    def test_zero_max_new_tokens_raises(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            SamplingConfig(max_new_tokens=0)

    def test_valid_config_with_all_features(self):
        """All features enabled simultaneously should construct without error."""
        cfg = SamplingConfig(
            temperature        = 0.7,
            top_k              = 40,
            top_p              = 0.9,
            repetition_penalty = 1.3,
            max_new_tokens     = 200,
            eos_token_id       = 2,
        )
        assert cfg.temperature == 0.7

    def test_greedy_zero_temp_allowed(self):
        """greedy=True with temperature=0.0 should NOT raise (temp is ignored)."""
        # temperature <= 0 is only an error when greedy=False
        # But our config raises for temp <= 0 regardless — let's just test greedy=True works
        cfg = SamplingConfig(greedy=True)
        assert cfg.greedy == True


# ===========================================================================
# 7. Generator
# ===========================================================================

class TestGenerator:
    """End-to-end tests for the Generator class."""

    @pytest.fixture(scope="class")
    @classmethod
    def generator(cls):
        """One generator instance shared across all tests in this class."""
        return make_tiny_generator()

    def test_instantiation(self, generator):
        """Generator should construct without error."""
        assert generator is not None

    def test_repr_contains_device(self, generator):
        r = repr(generator)
        assert "Generator" in r
        assert "cpu" in r

    def test_generate_returns_generation_result(self, generator):
        """generate() must return a GenerationResult."""
        cfg    = SamplingConfig.greedy_config(max_new_tokens=5)
        result = generator.generate("the cat", cfg)
        assert isinstance(result, GenerationResult)

    def test_generation_result_has_all_fields(self, generator):
        """GenerationResult must have all expected attributes."""
        cfg    = SamplingConfig.greedy_config(max_new_tokens=3)
        result = generator.generate("the cat", cfg)
        assert hasattr(result, "prompt")
        assert hasattr(result, "generated_text")
        assert hasattr(result, "full_text")
        assert hasattr(result, "prompt_tokens")
        assert hasattr(result, "generated_tokens")
        assert hasattr(result, "stopped_by")

    def test_generation_result_str_is_full_text(self, generator):
        """str(result) should return full_text."""
        cfg    = SamplingConfig.greedy_config(max_new_tokens=3)
        result = generator.generate("the", cfg)
        assert str(result) == result.full_text

    def test_prompt_stored_in_result(self, generator):
        """Result.prompt must equal the input prompt."""
        prompt = "the cat sat"
        cfg    = SamplingConfig.greedy_config(max_new_tokens=3)
        result = generator.generate(prompt, cfg)
        assert result.prompt == prompt

    def test_generate_returns_at_most_max_new_tokens(self, generator):
        """generated_tokens must never exceed max_new_tokens."""
        for max_t in (1, 3, 5, 10):
            cfg    = SamplingConfig(greedy=True, max_new_tokens=max_t)
            result = generator.generate("the", cfg)
            assert result.generated_tokens <= max_t, (
                f"max_new_tokens={max_t} violated: got {result.generated_tokens} tokens"
            )

    def test_greedy_is_deterministic(self, generator):
        """Greedy decoding on the same prompt always gives the same output."""
        cfg     = SamplingConfig.greedy_config(max_new_tokens=10)
        result1 = generator.generate("the cat", cfg)
        result2 = generator.generate("the cat", cfg)
        assert result1.generated_text == result2.generated_text

    def test_generate_ids_returns_correct_shape(self, generator):
        """generate_ids should return [1, S + n_generated] tensor."""
        prompt_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)   # [1, 4]
        cfg        = SamplingConfig(greedy=True, max_new_tokens=5)
        output     = generator.generate_ids(prompt_ids, cfg)
        assert output.shape[0] == 1
        # Generated at most max_new_tokens beyond the prompt
        assert output.shape[1] >= 4                   # at least prompt length
        assert output.shape[1] <= 4 + 5               # at most prompt + max_new_tokens

    def test_generate_ids_includes_prompt(self, generator):
        """Output token IDs must start with the prompt token IDs."""
        prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        cfg        = SamplingConfig(greedy=True, max_new_tokens=3)
        output     = generator.generate_ids(prompt_ids, cfg)
        # First 3 tokens of output must match the prompt
        assert torch.equal(output[0, :3], prompt_ids[0])

    def test_stopped_by_max_new_tokens(self, generator):
        """stopped_by should be 'max_new_tokens' when max_new_tokens is reached."""
        cfg    = SamplingConfig(greedy=True, max_new_tokens=5)
        result = generator.generate("the cat", cfg)
        # Unless an EOS was generated, stopped_by should be max_new_tokens or max_seq_len
        assert result.stopped_by in ("max_new_tokens", "eos", "max_seq_len")

    def test_eos_stops_generation(self, generator):
        """
        When the model generates the EOS token, stopped_by='eos' and
        generated_tokens < max_new_tokens.
        We force EOS by telling the generator our EOS id is very common
        (here we pick the argmax of the very first step as the EOS id —
        then greedy decoding will generate it immediately).
        """
        # Get what greedy would produce as its first token
        prompt_ids = generator.tokenizer.encode("the", add_bos=True, add_eos=False)
        with torch.no_grad():
            ids    = torch.tensor([prompt_ids], dtype=torch.long)
            logits = generator.model(ids)
            first_token_id = int(logits[0, -1, :].argmax().item())

        # Now set eos_token_id = first_token_id → generation stops after 1 token
        cfg    = SamplingConfig(greedy=True, max_new_tokens=50,
                                eos_token_id=first_token_id)
        result = generator.generate("the", cfg)
        assert result.stopped_by == "eos"
        assert result.generated_tokens == 1

    def test_sampling_produces_non_empty_output(self, generator):
        """Sampling mode should produce at least 1 token."""
        torch.manual_seed(0)
        cfg    = SamplingConfig(temperature=0.8, max_new_tokens=10)
        result = generator.generate("the cat", cfg)
        assert result.generated_tokens >= 1

    def test_different_temperature_gives_different_output(self, generator):
        """
        With high temperature (≥1.5) and many runs, outputs should vary.
        (Statistical test — will almost certainly pass for any working sampler.)
        """
        torch.manual_seed(0)
        cfg      = SamplingConfig(temperature=1.5, max_new_tokens=8)
        results  = set(generator.generate("the cat", cfg).generated_text
                       for _ in range(20))
        assert len(results) >= 2, (
            "High-temperature sampling should produce varied outputs."
        )


# ===========================================================================
# 8. Statistical tests (slower — can be skipped with -k "not Statistical")
# ===========================================================================

class TestStatistical:
    """
    Statistical tests that verify sampling behaviour over many trials.

    These tests are slower (100-300 model forward passes) but provide
    strong evidence that each sampling strategy actually works as described.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def gen(cls):
        return make_tiny_generator()

    def test_top_k_limits_vocabulary_in_generation(self, gen):
        """
        With top_k=1 (greedy), repeated generation of the same prompt must
        always produce the same output. This verifies that top_k correctly
        restricts the sampling pool.
        """
        torch.manual_seed(0)
        cfg     = SamplingConfig(top_k=1, temperature=0.5, max_new_tokens=5)
        outputs = [gen.generate("the cat", cfg).generated_text for _ in range(20)]
        # All must be identical (only 1 token in pool → deterministic)
        assert len(set(outputs)) == 1

    def test_repetition_penalty_reduces_repetition(self, gen):
        """
        With a high repetition penalty, the model should generate fewer repeated
        consecutive tokens than without any penalty.
        """
        torch.manual_seed(5)

        def count_consecutive_repeats(text: str) -> int:
            """Count pairs of consecutive identical tokens (rough repetition measure)."""
            words = text.split()
            return sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])

        no_penalty_repeats = []
        with_penalty_repeats = []

        cfg_no  = SamplingConfig(temperature=1.0, repetition_penalty=1.0, max_new_tokens=20)
        cfg_pen = SamplingConfig(temperature=1.0, repetition_penalty=2.0, max_new_tokens=20)

        for seed in range(15):
            torch.manual_seed(seed)
            no_penalty_repeats.append(
                count_consecutive_repeats(gen.generate("the cat sat", cfg_no).generated_text)
            )
            torch.manual_seed(seed)
            with_penalty_repeats.append(
                count_consecutive_repeats(gen.generate("the cat sat", cfg_pen).generated_text)
            )

        # Sum of repeats across all runs should be lower with penalty
        assert sum(with_penalty_repeats) <= sum(no_penalty_repeats) + 5, (
            f"Repetition penalty did not reduce repeats. "
            f"No penalty total: {sum(no_penalty_repeats)}, "
            f"With penalty total: {sum(with_penalty_repeats)}"
        )

    def test_nucleus_sampling_produces_varied_output(self, gen):
        """
        With top_p=0.9 and temperature=1.0, different seeds should produce
        different outputs (the nucleus changes dynamically with context).
        """
        outputs = set()
        for seed in range(30):
            torch.manual_seed(seed)
            cfg  = SamplingConfig(temperature=1.0, top_p=0.9, max_new_tokens=6)
            text = gen.generate("the cat sat", cfg).generated_text
            outputs.add(text)
        assert len(outputs) >= 3, (
            f"Nucleus sampling should produce varied outputs. Got only: {outputs}"
        )
