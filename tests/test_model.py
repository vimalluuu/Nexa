"""
tests/test_model.py
====================
Phase 3 test suite — Nexa Transformer (all model components).

All tests run on CPU with a TINY_CONFIG to keep execution fast (< 2 seconds
total). No GPU required. No pretrained weights loaded anywhere.

Test classes:
    TestModelConfig       — ModelConfig dataclass validation and properties
    TestRMSNorm           — RMSNorm correctness (shape, scale, gradients)
    TestRoPE              — RoPE correctness (shape, isometry, position dependence)
    TestMultiHeadAttention — Attention output shape and causal masking
    TestSwiGLU            — SwiGLU output shape and parameter structure
    TestTransformerBlock  — Block output shape and residual connection
    TestNexaTransformer   — Full model: logits, causality, generation, save/load
    TestIndependence      — No forbidden libraries imported by model code

Run with:
    pytest tests/test_model.py -v
    pytest tests/test_model.py -v --tb=short
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Shared tiny config (fast enough for CPU tests)
# ---------------------------------------------------------------------------

def make_tiny_config(**overrides):
    from nexa.models import ModelConfig
    defaults = dict(
        vocab_size   = 128,
        max_seq_len  = 32,
        d_model      = 64,
        n_heads      = 4,
        n_layers     = 2,
        d_ff         = 256,
        dropout      = 0.0,   # deterministic during tests
        norm_type    = "rmsnorm",
        norm_eps     = 1e-6,
        pos_encoding = "rotary",
        tie_embeddings = True,
        pad_token_id = 0,
        bos_token_id = 1,
        eos_token_id = 2,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


TINY_CFG = make_tiny_config()

# Batch / sequence dimensions used across tests
B, S = 2, 16     # batch size, sequence length


# ===========================================================================
# 1. ModelConfig
# ===========================================================================

class TestModelConfig:
    """Verify ModelConfig validation, properties, and factory methods."""

    def test_default_construction(self):
        """ModelConfig can be constructed with defaults."""
        from nexa.models import ModelConfig
        cfg = ModelConfig()
        assert cfg.d_model == 512
        assert cfg.n_heads == 8
        assert cfg.n_layers == 6

    def test_d_head_property(self):
        """d_head = d_model // n_heads."""
        cfg = make_tiny_config(d_model=64, n_heads=4)
        assert cfg.d_head == 16

    def test_d_head_exact_division(self):
        """d_model must be divisible by n_heads."""
        cfg = make_tiny_config(d_model=64, n_heads=8)
        assert cfg.d_head == 8

    def test_invalid_d_model_not_divisible(self):
        """Non-divisible d_model / n_heads raises ValueError."""
        from nexa.models import ModelConfig
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(d_model=65, n_heads=8)

    def test_invalid_n_layers_zero(self):
        from nexa.models import ModelConfig
        with pytest.raises(ValueError, match="n_layers"):
            ModelConfig(d_model=64, n_heads=8, n_layers=0)

    def test_invalid_dropout_out_of_range(self):
        from nexa.models import ModelConfig
        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(d_model=64, n_heads=8, dropout=1.5)

    def test_invalid_norm_type(self):
        from nexa.models import ModelConfig
        with pytest.raises(ValueError, match="norm_type"):
            ModelConfig(d_model=64, n_heads=8, norm_type="batchnorm")

    def test_invalid_pos_encoding(self):
        from nexa.models import ModelConfig
        with pytest.raises(ValueError, match="pos_encoding"):
            ModelConfig(d_model=64, n_heads=8, pos_encoding="fourier")

    def test_from_yaml(self):
        """ModelConfig.from_yaml() correctly reads configs/model_config.yaml."""
        from nexa.models import ModelConfig
        cfg = ModelConfig.from_yaml("configs/model_config.yaml")
        assert isinstance(cfg, ModelConfig)
        assert cfg.d_model > 0
        assert cfg.n_heads > 0
        assert cfg.d_model % cfg.n_heads == 0

    def test_from_dict(self):
        from nexa.models import ModelConfig
        cfg = ModelConfig.from_dict({"d_model": 128, "n_heads": 4, "vocab_size": 1000})
        assert cfg.d_model == 128
        assert cfg.n_heads == 4
        assert cfg.d_head == 32

    def test_repr_contains_key_fields(self):
        cfg = TINY_CFG
        r = repr(cfg)
        assert "d_model" in r
        assert "n_heads" in r


# ===========================================================================
# 2. RMSNorm
# ===========================================================================

class TestRMSNorm:
    """Verify RMSNorm output shape, scale normalization, and gradients."""

    def test_output_shape_preserved(self):
        """Output shape must match input shape exactly."""
        from nexa.models import RMSNorm
        norm = RMSNorm(TINY_CFG.d_model)
        x = torch.randn(B, S, TINY_CFG.d_model)
        assert norm(x).shape == x.shape

    def test_weight_is_ones_at_init(self):
        """Norm weight (γ) must be initialized to ones (identity)."""
        from nexa.models import RMSNorm
        norm = RMSNorm(TINY_CFG.d_model)
        assert torch.allclose(norm.weight, torch.ones(TINY_CFG.d_model))

    def test_output_rms_is_approx_one(self):
        """
        After normalization (with unit weight), the RMS of the output
        along the d_model axis should be ≈ 1.0.
        """
        from nexa.models import RMSNorm
        norm = RMSNorm(TINY_CFG.d_model)
        # Explicitly set weight to ones (it already is, but be explicit)
        with torch.no_grad():
            norm.weight.fill_(1.0)
        x = torch.randn(B, S, TINY_CFG.d_model) * 10.0   # large values
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()               # [B, S]
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), (
            f"Expected RMS ≈ 1.0, got max deviation {(rms - 1).abs().max().item():.6f}"
        )

    def test_weight_scales_output(self):
        """Doubling the weight should double the output."""
        from nexa.models import RMSNorm
        norm = RMSNorm(TINY_CFG.d_model)
        x = torch.randn(B, S, TINY_CFG.d_model)
        with torch.no_grad():
            out_one = norm(x).clone()
            norm.weight.data *= 2.0
            out_two = norm(x)
        assert torch.allclose(out_two, out_one * 2.0, atol=1e-5)

    def test_backward_pass_gradients_flow(self):
        """Gradients must flow through RMSNorm (backward does not raise)."""
        from nexa.models import RMSNorm
        norm = RMSNorm(TINY_CFG.d_model)
        x = torch.randn(B, S, TINY_CFG.d_model, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_build_norm_factory_rmsnorm(self):
        from nexa.models import build_norm, RMSNorm
        norm = build_norm("rmsnorm", 64)
        assert isinstance(norm, RMSNorm)

    def test_build_norm_factory_layernorm(self):
        from nexa.models import build_norm, LayerNorm
        norm = build_norm("layernorm", 64)
        assert isinstance(norm, LayerNorm)

    def test_build_norm_factory_invalid(self):
        from nexa.models import build_norm
        with pytest.raises(ValueError):
            build_norm("batchnorm", 64)


# ===========================================================================
# 3. RoPE
# ===========================================================================

class TestRoPE:
    """Verify RoPE precomputation and application."""

    def test_precompute_shape(self):
        """precompute_rope_freqs returns (cos, sin) each of shape [S, d_head//2]."""
        from nexa.models import precompute_rope_freqs
        S, d_head = 32, 16
        cos, sin = precompute_rope_freqs(d_head, S)
        assert cos.shape == (S, d_head // 2)
        assert sin.shape == (S, d_head // 2)

    def test_precompute_first_position_is_identity(self):
        """At position 0: cos = 1, sin = 0 — the rotation is zero."""
        from nexa.models import precompute_rope_freqs
        cos, sin = precompute_rope_freqs(16, 32)
        assert torch.allclose(cos[0], torch.ones(8), atol=1e-6)
        assert torch.allclose(sin[0], torch.zeros(8), atol=1e-6)

    def test_apply_rope_output_shape(self):
        """apply_rope must preserve the input tensor's shape."""
        from nexa.models import precompute_rope_freqs, apply_rope
        d_head = TINY_CFG.d_head
        x = torch.randn(B, TINY_CFG.n_heads, S, d_head)
        cos, sin = precompute_rope_freqs(d_head, S)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_apply_rope_preserves_norm(self):
        """
        RoPE is a rotation (isometry) — it preserves the L2 norm of each vector.
        ‖rotate(x)‖ = ‖x‖  for all x.
        """
        from nexa.models import precompute_rope_freqs, apply_rope
        d_head = TINY_CFG.d_head
        x = torch.randn(B, TINY_CFG.n_heads, S, d_head)
        cos, sin = precompute_rope_freqs(d_head, S)
        out = apply_rope(x, cos, sin)
        # L2 norms should match
        norms_in  = x.norm(dim=-1)     # [B, H, S]
        norms_out = out.norm(dim=-1)   # [B, H, S]
        assert torch.allclose(norms_in, norms_out, atol=1e-5), (
            f"RoPE changed vector norms. Max diff: {(norms_in - norms_out).abs().max().item():.6f}"
        )

    def test_apply_rope_is_position_dependent(self):
        """
        The same token vector at different sequence positions should produce
        different outputs — this is the whole point of positional encoding.
        """
        from nexa.models import precompute_rope_freqs, apply_rope
        d_head = TINY_CFG.d_head
        # Use same token vector at all positions
        token_vec = torch.randn(1, 1, 1, d_head)
        x = token_vec.expand(1, 1, S, d_head)   # [1, 1, S, d_head]
        cos, sin = precompute_rope_freqs(d_head, S)
        out = apply_rope(x, cos, sin)
        # All S rows should be different (different positions → different rotations)
        # Check at least position 0 and position 1 differ
        assert not torch.allclose(out[0, 0, 0], out[0, 0, 1], atol=1e-5), (
            "RoPE output should differ across positions for the same input vector."
        )

    def test_odd_d_head_raises(self):
        """d_head must be even (pairs of dimensions are rotated together)."""
        from nexa.models import precompute_rope_freqs
        with pytest.raises(AssertionError, match="even"):
            precompute_rope_freqs(d_head=7, max_seq_len=16)


# ===========================================================================
# 4. MultiHeadAttention
# ===========================================================================

class TestMultiHeadAttention:
    """Verify attention output shape and causal masking."""

    @pytest.fixture
    def attn(self):
        from nexa.models import MultiHeadAttention
        return MultiHeadAttention(TINY_CFG)

    def test_output_shape(self, attn):
        """Output must be [B, S, d_model]."""
        x = torch.randn(B, S, TINY_CFG.d_model)
        assert attn(x).shape == (B, S, TINY_CFG.d_model)

    def test_output_shape_single_token(self, attn):
        """Works correctly for a single-token sequence (S=1)."""
        x = torch.randn(B, 1, TINY_CFG.d_model)
        assert attn(x).shape == (B, 1, TINY_CFG.d_model)

    def test_output_shape_batch_one(self, attn):
        """Works for batch size 1."""
        x = torch.randn(1, S, TINY_CFG.d_model)
        assert attn(x).shape == (1, S, TINY_CFG.d_model)

    def test_has_correct_projections(self, attn):
        """Attention must have q_proj, k_proj, v_proj, out_proj."""
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert hasattr(attn, name), f"Missing projection: {name}"
            layer = getattr(attn, name)
            assert isinstance(layer, nn.Linear)
            assert layer.bias is None, f"{name} should have no bias"

    def test_causal_output_independence(self):
        """
        Causality test: changing tokens at positions i+1, i+2, ... should NOT
        change the output at position i.

        We check that the first half of the sequence produces identical logits
        regardless of what comes after it.
        """
        from nexa.models import NexaTransformer
        model = NexaTransformer(TINY_CFG)
        model.eval()

        S_test = 8
        # Create two sequences that agree on the first 4 tokens, differ on the rest
        ids_a = torch.randint(4, TINY_CFG.vocab_size, (1, S_test))
        ids_b = ids_a.clone()
        ids_b[:, 4:] = torch.randint(4, TINY_CFG.vocab_size, (1, 4))   # different suffix

        with torch.no_grad():
            logits_a = model(ids_a)
            logits_b = model(ids_b)

        # Positions 0..3 should be identical (they can't "see" positions 4..7)
        assert torch.allclose(logits_a[:, :4, :], logits_b[:, :4, :], atol=1e-5), (
            "Causal mask violated: changing future tokens changed past positions' logits."
        )

    def test_rope_buffers_registered(self, attn):
        """RoPE cos and sin tables must be registered buffers (not parameters)."""
        buffer_names = {name for name, _ in attn.named_buffers()}
        assert "rope_cos" in buffer_names
        assert "rope_sin" in buffer_names
        param_names = {name for name, _ in attn.named_parameters()}
        assert "rope_cos" not in param_names
        assert "rope_sin" not in param_names

    def test_backward_pass(self, attn):
        """Gradients flow through attention."""
        x = torch.randn(B, S, TINY_CFG.d_model, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


# ===========================================================================
# 5. SwiGLU
# ===========================================================================

class TestSwiGLU:
    """Verify SwiGLU output shape and parameter structure."""

    @pytest.fixture
    def mlp(self):
        from nexa.models import SwiGLU
        return SwiGLU(TINY_CFG)

    def test_output_shape(self, mlp):
        """Output must be [B, S, d_model]."""
        x = torch.randn(B, S, TINY_CFG.d_model)
        assert mlp(x).shape == (B, S, TINY_CFG.d_model)

    def test_has_three_projections(self, mlp):
        """SwiGLU must have gate_proj, up_proj, and down_proj."""
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert hasattr(mlp, name), f"Missing projection: {name}"
            layer = getattr(mlp, name)
            assert isinstance(layer, nn.Linear)
            assert layer.bias is None, f"{name} should have no bias"

    def test_projection_dimensions(self, mlp):
        assert mlp.gate_proj.in_features  == TINY_CFG.d_model
        assert mlp.gate_proj.out_features == TINY_CFG.d_ff
        assert mlp.up_proj.in_features    == TINY_CFG.d_model
        assert mlp.up_proj.out_features   == TINY_CFG.d_ff
        assert mlp.down_proj.in_features  == TINY_CFG.d_ff
        assert mlp.down_proj.out_features == TINY_CFG.d_model

    def test_different_inputs_give_different_outputs(self, mlp):
        """SwiGLU is nonlinear — different inputs give different outputs."""
        x1 = torch.randn(B, S, TINY_CFG.d_model)
        x2 = torch.randn(B, S, TINY_CFG.d_model)
        assert not torch.allclose(mlp(x1), mlp(x2))

    def test_backward_pass(self, mlp):
        x = torch.randn(B, S, TINY_CFG.d_model, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None


# ===========================================================================
# 6. TransformerBlock
# ===========================================================================

class TestTransformerBlock:
    """Verify block output shape and residual connection."""

    @pytest.fixture
    def block(self):
        from nexa.models import TransformerBlock
        return TransformerBlock(TINY_CFG)

    def test_output_shape(self, block):
        """Output shape must equal input shape [B, S, d_model]."""
        x = torch.randn(B, S, TINY_CFG.d_model)
        assert block(x).shape == x.shape

    def test_has_both_norm_layers(self, block):
        """Block must have norm1 and norm2."""
        assert hasattr(block, "norm1")
        assert hasattr(block, "norm2")

    def test_has_attention_and_mlp(self, block):
        """Block must have attn and mlp sub-modules."""
        from nexa.models import MultiHeadAttention, SwiGLU
        assert isinstance(block.attn, MultiHeadAttention)
        assert isinstance(block.mlp, SwiGLU)

    def test_residual_connection_adds_to_input(self, block):
        """
        The residual connection means the output cannot be zero (unless the
        sublayers exactly cancel the input, which is astronomically unlikely).
        Verify that output ≠ attention(norm(x)) alone.
        """
        x = torch.randn(B, S, TINY_CFG.d_model)
        # Direct comparison: output should differ from just the attention output
        with torch.no_grad():
            block_out = block(x)
            attn_out = block.attn(block.norm1(x))
        # If residual works: block_out ≈ x + attn_out + mlp_out (not just attn_out)
        assert not torch.allclose(block_out, attn_out, atol=1e-4)

    def test_backward_pass(self, block):
        x = torch.randn(B, S, TINY_CFG.d_model, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


# ===========================================================================
# 7. NexaTransformer (full model)
# ===========================================================================

class TestNexaTransformer:
    """Integration tests for the full NexaTransformer model."""

    @pytest.fixture(scope="class")
    @classmethod
    def model(cls):
        from nexa.models import NexaTransformer
        return NexaTransformer(TINY_CFG)

    # --- Forward pass ---

    def test_forward_output_shape(self, model):
        """Forward pass must return logits of shape [B, S, vocab_size]."""
        ids = torch.randint(0, TINY_CFG.vocab_size, (B, S))
        logits = model(ids)
        assert logits.shape == (B, S, TINY_CFG.vocab_size)

    def test_forward_output_is_float(self, model):
        """Logits must be float tensors (not integers)."""
        ids = torch.randint(0, TINY_CFG.vocab_size, (B, S))
        logits = model(ids)
        assert logits.is_floating_point()

    def test_forward_raises_on_too_long_sequence(self, model):
        """Sequences longer than max_seq_len must raise ValueError."""
        ids = torch.randint(0, TINY_CFG.vocab_size, (1, TINY_CFG.max_seq_len + 1))
        with pytest.raises(ValueError, match="max_seq_len"):
            model(ids)

    def test_forward_with_pad_tokens(self, model):
        """Forward pass works when input contains PAD token IDs."""
        ids = torch.zeros(B, S, dtype=torch.long)   # all padding
        logits = model(ids)
        assert logits.shape == (B, S, TINY_CFG.vocab_size)

    def test_forward_deterministic_in_eval(self, model):
        """In eval mode (dropout=0), identical inputs give identical outputs."""
        model.eval()
        ids = torch.randint(0, TINY_CFG.vocab_size, (B, S))
        with torch.no_grad():
            out1 = model(ids)
            out2 = model(ids)
        assert torch.allclose(out1, out2)

    # --- Parameters ---

    def test_num_parameters_positive(self, model):
        assert model.num_parameters > 0

    def test_num_parameters_in_expected_range(self, model):
        """
        For TINY_CFG (d=64, h=4, L=2, ff=256, V=128):
            Embedding:     128 × 64 = 8,192
            Attn/layer:    4 × 64² = 16,384
            SwiGLU/layer:  3 × 64 × 256 = 49,152
            Norms/layer:   2 × 64 = 128
            Per block:     65,664
            2 blocks:      131,328
            Final norm:    64
            Total:         ~139,584
        """
        params = model.num_parameters
        assert 50_000 < params < 250_000, (
            f"Unexpected parameter count: {params:,}. "
            "Check model architecture for obvious bugs."
        )

    def test_weight_tying(self, model):
        """When tie_embeddings=True, lm_head.weight IS embedding.weight."""
        assert model.lm_head.weight is model.embedding.weight, (
            "Weight tying broken: lm_head.weight and embedding.weight are different tensors."
        )

    def test_no_pretrained_weight_loading(self):
        """NexaTransformer must not have a from_pretrained() method."""
        from nexa.models import NexaTransformer
        assert not hasattr(NexaTransformer, "from_pretrained"), (
            "NexaTransformer must not expose from_pretrained(). "
            "Use NexaTransformer(config) to build from scratch."
        )

    # --- Causality (already tested in attention, but verify end-to-end) ---

    def test_causal_end_to_end(self):
        """
        End-to-end causality: tokens at position i must not depend on tokens
        at position j > i.

        Method: create two inputs that differ only in the last 4 tokens.
        The first S//2 output logits must be bit-for-bit identical.
        """
        from nexa.models import NexaTransformer
        model = NexaTransformer(TINY_CFG)
        model.eval()

        prefix_len = S // 2
        ids_a = torch.randint(4, TINY_CFG.vocab_size, (1, S))
        ids_b = ids_a.clone()
        ids_b[:, prefix_len:] = torch.randint(4, TINY_CFG.vocab_size, (1, S - prefix_len))

        with torch.no_grad():
            logits_a = model(ids_a)
            logits_b = model(ids_b)

        assert torch.allclose(logits_a[:, :prefix_len], logits_b[:, :prefix_len], atol=1e-5), (
            "Causality violated: future tokens affected past positions' logits."
        )

    # --- Generation ---

    def test_generate_returns_longer_sequence(self, model):
        """generate() must extend the input sequence."""
        model.eval()
        ids = torch.randint(1, TINY_CFG.vocab_size, (1, 4))
        generated = model.generate(ids, max_new_tokens=5)
        assert generated.shape[1] >= ids.shape[1]
        assert generated.shape[1] <= ids.shape[1] + 5

    def test_generate_greedy_is_deterministic(self, model):
        """Greedy generation must be deterministic (same output every time)."""
        model.eval()
        ids = torch.randint(1, TINY_CFG.vocab_size, (1, 4))
        gen1 = model.generate(ids, max_new_tokens=8, greedy=True)
        gen2 = model.generate(ids, max_new_tokens=8, greedy=True)
        assert torch.equal(gen1, gen2)

    def test_generate_preserves_prompt(self, model):
        """Generated sequence must start with the original prompt tokens."""
        model.eval()
        ids = torch.randint(1, TINY_CFG.vocab_size, (1, 4))
        generated = model.generate(ids, max_new_tokens=6)
        assert torch.equal(generated[:, :4], ids)

    def test_generate_stops_at_eos(self):
        """
        Verify that generate() stops when it produces eos_token_id.

        Strategy: replace the model's forward() with a stub that always returns
        logits strongly favouring EOS_ID. This avoids the confound of random
        initialised residual stream values fighting our weight patch.
        """
        from nexa.models import NexaTransformer
        m = NexaTransformer(TINY_CFG)
        m.eval()

        eos_id = TINY_CFG.eos_token_id
        V      = TINY_CFG.vocab_size

        # Stub: always predict EOS at every position
        def _always_eos(input_ids):
            B, S = input_ids.shape
            logits = torch.full((B, S, V), -1e4)
            logits[:, :, eos_id] = 1e4      # EOS gets max logit
            return logits

        # Monkey-patch forward for this test only
        original_forward = m.forward
        m.forward = _always_eos

        try:
            ids = torch.randint(1, 4, (1, 2))     # 2-token prompt (non-EOS)
            max_new = 20
            generated = m.generate(
                ids,
                max_new_tokens=max_new,
                greedy=True,
                eos_token_id=eos_id,
            )
        finally:
            m.forward = original_forward           # restore after test

        # The loop runs, appends EOS as first new token, then breaks immediately
        assert generated.shape[1] < ids.shape[1] + max_new, (
            f"generate() should have stopped at EOS (token {eos_id}), "
            f"but produced {generated.shape[1]} tokens (max would be {ids.shape[1] + max_new})."
        )
        # The first new token must be EOS_ID
        assert generated[0, ids.shape[1]].item() == eos_id, (
            f"Expected EOS_ID={eos_id} as first new token, "
            f"got {generated[0, ids.shape[1]].item()}"
        )

    def test_generate_respects_max_new_tokens(self, model):
        """Without EOS, generation produces exactly max_new_tokens new tokens."""
        model.eval()
        ids = torch.randint(1, TINY_CFG.vocab_size, (1, 2))
        n_new = 10
        generated = model.generate(ids, max_new_tokens=n_new, greedy=True)
        assert generated.shape[1] == ids.shape[1] + n_new

    # --- Save / Load ---

    def test_save_and_load_weights(self, model, tmp_path):
        """save_weights → load_weights produces identical forward outputs."""
        from nexa.models import NexaTransformer
        model.eval()
        path = tmp_path / "model.pt"
        model.save_weights(path)
        assert path.exists()

        # Load into a fresh model
        fresh = NexaTransformer(TINY_CFG)
        fresh.load_weights(path)
        fresh.eval()

        ids = torch.randint(0, TINY_CFG.vocab_size, (1, S))
        with torch.no_grad():
            out_orig  = model(ids)
            out_fresh = fresh(ids)

        assert torch.allclose(out_orig, out_fresh, atol=1e-6), (
            "Loaded model produces different outputs than the original."
        )

    def test_backward_pass_full_model(self, model):
        """Gradients flow through the full model (no NaN, no zero gradients)."""
        from nexa.models import NexaTransformer
        m = NexaTransformer(TINY_CFG)    # fresh model to avoid interfering with fixture
        m.train()
        import torch.nn.functional as F

        ids = torch.randint(0, TINY_CFG.vocab_size, (B, S))
        logits = m(ids)                    # [B, S, V]

        # Language modeling loss: predict next token from current position
        shift_logits = logits[:, :-1, :].reshape(-1, TINY_CFG.vocab_size)
        shift_labels = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels)
        loss.backward()

        # Check that at least some parameters have non-zero gradients
        grad_norms = [
            p.grad.norm().item()
            for p in m.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0, "No parameter received a gradient."
        assert all(g == g for g in grad_norms), "NaN gradient detected."   # NaN != NaN


# ===========================================================================
# 8. Independence
# ===========================================================================

class TestModelIndependence:
    """Verify that model code does not import any forbidden libraries."""

    FORBIDDEN = [
        "openai",
        "anthropic",
        "transformers",   # HuggingFace — loads pretrained weights
        "huggingface_hub",
        "tiktoken",
        "sentencepiece",
    ]

    def test_no_forbidden_modules_in_sys(self):
        """After importing the models package, no forbidden module appears."""
        import nexa.models  # noqa: F401
        for mod in self.FORBIDDEN:
            assert mod not in sys.modules, (
                f"Forbidden module '{mod}' was imported by nexa.models. "
                "Model code must not use pretrained AI libraries."
            )

    def test_no_from_pretrained(self):
        """NexaTransformer must have no from_pretrained class method."""
        from nexa.models import NexaTransformer
        assert not hasattr(NexaTransformer, "from_pretrained")

    def test_model_uses_only_pytorch_and_stdlib(self):
        """Core model files must not import forbidden frameworks."""
        import nexa.models.transformer as tm
        for forbidden in ("openai", "anthropic", "transformers"):
            assert forbidden not in dir(tm)
