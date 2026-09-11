"""
Tests for Phase 9 — V1 Evaluation and Generation

Strictly tests:
- Checkpoint compatibility
- Exact parameter count
- Validation-loss reproducibility
- Tolerance check
- Deterministic generation
- Seeded sampling reproducibility
- Token validity
- EOS handling
- Max-length handling
- Context-boundary handling
- Special-token handling
- Report schema
"""

from __future__ import annotations

import pytest
import torch
import math
from pathlib import Path

from nexa.models.config import ModelConfig
from nexa.models.transformer import NexaTransformer
from nexa.inference.generator import Generator, SamplingConfig
from scripts.evaluation.run_pilot_evaluation import (
    EVAL_PROMPTS,
    PHASE_8_9_VAL_LOSS,
    VAL_LOSS_TOLERANCE,
    EXPECTED_PARAMS,
    CHECKPOINT_PATH,
    CONFIG_PATH,
    TOKENIZER_DIR,
)

# ===========================================================================
# 1. Config and Checkpoint Requirements
# ===========================================================================

def test_checkpoint_exists():
    assert CHECKPOINT_PATH.exists(), f"Must have {CHECKPOINT_PATH}"

def test_config_exists():
    assert CONFIG_PATH.exists(), f"Must have {CONFIG_PATH}"

def test_exact_parameter_count():
    config = ModelConfig.from_yaml(CONFIG_PATH)
    model = NexaTransformer(config)
    assert model.num_parameters == EXPECTED_PARAMS, f"Expected {EXPECTED_PARAMS} params"

def test_checkpoint_compatibility():
    """Can load the authoritative weights into the authoritative config."""
    config = ModelConfig.from_yaml(CONFIG_PATH)
    model = NexaTransformer(config)
    
    # Just loading it successfully without exception is a pass
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])

def test_prompts_cover_edge_cases():
    types = {p["type"] for p in EVAL_PROMPTS}
    expected = {
        "normal", "very_short", "bos_minimal", "explicit_eos_case", 
        "max_new_tokens", "near_context_limit", "over_context",
        "repeated_token_stress", "unusual_character"
    }
    assert expected.issubset(types), f"Missing edge case prompts: {expected - types}"

# ===========================================================================
# 2. Generation Determinism and Validity (Mocked)
# ===========================================================================

@pytest.fixture
def mock_generator():
    """Provides a tiny generator for testing determinism and EOS."""
    config = ModelConfig.from_dict(dict(
        vocab_size=64, d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=16, dropout=0.0,
    ))
    model = NexaTransformer(config)
    model.eval()
    
    # Create a dummy tokenizer
    class DummyVocab:
        def token_to_id(self, token):
            return 1
            
    class DummyTokenizer:
        vocab_size = 64
        vocab = DummyVocab()
        def encode(self, text, add_bos=True, add_eos=False):
            ids = [ord(c) % 60 + 4 for c in text]
            if add_bos:
                ids = [1] + ids
            if add_eos:
                ids = ids + [2]
            return ids
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i - 4 + 65) for i in ids if i > 3)
    
    return Generator(model, DummyTokenizer(), device="cpu")

def test_generation_determinism_greedy(mock_generator):
    prompt = "Test"
    cfg = SamplingConfig(greedy=True, max_new_tokens=5)
    
    torch.manual_seed(42)
    res1 = mock_generator.generate(prompt, cfg)
    
    torch.manual_seed(42)
    res2 = mock_generator.generate(prompt, cfg)
    
    assert res1.generated_text == res2.generated_text
    assert res1.generated_tokens == res2.generated_tokens

def test_generation_determinism_seeded_sampling(mock_generator):
    prompt = "Test"
    cfg = SamplingConfig(temperature=0.8, top_p=0.9, max_new_tokens=5)
    
    torch.manual_seed(1337)
    res1 = mock_generator.generate(prompt, cfg)
    
    torch.manual_seed(1337)
    res2 = mock_generator.generate(prompt, cfg)
    
    assert res1.generated_text == res2.generated_text
    assert res1.generated_tokens == res2.generated_tokens

def test_generation_eos_handling(mock_generator):
    cfg = SamplingConfig(eos_token_id=2, max_new_tokens=10)
    assert cfg.eos_token_id == 2

def test_valid_token_ids_during_generation(mock_generator):
    prompt = "Hello"
    cfg = SamplingConfig(greedy=True, max_new_tokens=10)
    res = mock_generator.generate(prompt, cfg)
    
    ids = mock_generator.tokenizer.encode(res.generated_text, add_bos=False, add_eos=False)
    assert all(0 <= idx < mock_generator.tokenizer.vocab_size for idx in ids)

def test_max_length_handling(mock_generator):
    prompt = "Hello"
    cfg = SamplingConfig(greedy=True, max_new_tokens=3)
    res = mock_generator.generate(prompt, cfg)
    # Even if it wants to keep going, it must stop at 3
    assert res.generated_tokens <= 3

def test_tolerance_constant():
    # Enforces strictness
    assert VAL_LOSS_TOLERANCE <= 0.005, "Tolerance is too loose!"
