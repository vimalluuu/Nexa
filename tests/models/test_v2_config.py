"""
tests/models/test_v2_config.py
==============================
Tests the parameter definitions and config initialization for Nexa V2 models.
"""

from nexa.models.config import ModelConfig
from nexa.models.transformer import NexaTransformer

def test_v2_model_config_candidate_b():
    # Test Candidate B architecture
    cfg = ModelConfig(vocab_size=8192, d_model=512, n_layers=6, n_heads=8, d_ff=2048)
    
    assert cfg.vocab_size == 8192
    assert cfg.d_model == 512
    assert cfg.n_layers == 6
    assert cfg.n_heads == 8
    assert cfg.d_ff == 2048
    
    # Init transformer
    model = NexaTransformer(cfg)
    
    total = sum(p.numel() for p in model.parameters())
    emb = model.embedding.weight.numel()
    
    assert total == 29366784 # 29.3M
    assert emb == 4194304 # 4.1M

def test_v2_model_config_candidate_a():
    # Test Candidate A architecture
    cfg = ModelConfig(vocab_size=8192, d_model=384, n_layers=6, n_heads=12, d_ff=1536)
    model = NexaTransformer(cfg)
    
    total = sum(p.numel() for p in model.parameters())
    assert total == 17306496 # 17.3M
