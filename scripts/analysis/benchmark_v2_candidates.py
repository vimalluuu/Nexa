"""
scripts/analysis/benchmark_v2_candidates.py
============================================
Benchmarks Candidate A (~15M), Candidate B (~30M), and V1 Baseline on CPU.
Measures forward, backward, optimizer step time, and end-to-end throughput.
"""

import time
import json
import torch
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexa.models.config import ModelConfig
from nexa.models.transformer import NexaTransformer

# Device is CPU
device = torch.device("cpu")

REPORTS_DIR = Path("data/reports")
BATCH_SIZE = 4
STEPS = 5

def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    emb = model.embedding.weight.numel()
    return total, emb

def benchmark_config(name: str, config: ModelConfig, seq_len: int) -> dict:
    print(f"\n--- Benchmarking {name} (Context: {seq_len}) ---")
    
    config.max_seq_len = seq_len
    model = NexaTransformer(config)
    model.to(device)
    model.train()
    
    total_params, emb_params = count_parameters(model)
    print(f"Total Params: {total_params:,} | Embeddings: {emb_params:,} ({(emb_params/total_params)*100:.1f}%)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    # Dummy data
    x = torch.randint(0, config.vocab_size, (BATCH_SIZE, seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (BATCH_SIZE, seq_len), device=device)
    
    # Warmup
    print("Warming up...")
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    forward_times = []
    backward_times = []
    step_times = []
    end_to_end_times = []
    
    print(f"Running {STEPS} steps...")
    for i in range(STEPS):
        t0 = time.time()
        
        # Forward
        tf0 = time.time()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        tf1 = time.time()
        forward_times.append(tf1 - tf0)
        
        # Backward
        tb0 = time.time()
        loss.backward()
        tb1 = time.time()
        backward_times.append(tb1 - tb0)
        
        # Optimizer
        to0 = time.time()
        optimizer.step()
        optimizer.zero_grad()
        to1 = time.time()
        step_times.append(to1 - to0)
        
        t1 = time.time()
        end_to_end_times.append(t1 - t0)
        
    avg_fwd = sum(forward_times) / STEPS
    avg_bwd = sum(backward_times) / STEPS
    avg_opt = sum(step_times) / STEPS
    avg_e2e = sum(end_to_end_times) / STEPS
    
    tokens_per_sec = (BATCH_SIZE * seq_len) / avg_e2e
    
    print(f"Throughput: {tokens_per_sec:.1f} tok/s")
    print(f"Fwd: {avg_fwd*1000:.1f}ms | Bwd: {avg_bwd*1000:.1f}ms | Opt: {avg_opt*1000:.1f}ms")
    
    # Free memory
    del model
    del optimizer
    del x
    del y
    import gc
    gc.collect()
    
    return {
        "name": name,
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "seq_len": seq_len,
        "total_params": total_params,
        "embedding_params": emb_params,
        "forward_ms": avg_fwd * 1000,
        "backward_ms": avg_bwd * 1000,
        "optimizer_ms": avg_opt * 1000,
        "step_time_ms": avg_e2e * 1000,
        "tokens_per_sec": tokens_per_sec
    }

def main():
    torch.set_num_threads(6) # Ryzen 5 5600G physical cores
    
    results = []
    
    # 1. V1 Reference (6.8M)
    v1_cfg = ModelConfig(vocab_size=2048, d_model=256, n_layers=6, n_heads=8, d_ff=1024)
    results.append(benchmark_config("V1 Baseline", v1_cfg, 512))
    
    # 2. Candidate A (~15M)
    cand_a_cfg = ModelConfig(vocab_size=8192, d_model=384, n_layers=6, n_heads=12, d_ff=1536)
    results.append(benchmark_config("Candidate A (512)", cand_a_cfg, 512))
    results.append(benchmark_config("Candidate A (1024)", cand_a_cfg, 1024))
    results.append(benchmark_config("Candidate A (2048)", cand_a_cfg, 2048))
    
    # 3. Candidate B (~30M)
    cand_b_cfg = ModelConfig(vocab_size=8192, d_model=512, n_layers=6, n_heads=8, d_ff=2048)
    results.append(benchmark_config("Candidate B (512)", cand_b_cfg, 512))
    results.append(benchmark_config("Candidate B (1024)", cand_b_cfg, 1024))
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
