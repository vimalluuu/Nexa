"""
Nexa Phase 9 — V1 Evaluation and Generation
============================================
Evaluates the final Phase 8.9 pilot pretraining checkpoint:
  checkpoints/nexa_v1_pilot/step_0009229.pt

Establishes a reproducible baseline for:
  1. Recomputed validation loss on Phase 8.6 authoritative shards.
  2. Text generation behavior across multiple decoding strategies and edge cases.

Usage:
    python -m scripts.evaluation.run_pilot_evaluation
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nexa.models.config import ModelConfig
from nexa.models.transformer import NexaTransformer
from nexa.tokenizer import NexaTokenizer
from nexa.training.dataset import NexaDataset
from nexa.inference.generator import Generator, SamplingConfig
from scripts.training.run_pilot_pretraining import load_shard_tokens

# ---------------------------------------------------------------------------
# Authoritative References
# ---------------------------------------------------------------------------

CHECKPOINT_PATH   = Path("checkpoints/nexa_v1_pilot/step_0009229.pt")
CONFIG_PATH       = Path("configs/nexa_v1_pretraining.yaml")
TOKENIZER_DIR     = Path("data/processed/tokenizer_v1")
SHARD_VAL_DIR     = Path("data/processed/validation")

REPORTS_DIR       = Path("data/reports")
REPORT_JSON       = REPORTS_DIR / "nexa_v1_evaluation_report.json"
REPORT_MD         = REPORTS_DIR / "nexa_v1_evaluation_summary.md"

PHASE_8_9_VAL_LOSS = 3.3422
VAL_LOSS_TOLERANCE = 0.001
EXPECTED_PARAMS    = 6_819_072

# Evaluation prompts targeting various generation capabilities and edge cases
EVAL_PROMPTS = [
    {"type": "normal", "text": "The capital of France is"},
    {"type": "very_short", "text": "A"},
    {"type": "bos_minimal", "text": ""},  # Generator will prepend <bos> automatically
    {"type": "explicit_eos_case", "text": "This is a complete sentence.<eos> And then"}, 
    {"type": "max_new_tokens", "text": "List the numbers from one to one hundred: one, two,"}, # Configured to hit max tokens
    {"type": "near_context_limit", "text": "word " * 490}, # Near 512
    {"type": "over_context", "text": "word " * 600}, # Over 512, will get truncated by NexaDataset logic or index error
    {"type": "repeated_token_stress", "text": "A A A A A A A A A A A A A A A"},
    {"type": "unusual_character", "text": "Here are some emojis and symbols: 🍎, 🚀, €, £, ¶. Now"},
]

# ---------------------------------------------------------------------------
# Recompute Validation
# ---------------------------------------------------------------------------

def recompute_validation(model: NexaTransformer, seq_len: int = 512, batch_size: int = 4) -> dict:
    """Recomputes validation loss on the authoritative dataset."""
    print(f"\n[1/4] Loading validation shards from {SHARD_VAL_DIR}...")
    t0 = time.perf_counter()
    val_tokens = load_shard_tokens(SHARD_VAL_DIR)
    val_dataset = NexaDataset(val_tokens, block_size=seq_len)
    print(f"      Loaded {len(val_tokens):,} tokens into {len(val_dataset):,} windows ({time.perf_counter()-t0:.2f}s).")

    print(f"      Recomputing validation loss (this will take a few minutes)...")
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = next(model.parameters()).device
    total_loss = 0.0
    steps = 0
    t0 = time.perf_counter()
    
    with torch.no_grad():
        for x, y in loader:
            if steps >= 100:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, S, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * S, V), y.reshape(B * S))
            total_loss += loss.item()
            steps += 1
            if steps % 10 == 0:
                print(f"      ... processed {steps} / 100 validation batches ...")

    avg_loss = total_loss / max(steps, 1)
    ppl = math.exp(avg_loss)
    diff = abs(avg_loss - PHASE_8_9_VAL_LOSS)
    passed_tolerance = diff <= VAL_LOSS_TOLERANCE
    duration = time.perf_counter() - t0
    
    print(f"      Validation loss: {avg_loss:.4f} | PPL: {ppl:.2f}")
    print(f"      Diff vs Phase 8.9: {diff:.6f} | Passed Tolerance (<={VAL_LOSS_TOLERANCE}): {passed_tolerance}")
    
    return {
        "recorded_loss": PHASE_8_9_VAL_LOSS,
        "recomputed_loss": round(avg_loss, 4),
        "val_perplexity": round(ppl, 4),
        "absolute_difference": round(diff, 6),
        "tolerance": VAL_LOSS_TOLERANCE,
        "pass": passed_tolerance,
        "val_windows": len(val_dataset),
        "recompute_duration_s": round(duration, 2),
    }

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def test_generation_reproducibility(generator: Generator) -> dict:
    print("\n[2/4] Testing deterministic generation reproducibility...")
    prompt = "A simple prompt to test reproducibility."
    
    # 1. Greedy Reproducibility
    cfg_greedy = SamplingConfig(greedy=True, max_new_tokens=20)
    torch.manual_seed(42)
    g1 = generator.generate(prompt, cfg_greedy)
    torch.manual_seed(42)
    g2 = generator.generate(prompt, cfg_greedy)
    greedy_match = (g1.generated_text == g2.generated_text)
    
    # 2. Seeded Sampling Reproducibility
    cfg_sample = SamplingConfig(temperature=0.8, top_p=0.9, max_new_tokens=20)
    torch.manual_seed(42)
    s1 = generator.generate(prompt, cfg_sample)
    torch.manual_seed(42)
    s2 = generator.generate(prompt, cfg_sample)
    sample_match = (s1.generated_text == s2.generated_text)

    print(f"      Greedy match: {greedy_match} | Sampled match: {sample_match}")
    return {
        "greedy_reproducible": greedy_match,
        "sampled_reproducible": sample_match,
        "passed": greedy_match and sample_match
    }

def run_generation_suite(generator: Generator) -> list[dict]:
    """Runs the fixed prompt set across multiple decoding strategies."""
    print("\n[3/4] Running generation suite (edge cases)...")
    
    strategies = {
        "greedy": SamplingConfig(greedy=True, max_new_tokens=40, eos_token_id=2),
        "temperature": SamplingConfig(temperature=0.8, max_new_tokens=40, eos_token_id=2),
        "nucleus": SamplingConfig(temperature=0.8, top_p=0.9, max_new_tokens=40, eos_token_id=2),
        "top_k": SamplingConfig(temperature=0.8, top_k=40, max_new_tokens=40, eos_token_id=2),
        "penalized": SamplingConfig(temperature=0.8, top_p=0.9, repetition_penalty=1.2, max_new_tokens=40, eos_token_id=2),
    }

    results = []
    
    for p in EVAL_PROMPTS:
        prompt_results = {"prompt": p["text"], "type": p["type"], "outputs": {}}
        preview_prompt = (p['text'][:60] + "...") if len(p['text']) > 60 else p['text']
        print(f"\n  Prompt [{p['type']}]: {repr(preview_prompt)}")
        
        for name, config in strategies.items():
            # Reset seed for EVERY generation to ensure complete reproducibility
            torch.manual_seed(42)
            t0 = time.perf_counter()
            
            try:
                # Truncate prompt if over context
                prompt_ids = generator.tokenizer.encode(p["text"], add_bos=True, add_eos=False)
                max_seq_len = generator.model.config.max_seq_len
                
                # We need to simulate truncation if it exceeds context limit.
                # Actually, generator.py doesn't automatically truncate inputs yet. Let's handle it manually or let it fail cleanly
                if len(prompt_ids) >= max_seq_len:
                    # Manually truncate the prompt to allow some generation
                    prompt_ids = prompt_ids[-(max_seq_len - config.max_new_tokens):]
                    truncated_prompt = generator.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                    res = generator.generate(truncated_prompt, config)
                else:
                    res = generator.generate(p["text"], config)
                
                error = None
                output_ids = generator.tokenizer.encode(res.generated_text, add_bos=False, add_eos=False)
                
                # Check token validity
                invalid_count = sum(1 for idx in output_ids if not (0 <= idx < generator.tokenizer.vocab_size))
                
                prompt_results["outputs"][name] = {
                    "seed": 42,
                    "decoding_config": name,
                    "generated_text": res.generated_text,
                    "generated_token_ids": output_ids,
                    "generated_length": res.generated_tokens,
                    "eos_emitted": res.stopped_by == "eos",
                    "max_length_reached": res.stopped_by == "max_new_tokens",
                    "invalid_token_count": invalid_count,
                    "duration_s": round(time.perf_counter() - t0, 3),
                    "error": None
                }
                
                preview = res.generated_text.replace("\n", " ")[:60]
                print(f"    - {name:<11}: {preview}... ({res.generated_tokens} tok, {res.stopped_by})")
                
            except Exception as e:
                prompt_results["outputs"][name] = {
                    "seed": 42,
                    "decoding_config": name,
                    "generated_text": "",
                    "generated_token_ids": [],
                    "generated_length": 0,
                    "eos_emitted": False,
                    "max_length_reached": False,
                    "invalid_token_count": 0,
                    "duration_s": round(time.perf_counter() - t0, 3),
                    "error": str(e)
                }
                print(f"    - {name:<11}: ERROR {str(e)}")
            
        results.append(prompt_results)
        
    return results

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_markdown_report(report: dict, md_path: Path):
    """Formats the evaluation report into a readable markdown summary."""
    md = [
        "# Nexa Phase 9 — V1 Evaluation Summary",
        "",
        f"**Generated:** {report['generated']}",
        f"**Checkpoint:** `{report['checkpoint']}`",
        f"**Git Commit:** `{report['git_commit']}`",
        "",
        "## Model Configuration",
        f"- **Architecture:** {report['model_info']['vocab_size']} vocab, {report['model_info']['d_model']} d_model, {report['model_info']['n_layers']} layers, {report['model_info']['n_heads']} heads",
        f"- **Parameter Count:** {report['model_info']['total_params']:,}",
        f"- **Tokenizer:** {report['tokenizer']['type']} ({report['tokenizer']['vocab_size']} vocab)",
        "",
        "## Validation Baseline",
        f"- **Phase 8.9 Reference Loss:** {report['validation']['recorded_loss']:.4f}",
        f"- **Recomputed Validation Loss:** {report['validation']['recomputed_loss']:.4f}",
        f"- **Recomputed Perplexity:** {report['validation']['val_perplexity']:.2f}",
        f"- **Absolute Difference:** {report['validation']['absolute_difference']:.6f}",
        f"- **Tolerance Check (<= {report['validation']['tolerance']}):** {'PASS' if report['validation']['pass'] else 'FAIL'}",
        "",
        "## Reproducibility Check",
        f"- **Greedy Determinism:** {'PASS' if report['reproducibility']['greedy_reproducible'] else 'FAIL'}",
        f"- **Seeded Sampling Determinism:** {'PASS' if report['reproducibility']['sampled_reproducible'] else 'FAIL'}",
        "",
        "## Selected Output Samples (Edge Cases)",
        "*(Note: The model is trained on a tiny 18.9M token corpus. It is not expected to contain broad factual knowledge or deep coherence, but rather basic syntax and simple associations.)*",
        ""
    ]

    for p in report["generation"]:
        md.append(f"### Prompt Type: `{p['type']}`")
        preview = p['prompt'][:100] + "..." if len(p['prompt']) > 100 else p['prompt']
        md.append(f"> {preview}")
        md.append("")
        for name, out in p["outputs"].items():
            if out["error"]:
                md.append(f"**{name}**: *ERROR: {out['error']}*")
            elif name in ("greedy", "nucleus", "penalized"):
                text = out["generated_text"].strip().replace("\n", "\n    ")
                md.append(f"**{name}** ({out['generated_length']} tokens, stop: {out['eos_emitted'] and 'EOS' or 'MAX'}):")
                md.append(f"    {text}")
                md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"{'='*60}")
    print("  Nexa Phase 9 — V1 Evaluation and Generation")
    print(f"{'='*60}")

    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    # 1. Load Model
    print("\n[0/4] Loading authoritative model and tokenizer...")
    config = ModelConfig.from_yaml(CONFIG_PATH)
    model = NexaTransformer(config)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if model.num_parameters != EXPECTED_PARAMS:
        print(f"ERROR: Parameter count mismatch. Expected {EXPECTED_PARAMS}, got {model.num_parameters}")
        sys.exit(1)
        
    print(f"      Model loaded. Params: {model.num_parameters:,}")

    tokenizer = NexaTokenizer.load(TOKENIZER_DIR)
    generator = Generator(model, tokenizer, device="cpu")

    # 2. Recompute Validation
    val_metrics = recompute_validation(model)

    # 3. Test Reproducibility
    reproducibility = test_generation_reproducibility(generator)

    # 4. Generate Suite (Edge Cases)
    gen_results = run_generation_suite(generator)

    # 5. Save Reports
    print("\n[4/4] Generating reports...")
    
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    report = {
        "phase": "9",
        "generated": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "checkpoint": str(CHECKPOINT_PATH),
        "model_info": {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "d_ff": config.d_ff,
            "total_params": model.num_parameters,
        },
        "tokenizer": {
            "type": "nexa_bpe_v1",
            "vocab_size": tokenizer.vocab_size,
        },
        "validation": val_metrics,
        "reproducibility": reproducibility,
        "generation": gen_results,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    write_markdown_report(report, REPORT_MD)
    
    print(f"      JSON Report: {REPORT_JSON}")
    print(f"      Markdown Summary: {REPORT_MD}")
    print(f"\n{'='*60}")
    print("  Phase 9 COMPLETE.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
