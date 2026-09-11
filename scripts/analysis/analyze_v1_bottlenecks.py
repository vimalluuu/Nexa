"""
Nexa Phase 10: V1 Improvement Analysis and Scaling Plan
Produces findings and recommendations for Nexa V2.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
import subprocess

REPORTS_DIR = Path("data/reports")
JSON_PATH = REPORTS_DIR / "nexa_v1_improvement_analysis.json"
MD_PATH = REPORTS_DIR / "nexa_v1_improvement_analysis.md"

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def generate_analysis():
    return {
        "phase": "10",
        "generated": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        
        "baseline_preservation": {
            "checkpoint": "checkpoints/nexa_v1_pilot/step_0009229.pt",
            "validation_loss": 3.3635,
            "validation_ppl": 28.89,
            "status": "frozen"
        },
        
        "dataset_analysis": {
            "total_documents": 2257,
            "total_training_tokens": 18901546,
            "total_validation_tokens": 2000821,
            "source_distribution": "Wikimedia English Pilot",
            "findings": {
                "OBSERVED": [
                    "The current corpus is ~18.9M training tokens.",
                    "The pilot training pipeline successfully tokenizes and deduplicates the corpus."
                ],
                "INFERRED": [
                    "The dataset size is the primary limiter of factual generalization."
                ],
                "HYPOTHESIZED": [
                    "Chinchilla scaling laws suggest ~136M tokens as a theoretical compute-optimal target for a 6.8M parameter model, meaning we could safely scale the dataset by 5-10x.",
                    "Scaling actual training data needs will depend heavily on data quality, filtering, and model capacity, not just raw token volume."
                ]
            }
        },
        
        "tokenizer_analysis": {
            "findings": {
                "OBSERVED": [
                    "The tokenizer (nexa/tokenizer/tokenizer.py) implements Character-level BPE, not true byte-level BPE.",
                    "End-of-word is marked with `</w>`, which is replaced by a space during decoding.",
                    "Unseen characters are explicitly mapped to `<unk>`.",
                    "Tokenization efficiency is ~2.27 tokens per word on the Wikimedia dataset.",
                    "The vocabulary size is 2048."
                ],
                "INFERRED": [
                    "The strict character-level constraint (without byte-fallback) means any Unicode character not seen during BPE training immediately becomes `<unk>`.",
                    "The small vocabulary size forces frequent word fragmentation, leading to higher token-per-word counts."
                ],
                "HYPOTHESIZED": [
                    "Malformed text outputs like 'expeese' and 'Autoese' may stem from the model struggling to correctly predict long sequences of short token fragments.",
                    "Using a larger vocab size (4096, 8192, or 16384) or migrating to true Byte-Level BPE (which prevents <unk> tokens entirely) may reduce malformed-token artifacts, but this must be measured rather than assumed."
                ]
            }
        },
        
        "model_capacity_analysis": {
            "models": {
                "current_v1": {
                    "params_total": "6.8M",
                    "embedding_params": "~0.5M",
                    "memory_weights": "~27 MB",
                    "memory_optimizer": "~54 MB",
                    "memory_activations": "~6 MB",
                    "cpu_feasibility": "High (~1,145 tok/s)",
                    "relative_cost": "1x"
                },
                "moderate": {
                    "params_total": "~15M",
                    "embedding_params": "~1.5M",
                    "memory_weights": "~60 MB",
                    "memory_optimizer": "~120 MB",
                    "memory_activations": "~12 MB",
                    "cpu_feasibility": "Moderate (~400-600 tok/s)",
                    "relative_cost": "~2.2x"
                },
                "larger": {
                    "params_total": "~30M",
                    "embedding_params": "~2M",
                    "memory_weights": "~120 MB",
                    "memory_optimizer": "~240 MB",
                    "memory_activations": "~24 MB",
                    "cpu_feasibility": "Low (~150-250 tok/s)",
                    "relative_cost": "~4.5x"
                }
            },
            "findings": {
                "OBSERVED": [
                    "The 6.8M model executes comfortably on a Ryzen 5 5600G CPU at ~1,145 tok/s.",
                    "Memory usage is negligible (<100 MB total)."
                ],
                "INFERRED": [
                    "Scaling to ~15M parameters is well within the 16GB RAM budget but will halve throughput.",
                    "Scaling to ~30M parameters is memory-safe but will make single-epoch training on >100M tokens extremely slow on CPU."
                ],
                "HYPOTHESIZED": [
                    "A ~15M parameter model hits the optimal balance of representation capacity and CPU-bound turnaround times."
                ]
            }
        },
        
        "context_length_analysis": {
            "findings": {
                "OBSERVED": [
                    "Current context length is 512 tokens."
                ],
                "INFERRED": [
                    "Attention memory and compute cost scales quadratically. Moving to 1024 would 4x the attention compute bottleneck per forward pass.",
                    "The current dataset consists of encyclopedic extracts where short-range factual recall is common."
                ],
                "HYPOTHESIZED": [
                    "Given the small parameter count, the model likely lacks the capacity to effectively attend to 2048 tokens of context anyway.",
                    "Context length 512 or 1024 is sufficient; increasing to 2048 does not yield enough benefit for the massive CPU cost."
                ]
            }
        },
        
        "training_regime_analysis": {
            "findings": {
                "OBSERVED": [
                    "Training ran for exactly 1 pass (1 epoch) over 18.9M tokens.",
                    "Final train loss was 2.4716.",
                    "Final val loss was 3.3635."
                ],
                "INFERRED": [
                    "The model is not suffering from optimization limitation (loss decreased smoothly).",
                    "The model is not catastrophically overfitting, as validation loss did not aggressively diverge, though a ~0.89 gap exists between train and val.",
                    "The model is data-limited."
                ],
                "HYPOTHESIZED": [
                    "Multiple epochs or a larger dataset will continue to yield loss improvements.",
                    "The model is currently undertrained."
                ]
            }
        },
        
        "v2_recommendation": {
            "dataset_target": {"value": "~100M - 150M tokens", "status": "RECOMMENDED"},
            "tokenizer_design": {"value": "True Byte-Level BPE (No <unk>)", "status": "RECOMMENDED"},
            "vocab_size": {"value": 8192, "status": "RECOMMENDED"},
            "vocab_size_alternative": {"value": 4096, "status": "ALTERNATIVE"},
            "d_model": {"value": 384, "status": "RECOMMENDED"},
            "layers": {"value": 8, "status": "RECOMMENDED"},
            "heads": {"value": 12, "status": "RECOMMENDED"},
            "d_ff": {"value": 1536, "status": "RECOMMENDED"},
            "larger_model_architecture": {"value": "d_model=512, layers=10", "status": "DEFER"},
            "context_length": {"value": 512, "status": "RECOMMENDED"},
            "context_length_alternative": {"value": 1024, "status": "ALTERNATIVE"},
            "batch_size": {"value": 8, "status": "RECOMMENDED"},
            "learning_rate": {"value": "3e-4", "status": "RECOMMENDED"},
            "warmup_steps": {"value": 1000, "status": "RECOMMENDED"},
            "training_token_target": {"value": "~150M", "status": "RECOMMENDED"},
            "checkpoint_interval": {"value": 5000, "status": "RECOMMENDED"},
            "validation_interval": {"value": 500, "status": "RECOMMENDED"}
        },
        
        "experiment_priority_matrix": [
            {
                "priority": 1,
                "experiment": "Tokenizer Rewrite (Byte-Level BPE)",
                "compute_cost": "Very Low (No training)",
                "information_value": "Critical. Determines vocabulary and tokenization efficiency before any data expansion. It may reduce malformed-token artifacts, but this must be measured rather than assumed."
            },
            {
                "priority": 2,
                "experiment": "Dataset Expansion & Quality Filter",
                "compute_cost": "Low (Data processing only)",
                "information_value": "High. Prepares the pipeline for theoretical scaling-law limits."
            },
            {
                "priority": 3,
                "experiment": "Mini Architecture Test (Same Model vs Moderate Model on 10M tokens)",
                "compute_cost": "Moderate (A few hours)",
                "information_value": "High. Quantifies exact CPU throughput drop and loss trajectory curve before committing to full 150M run."
            },
            {
                "priority": 4,
                "experiment": "Full Nexa V2 Training Run",
                "compute_cost": "Very High (Days)",
                "information_value": "Final Validation."
            }
        ]
    }


def write_markdown_report(data: dict, md_path: Path):
    md = [
        "# Nexa Phase 10 — V1 Improvement Analysis and Scaling Plan",
        "",
        f"**Generated:** {data['generated']}",
        f"**Git Commit:** `{data['git_commit']}`",
        "",
        "## Executive Summary",
        "The Phase 10 analysis deconstructs the frozen Phase 9 evaluation baseline to identify exact bottlenecks in the Nexa V1 architecture. The analysis separates hard observations from inferences and theoretical hypotheses to dictate a concrete, CPU-safe roadmap for Nexa V2.",
        "",
        "## Frozen V1 Baseline",
        f"- **Checkpoint:** `{data['baseline_preservation']['checkpoint']}`",
        f"- **Validation Loss:** {data['baseline_preservation']['validation_loss']}",
        f"- **Status:** {data['baseline_preservation']['status'].upper()}",
        "",
        "## Dataset Analysis",
        "**OBSERVED**",
        *[f"- {x}" for x in data['dataset_analysis']['findings']['OBSERVED']],
        "**INFERRED**",
        *[f"- {x}" for x in data['dataset_analysis']['findings']['INFERRED']],
        "**HYPOTHESIZED**",
        *[f"- {x}" for x in data['dataset_analysis']['findings']['HYPOTHESIZED']],
        "",
        "## Tokenizer Analysis",
        "**OBSERVED**",
        *[f"- {x}" for x in data['tokenizer_analysis']['findings']['OBSERVED']],
        "**INFERRED**",
        *[f"- {x}" for x in data['tokenizer_analysis']['findings']['INFERRED']],
        "**HYPOTHESIZED**",
        *[f"- {x}" for x in data['tokenizer_analysis']['findings']['HYPOTHESIZED']],
        "",
        "## Model Capacity Analysis",
        "| Model | Params | CPU Feasibility | Est. Memory (Weights) | Relative Cost |",
        "|-------|--------|-----------------|-----------------------|---------------|",
        f"| Current V1 | {data['model_capacity_analysis']['models']['current_v1']['params_total']} | {data['model_capacity_analysis']['models']['current_v1']['cpu_feasibility']} | {data['model_capacity_analysis']['models']['current_v1']['memory_weights']} | {data['model_capacity_analysis']['models']['current_v1']['relative_cost']} |",
        f"| Moderate   | {data['model_capacity_analysis']['models']['moderate']['params_total']} | {data['model_capacity_analysis']['models']['moderate']['cpu_feasibility']} | {data['model_capacity_analysis']['models']['moderate']['memory_weights']} | {data['model_capacity_analysis']['models']['moderate']['relative_cost']} |",
        f"| Larger     | {data['model_capacity_analysis']['models']['larger']['params_total']} | {data['model_capacity_analysis']['models']['larger']['cpu_feasibility']} | {data['model_capacity_analysis']['models']['larger']['memory_weights']} | {data['model_capacity_analysis']['models']['larger']['relative_cost']} |",
        "",
        "**OBSERVED**",
        *[f"- {x}" for x in data['model_capacity_analysis']['findings']['OBSERVED']],
        "**INFERRED**",
        *[f"- {x}" for x in data['model_capacity_analysis']['findings']['INFERRED']],
        "**HYPOTHESIZED**",
        *[f"- {x}" for x in data['model_capacity_analysis']['findings']['HYPOTHESIZED']],
        "",
        "## Context Analysis",
        "**OBSERVED**",
        *[f"- {x}" for x in data['context_length_analysis']['findings']['OBSERVED']],
        "**INFERRED**",
        *[f"- {x}" for x in data['context_length_analysis']['findings']['INFERRED']],
        "**HYPOTHESIZED**",
        *[f"- {x}" for x in data['context_length_analysis']['findings']['HYPOTHESIZED']],
        "",
        "## Training Regime Analysis",
        "**OBSERVED**",
        *[f"- {x}" for x in data['training_regime_analysis']['findings']['OBSERVED']],
        "**INFERRED**",
        *[f"- {x}" for x in data['training_regime_analysis']['findings']['INFERRED']],
        "**HYPOTHESIZED**",
        *[f"- {x}" for x in data['training_regime_analysis']['findings']['HYPOTHESIZED']],
        "",
        "## V2 Recommendation",
        "The following is a concrete V2 proposal based on the bottleneck analysis.",
        ""
    ]
    
    for k, v in data['v2_recommendation'].items():
        md.append(f"- **{k.replace('_', ' ').title()}**: {v['value']} `[{v['status']}]`")
        
    md.extend([
        "",
        "## Experiment Priority Matrix",
        "Ranked strictly by Information Value per Unit of Compute:",
        ""
    ])
    
    for exp in data['experiment_priority_matrix']:
        md.append(f"### {exp['priority']}. {exp['experiment']}")
        md.append(f"- **Compute Cost:** {exp['compute_cost']}")
        md.append(f"- **Value:** {exp['information_value']}")
        md.append("")
    
    md_path.write_text("\n".join(md), encoding="utf-8")


def main():
    print("============================================================")
    print("  Nexa Phase 10 — V1 Improvement Analysis")
    print("============================================================")
    
    data = generate_analysis()
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    write_markdown_report(data, MD_PATH)
    
    print(f"JSON Report written to: {JSON_PATH}")
    print(f"Markdown Report written to: {MD_PATH}")
    print("\nPhase 10 Analysis Complete.")

if __name__ == "__main__":
    main()
