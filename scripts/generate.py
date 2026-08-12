"""
scripts/generate.py
====================
CLI entry point: generate text from a trained NexaTransformer.

Loads the trained model weights and tokenizer, then runs autoregressive
text generation using the specified sampling strategy.

Prerequisites
-------------
Run scripts/train.py first to produce:
    checkpoints/nexa_final.pt       (model weights)
    checkpoints/model_config.json   (model architecture config)
    data/processed/tokenizer/       (trained BPE tokenizer)

Usage
-----
# Greedy decoding (deterministic):
python scripts/generate.py --prompt "the cat sat" --greedy

# Balanced sampling (temperature + nucleus + repetition penalty):
python scripts/generate.py \\
    --prompt "morning is the time" \\
    --temperature 0.8 \\
    --top-p 0.9 \\
    --repetition-penalty 1.2 \\
    --max-new-tokens 50

# Creative sampling (higher temperature):
python scripts/generate.py \\
    --prompt "the rat ran" \\
    --temperature 1.2 \\
    --top-k 50 \\
    --max-new-tokens 80

# Interactive mode (keep prompting):
python scripts/generate.py --interactive

Independence
------------
No pretrained weights, no external AI APIs.
All generation uses weights trained from scratch in Phase 4.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexa.inference import Generator, SamplingConfig
from nexa.utils import get_logger

log = get_logger("nexa.generate", log_to_file=False)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_TOKENIZER_DIR  = Path("data/processed/tokenizer")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate text with Nexa's trained language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model + tokenizer paths
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
                   help="Directory containing nexa_final.pt and model_config.json.")
    p.add_argument("--tokenizer-dir",  type=Path, default=DEFAULT_TOKENIZER_DIR,
                   help="Directory containing the trained tokenizer.")
    p.add_argument("--device", default="auto",
                   help="Device: 'auto', 'cpu', 'cuda', or 'mps'.")

    # Prompt
    p.add_argument("--prompt", type=str, default="the cat sat",
                   help="Text prompt to generate from.")
    p.add_argument("--interactive", action="store_true",
                   help="Run in interactive mode (keep prompting).")

    # Sampling strategy
    g = p.add_argument_group("Sampling")
    g.add_argument("--greedy", action="store_true",
                   help="Greedy decoding (deterministic argmax). Ignores temperature.")
    g.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature. < 1 = focused, > 1 = creative.")
    g.add_argument("--top-k", type=int, default=0,
                   help="Top-K filter. 0 = disabled.")
    g.add_argument("--top-p", type=float, default=0.9,
                   help="Nucleus (top-p) filter. 1.0 = disabled.")
    g.add_argument("--repetition-penalty", type=float, default=1.2,
                   help="Repetition penalty. 1.0 = disabled.")
    g.add_argument("--max-new-tokens", type=int, default=60,
                   help="Maximum number of new tokens to generate.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> SamplingConfig:
    """Build a SamplingConfig from parsed CLI arguments."""
    if args.greedy:
        return SamplingConfig.greedy_config(max_new_tokens=args.max_new_tokens)
    return SamplingConfig(
        temperature        = args.temperature,
        top_k              = args.top_k,
        top_p              = args.top_p,
        repetition_penalty = args.repetition_penalty,
        max_new_tokens     = args.max_new_tokens,
    )


def print_config(config: SamplingConfig) -> None:
    print("\n-- Sampling Config -------------------------------------------")
    if config.greedy:
        print("  Strategy         : greedy (deterministic argmax)")
    else:
        print(f"  Strategy         : sampling")
        print(f"  Temperature      : {config.temperature}")
        print(f"  Top-K            : {config.top_k if config.top_k > 0 else 'disabled'}")
        print(f"  Top-P (nucleus)  : {config.top_p if config.top_p < 1.0 else 'disabled'}")
        print(f"  Rep. penalty     : {config.repetition_penalty if config.repetition_penalty != 1.0 else 'disabled'}")
    print(f"  Max new tokens   : {config.max_new_tokens}")
    print("-------------------------------------------------------------\n")


def run_generation(generator: Generator, prompt: str, config: SamplingConfig) -> None:
    """Run one generation and pretty-print the result."""
    print(f"\nPrompt    : {prompt!r}")
    print("Generating...", end=" ", flush=True)

    result = generator.generate(prompt, config)

    print(f"\rGenerated : {result.generated_text!r}")
    print(f"Full text : {result.full_text!r}")
    print(f"          [{result.prompt_tokens} prompt + {result.generated_tokens} new tokens, stopped: {result.stopped_by}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load model + tokenizer
    print(f"Loading model from: {args.checkpoint_dir}")
    try:
        generator = Generator.from_checkpoint(
            checkpoint_dir = args.checkpoint_dir,
            tokenizer_dir  = args.tokenizer_dir,
            device         = args.device,
        )
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("\nTip: Run  python scripts/train.py --demo  first to train a model.")
        sys.exit(1)

    model = generator.model
    print(f"Model     : {model.num_parameters:,} params, {model.config.n_layers} layers")
    print(f"Tokenizer : vocab_size={generator.tokenizer.vocab_size}")

    config = build_config(args)
    print_config(config)

    if args.interactive:
        print("Interactive mode. Type a prompt and press Enter. Ctrl-C to quit.\n")
        while True:
            try:
                prompt = input("Prompt> ").strip()
                if not prompt:
                    continue
                run_generation(generator, prompt, config)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
    else:
        run_generation(generator, args.prompt, config)


if __name__ == "__main__":
    main()
