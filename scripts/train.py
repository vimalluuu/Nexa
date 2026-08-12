"""
scripts/train.py
=================
CLI entry point: train Nexa's language model from scratch.

Full pipeline:
    1. Load (or train) the BPE tokenizer on your corpus.
    2. Tokenize the corpus → flat token ID sequence.
    3. Split into train / validation.
    4. Build NexaTransformer from ModelConfig.
    5. Run Trainer.train().
    6. Save the final model weights.

Usage
-----
    # Quick demo (tiny model, tiny corpus, trains in ~30s on CPU):
    python scripts/train.py --demo

    # Full training on custom corpus:
    python scripts/train.py \\
        --corpus   data/raw/my_dataset/tiny_train.txt \\
        --tokenizer data/processed/tokenizer \\
        --output   checkpoints/ \\
        --max-steps 1000 \\
        --batch-size 4

Independence
------------
No pretrained weights, no external AI APIs, no internet access required.
All weights are learned from the corpus you supply.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from nexa.models   import NexaTransformer, ModelConfig
from nexa.tokenizer import NexaTokenizer
from nexa.training  import Trainer, TrainerConfig, NexaDataset
from nexa.utils     import get_logger

log = get_logger("nexa.train", log_to_file=False)


# ---------------------------------------------------------------------------
# Demo config — tiny model + tiny corpus, trains on CPU in ~30 seconds
# ---------------------------------------------------------------------------

DEMO_MODEL_CFG = dict(
    d_model      = 128,
    n_heads      = 4,
    n_layers     = 3,
    d_ff         = 512,
    dropout      = 0.1,
    norm_type    = "rmsnorm",
    pos_encoding = "rotary",
    max_seq_len  = 128,
    tie_embeddings = True,
)

DEMO_TRAINER_CFG = dict(
    block_size       = 32,
    batch_size       = 4,
    grad_accum_steps = 2,
    learning_rate    = 5e-4,
    min_lr           = 5e-5,
    weight_decay     = 0.1,
    grad_clip        = 1.0,
    max_steps        = 300,
    warmup_steps     = 30,
    eval_interval    = 100,
    eval_steps       = 20,
    save_interval    = 300,
    log_interval     = 25,
    device           = "auto",
    seed             = 42,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_or_train_tokenizer(
    tokenizer_dir: Path,
    corpus_path:   Path,
    vocab_size:    int,
) -> NexaTokenizer:
    """Load tokenizer from disk, or train it fresh on the corpus."""
    if (tokenizer_dir / "tokenizer.json").exists():
        log.info("Loading existing tokenizer from %s", tokenizer_dir)
        return NexaTokenizer.load(tokenizer_dir)

    log.info("Training tokenizer on %s (vocab_size=%d) ...", corpus_path, vocab_size)
    corpus = [
        line.strip()
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = NexaTokenizer.train(corpus, vocab_size=vocab_size, min_frequency=1)
    tokenizer.save(tokenizer_dir)
    log.info("Tokenizer saved → %s  (vocab_size=%d)", tokenizer_dir, tokenizer.vocab_size)
    return tokenizer


def build_datasets(
    tokenizer:    NexaTokenizer,
    corpus_path:  Path,
    block_size:   int,
    val_fraction: float = 0.1,
) -> tuple[NexaDataset, Optional[NexaDataset]]:
    """Tokenize corpus and split into train/val."""
    text = corpus_path.read_text(encoding="utf-8")
    full_dataset = NexaDataset.from_text(
        text, tokenizer, block_size=block_size, add_bos=True, add_eos=True
    )
    token_ids = full_dataset.data.tolist()
    log.info("Corpus: %d tokens → %d training windows", len(token_ids), len(full_dataset))

    if len(token_ids) < block_size + 2:
        log.warning("Corpus very short (%d tokens). No validation split.", len(token_ids))
        return full_dataset, None

    train_ds, val_ds = NexaDataset.train_val_split(
        token_ids, block_size=block_size, val_fraction=val_fraction
    )
    log.info("Split: train=%d windows, val=%d windows", len(train_ds), len(val_ds))
    return train_ds, val_ds


def print_model_info(model: NexaTransformer) -> None:
    cfg = model.config
    print(f"\n{'='*60}")
    print(f"  NexaTransformer — built from scratch")
    print(f"{'='*60}")
    print(f"  Architecture   : {cfg.n_layers} layers x {cfg.n_heads} heads x d={cfg.d_model}")
    print(f"  FFN dim        : {cfg.d_ff}  (SwiGLU)")
    print(f"  Norm           : {cfg.norm_type.upper()}")
    print(f"  Pos encoding   : {cfg.pos_encoding}")
    print(f"  Vocab size     : {cfg.vocab_size:,}")
    print(f"  Parameters     : {model.num_parameters:,}")
    print(f"  Weight tying   : {cfg.tie_embeddings}")
    print(f"{'='*60}\n")


def print_training_summary(history, total_seconds: float) -> None:
    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    if history.train_loss:
        init_loss  = history.initial_train_loss
        final_loss = history.final_train_loss
        reduction  = (init_loss - final_loss) / init_loss * 100
        print(f"  Initial train loss  : {init_loss:.4f}")
        print(f"  Final   train loss  : {final_loss:.4f}")
        print(f"  Loss reduction      : {reduction:.1f}%")
    if history.val_loss:
        print(f"  Final   val   loss  : {history.final_val_loss:.4f}")
    print(f"  Training time       : {total_seconds:.1f}s")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Nexa's language model from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Run a quick demo (tiny model + tiny corpus, ~30s on CPU).",
    )
    p.add_argument(
        "--corpus", type=Path, default=Path("data/raw/nexa_demo/tiny_train.txt"),
        help="Path to the training corpus (plain text, UTF-8).",
    )
    p.add_argument(
        "--tokenizer", type=Path, default=Path("data/processed/tokenizer"),
        help="Tokenizer directory (will be trained if absent).",
    )
    p.add_argument(
        "--output", type=Path, default=Path("checkpoints"),
        help="Directory for saving model checkpoints.",
    )
    p.add_argument("--vocab-size",    type=int,   default=500,  help="BPE vocab size.")
    p.add_argument("--max-steps",     type=int,   default=500,  help="Optimizer steps.")
    p.add_argument("--batch-size",    type=int,   default=4,    help="Mini-batch size.")
    p.add_argument("--block-size",    type=int,   default=32,   help="Sequence length.")
    p.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate.")
    p.add_argument("--resume",        type=Path,  default=None, help="Resume from checkpoint.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import time
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Load / train tokenizer
    # ------------------------------------------------------------------
    tokenizer = load_or_train_tokenizer(
        tokenizer_dir = args.tokenizer,
        corpus_path   = args.corpus,
        vocab_size    = args.vocab_size,
    )

    # ------------------------------------------------------------------
    # Step 2: Build ModelConfig
    # ------------------------------------------------------------------
    if args.demo:
        model_kwargs = {**DEMO_MODEL_CFG, "vocab_size": tokenizer.vocab_size}
        trainer_kwargs = {**DEMO_TRAINER_CFG, "checkpoint_dir": args.output}
        log.info("Demo mode: tiny model, 300 steps, ~30s on CPU")
    else:
        model_kwargs = {
            **DEMO_MODEL_CFG,
            "vocab_size"  : tokenizer.vocab_size,
        }
        trainer_kwargs = {
            "block_size"     : args.block_size,
            "batch_size"     : args.batch_size,
            "learning_rate"  : args.learning_rate,
            "min_lr"         : args.learning_rate / 10,
            "max_steps"      : args.max_steps,
            "warmup_steps"   : max(10, args.max_steps // 10),
            "checkpoint_dir" : args.output,
            "log_interval"   : 25,
            "eval_interval"  : 100,
            "save_interval"  : args.max_steps,
        }

    model_cfg   = ModelConfig(**model_kwargs)
    trainer_cfg = TrainerConfig(**trainer_kwargs)

    # ------------------------------------------------------------------
    # Step 3: Build datasets
    # ------------------------------------------------------------------
    train_ds, val_ds = build_datasets(
        tokenizer,
        args.corpus,
        block_size=trainer_cfg.block_size,
    )

    # ------------------------------------------------------------------
    # Step 4: Build model
    # ------------------------------------------------------------------
    model = NexaTransformer(model_cfg)
    print_model_info(model)

    # ------------------------------------------------------------------
    # Step 5: Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model         = model,
        config        = trainer_cfg,
        train_dataset = train_ds,
        val_dataset   = val_ds,
        resume_from   = args.resume,
    )

    log.info("Starting training: %d steps, effective batch=%d",
             trainer_cfg.max_steps, trainer_cfg.effective_batch_size)

    history = trainer.train()

    # ------------------------------------------------------------------
    # Step 6: Save final weights
    # ------------------------------------------------------------------
    final_weights = args.output / "nexa_final.pt"
    model.save_weights(final_weights)
    log.info("Final weights saved → %s", final_weights)

    # Also save ModelConfig as JSON so Generator.from_checkpoint() can reload
    import dataclasses, json as _json
    config_json_path = args.output / "model_config.json"
    config_json_path.write_text(
        _json.dumps(dataclasses.asdict(model_cfg), indent=2), encoding="utf-8"
    )
    log.info("Model config saved → %s", config_json_path)


    t_end = time.perf_counter()
    print_training_summary(history, t_end - t_start)

    # Quick generation demo
    model.eval()
    prompt = "the cat sat"
    ids    = torch.tensor(
        [tokenizer.encode(prompt, add_bos=True)], dtype=torch.long
    )
    generated = model.generate(ids, max_new_tokens=20, temperature=0.8, greedy=False)
    tokens = [tokenizer.vocab.id_to_token(i) for i in generated[0].tolist()]
    text   = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    print(f"\nGeneration demo:")
    print(f"  Prompt   : {prompt!r}")
    print(f"  Generated: {text!r}")
    print(f"  Tokens   : {tokens}")


if __name__ == "__main__":
    main()
