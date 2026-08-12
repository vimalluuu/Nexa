"""
scripts/train_tokenizer.py
===========================
Train Nexa's BPE tokenizer on a corpus and save it.

Usage
-----
    # Train on sample corpus (default):
    python scripts/train_tokenizer.py

    # Train on your own corpus with a custom vocab size:
    python scripts/train_tokenizer.py \\
        --corpus data/raw/my_corpus.txt \\
        --output data/processed/tokenizer \\
        --vocab-size 2000 \\
        --min-frequency 2

After training, the tokenizer is saved to:
    data/processed/tokenizer/tokenizer.json  — merge rules
    data/processed/tokenizer/vocab.json      — token ↔ ID mapping

Independence
------------
This script downloads NO external data and loads NO pretrained models.
All vocabulary and merge rules are learned from the corpus you provide.
"""

import argparse
import sys
from pathlib import Path

# Ensure we can import nexa from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexa.tokenizer import NexaTokenizer
from nexa.utils import get_logger

log = get_logger("nexa.train_tokenizer", log_to_file=False)


def load_corpus(path: Path) -> list[str]:
    """Read a text file and return non-empty lines as a list of strings."""
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def show_sample_encodings(tokenizer: NexaTokenizer, texts: list[str]) -> None:
    """Print a few sample encode/decode pairs to verify the tokenizer."""
    print("\n" + "=" * 60)
    print("Sample tokenizations:")
    print("=" * 60)
    samples = texts[:5] if len(texts) >= 5 else texts
    for text in samples:
        ids = tokenizer.encode(text)
        tokens = [tokenizer.vocab.id_to_token(i) for i in ids]
        decoded = tokenizer.decode(ids)
        print(f"\n  Input  : {text!r}")
        print(f"  Tokens : {tokens}")
        print(f"  IDs    : {ids}")
        print(f"  Decoded: {decoded!r}")
    print("=" * 60)


def show_top_merges(tokenizer: NexaTokenizer, n: int = 20) -> None:
    """Print the first N merge rules learned by BPE."""
    print(f"\nTop {n} BPE merges learned:")
    for i, (a, b) in enumerate(tokenizer.merges[:n]):
        merged = a + b
        print(f"  [{i:3d}]  ({a!r:10} + {b!r:10}) -> {merged!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Nexa's BPE tokenizer from scratch on a text corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/raw/sample_corpus.txt"),
        help="Path to training corpus (plain text, one document per line).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tokenizer"),
        help="Directory to save the trained tokenizer.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Target vocabulary size (includes 4 special tokens).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum pair frequency for a merge to be considered.",
    )
    parser.add_argument(
        "--show-merges",
        type=int,
        default=20,
        metavar="N",
        help="Print the top N merge rules after training.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    if not args.corpus.exists():
        log.error("Corpus file not found: %s", args.corpus)
        sys.exit(1)

    corpus = load_corpus(args.corpus)
    log.info("Loaded corpus: %d lines from %s", len(corpus), args.corpus)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    tokenizer = NexaTokenizer.train(
        corpus=corpus,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{tokenizer}")
    print(f"  Special tokens : <pad>=0, <bos>=1, <eos>=2, <unk>=3")
    print(f"  Merges learned : {len(tokenizer.merges)}")
    print(f"  Final vocab    : {tokenizer.vocab_size} tokens")

    show_top_merges(tokenizer, n=args.show_merges)
    show_sample_encodings(tokenizer, corpus)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    tokenizer.save(args.output)
    log.info("Tokenizer saved to: %s", args.output)
    print(f"\n[OK] Saved to {args.output}/")


if __name__ == "__main__":
    main()
