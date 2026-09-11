"""
scripts/tokenizer/train_v2_tokenizer.py
=======================================
Trains three NexaByteTokenizer candidates (4096, 8192, 16384) on the pilot corpus.
"""

import time
import argparse
from pathlib import Path
import random

from nexa.tokenizer.byte_tokenizer import NexaByteTokenizer

RAW_CORPUS_DIR = Path("data/deduplicated/wikimedia_english")
OUTPUT_DIR = Path("data/processed/tokenizer_v2")

def load_documents(corpus_dir: Path):
    import json
    docs = []
    if not corpus_dir.exists():
        return docs
    for p in corpus_dir.rglob("*.jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        doc = json.loads(line)
                        docs.append(doc.get("text", ""))
                    except Exception:
                        pass
    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 8192, 16384], help="Vocab sizes to train")
    parser.add_argument("--sample_size", type=int, default=100000, help="Number of documents to sample for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    args = parser.parse_args()
    
    print(f"Loading corpus from {RAW_CORPUS_DIR}...")
    corpus = list(load_documents(RAW_CORPUS_DIR))
    
    # Deterministic sampling to keep training fast but representative
    random.seed(args.seed)
    if len(corpus) > args.sample_size:
        training_corpus = random.sample(corpus, args.sample_size)
    else:
        training_corpus = corpus
        
    print(f"Training on {len(training_corpus):,} documents.")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for vocab_size in args.sizes:
        print(f"\n--- Training Vocab Size: {vocab_size} ---")
        t0 = time.perf_counter()
        
        tokenizer = NexaByteTokenizer.train(training_corpus, vocab_size=vocab_size)
        
        duration = time.perf_counter() - t0
        print(f"Completed in {duration:.2f} seconds.")
        print(f"Final Vocab Size: {tokenizer.vocab_size} (Expected: {vocab_size})")
        
        save_path = OUTPUT_DIR / str(vocab_size)
        tokenizer.save(save_path)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
