"""
scripts/data/tokenize_v2_corpus.py
===================================
Tokenizes the deduplicated V2 corpus using the Phase 11 NexaByteTokenizer (8192).
Splits deterministically (90/10) and creates binary shards.
"""

import json
import random
import struct
from pathlib import Path
from collections import defaultdict
import sys

# Ensure nexa is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexa.tokenizer.byte_tokenizer import NexaByteTokenizer
from nexa.tokenizer.vocab import UNK_ID

CORPUS_JSONL = Path("data/processed/v2_corpus/v2_corpus.jsonl")
TOKENIZER_DIR = Path("data/processed/tokenizer_v2/8192")
OUT_DIR = Path("data/processed/v2")
REPORTS_DIR = Path("data/reports")

# Config
SEED = 42
TRAIN_RATIO = 0.90
SHARD_SIZE_TOKENS = 10_000_000

def generate_shards(split_name: str, docs: list[dict], tokenizer: NexaByteTokenizer):
    out_dir = OUT_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    current_shard = 0
    current_tokens = []
    
    stats = {
        "docs": 0,
        "total_tokens": 0,
        "total_chars": 0,
        "total_unks": 0,
        "shards_created": 0,
        "sources": defaultdict(int)
    }
    
    def flush_shard():
        nonlocal current_shard, current_tokens
        if not current_tokens: return
        shard_path = out_dir / f"shard_{current_shard:03d}.bin"
        with open(shard_path, "wb") as f:
            for token_id in current_tokens:
                f.write(struct.pack("<H", token_id)) # UInt16
        current_shard += 1
        stats["shards_created"] += 1
        current_tokens = []
        
    for doc in docs:
        text = doc["text"]
        source = doc.get("source", "unknown")
        
        # Tokenize (implicitly handles BOS/EOS if configured, but here we just encode text, we add BOS/EOS per document)
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        
        stats["docs"] += 1
        stats["total_tokens"] += len(ids)
        stats["total_chars"] += len(text)
        stats["total_unks"] += ids.count(UNK_ID)
        stats["sources"][source] += 1
        
        current_tokens.extend(ids)
        
        while len(current_tokens) >= SHARD_SIZE_TOKENS:
            # We keep the exact SHARD_SIZE_TOKENS and carry over the rest
            spillover = current_tokens[SHARD_SIZE_TOKENS:]
            current_tokens = current_tokens[:SHARD_SIZE_TOKENS]
            flush_shard()
            current_tokens = spillover
            
    # Flush remaining
    flush_shard()
    return stats

def main():
    if not CORPUS_JSONL.exists():
        print(f"Error: {CORPUS_JSONL} not found.")
        sys.exit(1)
        
    if not TOKENIZER_DIR.exists():
        print(f"Error: Tokenizer not found at {TOKENIZER_DIR}")
        sys.exit(1)
        
    print("Loading tokenizer...")
    tokenizer = NexaByteTokenizer.load(TOKENIZER_DIR)
    
    print("Loading documents...")
    docs = []
    with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
            
    # Deterministic split
    print("Splitting corpus (90/10)...")
    random.seed(SEED)
    
    # Sort docs by an exact text hash to ensure deterministic ordering regardless of filesystem reading order
    import hashlib
    docs.sort(key=lambda d: hashlib.sha256(d["text"].encode("utf-8")).hexdigest())
    
    random.shuffle(docs)
    
    split_idx = int(len(docs) * TRAIN_RATIO)
    train_docs = docs[:split_idx]
    val_docs = docs[split_idx:]
    
    print(f"Train documents: {len(train_docs)}")
    print(f"Validation documents: {len(val_docs)}")
    
    print("Generating Train Shards...")
    train_stats = generate_shards("train", train_docs, tokenizer)
    
    print("Generating Validation Shards...")
    val_stats = generate_shards("validation", val_docs, tokenizer)
    
    # Final check for token conservation
    total_generated = train_stats["total_tokens"] + val_stats["total_tokens"]
    
    report = {
        "tokenizer": {
            "version": "NexaByteTokenizer",
            "vocab_size": tokenizer.vocab_size
        },
        "split": {
            "seed": SEED,
            "train_ratio": TRAIN_RATIO
        },
        "train": train_stats,
        "validation": val_stats,
        "total_tokens_conserved": total_generated
    }
    
    out_report = REPORTS_DIR / "v2_corpus_tokenization_stats.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Complete! Tokenized {total_generated:,} total tokens.")

if __name__ == "__main__":
    main()
