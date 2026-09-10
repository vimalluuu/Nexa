"""
Nexa Phase 8.6 - Dataset Tokenization and Sharding
===================================================
Tokenizes the pilot corpus and writes it to sequential binary shards.

Pipeline:
1. Load Phase 8 pilot index and randomly split source_ids into train/validation (deterministic).
2. Load the trained NexaTokenizer (vocab_size=2048).
3. Stream documents from deduplicated JSONL files.
4. Tokenize each document, explicitly wrapping it with <bos> and <eos>.
5. Write token IDs continuously to `uint16` binary shard files.
   - When a shard reaches `max_tokens_per_shard`, roll over to a new file.
6. Generate manifests, checksums, and a final statistical report.

No entire corpus is loaded into RAM. Token IDs are verified to be < vocab_size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from nexa.tokenizer.tokenizer import NexaTokenizer
from nexa.tokenizer.vocab import BOS_ID, EOS_ID
from scripts.tokenizer.corpus_sampler import load_pilot_index, stream_pilot_texts

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PILOT_INDEX_PATH = Path("data/deduplicated/pilot_candidate_index.jsonl")
TOKENIZER_DIR    = Path("data/processed/tokenizer_v1")
OUTPUT_DIR       = Path("data/processed")

MANIFESTS_DIR    = Path("data/manifests")
REPORTS_DIR      = Path("data/reports")

# 10 million tokens per shard ~= 20 MB file size
# This allows fast sequential loading and easy shuffling at shard level.
DEFAULT_TOKENS_PER_SHARD = 10_000_000
DEFAULT_SEED             = 42
DEFAULT_VAL_FRACTION     = 0.10


# ---------------------------------------------------------------------------
# Shard Writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """
    Writes a continuous stream of integer token IDs into sequential binary files.
    Format: raw arrays of uint16 (little-endian).
    """

    def __init__(self, out_dir: Path, prefix: str, max_tokens: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.max_tokens = max_tokens
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.shard_idx = 0
        self.current_tokens = 0
        self.current_file = None
        self.current_hasher = hashlib.sha256()
        
        self.shards: list[dict] = []
        self._open_next_shard()

    def _open_next_shard(self):
        if self.current_file:
            self._close_current_shard()
            
        filename = f"{self.prefix}_{self.shard_idx:03d}.bin"
        self.current_path = self.out_dir / filename
        self.current_file = open(self.current_path, "wb")
        self.current_tokens = 0
        self.current_hasher = hashlib.sha256()

    def _close_current_shard(self):
        if self.current_file:
            self.current_file.close()
            size_bytes = self.current_path.stat().st_size
            checksum = self.current_hasher.hexdigest()
            self.shards.append({
                "filename": self.current_path.name,
                "tokens": self.current_tokens,
                "bytes": size_bytes,
                "sha256": checksum,
            })
            log.info("Wrote shard %s | %d tokens | %.2f MB",
                     self.current_path.name, self.current_tokens, size_bytes / 1024**2)
            self.shard_idx += 1
            self.current_file = None

    def write(self, tokens: list[int]):
        """Write tokens to binary, rolling over if necessary."""
        remaining = len(tokens)
        idx = 0
        
        while remaining > 0:
            space_left = self.max_tokens - self.current_tokens
            if space_left <= 0:
                self._open_next_shard()
                space_left = self.max_tokens
                
            chunk_size = min(remaining, space_left)
            chunk = tokens[idx : idx + chunk_size]
            
            # Pack as little-endian unsigned short (uint16)
            data = struct.pack(f"<{len(chunk)}H", *chunk)
            self.current_file.write(data)
            self.current_hasher.update(data)
            
            self.current_tokens += chunk_size
            remaining -= chunk_size
            idx += chunk_size

    def close(self):
        """Close the final shard."""
        self._close_current_shard()


# ---------------------------------------------------------------------------
# Pipeline logic
# ---------------------------------------------------------------------------

def deterministic_split(
    pilot_index: dict[str, set[str]],
    val_fraction: float,
    seed: int
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    Split the document source_ids into train and validation sets.
    
    Document-level split guarantees no overlap.
    A fixed seed guarantees reproducibility.
    """
    rng = random.Random(seed)
    train_index: dict[str, set[str]] = {}
    val_index: dict[str, set[str]] = {}
    
    for dataset, source_ids in pilot_index.items():
        sorted_ids = sorted(list(source_ids))
        rng.shuffle(sorted_ids)
        
        n_val = max(1, int(len(sorted_ids) * val_fraction))
        val_ids = set(sorted_ids[:n_val])
        train_ids = set(sorted_ids[n_val:])
        
        val_index[dataset] = val_ids
        train_index[dataset] = train_ids
        
    return train_index, val_index


def tokenize_and_shard(
    tokenizer: NexaTokenizer,
    pilot_index: dict[str, set[str]],
    output_dir: Path,
    val_fraction: float,
    seed: int,
    max_tokens_per_shard: int
) -> dict:
    """Run the complete tokenization and sharding pipeline."""
    
    train_idx, val_idx = deterministic_split(pilot_index, val_fraction, seed)
    
    train_dir = output_dir / "train"
    val_dir   = output_dir / "validation"
    
    train_writer = ShardWriter(train_dir, "shard", max_tokens_per_shard)
    val_writer   = ShardWriter(val_dir, "shard", max_tokens_per_shard)
    
    stats = {
        "total_documents": 0, "train_documents": 0, "val_documents": 0,
        "total_tokens": 0,    "train_tokens": 0,    "val_tokens": 0,
        "vocab_size": tokenizer.vocab_size,
    }
    
    # Track sequence lengths for reporting
    seq_lengths = []
    
    log.info("Tokenizing validation set...")
    for text in stream_pilot_texts(val_idx):
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if not ids:
            continue
        if any(t >= tokenizer.vocab_size or t < 0 for t in ids):
            raise ValueError("Token ID out of bounds! Ensure vocab_size is correct.")
            
        val_writer.write(ids)
        
        stats["val_documents"] += 1
        stats["val_tokens"] += len(ids)
        seq_lengths.append(len(ids))
        
    val_writer.close()
    log.info("Validation tokenization complete.")
    
    log.info("Tokenizing training set...")
    for text in stream_pilot_texts(train_idx):
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if not ids:
            continue
        if any(t >= tokenizer.vocab_size or t < 0 for t in ids):
            raise ValueError("Token ID out of bounds! Ensure vocab_size is correct.")
            
        train_writer.write(ids)
        
        stats["train_documents"] += 1
        stats["train_tokens"] += len(ids)
        seq_lengths.append(len(ids))
        
        if stats["train_documents"] % 500 == 0:
            log.info("  Processed %d train docs...", stats["train_documents"])
            
    train_writer.close()
    log.info("Training tokenization complete.")
    
    stats["total_documents"] = stats["train_documents"] + stats["val_documents"]
    stats["total_tokens"]    = stats["train_tokens"] + stats["val_tokens"]
    
    if seq_lengths:
        seq_lengths.sort()
        stats["avg_tokens_per_doc"] = round(sum(seq_lengths) / len(seq_lengths), 1)
        stats["median_tokens_per_doc"] = seq_lengths[len(seq_lengths) // 2]
        stats["p90_tokens_per_doc"] = seq_lengths[int(len(seq_lengths) * 0.90)]
        stats["p95_tokens_per_doc"] = seq_lengths[int(len(seq_lengths) * 0.95)]
        stats["p99_tokens_per_doc"] = seq_lengths[int(len(seq_lengths) * 0.99)]
    
    # Compile Manifest
    manifest = {
        "phase": "8.6",
        "generated": datetime.now(timezone.utc).isoformat(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "seed": seed,
        "val_fraction": val_fraction,
        "format": "uint16 little-endian binary",
        "bos_eos_included": True,
        "shards": {
            "train": train_writer.shards,
            "validation": val_writer.shards,
        }
    }
    
    # Compile checksums
    checksums = []
    for s in train_writer.shards:
        checksums.append(f"{s['sha256']}  train/{s['filename']}")
    for s in val_writer.shards:
        checksums.append(f"{s['sha256']}  validation/{s['filename']}")
        
    return {
        "stats": stats,
        "manifest": manifest,
        "checksums": "\n".join(checksums) + "\n"
    }


def main():
    parser = argparse.ArgumentParser(description="Nexa Phase 8.6 — Dataset Tokenization and Sharding")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_TOKENS_PER_SHARD)
    args = parser.parse_args()
    
    log.info("=== Nexa Phase 8.6 Dataset Sharding ===")
    
    if not TOKENIZER_DIR.exists():
        log.error("Tokenizer not found at %s. Run Phase 8.5 first.", TOKENIZER_DIR)
        sys.exit(1)
        
    tokenizer = NexaTokenizer.load(TOKENIZER_DIR)
    log.info("Loaded NexaTokenizer (vocab_size=%d)", tokenizer.vocab_size)
    
    pilot_index = load_pilot_index(PILOT_INDEX_PATH)
    
    t0 = time.time()
    reports = tokenize_and_shard(
        tokenizer=tokenizer,
        pilot_index=pilot_index,
        output_dir=OUTPUT_DIR,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_tokens_per_shard=args.max_tokens
    )
    elapsed = time.time() - t0
    
    # Write reports
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(MANIFESTS_DIR / "tokenized_dataset_v1.json", "w", encoding="utf-8") as f:
        json.dump(reports["manifest"], f, indent=2, ensure_ascii=False)
        
    with open(MANIFESTS_DIR / "tokenized_checksums.sha256", "w", encoding="utf-8") as f:
        f.write(reports["checksums"])
        
    report = reports["stats"]
    report["elapsed_s"] = round(elapsed, 1)
    
    total_mb = sum(s["bytes"] for s in reports["manifest"]["shards"]["train"] + reports["manifest"]["shards"]["validation"]) / 1024**2
    report["total_storage_mb"] = round(total_mb, 2)
    report["shard_count"] = len(reports["manifest"]["shards"]["train"]) + len(reports["manifest"]["shards"]["validation"])
    
    with open(REPORTS_DIR / "tokenized_dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    log.info("=== Pipeline Complete ===")
    log.info("Total tokens: %d | Docs: %d", report["total_tokens"], report["total_documents"])
    log.info("Train tokens: %d | Val tokens: %d", report["train_tokens"], report["val_tokens"])
    log.info("Storage: %.2f MB | Shards: %d", report["total_storage_mb"], report["shard_count"])
    log.info("Manifests and checksums written to %s", MANIFESTS_DIR)

if __name__ == "__main__":
    main()
