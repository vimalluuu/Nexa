"""
nexa/tokenizer/byte_tokenizer.py
=================================
True Byte-Level BPE Tokenizer for Nexa V2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import re

from nexa.tokenizer.vocab import PAD_ID, BOS_ID, EOS_ID, UNK_ID
from nexa.tokenizer.byte_bpe import Pair, train_byte_bpe, merge_tokens, text_to_byte_tokens

# Special tokens fixed at 0-3
SPECIAL_TOKENS = {
    "<pad>": PAD_ID,
    "<bos>": BOS_ID,
    "<eos>": EOS_ID,
    "<unk>": UNK_ID
}

class NexaByteTokenizer:
    """
    True Byte-Level BPE Tokenizer.
    IDs 0-3: Special tokens
    IDs 4-259: The 256 fundamental byte values (0x00 - 0xFF)
    IDs 260+: Learned BPE merges
    """
    
    def __init__(self, merges: list[Pair], vocab: dict[bytes, int]) -> None:
        self.merges = merges
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        self._merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self._pattern = re.compile(r'\s*\S+|\s+')
        
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(SPECIAL_TOKENS)

    @classmethod
    def train(cls, corpus: list[str], vocab_size: int = 4096) -> NexaByteTokenizer:
        # Base vocabulary is all 256 bytes.
        # Plus 4 special tokens.
        # So we can do `vocab_size - 260` merges.
        num_merges = vocab_size - 260
        if num_merges < 0:
            raise ValueError(f"vocab_size must be >= 260. Got {vocab_size}")
            
        print(f"Training Byte-Level BPE for {num_merges} merges...")
        merges = train_byte_bpe(corpus, num_merges)
        
        # Construct Vocabulary
        vocab: dict[bytes, int] = {}
        
        # 1. Add base bytes (mapped to IDs 4..259)
        for b in range(256):
            vocab[bytes([b])] = b + 4
            
        # 2. Add learned merges (mapped to 260+)
        next_id = 260
        for p1, p2 in merges:
            merged_token = p1 + p2
            if merged_token not in vocab:
                vocab[merged_token] = next_id
                next_id += 1
                
        return cls(merges, vocab)

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert bytes to hex strings for JSON serialization
        hex_merges = [[p[0].hex(), p[1].hex()] for p in self.merges]
        hex_vocab = {k.hex(): v for k, v in self.vocab.items()}
        
        with open(save_dir / "merges.json", "w", encoding="utf-8") as f:
            json.dump(hex_merges, f, indent=2)
            
        with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(hex_vocab, f, indent=2)

    @classmethod
    def load(cls, load_dir: str | Path) -> NexaByteTokenizer:
        load_dir = Path(load_dir)
        
        with open(load_dir / "merges.json", "r", encoding="utf-8") as f:
            hex_merges = json.load(f)
            
        with open(load_dir / "vocab.json", "r", encoding="utf-8") as f:
            hex_vocab = json.load(f)
            
        merges = [(bytes.fromhex(p1), bytes.fromhex(p2)) for p1, p2 in hex_merges]
        vocab = {bytes.fromhex(k): v for k, v in hex_vocab.items()}
        
        return cls(merges, vocab)

    def _encode_word(self, word: str) -> list[int]:
        tokens = text_to_byte_tokens(word)
        
        while len(tokens) >= 2:
            # Find the best valid pair among adjacent tokens
            best_pair = None
            best_rank = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self._merge_rank and self._merge_rank[pair] < best_rank:
                    best_pair = pair
                    best_rank = self._merge_rank[pair]
                    
            if best_pair is None:
                break
                
            tokens = merge_tokens(tokens, best_pair)
            
        # Map back to IDs. If somehow a token isn't in vocab (impossible in byte BPE unless bug), map to UNK
        return [self.vocab.get(tok, UNK_ID) for tok in tokens]

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(BOS_ID)
            
        words = self._pattern.findall(text)
        for word in words:
            ids.extend(self._encode_word(word))
            
        if add_eos:
            ids.append(EOS_ID)
            
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        b_array = bytearray()
        
        for idx in ids:
            if idx in SPECIAL_TOKENS.values():
                if not skip_special_tokens:
                    # In a real decoder, you might yield '<pad>' string, but since this operates on bytes, 
                    # there is no perfect way to mix string tags and bytes unless we return bytes and decode later.
                    # We'll just ignore them to prevent UnicodeDecodeError.
                    pass
                continue
                
            token_bytes = self.inv_vocab.get(idx)
            if token_bytes is not None:
                b_array.extend(token_bytes)
                
        # Handle surrogate pairs and decode errors gracefully
        return b_array.decode("utf-8", errors="replace")
