"""
nexa/tokenizer/byte_bpe.py
===========================
Optimized Pure Python Byte-Level BPE.
Maintains pair statistics incrementally for fast O(N) merge loops.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

Pair = tuple[bytes, bytes]
ByteTokens = list[bytes]

def text_to_byte_tokens(text: str) -> ByteTokens:
    if not text:
        return []
    return [bytes([b]) for b in text.encode("utf-8")]

def train_byte_bpe(corpus: list[str], num_merges: int) -> list[Pair]:
    """
    Highly optimized pure-python BPE training.
    Instead of recreating the stats dict on every iteration, we maintain:
    1. word_freqs: mapping of word (as list of tokens) to its frequency in corpus.
    2. pair_stats: mapping of adjacent token pairs to their total frequency.
    """
    pattern = re.compile(r'\s*\S+|\s+')
    
    # 1. Build initial words and frequencies
    # We use tuples internally so they can be hashed in word_freqs, but lists inside the values for mutability.
    word_freqs_raw = defaultdict(int)
    for doc in corpus:
        words = pattern.findall(doc)
        for w in words:
            word_freqs_raw[tuple(text_to_byte_tokens(w))] += 1
            
    # We need mutable lists to apply merges in-place for speed.
    # We'll map an index to each unique word to track them.
    words = []
    word_counts = []
    for w_tuple, count in word_freqs_raw.items():
        if len(w_tuple) > 1: # Only care about words with at least 2 tokens
            words.append(list(w_tuple))
            word_counts.append(count)
            
    # 2. Build initial pair stats and reverse index (pair -> list of word indices)
    pair_stats = defaultdict(int)
    pair_to_words = defaultdict(set)
    
    for wid, word in enumerate(words):
        count = word_counts[wid]
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_stats[pair] += count
            pair_to_words[pair].add(wid)
            
    merges = []
    
    for _ in range(num_merges):
        if not pair_stats:
            break
            
        # Find the best pair
        best_pair = max(pair_stats, key=pair_stats.get)
        if pair_stats[best_pair] < 1:
            break
            
        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        
        # Apply merge to all words containing the best_pair
        affected_wids = list(pair_to_words[best_pair])
        
        # Zero out the old pair in stats
        del pair_stats[best_pair]
        del pair_to_words[best_pair]
        
        for wid in affected_wids:
            word = words[wid]
            count = word_counts[wid]
            
            # Remove all old pairs for this word from pair_stats
            for i in range(len(word) - 1):
                p = (word[i], word[i+1])
                if p != best_pair: # best_pair is already deleted
                    pair_stats[p] -= count
                    if pair_stats[p] <= 0:
                        del pair_stats[p]
                        pair_to_words[p].discard(wid)
            
            # Perform the merge on the word
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[wid] = new_word
            
            # Add all new pairs for this word to pair_stats
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i+1])
                pair_stats[p] += count
                pair_to_words[p].add(wid)
                
    return merges

def merge_tokens(tokens: tuple[bytes, ...], pair: Pair) -> tuple[bytes, ...]:
    if len(tokens) < 2:
        return tokens
        
    merged_token = pair[0] + pair[1]
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(merged_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return tuple(new_tokens)
