"""
docs/concepts/phase2_bpe_tokenizer.md

BPE Tokenizer — Concept Reference
===================================
(See inline code docs in nexa/tokenizer/ for implementation details)

Quick reference card — algorithm invariants and design choices for Nexa's
BPE tokenizer. Read this alongside bpe.py and tokenizer.py.

Key invariants
--------------
1. Every word in the corpus is represented as:   characters + ["</w>"]
   e.g.  "cat"  →  ("c", "a", "t", "</w>")

2. Merge rules are ORDERED. Lower index = higher priority at inference.
   merge[0] was the most frequent pair in the training corpus.

3. Token IDs are deterministic and stable after training:
   0       <pad>
   1       <bos>
   2       <eos>
   3       <unk>
   4..N    base characters, sorted alphabetically
   N+1..   merged tokens, in order of merge rank

4. Decoding: join token strings, replace "</w>" → " ", strip trailing space.

5. OOV characters (not seen during training) → mapped to <unk> at encode time.

6. Independence guarantee: No pretrained tokenizer weights are loaded.
   The vocabulary and merge table come entirely from the corpus you provide.
"""
