# Nexa — Architecture Overview

> **Independence Principle**: Nexa uses no pretrained AI model weights.
> Every neural network in Nexa is architected and trained from scratch by us.

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                         User Interface                        │
│   CLI  ·  Gradio Chat UI  ·  FastAPI REST  ·  Speech I/O    │
└────────────────────────────┬─────────────────────────────────┘
                             │  text / audio
┌────────────────────────────▼─────────────────────────────────┐
│                       Inference Core                          │
│   generator.py — sampling loop with KV cache                 │
└────────────────────────────┬─────────────────────────────────┘
                             │  token IDs
┌────────────────────────────▼─────────────────────────────────┐
│                     NexaTransformer                           │
│   Embedding → N × TransformerBlock → LM Head → logits        │
│                                                               │
│   TransformerBlock:                                           │
│     RMSNorm → MultiHeadAttention (RoPE) → residual           │
│     RMSNorm → FeedForwardNetwork (SwiGLU) → residual         │
└────────────────────────────┬─────────────────────────────────┘
                             │  tokens
┌────────────────────────────▼─────────────────────────────────┐
│                      BPE Tokenizer                            │
│   Raw text → byte-pair encoding → integer token IDs          │
│   (Implemented from scratch, no SentencePiece weights)        │
└──────────────────────────────────────────────────────────────┘

Supporting Systems (attached to Inference Core):

  Memory System        Vector store (FAISS) + sliding-window context
  Tool System          Calculator, web search, code execution
  Speech I/O           STT (our acoustic model) + TTS (pyttsx3)
```

---

## Component Reference

### `nexa/tokenizer/` — Phase 2
| File | Role |
|------|------|
| `bpe.py` | Byte-Pair Encoding: counts character bigrams, iteratively merges most-frequent pairs |
| `vocab.py` | Saves/loads the merge table and vocab as JSON |

**Key concept**: BPE doesn't need a neural network — it's a pure frequency-counting algorithm on a text corpus. We train it ourselves.

---

### `nexa/models/` — Phase 3

| File | Role |
|------|------|
| `config.py` | `ModelConfig` dataclass — the single source of truth for architecture |
| `attention.py` | Multi-Head Self-Attention with optional Rotary Positional Embedding |
| `mlp.py` | Feed-Forward block (Linear → SwiGLU → Linear) |
| `transformer.py` | Assembles N decoder blocks + embedding + LM head |

**Key concept**: A Transformer is a stack of identical blocks. Each block has two operations: attend to context (attention), then process each token independently (MLP). Residual connections and normalization keep gradients stable.

---

### `nexa/training/` — Phase 4

| File | Role |
|------|------|
| `dataset.py` | Reads `.jsonl` text files, tokenizes on the fly, returns `(input_ids, target_ids)` pairs |
| `trainer.py` | Main loop: forward pass → cross-entropy loss → backprop → AdamW step → checkpoint |
| `scheduler.py` | Cosine LR schedule with linear warmup |

**Key concept**: Language modelling is self-supervised — we train the model to predict the next token. No labels needed; the input text IS the label (shifted by one position).

---

### `nexa/inference/` — Phase 5

| File | Role |
|------|------|
| `generator.py` | Autoregressive loop: feed prompt → sample next token → append → repeat until EOS |

**Key concept**: At each step the model sees ALL tokens so far and outputs a probability distribution. We sample from that distribution using temperature + top-k/p. The KV cache saves us from recomputing attention for already-seen tokens.

---

### `nexa/memory/` — Phase 7

| File | Role |
|------|------|
| `context.py` | Maintains a rolling window of recent tokens (short-term) |
| `vector_store.py` | Stores past conversation summaries as dense vectors in FAISS (long-term) |

---

### `nexa/speech/` — Future Research Phase

Speech is treated as a separate research phase, not part of the core roadmap.
When it begins, we will design and train our own acoustic model from scratch.

| Constraint | Detail |
|------------|--------|
| ❌ Forbidden | Whisper, Vosk, DeepSpeech, SpeechBrain, pyttsx3, Coqui TTS |
| ✅ Allowed | sounddevice (audio I/O), scipy/numpy (signal processing), PyTorch (our own acoustic model) |
| Data | Open-licensed speech corpora (LibriSpeech — CC-BY 4.0) |

---

## Data Flow: Chat Interaction

```
1. User speaks           → sounddevice captures audio
2. STT                   → acoustic model converts to text
3. Memory retrieval      → FAISS fetches relevant past context
4. Prompt assembly       → system prompt + memory + history + user message
5. Tokenization          → BPE encodes prompt to token IDs
6. Model forward pass    → NexaTransformer produces logits
7. Sampling              → generator samples next tokens one at a time
8. Detokenization        → BPE decodes token IDs back to text
9. Memory update         → new turn stored in FAISS
10. TTS                  → pyttsx3 speaks the response
11. Display              → Gradio UI shows the conversation
```

---

## Training Data Policy

All training data must be:
- ✅ Public domain (e.g. Project Gutenberg, Wikipedia dumps)
- ✅ Open-licensed for ML training (e.g. The Pile subsets with permissive licenses)
- ✅ CC0 or CC-BY licensed text corpora
- ❌ Not scraped without permission
- ❌ Not behind a paywall or proprietary license
