# Nexa — Independent AI System

> An end-to-end AI assistant built from scratch using Python and PyTorch.
> No proprietary APIs. No pretrained weights. Every component is built, trained, and owned by us.

---

## Core Independence Rule

> **Nexa does not use OpenAI, Anthropic, Google Gemini, Claude, LLaMA, Mistral, GPT, Whisper,
> or any other pretrained AI model as a component of its intelligence.**
>
> Nexa is an independent research/learning project. Its language-model weights are trained from scratch and it does not use external LLM inference APIs or pretrained language-model weights.
>
> - ✅ Allowed: PyTorch, NumPy, FastAPI, PyYAML, Rich, and similar dev/utility libraries
> - ✅ Allowed: Publicly available text datasets (legal for training)
> - ❌ Forbidden: Downloading or importing any pretrained model weights
> - ❌ Forbidden: Wrapping any third-party AI API


---

## Vision

Nexa is a learning-first, production-ready AI system that teaches the internals of modern
language models by building them from first principles. By the time Nexa is complete, you will
have built and understood:

- A **Byte-Pair Encoding (BPE) tokenizer** — the same class of tokenizer used by GPT-4
- A **decoder-only Transformer** — the architecture behind every major LLM
- A **training loop** with gradient descent, mixed precision, and checkpointing
- A **text generation engine** with sampling strategies (top-k, top-p, temperature)
- A **chat interface** via Gradio + FastAPI
- A **memory system** combining sliding window context and FAISS vector search
- **Speech I/O** via Whisper (STT) and pyttsx3 (TTS)
- **Optimization** via quantization and ONNX export

---

## Development Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Project Setup |
| 2 | ✅ Complete | BPE Tokenizer |
| 3 | ✅ Complete | Transformer |
| 4 | ✅ Complete | Training Loop |
| 5 | ✅ Complete | Inference |
| 6 | ✅ Complete | Local Chat UI |
| 7 | ✅ Complete | Memory |
| 8 | ⬜ Not Started | Speech & Optimization |

---

## Quick Start

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/your-org/nexa.git
cd nexa

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install Nexa in editable mode (all dependencies included)
pip install -e .
```

### 2. Verify Installation

```bash
# Check logger works
python -c "from nexa.utils import get_logger; get_logger('nexa').info('Nexa is ready!')"

# Check config loader works
python -c "from nexa.utils import load_config; cfg = load_config('configs/model_config.yaml'); print(cfg)"
```

---

## Project Structure

```
nexa/
├── configs/          # YAML config files (model, train, inference)
├── data/             # Raw and processed training data
├── docs/             # Architecture notes and concept explanations
├── nexa/             # Main Python package
│   ├── tokenizer/    # BPE tokenizer (Phase 2)
│   ├── models/       # Transformer model (Phase 3)
│   ├── training/     # Training loop (Phase 4)
│   ├── inference/    # Text generation (Phase 5)
│   ├── memory/       # Memory system (Phase 7)
│   ├── tools/        # Tool use (web, calculator)
│   ├── speech/       # STT + TTS (Phase 8)
│   ├── app/          # Chat UI + REST API (Phase 6)
│   └── utils/        # Shared utilities (logger, config)
├── tests/            # Unit and integration tests
├── scripts/          # One-off data prep and eval scripts
├── checkpoints/      # Saved model weights (git-ignored)
└── logs/             # Training logs (git-ignored)
```

---

## Philosophy

- **Understand first, optimize later** — every component has a concept explanation before code
- **No magic** — we avoid abstractions we don't understand
- **Modular** — each sub-package is independent and testable
- **One dependency rule** — if it can be built in <100 lines, we build it

---

## Model Checkpoint Policy

Source code and configuration files belong in Git. For future large Nexa models, large trained model artifacts (like checkpoints) should not be committed directly to Git, but instead should use an appropriate artifact-storage mechanism (e.g., Git LFS, cloud storage, etc.).

---

## Dataset Policy

Future Nexa training datasets must be checked for licensing and redistribution rights before being committed. Small, project-owned demonstration or sample datasets can be tracked in Git. Private datasets, copyrighted data without permission, personal data, and huge generated datasets must not be uploaded to the repository.

---

## License

MIT License — see [LICENSE](LICENSE)
