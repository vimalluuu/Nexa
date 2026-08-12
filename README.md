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

Nexa is an independent research and learning AI system that teaches the internals of modern
language models by building them from first principles. It is intentionally not production-grade or comparable to massive commercial systems like ChatGPT or Claude. By the time Nexa is complete, you will
have built and understood:

- A **Byte-Pair Encoding (BPE) tokenizer** — built from scratch
- A **decoder-only Transformer** — the fundamental architecture behind modern LLMs
- A **training loop** with gradient descent, mixed precision, and checkpointing
- A **text generation engine** with sampling strategies (top-k, top-p, temperature)
- A **chat interface** via FastAPI and vanilla web technologies
- A **memory system** combining sliding window context and TF-IDF vector search
- **Scalable Pretraining** — dataset engineering, distributed training, and reproducibility

**Note on Speech Systems:** Nexa will NOT use Whisper, Vosk, DeepSpeech, pyttsx3, or any other pretrained/external speech intelligence. Speech is deferred to a future independent research phase where Nexa's own speech systems may be studied and trained from scratch.

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
| 8 | ⬜ Not Started | Pretraining v1 — Dataset Engineering & Model Scaling |

### Next Phase: Pretraining v1
Phase 8 will focus on:
- Identifying legally usable training datasets
- Dataset cleaning and deduplication
- Train/validation splitting
- Tokenizer retraining on the real corpus
- Model scaling from the current tiny research model
- Longer-context experiments
- Scalable pretraining infrastructure
- Training and evaluation metrics
- Checkpointing and reproducibility


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

## Current Model

- **Parameters**: Approximately 803K parameters
- **Vocabulary Size**: 124 tokens for the current toy model
- **Origin**: Trained entirely from scratch
- **Training Data**: Current training corpus is a small demonstration corpus
- **Capability**: Generated text quality is intentionally limited at this stage as it serves only to prove the architecture works.

---

## Project Structure

```
nexa/
├── configs/          # YAML config files (model, train, inference)
├── data/             # Dataset pipeline stages, manifests, and reports
├── docs/             # Architecture notes and concept explanations
├── nexa/             # Main Python package
│   ├── tokenizer/    # BPE tokenizer (Phase 2)
│   ├── models/       # Transformer model (Phase 3)
│   ├── training/     # Training loop (Phase 4)
│   ├── inference/    # Text generation (Phase 5)
│   ├── memory/       # TF-IDF Memory system (Phase 7)
│   ├── tools/        # Tool use
│   ├── speech/       # Deferred to future independent research
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
- **Independent AI** — no pretrained AI intelligence is allowed
- **One dependency rule** — if it can be built in <100 lines, we build it

---

## Model Checkpoint Policy

Source code and configuration files belong in Git. For future large Nexa models, large trained model artifacts (like checkpoints) should not be committed directly to Git, but instead should use an appropriate artifact-storage mechanism (e.g., Git LFS, cloud storage, etc.).

---

## Dataset Policy

Future Nexa training datasets must be checked for licensing and redistribution rights before being committed. Small, project-owned demonstration or sample datasets can be tracked in Git. Private datasets, copyrighted data without permission, personal data, and huge generated datasets must not be uploaded to the repository.

Dataset work must use separate pipeline stages:

```text
data/raw/<dataset_name>/
data/cleaned/<dataset_name>/
data/deduplicated/<dataset_name>/
data/splits/train/
data/splits/validation/
data/processed/train/
data/processed/validation/
data/manifests/
data/reports/
```

Raw source files from different datasets must never be mixed in one directory.
Every independently sourced dataset directory must include metadata for name,
source, official URL, version, license, download date, original filename,
checksum, and approximate size. Raw data is never modified in place or deleted
automatically. See `docs/datasets/dataset_folder_organization.md`.

---

## License

MIT License — see [LICENSE](LICENSE)
