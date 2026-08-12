# Phase 1: Project Setup — Concepts Explained

## Why does project structure matter for an AI system?

An AI project has an unusual shape: it spans **data engineering**, **mathematics**,
**software engineering**, and **deployment** — all in one codebase. Without clear
structure from day one, the project becomes a pile of scripts that nobody (including
future-you) can navigate or trust.

We front-load structure in Phase 1 so every future phase has a home.

---

## Concept 1: Python Packages vs. Modules

A **module** is a single `.py` file. A **package** is a directory that contains
an `__init__.py` file, making it importable as a namespace.

```
nexa/                  ← package (has __init__.py)
    utils/             ← sub-package (has __init__.py)
        logger.py      ← module
        config_loader.py  ← module
```

With `pip install -e .`, Python registers `nexa` as an installed package.
Any script anywhere on the machine can then do:

```python
from nexa.utils import get_logger   # works without sys.path hacks
```

The `-e` flag stands for **editable install**: Python symlinks to our actual
source files instead of copying them. Every edit we make is immediately live —
no reinstall needed.

---

## Concept 2: YAML Configuration

YAML (Yet Another Markup Language) is a human-readable data format. We use it
to externalise every numeric hyperparameter that controls model behaviour:

```yaml
model:
  d_model: 512        # embedding width
  n_layers: 6         # number of transformer blocks
  learning_rate: 3e-4
```

**Why not `.py` config files?**
YAML files are not code — they can't have bugs, can't have imports, and can't
accidentally break the runtime. They're also easy to diff in git when you
change a hyperparameter.

**OmegaConf** converts the raw YAML dict into a `DictConfig` object that:
- Supports dot-access: `cfg.model.d_model` instead of `cfg["model"]["d_model"]`
- Supports CLI overrides: `python train.py model.d_model=256`
- Validates types at access time

---

## Concept 3: Logging Levels

Python's `logging` module defines five severity levels:

| Level | Value | When to use |
|-------|-------|-------------|
| `DEBUG` | 10 | Verbose internal details (tensor shapes, step counts) |
| `INFO` | 20 | Normal milestones (epoch start, checkpoint saved) |
| `WARNING` | 30 | Recoverable issues (learning rate too high) |
| `ERROR` | 40 | Failures that interrupt a specific operation |
| `CRITICAL` | 50 | System-level failures (OOM, corrupt checkpoint) |

When you set a logger to `INFO`, only messages at `INFO` and above are shown.
`DEBUG` messages are silently discarded. This lets you toggle verbosity
without changing any code.

---

## Concept 4: Separation of Concerns

Each sub-package in `nexa/` has exactly one responsibility:

| Package | Responsibility |
|---------|---------------|
| `tokenizer/` | Convert raw text ↔ integer token IDs |
| `models/` | Define the neural network architecture |
| `training/` | Manage the gradient descent loop |
| `inference/` | Generate text from a trained model |
| `memory/` | Store and retrieve conversation history |
| `tools/` | Implement callable tools (calculator, search) |
| `speech/` | Convert between audio and text (future) |
| `app/` | Expose the system via web UI or REST API |
| `utils/` | Shared helpers (logger, config) used by all of the above |

No package imports from a "sibling" at the same level — dependencies only
flow downward (e.g., `training/` uses `models/` and `tokenizer/`, never the
reverse). This prevents circular imports and keeps the architecture clean.

---

## Concept 5: The Independence Rule

Nexa's core principle is **no pretrained AI weights**. Here is why this matters:

1. **Intellectual ownership**: If Nexa's intelligence comes from someone else's
   pretrained model, Nexa is just a wrapper — not an independent system.

2. **Legal clarity**: Pretrained model weights carry complex license terms
   (commercial restrictions, redistribution limits, usage policies). Weights
   we train ourselves on permissive datasets have no such ambiguity.

3. **Deep understanding**: Every concept in Nexa's architecture must be
   implementable and explainable by us. Importing a pretrained model shortcuts
   that understanding.

4. **Control**: We can modify, quantize, distil, or retrain every component.
   With a black-box external model, we have no such control.

The allowed libraries (PyTorch, NumPy, FastAPI, etc.) are **tools** — like
a compiler or a database engine. They provide no intelligence; they provide
computation infrastructure.

---

## What Phase 1 Produces

After Phase 1, we have:

```
✅ An installable Python package (pip install -e .)
✅ Three YAML config files (model, training, inference)
✅ A logger that writes rich terminal output + rotating file logs
✅ A config loader with dot-access and CLI override support
✅ Stub __init__.py files for all future sub-packages
✅ A full test suite for Phase 1 utilities
✅ Architecture documentation
✅ .gitignore, README, requirements.txt
```

We do NOT yet have:
```
⬜ A tokenizer (Phase 2)
⬜ A model (Phase 3)
⬜ Training data (Phase 4)
⬜ Any neural network weights
```
