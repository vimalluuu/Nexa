"""
nexa/training/__init__.py
==========================
Public API for Nexa's training sub-package.

Primary entry points:

    from nexa.training import Trainer, TrainerConfig, TrainingHistory
    from nexa.training import NexaDataset
    from nexa.training import CosineWarmupScheduler, cosine_warmup_lr

Independence
------------
All training is performed from scratch using your own corpus and
NexaTokenizer (Phase 2). No pretrained weights, no external AI APIs.
"""

from nexa.training.dataset   import NexaDataset
from nexa.training.scheduler import CosineWarmupScheduler, cosine_warmup_lr
from nexa.training.trainer   import Trainer, TrainerConfig, TrainingHistory

__all__ = [
    "NexaDataset",
    "CosineWarmupScheduler",
    "cosine_warmup_lr",
    "Trainer",
    "TrainerConfig",
    "TrainingHistory",
]
