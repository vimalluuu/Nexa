"""
nexa/training/trainer.py
=========================
Trainer — the complete training loop for NexaTransformer.

Responsibilities
-----------------
1. Build AdamW optimizer with separate weight-decay parameter groups.
2. Run the training loop: forward → loss → backward → clip → step.
3. Schedule the learning rate (cosine warmup).
4. Accumulate gradients across multiple mini-batches (simulating large batches).
5. Evaluate on a validation set at regular intervals.
6. Save and load checkpoints (full training state).
7. Log metrics and return a structured training history.

Independence
------------
All training is from scratch on the corpus you provide.
No pretrained weights, no external AI APIs, no data downloads.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from nexa.training.scheduler import CosineWarmupScheduler
from nexa.utils import get_logger

log = get_logger(__name__, log_to_file=False)


# ===========================================================================
# TrainerConfig
# ===========================================================================

@dataclass
class TrainerConfig:
    """
    All hyperparameters for the training loop.

    Every number that controls training lives here.
    Nothing is hardcoded in Trainer — it all reads from this config.

    Parameters
    ----------
    block_size : int
        Sequence length for each training window (tokens per example).
        Must match or be ≤ ModelConfig.max_seq_len.
    batch_size : int
        Number of windows per mini-batch.
    grad_accum_steps : int
        How many mini-batches to accumulate before one optimizer step.
        Effective batch size = batch_size × grad_accum_steps.
    learning_rate : float
        Peak learning rate (reached at end of warmup).
    min_lr : float
        Floor of cosine decay. Typically learning_rate / 10.
    weight_decay : float
        L2 regularization coefficient for AdamW (applied to 2D+ weights).
    beta1, beta2 : float
        AdamW momentum parameters. Standard: (0.9, 0.95).
    eps : float
        AdamW numerical stability epsilon.
    grad_clip : float
        Maximum global gradient norm. Clip if exceeded.
    max_steps : int
        Total optimizer steps. Training ends when step == max_steps.
    warmup_steps : int
        Number of steps for linear LR warmup (0 → max_lr).
    eval_interval : int
        Run validation every N optimizer steps (0 = never).
    eval_steps : int
        Maximum batches to evaluate on per validation run.
    save_interval : int
        Save a checkpoint every N optimizer steps (0 = never).
    checkpoint_dir : str or Path
        Directory for saving checkpoints.
    keep_last_n : int
        Delete older checkpoints, keeping only the last N.
    log_interval : int
        Print/log metrics every N optimizer steps (0 = never).
    device : str
        "auto" → cuda > mps > cpu. Or explicitly "cpu", "cuda", "mps".
    seed : int
        Random seed for reproducibility.
    """

    # Data
    block_size:        int   = 128
    batch_size:        int   = 4

    # Optimizer
    learning_rate:     float = 3e-4
    min_lr:            float = 3e-5
    weight_decay:      float = 0.1
    beta1:             float = 0.9
    beta2:             float = 0.95
    eps:               float = 1e-8
    grad_clip:         float = 1.0

    # Schedule
    max_steps:         int   = 1000
    warmup_steps:      int   = 100

    # Accumulation
    grad_accum_steps:  int   = 1

    # Evaluation
    eval_interval:     int   = 200
    eval_steps:        int   = 20

    # Checkpointing
    save_interval:     int   = 500
    checkpoint_dir:    Union[str, Path] = "checkpoints"
    keep_last_n:       int   = 3

    # Logging
    log_interval:      int   = 20

    # Device & reproducibility
    device:            str   = "auto"
    seed:              int   = 42

    def __post_init__(self) -> None:
        if self.warmup_steps >= self.max_steps and self.max_steps > 1:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps ({self.max_steps})."
            )
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be ≥ 1, got {self.grad_accum_steps}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {self.batch_size}.")
        if not (0.0 < self.learning_rate):
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        self.checkpoint_dir = Path(self.checkpoint_dir)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainerConfig":
        """Load TrainerConfig from configs/train_config.yaml."""
        from nexa.utils import load_config
        from omegaconf import OmegaConf

        raw = load_config(path)
        train_dict: dict = OmegaConf.to_container(raw.training, resolve=True)
        return cls(**{k: v for k, v in train_dict.items() if k in cls.__dataclass_fields__})

    @property
    def effective_batch_size(self) -> int:
        """The batch size seen by the optimizer after accumulation."""
        return self.batch_size * self.grad_accum_steps


# ===========================================================================
# Training History (structured result)
# ===========================================================================

@dataclass
class TrainingHistory:
    """Structured record of all metrics collected during training."""

    train_loss: list[tuple[int, float]] = field(default_factory=list)
    val_loss:   list[tuple[int, float]] = field(default_factory=list)
    lr_history: list[tuple[int, float]] = field(default_factory=list)

    def record_train(self, step: int, loss: float, lr: float) -> None:
        self.train_loss.append((step, loss))
        self.lr_history.append((step, lr))

    def record_val(self, step: int, loss: float) -> None:
        self.val_loss.append((step, loss))

    @property
    def final_train_loss(self) -> Optional[float]:
        return self.train_loss[-1][1] if self.train_loss else None

    @property
    def final_val_loss(self) -> Optional[float]:
        return self.val_loss[-1][1] if self.val_loss else None

    @property
    def initial_train_loss(self) -> Optional[float]:
        return self.train_loss[0][1] if self.train_loss else None


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:
    """
    Training loop for NexaTransformer.

    Usage
    -----
    trainer = Trainer(
        model        = NexaTransformer(model_cfg),
        config       = TrainerConfig(max_steps=500, batch_size=4),
        train_dataset = NexaDataset(token_ids, block_size=64),
        val_dataset   = NexaDataset(val_ids, block_size=64),   # optional
    )
    history = trainer.train()
    print(f"Final loss: {history.final_train_loss:.4f}")

    Parameters
    ----------
    model : nn.Module
        The model to train (NexaTransformer).
    config : TrainerConfig
        All training hyperparameters.
    train_dataset : Dataset
        Training data (NexaDataset).
    val_dataset : Dataset, optional
        Validation data. If None, no validation is run.
    resume_from : Path, optional
        Path to a checkpoint to resume from. Restores model + optimizer state.
    """

    def __init__(
        self,
        model:         nn.Module,
        config:        TrainerConfig,
        train_dataset: Dataset,
        val_dataset:   Optional[Dataset] = None,
        resume_from:   Optional[Path]    = None,
    ) -> None:
        self.model         = model
        self.cfg           = config
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset

        self.device = self._resolve_device(config.device)
        self._set_seed(config.seed)

        self._start_step = 0
        self._resume_path = resume_from

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self) -> TrainingHistory:
        """
        Run the full training loop.

        Algorithm (one optimizer step):
            1.  Sample a mini-batch (x, y) from the DataLoader.
            2.  Forward pass: logits = model(x).
            3.  Compute cross-entropy loss on shifted targets.
            4.  Scale loss by 1/grad_accum_steps and call .backward().
            5.  Repeat steps 1-4 for grad_accum_steps mini-batches.
            6.  Clip global gradient norm to grad_clip.
            7.  Update learning rate via cosine warmup schedule.
            8.  optimizer.step() and optimizer.zero_grad().
            9.  Increment step counter; log / eval / checkpoint as needed.

        Returns
        -------
        TrainingHistory
            Contains lists of (step, value) pairs for train_loss,
            val_loss, and lr.
        """
        self.model.to(self.device)
        self.model.train()

        optimizer = self._build_optimizer()
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps = self.cfg.warmup_steps,
            max_steps    = self.cfg.max_steps,
            max_lr       = self.cfg.learning_rate,
            min_lr       = self.cfg.min_lr,
        )
        history = TrainingHistory()

        # Resume from checkpoint if requested
        step = self._start_step
        if self._resume_path is not None:
            step = self._load_checkpoint(self._resume_path, optimizer, scheduler)
            log.info("Resumed from step %d", step)

        # Infinite data loader (reshuffles each epoch)
        data_iter = self._make_infinite_loader(self.train_dataset)

        # Gradient accumulation state
        accum_count  = 0
        running_loss = 0.0
        optimizer.zero_grad()

        t0 = time.perf_counter()

        while step < self.cfg.max_steps:
            # ----------------------------------------------------------------
            # Accumulation inner loop
            # ----------------------------------------------------------------
            x, y = next(data_iter)
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            logits = self.model(x)                  # [B, S, vocab_size]
            loss   = self._compute_loss(logits, y)  # scalar

            # Scale for gradient accumulation then backpropagate
            (loss / self.cfg.grad_accum_steps).backward()
            running_loss += loss.item()
            accum_count  += 1

            # Not yet ready for an optimizer step
            if accum_count < self.cfg.grad_accum_steps:
                continue

            # ----------------------------------------------------------------
            # Optimizer step (every grad_accum_steps mini-batches)
            # ----------------------------------------------------------------

            # Clip gradient norm (prevents catastrophic updates)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )

            # Apply LR for this step, then advance the scheduler
            lr = scheduler.step()

            # Weight update
            optimizer.step()
            optimizer.zero_grad()

            # Compute average loss across accumulation steps
            avg_loss    = running_loss / accum_count
            running_loss = 0.0
            accum_count  = 0
            step        += 1

            # ----------------------------------------------------------------
            # Logging
            # ----------------------------------------------------------------
            if self.cfg.log_interval > 0 and step % self.cfg.log_interval == 0:
                t1       = time.perf_counter()
                ms_step  = (t1 - t0) * 1000 / self.cfg.log_interval
                t0       = t1
                history.record_train(step, avg_loss, lr)
                log.info(
                    "step %5d/%d | loss %.4f | lr %.2e | grad_norm %.2f | %.1f ms/step",
                    step, self.cfg.max_steps, avg_loss, lr, grad_norm, ms_step,
                )

            # ----------------------------------------------------------------
            # Validation
            # ----------------------------------------------------------------
            if (
                self.val_dataset is not None
                and self.cfg.eval_interval > 0
                and step % self.cfg.eval_interval == 0
            ):
                val_loss = self.evaluate()
                history.record_val(step, val_loss)
                log.info("  → val_loss %.4f", val_loss)
                self.model.train()

            # ----------------------------------------------------------------
            # Checkpointing
            # ----------------------------------------------------------------
            if (
                self.cfg.save_interval > 0
                and step % self.cfg.save_interval == 0
            ):
                self._save_checkpoint(step, avg_loss, optimizer, scheduler)

        # Final checkpoint at the end of training
        if self.cfg.save_interval > 0:
            self._save_checkpoint(step, avg_loss if accum_count == 0 else running_loss, optimizer, scheduler)

        log.info("Training complete. Final train loss: %.4f", history.final_train_loss or float("nan"))
        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Compute average cross-entropy loss on the validation set.

        Returns
        -------
        float
            Mean validation loss over up to `eval_steps` batches.
        """
        if self.val_dataset is None:
            raise RuntimeError("No validation dataset was provided.")

        self.model.eval()
        loader = DataLoader(
            self.val_dataset,
            batch_size = self.cfg.batch_size,
            shuffle    = False,
            drop_last  = False,
        )

        total_loss = 0.0
        n_batches  = 0
        for x, y in loader:
            if n_batches >= self.cfg.eval_steps:
                break
            x, y  = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            total_loss += self._compute_loss(logits, y).item()
            n_batches  += 1

        return total_loss / max(1, n_batches)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        step:      int,
        loss:      float,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWarmupScheduler,
    ) -> Path:
        """
        Save a full training checkpoint.

        The checkpoint contains:
            - model state dict (all weights + registered buffers)
            - optimizer state dict (Adam moments)
            - scheduler step counter
            - current step and loss
            - TrainerConfig (so training can be resumed without re-specifying args)

        Parameters
        ----------
        step : int
            Current optimizer step.
        loss : float
            Current training loss.
        optimizer : Optimizer
        scheduler : CosineWarmupScheduler

        Returns
        -------
        Path
            Path to the saved checkpoint file.
        """
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.checkpoint_dir / f"step_{step:07d}.pt"

        # Serialize config with Path converted to str (weights_only=True safe)
        cfg_dict = dataclasses.asdict(self.cfg)
        cfg_dict["checkpoint_dir"] = str(cfg_dict["checkpoint_dir"])

        torch.save(
            {
                "step":              step,
                "loss":              loss,
                "model_state":       self.model.state_dict(),
                "optimizer_state":   optimizer.state_dict(),
                "scheduler_step":    scheduler.steps_completed,
                "trainer_config":    cfg_dict,
            },
            path,
        )
        log.info("Checkpoint saved → %s", path)

        # Also write a 'latest.pt' for easy resume
        latest = self.cfg.checkpoint_dir / "latest.pt"
        torch.save({"latest_path": str(path)}, latest)

        # Prune old checkpoints
        self._prune_checkpoints()
        return path

    def _load_checkpoint(
        self,
        path:      Path,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWarmupScheduler,
    ) -> int:
        """
        Load a checkpoint and restore model, optimizer, and scheduler state.

        Parameters
        ----------
        path : Path
            Path to the checkpoint .pt file.
        optimizer : Optimizer
            Optimizer whose state will be restored.
        scheduler : CosineWarmupScheduler
            Scheduler whose step counter will be restored.

        Returns
        -------
        int
            The step to resume from (the step saved in the checkpoint).
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler._step = ckpt.get("scheduler_step", ckpt["step"])
        log.info("Checkpoint loaded ← %s (step %d)", path, ckpt["step"])
        return ckpt["step"]

    def _prune_checkpoints(self) -> None:
        """Delete old checkpoints, keeping the last `keep_last_n`."""
        if self.cfg.keep_last_n <= 0:
            return
        ckpts = sorted(self.cfg.checkpoint_dir.glob("step_*.pt"))
        for old in ckpts[: -self.cfg.keep_last_n]:
            old.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        Build AdamW with separate parameter groups for weight decay.

        Weight decay (L2 regularization) should NOT be applied to:
            - 1D tensors: norm scales (γ), biases
            - Any embedding-style lookups

        Weight decay SHOULD be applied to:
            - 2D+ tensors: all linear weight matrices (W_q, W_k, W_v, W_o, FFN)

        This separation is what makes AdamW correct. Applying weight decay to
        norm parameters would shrink their scale toward zero, breaking training.

        Returns
        -------
        torch.optim.AdamW
        """
        decay_params   = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2:
                # 1D: norm weights, biases → no decay
                nodecay_params.append(param)
            else:
                # 2D+: linear / embedding weight matrices → decay
                decay_params.append(param)

        n_decay   = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        log.debug(
            "Optimizer param groups: decay=%s params, no_decay=%s params",
            f"{n_decay:,}", f"{n_nodecay:,}",
        )

        return torch.optim.AdamW(
            [
                {"params": decay_params,   "weight_decay": self.cfg.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr    = self.cfg.learning_rate,
            betas = (self.cfg.beta1, self.cfg.beta2),
            eps   = self.cfg.eps,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _make_infinite_loader(
        self,
        dataset: Dataset,
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        """
        Yield (x, y) batches from `dataset` indefinitely.

        Creates a new shuffled DataLoader each epoch so the order is
        different every pass through the data (proper stochastic training).
        This is important for small datasets where we loop many times.

        Yields
        ------
        (x, y) : tuple[Tensor, Tensor]
            Each of shape [batch_size, block_size].
        """
        drop = len(dataset) >= self.cfg.batch_size
        while True:
            loader = DataLoader(
                dataset,
                batch_size = self.cfg.batch_size,
                shuffle    = True,
                drop_last  = drop,
            )
            for batch in loader:
                yield batch

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
        """
        Cross-entropy loss for language modeling.

        Language modeling loss: given a sequence x[0..S-1], predict x[1..S].
        The model outputs logits of shape [B, S, V].
        The target is the same sequence shifted by 1 (already done in NexaDataset):
            logits[:, t, :] predicts targets[:, t].

        Parameters
        ----------
        logits : Tensor  shape [B, S, vocab_size]
            Raw model output.
        targets : Tensor  shape [B, S]
            Token IDs to predict.

        Returns
        -------
        Tensor  scalar
            Mean cross-entropy loss over all (batch, position) pairs.
        """
        B, S, V = logits.shape
        # Flatten to [B*S, V] and [B*S] for F.cross_entropy
        return F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve "auto" → the best available device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __repr__(self) -> str:
        return (
            f"Trainer("
            f"device={self.device}, "
            f"max_steps={self.cfg.max_steps}, "
            f"eff_batch={self.cfg.effective_batch_size})"
        )
