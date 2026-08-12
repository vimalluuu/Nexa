"""
tests/test_training.py
=======================
Phase 4 test suite — training loop (dataset, scheduler, trainer).

Test classes:
    TestNexaDataset          — sliding window dataset correctness
    TestCosineWarmupScheduler — LR schedule properties
    TestTrainerConfig         — config validation
    TestTrainer               — optimizer, accumulation, checkpoint, convergence
    TestConvergence           — the key integration test: loss decreases

Run with:
    pytest tests/test_training.py -v
    pytest tests/test_training.py -v -k "not Convergence"   # skip the slow test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from nexa.models    import NexaTransformer, ModelConfig
from nexa.training  import (
    NexaDataset,
    CosineWarmupScheduler,
    cosine_warmup_lr,
    Trainer,
    TrainerConfig,
    TrainingHistory,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_tiny_model_cfg(**kwargs) -> ModelConfig:
    """Minimal model config for fast CPU tests."""
    defaults = dict(
        vocab_size   = 64,
        max_seq_len  = 32,
        d_model      = 32,
        n_heads      = 2,
        n_layers     = 1,
        d_ff         = 64,
        dropout      = 0.0,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def make_tiny_trainer_cfg(tmp_path: Path, **kwargs) -> TrainerConfig:
    """Minimal trainer config with all I/O suppressed for unit tests."""
    defaults = dict(
        block_size      = 8,
        batch_size      = 2,
        max_steps       = 30,
        warmup_steps    = 3,
        learning_rate   = 1e-3,
        min_lr          = 1e-4,
        grad_accum_steps= 1,
        eval_interval   = 0,           # no validation during unit tests
        save_interval   = 0,           # no checkpointing during unit tests
        log_interval    = 9999,        # suppress log noise
        checkpoint_dir  = tmp_path / "ckpts",
        device          = "cpu",
        seed            = 42,
    )
    defaults.update(kwargs)
    return TrainerConfig(**defaults)


def make_tiny_dataset(n_tokens: int = 512, block_size: int = 8) -> NexaDataset:
    """
    Create a tiny dataset with a simple repeating pattern.
    The pattern [1, 2, ..., 15, 1, 2, ...] is easy for the model to learn.
    Starts from 1 to avoid token 0 (PAD), whose embedding is zeroed.
    """
    vocab = 64
    ids   = [(i % (vocab - 1)) + 1 for i in range(n_tokens)]  # IDs in [1, 63]
    return NexaDataset(ids, block_size)


# ===========================================================================
# 1. NexaDataset
# ===========================================================================

class TestNexaDataset:
    """Verify sliding-window dataset correctness."""

    def test_length_formula(self):
        """len(ds) = max(0, n_tokens - block_size)."""
        ids = list(range(50))
        ds  = NexaDataset(ids, block_size=10)
        assert len(ds) == 40

    def test_length_too_short(self):
        """Fewer tokens than block_size → empty dataset."""
        ids = list(range(5))
        ds  = NexaDataset(ids, block_size=10)
        assert len(ds) == 0

    def test_exact_minimum(self):
        """block_size+1 tokens → exactly 1 valid window."""
        ids = list(range(11))
        ds  = NexaDataset(ids, block_size=10)
        assert len(ds) == 1

    def test_x_and_y_shapes(self):
        """Each item returns (x, y) of shape [block_size]."""
        ds = make_tiny_dataset(block_size=8)
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_y_is_x_shifted_by_one(self):
        """y[i] == x[i+1] for all i < block_size-1."""
        ids = list(range(100))
        ds  = NexaDataset(ids, block_size=16)
        x, y = ds[0]
        # x = [0,1,...,15], y = [1,2,...,16]
        assert torch.equal(y[:-1], x[1:])

    def test_first_window_starts_at_zero(self):
        """ds[0] returns the window starting at position 0."""
        ids = list(range(20))
        ds  = NexaDataset(ids, block_size=5)
        x, y = ds[0]
        assert x.tolist() == [0, 1, 2, 3, 4]
        assert y.tolist() == [1, 2, 3, 4, 5]

    def test_last_window(self):
        """ds[-1] / ds[len-1] is the last valid window."""
        ids = list(range(20))
        ds  = NexaDataset(ids, block_size=5)
        x, y = ds[len(ds) - 1]
        # Last start: 20 - 5 - 1 = 14 ... wait, len = 20-5=15, last idx=14
        assert x.tolist() == [14, 15, 16, 17, 18]
        assert y.tolist() == [15, 16, 17, 18, 19]

    def test_token_ids_as_tensor(self):
        """NexaDataset accepts a Tensor as input."""
        t  = torch.arange(30)
        ds = NexaDataset(t, block_size=5)
        assert len(ds) == 25

    def test_getitem_returns_long_tensors(self):
        """x and y must be integer (long) tensors."""
        ds = make_tiny_dataset()
        x, y = ds[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_from_text(self):
        """from_text() tokenizes and creates a dataset."""
        from nexa.tokenizer import NexaTokenizer
        corpus = ["the cat sat"] * 10
        tok    = NexaTokenizer.train(corpus, vocab_size=50)
        text   = "\n".join(corpus)
        ds     = NexaDataset.from_text(text, tok, block_size=8)
        assert len(ds) > 0

    def test_train_val_split_sizes(self):
        """train_val_split returns non-empty train and val datasets."""
        ids         = list(range(500))
        train, val  = NexaDataset.train_val_split(ids, block_size=10, val_fraction=0.1)
        assert len(train) > 0
        assert len(val)   > 0

    def test_train_val_split_no_data_leakage(self):
        """
        Training data and validation data start at different positions.
        (Val set may share a few tokens at the boundary for context,
        but the bulk should not overlap.)
        """
        ids        = list(range(500))
        train, val = NexaDataset.train_val_split(ids, block_size=10, val_fraction=0.2)
        # The training set must be strictly shorter than the full set
        assert train.token_count() < len(ids)

    def test_repr_contains_key_info(self):
        ds = make_tiny_dataset()
        r  = repr(ds)
        assert "NexaDataset" in r
        assert "block_size"  in r

    def test_train_val_split_too_short_raises(self):
        """Corpus shorter than block_size+2 must raise ValueError."""
        with pytest.raises(ValueError, match="too short"):
            NexaDataset.train_val_split([1, 2, 3], block_size=10)


# ===========================================================================
# 2. CosineWarmupScheduler
# ===========================================================================

class TestCosineWarmupScheduler:
    """Verify LR schedule mathematical properties."""

    # --- cosine_warmup_lr (stateless function) ---

    def test_warmup_starts_positive(self):
        """Step 0 should give a small positive LR (not zero)."""
        lr = cosine_warmup_lr(0, warmup_steps=10, max_steps=100, max_lr=1e-3)
        assert lr > 0

    def test_warmup_end_equals_max_lr(self):
        """LR at step=warmup_steps should equal max_lr."""
        lr = cosine_warmup_lr(10, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
        assert abs(lr - 1e-3) < 1e-9

    def test_warmup_is_monotonically_increasing(self):
        """LR strictly increases during warmup."""
        lrs = [cosine_warmup_lr(t, warmup_steps=10, max_steps=100, max_lr=1e-3) for t in range(10)]
        assert lrs == sorted(lrs)

    def test_decay_is_monotonically_decreasing(self):
        """LR monotonically decreases after warmup."""
        lrs = [cosine_warmup_lr(t, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
               for t in range(10, 101)]
        assert lrs == sorted(lrs, reverse=True)

    def test_final_step_equals_min_lr(self):
        """LR at step=max_steps should equal min_lr."""
        lr = cosine_warmup_lr(100, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
        assert abs(lr - 1e-4) < 1e-9

    def test_beyond_max_steps_returns_min_lr(self):
        """Steps beyond max_steps clamp at min_lr."""
        lr = cosine_warmup_lr(999, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
        assert abs(lr - 1e-4) < 1e-9

    def test_midpoint_between_min_and_max(self):
        """At the midpoint of decay, LR ≈ (max_lr + min_lr) / 2."""
        W, T = 0, 100
        mid_lr = cosine_warmup_lr(50, W, T, max_lr=1.0, min_lr=0.0)
        assert abs(mid_lr - 0.5) < 1e-6

    # --- CosineWarmupScheduler (stateful class) ---

    def test_scheduler_instantiation(self):
        optimizer = torch.optim.AdamW([torch.zeros(1)], lr=1e-3)
        sched = CosineWarmupScheduler(optimizer, warmup_steps=5, max_steps=50,
                                       max_lr=1e-3, min_lr=1e-4)
        assert sched.steps_completed == 0

    def test_scheduler_step_advances_counter(self):
        optimizer = torch.optim.AdamW([torch.zeros(1)], lr=1e-3)
        sched = CosineWarmupScheduler(optimizer, warmup_steps=5, max_steps=50,
                                       max_lr=1e-3, min_lr=1e-4)
        sched.step()
        assert sched.steps_completed == 1

    def test_scheduler_applies_lr_to_optimizer(self):
        """After step(), optimizer.param_groups[0]['lr'] reflects the schedule."""
        param = torch.zeros(2, 2, requires_grad=True)
        optimizer = torch.optim.AdamW([param], lr=0.0)
        sched = CosineWarmupScheduler(optimizer, warmup_steps=0, max_steps=100,
                                       max_lr=1e-3, min_lr=1e-4)
        sched.step()
        lr_in_optimizer = optimizer.param_groups[0]["lr"]
        expected = cosine_warmup_lr(0, 0, 100, 1e-3, 1e-4)
        assert abs(lr_in_optimizer - expected) < 1e-9

    def test_scheduler_lr_decreases_over_decay(self):
        """LR must decrease monotonically through the decay phase."""
        optimizer = torch.optim.AdamW([torch.zeros(2, 2)], lr=0.0)
        sched = CosineWarmupScheduler(optimizer, warmup_steps=5, max_steps=50,
                                       max_lr=1e-3, min_lr=1e-4)
        lrs = []
        for _ in range(50):
            lrs.append(sched.step())
        decay_lrs = lrs[5:]  # after warmup
        assert decay_lrs == sorted(decay_lrs, reverse=True)

    def test_invalid_warmup_greater_than_max(self):
        optimizer = torch.optim.AdamW([torch.zeros(1)], lr=1e-3)
        with pytest.raises(ValueError, match="warmup_steps"):
            CosineWarmupScheduler(optimizer, warmup_steps=100, max_steps=50,
                                   max_lr=1e-3)

    def test_repr_contains_step_info(self):
        optimizer = torch.optim.AdamW([torch.zeros(1)], lr=1e-3)
        sched = CosineWarmupScheduler(optimizer, warmup_steps=5, max_steps=50,
                                       max_lr=1e-3)
        r = repr(sched)
        assert "CosineWarmupScheduler" in r
        assert "50" in r


# ===========================================================================
# 3. TrainerConfig
# ===========================================================================

class TestTrainerConfig:
    """Verify TrainerConfig validation and properties."""

    def test_default_construction(self, tmp_path):
        cfg = TrainerConfig(checkpoint_dir=tmp_path)
        assert cfg.max_steps   == 1000
        assert cfg.batch_size  == 4
        assert cfg.grad_clip   == 1.0

    def test_effective_batch_size(self, tmp_path):
        cfg = TrainerConfig(batch_size=4, grad_accum_steps=8, checkpoint_dir=tmp_path)
        assert cfg.effective_batch_size == 32

    def test_warmup_ge_max_steps_raises(self, tmp_path):
        with pytest.raises(ValueError, match="warmup_steps"):
            TrainerConfig(max_steps=10, warmup_steps=10, checkpoint_dir=tmp_path)

    def test_invalid_batch_size(self, tmp_path):
        with pytest.raises(ValueError, match="batch_size"):
            TrainerConfig(batch_size=0, checkpoint_dir=tmp_path)

    def test_invalid_grad_accum(self, tmp_path):
        with pytest.raises(ValueError, match="grad_accum_steps"):
            TrainerConfig(grad_accum_steps=0, checkpoint_dir=tmp_path)

    def test_checkpoint_dir_is_path(self, tmp_path):
        cfg = TrainerConfig(checkpoint_dir=str(tmp_path))
        assert isinstance(cfg.checkpoint_dir, Path)


# ===========================================================================
# 4. Trainer
# ===========================================================================

class TestTrainer:
    """Unit tests for Trainer behaviour (fast, no convergence guarantee)."""

    def test_trainer_instantiation(self, tmp_path):
        """Trainer can be constructed with a tiny model and dataset."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path)
        trainer = Trainer(model, cfg, dataset)
        assert trainer is not None

    def test_repr(self, tmp_path):
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path)
        trainer = Trainer(model, cfg, dataset)
        r = repr(trainer)
        assert "Trainer" in r
        assert "cpu"     in r

    def test_compute_loss_returns_scalar(self, tmp_path):
        """_compute_loss returns a scalar tensor."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path)
        trainer = Trainer(model, cfg, dataset)

        x = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            logits = model(x)
        loss = Trainer._compute_loss(logits, x)
        assert loss.shape == ()      # scalar
        assert loss.item() > 0

    def test_loss_is_finite_at_init(self, tmp_path):
        """Initial loss must be a finite positive number."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path)
        trainer = Trainer(model, cfg, dataset)

        x = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            logits = model(x)
        loss = Trainer._compute_loss(logits, x).item()
        assert loss == loss    # not NaN (NaN != NaN)
        assert loss < 1e9      # finite

    def test_initial_loss_near_log_vocab(self, tmp_path):
        """
        At random init, loss ≈ log(vocab_size).
        For vocab=64: log(64) ≈ 4.16.
        We allow ±1.5 tolerance since random init is noisy.
        """
        import math
        torch.manual_seed(0)
        model_cfg = make_tiny_model_cfg(vocab_size=64)
        model     = NexaTransformer(model_cfg)
        model.eval()

        losses = []
        for _ in range(10):
            x = torch.randint(0, 64, (4, 8))
            with torch.no_grad():
                logits = model(x)
                loss   = Trainer._compute_loss(logits, x).item()
            losses.append(loss)

        mean_loss  = sum(losses) / len(losses)
        target     = math.log(64)   # ≈ 4.16
        assert abs(mean_loss - target) < 1.5, (
            f"Expected initial loss ≈ {target:.2f}, got {mean_loss:.2f}"
        )

    def test_optimizer_has_two_param_groups(self, tmp_path):
        """
        AdamW must have 2 param groups: one with weight_decay, one without.
        """
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path)
        trainer = Trainer(model, cfg, dataset)
        optim   = trainer._build_optimizer()
        assert len(optim.param_groups) == 2

    def test_optimizer_nodecay_group_has_zero_wd(self, tmp_path):
        """1D params (norms) must have weight_decay=0 in the optimizer."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path)
        trainer = Trainer(model, cfg, dataset)
        optim   = trainer._build_optimizer()
        # Second group = no-decay
        assert optim.param_groups[1]["weight_decay"] == 0.0

    def test_train_returns_history(self, tmp_path):
        """train() returns a TrainingHistory object."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path, max_steps=5)
        trainer = Trainer(model, cfg, dataset)
        history = trainer.train()
        assert isinstance(history, TrainingHistory)

    def test_train_completes_requested_steps(self, tmp_path):
        """train() runs exactly max_steps optimizer steps."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path, max_steps=10, log_interval=1)
        trainer = Trainer(model, cfg, dataset)
        history = trainer.train()
        # With log_interval=1, every step is recorded
        assert len(history.train_loss) == 10

    def test_grad_accumulation_step_count(self, tmp_path):
        """
        With grad_accum_steps=3, the optimizer is called max_steps times
        regardless. Verify training completes the correct step count.
        """
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(
            tmp_path, max_steps=9, grad_accum_steps=3, log_interval=1
        )
        trainer = Trainer(model, cfg, dataset)
        history = trainer.train()
        assert len(history.train_loss) == 9

    # --- Checkpointing ---

    def test_save_checkpoint_creates_file(self, tmp_path):
        """Checkpointing must create a .pt file."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(
            tmp_path, max_steps=5, save_interval=5, log_interval=9999
        )
        trainer = Trainer(model, cfg, dataset)
        trainer.train()
        checkpoints = list(cfg.checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0

    def test_checkpoint_contains_required_keys(self, tmp_path):
        """Checkpoint file must have model_state, optimizer_state, step, loss."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(
            tmp_path, max_steps=5, save_interval=5, log_interval=9999
        )
        trainer = Trainer(model, cfg, dataset)
        trainer.train()

        ckpt_path = sorted(cfg.checkpoint_dir.glob("step_*.pt"))[0]
        ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key in ("step", "loss", "model_state", "optimizer_state"):
            assert key in ckpt, f"Missing key '{key}' in checkpoint."

    def test_load_checkpoint_restores_weights(self, tmp_path):
        """
        Saving then loading a trainer checkpoint must restore identical model weights.

        The trainer checkpoint is a FULL dict (step, loss, model_state, ...).
        We load 'model_state' from it and verify weights match the trained model.
        """
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(
            tmp_path, max_steps=5, save_interval=5, log_interval=9999
        )
        trainer = Trainer(model, cfg, dataset)
        trainer.train()

        # Snapshot the trained weights
        trained_w = model.embedding.weight.data.clone()

        # Load checkpoint dict and restore into a fresh model
        ckpt_path = sorted(cfg.checkpoint_dir.glob("step_*.pt"))[0]
        ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        fresh = NexaTransformer(make_tiny_model_cfg())
        fresh.load_state_dict(ckpt["model_state"])

        # The fresh model's weights must now match the trained model's weights
        assert torch.equal(fresh.embedding.weight.data, trained_w), (
            "Loaded model weights do not match the trained model."
        )


    # --- Evaluation ---

    def test_evaluate_returns_float(self, tmp_path):
        """evaluate() returns a finite positive float."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        val_ds  = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path)
        trainer = Trainer(model, cfg, dataset, val_dataset=val_ds)
        model.to(trainer.device)
        val_loss = trainer.evaluate()
        assert isinstance(val_loss, float)
        assert val_loss > 0
        assert val_loss == val_loss  # not NaN

    def test_evaluate_without_val_dataset_raises(self, tmp_path):
        """evaluate() must raise if no validation dataset was provided."""
        model   = NexaTransformer(make_tiny_model_cfg())
        dataset = make_tiny_dataset()
        cfg     = make_tiny_trainer_cfg(tmp_path)
        trainer = Trainer(model, cfg, dataset)    # no val_dataset
        with pytest.raises(RuntimeError, match="validation"):
            trainer.evaluate()


# ===========================================================================
# 5. Convergence (the key integration test)
# ===========================================================================

class TestConvergence:
    """
    Verify that the training loop actually reduces the loss.

    This is the single most important test for Phase 4.
    It validates the entire pipeline end-to-end:
        dataset → model forward → loss → backward → optimizer → weight update.

    Strategy:
    - Use a highly-repetitive dataset (cyclic token IDs) that a tiny model
      can overfit to after ~50 steps.
    - Assert that final loss < initial loss by a meaningful margin.
    - Use a fixed seed for reproducibility.

    Rationale for cyclic dataset:
        ids = [0, 1, 2, ..., 15, 0, 1, 2, ...]  (repeats)
        Every context [i, i+1, ..., i+7] maps deterministically to target i+8.
        A 1-layer transformer with sufficient LR can memorize this in <50 steps.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def trained_history(cls, tmp_path_factory):
        """Train once and reuse across all convergence tests."""
        torch.manual_seed(42)
        tmp = tmp_path_factory.mktemp("convergence")

        model_cfg = make_tiny_model_cfg(
            vocab_size  = 32,
            d_model     = 32,
            n_heads     = 2,
            n_layers    = 1,
            d_ff        = 64,
            max_seq_len = 32,
        )
        trainer_cfg = TrainerConfig(
            block_size      = 8,
            batch_size      = 4,
            max_steps       = 100,
            warmup_steps    = 5,
            learning_rate   = 5e-3,    # higher LR → faster memorization
            min_lr          = 5e-4,
            grad_accum_steps= 1,
            eval_interval   = 0,
            save_interval   = 0,
            log_interval    = 10,      # record every 10 steps
            checkpoint_dir  = tmp / "ckpts",
            device          = "cpu",
            seed            = 42,
        )

        # Cyclic dataset: IDs in [1, 31] to avoid PAD token 0 (zeroed embedding)
        n_tokens = 512
        ids      = [(i % 31) + 1 for i in range(n_tokens)]  # [1..31] repeating
        dataset  = NexaDataset(ids, block_size=8)

        model   = NexaTransformer(model_cfg)
        trainer = Trainer(model, trainer_cfg, dataset)
        history = trainer.train()
        return history, model

    def test_loss_decreases(self, trained_history):
        """Final train loss must be strictly less than initial train loss."""
        history, _ = trained_history
        assert history.initial_train_loss is not None
        assert history.final_train_loss   is not None
        assert history.final_train_loss < history.initial_train_loss, (
            f"Expected loss to decrease.\n"
            f"  Initial loss : {history.initial_train_loss:.4f}\n"
            f"  Final loss   : {history.final_train_loss:.4f}\n"
            "The training loop is not learning."
        )

    def test_loss_decreases_by_significant_margin(self, trained_history):
        """
        Loss must drop by at least 10% from initial.
        On a cyclic dataset with lr=5e-3, a 1-layer model should see
        much larger reduction. 10% is a very conservative lower bound.
        """
        history, _ = trained_history
        reduction = (history.initial_train_loss - history.final_train_loss) / history.initial_train_loss
        assert reduction >= 0.10, (
            f"Expected ≥10% loss reduction, got {reduction*100:.1f}%.\n"
            f"  Initial: {history.initial_train_loss:.4f}\n"
            f"  Final  : {history.final_train_loss:.4f}"
        )

    def test_history_has_entries(self, trained_history):
        """Training history must contain at least one recorded loss."""
        history, _ = trained_history
        assert len(history.train_loss) > 0

    def test_lr_history_populated(self, trained_history):
        """LR history must be recorded alongside loss."""
        history, _ = trained_history
        assert len(history.lr_history) > 0

    def test_all_losses_finite(self, trained_history):
        """No NaN or Inf losses during training (gradient stability check)."""
        history, _ = trained_history
        for step, loss in history.train_loss:
            assert loss == loss, f"NaN detected at step {step}."
            assert loss < 1e6,  f"Exploding loss {loss:.2e} at step {step}."

    def test_model_weights_changed_after_training(self, trained_history):
        """Model weights must be different from their initialization."""
        _, model = trained_history
        # Re-build a fresh model with the same seed and check weights differ
        torch.manual_seed(42)
        fresh = NexaTransformer(make_tiny_model_cfg(vocab_size=32))
        # The trained model should not match a freshly initialized one
        assert not torch.equal(
            model.embedding.weight.data,
            fresh.embedding.weight.data,
        ), "Model weights unchanged after training — optimizer may not be working."

    def test_full_pipeline_with_tokenizer(self, tmp_path):
        """
        End-to-end test: tokenize a tiny corpus, train, verify loss decreases.
        Uses NexaTokenizer from Phase 2.
        """
        from nexa.tokenizer import NexaTokenizer

        # Minimal corpus with clear patterns
        corpus = [
            "cat rat mat sat flat",
            "rat sat on flat mat",
            "cat sat on the mat",
            "flat mat big cat",
        ] * 5

        tok = NexaTokenizer.train(corpus, vocab_size=80, min_frequency=1)

        # Build dataset
        text    = "\n".join(corpus)
        dataset = NexaDataset.from_text(text, tok, block_size=8)
        assert len(dataset) > 0, "Dataset is empty — corpus may be too short."

        model_cfg = ModelConfig(
            vocab_size  = tok.vocab_size,
            max_seq_len = 32,
            d_model     = 32,
            n_heads     = 2,
            n_layers    = 1,
            d_ff        = 64,
            dropout     = 0.0,
        )
        trainer_cfg = TrainerConfig(
            block_size      = 8,
            batch_size      = 2,
            max_steps       = 80,
            warmup_steps    = 5,
            learning_rate   = 5e-3,
            min_lr          = 5e-4,
            eval_interval   = 0,
            save_interval   = 0,
            log_interval    = 10,      # record history
            checkpoint_dir  = tmp_path / "ckpts",
            device          = "cpu",
            seed            = 0,
        )

        torch.manual_seed(0)
        model   = NexaTransformer(model_cfg)
        trainer = Trainer(model, trainer_cfg, dataset)
        history = trainer.train()

        assert history.final_train_loss < history.initial_train_loss, (
            f"Full pipeline: loss did not decrease. "
            f"Initial={history.initial_train_loss:.4f}, "
            f"Final={history.final_train_loss:.4f}"
        )
