"""
nexa/models/transformer.py
===========================
NexaTransformer — the complete decoder-only language model.

Architecture Summary
--------------------
NexaTransformer is a decoder-only Transformer (GPT-style) built entirely
from scratch. It has the following layers in order:

    Embedding         [vocab_size, d_model]
    Dropout
    × n_layers:
        TransformerBlock (Pre-RMSNorm + MultiHeadAttention + SwiGLU)
    Final RMSNorm     [d_model]
    LM Head           [d_model, vocab_size]  — weight-tied to Embedding

Weight Tying
------------
The LM head weight matrix is set equal to the embedding weight matrix.
This means the same vector space used to ENCODE input tokens is also used
to DECODE output logits. Benefits:
    1. Reduces parameters by vocab_size × d_model (≈ 16M for a 32k vocab)
    2. Improves generalization — the model learns consistent representations
    3. Well-established technique (Press & Wolf, 2016)

Initialization
--------------
All linear layers: normal distribution with std = 0.02.
All embedding layers: normal distribution with std = 0.02.
All norm weights: ones (= identity at start of training).
This closely follows the GPT-2 initialization scheme.

Parameter Budget (default config: d=512, h=8, L=6, ff=2048, V=32000)
----------------------------------------------------------------------
Embedding:      32000 × 512         =  16.4M  (shared with LM head)
Attention/layer: 4 × 512 × 512      =   1.05M
SwiGLU/layer:    3 × 512 × 2048     =   3.15M
Norms/layer:     2 × 512            ≈   0 (negligible)
6 layers:        6 × (1.05 + 3.15)  =  25.2M
Final norm:      512                ≈   0
Total (unique):  16.4 + 25.2        ≈  41.6M parameters

Independence
------------
NexaTransformer is initialized from scratch using standard PyTorch random
initialization. No pretrained weights are loaded at any point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from nexa.models.config import ModelConfig
from nexa.models.norm import build_norm
from nexa.models.block import TransformerBlock
from nexa.utils import get_logger

log = get_logger(__name__, log_to_file=False)


class NexaTransformer(nn.Module):
    """
    Nexa's decoder-only Transformer language model.

    Usage
    -----
    Training (compute logits, then CrossEntropyLoss):

        model = NexaTransformer(config)
        logits = model(input_ids)          # [B, S, vocab_size]
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            input_ids[:, 1:].reshape(-1),
            ignore_index=config.pad_token_id,
        )

    Inference (greedy / sampled generation):

        tokens = model.generate(input_ids, max_new_tokens=100, temperature=0.8)

    Parameters
    ----------
    config : ModelConfig
        Full architecture specification.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # Token embedding: maps integer IDs to d_model-dimensional vectors.
        # padding_idx=pad_token_id means PAD embeddings are zeroed (not updated).
        # ------------------------------------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.embed_dropout = nn.Dropout(config.dropout)

        # ------------------------------------------------------------------
        # N stacked Transformer decoder blocks
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # ------------------------------------------------------------------
        # Final normalization before the LM head
        # (Pre-norm style: the blocks normalize before their sublayers, so the
        # residual stream exiting the last block is NOT normalized. We do it here.)
        # ------------------------------------------------------------------
        self.final_norm = build_norm(config.norm_type, config.d_model, config.norm_eps)

        # ------------------------------------------------------------------
        # LM Head: projects from d_model to vocab_size (logits)
        # No bias — standard for modern language models.
        # ------------------------------------------------------------------
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share the embedding and LM head weight matrix.
        # They operate in the same semantic space, so sharing is beneficial.
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # ------------------------------------------------------------------
        # Initialize all weights
        # ------------------------------------------------------------------
        self._init_weights()

        log.debug(
            "NexaTransformer initialized | params=%s | config=%s",
            f"{self.num_parameters:,}",
            config,
        )

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Initialize model weights.

        Scheme (following GPT-2):
            Linear weights:    Normal(0, 0.02)
            Linear biases:     zeros (rare in our model — most layers are bias=False)
            Embedding weights: Normal(0, 0.02)
            Norm weights:      ones (RMSNorm / LayerNorm scales start at 1)

        The 0.02 std is chosen so that the initial attention logits
        (d_model-dimensional dot products) have variance ≈ 1.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # Zero out padding embedding (if padding_idx is set)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Compute next-token logits for a batch of token ID sequences.

        Parameters
        ----------
        input_ids : Tensor  shape [B, S]
            Batch of integer token ID sequences. All values must be in
            [0, vocab_size). Sequences shorter than max_seq_len are fine;
            the RoPE tables are sliced to the actual sequence length.

        Returns
        -------
        logits : Tensor  shape [B, S, vocab_size]
            Raw (unnormalized) scores for every token at every position.
            logits[:, t, :] = scores for position t+1 (the NEXT token after t).

            To compute training loss:
                loss = CrossEntropy(logits[:, :-1], input_ids[:, 1:])
        """
        B, S = input_ids.shape
        if S > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {S} exceeds max_seq_len={self.config.max_seq_len}. "
                f"Truncate the input or increase max_seq_len in ModelConfig."
            )

        # Step 1: Embed token IDs → continuous vectors
        x = self.embedding(input_ids)      # [B, S, d_model]
        x = self.embed_dropout(x)

        # Step 2: Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)                   # [B, S, d_model]

        # Step 3: Final normalization
        x = self.final_norm(x)             # [B, S, d_model]

        # Step 4: Project to vocabulary logits
        logits = self.lm_head(x)           # [B, S, vocab_size]

        return logits

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        greedy: bool = False,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressively generate new tokens given a prompt.

        This is the "inference loop": at each step, we run the full forward
        pass on all tokens so far, take the logits at the LAST position,
        sample (or argmax) the next token, and append it.

        Parameters
        ----------
        input_ids : Tensor  shape [B, S]
            Prompt token IDs. Use batch size 1 for simple text generation.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        temperature : float
            Sampling temperature.
            - temperature = 1.0: sample from the model's distribution as-is.
            - temperature < 1.0 (e.g., 0.5): sharper, more deterministic.
            - temperature > 1.0 (e.g., 1.5): flatter, more random/creative.
        greedy : bool
            If True, always pick the highest-probability token (argmax).
            Equivalent to temperature → 0. Deterministic.
        eos_token_id : int, optional
            If provided, stop generation when ALL sequences in the batch
            have produced this token.

        Returns
        -------
        Tensor  shape [B, S + n_generated]
            The full sequence (prompt + generated tokens).
            n_generated ≤ max_new_tokens (may be less if EOS was hit).
        """
        self.eval()
        ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len if needed
            # (simple sliding window — more sophisticated KV-cache in Phase 5)
            ctx = ids if ids.shape[1] <= self.config.max_seq_len else ids[:, -self.config.max_seq_len:]

            # Forward pass — only need the last position's logits
            logits = self(ctx)                       # [B, S, vocab_size]
            next_logits = logits[:, -1, :]           # [B, vocab_size]

            if greedy or temperature == 0.0:
                # Greedy: pick the most probable token
                next_token = next_logits.argmax(dim=-1, keepdim=True)    # [B, 1]
            else:
                # Temperature sampling: scale logits, then sample
                scaled_logits = next_logits / max(temperature, 1e-8)
                probs = torch.softmax(scaled_logits, dim=-1)              # [B, vocab_size]
                next_token = torch.multinomial(probs, num_samples=1)      # [B, 1]

            # Append new token to the sequence
            ids = torch.cat([ids, next_token], dim=1)                     # [B, S+1]

            # Early stopping: all sequences have emitted EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return ids

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters_non_embedding(self) -> int:
        """Trainable parameters excluding the embedding (for cleaner scaling laws)."""
        embed_params = sum(p.numel() for p in self.embedding.parameters())
        return self.num_parameters - embed_params

    def save_weights(self, path: Path | str) -> None:
        """
        Save model weights to a file.

        Only saves the state_dict (weights and buffers), NOT the architecture.
        The ModelConfig must be saved separately and passed to __init__ to reload.

        Parameters
        ----------
        path : Path or str
            Destination file path (e.g., "checkpoints/model.pt").
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        log.info("Model weights saved → %s  (%s params)", path, f"{self.num_parameters:,}")

    def load_weights(self, path: Path | str, strict: bool = True) -> None:
        """
        Load model weights from a file previously saved by save_weights().

        Parameters
        ----------
        path : Path or str
            Path to the saved .pt / .pth file.
        strict : bool
            If True (default), the saved keys must exactly match this model's keys.
            Set False to load partial checkpoints (e.g., missing LM head).
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=strict)
        log.info("Model weights loaded ← %s", path)

    def __repr__(self) -> str:
        return (
            f"NexaTransformer("
            f"params={self.num_parameters:,}, "
            f"config={self.config})"
        )
