"""
nexa/inference/generator.py
============================
Generator — the full-featured text generation interface for NexaTransformer.

This class wraps a trained NexaTransformer and NexaTokenizer to provide a
high-level generate(prompt, config) → str interface.

Architecture
------------
The generation loop is an **autoregressive** process:

    Step 0: Encode prompt → token IDs
    Loop:
        1. Forward pass: model(all_ids_so_far) → logits [1, S, V]
        2. Take logits at the LAST position: logits[0, -1, :] → [V]
        3. Sample next token using SamplingConfig
        4. Append next token to the sequence
        5. If next_token == eos_token_id → stop
        6. If len(generated) == max_new_tokens → stop
    Step N: Decode generated IDs → text

Why is it slow? Each step requires a full forward pass (O(S²) attention).
Future phases will add KV-caching (O(S) per step) to speed this up.

Independence
------------
No pretrained models loaded. The Generator uses whatever NexaTransformer
weights you provide (trained from scratch in Phase 4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from nexa.inference.sampler import SamplingConfig, sample_next_token
from nexa.utils import get_logger

if TYPE_CHECKING:
    from nexa.models import NexaTransformer, ModelConfig
    from nexa.tokenizer import NexaTokenizer

log = get_logger(__name__, log_to_file=False)


@dataclass
class GenerationResult:
    """
    Structured result from a single generation call.

    Attributes
    ----------
    prompt : str
        The original input prompt.
    generated_text : str
        The newly generated text (not including the prompt).
    full_text : str
        prompt + generated_text (full output).
    prompt_tokens : int
        Number of tokens in the encoded prompt.
    generated_tokens : int
        Number of newly generated tokens.
    stopped_by : str
        Why generation stopped: "eos", "max_new_tokens", or "max_seq_len".
    """
    prompt:           str
    generated_text:   str
    full_text:        str
    prompt_tokens:    int
    generated_tokens: int
    stopped_by:       str

    def __str__(self) -> str:
        return self.full_text

    def __repr__(self) -> str:
        return (
            f"GenerationResult("
            f"prompt_tokens={self.prompt_tokens}, "
            f"generated={self.generated_tokens}, "
            f"stopped_by='{self.stopped_by}')"
        )


class Generator:
    """
    High-level text generation interface for NexaTransformer.

    Combines a trained model, its tokenizer, and a SamplingConfig to
    produce text from a prompt string.

    Parameters
    ----------
    model : NexaTransformer
        A trained language model. Should be in eval() mode.
    tokenizer : NexaTokenizer
        The tokenizer used to train the model (must share the same vocabulary).
    device : str
        "auto" → cuda > mps > cpu. Or explicitly "cpu", "cuda", "mps".

    Usage
    -----
    # Build from in-memory objects (typical after training):
    generator = Generator(model, tokenizer)

    # Greedy generation:
    result = generator.generate("the cat sat", SamplingConfig.greedy_config())

    # Sampling with nucleus + repetition penalty:
    config = SamplingConfig(temperature=0.8, top_p=0.9, repetition_penalty=1.2,
                            max_new_tokens=50)
    result = generator.generate("the cat sat", config)
    print(result.generated_text)

    # Load from saved checkpoint:
    generator = Generator.from_checkpoint("checkpoints/", "data/processed/tokenizer")
    """

    def __init__(
        self,
        model:     nn.Module,
        tokenizer: "NexaTokenizer",
        device:    str = "auto",
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = self._resolve_device(device)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Primary generation API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt:      str,
        config:      Optional[SamplingConfig] = None,
        return_full: bool = False,
    ) -> GenerationResult:
        """
        Generate text from a natural-language prompt.

        Parameters
        ----------
        prompt : str
            The input text. Will be tokenized and prepended with <bos>.
        config : SamplingConfig, optional
            Sampling hyperparameters. Defaults to balanced sampling.
        return_full : bool
            Unused (full_text is always in GenerationResult). Kept for
            backwards compatibility.

        Returns
        -------
        GenerationResult
            Contains prompt, generated_text, full_text, and metadata.
        """
        config = config or SamplingConfig.default_sampling()

        # Encode the prompt (with BOS, without EOS)
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids  = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Run generation loop
        output_ids, stopped_by = self._generation_loop(input_ids, config)

        # Decode results
        full_ids       = output_ids[0].tolist()
        new_ids        = full_ids[len(prompt_ids):]

        generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        full_text      = self.tokenizer.decode(
            [i for i in full_ids if i not in
             (self.tokenizer.vocab.token_to_id("<bos>"),)],
            skip_special_tokens=True,
        )

        return GenerationResult(
            prompt           = prompt,
            generated_text   = generated_text,
            full_text        = full_text,
            prompt_tokens    = len(prompt_ids),
            generated_tokens = len(new_ids),
            stopped_by       = stopped_by,
        )

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: Tensor,
        config:    Optional[SamplingConfig] = None,
    ) -> Tensor:
        """
        Generate token IDs autoregressively from a prompt token ID tensor.

        Lower-level API: operates on raw token IDs rather than text strings.
        Useful for testing and for batch generation pipelines.

        Parameters
        ----------
        input_ids : Tensor  shape [1, S]
            Prompt token IDs. Batch size must be 1.
        config : SamplingConfig, optional
            Sampling hyperparameters.

        Returns
        -------
        Tensor  shape [1, S + n_generated]
            Full sequence including prompt and generated tokens.
        """
        config = config or SamplingConfig.default_sampling()
        ids, _ = self._generation_loop(input_ids.to(self.device), config)
        return ids

    # ------------------------------------------------------------------
    # Core autoregressive generation loop
    # ------------------------------------------------------------------

    def _generation_loop(
        self,
        input_ids: Tensor,
        config:    SamplingConfig,
    ) -> tuple[Tensor, str]:
        """
        The autoregressive decoding loop.

        At each step:
        1. Forward pass on the current token sequence.
        2. Extract logits at the last position.
        3. Sample the next token using SamplingConfig.
        4. Append to the sequence and check stopping conditions.

        Parameters
        ----------
        input_ids : Tensor  shape [1, S]
        config : SamplingConfig

        Returns
        -------
        (output_ids, stopped_by)
            output_ids : Tensor  shape [1, S + n_generated]
            stopped_by : str — "eos", "max_new_tokens", or "max_seq_len"
        """
        ids      = input_ids.clone()
        max_seq  = self.model.config.max_seq_len
        stopped  = "max_new_tokens"

        for _ in range(config.max_new_tokens):
            # Truncate context to the model's maximum sequence length
            # (simple sliding window — no KV-cache yet)
            ctx = ids[:, -max_seq:]

            # Forward pass — we only need the final position's logits
            logits      = self.model(ctx)           # [1, S, vocab_size]
            next_logits = logits[0, -1, :]          # [vocab_size]

            # Sample next token
            next_id = sample_next_token(next_logits, ids[0], config)

            # Append to sequence
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            ids = torch.cat([ids, next_tensor], dim=1)

            # Check EOS condition
            if config.eos_token_id is not None and next_id == config.eos_token_id:
                stopped = "eos"
                break

            # Check max sequence length condition
            if ids.shape[1] >= max_seq:
                stopped = "max_seq_len"
                break

        return ids, stopped

    # ------------------------------------------------------------------
    # Factory: load from checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        tokenizer_dir:  str | Path,
        device:         str = "auto",
    ) -> "Generator":
        """
        Load a Generator from a training output directory.

        Expects the directory to contain:
            - nexa_final.pt       (model weights, from scripts/train.py)
            - model_config.json   (ModelConfig as JSON, saved alongside weights)

        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory containing nexa_final.pt and model_config.json.
        tokenizer_dir : str or Path
            Directory containing tokenizer.json (from NexaTokenizer.save()).
        device : str
            Device to load model onto.

        Returns
        -------
        Generator

        Raises
        ------
        FileNotFoundError
            If nexa_final.pt or model_config.json are missing.
        """
        from nexa.models import NexaTransformer, ModelConfig
        from nexa.tokenizer import NexaTokenizer

        checkpoint_dir = Path(checkpoint_dir)
        weights_path   = checkpoint_dir / "nexa_final.pt"
        config_path    = checkpoint_dir / "model_config.json"

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}\n"
                "Run scripts/train.py first to generate model weights."
            )
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {config_path}\n"
                "Run scripts/train.py first to generate model_config.json."
            )

        # Load ModelConfig
        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        model_cfg   = ModelConfig.from_dict(config_dict)

        # Build model from scratch (no pretrained weights — only OUR weights)
        model = NexaTransformer(model_cfg)
        model.load_weights(weights_path)
        log.info("Model loaded ← %s  (%s params)", weights_path, f"{model.num_parameters:,}")

        # Load tokenizer
        tokenizer = NexaTokenizer.load(tokenizer_dir)
        log.info("Tokenizer loaded ← %s  (vocab_size=%d)", tokenizer_dir, tokenizer.vocab_size)

        return cls(model, tokenizer, device=device)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def __repr__(self) -> str:
        return (
            f"Generator("
            f"device={self.device}, "
            f"vocab={self.tokenizer.vocab_size}, "
            f"params={getattr(self.model, 'num_parameters', '?'):,})"
            if hasattr(self.model, 'num_parameters') else
            f"Generator(device={self.device})"
        )
