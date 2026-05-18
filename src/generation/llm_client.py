"""LLM client abstraction with disk caching.

Exposes a uniform `.complete(prompt, prompt_type) -> str` interface over local
Hugging Face causal (decoder-only) instruction-tuned models.  Responses are
cached to disk keyed on (prompt, model, prompt_type) so notebook re-runs
are near-free.

Attribution:
    Multi-model comparison motivated by Zhou et al. 2024 §4, which benchmarks
    multiple LLMs across trustworthiness dimensions.
    RAG + LLM pipeline shape for veracity prediction - Singal et al. 2024 §4.
    Cost-bounded experimentation via response caching - Zhou et al. 2024 §2.1.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod

import diskcache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_BACKOFF_BASE = 5.0  # seconds; wait = _BACKOFF_BASE * 2^attempt
_MAX_NEW_TOKENS = 512  # fixed generation budget; large enough for all prompt types


def _get_device() -> str:
    """Return the best available device string for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract LLM client.

    Args:
        model: Model identifier string.
        temperature: Sampling temperature.
        cache_dir: Directory for the ``diskcache`` response store.
        requests_per_minute: Optional rate limit (``None`` = unlimited).
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        cache_dir: str | os.PathLike = ".cache/llm_responses",
        requests_per_minute: int | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._cache = diskcache.Cache(str(cache_dir))
        self._min_interval = (60.0 / requests_per_minute) if requests_per_minute else 0.0
        self._last_call_time: float = 0.0
        self._rate_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, prompt: str, prompt_type: str = "standard") -> str:
        """Return the model completion for *prompt*, with caching and retry.

        Cache key: SHA-256 of ``(prompt, model, prompt_type)``.
        """
        key = _cache_key(prompt, self._model, prompt_type)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("Cache hit  model=%s key=%.8s", self._model, key)
            return cached
        response = self._complete_with_retry(prompt)
        self._cache.set(key, response)
        return response

    def close(self) -> None:
        """Flush and close the disk cache."""
        self._cache.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Make a single model call and return the response text."""

    def _complete_with_retry(self, prompt: str) -> str:
        """Call :meth:`_call_api` with proactive throttling and exponential backoff.

        Thread-safe: multiple threads share the rate limit via slot reservation.
        """
        wait = 0.0
        if self._min_interval:
            with self._rate_lock:
                now = time.time()
                gap = now - self._last_call_time
                wait = max(0.0, self._min_interval - gap)
                self._last_call_time = now + wait
        if wait:
            time.sleep(wait)
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._call_api(prompt)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if not isinstance(status, int) or status not in {429, 500, 502, 503, 529}:
                    raise
                if attempt == _MAX_RETRIES:
                    raise
                sleep_s = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed (status=%s). Retrying in %.1fs.",
                    attempt + 1, _MAX_RETRIES, status, sleep_s,
                )
                time.sleep(sleep_s)
        raise RuntimeError("unreachable")  # satisfies type checker


# ---------------------------------------------------------------------------
# Concrete client
# ---------------------------------------------------------------------------


class HuggingFaceClient(LLMClient):
    """Local Hugging Face causal LM client (decoder-only, instruction-tuned).

    Uses AutoTokenizer + AutoModelForCausalLM with chat template formatting.
    Compatible with any instruction-tuned model that exposes a chat template
    (Qwen2.5-Instruct, Llama-3.2-Instruct, SmolLM2-Instruct, Gemma-it, Phi-3.5, …).

    Args:
        model: HuggingFace model ID (e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``).
        temperature: Sampling temperature. ``0.0`` = greedy / deterministic.
        cache_dir: Disk cache directory.
        max_tokens: Default max new tokens per generation.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        temperature: float = 0.0,
        cache_dir: str | os.PathLike = ".cache/llm_responses",
    ) -> None:
        super().__init__(model=model, temperature=temperature, cache_dir=cache_dir)
        self._device = _get_device()
        self._hf_model = None
        self._tokenizer = None
        self._inference_lock = threading.Lock()

    def _load_model(self) -> None:
        dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self._model)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        # low_cpu_mem_usage loads shards sequentially to avoid the full fp32
        # copy in RAM; explicit .to() then moves to target device in one step.
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self._model, dtype=dtype, low_cpu_mem_usage=True
        ).to(self._device)

    def close(self) -> None:
        if self._hf_model is not None:
            del self._hf_model
            del self._tokenizer
            self._hf_model = None
            self._tokenizer = None
            gc.collect()
            if self._device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        super().close()

    def _call_api(self, prompt: str) -> str:
        with self._inference_lock:
            if self._hf_model is None:
                self._load_model()
            return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048
        )
        input_length = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        gen_kwargs: dict = {
            "max_new_tokens": _MAX_NEW_TOKENS,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self._temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self._temperature
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            output_ids = self._hf_model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][input_length:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_key(prompt: str, model: str, prompt_type: str) -> str:
    """SHA-256 of the ``(prompt, model, prompt_type)`` tuple."""
    raw = f"{prompt}\x00{model}\x00{prompt_type}"
    return hashlib.sha256(raw.encode()).hexdigest()
