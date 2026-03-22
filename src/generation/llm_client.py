"""LLM client abstraction with retry/backoff, rate-limit handling, and response caching.

Exposes a uniform `.complete(prompt) -> str` interface over Anthropic Claude and
OpenAI GPT.  Responses are cached to disk keyed on (prompt_hash, model, temperature)
so notebook re-runs are near-free.

Attribution:
    Multi-model comparison (Claude + GPT) motivated by Zhou et al. 2024 §4, which
    benchmarks 10 LLMs across trustworthiness dimensions.
    RAG + LLM pipeline shape for veracity prediction — Singal et al. 2024 §4.
    Cost-bounded experimentation via response caching — Zhou et al. 2024 §2.1.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod

import diskcache

logger = logging.getLogger(__name__)

# HTTP status codes that warrant a retry (transient errors + rate limits).
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 529})
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0  # seconds; wait = _BACKOFF_BASE * 2^attempt


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract LLM client.

    Args:
        model: Model identifier string.
        temperature: Sampling temperature.
        cache_dir: Directory for the ``diskcache`` response store.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        cache_dir: str | os.PathLike = ".cache/llm_responses",
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._cache = diskcache.Cache(str(cache_dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> str:
        """Return the model completion for *prompt*, with caching and retry.

        Cache key: SHA-256 of ``(prompt, model, temperature)``.

        Args:
            prompt: Full prompt string to send to the model.

        Returns:
            Model response text.
        """
        key = _cache_key(prompt, self._model, self._temperature)
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
        """Make a single API call and return the response text."""

    def _complete_with_retry(self, prompt: str) -> str:
        """Call :meth:`_call_api` with exponential backoff on transient errors."""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._call_api(prompt)
            except Exception as exc:
                status = _extract_status(exc)
                # Non-transient HTTP errors (e.g. 401 auth) → fail immediately.
                if status is not None and status not in _RETRYABLE_STATUSES:
                    raise
                if attempt == _MAX_RETRIES:
                    raise
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed (status=%s). Retrying in %.1fs.",
                    attempt + 1, _MAX_RETRIES, status, wait,
                )
                time.sleep(wait)
        raise RuntimeError("unreachable")  # satisfies type checker


# ---------------------------------------------------------------------------
# Concrete clients
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    """Anthropic Claude client.

    Args:
        model: Claude model ID (e.g. ``"claude-haiku-4-5-20251001"``).
        temperature: Sampling temperature.
        cache_dir: Disk cache directory.
        max_tokens: Maximum tokens in the completion.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.0,
        cache_dir: str | os.PathLike = ".cache/llm_responses",
        max_tokens: int = 256,
    ) -> None:
        super().__init__(model=model, temperature=temperature, cache_dir=cache_dir)
        import anthropic  # deferred so tests can mock before import
        self._client = anthropic.Anthropic(max_retries=0)  # retries handled here
        self._max_tokens = max_tokens

    def _call_api(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class OpenAIClient(LLMClient):
    """OpenAI GPT client.

    Args:
        model: OpenAI model ID (e.g. ``"gpt-4o-mini"``).
        temperature: Sampling temperature.
        cache_dir: Disk cache directory.
        max_tokens: Maximum tokens in the completion.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        cache_dir: str | os.PathLike = ".cache/llm_responses",
        max_tokens: int = 256,
    ) -> None:
        super().__init__(model=model, temperature=temperature, cache_dir=cache_dir)
        import openai  # deferred so tests can mock before import
        self._client = openai.OpenAI(max_retries=0)  # retries handled here
        self._max_tokens = max_tokens

    def _call_api(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_key(prompt: str, model: str, temperature: float) -> str:
    """SHA-256 of the ``(prompt, model, temperature)`` triple."""
    raw = f"{prompt}\x00{model}\x00{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _extract_status(exc: Exception) -> int | None:
    """Try to extract an HTTP status code from *exc*."""
    for attr in ("status_code", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None
