"""Tests for src/generation/llm_client.py (step 13).

Assertions:
- complete() returns the mocked response text.
- Second call is served from cache (API not called again).
- Retry behavior: API called multiple times on transient (429) errors.
- Non-retryable error (401) raises immediately without retry.
- Cache key is sensitive to model and temperature.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.generation.llm_client import AnthropicClient, OpenAIClient, _cache_key


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _anthropic_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.content[0].text = text
    return mock


def _openai_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = text
    return mock


def _rate_limit_error() -> Exception:
    exc = Exception("rate limit exceeded")
    exc.status_code = 429  # type: ignore[attr-defined]
    return exc


def _auth_error() -> Exception:
    exc = Exception("invalid api key")
    exc.status_code = 401  # type: ignore[attr-defined]
    return exc


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------


class TestAnthropicClientComplete:
    def test_returns_text(self, tmp_path):
        client = AnthropicClient(cache_dir=tmp_path / "llm")
        with patch.object(
            client._client.messages, "create",
            return_value=_anthropic_response("SUPPORTS"),
        ) as mock_api:
            result = client.complete("Is Paris the capital of France?")
        assert result == "SUPPORTS"
        mock_api.assert_called_once()

    def test_cache_hit_skips_api(self, tmp_path):
        client = AnthropicClient(cache_dir=tmp_path / "llm")
        with patch.object(
            client._client.messages, "create",
            return_value=_anthropic_response("REFUTES"),
        ) as mock_api:
            first = client.complete("same prompt")
            second = client.complete("same prompt")
        assert first == second == "REFUTES"
        mock_api.assert_called_once()  # second call served from cache

    def test_different_prompts_call_api_twice(self, tmp_path):
        client = AnthropicClient(cache_dir=tmp_path / "llm")
        with patch.object(
            client._client.messages, "create",
            side_effect=[_anthropic_response("SUPPORTS"), _anthropic_response("REFUTES")],
        ) as mock_api:
            r1 = client.complete("prompt A")
            r2 = client.complete("prompt B")
        assert r1 == "SUPPORTS"
        assert r2 == "REFUTES"
        assert mock_api.call_count == 2


class TestAnthropicClientRetry:
    def test_retries_on_rate_limit_then_succeeds(self, tmp_path):
        client = AnthropicClient(cache_dir=tmp_path / "llm_retry")
        side_effects = [_rate_limit_error(), _rate_limit_error(), _anthropic_response("SUPPORTS")]
        with patch.object(client._client.messages, "create", side_effect=side_effects) as mock_api:
            with patch("src.generation.llm_client.time.sleep"):
                result = client.complete("retry prompt")
        assert result == "SUPPORTS"
        assert mock_api.call_count == 3

    def test_no_retry_on_auth_error(self, tmp_path):
        client = AnthropicClient(cache_dir=tmp_path / "llm_auth")
        with patch.object(
            client._client.messages, "create", side_effect=_auth_error(),
        ) as mock_api:
            with patch("src.generation.llm_client.time.sleep"):
                with pytest.raises(Exception, match="invalid api key"):
                    client.complete("fail prompt")
        mock_api.assert_called_once()  # raised immediately, no retry

    def test_retry_exhaustion_raises(self, tmp_path):
        """After MAX_RETRIES+1 transient failures the last exception is raised."""
        client = AnthropicClient(cache_dir=tmp_path / "llm_exhaust")
        side_effects = [_rate_limit_error()] * 5  # 1 initial + 4 retries = 5
        with patch.object(client._client.messages, "create", side_effect=side_effects) as mock_api:
            with patch("src.generation.llm_client.time.sleep"):
                with pytest.raises(Exception, match="rate limit"):
                    client.complete("exhaust prompt")
        assert mock_api.call_count == 5

    @pytest.mark.parametrize("status", [500, 503])
    def test_retries_on_server_error(self, tmp_path, status):
        """Transient server errors (500, 503) are retried like 429."""
        client = AnthropicClient(cache_dir=tmp_path / f"llm_{status}")
        err = Exception(f"server error {status}")
        err.status_code = status  # type: ignore[attr-defined]
        side_effects = [err, _anthropic_response("SUPPORTS")]
        with patch.object(client._client.messages, "create", side_effect=side_effects):
            with patch("src.generation.llm_client.time.sleep"):
                result = client.complete("server error prompt")
        assert result == "SUPPORTS"

    def test_no_retry_on_programming_error(self, tmp_path):
        """Exceptions without HTTP status (e.g. TypeError) must not be retried."""
        client = AnthropicClient(cache_dir=tmp_path / "llm_prog")
        with patch.object(
            client._client.messages, "create", side_effect=TypeError("bad arg"),
        ) as mock_api:
            with patch("src.generation.llm_client.time.sleep"):
                with pytest.raises(TypeError, match="bad arg"):
                    client.complete("prog error prompt")
        mock_api.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


class TestOpenAIClientComplete:
    """openai.OpenAI() validates the API key at construction time, so we patch
    the constructor to avoid requiring OPENAI_API_KEY in CI/tests."""

    def test_returns_text(self, tmp_path):
        with patch("openai.OpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            client = OpenAIClient(cache_dir=tmp_path / "llm_oai")
            mock_instance.chat.completions.create.return_value = _openai_response("NOT ENOUGH INFO")
            result = client.complete("openai prompt")
        assert result == "NOT ENOUGH INFO"
        mock_instance.chat.completions.create.assert_called_once()

    def test_cache_hit_skips_api(self, tmp_path):
        with patch("openai.OpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            client = OpenAIClient(cache_dir=tmp_path / "llm_oai_cache")
            mock_instance.chat.completions.create.return_value = _openai_response("SUPPORTS")
            first = client.complete("cached openai prompt")
            second = client.complete("cached openai prompt")
        assert first == second == "SUPPORTS"
        mock_instance.chat.completions.create.assert_called_once()  # second call from cache


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_deterministic(self):
        assert _cache_key("p", "m", 0.0) == _cache_key("p", "m", 0.0)

    def test_sensitive_to_model(self):
        assert _cache_key("p", "model-a", 0.0) != _cache_key("p", "model-b", 0.0)

    def test_sensitive_to_temperature(self):
        assert _cache_key("p", "m", 0.0) != _cache_key("p", "m", 0.7)

    def test_sensitive_to_prompt(self):
        assert _cache_key("prompt-1", "m", 0.0) != _cache_key("prompt-2", "m", 0.0)
