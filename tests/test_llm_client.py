"""Tests for src/generation/llm_client.py.

Assertions:
- complete() returns the model response text (new tokens only, not input).
- Second call with same prompt is served from cache (model not called again).
- Cache key is sensitive to model, temperature, and max_tokens.
- Greedy decoding when temperature == 0; sampling when temperature > 0.
- truncation=True is always passed to the tokenizer.
- pad_token_id is set to eos_token_id when missing.
- apply_chat_template is called to format the prompt as a chat message.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import torch

from src.generation.llm_client import HuggingFaceClient, _cache_key

_INPUT_LEN = 10  # simulated number of input tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    tmp_path,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> tuple[HuggingFaceClient, MagicMock, MagicMock]:
    """Instantiate HuggingFaceClient with mocked tokenizer and causal model."""
    with (
        patch("src.generation.llm_client.AutoTokenizer") as mock_tok_cls,
        patch("src.generation.llm_client.AutoModelForCausalLM") as mock_model_cls,
        patch("src.generation.llm_client._get_device", return_value="cpu"),
    ):
        mock_tokenizer = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.pad_token_id = 0  # already set - no override needed
        mock_tokenizer.eos_token_id = 1

        mock_hf_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_hf_model.to.return_value = mock_hf_model

        client = HuggingFaceClient(
            model=model,
            temperature=temperature,
            cache_dir=tmp_path / "llm",
            max_tokens=max_tokens,
        )

    client._tokenizer = mock_tokenizer
    client._hf_model = mock_hf_model
    client._device = "cpu"
    return client, mock_tokenizer, mock_hf_model


def _setup_generate(mock_tokenizer: MagicMock, mock_hf_model: MagicMock, text: str) -> None:
    """Configure mocks for a chat-template → generate → decode round-trip."""
    mock_tokenizer.apply_chat_template.return_value = "<formatted prompt>"

    # Simulate tokenizer returning _INPUT_LEN input token ids
    fake_input_ids = torch.zeros(1, _INPUT_LEN, dtype=torch.long)
    mock_tokenizer.return_value = {
        "input_ids": fake_input_ids,
        "attention_mask": torch.ones(1, _INPUT_LEN, dtype=torch.long),
    }

    # generate returns input_ids + new_ids (total length > _INPUT_LEN)
    new_token_count = 5
    fake_output_ids = torch.zeros(1, _INPUT_LEN + new_token_count, dtype=torch.long)
    mock_hf_model.generate.return_value = fake_output_ids

    mock_tokenizer.decode.return_value = text


# ---------------------------------------------------------------------------
# HuggingFaceClient - basic completion
# ---------------------------------------------------------------------------


class TestHuggingFaceClientComplete:
    def test_returns_text(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path)
        _setup_generate(tok, mdl, "SUPPORTS")
        assert client.complete("Is Paris the capital of France?") == "SUPPORTS"
        mdl.generate.assert_called_once()

    def test_cache_hit_skips_model(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path)
        _setup_generate(tok, mdl, "REFUTES")
        first = client.complete("same prompt")
        second = client.complete("same prompt")
        assert first == second == "REFUTES"
        mdl.generate.assert_called_once()  # second call served from cache

    def test_different_prompts_call_model_twice(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path)
        tok.apply_chat_template.return_value = "<formatted>"
        fake_input_ids = torch.zeros(1, _INPUT_LEN, dtype=torch.long)
        tok.return_value = {
            "input_ids": fake_input_ids,
            "attention_mask": torch.ones(1, _INPUT_LEN, dtype=torch.long),
        }
        fake_output = torch.zeros(1, _INPUT_LEN + 3, dtype=torch.long)
        mdl.generate.return_value = fake_output
        tok.decode.side_effect = ["SUPPORTS", "REFUTES"]
        assert client.complete("prompt A") == "SUPPORTS"
        assert client.complete("prompt B") == "REFUTES"
        assert mdl.generate.call_count == 2

    def test_strips_whitespace_from_output(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path)
        _setup_generate(tok, mdl, "  SUPPORTS\n")
        assert client.complete("any prompt") == "SUPPORTS"

    def test_decode_called_on_new_tokens_only(self, tmp_path):
        """decode must receive only the sliced new tokens, not the full sequence."""
        client, tok, mdl = _make_client(tmp_path)
        _setup_generate(tok, mdl, "SUPPORTS")
        client.complete("test")
        decode_call_arg = tok.decode.call_args[0][0]
        # The decoded tensor must have length == total_output - input_length
        assert len(decode_call_arg) == 5  # new_token_count set in _setup_generate


# ---------------------------------------------------------------------------
# HuggingFaceClient - chat template
# ---------------------------------------------------------------------------


class TestChatTemplate:
    def test_apply_chat_template_called(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path)
        _setup_generate(tok, mdl, "SUPPORTS")
        client.complete("my prompt")
        tok.apply_chat_template.assert_called_once()
        call_args = tok.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages == [{"role": "user", "content": "my prompt"}]

    def test_add_generation_prompt_true(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path)
        _setup_generate(tok, mdl, "SUPPORTS")
        client.complete("prompt")
        kwargs = tok.apply_chat_template.call_args[1]
        assert kwargs.get("add_generation_prompt") is True


# ---------------------------------------------------------------------------
# HuggingFaceClient - pad_token_id fallback
# ---------------------------------------------------------------------------


class TestPadTokenId:
    def test_pad_token_set_to_eos_when_missing(self, tmp_path):
        with (
            patch("src.generation.llm_client.AutoTokenizer") as mock_tok_cls,
            patch("src.generation.llm_client.AutoModelForCausalLM") as mock_model_cls,
            patch("src.generation.llm_client._get_device", return_value="cpu"),
        ):
            mock_tokenizer = MagicMock()
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            mock_tokenizer.pad_token_id = None   # missing - should be overridden
            mock_tokenizer.eos_token_id = 42

            mock_hf_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_hf_model
            mock_hf_model.to.return_value = mock_hf_model

            client = HuggingFaceClient(cache_dir=tmp_path / "llm")
            client._load_model()  # lazy - must be triggered explicitly to exercise fallback

        assert mock_tokenizer.pad_token_id == 42

    def test_pad_token_not_overridden_when_present(self, tmp_path):
        with (
            patch("src.generation.llm_client.AutoTokenizer") as mock_tok_cls,
            patch("src.generation.llm_client.AutoModelForCausalLM") as mock_model_cls,
            patch("src.generation.llm_client._get_device", return_value="cpu"),
        ):
            mock_tokenizer = MagicMock()
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            mock_tokenizer.pad_token_id = 7   # already set
            mock_tokenizer.eos_token_id = 99

            mock_hf_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_hf_model
            mock_hf_model.to.return_value = mock_hf_model

            client = HuggingFaceClient(cache_dir=tmp_path / "llm")
            client._load_model()  # lazy - must be triggered explicitly

        assert mock_tokenizer.pad_token_id == 7  # unchanged


# ---------------------------------------------------------------------------
# HuggingFaceClient - generation parameters
# ---------------------------------------------------------------------------


class TestHuggingFaceClientGenerationParams:
    def test_greedy_when_temperature_zero(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path, temperature=0.0)
        _setup_generate(tok, mdl, "SUPPORTS")
        client.complete("greedy prompt")
        gen_kwargs = mdl.generate.call_args[1]
        assert gen_kwargs.get("do_sample") is False

    def test_sampling_when_temperature_nonzero(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path, temperature=0.7)
        _setup_generate(tok, mdl, "SUPPORTS")
        client.complete("sample prompt")
        gen_kwargs = mdl.generate.call_args[1]
        assert gen_kwargs.get("do_sample") is True
        assert gen_kwargs.get("temperature") == pytest.approx(0.7)

    def test_truncation_always_enabled(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path)
        _setup_generate(tok, mdl, "SUPPORTS")
        client.complete("long prompt")
        tok_call_kwargs = tok.call_args[1]
        assert tok_call_kwargs.get("truncation") is True

    def test_max_tokens_passed_to_generate(self, tmp_path):
        client, tok, mdl = _make_client(tmp_path, max_tokens=64)
        _setup_generate(tok, mdl, "SUPPORTS")
        client.complete("prompt")
        gen_kwargs = mdl.generate.call_args[1]
        assert gen_kwargs.get("max_new_tokens") == 64


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_deterministic(self):
        assert _cache_key("p", "m", 0.0, 64) == _cache_key("p", "m", 0.0, 64)

    def test_sensitive_to_model(self):
        assert _cache_key("p", "model-a", 0.0, 64) != _cache_key("p", "model-b", 0.0, 64)

    def test_sensitive_to_temperature(self):
        assert _cache_key("p", "m", 0.0, 64) != _cache_key("p", "m", 0.7, 64)

    def test_sensitive_to_prompt(self):
        assert _cache_key("prompt-1", "m", 0.0, 64) != _cache_key("prompt-2", "m", 0.0, 64)

    def test_sensitive_to_max_tokens(self):
        assert _cache_key("p", "m", 0.0, 64) != _cache_key("p", "m", 0.0, 256)


# ---------------------------------------------------------------------------
# LLMClient abstract base - _max_tokens must be declared in the base
# ---------------------------------------------------------------------------


class TestLLMClientBase:
    """Any minimal subclass must work without re-declaring _max_tokens."""

    def _minimal_client(self, tmp_path, max_tokens=256):
        from src.generation.llm_client import LLMClient

        class MinimalClient(LLMClient):
            def _call_api(self, prompt: str, max_tokens=None) -> str:
                return "ok"

        return MinimalClient(model="test-model", cache_dir=tmp_path / "llm", max_tokens=max_tokens)

    def test_complete_without_explicit_max_tokens_does_not_raise(self, tmp_path):
        """complete() must not raise AttributeError when max_tokens is omitted."""
        client = self._minimal_client(tmp_path)
        assert client.complete("hello") == "ok"

    def test_base_default_max_tokens_respected(self, tmp_path):
        """The max_tokens passed to __init__ must propagate to _call_api."""
        from src.generation.llm_client import LLMClient

        received: list[int] = []

        class CapturingClient(LLMClient):
            def _call_api(self, prompt: str, max_tokens=None) -> str:
                received.append(max_tokens)
                return "ok"

        client = CapturingClient(model="m", cache_dir=tmp_path / "llm", max_tokens=128)
        client.complete("hello")
        assert received == [128]

    def test_explicit_max_tokens_overrides_default(self, tmp_path):
        """max_tokens passed to complete() must win over the instance default."""
        from src.generation.llm_client import LLMClient

        received: list[int] = []

        class CapturingClient(LLMClient):
            def _call_api(self, prompt: str, max_tokens=None) -> str:
                received.append(max_tokens)
                return "ok"

        client = CapturingClient(model="m", cache_dir=tmp_path / "llm", max_tokens=64)
        client.complete("hello", max_tokens=512)
        assert received == [512]


# ---------------------------------------------------------------------------
# HuggingFaceClient - close() / memory cleanup
# ---------------------------------------------------------------------------


class TestHuggingFaceClientClose:
    """close() must release model references and trigger gc.collect()."""

    def _make_loaded_client(self, tmp_path, device: str = "cpu") -> HuggingFaceClient:
        client, _, _ = _make_client(tmp_path)
        client._device = device
        return client

    def test_close_sets_model_and_tokenizer_to_none(self, tmp_path):
        client = self._make_loaded_client(tmp_path)
        assert client._hf_model is not None
        client.close()
        assert client._hf_model is None
        assert client._tokenizer is None

    def test_close_calls_gc_collect(self, tmp_path):
        client = self._make_loaded_client(tmp_path)
        with patch("src.generation.llm_client.gc") as mock_gc:
            client.close()
        mock_gc.collect.assert_called_once()

    def test_close_mps_synchronizes_before_empty_cache(self, tmp_path):
        client = self._make_loaded_client(tmp_path, device="mps")
        call_order: list[str] = []
        with (
            patch("torch.mps.synchronize", side_effect=lambda: call_order.append("sync")),
            patch("torch.mps.empty_cache", side_effect=lambda: call_order.append("empty")),
            patch("src.generation.llm_client.gc"),
        ):
            client.close()
        assert call_order == ["sync", "empty"], "synchronize must precede empty_cache on MPS"

    def test_close_cuda_synchronizes_before_empty_cache(self, tmp_path):
        client = self._make_loaded_client(tmp_path, device="cuda")
        call_order: list[str] = []
        with (
            patch("torch.cuda.synchronize", side_effect=lambda: call_order.append("sync")),
            patch("torch.cuda.empty_cache", side_effect=lambda: call_order.append("empty")),
            patch("src.generation.llm_client.gc"),
        ):
            client.close()
        assert call_order == ["sync", "empty"], "synchronize must precede empty_cache on CUDA"

    def test_close_idempotent(self, tmp_path):
        """Calling close() twice must not raise."""
        client = self._make_loaded_client(tmp_path)
        with patch("src.generation.llm_client.gc"):
            client.close()
            client.close()  # second call must be a no-op, not an error

    def test_context_manager_triggers_close(self, tmp_path):
        client, _, _ = _make_client(tmp_path)
        with patch.object(client, "close") as mock_close:
            with client:
                pass
        mock_close.assert_called_once()
