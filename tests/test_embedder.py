"""Tests for src/retrieval/embedder.py - no real model download required.

The SentenceTransformer model is mocked so tests run offline and fast.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.embedder import Embedder, _best_device


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------


def test_best_device_prefers_cuda():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.backends.mps.is_available", return_value=True):
        assert _best_device() == "cuda"


def test_best_device_falls_back_to_mps():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        assert _best_device() == "mps"


def test_best_device_falls_back_to_cpu():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert _best_device() == "cpu"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_DIM = 384
_TEXTS = ["Alice lives in Wonderland.", "Bob never left Paris.", "Carol is unknown."]


def _make_fake_model(dim: int = _DIM) -> MagicMock:
    """Return a mock SentenceTransformer that produces deterministic vectors."""
    model = MagicMock()
    model.get_embedding_dimension.return_value = dim

    def _encode(texts, **kwargs):
        rng = np.random.default_rng(seed=hash(tuple(texts)) % (2**31))
        return rng.random((len(texts), dim)).astype(np.float32)

    model.encode.side_effect = _encode
    return model


@pytest.fixture()
def embedder(tmp_path):
    """Embedder with mocked SentenceTransformer and temp cache dir."""
    with patch("src.retrieval.embedder.SentenceTransformer", return_value=_make_fake_model()):
        with Embedder(cache_dir=tmp_path / "emb") as emb:
            yield emb


# ---------------------------------------------------------------------------
# Shape / dtype
# ---------------------------------------------------------------------------


def test_encode_returns_correct_shape(embedder):
    vecs = embedder.encode(_TEXTS)
    assert vecs.shape == (len(_TEXTS), _DIM)


def test_encode_returns_float32(embedder):
    vecs = embedder.encode(_TEXTS)
    assert vecs.dtype == np.float32


def test_encode_empty_list_returns_zero_rows(embedder):
    vecs = embedder.encode([])
    assert vecs.shape == (0, _DIM)


def test_single_text_returns_2d(embedder):
    vecs = embedder.encode(["single passage"])
    assert vecs.ndim == 2
    assert vecs.shape[0] == 1


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------


def test_cache_hit_on_second_call(embedder):
    """Model.encode must NOT be called on the second call for the same texts."""
    embedder.encode(_TEXTS)
    call_count_before = embedder._model.encode.call_count
    embedder.encode(_TEXTS)
    assert embedder._model.encode.call_count == call_count_before, (
        "Model was called again despite all texts being cached"
    )


def test_cache_hit_returns_same_vectors(embedder):
    first = embedder.encode(_TEXTS)
    second = embedder.encode(_TEXTS)
    np.testing.assert_array_equal(first, second)


def test_partial_cache_hit(embedder):
    """Only uncached texts should be forwarded to the model."""
    embedder.encode(_TEXTS[:2])  # cache first two
    call_count_before = embedder._model.encode.call_count

    embedder.encode(_TEXTS)  # third text is new → model called once more
    assert embedder._model.encode.call_count == call_count_before + 1
    # The new call should only have received the uncached text
    last_call_texts = embedder._model.encode.call_args[0][0]
    assert last_call_texts == [_TEXTS[2]]


# ---------------------------------------------------------------------------
# Embedding dimension property
# ---------------------------------------------------------------------------


def test_embedding_dim(embedder):
    assert embedder.embedding_dim == _DIM
