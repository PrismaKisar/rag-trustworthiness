"""Tests for src/retrieval/index.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.index import FaissIndex

_DIM = 8
_N = 10
_RNG = np.random.default_rng(42)


def _unit_vecs(n: int, dim: int = _DIM) -> np.ndarray:
    """Return *n* L2-normalised float32 vectors."""
    vecs = _RNG.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture()
def index_with_data() -> FaissIndex:
    idx = FaissIndex(dim=_DIM)
    idx.add(_unit_vecs(_N))
    return idx


# ---------------------------------------------------------------------------
# add / ntotal
# ---------------------------------------------------------------------------


def test_ntotal_after_add():
    idx = FaissIndex(dim=_DIM)
    idx.add(_unit_vecs(5))
    assert idx.ntotal == 5


def test_ntotal_accumulates_across_add_calls():
    idx = FaissIndex(dim=_DIM)
    idx.add(_unit_vecs(3))
    idx.add(_unit_vecs(4))
    assert idx.ntotal == 7


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_returns_k_results(index_with_data):
    results = index_with_data.search(_unit_vecs(1)[0], k=3)
    assert len(results) == 3


def test_search_returns_valid_indices(index_with_data):
    results = index_with_data.search(_unit_vecs(1)[0], k=5)
    assert all(0 <= i < _N for i in results)


def test_search_k_larger_than_ntotal_returns_ntotal(index_with_data):
    """k > ntotal should return ntotal results, not raise."""
    results = index_with_data.search(_unit_vecs(1)[0], k=_N + 100)
    assert len(results) == _N


def test_search_empty_index_returns_empty():
    idx = FaissIndex(dim=_DIM)
    assert idx.search(_unit_vecs(1)[0], k=5) == []


def test_top1_is_self(index_with_data):
    """A vector added to the index should be its own nearest neighbour."""
    # Re-build with a fixed set so we know the exact vectors
    idx = FaissIndex(dim=_DIM)
    vecs = _unit_vecs(_N)
    idx.add(vecs)
    result = idx.search(vecs[3], k=1)
    assert result[0] == 3


def test_search_returns_list_of_int(index_with_data):
    results = index_with_data.search(_unit_vecs(1)[0], k=3)
    assert isinstance(results, list)
    assert all(isinstance(i, int) for i in results)

