"""Tests for src/retrieval/retriever.py (step 11).

Assertions:
- retrieve() returns exactly K results.
- Precision@k is computable (gold_indices available from RetrievalCorpus).
- RuntimeError raised when build() has not been called.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.corpus import RetrievalCorpus
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PASSAGES = [
    "Paris is the capital of France.",
    "The Eiffel Tower is located in Paris.",
    "Berlin is the capital of Germany.",
    "The Amazon river is in South America.",
    "Water boils at 100 degrees Celsius.",
    "Dogs are domesticated mammals.",
    "The moon orbits the Earth.",
    "Python is a programming language.",
]

GOLD_INDICES = {0, 1}  # first two passages are "gold"


@pytest.fixture(scope="session")
def embedder(tmp_path_factory):
    emb = Embedder(cache_dir=tmp_path_factory.mktemp("emb_cache"))
    yield emb
    emb.close()


@pytest.fixture(scope="session")
def corpus():
    return RetrievalCorpus(passages=PASSAGES, gold_indices=GOLD_INDICES)


@pytest.fixture(scope="session")
def built_retriever(embedder, corpus):
    r = Retriever(embedder=embedder, k=3)
    r.build(corpus)
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetrieverReturnsExactlyK:
    def test_returns_k_default(self, built_retriever):
        results = built_retriever.retrieve("What is the capital of France?")
        assert len(results) == 3

    def test_returns_k_override(self, built_retriever):
        for k in (1, 2, 5):
            results = built_retriever.retrieve("river in South America", k=k)
            assert len(results) == k

    def test_returns_strings(self, built_retriever):
        results = built_retriever.retrieve("Paris capital", k=2)
        assert all(isinstance(p, str) for p in results)

    def test_results_are_subset_of_corpus(self, built_retriever):
        results = built_retriever.retrieve("capital city", k=3)
        assert all(p in PASSAGES for p in results)


class TestRecallAtK:
    """recall@k = |retrieved ∩ gold| / |gold| - must be computable from corpus."""

    def _recall_at_k(self, retrieved: list[str], corpus: RetrievalCorpus) -> float:
        gold_passages = {corpus.passages[i] for i in corpus.gold_indices}
        hits = sum(1 for p in retrieved if p in gold_passages)
        return hits / len(gold_passages) if gold_passages else 0.0

    def test_recall_at_k_in_range(self, built_retriever, corpus):
        results = built_retriever.retrieve("Paris Eiffel Tower", k=3)
        r_at_k = self._recall_at_k(results, corpus)
        assert 0.0 <= r_at_k <= 1.0

    def test_recall_at_k_relevant_query(self, built_retriever, corpus):
        """A query about Paris/Eiffel should surface gold passages."""
        results = built_retriever.retrieve("Paris Eiffel Tower capital France", k=2)
        r_at_k = self._recall_at_k(results, corpus)
        assert r_at_k > 0.0, "Expected at least one gold passage in top-2 for a Paris query"


class TestRetrieveBuildGuard:
    def test_raises_before_build(self, embedder):
        r = Retriever(embedder=embedder, k=3)
        with pytest.raises(RuntimeError):
            r.retrieve("some claim")

    def test_rebuild_replaces_index(self, embedder):
        new_passages = ["Only this passage exists."]
        r = Retriever(embedder=embedder, k=1)
        r.build(RetrievalCorpus(passages=new_passages))
        results = r.retrieve("passage", k=1)
        assert results == new_passages
