"""Tests for src/retrieval/retriever.py."""

from __future__ import annotations

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


@pytest.fixture(scope="session")
def embedder(tmp_path_factory):
    emb = Embedder(cache_dir=tmp_path_factory.mktemp("emb_cache"))
    yield emb
    emb.close()


@pytest.fixture(scope="session")
def corpus():
    return RetrievalCorpus(passages=PASSAGES)


@pytest.fixture(scope="session")
def built_retriever(embedder, corpus):
    r = Retriever(embedder=embedder, k=3)
    r.build(corpus)
    return r


# ---------------------------------------------------------------------------
# k property
# ---------------------------------------------------------------------------

class TestKProperty:
    def test_k_matches_constructor(self, embedder):
        r = Retriever(embedder=embedder, k=7)
        assert r.k == 7

    def test_default_k(self, embedder):
        r = Retriever(embedder=embedder)
        assert r.k == 5


# ---------------------------------------------------------------------------
# retrieve returns exactly k results
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

    def test_relevant_query_surfaces_related_passages(self, built_retriever):
        results = built_retriever.retrieve("Paris Eiffel Tower capital France", k=2)
        paris_related = {PASSAGES[0], PASSAGES[1]}
        assert any(p in paris_related for p in results)


# ---------------------------------------------------------------------------
# build guard
# ---------------------------------------------------------------------------

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
