"""Tests for src/retrieval/corpus.py."""

import pytest

from src.retrieval.corpus import RetrievalCorpus, build_all_corpora, build_corpus, build_hotpotqa_corpus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLES = [
    {"claim": "A", "evidence": ["A1", "A2"], "label": "SUPPORTS"},
    {"claim": "B", "evidence": ["B1", "B2", "B3"], "label": "REFUTES"},
    {"claim": "C", "evidence": ["C1"], "label": "NOT ENOUGH INFO"},
    {"claim": "D", "evidence": ["D1", "D2", "D3", "D4"], "label": "SUPPORTS"},
]


# ---------------------------------------------------------------------------
# build_corpus - global pool
# ---------------------------------------------------------------------------


def test_corpus_excludes_own_evidence():
    corpus = build_corpus(EXAMPLES[0], EXAMPLES, example_index=0)
    for ev in EXAMPLES[0]["evidence"]:
        assert ev not in corpus.passages


def test_corpus_contains_all_other_evidence():
    corpus = build_corpus(EXAMPLES[0], EXAMPLES, example_index=0)
    for ex in EXAMPLES[1:]:
        for ev in ex["evidence"]:
            assert ev in corpus.passages


def test_corpus_size_equals_sum_of_other_evidence():
    corpus = build_corpus(EXAMPLES[0], EXAMPLES, example_index=0)
    expected = sum(len(ex["evidence"]) for ex in EXAMPLES[1:])
    assert len(corpus.passages) == expected


def test_corpus_excludes_self_by_identity_without_index():
    corpus = build_corpus(EXAMPLES[1], EXAMPLES)
    for ev in EXAMPLES[1]["evidence"]:
        assert ev not in corpus.passages


def test_corpus_empty_evidence_example():
    empty_ex = {"claim": "Unknown.", "evidence": [], "label": "NOT ENOUGH INFO"}
    dataset = EXAMPLES + [empty_ex]
    corpus = build_corpus(empty_ex, dataset)
    expected = sum(len(ex["evidence"]) for ex in EXAMPLES)
    assert len(corpus.passages) == expected


# ---------------------------------------------------------------------------
# build_all_corpora
# ---------------------------------------------------------------------------


def test_build_all_corpora_returns_correct_count():
    corpora = build_all_corpora(EXAMPLES)
    assert len(corpora) == len(EXAMPLES)


def test_build_all_corpora_each_is_retrieval_corpus():
    corpora = build_all_corpora(EXAMPLES)
    assert all(isinstance(c, RetrievalCorpus) for c in corpora)


def test_build_all_corpora_excludes_own_evidence():
    corpora = build_all_corpora(EXAMPLES)
    for ex, corpus in zip(EXAMPLES, corpora):
        for ev in ex["evidence"]:
            assert ev not in corpus.passages


def test_build_all_corpora_accepts_full_dataset():
    """full_dataset parameter allows evaluating a subset over the full pool."""
    subset = EXAMPLES[:2]
    corpora = build_all_corpora(subset, full_dataset=EXAMPLES)
    assert len(corpora) == len(subset)



# ---------------------------------------------------------------------------
# build_hotpotqa_corpus - global pool
# ---------------------------------------------------------------------------

_HOTPOT_A = {
    "question": "Where was Marie Curie born?",
    "answer": "Warsaw",
    "supporting_facts": [["Marie Curie", 0], ["Warsaw", 0]],
    "context": [
        ["Marie Curie", ["Marie Curie was born in Warsaw.", "She won Nobel prizes."]],
        ["Warsaw", ["Warsaw is the capital of Poland.", "It is in central Poland."]],
        ["Distractor A", ["A totally unrelated paragraph.", "More filler."]],
    ],
}

_HOTPOT_B = {
    "question": "Who founded Apple?",
    "answer": "Steve Jobs",
    "supporting_facts": [["Apple Inc.", 0], ["Steve Jobs", 0]],
    "context": [
        ["Apple Inc.", ["Apple was founded in 1976.", "It is based in Cupertino."]],
        ["Steve Jobs", ["Steve Jobs co-founded Apple.", "He was born in San Francisco."]],
        ["Distractor B", ["Unrelated sentence."]],
    ],
}

_HOTPOT_EXAMPLES = [_HOTPOT_A, _HOTPOT_B]


def test_build_hotpotqa_global_excludes_own_supporting():
    corpus = build_hotpotqa_corpus(_HOTPOT_A, _HOTPOT_EXAMPLES, example_index=0)
    own_sup = {"Marie Curie was born in Warsaw. She won Nobel prizes.",
               "Warsaw is the capital of Poland. It is in central Poland."}
    for p in corpus.passages:
        assert p not in own_sup


def test_build_hotpotqa_global_contains_other_supporting():
    corpus = build_hotpotqa_corpus(_HOTPOT_A, _HOTPOT_EXAMPLES, example_index=0)
    assert "Apple was founded in 1976. It is based in Cupertino." in corpus.passages
    assert "Steve Jobs co-founded Apple. He was born in San Francisco." in corpus.passages


def test_build_hotpotqa_global_excludes_non_supporting_from_others():
    corpus = build_hotpotqa_corpus(_HOTPOT_A, _HOTPOT_EXAMPLES, example_index=0)
    assert "Unrelated sentence." not in corpus.passages


def test_build_hotpotqa_requires_all_examples():
    """build_hotpotqa_corpus requires all_examples."""
    import inspect
    sig = inspect.signature(build_hotpotqa_corpus)
    assert "all_examples" in sig.parameters
