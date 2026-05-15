"""Tests for src/retrieval/corpus.py."""

import pytest

from src.retrieval.corpus import RetrievalCorpus, build_all_corpora, build_corpus

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
# build_corpus
# ---------------------------------------------------------------------------


def test_corpus_contains_own_evidence():
    corpus = build_corpus(EXAMPLES[0], EXAMPLES, example_index=0)
    for ev in EXAMPLES[0]["evidence"]:
        assert ev in corpus.passages


def test_gold_indices_match_evidence_length():
    corpus = build_corpus(EXAMPLES[0], EXAMPLES, example_index=0)
    assert corpus.gold_indices == {0, 1}  # evidence has 2 items → indices 0,1


def test_distractor_passages_added():
    corpus = build_corpus(EXAMPLES[0], EXAMPLES, distractor_pool_size=5, example_index=0)
    n_gold = len(EXAMPLES[0]["evidence"])
    assert len(corpus.passages) > n_gold


def test_no_own_evidence_in_distractors():
    """Distractors must come from other claims, not from the example itself."""
    corpus = build_corpus(EXAMPLES[0], EXAMPLES, distractor_pool_size=20, example_index=0)
    distractors = [p for i, p in enumerate(corpus.passages) if i not in corpus.gold_indices]
    own = set(EXAMPLES[0]["evidence"])
    assert all(p not in own for p in distractors)


def test_distractor_pool_capped_when_small_dataset():
    """If fewer distractors are available than requested, cap silently."""
    small = [
        {"claim": "X", "evidence": ["X1"], "label": "SUPPORTS"},
        {"claim": "Y", "evidence": ["Y1"], "label": "REFUTES"},
    ]
    corpus = build_corpus(small[0], small, distractor_pool_size=100, example_index=0)
    # Only 1 distractor candidate ("Y1") is available
    assert len(corpus.passages) <= 2  # 1 gold + up to 1 distractor


def test_corpus_reproducible_with_same_seed():
    c1 = build_corpus(EXAMPLES[0], EXAMPLES, distractor_pool_size=5, seed=0, example_index=0)
    c2 = build_corpus(EXAMPLES[0], EXAMPLES, distractor_pool_size=5, seed=0, example_index=0)
    assert c1.passages == c2.passages


def test_corpus_differs_with_different_seed():
    c1 = build_corpus(EXAMPLES[0], EXAMPLES, distractor_pool_size=3, seed=0, example_index=0)
    c2 = build_corpus(EXAMPLES[0], EXAMPLES, distractor_pool_size=3, seed=99, example_index=0)
    # Different seeds should (almost certainly) yield different distractor order
    assert c1.passages != c2.passages


# ---------------------------------------------------------------------------
# build_all_corpora
# ---------------------------------------------------------------------------


def test_build_all_corpora_returns_correct_count():
    corpora = build_all_corpora(EXAMPLES, distractor_pool_size=5)
    assert len(corpora) == len(EXAMPLES)


def test_build_all_corpora_each_is_retrieval_corpus():
    corpora = build_all_corpora(EXAMPLES)
    assert all(isinstance(c, RetrievalCorpus) for c in corpora)


def test_build_all_corpora_gold_indices_per_example():
    corpora = build_all_corpora(EXAMPLES)
    for ex, corpus in zip(EXAMPLES, corpora):
        poisoned = ex.get("poisoned_positions", set())
        assert corpus.gold_indices == set(range(len(ex["evidence"]))) - poisoned


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_evidence_produces_only_distractors():
    """An example with empty evidence (e.g. NOT ENOUGH INFO) should have no gold indices."""
    empty_ex = {"claim": "Unknown.", "evidence": [], "label": "NOT ENOUGH INFO"}
    corpus = build_corpus(empty_ex, EXAMPLES, distractor_pool_size=5, example_index=None)
    assert corpus.gold_indices == set()
    assert len(corpus.passages) > 0  # only distractors


def test_poisoned_positions_excluded_from_gold():
    """Passages replaced by the poisoner must not appear in gold_indices."""
    poisoned_ex = {
        "claim": "A",
        "evidence": ["real", "fake", "real2"],
        "poisoned_positions": {1},
        "label": "SUPPORTS",
    }
    corpus = build_corpus(poisoned_ex, EXAMPLES, distractor_pool_size=0, example_index=None)
    assert 1 not in corpus.gold_indices
    assert corpus.gold_indices == {0, 2}


# ---------------------------------------------------------------------------
# build_hotpotqa_corpus
# ---------------------------------------------------------------------------

_HOTPOT_EX = {
    "question": "Where was Marie Curie born?",
    "answer": "Warsaw",
    "supporting_facts": [["Marie Curie", 0], ["Warsaw", 0]],
    "context": [
        ["Marie Curie", ["Marie Curie was born in Warsaw.", "She won Nobel prizes."]],
        ["Warsaw", ["Warsaw is the capital of Poland.", "It is in central Poland."]],
        ["Distractor A", ["A totally unrelated paragraph.", "More filler."]],
    ],
}


def test_build_hotpotqa_corpus_returns_retrieval_corpus():
    from src.retrieval.corpus import build_hotpotqa_corpus
    corpus = build_hotpotqa_corpus(_HOTPOT_EX)
    assert isinstance(corpus, RetrievalCorpus)


def test_build_hotpotqa_corpus_passages_are_joined_sentences():
    from src.retrieval.corpus import build_hotpotqa_corpus
    corpus = build_hotpotqa_corpus(_HOTPOT_EX)
    assert corpus.passages[0] == "Marie Curie was born in Warsaw. She won Nobel prizes."
    assert corpus.passages[1] == "Warsaw is the capital of Poland. It is in central Poland."


def test_build_hotpotqa_corpus_supporting_titles_are_gold():
    from src.retrieval.corpus import build_hotpotqa_corpus
    corpus = build_hotpotqa_corpus(_HOTPOT_EX)
    # "Marie Curie" → index 0, "Warsaw" → index 1; "Distractor A" → not gold
    assert corpus.gold_indices == {0, 1}


def test_build_hotpotqa_corpus_non_supporting_not_gold():
    from src.retrieval.corpus import build_hotpotqa_corpus
    corpus = build_hotpotqa_corpus(_HOTPOT_EX)
    assert 2 not in corpus.gold_indices  # "Distractor A" is not supporting


def test_build_hotpotqa_corpus_poisoned_title_excluded_from_gold():
    from src.retrieval.corpus import build_hotpotqa_corpus
    ex = {**_HOTPOT_EX, "poisoned_positions": [["Marie Curie", 0]]}
    corpus = build_hotpotqa_corpus(ex)
    # "Marie Curie" is supporting but poisoned → excluded; "Warsaw" remains
    assert 0 not in corpus.gold_indices
    assert 1 in corpus.gold_indices


def test_build_hotpotqa_corpus_no_poisoned_positions_key():
    from src.retrieval.corpus import build_hotpotqa_corpus
    ex = {k: v for k, v in _HOTPOT_EX.items() if k != "poisoned_positions"}
    corpus = build_hotpotqa_corpus(ex)
    assert corpus.gold_indices == {0, 1}


def test_build_hotpotqa_corpus_empty_supporting_facts():
    from src.retrieval.corpus import build_hotpotqa_corpus
    ex = {**_HOTPOT_EX, "supporting_facts": []}
    corpus = build_hotpotqa_corpus(ex)
    assert corpus.gold_indices == set()
