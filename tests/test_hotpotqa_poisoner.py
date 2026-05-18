"""Tests for src/data/hotpotqa_poisoner.py."""

import copy

import pytest

from src.data.hotpotqa_poisoner import poison_hotpotqa


def _make_example(qid: str, hop_a_sent: str, hop_b_sent: str) -> dict:
    """Build a HotpotQA example with two supporting hops."""
    return {
        "question": f"q-{qid}",
        "answer": f"a-{qid}",
        "supporting_facts": [[f"{qid}_A", 0], [f"{qid}_B", 1]],
        "context": [
            [f"{qid}_A", [hop_a_sent, "filler A1."]],
            [f"{qid}_B", ["filler B0.", hop_b_sent]],
        ],
    }


EXAMPLES = [
    _make_example("E1", "Switzerland borders France.", "Italy borders Switzerland."),
    _make_example("E2", "Clarke wrote 2001.", "The film premiered in 1968."),
    _make_example("E3", "Curie was born in Warsaw.", "She won two Nobel prizes."),
    _make_example("E4", "Mount Everest is in Nepal.", "It is the highest peak."),
]


class _FakeLLM:
    """Deterministic mock that returns a sentinel-tagged negation."""

    def __init__(self):
        self.calls: list[str] = []

    def complete(self, prompt, prompt_type=None):
        self.calls.append(prompt)
        return f"NEGATED::{len(self.calls)}"


def _sentence_at(example: dict, title: str, sent_idx: int) -> str:
    for ctx_title, sents in example["context"]:
        if ctx_title == title:
            return sents[sent_idx]
    raise KeyError(title)


# ---------------------------------------------------------------------------
# rate=0.0 — no LLM needed
# ---------------------------------------------------------------------------

def test_no_poisoning_leaves_context_unchanged():
    result = poison_hotpotqa(EXAMPLES, poison_rate=0.0)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["context"] == poisoned["context"]


def test_no_poisoned_positions_at_rate_zero():
    result = poison_hotpotqa(EXAMPLES, poison_rate=0.0)
    for ex in result:
        assert "poisoned_positions" not in ex


# ---------------------------------------------------------------------------
# Single-hop LLM negation at rate=1.0
# ---------------------------------------------------------------------------

def test_exactly_one_hop_poisoned_per_example():
    llm = _FakeLLM()
    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0, llm=llm)
    for original, poisoned in zip(EXAMPLES, result):
        changed = sum(
            1 for title, sent_idx in original["supporting_facts"]
            if _sentence_at(poisoned, title, sent_idx) != _sentence_at(original, title, sent_idx)
        )
        assert changed == 1, f"Expected 1 poisoned hop, got {changed}"


def test_non_targeted_hop_is_unchanged():
    llm = _FakeLLM()
    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0, llm=llm)
    for original, poisoned in zip(EXAMPLES, result):
        poisoned_positions = {(p[0], p[1]) for p in poisoned.get("poisoned_positions", [])}
        for title, sent_idx in original["supporting_facts"]:
            if (title, sent_idx) not in poisoned_positions:
                assert (
                    _sentence_at(poisoned, title, sent_idx)
                    == _sentence_at(original, title, sent_idx)
                )


def test_poisoned_positions_has_exactly_one_entry():
    llm = _FakeLLM()
    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0, llm=llm)
    for poisoned in result:
        assert "poisoned_positions" in poisoned
        assert len(poisoned["poisoned_positions"]) == 1


def test_poisoned_position_is_a_supporting_fact():
    llm = _FakeLLM()
    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0, llm=llm)
    for original, poisoned in zip(EXAMPLES, result):
        pos = tuple(poisoned["poisoned_positions"][0])
        expected = {tuple(f) for f in original["supporting_facts"]}
        assert pos in expected


def test_llm_negation_replaces_sentence_with_llm_output():
    llm = _FakeLLM()
    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0, llm=llm)
    for poisoned in result:
        title, sent_idx = poisoned["poisoned_positions"][0]
        new_sent = _sentence_at(poisoned, title, sent_idx)
        assert new_sent.startswith("NEGATED::")


def test_llm_called_once_per_example():
    llm = _FakeLLM()
    poison_hotpotqa(EXAMPLES, poison_rate=1.0, llm=llm)
    assert len(llm.calls) == len(EXAMPLES)


def test_llm_prompt_includes_original_sentence():
    llm = _FakeLLM()
    poison_hotpotqa(EXAMPLES, poison_rate=1.0, llm=llm)
    all_supporting = {
        _sentence_at(ex, t, i)
        for ex in EXAMPLES
        for t, i in ex["supporting_facts"]
    }
    for prompt in llm.calls:
        assert any(s in prompt for s in all_supporting)


# ---------------------------------------------------------------------------
# Immutability and stable fields
# ---------------------------------------------------------------------------

def test_originals_not_mutated():
    before = copy.deepcopy(EXAMPLES)
    poison_hotpotqa(EXAMPLES, poison_rate=1.0, llm=_FakeLLM())
    assert EXAMPLES == before


def test_question_and_answer_unchanged():
    llm = _FakeLLM()
    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0, llm=llm)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["question"] == poisoned["question"]
        assert original["answer"] == poisoned["answer"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility():
    a = poison_hotpotqa(EXAMPLES, poison_rate=1.0, seed=42, llm=_FakeLLM())
    b = poison_hotpotqa(EXAMPLES, poison_rate=1.0, seed=42, llm=_FakeLLM())
    assert a == b


def test_different_seeds_target_different_hops():
    """Across seeds, different hop positions get targeted."""
    seen_hops: set[tuple[str, int]] = set()
    for s in range(30):
        result = poison_hotpotqa(EXAMPLES, poison_rate=1.0, seed=s, llm=_FakeLLM())
        for ex in result:
            for p in ex.get("poisoned_positions", []):
                seen_hops.add((p[0], p[1]))
    expected_hops = {(f[0], f[1]) for ex in EXAMPLES for f in ex["supporting_facts"]}
    assert seen_hops == expected_hops


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rate", [1.5, -0.1])
def test_invalid_poison_rate_raises(rate):
    with pytest.raises(ValueError, match="poison_rate"):
        poison_hotpotqa(EXAMPLES, poison_rate=rate)


def test_llm_required_when_rate_positive():
    with pytest.raises(ValueError, match="llm"):
        poison_hotpotqa(EXAMPLES, poison_rate=1.0)


def test_single_supporting_fact_example():
    """Example with only one supporting fact: that one gets poisoned."""
    single_hop = [
        {
            "question": "Who wrote Hamlet?",
            "answer": "Shakespeare",
            "supporting_facts": [["Shakespeare", 0]],
            "context": [["Shakespeare", ["Shakespeare wrote Hamlet.", "Extra sentence."]]],
        },
        {
            "question": "Where is Paris?",
            "answer": "France",
            "supporting_facts": [["Paris", 0]],
            "context": [["Paris", ["Paris is in France.", "Extra."]]],
        },
    ]
    llm = _FakeLLM()
    result = poison_hotpotqa(single_hop, poison_rate=1.0, llm=llm)
    for original, poisoned in zip(single_hop, result):
        assert len(poisoned["poisoned_positions"]) == 1
        title, sent_idx = poisoned["poisoned_positions"][0]
        new_sent = _sentence_at(poisoned, title, sent_idx)
        assert new_sent.startswith("NEGATED::")
        assert new_sent != _sentence_at(original, title, sent_idx)
