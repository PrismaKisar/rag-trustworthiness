"""Tests for src/data/hotpotqa_poisoner.py."""

import copy

import pytest


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


# ---------------------------------------------------------------------------
# Tracer bullet
# ---------------------------------------------------------------------------


def _sentence_at(example: dict, title: str, sent_idx: int) -> str:
    for ctx_title, sents in example["context"]:
        if ctx_title == title:
            return sents[sent_idx]
    raise KeyError(title)


def test_no_poisoning_leaves_context_unchanged():
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    result = poison_hotpotqa(EXAMPLES, poison_rate=0.0)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["context"] == poisoned["context"]


def test_full_poisoning_replaces_every_supporting_fact():
    """At rate 1.0 every supporting-fact sentence is replaced."""
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0)
    for original, poisoned in zip(EXAMPLES, result):
        for title, sent_idx in original["supporting_facts"]:
            assert _sentence_at(poisoned, title, sent_idx) != _sentence_at(
                original, title, sent_idx
            ), f"hop ({title}, {sent_idx}) was not replaced"


def test_originals_not_mutated():
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    before = copy.deepcopy(EXAMPLES)
    poison_hotpotqa(EXAMPLES, poison_rate=1.0)
    assert EXAMPLES == before


def test_question_and_answer_unchanged():
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["question"] == poisoned["question"]
        assert original["answer"] == poisoned["answer"]


def test_distractors_come_from_other_examples():
    """Replaced sentences must be supporting-fact sentences of OTHER questions."""
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0)
    for idx, (original, poisoned) in enumerate(zip(EXAMPLES, result)):
        own_supporting = {
            _sentence_at(original, t, i) for t, i in original["supporting_facts"]
        }
        other_supporting = {
            _sentence_at(EXAMPLES[j], t, i)
            for j, other in enumerate(EXAMPLES) if j != idx
            for t, i in other["supporting_facts"]
        }
        for title, sent_idx in original["supporting_facts"]:
            new_sent = _sentence_at(poisoned, title, sent_idx)
            assert new_sent in other_supporting
            assert new_sent not in own_supporting


def test_seed_reproducibility():
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    a = poison_hotpotqa(EXAMPLES, poison_rate=0.5, seed=42)
    b = poison_hotpotqa(EXAMPLES, poison_rate=0.5, seed=42)
    assert a == b


def test_different_seeds_produce_different_output():
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    a = poison_hotpotqa(EXAMPLES, poison_rate=0.5, seed=42)
    b = poison_hotpotqa(EXAMPLES, poison_rate=0.5, seed=99)
    contexts_a = [ex["context"] for ex in a]
    contexts_b = [ex["context"] for ex in b]
    assert contexts_a != contexts_b


@pytest.mark.parametrize("rate", [1.5, -0.1])
def test_invalid_poison_rate_raises(rate):
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    with pytest.raises(ValueError, match="poison_rate"):
        poison_hotpotqa(EXAMPLES, poison_rate=rate)


def test_poisoned_positions_tracked():
    """Each poisoned example carries its replaced (title, sent_idx) pairs."""
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    result = poison_hotpotqa(EXAMPLES, poison_rate=1.0)
    for original, poisoned in zip(EXAMPLES, result):
        assert "poisoned_positions" in poisoned
        positions = [tuple(p) for p in poisoned["poisoned_positions"]]
        expected = [tuple(f) for f in original["supporting_facts"]]
        assert set(positions) == set(expected)


def test_no_poisoned_positions_at_rate_zero():
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    result = poison_hotpotqa(EXAMPLES, poison_rate=0.0)
    for ex in result:
        assert "poisoned_positions" not in ex


def test_both_hops_addressable():
    """Across many seeds at rate ~0.5 each hop position gets poisoned at least once."""
    from src.data.hotpotqa_poisoner import poison_hotpotqa

    seen_hops: set[tuple[str, int]] = set()
    for s in range(50):
        result = poison_hotpotqa(EXAMPLES, poison_rate=0.5, seed=s)
        for ex in result:
            for p in ex.get("poisoned_positions", []):
                seen_hops.add(tuple(p))
    expected_hops = {tuple(f) for ex in EXAMPLES for f in ex["supporting_facts"]}
    assert seen_hops == expected_hops
