"""Tests for src/data/poisoner.py — no real FEVER data required."""

import copy

import pytest

from src.data.poisoner import poison_dataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SUPPORTS_EV = ["Alice appeared in Wonderland.", "She was a fictional character."]
_REFUTES_EV = ["Bob never visited Paris.", "He stayed home the entire time."]

EXAMPLES = [
    {"claim": "Alice appeared.", "evidence": list(_SUPPORTS_EV), "label": "SUPPORTS"},
    {"claim": "Alice was real.", "evidence": list(_SUPPORTS_EV), "label": "SUPPORTS"},
    {"claim": "Bob stayed home.", "evidence": list(_REFUTES_EV), "label": "REFUTES"},
    {"claim": "Bob left Paris.", "evidence": list(_REFUTES_EV), "label": "REFUTES"},
    {"claim": "Carol might exist.", "evidence": [], "label": "NOT ENOUGH INFO"},
]


# ---------------------------------------------------------------------------
# Tests — basic invariants
# ---------------------------------------------------------------------------


def test_no_poisoning_leaves_evidence_unchanged():
    result = poison_dataset(EXAMPLES, poison_rate=0.0)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["evidence"] == poisoned["evidence"]


def test_claim_unchanged():
    result = poison_dataset(EXAMPLES, poison_rate=0.5)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["claim"] == poisoned["claim"]


def test_label_unchanged():
    result = poison_dataset(EXAMPLES, poison_rate=0.5)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["label"] == poisoned["label"]


def test_originals_not_mutated():
    before = copy.deepcopy(EXAMPLES)
    poison_dataset(EXAMPLES, poison_rate=1.0)
    assert EXAMPLES == before


# ---------------------------------------------------------------------------
# Tests — poisoning rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
def test_poisoning_rate_per_example(rate):
    """Each example should have exactly round(rate * n) passages replaced."""
    result = poison_dataset(EXAMPLES, poison_rate=rate)
    for original, poisoned in zip(EXAMPLES, result):
        if not original["evidence"]:
            continue
        n = len(original["evidence"])
        changed = sum(o != p for o, p in zip(original["evidence"], poisoned["evidence"]))
        assert changed == round(rate * n), (
            f"label={original['label']}, n={n}, rate={rate}: "
            f"expected {round(rate * n)} changed, got {changed}"
        )


# ---------------------------------------------------------------------------
# Tests — distractors come from opposite label
# ---------------------------------------------------------------------------


def test_distractors_from_opposite_label_supports():
    """SUPPORTS claims should only receive distractors from REFUTES evidence."""
    result = poison_dataset(EXAMPLES, poison_rate=1.0)
    refutes_pool = set(_REFUTES_EV)
    for original, poisoned in zip(EXAMPLES, result):
        if original["label"] != "SUPPORTS" or not original["evidence"]:
            continue
        for passage in poisoned["evidence"]:
            assert passage in refutes_pool, f"Unexpected distractor: {passage!r}"


def test_distractors_from_opposite_label_refutes():
    """REFUTES claims should only receive distractors from SUPPORTS evidence."""
    result = poison_dataset(EXAMPLES, poison_rate=1.0)
    supports_pool = set(_SUPPORTS_EV)
    for original, poisoned in zip(EXAMPLES, result):
        if original["label"] != "REFUTES" or not original["evidence"]:
            continue
        for passage in poisoned["evidence"]:
            assert passage in supports_pool, f"Unexpected distractor: {passage!r}"
