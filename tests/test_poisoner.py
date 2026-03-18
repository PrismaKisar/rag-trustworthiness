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
