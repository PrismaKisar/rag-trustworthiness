"""Tests for src/data/poisoner.py - llm_negation only."""

import copy

import pytest

from src.data.poisoner import poison_dataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLES = [
    {"claim": "Alice appeared.", "evidence": ["Alice appeared in Wonderland.", "She was fictional."], "label": "SUPPORTS"},
    {"claim": "Bob stayed home.", "evidence": ["Bob never visited Paris.", "He stayed home."], "label": "REFUTES"},
    {"claim": "Carol might exist.", "evidence": [], "label": "NOT ENOUGH INFO"},
]


class _FakeLLM:
    """Deterministic mock that returns a sentinel-tagged negation."""

    def __init__(self):
        self.calls: list[str] = []

    def complete(self, prompt, prompt_type=None):
        self.calls.append(prompt)
        return f"NEGATED::{len(self.calls)}"


# ---------------------------------------------------------------------------
# rate=0.0 — no LLM needed
# ---------------------------------------------------------------------------

def test_no_poisoning_leaves_evidence_unchanged():
    result = poison_dataset(EXAMPLES, poison_rate=0.0)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["evidence"] == poisoned["evidence"]


def test_no_poisoned_positions_at_rate_zero():
    result = poison_dataset(EXAMPLES, poison_rate=0.0)
    for ex in result:
        assert "poisoned_positions" not in ex


def test_returns_same_length_at_zero():
    result = poison_dataset(EXAMPLES, poison_rate=0.0)
    assert len(result) == len(EXAMPLES)


# ---------------------------------------------------------------------------
# rate=1.0 — requires LLM
# ---------------------------------------------------------------------------

def test_llm_negation_replaces_all_evidence():
    llm = _FakeLLM()
    result = poison_dataset(EXAMPLES, poison_rate=1.0, llm=llm)
    for original, poisoned in zip(EXAMPLES, result):
        if not original["evidence"]:
            continue
        for passage in poisoned["evidence"]:
            assert passage.startswith("NEGATED::")


def test_llm_negation_calls_llm_once_per_poisoned_passage():
    llm = _FakeLLM()
    poison_dataset(EXAMPLES, poison_rate=1.0, llm=llm)
    expected = sum(len(ex["evidence"]) for ex in EXAMPLES)
    assert len(llm.calls) == expected


def test_llm_negation_prompt_includes_original_passage():
    llm = _FakeLLM()
    poison_dataset(EXAMPLES, poison_rate=1.0, llm=llm)
    all_evidence = {p for ex in EXAMPLES for p in ex["evidence"]}
    for prompt in llm.calls:
        assert any(p in prompt for p in all_evidence)


def test_llm_negation_tracks_poisoned_positions():
    llm = _FakeLLM()
    result = poison_dataset(EXAMPLES, poison_rate=1.0, llm=llm)
    for original, poisoned in zip(EXAMPLES, result):
        if not original["evidence"]:
            assert "poisoned_positions" not in poisoned
        else:
            assert poisoned["poisoned_positions"] == list(range(len(original["evidence"])))


def test_claim_unchanged():
    llm = _FakeLLM()
    result = poison_dataset(EXAMPLES, poison_rate=1.0, llm=llm)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["claim"] == poisoned["claim"]


def test_label_unchanged():
    llm = _FakeLLM()
    result = poison_dataset(EXAMPLES, poison_rate=1.0, llm=llm)
    for original, poisoned in zip(EXAMPLES, result):
        assert original["label"] == poisoned["label"]


def test_originals_not_mutated():
    before = copy.deepcopy(EXAMPLES)
    llm = _FakeLLM()
    poison_dataset(EXAMPLES, poison_rate=1.0, llm=llm)
    assert EXAMPLES == before


def test_returns_same_length_at_one():
    llm = _FakeLLM()
    result = poison_dataset(EXAMPLES, poison_rate=1.0, llm=llm)
    assert len(result) == len(EXAMPLES)


def test_empty_evidence_unaffected():
    llm = _FakeLLM()
    result = poison_dataset(EXAMPLES, poison_rate=1.0, llm=llm)
    nei = next(r for r in result if r["label"] == "NOT ENOUGH INFO")
    assert nei["evidence"] == []


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility():
    a = poison_dataset(EXAMPLES, poison_rate=1.0, seed=42, llm=_FakeLLM())
    b = poison_dataset(EXAMPLES, poison_rate=1.0, seed=42, llm=_FakeLLM())
    assert a == b


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rate", [1.5, -0.1])
def test_invalid_poison_rate_raises(rate):
    with pytest.raises(ValueError, match="poison_rate"):
        poison_dataset(EXAMPLES, poison_rate=rate)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        poison_dataset(EXAMPLES, poison_rate=0.5, strategy="opposite_label")


def test_llm_negation_requires_llm():
    with pytest.raises(ValueError, match="llm"):
        poison_dataset(EXAMPLES, poison_rate=1.0, strategy="llm_negation")
