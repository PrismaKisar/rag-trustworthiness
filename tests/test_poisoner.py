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


# ---------------------------------------------------------------------------
# Tests — reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility():
    result_a = poison_dataset(EXAMPLES, poison_rate=0.5, seed=42)
    result_b = poison_dataset(EXAMPLES, poison_rate=0.5, seed=42)
    assert result_a == result_b


def test_different_seeds_produce_different_output():
    result_42 = poison_dataset(EXAMPLES, poison_rate=0.5, seed=42)
    result_99 = poison_dataset(EXAMPLES, poison_rate=0.5, seed=99)
    evidences_42 = [e["evidence"] for e in result_42 if e["evidence"]]
    evidences_99 = [e["evidence"] for e in result_99 if e["evidence"]]
    assert evidences_42 != evidences_99


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [1.5, -0.1])
def test_invalid_poison_rate_raises(rate):
    with pytest.raises(ValueError, match="poison_rate"):
        poison_dataset(EXAMPLES, poison_rate=rate)


def test_empty_evidence_unaffected():
    result = poison_dataset(EXAMPLES, poison_rate=1.0)
    nei = next(r for r in result if r["label"] == "NOT ENOUGH INFO")
    assert nei["evidence"] == []


def test_returns_same_length():
    result = poison_dataset(EXAMPLES, poison_rate=0.5)
    assert len(result) == len(EXAMPLES)


def test_poisoned_positions_tracked():
    """Poisoned examples must carry ``poisoned_positions`` marking replaced indices."""
    result = poison_dataset(EXAMPLES, poison_rate=1.0)
    for original, poisoned in zip(EXAMPLES, result):
        if not original["evidence"]:
            assert "poisoned_positions" not in poisoned
        else:
            assert "poisoned_positions" in poisoned
            assert len(poisoned["poisoned_positions"]) == len(original["evidence"])


def test_no_poisoned_positions_at_rate_zero():
    """At rate 0 no passages are replaced, so no poisoned_positions key."""
    result = poison_dataset(EXAMPLES, poison_rate=0.0)
    for item in result:
        assert "poisoned_positions" not in item


# ---------------------------------------------------------------------------
# Tests — strategy parameter
# ---------------------------------------------------------------------------


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        poison_dataset(EXAMPLES, poison_rate=0.5, strategy="bogus")


def test_explicit_opposite_label_strategy_matches_default():
    default = poison_dataset(EXAMPLES, poison_rate=0.5, seed=42)
    explicit = poison_dataset(EXAMPLES, poison_rate=0.5, seed=42, strategy="opposite_label")
    assert default == explicit


def test_llm_negation_requires_llm():
    with pytest.raises(ValueError, match="llm"):
        poison_dataset(EXAMPLES, poison_rate=0.5, strategy="llm_negation")


class _FakeLLM:
    """Deterministic mock that returns a sentinel-tagged negation."""

    def __init__(self):
        self.calls: list[str] = []

    def complete(self, prompt, max_tokens=None):
        self.calls.append(prompt)
        return f"NEGATED::{len(self.calls)}"


def test_llm_negation_replaces_passages_with_llm_output():
    llm = _FakeLLM()
    result = poison_dataset(
        EXAMPLES, poison_rate=1.0, seed=42, strategy="llm_negation", llm=llm,
    )
    for original, poisoned in zip(EXAMPLES, result):
        if not original["evidence"]:
            continue
        for passage in poisoned["evidence"]:
            assert passage.startswith("NEGATED::"), (
                f"expected LLM-generated passage, got {passage!r}"
            )


def test_llm_negation_calls_llm_once_per_poisoned_passage():
    llm = _FakeLLM()
    poison_dataset(
        EXAMPLES, poison_rate=1.0, seed=42, strategy="llm_negation", llm=llm,
    )
    expected_calls = sum(len(ex["evidence"]) for ex in EXAMPLES)
    assert len(llm.calls) == expected_calls


def test_llm_negation_tracks_poisoned_positions():
    llm = _FakeLLM()
    result = poison_dataset(
        EXAMPLES, poison_rate=1.0, seed=42, strategy="llm_negation", llm=llm,
    )
    for original, poisoned in zip(EXAMPLES, result):
        if not original["evidence"]:
            assert "poisoned_positions" not in poisoned
        else:
            assert poisoned["poisoned_positions"] == set(range(len(original["evidence"])))


def test_llm_negation_prompt_includes_original_passage():
    llm = _FakeLLM()
    poison_dataset(
        EXAMPLES, poison_rate=1.0, seed=42, strategy="llm_negation", llm=llm,
    )
    all_evidence = {p for ex in EXAMPLES for p in ex["evidence"]}
    for prompt in llm.calls:
        assert any(p in prompt for p in all_evidence), (
            f"prompt does not include any original passage: {prompt!r}"
        )


def test_sample_without_replacement():
    """When pool is large enough, distractors must be unique (no duplicates)."""
    result = poison_dataset(EXAMPLES, poison_rate=1.0)
    for original, poisoned in zip(EXAMPLES, result):
        if len(original["evidence"]) > 1:
            poisoned_passages = [
                poisoned["evidence"][i] for i in poisoned.get("poisoned_positions", set())
            ]
            # With a large-enough pool, sample() produces unique distractors
            assert len(poisoned_passages) == len(set(poisoned_passages))
