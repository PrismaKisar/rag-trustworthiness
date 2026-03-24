"""Tests for src/evaluation/metrics.py — all five metric functions."""

import pytest

from src.evaluation.metrics import (
    accuracy,
    hallucination_rate,
    macro_f1,
    precision_at_k,
    self_consistency,
)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_perfect(self):
        preds = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        gold  = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        assert accuracy(preds, gold) == 1.0

    def test_zero(self):
        preds = ["REFUTES", "SUPPORTS", "SUPPORTS"]
        gold  = ["SUPPORTS", "REFUTES",  "NOT ENOUGH INFO"]
        assert accuracy(preds, gold) == 0.0

    def test_partial(self):
        preds = ["SUPPORTS", "REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
        gold  = ["SUPPORTS", "SUPPORTS", "SUPPORTS", "NOT ENOUGH INFO"]
        assert accuracy(preds, gold) == pytest.approx(0.75)

    def test_empty(self):
        assert accuracy([], []) == 0.0


# ---------------------------------------------------------------------------
# macro_f1
# ---------------------------------------------------------------------------

class TestMacroF1:
    def test_perfect(self):
        labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        assert macro_f1(labels, labels) == pytest.approx(1.0)

    def test_in_range(self):
        preds = ["SUPPORTS", "SUPPORTS", "SUPPORTS"]
        gold  = ["SUPPORTS", "REFUTES",  "NOT ENOUGH INFO"]
        score = macro_f1(preds, gold)
        assert 0.0 <= score <= 1.0

    def test_all_wrong_is_low(self):
        preds = ["REFUTES",  "SUPPORTS", "SUPPORTS"]
        gold  = ["SUPPORTS", "REFUTES",  "NOT ENOUGH INFO"]
        assert macro_f1(preds, gold) < 0.5

    def test_empty(self):
        assert macro_f1([], []) == 0.0


# ---------------------------------------------------------------------------
# hallucination_rate
# ---------------------------------------------------------------------------

class TestHallucinationRate:
    def test_full_hallucination(self):
        preds = ["SUPPORTS", "REFUTES"]
        gold  = ["NOT ENOUGH INFO", "NOT ENOUGH INFO"]
        assert hallucination_rate(preds, gold) == 1.0

    def test_no_hallucination(self):
        preds = ["NOT ENOUGH INFO", "NOT ENOUGH INFO"]
        gold  = ["NOT ENOUGH INFO", "NOT ENOUGH INFO"]
        assert hallucination_rate(preds, gold) == 0.0

    def test_partial(self):
        # 2 out of 4 NEI gold → 1 hallucinated
        preds = ["SUPPORTS", "NOT ENOUGH INFO", "REFUTES", "SUPPORTS"]
        gold  = ["NOT ENOUGH INFO", "NOT ENOUGH INFO", "SUPPORTS", "REFUTES"]
        # NEI gold at indices 0,1 → preds: SUPPORTS (hallucinated), NOT ENOUGH INFO (ok)
        assert hallucination_rate(preds, gold) == pytest.approx(0.5)

    def test_no_nei_gold_returns_zero(self):
        preds = ["SUPPORTS", "REFUTES"]
        gold  = ["SUPPORTS", "REFUTES"]
        assert hallucination_rate(preds, gold) == 0.0

    def test_empty(self):
        assert hallucination_rate([], []) == 0.0


# ---------------------------------------------------------------------------
# self_consistency
# ---------------------------------------------------------------------------

class TestSelfConsistency:
    def test_perfectly_consistent(self):
        runs = [["SUPPORTS"] * 5, ["REFUTES"] * 5]
        assert self_consistency(runs) == 1.0

    def test_perfectly_inconsistent(self):
        # 2 runs, split 1/1 → majority count = 1, score = 0.5 each
        runs = [["SUPPORTS", "REFUTES"]]
        assert self_consistency(runs) == pytest.approx(0.5)

    def test_partial(self):
        # Claim 1: 4/5 agree → 0.8; Claim 2: 5/5 agree → 1.0; mean = 0.9
        runs = [
            ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "REFUTES"],
            ["REFUTES"] * 5,
        ]
        assert self_consistency(runs) == pytest.approx(0.9)

    def test_single_run(self):
        # 1/1 agrees with majority → 1.0
        runs = [["SUPPORTS"]]
        assert self_consistency(runs) == 1.0

    def test_empty(self):
        assert self_consistency([]) == 0.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_gold(self):
        retrieved = ["p1", "p2", "p3"]
        gold      = ["p1", "p2", "p3", "p4"]
        assert precision_at_k(retrieved, gold) == 1.0

    def test_none_gold(self):
        retrieved = ["p5", "p6"]
        gold      = ["p1", "p2"]
        assert precision_at_k(retrieved, gold) == 0.0

    def test_partial(self):
        retrieved = ["p1", "p2", "p5", "p6"]
        gold      = ["p1", "p2"]
        assert precision_at_k(retrieved, gold) == pytest.approx(0.5)

    def test_empty_retrieved(self):
        assert precision_at_k([], ["p1"]) == 0.0

    def test_empty_gold(self):
        assert precision_at_k(["p1", "p2"], []) == 0.0
