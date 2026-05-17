"""Tests for src/evaluation/metrics.py - all five metric functions."""

import pytest

from src.evaluation.metrics import (
    accuracy,
    contradiction_detection_rate,
    hallucination_rate,
    macro_f1,
    precision_at_k,
    qa_hallucination_rate,
    retrieval_accuracy_correlation,
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


# ---------------------------------------------------------------------------
# exact_match (QA - HotpotQA)
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_identical(self):
        from src.evaluation.metrics import exact_match
        assert exact_match("Switzerland", "Switzerland") == 1.0

    def test_case_insensitive(self):
        from src.evaluation.metrics import exact_match
        assert exact_match("switzerland", "Switzerland") == 1.0

    def test_strips_articles(self):
        from src.evaluation.metrics import exact_match
        assert exact_match("the eiffel tower", "Eiffel Tower") == 1.0

    def test_strips_punctuation(self):
        from src.evaluation.metrics import exact_match
        assert exact_match("Switzerland.", "Switzerland") == 1.0

    def test_collapses_whitespace(self):
        from src.evaluation.metrics import exact_match
        assert exact_match("Arthur  C.   Clarke", "Arthur C Clarke") == 1.0

    def test_mismatch(self):
        from src.evaluation.metrics import exact_match
        assert exact_match("France", "Switzerland") == 0.0


# ---------------------------------------------------------------------------
# token_f1 (QA - HotpotQA)
# ---------------------------------------------------------------------------


class TestTokenF1:
    def test_perfect_match(self):
        from src.evaluation.metrics import token_f1
        assert token_f1("Marie Curie", "Marie Curie") == pytest.approx(1.0)

    def test_no_overlap(self):
        from src.evaluation.metrics import token_f1
        assert token_f1("dog", "cat") == 0.0

    def test_partial_overlap(self):
        # pred tokens: {arthur, clarke}; gold tokens: {arthur, c, clarke}
        # precision = 2/2 = 1.0, recall = 2/3, f1 = 2*1*(2/3)/(1+2/3) = 0.8
        from src.evaluation.metrics import token_f1
        assert token_f1("Arthur Clarke", "Arthur C Clarke") == pytest.approx(0.8)

    def test_normalises_like_em(self):
        from src.evaluation.metrics import token_f1
        assert token_f1("THE Eiffel, Tower.", "eiffel tower") == pytest.approx(1.0)

    def test_empty_prediction_returns_zero(self):
        from src.evaluation.metrics import token_f1
        assert token_f1("", "Marie Curie") == 0.0

    def test_empty_gold_returns_zero(self):
        from src.evaluation.metrics import token_f1
        assert token_f1("Marie Curie", "") == 0.0


# ---------------------------------------------------------------------------
# qa_hallucination_rate
# ---------------------------------------------------------------------------

class TestQaHallucinationRate:
    def test_all_grounded_returns_zero(self):
        predicted = ["Warsaw"]
        passages = [["Warsaw is the capital of Poland."]]
        assert qa_hallucination_rate(predicted, passages) == pytest.approx(0.0)

    def test_none_grounded_returns_one(self):
        predicted = ["xyzzyquux"]
        passages = [["Marie Curie was born in Poland."]]
        assert qa_hallucination_rate(predicted, passages) == pytest.approx(1.0)

    def test_mixed_returns_correct_fraction(self):
        predicted = ["Warsaw", "xyzzyquux"]
        passages = [
            ["Warsaw is the capital of Poland."],
            ["Marie Curie was born in Poland."],
        ]
        assert qa_hallucination_rate(predicted, passages) == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        assert qa_hallucination_rate([], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# contradiction_detection_rate
# ---------------------------------------------------------------------------

class TestContradictionDetectionRate:
    def test_proportion_of_true_flags(self):
        assert contradiction_detection_rate([True, True, False, True, False]) == pytest.approx(0.6)

    def test_all_true(self):
        assert contradiction_detection_rate([True, True, True]) == pytest.approx(1.0)

    def test_all_false(self):
        assert contradiction_detection_rate([False, False]) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self):
        assert contradiction_detection_rate([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# retrieval_accuracy_correlation
# ---------------------------------------------------------------------------

class TestRetrievalAccuracyCorrelation:
    def test_perfect_positive_correlation(self):
        vals = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = retrieval_accuracy_correlation(vals, vals)
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-9)
        assert result["spearman_r"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_correlation(self):
        precision = [0.0, 0.25, 0.5, 0.75, 1.0]
        accuracy_vals = [1.0, 0.75, 0.5, 0.25, 0.0]
        result = retrieval_accuracy_correlation(precision, accuracy_vals)
        assert result["pearson_r"] == pytest.approx(-1.0, abs=1e-9)
        assert result["spearman_r"] == pytest.approx(-1.0, abs=1e-9)

    def test_returns_all_keys(self):
        result = retrieval_accuracy_correlation([0.1, 0.5, 0.9], [0.2, 0.6, 0.8])
        assert set(result.keys()) == {"pearson_r", "pearson_p", "spearman_r", "spearman_p"}

    def test_p_values_in_range(self):
        import math
        precision = [0.1 * i for i in range(10)]
        accuracy_vals = [0.05 + 0.09 * i for i in range(10)]
        result = retrieval_accuracy_correlation(precision, accuracy_vals)
        assert 0.0 <= result["pearson_p"] <= 1.0
        assert 0.0 <= result["spearman_p"] <= 1.0
        assert not math.isnan(result["pearson_r"])
        assert not math.isnan(result["spearman_r"])

    def test_constant_series_returns_nan(self):
        import math
        result = retrieval_accuracy_correlation([0.5, 0.5, 0.5], [0.1, 0.5, 0.9])
        assert math.isnan(result["pearson_r"])
        assert math.isnan(result["spearman_r"])

    def test_empty_returns_nan(self):
        import math
        result = retrieval_accuracy_correlation([], [])
        assert math.isnan(result["pearson_r"])
        assert math.isnan(result["spearman_r"])
