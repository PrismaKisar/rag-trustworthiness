"""Tests for src/evaluation/failures.py."""

import pytest

from src.evaluation.cases import EvaluationCase, EvaluationResult
from src.evaluation.failures import (
    classify_failure,
    find_failure_examples,
    find_vigilant_contrast_examples,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _case(gold_label="SUPPORTS", passages=None):
    return EvaluationCase(
        claim="The sky is blue.",
        gold_label=gold_label,
        passages=passages or ["p1", "p2", "p3"],
        prompts=["prompt text"],
        prompt_type="standard",
    )


def _result(predicted, case_index=0):
    return EvaluationResult(case_index=case_index, runs=[predicted], predicted_label=predicted)


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------

class TestClassifyFailure:
    def test_correct_prediction_returns_correct(self):
        case = _case(gold_label="SUPPORTS")
        result = _result("SUPPORTS")
        assert classify_failure(case, result) == "correct"

    def test_correct_for_refutes(self):
        case = _case(gold_label="REFUTES")
        result = _result("REFUTES")
        assert classify_failure(case, result) == "correct"

    def test_nei_collapse_for_supports_claim(self):
        case = _case(gold_label="SUPPORTS")
        result = _result("NOT ENOUGH INFO")
        assert classify_failure(case, result) == "nei_collapse"

    def test_nei_collapse_for_refutes_claim(self):
        case = _case(gold_label="REFUTES")
        result = _result("NOT ENOUGH INFO")
        assert classify_failure(case, result) == "nei_collapse"

    def test_nei_correct_when_gold_is_nei(self):
        case = _case(gold_label="NOT ENOUGH INFO")
        result = _result("NOT ENOUGH INFO")
        assert classify_failure(case, result) == "correct"

    def test_other_error_wrong_factual_label(self):
        case = _case(gold_label="SUPPORTS")
        result = _result("REFUTES")
        assert classify_failure(case, result) == "other_error"

    def test_other_error_wrong_non_nei(self):
        case = _case(gold_label="REFUTES")
        result = _result("SUPPORTS")
        assert classify_failure(case, result) == "other_error"


# ---------------------------------------------------------------------------
# find_failure_examples
# ---------------------------------------------------------------------------

class TestFindFailureExamples:
    def _make_data(self):
        cases = [
            _case(gold_label="SUPPORTS"),   # index 0: correct
            _case(gold_label="SUPPORTS"),   # index 1: other_error
            _case(gold_label="SUPPORTS"),   # index 2: nei_collapse
            _case(gold_label="REFUTES"),    # index 3: nei_collapse
        ]
        results = [
            _result("SUPPORTS", 0),
            _result("REFUTES", 1),
            _result("NOT ENOUGH INFO", 2),
            _result("NOT ENOUGH INFO", 3),
        ]
        return cases, results

    def test_finds_nei_collapse_examples(self):
        cases, results = self._make_data()
        found = find_failure_examples(cases, results, "nei_collapse")
        assert len(found) == 2
        assert all(classify_failure(c, r) == "nei_collapse" for c, r in found)

    def test_finds_other_error_examples(self):
        cases, results = self._make_data()
        found = find_failure_examples(cases, results, "other_error")
        assert len(found) == 1
        assert classify_failure(found[0][0], found[0][1]) == "other_error"

    def test_max_examples_is_respected(self):
        cases, results = self._make_data()
        found = find_failure_examples(cases, results, "nei_collapse", max_examples=1)
        assert len(found) == 1

    def test_returns_empty_when_none_match(self):
        cases = [_case(gold_label="SUPPORTS")]
        results = [_result("SUPPORTS", 0)]
        found = find_failure_examples(cases, results, "nei_collapse")
        assert found == []

    def test_returns_case_result_pairs(self):
        cases, results = self._make_data()
        found = find_failure_examples(cases, results, "nei_collapse")
        c, r = found[0]
        assert hasattr(c, "claim")
        assert hasattr(r, "predicted_label")


# ---------------------------------------------------------------------------
# find_vigilant_contrast_examples
# ---------------------------------------------------------------------------

class TestFindVigilantContrastExamples:
    def _make_contrast_data(self):
        cases = [_case(gold_label="SUPPORTS") for _ in range(4)]
        std_results = [
            _result("REFUTES", 0),   # standard wrong
            _result("REFUTES", 1),   # standard wrong
            _result("SUPPORTS", 2),  # standard correct
            _result("SUPPORTS", 3),  # standard correct
        ]
        vig_results = [
            _result("SUPPORTS", 0),  # vigilant correct → "helped"
            _result("REFUTES", 1),   # vigilant wrong   → "failed"
            _result("SUPPORTS", 2),  # both correct
            _result("REFUTES", 3),   # vigilant regressed
        ]
        return cases, std_results, vig_results

    def test_helped_finds_correct_cases(self):
        cases, std, vig = self._make_contrast_data()
        found = find_vigilant_contrast_examples(cases, std, vig, kind="helped")
        assert len(found) == 1
        c, sr, vr = found[0]
        assert sr.predicted_label != c.gold_label
        assert vr.predicted_label == c.gold_label

    def test_failed_finds_correct_cases(self):
        cases, std, vig = self._make_contrast_data()
        found = find_vigilant_contrast_examples(cases, std, vig, kind="failed")
        assert len(found) == 1
        c, sr, vr = found[0]
        assert sr.predicted_label != c.gold_label
        assert vr.predicted_label != c.gold_label

    def test_max_examples_respected(self):
        cases = [_case(gold_label="SUPPORTS")] * 3
        std = [_result("REFUTES", i) for i in range(3)]
        vig = [_result("SUPPORTS", i) for i in range(3)]
        found = find_vigilant_contrast_examples(cases, std, vig, kind="helped", max_examples=2)
        assert len(found) == 2

    def test_returns_triple_tuples(self):
        cases, std, vig = self._make_contrast_data()
        found = find_vigilant_contrast_examples(cases, std, vig, kind="helped")
        assert len(found[0]) == 3

    def test_empty_when_no_match(self):
        cases = [_case(gold_label="SUPPORTS")]
        std = [_result("SUPPORTS", 0)]
        vig = [_result("SUPPORTS", 0)]
        assert find_vigilant_contrast_examples(cases, std, vig, kind="helped") == []
        assert find_vigilant_contrast_examples(cases, std, vig, kind="failed") == []
