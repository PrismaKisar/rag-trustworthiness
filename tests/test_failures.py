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

def _case(gold_label="SUPPORTS", passages=None, gold_passages=None):
    return EvaluationCase(
        claim="The sky is blue.",
        gold_label=gold_label,
        passages=passages or ["g1", "g2", "g3", "g4", "g5"],
        gold_passages=gold_passages or ["g1", "g2", "g3", "g4", "g5"],
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

    def test_nei_collapse_no_gold_retrieved(self):
        # No gold passages retrieved → model gives up and says NEI
        case = _case(
            gold_label="SUPPORTS",
            passages=["distractor1", "distractor2", "distractor3"],
            gold_passages=["g1", "g2"],
        )
        result = _result("NOT ENOUGH INFO")
        assert classify_failure(case, result) == "nei_collapse"

    def test_nei_collapse_requires_no_gold_retrieved(self):
        # If at least one gold passage is retrieved, it is not a NEI collapse
        case = _case(
            gold_label="SUPPORTS",
            passages=["g1", "distractor1", "distractor2"],
            gold_passages=["g1", "g2"],
        )
        result = _result("NOT ENOUGH INFO")
        assert classify_failure(case, result) != "nei_collapse"

    def test_nei_collapse_requires_factual_gold(self):
        # Gold is already NEI - predicting NEI is correct, not a collapse
        case = _case(
            gold_label="NOT ENOUGH INFO",
            passages=["d1", "d2", "d3"],
            gold_passages=["g1"],
        )
        result = _result("NOT ENOUGH INFO")
        assert classify_failure(case, result) == "correct"

    def test_convinced_by_minority_poison(self):
        # 4 gold passages retrieved, 1 non-gold → minority poisons the prediction
        case = _case(
            gold_label="SUPPORTS",
            passages=["g1", "g2", "g3", "g4", "poison1"],
            gold_passages=["g1", "g2", "g3", "g4", "g5"],
        )
        result = _result("REFUTES")
        assert classify_failure(case, result) == "convinced_by_minority_poison"

    def test_convinced_by_minority_requires_majority_gold(self):
        # 2 gold, 3 non-gold → non-gold majority, should NOT be convinced_by_minority
        case = _case(
            gold_label="SUPPORTS",
            passages=["g1", "g2", "p1", "p2", "p3"],
            gold_passages=["g1", "g2", "g3", "g4"],
        )
        result = _result("REFUTES")
        assert classify_failure(case, result) != "convinced_by_minority_poison"

    def test_other_error_all_gold_retrieved_but_wrong(self):
        # All retrieved passages are gold, yet model still predicts wrong
        case = _case(
            gold_label="SUPPORTS",
            passages=["g1", "g2"],
            gold_passages=["g1", "g2"],
        )
        result = _result("REFUTES")
        assert classify_failure(case, result) == "other_error"

    def test_other_error_no_gold_wrong_non_nei(self):
        # No gold retrieved, but model predicts REFUTES (not NEI)
        case = _case(
            gold_label="SUPPORTS",
            passages=["d1", "d2"],
            gold_passages=["g1", "g2"],
        )
        result = _result("REFUTES")
        assert classify_failure(case, result) == "other_error"


# ---------------------------------------------------------------------------
# find_failure_examples
# ---------------------------------------------------------------------------

class TestFindFailureExamples:
    def _make_data(self):
        cases = [
            # index 0: correct
            _case(gold_label="SUPPORTS", passages=["g1", "g2"], gold_passages=["g1", "g2"]),
            # index 1: convinced_by_minority_poison
            _case(gold_label="SUPPORTS", passages=["g1", "g2", "g3", "g4", "p1"],
                  gold_passages=["g1", "g2", "g3", "g4", "g5"]),
            # index 2: nei_collapse
            _case(gold_label="SUPPORTS", passages=["d1", "d2", "d3"],
                  gold_passages=["g1", "g2"]),
            # index 3: another convinced_by_minority_poison
            _case(gold_label="REFUTES", passages=["g1", "g2", "g3", "g4", "p2"],
                  gold_passages=["g1", "g2", "g3", "g4", "g5"]),
        ]
        results = [
            _result("SUPPORTS", 0),
            _result("REFUTES", 1),
            _result("NOT ENOUGH INFO", 2),
            _result("SUPPORTS", 3),
        ]
        return cases, results

    def test_finds_convinced_by_minority_examples(self):
        cases, results = self._make_data()
        found = find_failure_examples(cases, results, "convinced_by_minority_poison")
        assert len(found) == 2
        assert all(classify_failure(c, r) == "convinced_by_minority_poison" for c, r in found)

    def test_finds_nei_collapse_examples(self):
        cases, results = self._make_data()
        found = find_failure_examples(cases, results, "nei_collapse")
        assert len(found) == 1
        assert classify_failure(found[0][0], found[0][1]) == "nei_collapse"

    def test_max_examples_is_respected(self):
        cases, results = self._make_data()
        found = find_failure_examples(cases, results, "convinced_by_minority_poison", max_examples=1)
        assert len(found) == 1

    def test_returns_empty_when_none_match(self):
        cases, results = self._make_data()
        found = find_failure_examples(cases, results, "other_error")
        assert found == []

    def test_returns_case_result_pairs(self):
        cases, results = self._make_data()
        found = find_failure_examples(cases, results, "nei_collapse")
        assert len(found) == 1
        c, r = found[0]
        assert hasattr(c, "claim")
        assert hasattr(r, "predicted_label")


# ---------------------------------------------------------------------------
# find_vigilant_contrast_examples
# ---------------------------------------------------------------------------

class TestFindVigilantContrastExamples:
    def _make_contrast_data(self):
        # Case 0: standard wrong, vigilant correct  → "helped"
        # Case 1: both wrong                        → "failed"
        # Case 2: both correct                      → neither
        # Case 3: standard correct, vigilant wrong  → neither (vigilant regressed)
        cases = [_case(gold_label="SUPPORTS") for _ in range(4)]
        std_results = [
            _result("REFUTES", 0),
            _result("REFUTES", 1),
            _result("SUPPORTS", 2),
            _result("SUPPORTS", 3),
        ]
        vig_results = [
            _result("SUPPORTS", 0),
            _result("REFUTES", 1),
            _result("SUPPORTS", 2),
            _result("REFUTES", 3),
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
        # Duplicate case 0 to get 2 "helped" candidates
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
