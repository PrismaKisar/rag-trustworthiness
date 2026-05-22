"""Tests for the three phases of scorer: prepare_cases, resolve, aggregate."""

from unittest.mock import MagicMock

import pytest

from src.evaluation import scorer
from src.evaluation.cases import EvaluationCase, EvaluationResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EXAMPLES = [
    {"claim": "The sky is blue.", "evidence": ["The sky appears blue."], "label": "SUPPORTS"},
    {"claim": "Cats can fly.", "evidence": ["Cats are mammals."], "label": "REFUTES"},
    {"claim": "Unknown fact.", "evidence": ["Some text."], "label": "NOT ENOUGH INFO"},
]


def _retriever(passages=("passage A", "passage B")):
    r = MagicMock()
    r.retrieve.return_value = list(passages)
    r.k = 10
    return r


def _llm(response="SUPPORTS"):
    m = MagicMock()
    m.complete.return_value = response
    return m


# ---------------------------------------------------------------------------
# prepare_cases
# ---------------------------------------------------------------------------

class TestPrepareCases:
    def test_returns_one_case_per_example(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever())
        assert len(cases) == len(EXAMPLES)

    def test_gold_labels_match_examples(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever())
        for case, ex in zip(cases, EXAMPLES):
            assert case.gold_label == ex["label"]

    def test_claims_match_examples(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever())
        for case, ex in zip(cases, EXAMPLES):
            assert case.claim == ex["claim"]

    def test_one_prompt_per_case(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever())
        for case in cases:
            assert len(case.prompts) == 1

    def test_prompt_contains_claim_text(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever())
        for case in cases:
            assert case.claim in case.prompts[0]

    def test_prompt_contains_retrieved_passages(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(("distractor A", "distractor B")))
        for case in cases:
            assert "distractor A" in case.prompts[0]
            assert "distractor B" in case.prompts[0]

    def test_prompt_contains_injected_evidence(self):
        """Evidence from example is injected directly into the context."""
        cases = scorer.prepare_cases(EXAMPLES, _retriever())
        for case, ex in zip(cases, EXAMPLES):
            for ev in ex["evidence"]:
                assert ev in case.prompts[0]

    def test_passages_include_injected_evidence(self):
        """case.passages must contain the example's evidence."""
        cases = scorer.prepare_cases(EXAMPLES, _retriever())
        for case, ex in zip(cases, EXAMPLES):
            for ev in ex["evidence"]:
                assert ev in case.passages

    def test_passages_retrieved_count_is_k_minus_n_gold(self):
        """Retriever is called with k - n_gold, not full k."""
        for ex in EXAMPLES:
            r = _retriever()
            r.k = 10
            scorer.prepare_cases([ex], r)
            n_gold = len(ex["evidence"])
            _, call_kwargs = r.retrieve.call_args
            assert call_kwargs.get("k", None) == 10 - n_gold

    def test_prompt_type_stored_on_case(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), prompt_type="chain_of_thought")
        for case in cases:
            assert case.prompt_type == "chain_of_thought"

    def test_retriever_build_called_per_example(self):
        r = _retriever()
        scorer.prepare_cases(EXAMPLES, r)
        assert r.build.call_count == len(EXAMPLES)

    def test_retriever_retrieve_called_per_example(self):
        r = _retriever()
        scorer.prepare_cases(EXAMPLES, r)
        assert r.retrieve.call_count == len(EXAMPLES)

    def test_returns_evaluation_case_instances(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever())
        for case in cases:
            assert isinstance(case, EvaluationCase)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

class TestResolve:
    def _make_cases(self):
        return scorer.prepare_cases(EXAMPLES, _retriever())

    def test_empty_cases_returns_empty_list(self):
        assert scorer.resolve([], _llm()) == []

    def test_returns_one_result_per_case(self):
        cases = self._make_cases()
        results = scorer.resolve(cases, _llm())
        assert len(results) == len(cases)

    def test_runs_length_is_one(self):
        cases = self._make_cases()
        results = scorer.resolve(cases, _llm())
        for result in results:
            assert len(result.runs) == 1

    def test_llm_called_once_per_example(self):
        cases = self._make_cases()
        llm = _llm()
        scorer.resolve(cases, llm)
        assert llm.complete.call_count == len(EXAMPLES)

    def test_predicted_label_from_llm_response(self):
        cases = self._make_cases()
        results = scorer.resolve(cases, _llm("REFUTES"))
        for result in results:
            assert result.predicted_label == "REFUTES"

    def test_case_index_matches_position(self):
        cases = self._make_cases()
        results = scorer.resolve(cases, _llm())
        for i, result in enumerate(results):
            assert result.case_index == i

    def test_returns_evaluation_result_instances(self):
        cases = self._make_cases()
        results = scorer.resolve(cases, _llm())
        for result in results:
            assert isinstance(result, EvaluationResult)

    def test_contradiction_flag_true_when_response_contains_contradiction(self):
        cases = scorer.prepare_cases(
            [{"claim": "C", "evidence": ["p1", "p2"], "label": "REFUTES"}],
            _retriever(),
            prompt_type="vigilant",
        )
        vigilant_response = (
            "Consistency check: The passages contradict each other.\n"
            "Final Label (SUPPORTS / REFUTES / NOT ENOUGH INFO): REFUTES"
        )
        results = scorer.resolve(cases, _llm(vigilant_response))
        assert results[0].contradiction_flag is True

    def test_contradiction_flag_false_when_passages_consistent(self):
        cases = scorer.prepare_cases(
            [{"claim": "C", "evidence": ["p1"], "label": "SUPPORTS"}],
            _retriever(),
            prompt_type="vigilant",
        )
        consistent_response = (
            "Consistency check: The passages are consistent.\n"
            "Final Label (SUPPORTS / REFUTES / NOT ENOUGH INFO): SUPPORTS"
        )
        results = scorer.resolve(cases, _llm(consistent_response))
        assert results[0].contradiction_flag is False


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def _make_cases_and_results(self, predictions, gold_labels):
        assert len(predictions) == len(gold_labels)
        cases = [
            EvaluationCase(
                claim=f"claim {i}",
                gold_label=gold,
                passages=["p"],
                prompts=["prompt"],
                prompt_type="standard",
            )
            for i, gold in enumerate(gold_labels)
        ]
        results = [
            EvaluationResult(
                case_index=i,
                runs=[pred],
                predicted_label=pred,
            )
            for i, pred in enumerate(predictions)
        ]
        return cases, results

    def test_returns_required_keys(self):
        cases, results = self._make_cases_and_results(["SUPPORTS"], ["SUPPORTS"])
        out = scorer.aggregate(cases, results)
        assert {"accuracy", "macro_f1", "hallucination_rate"} <= out.keys()

    def test_no_recall_at_k_key(self):
        cases, results = self._make_cases_and_results(["SUPPORTS"], ["SUPPORTS"])
        out = scorer.aggregate(cases, results)
        assert "recall_at_k" not in out

    def test_no_self_consistency_key(self):
        cases, results = self._make_cases_and_results(["SUPPORTS"], ["SUPPORTS"])
        out = scorer.aggregate(cases, results)
        assert "self_consistency" not in out

    def test_perfect_accuracy(self):
        preds = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        gold = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        cases, results = self._make_cases_and_results(preds, gold)
        out = scorer.aggregate(cases, results)
        assert out["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        preds = ["REFUTES", "SUPPORTS", "SUPPORTS"]
        gold = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        cases, results = self._make_cases_and_results(preds, gold)
        out = scorer.aggregate(cases, results)
        assert out["accuracy"] == pytest.approx(0.0)

    def test_all_values_in_range(self):
        preds = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        gold = ["SUPPORTS", "REFUTES", "SUPPORTS"]
        cases, results = self._make_cases_and_results(preds, gold)
        out = scorer.aggregate(cases, results)
        for v in out.values():
            assert 0.0 <= v <= 1.0

    def test_contradiction_detection_rate_present_for_vigilant(self):
        cases, _ = self._make_cases_and_results(["SUPPORTS", "REFUTES"], ["SUPPORTS", "REFUTES"])
        results = [
            EvaluationResult(case_index=0, runs=["SUPPORTS"], predicted_label="SUPPORTS", contradiction_flag=True),
            EvaluationResult(case_index=1, runs=["REFUTES"], predicted_label="REFUTES", contradiction_flag=False),
        ]
        out = scorer.aggregate(cases, results, prompt_type="vigilant")
        assert "contradiction_detection_rate" in out
        assert out["contradiction_detection_rate"] == pytest.approx(0.5)

    def test_contradiction_detection_rate_absent_for_standard(self):
        cases, results = self._make_cases_and_results(["SUPPORTS"], ["SUPPORTS"])
        out = scorer.aggregate(cases, results, prompt_type="standard")
        assert "contradiction_detection_rate" not in out


# ---------------------------------------------------------------------------
# EvaluationResult.contradiction_flag
# ---------------------------------------------------------------------------

class TestEvaluationResultContradictionFlag:
    def test_default_is_false(self):
        r = EvaluationResult(case_index=0, runs=["SUPPORTS"], predicted_label="SUPPORTS")
        assert r.contradiction_flag is False

    def test_can_set_true(self):
        r = EvaluationResult(case_index=0, runs=["REFUTES"], predicted_label="REFUTES", contradiction_flag=True)
        assert r.contradiction_flag is True
