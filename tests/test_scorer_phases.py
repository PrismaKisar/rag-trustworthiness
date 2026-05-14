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

_MAX_TOKENS = {"standard": 64, "chain_of_thought": 512, "vigilant": 256}


def _retriever(passages=("passage A", "passage B")):
    r = MagicMock()
    r.retrieve.return_value = list(passages)
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
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), max_tokens_by_prompt=_MAX_TOKENS)
        assert len(cases) == len(EXAMPLES)

    def test_gold_labels_match_examples(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), max_tokens_by_prompt=_MAX_TOKENS)
        for case, ex in zip(cases, EXAMPLES):
            assert case.gold_label == ex["label"]

    def test_claims_match_examples(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), max_tokens_by_prompt=_MAX_TOKENS)
        for case, ex in zip(cases, EXAMPLES):
            assert case.claim == ex["claim"]

    def test_single_run_yields_one_prompt_per_case(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), sc_runs=1, max_tokens_by_prompt=_MAX_TOKENS)
        for case in cases:
            assert len(case.prompts) == 1

    def test_multiple_runs_yields_matching_prompt_count(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), sc_runs=3, max_tokens_by_prompt=_MAX_TOKENS)
        for case in cases:
            assert len(case.prompts) == 3

    def test_prompt_contains_claim_text(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), sc_runs=1, max_tokens_by_prompt=_MAX_TOKENS)
        for case in cases:
            assert case.claim in case.prompts[0]

    def test_prompt_contains_retrieved_passages(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(("gold A", "gold B")), sc_runs=1, max_tokens_by_prompt=_MAX_TOKENS)
        for case in cases:
            assert "gold A" in case.prompts[0]
            assert "gold B" in case.prompts[0]

    def test_passages_come_from_retriever(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(("x", "y", "z")), max_tokens_by_prompt=_MAX_TOKENS)
        for case in cases:
            assert case.passages == ["x", "y", "z"]

    def test_gold_passages_are_strings(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), max_tokens_by_prompt=_MAX_TOKENS)
        for case in cases:
            assert isinstance(case.gold_passages, list)
            assert all(isinstance(p, str) for p in case.gold_passages)

    def test_gold_passages_contain_original_evidence(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), max_tokens_by_prompt=_MAX_TOKENS)
        for case, ex in zip(cases, EXAMPLES):
            for ev in ex["evidence"]:
                assert ev in case.gold_passages

    def test_max_tokens_from_dict(self):
        custom = {"standard": 99, "chain_of_thought": 999, "vigilant": 499}
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), prompt_type="chain_of_thought", max_tokens_by_prompt=custom)
        for case in cases:
            assert case.max_tokens == 999

    def test_max_tokens_default_standard(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), prompt_type="standard")
        for case in cases:
            assert case.max_tokens == 64

    def test_retriever_build_called_per_example(self):
        r = _retriever()
        scorer.prepare_cases(EXAMPLES, r, max_tokens_by_prompt=_MAX_TOKENS)
        assert r.build.call_count == len(EXAMPLES)

    def test_retriever_retrieve_called_per_example(self):
        r = _retriever()
        scorer.prepare_cases(EXAMPLES, r, max_tokens_by_prompt=_MAX_TOKENS)
        assert r.retrieve.call_count == len(EXAMPLES)

    def test_sc_run_prompts_contain_same_passages(self):
        """All sc-run prompts for one case contain the same passages (just shuffled)."""
        passages = ("A", "B", "C")
        cases = scorer.prepare_cases(EXAMPLES[:1], _retriever(passages), sc_runs=5, seed=0, max_tokens_by_prompt=_MAX_TOKENS)
        case = cases[0]
        for prompt in case.prompts:
            for p in passages:
                assert p in prompt

    def test_deterministic_with_same_seed(self):
        r1, r2 = _retriever(), _retriever()
        cases1 = scorer.prepare_cases(EXAMPLES, r1, sc_runs=3, seed=7, max_tokens_by_prompt=_MAX_TOKENS)
        cases2 = scorer.prepare_cases(EXAMPLES, r2, sc_runs=3, seed=7, max_tokens_by_prompt=_MAX_TOKENS)
        for c1, c2 in zip(cases1, cases2):
            assert c1.prompts == c2.prompts

    def test_returns_evaluation_case_instances(self):
        cases = scorer.prepare_cases(EXAMPLES, _retriever(), max_tokens_by_prompt=_MAX_TOKENS)
        for case in cases:
            assert isinstance(case, EvaluationCase)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

class TestResolve:
    def _make_cases(self, sc_runs=1):
        return scorer.prepare_cases(EXAMPLES, _retriever(), sc_runs=sc_runs, max_tokens_by_prompt=_MAX_TOKENS)

    def test_returns_one_result_per_case(self):
        cases = self._make_cases()
        results = scorer.resolve(cases, _llm())
        assert len(results) == len(cases)

    def test_runs_length_matches_prompt_count(self):
        cases = self._make_cases(sc_runs=3)
        results = scorer.resolve(cases, _llm())
        for case, result in zip(cases, results):
            assert len(result.runs) == len(case.prompts)

    def test_llm_called_once_per_prompt_total(self):
        sc_runs = 3
        cases = self._make_cases(sc_runs=sc_runs)
        llm = _llm()
        scorer.resolve(cases, llm)
        assert llm.complete.call_count == len(EXAMPLES) * sc_runs

    def test_predicted_label_is_majority(self):
        cases = self._make_cases(sc_runs=3)
        llm = MagicMock()
        # First case: 2 SUPPORTS + 1 REFUTES → SUPPORTS
        llm.complete.side_effect = ["SUPPORTS", "SUPPORTS", "REFUTES"] * len(EXAMPLES)
        results = scorer.resolve(cases, llm)
        assert results[0].predicted_label == "SUPPORTS"

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

    def test_runs_contain_extracted_labels(self):
        cases = self._make_cases(sc_runs=1)
        results = scorer.resolve(cases, _llm("REFUTES"))
        for result in results:
            assert result.runs[0] in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def _make_cases_and_results(self, predictions, gold_labels, sc_runs=1):
        """Build minimal EvaluationCase + EvaluationResult pairs from explicit labels."""
        assert len(predictions) == len(gold_labels)
        cases = [
            EvaluationCase(
                claim=f"claim {i}",
                gold_label=gold,
                passages=["p"],
                gold_passages=["p"],
                prompts=["prompt"] * sc_runs,
                max_tokens=64,
            )
            for i, gold in enumerate(gold_labels)
        ]
        results = [
            EvaluationResult(
                case_index=i,
                runs=[pred] * sc_runs,
                predicted_label=pred,
            )
            for i, pred in enumerate(predictions)
        ]
        return cases, results

    def test_returns_required_keys(self):
        cases, results = self._make_cases_and_results(["SUPPORTS"], ["SUPPORTS"])
        out = scorer.aggregate(cases, results)
        assert {"accuracy", "macro_f1", "hallucination_rate", "precision_at_k"} <= out.keys()

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

    def test_precision_at_k_perfect_when_passages_match_gold(self):
        cases = [
            EvaluationCase(
                claim="c", gold_label="SUPPORTS",
                passages=["gold passage"],
                gold_passages=["gold passage"],
                prompts=["p"], max_tokens=64,
            )
        ]
        results = [EvaluationResult(case_index=0, runs=["SUPPORTS"], predicted_label="SUPPORTS")]
        out = scorer.aggregate(cases, results)
        assert out["precision_at_k"] == pytest.approx(1.0)

    def test_precision_at_k_zero_when_no_overlap(self):
        cases = [
            EvaluationCase(
                claim="c", gold_label="SUPPORTS",
                passages=["distractor"],
                gold_passages=["gold passage"],
                prompts=["p"], max_tokens=64,
            )
        ]
        results = [EvaluationResult(case_index=0, runs=["SUPPORTS"], predicted_label="SUPPORTS")]
        out = scorer.aggregate(cases, results)
        assert out["precision_at_k"] == pytest.approx(0.0)

    def test_no_self_consistency_key_for_single_run(self):
        cases, results = self._make_cases_and_results(["SUPPORTS"], ["SUPPORTS"], sc_runs=1)
        out = scorer.aggregate(cases, results)
        assert "self_consistency" not in out

    def test_self_consistency_key_present_for_multiple_runs(self):
        cases, results = self._make_cases_and_results(["SUPPORTS"], ["SUPPORTS"], sc_runs=3)
        out = scorer.aggregate(cases, results)
        assert "self_consistency" in out

    def test_self_consistency_perfect_when_all_runs_agree(self):
        cases, results = self._make_cases_and_results(["SUPPORTS"], ["SUPPORTS"], sc_runs=3)
        out = scorer.aggregate(cases, results)
        assert out["self_consistency"] == pytest.approx(1.0)

    def test_all_values_in_range(self):
        preds = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        gold = ["SUPPORTS", "REFUTES", "SUPPORTS"]
        cases, results = self._make_cases_and_results(preds, gold)
        out = scorer.aggregate(cases, results)
        for v in out.values():
            assert 0.0 <= v <= 1.0
