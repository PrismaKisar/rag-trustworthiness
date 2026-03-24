"""Tests for src/evaluation/scorer.py."""

from unittest.mock import MagicMock

import pytest

from src.evaluation import scorer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLES = [
    {"claim": "The sky is blue.", "evidence": ["The sky appears blue."], "label": "SUPPORTS"},
    {"claim": "Cats can fly.", "evidence": ["Cats are mammals."], "label": "REFUTES"},
    {"claim": "Unknown fact.", "evidence": ["Some text."], "label": "NOT ENOUGH INFO"},
]


def _retriever(passages=("passage A", "passage B")):
    r = MagicMock()
    r.retrieve.return_value = list(passages)
    return r


def _llm(response="SUPPORTS"):
    m = MagicMock()
    m.complete.return_value = response
    return m


# ---------------------------------------------------------------------------
# Basic runs
# ---------------------------------------------------------------------------

class TestRunBasic:
    def test_returns_required_keys(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        assert {"accuracy", "macro_f1", "hallucination_rate"} <= result.keys()

    def test_no_self_consistency_key_by_default(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        assert "self_consistency" not in result

    def test_metrics_in_range(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        for val in result.values():
            assert 0.0 <= val <= 1.0

    def test_perfect_accuracy(self):
        llm = MagicMock()
        llm.complete.side_effect = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        result = scorer.run(EXAMPLES, _retriever(), llm, prompt_type="standard")
        assert result["accuracy"] == pytest.approx(1.0)

    def test_retriever_build_called_per_example(self):
        r = _retriever()
        scorer.run(EXAMPLES, r, _llm())
        assert r.build.call_count == len(EXAMPLES)

    def test_retriever_retrieve_called_per_example(self):
        r = _retriever()
        scorer.run(EXAMPLES, r, _llm())
        assert r.retrieve.call_count == len(EXAMPLES)


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------

class TestSelfConsistency:
    def test_key_present_when_runs_gt_1(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm(), self_consistency_runs=3)
        assert "self_consistency" in result

    def test_in_range(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm(), self_consistency_runs=3)
        assert 0.0 <= result["self_consistency"] <= 1.0

    def test_llm_called_n_times_per_example(self):
        llm = _llm("SUPPORTS")
        n_runs = 3
        scorer.run(EXAMPLES, _retriever(), llm, self_consistency_runs=n_runs)
        assert llm.complete.call_count == len(EXAMPLES) * n_runs

    def test_consistent_prediction_is_1(self):
        # All runs return SUPPORTS → consistency = 1.0
        result = scorer.run(EXAMPLES, _retriever(), _llm("SUPPORTS"), self_consistency_runs=5)
        assert result["self_consistency"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Prompt types
# ---------------------------------------------------------------------------

class TestPromptTypes:
    @pytest.mark.parametrize("prompt_type", ["standard", "chain_of_thought", "vigilant"])
    def test_all_prompt_types_run(self, prompt_type):
        result = scorer.run(EXAMPLES, _retriever(), _llm(), prompt_type=prompt_type)
        assert "accuracy" in result
