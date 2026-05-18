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
    r.k = 10
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

    def test_no_recall_at_k_key(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        assert "recall_at_k" not in result

    def test_no_self_consistency_key(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        assert "self_consistency" not in result

    def test_metrics_in_range(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        for val in result.values():
            assert 0.0 <= val <= 1.0

    def test_perfect_accuracy(self):
        label_by_claim = {
            "The sky is blue.": "SUPPORTS",
            "Cats can fly.": "REFUTES",
            "Unknown fact.": "NOT ENOUGH INFO",
        }
        llm = MagicMock()
        llm.complete.side_effect = lambda prompt, _pt=None: next(
            v for k, v in label_by_claim.items() if k in prompt
        )
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
# Prompt types
# ---------------------------------------------------------------------------

class TestPromptTypes:
    @pytest.mark.parametrize("prompt_type", ["standard", "chain_of_thought", "vigilant"])
    def test_all_prompt_types_run(self, prompt_type):
        result = scorer.run(EXAMPLES, _retriever(), _llm(), prompt_type=prompt_type)
        assert "accuracy" in result


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_metrics(self):
        llm1 = _llm("SUPPORTS")
        llm2 = _llm("SUPPORTS")
        r1 = scorer.run(EXAMPLES, _retriever(), llm1, seed=42)
        r2 = scorer.run(EXAMPLES, _retriever(), llm2, seed=42)
        assert r1 == r2
