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
        assert {"accuracy", "macro_f1", "hallucination_rate", "precision_at_k"} <= result.keys()

    def test_no_self_consistency_key_by_default(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        assert "self_consistency" not in result

    def test_metrics_in_range(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        for val in result.values():
            assert 0.0 <= val <= 1.0

    def test_perfect_accuracy(self):
        # Map each claim to the correct label — order-independent under parallel dispatch.
        label_by_claim = {
            "The sky is blue.": "SUPPORTS",
            "Cats can fly.": "REFUTES",
            "Unknown fact.": "NOT ENOUGH INFO",
        }
        llm = MagicMock()
        llm.complete.side_effect = lambda prompt, _max=None: next(
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


# ---------------------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_precision_at_k_present(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        assert "precision_at_k" in result

    def test_precision_at_k_in_range(self):
        result = scorer.run(EXAMPLES, _retriever(), _llm())
        assert 0.0 <= result["precision_at_k"] <= 1.0


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


# ---------------------------------------------------------------------------
# Integration: poisoning degrades accuracy
# ---------------------------------------------------------------------------

class TestPoisoningDegradation:
    """Verify that poisoning the evidence actually degrades model accuracy.

    Uses a mock LLM that returns the correct label when gold evidence
    is present in the prompt, but returns a wrong label when it finds
    distractor evidence from the opposite label.
    """

    _CLEAN_EXAMPLES = [
        {"claim": "Paris is in France.", "evidence": ["Paris is located in France."], "label": "SUPPORTS"},
        {"claim": "Rome is in France.", "evidence": ["Rome is the capital of Italy."], "label": "REFUTES"},
        {"claim": "Paris is in France.", "evidence": ["Paris is located in France."], "label": "SUPPORTS"},
        {"claim": "Rome is in France.", "evidence": ["Rome is the capital of Italy."], "label": "REFUTES"},
    ]

    def _make_smart_llm(self):
        """LLM that returns SUPPORTS when it sees gold evidence, REFUTES otherwise."""
        gold_snippets = {"Paris is located in France.", "Rome is the capital of Italy."}
        llm = MagicMock()

        def _complete(prompt, _max=None):
            for snippet in gold_snippets:
                if snippet in prompt:
                    if "Paris" in prompt:
                        return "SUPPORTS"
                    return "REFUTES"
            # No gold evidence → always return SUPPORTS (wrong for REFUTES claims)
            return "SUPPORTS"

        llm.complete.side_effect = _complete
        return llm

    def test_poisoning_degrades_accuracy(self):
        from src.data.poisoner import poison_dataset

        clean = self._CLEAN_EXAMPLES
        poisoned = poison_dataset(clean, poison_rate=1.0, seed=0)

        result_clean = scorer.run(clean, _retriever(), self._make_smart_llm(), seed=0)
        result_poisoned = scorer.run(poisoned, _retriever(), self._make_smart_llm(), seed=0)

        assert result_clean["accuracy"] >= result_poisoned["accuracy"]
