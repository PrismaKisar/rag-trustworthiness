"""Tests for src/evaluation/pipeline.py - EvaluationTask Protocol + run_pipeline composer."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.evaluation.pipeline import run_pipeline


class _FakeCase:
    def __init__(self, idx: int):
        self.prompts = [f"prompt_{idx}"]
        self.prompt_type = "standard"


class _StubTask:
    def build_cases(self, examples, retriever, prompt_type, seed, **kwargs):
        return [_FakeCase(i) for i in range(len(examples))]

    def parse_result(self, case_index: int, raw_runs: list[str]):
        return f"result_{case_index}"

    def compute_metrics(self, cases, results, prompt_type: str) -> dict[str, float]:
        return {"stub_metric": 42.0}


class _StubLLM:
    def complete(self, prompt: str, prompt_type: str = "standard") -> str:
        return "raw_answer"


# ---------------------------------------------------------------------------
# Behavior 1: run_pipeline returns compute_metrics output
# ---------------------------------------------------------------------------


class TestRunPipelineReturnValue:
    def test_returns_compute_metrics_output(self):
        metrics = run_pipeline(
            task=_StubTask(),
            examples=[{"q": "x"}, {"q": "y"}],
            retriever=object(),
            llm=_StubLLM(),
        )
        assert metrics == {"stub_metric": 42.0}


# ---------------------------------------------------------------------------
# Behavior 2: parse_result called once per case with correct case_index
# ---------------------------------------------------------------------------


class _RecordingTask:
    def __init__(self):
        self.parse_calls: list[tuple[int, list[str]]] = []

    def build_cases(self, examples, retriever, prompt_type, seed, **kwargs):
        return [_FakeCase(i) for i in range(len(examples))]

    def parse_result(self, case_index: int, raw_runs: list[str]):
        self.parse_calls.append((case_index, raw_runs))
        return f"result_{case_index}"

    def compute_metrics(self, cases, results, prompt_type: str) -> dict[str, float]:
        return {}


class TestParseResultCalled:
    def test_called_once_per_case_with_correct_index(self):
        task = _RecordingTask()
        run_pipeline(
            task=task,
            examples=[{"q": "a"}, {"q": "b"}, {"q": "c"}],
            retriever=object(),
            llm=_StubLLM(),
        )
        assert len(task.parse_calls) == 3
        assert [idx for idx, _ in task.parse_calls] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Behavior 3: FeverTask integration produces FEVER metrics
# ---------------------------------------------------------------------------

_FEVER_EXAMPLES = [
    {"claim": "The sky is blue.", "evidence": ["The sky appears blue."], "label": "SUPPORTS"},
    {"claim": "Cats can fly.", "evidence": ["Cats are mammals."], "label": "REFUTES"},
]


class TestFeverTaskIntegration:
    def test_produces_fever_metric_keys(self):
        from src.evaluation.scorer import FeverTask

        retriever = MagicMock()
        retriever.retrieve.return_value = ["passage A", "passage B"]
        retriever.k = 10

        llm = MagicMock()
        llm.complete.return_value = "SUPPORTS"

        metrics = run_pipeline(
            task=FeverTask(),
            examples=_FEVER_EXAMPLES,
            retriever=retriever,
            llm=llm,
        )
        assert {"accuracy", "macro_f1", "hallucination_rate"} <= metrics.keys()
        assert "recall_at_k" not in metrics
        for val in metrics.values():
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Behavior 4: HotpotQATask integration produces QA metrics
# ---------------------------------------------------------------------------

_HOTPOT_EXAMPLES = [
    {
        "question": "Where was Marie Curie born?",
        "answer": "Warsaw",
        "supporting_facts": [["Marie Curie", 0]],
        "context": [
            ["Marie Curie", ["Marie Curie was born in Warsaw.", "She won Nobel prizes."]],
            ["Distractor", ["Unrelated content.", "More filler."]],
        ],
    },
    {
        "question": "Who wrote Hamlet?",
        "answer": "Shakespeare",
        "supporting_facts": [["Shakespeare", 0]],
        "context": [
            ["Shakespeare", ["William Shakespeare wrote Hamlet.", "He also wrote Macbeth."]],
            ["Distractor", ["Unrelated content.", "More filler."]],
        ],
    },
]


class TestHotpotQATaskIntegration:
    def test_produces_qa_metric_keys(self):
        from src.evaluation.qa_scorer import HotpotQATask

        retriever = MagicMock()
        retriever.retrieve.return_value = ["passage A", "passage B"]
        retriever.k = 10

        llm = MagicMock()
        llm.complete.return_value = "Answer: Warsaw"

        metrics = run_pipeline(
            task=HotpotQATask(),
            examples=_HOTPOT_EXAMPLES,
            retriever=retriever,
            llm=llm,
            prompt_type="standard_qa",
        )
        assert {"exact_match", "token_f1"} <= metrics.keys()
        assert "recall_at_k" not in metrics
        for val in metrics.values():
            assert 0.0 <= val <= 1.0
