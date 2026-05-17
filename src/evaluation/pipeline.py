"""EvaluationTask Protocol and run_pipeline composer.

Decouples the three-phase scorer structure from dataset-specific logic.
Any object implementing EvaluationTask can be driven through run_pipeline.

Attribution:
    Deep-module pattern - Ousterhout, "A Philosophy of Software Design" §4.
"""

from __future__ import annotations

from typing import Protocol

from src.evaluation.dispatch import resolve_raw


class EvaluationTask(Protocol):
    def build_cases(
        self,
        examples: list[dict],
        retriever,
        prompt_type: str,
        sc_runs: int,
        seed: int,
        **kwargs,
    ) -> list:
        ...

    def parse_result(self, case_index: int, raw_runs: list[str]) -> object:
        ...

    def compute_metrics(
        self,
        cases: list,
        results: list,
        prompt_type: str,
    ) -> dict[str, float]:
        ...


def run_pipeline(
    task: EvaluationTask,
    examples: list[dict],
    retriever,
    llm,
    prompt_type: str = "standard",
    sc_runs: int = 1,
    seed: int = 42,
    n_workers: int = 4,
    **task_kwargs,
) -> dict[str, float]:
    """Drive *task* through the three-phase pipeline and return metrics."""
    cases = task.build_cases(examples, retriever, prompt_type, sc_runs, seed, **task_kwargs)
    raw = resolve_raw(cases, llm, n_workers=n_workers)
    results = [task.parse_result(i, runs) for i, runs in enumerate(raw)]
    return task.compute_metrics(cases, results, prompt_type)
