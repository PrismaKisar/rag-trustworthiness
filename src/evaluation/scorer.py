"""Scorer: run a model over FEVER examples and return evaluation metrics.

Three-phase pipeline:
    prepare_cases  - sequential retrieval + injection → list[EvaluationCase]
    resolve        - parallel LLM calls              → list[EvaluationResult]
    aggregate      - pure metric rollup              → dict[str, float]

Context composition per example:
    passages = top-(k - n_gold) retrieved distractors + example["evidence"]

The example's own evidence (clean at r=0, poisoned at r=1) is injected
directly after retrieval; the retriever only sources distractors.

Attribution:
    Sequential RAG pipeline for veracity prediction - Singal et al. 2024 §4.
    Factuality metrics (accuracy, macro-F1, hallucination rate) - Zhou et al. 2024.
"""

from __future__ import annotations

import logging
from collections import Counter

from src.evaluation.cases import EvaluationCase, EvaluationResult
from src.evaluation.dispatch import resolve_raw
from src.evaluation.metrics import (
    accuracy,
    contradiction_detection_rate,
    hallucination_rate,
    macro_f1,
)
from src.generation.llm_client import LLMClient
from src.generation.parser import extract_contradiction_flag, extract_label
from src.generation.prompts import PromptType, format_prompt
from src.retrieval.corpus import build_all_corpora
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 - retrieval + injection (sequential: embedder / FAISS not thread-safe)
# ---------------------------------------------------------------------------

def prepare_cases(
    examples: list[dict],
    retriever: Retriever,
    prompt_type: PromptType = "standard",
    seed: int = 42,
    full_dataset: list[dict] | None = None,
) -> list[EvaluationCase]:
    """Build one EvaluationCase per example via retrieval and direct injection.

    For each example the context is:
        top-(k - n_gold) distractors retrieved from the global gold pool
        + example["evidence"]  (original gold at r=0; passages at r=1)

    Args:
        examples: FEVER examples (keys: ``claim``, ``evidence``, ``label``).
        retriever: Retriever whose index is rebuilt per example.
        prompt_type: One of ``"standard"``, ``"chain_of_thought"``, ``"vigilant"``.
        seed: Unused; kept for API compatibility.
        full_dataset: Fixed global passage pool for retrieval.  Pass the
            complete dev set so the retrieval results — and therefore prompt
            text and LLM cache keys — remain stable when *N* changes.
            Defaults to *examples* (original behaviour) when omitted.

    Returns:
        List of :class:`~src.evaluation.cases.EvaluationCase` objects,
        one per example, in the same order.
    """
    corpora = build_all_corpora(examples, full_dataset=full_dataset)
    cases: list[EvaluationCase] = []
    for example, corpus in zip(examples, corpora):
        retriever.build(corpus)
        n_gold = len(example["evidence"])
        k_distractors = max(0, retriever.k - n_gold)
        retrieved = retriever.retrieve(example["claim"], k=k_distractors)
        passages = retrieved + list(example["evidence"])

        cases.append(EvaluationCase(
            claim=example["claim"],
            gold_label=example["label"],
            passages=passages,
            prompts=[format_prompt(example["claim"], passages, prompt_type)],
            prompt_type=prompt_type,
        ))

    return cases


# ---------------------------------------------------------------------------
# Phase 2 - LLM calls (parallel: I/O-bound, cache is thread-safe)
# ---------------------------------------------------------------------------

def resolve(
    cases: list[EvaluationCase],
    llm: LLMClient,
    n_workers: int = 4,
) -> list[EvaluationResult]:
    """Dispatch LLM calls for every prompt in *cases* and return results."""
    if not cases:
        return []

    raw_per_case = resolve_raw(cases, llm, n_workers=n_workers)
    results: list[EvaluationResult] = []
    for case_idx, (case, raw_runs) in enumerate(zip(cases, raw_per_case)):
        runs = [extract_label(r) for r in raw_runs]
        predicted = Counter(runs).most_common(1)[0][0]
        contradiction_flag = extract_contradiction_flag(raw_runs[0])
        logger.debug(
            "Case %d/%d  gold=%s  pred=%s",
            case_idx + 1, len(cases), case.gold_label, predicted,
        )
        results.append(EvaluationResult(
            case_index=case_idx,
            runs=runs,
            predicted_label=predicted,
            contradiction_flag=contradiction_flag,
        ))

    return results


# ---------------------------------------------------------------------------
# Phase 3 - aggregate
# ---------------------------------------------------------------------------

def aggregate(
    cases: list[EvaluationCase],
    results: list[EvaluationResult],
    prompt_type: PromptType = "standard",
) -> dict[str, float]:
    """Compute evaluation metrics from *cases* and *results*.

    Args:
        cases: Output of :func:`prepare_cases`.
        results: Output of :func:`resolve`, same order as *cases*.
        prompt_type: ``"vigilant"`` adds ``contradiction_detection_rate``.

    Returns:
        Dict with ``accuracy``, ``macro_f1``, ``hallucination_rate``,
        and ``contradiction_detection_rate`` (only when prompt_type == "vigilant").
    """
    gold_labels = [case.gold_label for case in cases]
    predictions = [result.predicted_label for result in results]

    metrics: dict[str, float] = {
        "accuracy": accuracy(predictions, gold_labels),
        "macro_f1": macro_f1(predictions, gold_labels),
        "hallucination_rate": hallucination_rate(predictions, gold_labels),
    }

    if prompt_type == "vigilant":
        metrics["contradiction_detection_rate"] = contradiction_detection_rate(
            [r.contradiction_flag for r in results]
        )

    return metrics


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def run(
    examples: list[dict],
    retriever: Retriever,
    llm: LLMClient,
    prompt_type: PromptType = "standard",
    seed: int = 42,
    n_workers: int = 4,
    full_dataset: list[dict] | None = None,
) -> dict[str, float]:
    """Run *llm* on every example and return aggregated metrics."""
    from src.evaluation.pipeline import run_pipeline
    return run_pipeline(
        task=FeverTask(),
        examples=examples,
        retriever=retriever,
        llm=llm,
        prompt_type=prompt_type,
        seed=seed,
        n_workers=n_workers,
        full_dataset=full_dataset,
    )


# ---------------------------------------------------------------------------
# EvaluationTask adapter
# ---------------------------------------------------------------------------


class FeverTask:
    """Adapts the FEVER three-phase scorer to the EvaluationTask protocol."""

    def build_cases(
        self,
        examples: list[dict],
        retriever,
        prompt_type: str,
        seed: int,
        **kwargs,
    ) -> list[EvaluationCase]:
        return prepare_cases(
            examples=examples,
            retriever=retriever,
            prompt_type=prompt_type,
            seed=seed,
            full_dataset=kwargs.get("full_dataset"),
        )

    def parse_result(self, case_index: int, raw_runs: list[str]) -> EvaluationResult:
        runs = [extract_label(r) for r in raw_runs]
        predicted = Counter(runs).most_common(1)[0][0]
        contradiction_flag = extract_contradiction_flag(raw_runs[0])
        return EvaluationResult(
            case_index=case_index,
            runs=runs,
            predicted_label=predicted,
            contradiction_flag=contradiction_flag,
        )

    def compute_metrics(
        self,
        cases: list[EvaluationCase],
        results: list[EvaluationResult],
        prompt_type: str,
    ) -> dict[str, float]:
        return aggregate(cases, results, prompt_type=prompt_type)
