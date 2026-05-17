"""Scorer: run a model over FEVER examples and return evaluation metrics.

Three-phase pipeline:
    prepare_cases  - sequential retrieval → list[EvaluationCase]
    resolve        - parallel LLM calls   → list[EvaluationResult]
    aggregate      - pure metric rollup   → dict[str, float]

run() is a thin composer over the three phases with the same public signature
as before.

Attribution:
    Sequential RAG pipeline for veracity prediction - Singal et al. 2024 §4.
    Factuality metrics (accuracy, macro-F1, hallucination rate) - Zhou et al. 2024.
    Self-consistency under passage-order perturbation - Wang et al. 2022
    (cited in Zhou 2024 §2.1 as a robustness-improving prompting technique).
"""

from __future__ import annotations

import logging
import random
from collections import Counter

from src.evaluation.cases import EvaluationCase, EvaluationResult
from src.evaluation.dispatch import resolve_raw
from src.evaluation.metrics import (
    accuracy,
    contradiction_detection_rate,
    hallucination_rate,
    macro_f1,
    recall_at_k,
    self_consistency,
)
from src.generation.llm_client import LLMClient
from src.generation.parser import extract_contradiction_flag, extract_label
from src.generation.prompts import PromptType, format_prompt
from src.retrieval.corpus import build_all_corpora
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 - retrieval (sequential: embedder / FAISS not thread-safe)
# ---------------------------------------------------------------------------

def prepare_cases(
    examples: list[dict],
    retriever: Retriever,
    prompt_type: PromptType = "standard",
    sc_runs: int = 1,
    distractor_pool_size: int = 20,
    seed: int = 42,
) -> list[EvaluationCase]:
    """Build one EvaluationCase per example via sequential retrieval.

    Args:
        examples: FEVER examples (keys: ``claim``, ``evidence``, ``label``).
        retriever: Retriever whose index is rebuilt per example.
        prompt_type: One of ``"standard"``, ``"chain_of_thought"``, ``"vigilant"``.
        sc_runs: Number of prompts per case (passages shuffled between runs ≥2).
        distractor_pool_size: Distractor passages added to each example corpus.
        seed: Base random seed for corpus building and passage shuffling.

    Returns:
        List of :class:`~src.evaluation.cases.EvaluationCase` objects,
        one per example, in the same order.
    """
    corpora = build_all_corpora(examples, distractor_pool_size=distractor_pool_size, seed=seed)
    cases: list[EvaluationCase] = []
    for i, (example, corpus) in enumerate(zip(examples, corpora)):
        retriever.build(corpus)
        passages = retriever.retrieve(example["claim"])
        gold_passages = [corpus.passages[j] for j in corpus.gold_indices]

        rng = random.Random(seed + i)
        prompts = [format_prompt(example["claim"], passages, prompt_type)]
        for _ in range(sc_runs - 1):
            shuffled = list(passages)
            rng.shuffle(shuffled)
            prompts.append(format_prompt(example["claim"], shuffled, prompt_type))

        cases.append(EvaluationCase(
            claim=example["claim"],
            gold_label=example["label"],
            passages=list(passages),
            gold_passages=gold_passages,
            prompts=prompts,
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
    """Dispatch LLM calls for every prompt in *cases* and return results.

    Args:
        cases: Output of :func:`prepare_cases`.
        llm: LLM client implementing ``.complete(prompt, max_tokens) -> str``.
        n_workers: Thread-pool size for parallel dispatch.

    Returns:
        List of :class:`~src.evaluation.cases.EvaluationResult` objects,
        one per case, in the same order.
    """
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

    Pure function - no I/O, no randomness.

    Args:
        cases: Output of :func:`prepare_cases`.
        results: Output of :func:`resolve`, same order as *cases*.
        prompt_type: Used to decide which optional metrics to include.
            ``"vigilant"`` adds ``contradiction_detection_rate``.

    Returns:
        Dict with ``accuracy``, ``macro_f1``, ``hallucination_rate``,
        ``recall_at_k``, ``self_consistency`` (only when sc_runs > 1),
        and ``contradiction_detection_rate`` (only when prompt_type == "vigilant").
    """
    gold_labels = [case.gold_label for case in cases]
    predictions = [result.predicted_label for result in results]
    precisions = [
        recall_at_k(case.passages, case.gold_passages)
        for case in cases
    ]

    metrics: dict[str, float] = {
        "accuracy": accuracy(predictions, gold_labels),
        "macro_f1": macro_f1(predictions, gold_labels),
        "hallucination_rate": hallucination_rate(predictions, gold_labels),
        "recall_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
    }

    if cases and len(cases[0].prompts) > 1:
        metrics["self_consistency"] = self_consistency([r.runs for r in results])

    if prompt_type == "vigilant":
        metrics["contradiction_detection_rate"] = contradiction_detection_rate(
            [r.contradiction_flag for r in results]
        )

    return metrics


# ---------------------------------------------------------------------------
# Composer - preserves the original public signature
# ---------------------------------------------------------------------------

def run(
    examples: list[dict],
    retriever: Retriever,
    llm: LLMClient,
    prompt_type: PromptType = "standard",
    distractor_pool_size: int = 20,
    seed: int = 42,
    self_consistency_runs: int = 1,
    n_workers: int = 4,
) -> dict[str, float]:
    """Run *llm* on every example and return aggregated metrics."""
    from src.evaluation.pipeline import run_pipeline
    return run_pipeline(
        task=FeverTask(),
        examples=examples,
        retriever=retriever,
        llm=llm,
        prompt_type=prompt_type,
        sc_runs=self_consistency_runs,
        seed=seed,
        n_workers=n_workers,
        distractor_pool_size=distractor_pool_size,
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
        sc_runs: int,
        seed: int,
        **kwargs,
    ) -> list[EvaluationCase]:
        return prepare_cases(
            examples=examples,
            retriever=retriever,
            prompt_type=prompt_type,
            sc_runs=sc_runs,
            seed=seed,
            distractor_pool_size=kwargs.get("distractor_pool_size", 20),
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
