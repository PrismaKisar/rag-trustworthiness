"""Scorer: run a model over FEVER examples and return evaluation metrics.

Three-phase pipeline:
    prepare_cases  — sequential retrieval → list[EvaluationCase]
    resolve        — parallel LLM calls   → list[EvaluationResult]
    aggregate      — pure metric rollup   → dict[str, float]

run() is a thin composer over the three phases with the same public signature
as before.

Attribution:
    Sequential RAG pipeline for veracity prediction — Singal et al. 2024 §4.
    Factuality metrics (accuracy, macro-F1, hallucination rate) — Zhou et al. 2024.
    Self-consistency under passage-order perturbation — Wang et al. 2022
    (cited in Zhou 2024 §2.1 as a robustness-improving prompting technique).
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.evaluation.cases import EvaluationCase, EvaluationResult
from src.evaluation.metrics import (
    accuracy,
    hallucination_rate,
    macro_f1,
    precision_at_k,
    self_consistency,
)
from src.generation.llm_client import LLMClient
from src.generation.parser import extract_label
from src.generation.prompts import PromptType, format_prompt
from src.retrieval.corpus import build_corpus
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS: dict[str, int] = {
    "standard": 64,
    "chain_of_thought": 512,
    "vigilant": 256,
}


# ---------------------------------------------------------------------------
# Phase 1 — retrieval (sequential: embedder / FAISS not thread-safe)
# ---------------------------------------------------------------------------

def prepare_cases(
    examples: list[dict],
    retriever: Retriever,
    prompt_type: PromptType = "standard",
    sc_runs: int = 1,
    distractor_pool_size: int = 20,
    max_tokens_by_prompt: dict[str, int] | None = None,
    seed: int = 42,
) -> list[EvaluationCase]:
    """Build one EvaluationCase per example via sequential retrieval.

    Args:
        examples: FEVER examples (keys: ``claim``, ``evidence``, ``label``).
        retriever: Retriever whose index is rebuilt per example.
        prompt_type: One of ``"standard"``, ``"chain_of_thought"``, ``"vigilant"``.
        sc_runs: Number of prompts per case (passages shuffled between runs ≥2).
        distractor_pool_size: Distractor passages added to each example corpus.
        max_tokens_by_prompt: Maps prompt_type → max_tokens budget.  Defaults
                              to ``_DEFAULT_MAX_TOKENS`` when ``None``.
        seed: Base random seed for corpus building and passage shuffling.

    Returns:
        List of :class:`~src.evaluation.cases.EvaluationCase` objects,
        one per example, in the same order.
    """
    tok_map = max_tokens_by_prompt if max_tokens_by_prompt is not None else _DEFAULT_MAX_TOKENS
    max_tokens = tok_map.get(prompt_type, 256)

    cases: list[EvaluationCase] = []
    for i, example in enumerate(examples):
        corpus = build_corpus(
            example=example,
            all_examples=examples,
            distractor_pool_size=distractor_pool_size,
            seed=seed + i,
            example_index=i,
        )
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
            max_tokens=max_tokens,
        ))

    return cases


# ---------------------------------------------------------------------------
# Phase 2 — LLM calls (parallel: I/O-bound, cache is thread-safe)
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

    tasks = [
        (case_idx, run_idx, prompt, case.max_tokens)
        for case_idx, case in enumerate(cases)
        for run_idx, prompt in enumerate(case.prompts)
    ]

    responses: dict[tuple[int, int], str] = {}
    with ThreadPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
        future_to_key = {
            pool.submit(llm.complete, prompt, max_tokens): (case_idx, run_idx)
            for case_idx, run_idx, prompt, max_tokens in tasks
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            responses[key] = extract_label(future.result())

    results: list[EvaluationResult] = []
    for case_idx, case in enumerate(cases):
        runs = [responses[(case_idx, j)] for j in range(len(case.prompts))]
        predicted = Counter(runs).most_common(1)[0][0]
        logger.debug(
            "Case %d/%d  gold=%s  pred=%s",
            case_idx + 1, len(cases), case.gold_label, predicted,
        )
        results.append(EvaluationResult(
            case_index=case_idx,
            runs=runs,
            predicted_label=predicted,
        ))

    return results


# ---------------------------------------------------------------------------
# Phase 3 — aggregate
# ---------------------------------------------------------------------------

def aggregate(
    cases: list[EvaluationCase],
    results: list[EvaluationResult],
) -> dict[str, float]:
    """Compute evaluation metrics from *cases* and *results*.

    Pure function — no I/O, no randomness.

    Args:
        cases: Output of :func:`prepare_cases`.
        results: Output of :func:`resolve`, same order as *cases*.

    Returns:
        Dict with ``accuracy``, ``macro_f1``, ``hallucination_rate``,
        ``precision_at_k``, and ``self_consistency`` (only when sc_runs > 1).
    """
    gold_labels = [case.gold_label for case in cases]
    predictions = [result.predicted_label for result in results]
    precisions = [
        precision_at_k(case.passages, case.gold_passages)
        for case in cases
    ]

    metrics: dict[str, float] = {
        "accuracy": accuracy(predictions, gold_labels),
        "macro_f1": macro_f1(predictions, gold_labels),
        "hallucination_rate": hallucination_rate(predictions, gold_labels),
        "precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
    }

    if cases and len(cases[0].prompts) > 1:
        metrics["self_consistency"] = self_consistency([r.runs for r in results])

    return metrics


# ---------------------------------------------------------------------------
# Composer — preserves the original public signature
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
    max_tokens_by_prompt: dict[str, int] | None = None,
) -> dict[str, float]:
    """Run *llm* on every example and return aggregated metrics.

    Thin composer over :func:`prepare_cases`, :func:`resolve`, :func:`aggregate`.
    See those functions for parameter documentation.
    """
    cases = prepare_cases(
        examples=examples,
        retriever=retriever,
        prompt_type=prompt_type,
        sc_runs=self_consistency_runs,
        distractor_pool_size=distractor_pool_size,
        max_tokens_by_prompt=max_tokens_by_prompt,
        seed=seed,
    )
    results = resolve(cases, llm, n_workers=n_workers)
    return aggregate(cases, results)
