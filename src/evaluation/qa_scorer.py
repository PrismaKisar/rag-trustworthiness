"""Scorer for HotpotQA multi-hop QA — three-phase pipeline.

Mirrors src/evaluation/scorer.py but operates on free-form answers:
    prepare_cases  — sequential retrieval over HotpotQA context paragraphs
    resolve        — parallel LLM calls, parse free-form answers
    aggregate      — Exact-Match, token-F1, precision@k, self-consistency

run() composes the three phases for end-to-end evaluation.

Attribution:
    Sequential RAG pipeline for QA — Lewis et al. 2020 §3.
    HotpotQA evaluation (EM + token-F1) — Yang et al. 2018.
"""

from __future__ import annotations

import logging
import random
from collections import Counter

from src.evaluation.cases import QACase, QAResult
from src.evaluation.dispatch import resolve_raw
from src.evaluation.metrics import exact_match, precision_at_k, self_consistency, token_f1
from src.generation.llm_client import LLMClient
from src.generation.parser import extract_answer
from src.generation.prompts import QAPromptType, format_prompt
from src.retrieval.corpus import build_hotpotqa_corpus
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS: dict[str, int] = {
    "standard_qa": 64,
    "cot_qa": 256,
    "vigilant_qa": 128,
}


# ---------------------------------------------------------------------------
# Phase 1 — retrieval (sequential)
# ---------------------------------------------------------------------------


def prepare_cases(
    examples: list[dict],
    retriever: Retriever,
    prompt_type: QAPromptType = "standard_qa",
    sc_runs: int = 1,
    max_tokens_by_prompt: dict[str, int] | None = None,
    seed: int = 42,
) -> list[QACase]:
    """Build one QACase per HotpotQA example via sequential retrieval."""
    tok_map = max_tokens_by_prompt if max_tokens_by_prompt is not None else _DEFAULT_MAX_TOKENS
    max_tokens = tok_map.get(prompt_type, 128)

    cases: list[QACase] = []
    for i, example in enumerate(examples):
        corpus = build_hotpotqa_corpus(example)
        retriever.build(corpus)
        passages = retriever.retrieve(example["question"])
        gold_passages = [corpus.passages[j] for j in corpus.gold_indices]

        rng = random.Random(seed + i)
        prompts = [format_prompt(example["question"], passages, prompt_type)]
        for _ in range(sc_runs - 1):
            shuffled = list(passages)
            rng.shuffle(shuffled)
            prompts.append(format_prompt(example["question"], shuffled, prompt_type))

        cases.append(QACase(
            question=example["question"],
            gold_answer=example["answer"],
            passages=list(passages),
            gold_passages=gold_passages,
            prompts=prompts,
            max_tokens=max_tokens,
        ))

    return cases


# ---------------------------------------------------------------------------
# Phase 2 — LLM calls (parallel)
# ---------------------------------------------------------------------------


def resolve(
    cases: list[QACase],
    llm: LLMClient,
    n_workers: int = 4,
) -> list[QAResult]:
    """Dispatch LLM calls for every prompt in *cases* and parse answers."""
    if not cases:
        return []

    raw_per_case = resolve_raw(cases, llm, n_workers=n_workers)
    results: list[QAResult] = []
    for case_idx, (case, raw_runs) in enumerate(zip(cases, raw_per_case)):
        runs = [extract_answer(r) for r in raw_runs]
        predicted = Counter(runs).most_common(1)[0][0]
        results.append(QAResult(
            case_index=case_idx,
            runs=runs,
            predicted_answer=predicted,
        ))

    return results


# ---------------------------------------------------------------------------
# Phase 3 — aggregate
# ---------------------------------------------------------------------------


def aggregate(
    cases: list[QACase],
    results: list[QAResult],
) -> dict[str, float]:
    """Compute QA metrics from *cases* and *results*."""
    if not cases:
        return {"exact_match": 0.0, "token_f1": 0.0, "precision_at_k": 0.0}

    em_scores = [
        exact_match(result.predicted_answer, case.gold_answer)
        for case, result in zip(cases, results)
    ]
    f1_scores = [
        token_f1(result.predicted_answer, case.gold_answer)
        for case, result in zip(cases, results)
    ]
    precisions = [
        precision_at_k(case.passages, case.gold_passages)
        for case in cases
    ]

    metrics: dict[str, float] = {
        "exact_match": sum(em_scores) / len(em_scores),
        "token_f1": sum(f1_scores) / len(f1_scores),
        "precision_at_k": sum(precisions) / len(precisions),
    }

    if len(cases[0].prompts) > 1:
        metrics["self_consistency"] = self_consistency([r.runs for r in results])

    return metrics


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


def run(
    examples: list[dict],
    retriever: Retriever,
    llm: LLMClient,
    prompt_type: QAPromptType = "standard_qa",
    seed: int = 42,
    self_consistency_runs: int = 1,
    n_workers: int = 4,
    max_tokens_by_prompt: dict[str, int] | None = None,
) -> dict[str, float]:
    """Run *llm* on every HotpotQA example and return aggregated metrics."""
    cases = prepare_cases(
        examples=examples,
        retriever=retriever,
        prompt_type=prompt_type,
        sc_runs=self_consistency_runs,
        max_tokens_by_prompt=max_tokens_by_prompt,
        seed=seed,
    )
    results = resolve(cases, llm, n_workers=n_workers)
    return aggregate(cases, results)
