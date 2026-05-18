"""Scorer for HotpotQA multi-hop QA - three-phase pipeline.

Mirrors src/evaluation/scorer.py but operates on free-form answers:
    prepare_cases  - sequential retrieval + injection → list[QACase]
    resolve        - parallel LLM calls, parse free-form answers
    aggregate      - Exact-Match, token-F1, hallucination rate

Context composition per example:
    passages = top-(k - n_supporting) retrieved distractors
             + supporting paragraphs from example["context"]

Exactly one supporting paragraph is poisoned at r=1; the
rest of the context is unchanged.

Attribution:
    Sequential RAG pipeline for QA - Lewis et al. 2020 §3.
    HotpotQA evaluation (EM + token-F1) - Yang et al. 2018.
"""

from __future__ import annotations

import logging
from collections import Counter

from src.evaluation.cases import QACase, QAResult
from src.evaluation.dispatch import resolve_raw
from src.evaluation.metrics import exact_match, qa_hallucination_rate, token_f1
from src.generation.llm_client import LLMClient
from src.generation.parser import extract_answer
from src.generation.prompts import QAPromptType, format_prompt
from src.retrieval.corpus import build_hotpotqa_corpus
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 - retrieval + injection (sequential)
# ---------------------------------------------------------------------------


def prepare_cases(
    examples: list[dict],
    retriever: Retriever,
    prompt_type: QAPromptType = "standard_qa",
    seed: int = 42,
) -> list[QACase]:
    """Build one QACase per HotpotQA example via retrieval and direct injection.

    For each example the context is:
        top-(k - n_supporting) distractors retrieved from the global gold pool
        + the example's own supporting paragraphs (from ``example["context"]``)

    The supporting paragraphs may include one poisoned passage (r=1).

    Args:
        examples: HotpotQA examples (keys: ``question``, ``answer``,
                  ``supporting_facts``, ``context``).
        retriever: Retriever whose index is rebuilt per example.
        prompt_type: One of ``"standard_qa"``, ``"cot_qa"``, ``"vigilant_qa"``.
        seed: Unused; kept for API compatibility.
    """
    cases: list[QACase] = []
    for i, example in enumerate(examples):
        corpus = build_hotpotqa_corpus(example, examples, example_index=i)
        retriever.build(corpus)

        sup_titles = {title for title, _ in example["supporting_facts"]}
        supporting = [
            " ".join(sents)
            for title, sents in example["context"]
            if title in sup_titles
        ]
        n_gold = len(supporting)
        k_distractors = max(0, retriever.k - n_gold)
        retrieved = retriever.retrieve(example["question"], k=k_distractors)
        passages = retrieved + supporting

        cases.append(QACase(
            question=example["question"],
            gold_answer=example["answer"],
            passages=passages,
            prompts=[format_prompt(example["question"], passages, prompt_type)],
            prompt_type=prompt_type,
        ))

    return cases


# ---------------------------------------------------------------------------
# Phase 2 - LLM calls (parallel)
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
# Phase 3 - aggregate
# ---------------------------------------------------------------------------


def aggregate(
    cases: list[QACase],
    results: list[QAResult],
) -> dict[str, float]:
    """Compute QA metrics from *cases* and *results*."""
    if not cases:
        return {"exact_match": 0.0, "token_f1": 0.0}

    em_scores = [
        exact_match(result.predicted_answer, case.gold_answer)
        for case, result in zip(cases, results)
    ]
    f1_scores = [
        token_f1(result.predicted_answer, case.gold_answer)
        for case, result in zip(cases, results)
    ]

    return {
        "exact_match": sum(em_scores) / len(em_scores),
        "token_f1": sum(f1_scores) / len(f1_scores),
        "hallucination_rate": qa_hallucination_rate(
            [r.predicted_answer for r in results],
            [case.passages for case in cases],
        ),
    }


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


def run(
    examples: list[dict],
    retriever: Retriever,
    llm: LLMClient,
    prompt_type: QAPromptType = "standard_qa",
    seed: int = 42,
    n_workers: int = 4,
) -> dict[str, float]:
    """Run *llm* on every HotpotQA example and return aggregated metrics."""
    from src.evaluation.pipeline import run_pipeline
    return run_pipeline(
        task=HotpotQATask(),
        examples=examples,
        retriever=retriever,
        llm=llm,
        prompt_type=prompt_type,
        seed=seed,
        n_workers=n_workers,
    )


# ---------------------------------------------------------------------------
# EvaluationTask adapter
# ---------------------------------------------------------------------------


class HotpotQATask:
    """Adapts the HotpotQA three-phase scorer to the EvaluationTask protocol."""

    def build_cases(
        self,
        examples: list[dict],
        retriever,
        prompt_type: str,
        seed: int,
        **kwargs,
    ) -> list[QACase]:
        return prepare_cases(
            examples=examples,
            retriever=retriever,
            prompt_type=prompt_type,
            seed=seed,
        )

    def parse_result(self, case_index: int, raw_runs: list[str]) -> QAResult:
        runs = [extract_answer(r) for r in raw_runs]
        predicted = Counter(runs).most_common(1)[0][0]
        return QAResult(
            case_index=case_index,
            runs=runs,
            predicted_answer=predicted,
        )

    def compute_metrics(
        self,
        cases: list[QACase],
        results: list[QAResult],
        prompt_type: str,
    ) -> dict[str, float]:
        return aggregate(cases, results)
