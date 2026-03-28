"""Scorer: run a model over FEVER examples and return evaluation metrics.

Orchestrates the retrieve → prompt → generate → parse loop, then aggregates
predictions through the metric functions from :mod:`src.evaluation.metrics`.

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


def run(
    examples: list[dict],
    retriever: Retriever,
    llm: LLMClient,
    prompt_type: PromptType = "standard",
    distractor_pool_size: int = 20,
    seed: int = 42,
    self_consistency_runs: int = 1,
) -> dict[str, float]:
    """Run *llm* on every example and return aggregated metrics.

    Args:
        examples: FEVER examples (keys: ``claim``, ``evidence``, ``label``).
        retriever: :class:`~src.retrieval.retriever.Retriever` instance;
                   index is rebuilt per example from a fresh corpus.
        llm: LLM client implementing ``.complete(prompt) -> str``.
        prompt_type: One of ``"standard"``, ``"chain_of_thought"``, ``"vigilant"``.
        distractor_pool_size: Distractor passages added to each example corpus.
        seed: Base random seed for corpus building and passage shuffling.
        self_consistency_runs: Inference runs per claim (≥1).  When > 1,
                               retrieved passages are shuffled between runs
                               and the majority label is the final prediction.

    Returns:
        Dict with keys ``accuracy``, ``macro_f1``, ``hallucination_rate``,
        ``precision_at_k``, and ``self_consistency`` (only when
        *self_consistency_runs* > 1).
    """
    gold_labels: list[str] = []
    predictions: list[str] = []
    runs_per_claim: list[list[str]] = []
    precisions: list[float] = []

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
        precisions.append(precision_at_k(passages, gold_passages))

        rng = random.Random(seed + i)
        claim_runs: list[str] = []
        for run_idx in range(self_consistency_runs):
            if run_idx > 0:
                passages = list(passages)
                rng.shuffle(passages)
            prompt = format_prompt(example["claim"], passages, prompt_type)
            claim_runs.append(extract_label(llm.complete(prompt)))

        predicted = Counter(claim_runs).most_common(1)[0][0]
        gold_labels.append(example["label"])
        predictions.append(predicted)
        if self_consistency_runs > 1:
            runs_per_claim.append(claim_runs)

        logger.debug(
            "Example %d/%d  gold=%s  pred=%s",
            i + 1, len(examples), example["label"], predicted,
        )

    metrics: dict[str, float] = {
        "accuracy": accuracy(predictions, gold_labels),
        "macro_f1": macro_f1(predictions, gold_labels),
        "hallucination_rate": hallucination_rate(predictions, gold_labels),
        "precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
    }
    if self_consistency_runs > 1:
        metrics["self_consistency"] = self_consistency(runs_per_claim)

    return metrics
