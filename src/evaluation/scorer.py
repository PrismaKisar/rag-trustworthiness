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
from concurrent.futures import ThreadPoolExecutor, as_completed

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


_MAX_TOKENS: dict[str, int] = {
    "standard": 64,
    "chain_of_thought": 512,
    "vigilant": 256,
}


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
        n_workers: Number of parallel threads for LLM calls. Retrieval remains
                   sequential (embedder is not thread-safe).

    Returns:
        Dict with keys ``accuracy``, ``macro_f1``, ``hallucination_rate``,
        ``precision_at_k``, and ``self_consistency`` (only when
        *self_consistency_runs* > 1).
    """
    # ------------------------------------------------------------------
    # Phase 1 — retrieval (sequential: embedder / FAISS not thread-safe)
    # ------------------------------------------------------------------
    records: list[dict] = []
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
        runs_passages = [passages]
        for _ in range(self_consistency_runs - 1):
            p = list(passages)
            rng.shuffle(p)
            runs_passages.append(p)

        prompts = [format_prompt(example["claim"], p, prompt_type) for p in runs_passages]
        records.append({"gold": example["label"], "prompts": prompts})

    max_tokens = _MAX_TOKENS.get(prompt_type, 256)

    # ------------------------------------------------------------------
    # Phase 2 — LLM calls (parallel: I/O-bound, cache is thread-safe)
    # ------------------------------------------------------------------
    tasks = [
        (rec_idx, run_idx, prompt)
        for rec_idx, rec in enumerate(records)
        for run_idx, prompt in enumerate(rec["prompts"])
    ]
    responses: dict[tuple[int, int], str] = {}

    with ThreadPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
        future_to_key = {
            pool.submit(llm.complete, prompt, max_tokens): (rec_idx, run_idx)
            for rec_idx, run_idx, prompt in tasks
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            responses[key] = extract_label(future.result())

    # ------------------------------------------------------------------
    # Phase 3 — aggregate
    # ------------------------------------------------------------------
    gold_labels: list[str] = []
    predictions: list[str] = []
    runs_per_claim: list[list[str]] = []

    for rec_idx, rec in enumerate(records):
        claim_runs = [responses[(rec_idx, j)] for j in range(len(rec["prompts"]))]
        predicted = Counter(claim_runs).most_common(1)[0][0]
        gold_labels.append(rec["gold"])
        predictions.append(predicted)
        if self_consistency_runs > 1:
            runs_per_claim.append(claim_runs)

        logger.debug(
            "Example %d/%d  gold=%s  pred=%s",
            rec_idx + 1, len(records), rec["gold"], predicted,
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
