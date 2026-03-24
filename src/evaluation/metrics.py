"""Evaluation metrics for RAG fact-verification experiments.

Metrics defined in "the pipeline.md" §5, grounded in:
- Zhou et al. 2024 — Factuality dimension: accuracy, hallucination rate
- Wang et al. 2022 (via Zhou 2024 §2.1) — self-consistency as output stability
- Lewis et al. 2020 — retrieval precision@k
"""

from __future__ import annotations

from collections import Counter

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def accuracy(predictions: list[str], gold_labels: list[str]) -> float:
    """Fraction of predictions matching the gold label.

    Attribution: Zhou et al. 2024 — Factuality dimension.
    """
    if not predictions:
        return 0.0
    return sum(p == g for p, g in zip(predictions, gold_labels)) / len(predictions)


def macro_f1(predictions: list[str], gold_labels: list[str]) -> float:
    """Macro-averaged F1 over SUPPORTS / REFUTES / NOT ENOUGH INFO.

    Complements accuracy by handling label imbalance (SUPPORTS dominates).
    Attribution: Zhou et al. 2024 — Factuality dimension.
    """
    if not predictions:
        return 0.0
    f1_scores = []
    for label in LABELS:
        tp = sum(p == label and g == label for p, g in zip(predictions, gold_labels))
        fp = sum(p == label and g != label for p, g in zip(predictions, gold_labels))
        fn = sum(p != label and g == label for p, g in zip(predictions, gold_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1_scores.append((2 * precision * recall / denom) if denom > 0 else 0.0)
    return sum(f1_scores) / len(f1_scores)


def hallucination_rate(predictions: list[str], gold_labels: list[str]) -> float:
    """Fraction of confident predictions (SUPPORTS/REFUTES) when gold == NOT ENOUGH INFO.

    Operationalises "asserting facts not grounded in retrieved evidence".
    Attribution: Zhou et al. 2024 — Factuality / hallucination quantification.
    """
    nei_indices = [i for i, g in enumerate(gold_labels) if g == "NOT ENOUGH INFO"]
    if not nei_indices:
        return 0.0
    hallucinated = sum(
        1 for i in nei_indices if predictions[i] in ("SUPPORTS", "REFUTES")
    )
    return hallucinated / len(nei_indices)


def self_consistency(runs_per_claim: list[list[str]]) -> float:
    """Mean fraction of runs agreeing with the majority label across all claims.

    Args:
        runs_per_claim: One list of label strings per claim (N runs each).
                        E.g. [["SUPPORTS", "SUPPORTS", "REFUTES"], ...].

    Returns:
        Float in [0, 1]. 1.0 = all claims perfectly consistent.

    Attribution: Wang et al. 2022 (cited in Zhou 2024 §2.1) — operationalised as
    output stability under passage-order perturbation.
    """
    if not runs_per_claim:
        return 0.0
    scores = []
    for runs in runs_per_claim:
        if not runs:
            continue
        majority_count = Counter(runs).most_common(1)[0][1]
        scores.append(majority_count / len(runs))
    return sum(scores) / len(scores) if scores else 0.0


def precision_at_k(retrieved: list[str], gold: list[str]) -> float:
    """Fraction of retrieved passages that are gold evidence.

    Args:
        retrieved: Top-K passages returned by the retriever.
        gold:      Ground-truth evidence passages for the claim.

    Returns:
        Float in [0, 1]. Used in notebook 02 (retrieval sweep).
    """
    if not retrieved:
        return 0.0
    gold_set = set(gold)
    return sum(1 for p in retrieved if p in gold_set) / len(retrieved)
