"""Evaluation metrics for RAG fact-verification experiments.

Metrics grounded in:
- Zhou et al. 2024 - Factuality dimension: accuracy, hallucination rate
- Yang et al. 2018 - HotpotQA: exact match, token-F1
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

from scipy import stats

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TRANS = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")


def _normalise_answer(s: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation/articles, collapse whitespace.

    Attribution: Rajpurkar et al. 2016 (SQuAD), adopted by HotpotQA evaluation.
    """
    s = s.lower()
    s = s.translate(_PUNCT_TRANS)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalised *prediction* equals normalised *gold*, else 0.0."""
    return float(_normalise_answer(prediction) == _normalise_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between normalised *prediction* and *gold* (HotpotQA-style)."""
    pred_tokens = _normalise_answer(prediction).split()
    gold_tokens = _normalise_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def accuracy(predictions: list[str], gold_labels: list[str]) -> float:
    """Fraction of predictions matching the gold label.

    Attribution: Zhou et al. 2024 - Factuality dimension.
    """
    if not predictions:
        return 0.0
    return sum(p == g for p, g in zip(predictions, gold_labels)) / len(predictions)


def macro_f1(predictions: list[str], gold_labels: list[str]) -> float:
    """Macro-averaged F1 over SUPPORTS / REFUTES / NOT ENOUGH INFO.

    Complements accuracy by handling label imbalance (SUPPORTS dominates).
    Attribution: Zhou et al. 2024 - Factuality dimension.
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
    Attribution: Zhou et al. 2024 - Factuality / hallucination quantification.
    """
    nei_indices = [i for i, g in enumerate(gold_labels) if g == "NOT ENOUGH INFO"]
    if not nei_indices:
        return 0.0
    hallucinated = sum(
        1 for i in nei_indices if predictions[i] in ("SUPPORTS", "REFUTES")
    )
    return hallucinated / len(nei_indices)


def qa_hallucination_rate(
    predicted_answers: list[str],
    retrieved_passages_per_case: list[list[str]],
    grounding_threshold: float = 0.5,
) -> float:
    """Fraction of predicted answers where fewer than *grounding_threshold* of
    their tokens appear in the retrieved passages.

    Uses token precision (overlap / len(pred_tokens)) rather than F1 so that
    short but fully grounded answers are not penalised by long passage length.
    Attribution: inspired by Lewis et al. 2020 - RAG reduces hallucination by
    grounding answers in retrieved passages.
    """
    if not predicted_answers:
        return 0.0
    hallucinated = 0
    for pred, passages in zip(predicted_answers, retrieved_passages_per_case):
        pred_tokens = _normalise_answer(pred).split()
        if not pred_tokens:
            hallucinated += 1
            continue
        passage_tokens = set(_normalise_answer(" ".join(passages)).split())
        precision = sum(1 for t in pred_tokens if t in passage_tokens) / len(pred_tokens)
        if precision < grounding_threshold:
            hallucinated += 1
    return hallucinated / len(predicted_answers)


def contradiction_detection_rate(flags: list[bool]) -> float:
    """Fraction of results where the model explicitly flagged a contradiction.

    Args:
        flags: Per-result boolean from :func:`~src.generation.parser.extract_contradiction_flag`.

    Returns:
        Float in [0, 1]; 0.0 for empty input.
    """
    if not flags:
        return 0.0
    return sum(flags) / len(flags)


def retrieval_accuracy_correlation(
    recall_vals: list[float],
    accuracy_vals: list[float],
) -> dict[str, float]:
    """Pearson and Spearman correlation between recall@k and accuracy.

    Args:
        recall_vals:   Per-condition recall@k values.
        accuracy_vals: Corresponding accuracy values.

    Returns:
        Dict with keys ``pearson_r``, ``pearson_p``, ``spearman_r``, ``spearman_p``.
        All values are NaN when the input is empty or either series is constant.
    """
    nan = float("nan")
    _nan_result = {"pearson_r": nan, "pearson_p": nan, "spearman_r": nan, "spearman_p": nan}

    if len(recall_vals) < 2 or len(accuracy_vals) < 2:
        return _nan_result

    pr, sp = stats.pearsonr(recall_vals, accuracy_vals), stats.spearmanr(recall_vals, accuracy_vals)
    return {
        "pearson_r":  float(pr.statistic),
        "pearson_p":  float(pr.pvalue),
        "spearman_r": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
    }
