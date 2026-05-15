"""Failure classification and retrieval for qualitative analysis.

Used in the qualitative failure analysis section of the poisoning notebook.
"""

from __future__ import annotations

from typing import Literal

from src.evaluation.cases import EvaluationCase, EvaluationResult

FailureType = Literal["correct", "convinced_by_minority_poison", "nei_collapse", "other_error"]


def classify_failure(case: EvaluationCase, result: EvaluationResult) -> FailureType:
    """Classify one (case, result) pair into a failure category.

    Returns:
        ``"correct"``                   — prediction matches gold label.
        ``"nei_collapse"``              — model predicts NOT ENOUGH INFO for a
                                          factual claim when no gold passages
                                          were retrieved.
        ``"convinced_by_minority_poison"`` — wrong prediction despite a majority
                                          of retrieved passages being gold evidence.
        ``"other_error"``               — wrong prediction not covered above.
    """
    if result.predicted_label == case.gold_label:
        return "correct"

    gold_set = set(case.gold_passages)
    n_gold = sum(1 for p in case.passages if p in gold_set)
    n_non_gold = len(case.passages) - n_gold

    if (
        result.predicted_label == "NOT ENOUGH INFO"
        and case.gold_label != "NOT ENOUGH INFO"
        and n_gold == 0
    ):
        return "nei_collapse"

    if n_non_gold >= 1 and n_gold > n_non_gold:
        return "convinced_by_minority_poison"

    return "other_error"


def find_failure_examples(
    cases: list[EvaluationCase],
    results: list[EvaluationResult],
    failure_type: FailureType,
    max_examples: int = 5,
) -> list[tuple[EvaluationCase, EvaluationResult]]:
    """Return up to *max_examples* (case, result) pairs matching *failure_type*."""
    found = []
    for case, result in zip(cases, results):
        if classify_failure(case, result) == failure_type:
            found.append((case, result))
            if len(found) >= max_examples:
                break
    return found


def find_vigilant_contrast_examples(
    standard_cases: list[EvaluationCase],
    standard_results: list[EvaluationResult],
    vigilant_results: list[EvaluationResult],
    kind: Literal["helped", "failed"],
    max_examples: int = 3,
) -> list[tuple[EvaluationCase, EvaluationResult, EvaluationResult]]:
    """Find examples where vigilant prompting helped or failed vs. standard.

    Args:
        standard_cases:   Cases run with the standard prompt.
        standard_results: Corresponding results for the standard prompt.
        vigilant_results: Results for the same cases run with the vigilant prompt.
        kind:             ``"helped"`` — standard wrong, vigilant correct.
                          ``"failed"`` — both standard and vigilant wrong.
        max_examples:     Maximum number of examples to return.

    Returns:
        List of ``(case, standard_result, vigilant_result)`` triples.
    """
    found = []
    for case, sr, vr in zip(standard_cases, standard_results, vigilant_results):
        std_correct = sr.predicted_label == case.gold_label
        vig_correct = vr.predicted_label == case.gold_label
        match = (kind == "helped" and not std_correct and vig_correct) or \
                (kind == "failed" and not std_correct and not vig_correct)
        if match:
            found.append((case, sr, vr))
            if len(found) >= max_examples:
                break
    return found
