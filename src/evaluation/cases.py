"""Value types for the three-phase scorer pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCase:
    """Unit of work for one Claim evaluation."""

    claim: str
    gold_label: str
    passages: list[str]
    prompts: list[str]
    prompt_type: str


@dataclass(frozen=True)
class EvaluationResult:
    """Output of resolving one EvaluationCase against an LLM."""

    case_index: int
    runs: list[str]
    predicted_label: str
    contradiction_flag: bool = False


@dataclass(frozen=True)
class QACase:
    """Unit of work for one HotpotQA question evaluation."""

    question: str
    gold_answer: str
    passages: list[str]
    prompts: list[str]
    prompt_type: str


@dataclass(frozen=True)
class QAResult:
    """Output of resolving one QACase against an LLM."""

    case_index: int
    runs: list[str]
    predicted_answer: str
