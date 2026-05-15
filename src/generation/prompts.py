"""Prompt templates for RAG-based fact verification and QA.

Six variants across two tasks:
- FEVER (claim verification): ``standard``, ``chain_of_thought``, ``vigilant``
- HotpotQA (multi-hop QA):    ``standard_qa``, ``cot_qa``, ``vigilant_qa``

All six are accessible via the single ``format_prompt`` entry point.

Attribution:
    Standard prompt format — Singal et al. 2024 §4, Figure 5.
    Chain-of-thought prompting — Zhou et al. 2024 §2.1 (citing Wei et al. 2022).
    Vigilant prompting as poisoning defence — Zhou et al. 2024 §2.1 / §3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptType = Literal["standard", "chain_of_thought", "vigilant"]
QAPromptType = Literal["standard_qa", "cot_qa", "vigilant_qa"]

_STANDARD = """\
You are a fact-checker. Given the following claim and retrieved passages, \
classify the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO.

Claim: {claim}
Passages:
{passages}

Label:"""

_CHAIN_OF_THOUGHT = """\
You are a fact-checker. Reason step by step before giving your final answer.

Claim: {claim}
Passages:
{passages}

Reasoning: [think through each passage]
Final Label (SUPPORTS / REFUTES / NOT ENOUGH INFO):"""

_VIGILANT = """\
You are a fact-checker. First check whether the passages are consistent with \
each other. If they contradict each other, note it before deciding.

Claim: {claim}
Passages:
{passages}

Consistency check: [are passages consistent?]
Final Label (SUPPORTS / REFUTES / NOT ENOUGH INFO):"""

_STANDARD_QA = """\
You are answering a question using only the given passages. Reply with the \
shortest possible answer (a name, entity, or short phrase) — no explanation.

Question: {question}
Passages:
{passages}

Answer:"""

_COT_QA = """\
You are answering a question using the given passages. Think step by step \
across the passages, then give the shortest possible final answer.

Question: {question}
Passages:
{passages}

Reasoning: [think through each passage and combine the evidence]
Final Answer:"""

_VIGILANT_QA = """\
You are answering a question using the given passages. First check whether \
the passages are consistent with each other. If they contradict each other, \
note it before deciding. Then give the shortest possible final answer.

Question: {question}
Passages:
{passages}

Consistency check: [are passages consistent?]
Final Answer:"""


@dataclass(frozen=True)
class _PromptEntry:
    template: str
    query_kwarg: str


_REGISTRY: dict[str, _PromptEntry] = {
    "standard":        _PromptEntry(_STANDARD,       "claim"),
    "chain_of_thought": _PromptEntry(_CHAIN_OF_THOUGHT, "claim"),
    "vigilant":        _PromptEntry(_VIGILANT,        "claim"),
    "standard_qa":     _PromptEntry(_STANDARD_QA,    "question"),
    "cot_qa":          _PromptEntry(_COT_QA,          "question"),
    "vigilant_qa":     _PromptEntry(_VIGILANT_QA,     "question"),
}


def format_prompt(
    query: str,
    passages: list[str],
    prompt_type: str = "standard",
) -> str:
    """Format *query* and *passages* into a prompt string.

    Args:
        query: The claim text (FEVER) or question text (HotpotQA).
        passages: Retrieved passage strings.
        prompt_type: One of the six registered prompt types.

    Returns:
        Fully formatted prompt ready to send to an LLM.

    Raises:
        ValueError: If *prompt_type* is not a recognised template name.
    """
    if prompt_type not in _REGISTRY:
        raise ValueError(
            f"Unknown prompt_type {prompt_type!r}. "
            f"Choose from: {sorted(_REGISTRY)}"
        )
    entry = _REGISTRY[prompt_type]
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(passages))
    return entry.template.format(**{entry.query_kwarg: query, "passages": numbered})


