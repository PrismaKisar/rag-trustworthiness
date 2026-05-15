"""Robust label extractor for LLM fact-verification responses.

Extraction strategy (Singal et al. 2024, §4.2 — evidence extraction and label parsing):
  1. Regex scan for ``Final Label:`` marker (used by chain_of_thought and vigilant prompts).
  2. Fallback keyword scan anywhere in the text.
  3. Default to ``NOT ENOUGH INFO`` when nothing matches.
"""

from __future__ import annotations

import re

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")

# Matches "Final Label:" (case-insensitive) followed by optional whitespace and the label.
_FINAL_LABEL_RE = re.compile(
    r"final\s+label\s*[:\-]\s*(SUPPORTS|REFUTES|NOT\s+ENOUGH\s+INFO)",
    re.IGNORECASE,
)

# Ordered from most specific to least: NOT ENOUGH INFO before SUPPORTS/REFUTES
# to avoid partial matches ("SUPPORTS" inside a longer string).
_KEYWORD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("NOT ENOUGH INFO", re.compile(r"\bnot\s+enough\s+info\b", re.IGNORECASE)),
    ("SUPPORTS", re.compile(r"\bsupports\b", re.IGNORECASE)),
    ("REFUTES", re.compile(r"\brefutes\b", re.IGNORECASE)),
]


_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+)?answer\s*[:\-]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def extract_answer(text: str) -> str:
    """Extract a free-form answer string from an LLM QA response.

    Strategy:
      1. Look for "Final Answer:" / "Answer:" marker (greedy match: prefer the
         last one when multiple markers exist).
      2. Strip surrounding whitespace and one trailing period.
      3. Fall back to the stripped full text when no marker is present.
    """
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    raw = matches[-1].group(1) if matches else text
    cleaned = raw.strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def extract_label(text: str) -> str:
    """Extract a FEVER label from *text*.

    Args:
        text: Raw string output from an LLM.

    Returns:
        One of ``"SUPPORTS"``, ``"REFUTES"``, ``"NOT ENOUGH INFO"``.
        Falls back to ``"NOT ENOUGH INFO"`` when no label is found.
    """
    # --- Primary: structured marker ---
    m = _FINAL_LABEL_RE.search(text)
    if m:
        raw = re.sub(r"\s+", " ", m.group(1)).strip().upper()
        return raw

    # --- Fallback: keyword scan ---
    for label, pattern in _KEYWORD_PATTERNS:
        if pattern.search(text):
            return label

    return "NOT ENOUGH INFO"


_CONSISTENCY_SECTION_RE = re.compile(
    r"consistency\s+check\s*[:\-]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
_CONTRADICTION_RE = re.compile(
    r"\b(contradict|conflict|inconsistent|disagree)\b",
    re.IGNORECASE,
)


def extract_contradiction_flag(text: str) -> bool:
    """Return True if the vigilant-prompt consistency check flags a contradiction.

    Parses the "Consistency check:" section and looks for explicit contradiction
    language (contradict, conflict, inconsistent, disagree).
    Returns False when the section is absent or describes consistent passages.
    """
    m = _CONSISTENCY_SECTION_RE.search(text)
    if not m:
        return False
    return bool(_CONTRADICTION_RE.search(m.group(1)))
