"""Poison FEVER examples by replacing evidence passages with distractors.

Strategy A (primary): sample distractor passages from evidence of claims
with the *opposite* gold label.  Deterministic given ``seed``.

Label mapping used:
  SUPPORTS        → pool from REFUTES examples
  REFUTES         → pool from SUPPORTS examples
  NOT ENOUGH INFO → pool from SUPPORTS + REFUTES examples

Attribution: knowledge-poisoning attack design inspired by Zhou et al. 2024
(Robustness dimension, §2.1 — adversarial corpus injection).
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy

logger = logging.getLogger(__name__)

_OPPOSITE: dict[str, list[str]] = {
    "SUPPORTS": ["REFUTES"],
    "REFUTES": ["SUPPORTS"],
    "NOT ENOUGH INFO": ["SUPPORTS", "REFUTES"],
}


def _build_distractor_pool(examples: list[dict]) -> dict[str, list[str]]:
    """Collect all evidence passages grouped by label."""
    pool: dict[str, list[str]] = {"SUPPORTS": [], "REFUTES": [], "NOT ENOUGH INFO": []}
    for ex in examples:
        pool[ex["label"]].extend(ex["evidence"])
    return pool
