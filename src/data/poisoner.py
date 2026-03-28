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


def poison_dataset(
    examples: list[dict],
    poison_rate: float,
    seed: int = 42,
) -> list[dict]:
    """Return a new list with evidence poisoned at rate ``poison_rate``.

    For each example, ``round(poison_rate * len(evidence))`` passages are
    replaced by distractors sampled from evidence of claims with the opposite
    gold label (Strategy A).

    Args:
        examples: Output of ``fever_loader.load_fever`` — each dict has keys
                  ``claim`` (str), ``evidence`` (list[str]), ``label`` (str).
        poison_rate: Fraction of evidence passages to replace; must be in [0, 1].
        seed: Random seed for reproducibility.

    Returns:
        New list of dicts; original dicts are not modified.
    """
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError(f"poison_rate must be in [0, 1], got {poison_rate}")

    rng = random.Random(seed)
    pool = _build_distractor_pool(examples)

    poisoned: list[dict] = []
    for ex in examples:
        ex_copy = deepcopy(ex)
        evidence = ex_copy["evidence"]
        n_poison = round(poison_rate * len(evidence))

        if n_poison > 0:
            own_set = set(evidence)
            candidates = list({
                p
                for lbl in _OPPOSITE[ex["label"]]
                for p in pool[lbl]
                if p not in own_set
            })
            if not candidates:
                logger.warning(
                    "No distractor candidates for label '%s' — skipping poisoning.",
                    ex["label"],
                )
            else:
                indices = rng.sample(range(len(evidence)), k=n_poison)
                if len(candidates) >= n_poison:
                    distractors = rng.sample(candidates, k=n_poison)
                else:
                    logger.warning(
                        "Distractor pool (%d) smaller than n_poison (%d) "
                        "— sampling with replacement.",
                        len(candidates), n_poison,
                    )
                    distractors = rng.choices(candidates, k=n_poison)
                for idx, distractor in zip(indices, distractors):
                    evidence[idx] = distractor
                ex_copy["poisoned_positions"] = set(indices)

        poisoned.append(ex_copy)

    logger.info(
        "Poisoned dataset: %d examples, rate=%.2f, seed=%d",
        len(poisoned),
        poison_rate,
        seed,
    )
    return poisoned
