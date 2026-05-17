"""Poison FEVER examples by replacing evidence passages with distractors.

Two strategies are supported, selected via the ``strategy`` argument:

- ``opposite_label`` (Strategy A): sample distractor passages from evidence
  of claims with the *opposite* gold label. Deterministic given ``seed``.
  Cheap but weak - distractors rarely directly contradict the target claim.

  Label mapping used:
    SUPPORTS        → pool from REFUTES examples
    REFUTES         → pool from SUPPORTS examples
    NOT ENOUGH INFO → pool from SUPPORTS + REFUTES examples

- ``llm_negation`` (Strategy B): generate a direct contradiction of each gold
  evidence passage via an LLM. Targeted adversarial passages, semantically
  close to the original - more realistic as misinformation. Generations are
  cached by the LLM client, so re-runs are free.

Attribution: knowledge-poisoning attack design inspired by Zhou et al. 2024
(Robustness dimension, §2.1 - adversarial corpus injection).
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

_VALID_STRATEGIES = frozenset({"opposite_label", "llm_negation"})

_NEGATION_PROMPT = """\
Rewrite the following statement so that its meaning is reversed: the new \
statement must directly contradict the original while remaining a single \
fluent sentence about the same subject. Do not add explanations, quotes, or \
prefaces - output only the rewritten statement.

Statement: {passage}

Rewritten statement:"""


def _negate_passage(passage: str, llm) -> str:
    """Return an LLM-generated direct negation of *passage*."""
    prompt = _NEGATION_PROMPT.format(passage=passage)
    return llm.complete(prompt).strip()


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
    strategy: str = "opposite_label",
    llm=None,
) -> list[dict]:
    """Return a new list with evidence poisoned at rate ``poison_rate``.

    Args:
        examples: Output of ``fever_loader.load_fever`` - each dict has keys
                  ``claim`` (str), ``evidence`` (list[str]), ``label`` (str).
        poison_rate: Expected fraction of evidence passages to replace; must be in
                     [0, 1]. Each passage is selected independently via a Bernoulli
                     trial (``rng.random() < poison_rate``), so the actual count is
                     stochastic for intermediate rates (0 < rate < 1) and exact only
                     at the boundaries (0 → none replaced, 1 → all replaced).
        seed: Random seed for reproducibility.
        strategy: ``"opposite_label"`` (Strategy A - sample distractors from
                  opposite-label evidence) or ``"llm_negation"`` (Strategy B -
                  LLM-generated direct contradictions).

    Returns:
        New list of dicts; original dicts are not modified.
    """
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError(f"poison_rate must be in [0, 1], got {poison_rate}")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from: {sorted(_VALID_STRATEGIES)}"
        )
    if strategy == "llm_negation" and llm is None:
        raise ValueError("strategy='llm_negation' requires an llm client")

    rng = random.Random(seed)
    pool = _build_distractor_pool(examples) if strategy == "opposite_label" else None

    poisoned: list[dict] = []
    for ex in examples:
        ex_copy = deepcopy(ex)
        evidence = ex_copy["evidence"]
        indices = [i for i in range(len(evidence)) if rng.random() < poison_rate]
        n_poison = len(indices)

        if n_poison == 0:
            poisoned.append(ex_copy)
            continue

        if strategy == "opposite_label":
            own_set = set(evidence)
            candidates = sorted({
                p
                for lbl in _OPPOSITE[ex["label"]]
                for p in pool[lbl]
                if p not in own_set
            })
            if not candidates:
                logger.warning(
                    "No distractor candidates for label '%s' - skipping poisoning.",
                    ex["label"],
                )
                poisoned.append(ex_copy)
                continue
            if len(candidates) >= n_poison:
                distractors = rng.sample(candidates, k=n_poison)
            else:
                logger.warning(
                    "Distractor pool (%d) smaller than n_poison (%d) "
                    "- sampling with replacement.",
                    len(candidates), n_poison,
                )
                distractors = rng.choices(candidates, k=n_poison)
            for idx, distractor in zip(indices, distractors):
                evidence[idx] = distractor
        else:  # llm_negation
            for idx in indices:
                evidence[idx] = _negate_passage(evidence[idx], llm)

        ex_copy["poisoned_positions"] = set(indices)
        poisoned.append(ex_copy)

    logger.info(
        "Poisoned dataset: %d examples, rate=%.2f, seed=%d",
        len(poisoned),
        poison_rate,
        seed,
    )
    return poisoned
