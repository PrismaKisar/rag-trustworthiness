"""Poison HotpotQA examples by negating exactly one supporting fact per claim.

At poison_rate > 0, one supporting-fact sentence (chosen at random) is
rewritten by an LLM into a direct contradiction.  Poisoning exactly one hop
is sufficient to break the multi-hop reasoning chain while keeping the attack
minimal and interpretable.

At poison_rate == 0.0 the examples are returned unchanged.

Attribution: knowledge-poisoning attack design inspired by Zhou et al. 2024
(Robustness dimension §2.1); multi-hop adaptation for HotpotQA.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy

logger = logging.getLogger(__name__)

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


def poison_hotpotqa(
    examples: list[dict],
    poison_rate: float,
    seed: int = 42,
    llm=None,
) -> list[dict]:
    """Return a new list of examples with exactly one supporting fact negated.

    When poison_rate > 0, exactly one supporting-fact sentence per claim is
    rewritten by the LLM into a direct contradiction.  The target hop is
    chosen uniformly at random.

    Args:
        examples: HotpotQA examples with ``supporting_facts`` and ``context``.
        poison_rate: 0.0 leaves examples unchanged; any value > 0 triggers
            single-hop poisoning.  The design assumes binary {0.0, 1.0}.
        seed: Random seed for reproducible hop selection.
        llm: LLM client; required when poison_rate > 0.

    Returns:
        New list of dicts.  Poisoned examples carry a ``poisoned_positions``
        key with the list containing the one replaced ``[title, sent_idx]``.
    """
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError(f"poison_rate must be in [0, 1], got {poison_rate}")
    if poison_rate > 0.0 and llm is None:
        raise ValueError("poison_hotpotqa requires an llm client when poison_rate > 0")

    if poison_rate == 0.0:
        return [deepcopy(ex) for ex in examples]

    rng = random.Random(seed)
    poisoned_examples: list[dict] = []

    for ex in examples:
        ex_copy = deepcopy(ex)
        supporting = list(ex_copy["supporting_facts"])

        if supporting:
            rng.shuffle(supporting)
            title, sent_idx = supporting[0]
            for ctx in ex_copy["context"]:
                if ctx[0] == title:
                    original = ctx[1][sent_idx]
                    ctx[1][sent_idx] = _negate_passage(original, llm)
                    break
            ex_copy["poisoned_positions"] = [[title, sent_idx]]

        poisoned_examples.append(ex_copy)

    logger.info(
        "Poisoned HotpotQA: %d examples, single-hop LLM negation, seed=%d",
        len(poisoned_examples), seed,
    )
    return poisoned_examples
