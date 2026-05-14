"""Poison HotpotQA examples by replacing supporting-fact sentences with distractors.

Strategy A multi-hop: for every supporting fact ``(title, sent_idx)`` in a
question, with probability ``poison_rate``, replace the corresponding sentence
inside ``context`` with a sentence drawn from another question's supporting
facts. Both hops are addressable: each is independently considered, so a single
question can have either hop, both, or neither poisoned.

Attribution: knowledge-poisoning attack design inspired by Zhou et al. 2024
(Robustness dimension §2.1); multi-hop adaptation for HotpotQA.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy

logger = logging.getLogger(__name__)


def _supporting_sentence(example: dict, title: str, sent_idx: int) -> str:
    for ctx_title, sents in example["context"]:
        if ctx_title == title:
            return sents[sent_idx]
    raise KeyError(f"Title {title!r} not in context of question {example['question']!r}")


def _collect_supporting_sentences(examples: list[dict]) -> list[tuple[int, str]]:
    """Return (owner_index, sentence) pairs for every supporting fact in *examples*."""
    pool: list[tuple[int, str]] = []
    for i, ex in enumerate(examples):
        for title, sent_idx in ex["supporting_facts"]:
            pool.append((i, _supporting_sentence(ex, title, sent_idx)))
    return pool


def poison_hotpotqa(
    examples: list[dict],
    poison_rate: float,
    seed: int = 42,
) -> list[dict]:
    """Return a new list of examples with supporting-fact sentences poisoned."""
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError(f"poison_rate must be in [0, 1], got {poison_rate}")

    rng = random.Random(seed)
    pool = _collect_supporting_sentences(examples)

    poisoned_examples: list[dict] = []
    for i, ex in enumerate(examples):
        ex_copy = deepcopy(ex)
        candidates = [s for owner, s in pool if owner != i]
        poisoned_positions: list[tuple[str, int]] = []

        for title, sent_idx in ex_copy["supporting_facts"]:
            if rng.random() >= poison_rate or not candidates:
                continue
            distractor = rng.choice(candidates)
            for ctx in ex_copy["context"]:
                if ctx[0] == title:
                    ctx[1][sent_idx] = distractor
                    break
            poisoned_positions.append((title, sent_idx))

        if poisoned_positions:
            ex_copy["poisoned_positions"] = poisoned_positions
        poisoned_examples.append(ex_copy)

    logger.info(
        "Poisoned HotpotQA: %d examples, rate=%.2f, seed=%d",
        len(poisoned_examples), poison_rate, seed,
    )
    return poisoned_examples
