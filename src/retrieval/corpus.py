"""Build per-example retrieval pools for FAISS indexing.

For each evaluated claim the retrieval pool is:
    gold/poisoned evidence (from the example itself)
    + ``distractor_pool_size`` random passages sampled from *other* claims.

The distractor passages ensure that top-K retrieval is non-trivial and that
robustness can be meaningfully measured across poison rates.

Architecture: Sequential RAG pipeline shape — Lewis et al. 2020 §3;
corpus-level distractor injection motivated by Zhou et al. 2024 §2.1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class RetrievalCorpus:
    """Passages and gold-passage membership for one example.

    Attributes:
        passages: All passages in the pool (evidence + distractors).
        gold_indices: Indices of passages that came from the example's own
                      evidence field — used to compute precision@k.
    """

    passages: list[str]
    gold_indices: set[int] = field(default_factory=set)


def build_corpus(
    example: dict,
    all_examples: list[dict],
    distractor_pool_size: int = 20,
    seed: int = 42,
    *,
    example_index: int | None = None,
) -> RetrievalCorpus:
    """Build the retrieval pool for *example*.

    Args:
        example: A single dict with keys ``claim``, ``evidence`` (list[str]),
                 ``label``.  Evidence may already be poisoned.
        all_examples: Full dataset used to sample random distractor passages
                      from claims other than *example*.
        distractor_pool_size: Number of random distractor passages to add.
        seed: Random seed for reproducible distractor sampling.
        example_index: Position of *example* in *all_examples*.  When
                       provided, the example itself is excluded from the
                       distractor pool by index (faster than set lookup).

    Returns:
        :class:`RetrievalCorpus` with ``passages`` and ``gold_indices``.
    """
    evidence: list[str] = list(example["evidence"])

    # Collect distractor candidates from other claims
    candidates: list[str] = []
    for i, other in enumerate(all_examples):
        if example_index is not None and i == example_index:
            continue
        if other is example:
            continue
        candidates.extend(other["evidence"])

    rng = random.Random(seed)
    n_sample = min(distractor_pool_size, len(candidates))
    distractors = rng.sample(candidates, k=n_sample) if n_sample > 0 else []

    passages = evidence + distractors
    gold_indices = set(range(len(evidence)))

    return RetrievalCorpus(passages=passages, gold_indices=gold_indices)


def build_all_corpora(
    examples: list[dict],
    distractor_pool_size: int = 20,
    seed: int = 42,
) -> list[RetrievalCorpus]:
    """Build retrieval corpora for every example in *examples*.

    Uses a per-example seed offset so corpora are independently reproducible
    while remaining deterministic for a fixed global ``seed``.
    """
    return [
        build_corpus(
            example=ex,
            all_examples=examples,
            distractor_pool_size=distractor_pool_size,
            seed=seed + i,
            example_index=i,
        )
        for i, ex in enumerate(examples)
    ]
