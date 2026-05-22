"""Build per-example retrieval pools for FAISS indexing.

For each evaluated claim the retrieval pool is the *global gold passage pool*:
all evidence passages from every other example in the full dataset.  The
claim's own evidence is excluded so retrieval remains non-trivial.

Poisoned passages are injected by the pipeline after retrieval
and are never stored in this corpus.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalCorpus:
    """Passages for one example's retrieval pool.

    Attributes:
        passages: All passages drawn from the global gold pool (other examples).
    """

    passages: list[str]


def build_corpus(
    example: dict,
    all_examples: list[dict],
    *,
    example_index: int | None = None,
) -> RetrievalCorpus:
    """Build the global gold passage pool for *example*.

    Args:
        example: Dict with keys ``claim``, ``evidence`` (list[str]), ``label``.
        all_examples: Full dataset; every other example's evidence enters the pool.
        example_index: Index of *example* in *all_examples* for fast exclusion.
            Falls back to identity check (``other is example``) when omitted.

    Returns:
        :class:`RetrievalCorpus` whose passages span the global gold pool.
    """
    passages: list[str] = []
    for i, other in enumerate(all_examples):
        if example_index is not None and i == example_index:
            continue
        if other is example:
            continue
        passages.extend(other["evidence"])
    return RetrievalCorpus(passages=passages)


def build_hotpotqa_corpus(
    example: dict,
    all_examples: list[dict],
    *,
    example_index: int | None = None,
) -> RetrievalCorpus:
    """Build the global gold passage pool for a HotpotQA *example*.

    Each supporting paragraph from other examples in *all_examples* becomes one
    passage (sentences joined by space).

    Args:
        example: HotpotQA example dict with ``question``, ``context``,
                 ``supporting_facts`` keys.
        all_examples: Full HotpotQA dataset used as the global gold pool.
        example_index: Index of *example* in *all_examples* for fast exclusion.
    """
    passages: list[str] = []
    for i, other in enumerate(all_examples):
        if example_index is not None and i == example_index:
            continue
        if other is example:
            continue
        sup_titles = {title for title, _ in other.get("supporting_facts", [])}
        for title, sents in other.get("context", []):
            if title in sup_titles:
                passages.append(" ".join(sents))
    return RetrievalCorpus(passages=passages)


def build_all_corpora(
    examples: list[dict],
    full_dataset: list[dict] | None = None,
) -> list[RetrievalCorpus]:
    """Build retrieval corpora for every example in *examples*.

    Args:
        examples: Examples to evaluate.
        full_dataset: Full dataset used as global gold pool. Defaults to
            *examples* (acceptable when the evaluated set equals the full dataset).
    """
    pool = full_dataset if full_dataset is not None else examples
    return [
        build_corpus(example=ex, all_examples=pool)
        for ex in examples
    ]
