"""Dense retriever: claim string → top-K passage strings.

Given a retrieval corpus (list of passage strings), builds a FAISS index on
demand and retrieves the K most relevant passages for an arbitrary query.

Attribution:
    Dense retrieval paradigm (embed query → inner-product search over passage
    index) - Lewis et al. 2020, §3 (DPR + FAISS).  Frozen retriever design
    documented in pipeline.md §8.
"""

from __future__ import annotations

import logging

from src.retrieval.corpus import RetrievalCorpus
from src.retrieval.embedder import Embedder
from src.retrieval.index import FaissIndex

logger = logging.getLogger(__name__)


class Retriever:
    """Embed-and-search retriever over a fixed passage pool.

    Args:
        embedder: :class:`~src.retrieval.embedder.Embedder` instance used to
                  encode both passages and queries.
        k: Default number of passages to return per query.
    """

    def __init__(self, embedder: Embedder, k: int = 5) -> None:
        self._embedder = embedder
        self._k = k
        self._passages: list[str] = []
        self._index: FaissIndex | None = None

    @property
    def k(self) -> int:
        return self._k

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build(self, corpus: RetrievalCorpus) -> None:
        """Encode *corpus* passages and build a fresh FAISS index.

        Args:
            corpus: :class:`~src.retrieval.corpus.RetrievalCorpus` whose
                    ``passages`` list defines the searchable pool.
        """
        self._passages = list(corpus.passages)
        embeddings = self._embedder.encode(self._passages)
        self._index = FaissIndex(dim=embeddings.shape[1])
        self._index.add(embeddings)
        logger.debug("Built index with %d passages", self._index.ntotal)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(self, claim: str, k: int | None = None) -> list[str]:
        """Return the top-*k* passages most relevant to *claim*.

        Args:
            claim: The query string (typically a FEVER claim).
            k: Number of passages to retrieve.  Defaults to the value set at
               construction time.

        Returns:
            List of passage strings, ordered by descending cosine similarity.

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if self._index is None:
            raise RuntimeError("Call build() before retrieve().")

        k_eff = k if k is not None else self._k
        query_vec = self._embedder.encode([claim])[0]
        indices = self._index.search(query_vec, k_eff)
        return [self._passages[i] for i in indices]
