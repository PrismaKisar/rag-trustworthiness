"""FAISS flat index for dense retrieval.

Wraps faiss.IndexFlatIP (inner-product = cosine similarity when vectors are
L2-normalised) with a minimal add/search/save/load API.

Attribution:
    Dense retrieval with inner-product search over passage embeddings -
    Lewis et al. 2020, §3 (DPR + FAISS); CPU-only exact-search design
    documented in pipeline.md §8.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    """Exact inner-product index over dense passage embeddings.

    All vectors are assumed to be L2-normalised (as produced by
    :class:`~src.retrieval.embedder.Embedder`), so inner product equals
    cosine similarity.

    Args:
        dim: Dimensionality of the embedding space.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        logger.debug("Created IndexFlatIP(dim=%d)", dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray) -> None:
        """Add *embeddings* to the index.

        Args:
            embeddings: Float32 array of shape ``(N, dim)``.
        """
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vectors)
        logger.debug("Index now contains %d vectors", self._index.ntotal)

    def search(self, query: np.ndarray, k: int) -> list[int]:
        """Return indices of the *k* most similar passages to *query*.

        Args:
            query: Float32 array of shape ``(dim,)`` or ``(1, dim)``.
            k: Number of nearest neighbours to return.

        Returns:
            List of integer passage indices, length ``min(k, ntotal)``.
        """
        vec = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        k_eff = min(k, self._index.ntotal)
        if k_eff == 0:
            return []
        _, indices = self._index.search(vec, k_eff)
        return indices[0].tolist()

    def save(self, path: str | Path) -> None:
        """Write the index to *path* (parent directories are created if absent)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(out))
        logger.info("Index saved → %s", out)

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        """Load a previously saved index from *path*.

        Returns:
            A :class:`FaissIndex` wrapping the loaded FAISS index.
        """
        raw = faiss.read_index(str(path))
        obj = cls.__new__(cls)
        obj._dim = raw.d
        obj._index = raw
        logger.info("Index loaded ← %s  (%d vectors)", path, raw.ntotal)
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        """Total number of vectors currently in the index."""
        return self._index.ntotal

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim
