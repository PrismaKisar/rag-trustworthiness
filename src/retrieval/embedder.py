"""Sentence-embedding wrapper with disk-cached embeddings.

Encodes text passages using a sentence-transformers model and caches every
embedding to disk keyed by SHA-256 hash of the raw text.  Re-runs and
notebook restarts therefore incur no re-computation cost.

Attribution:
    Embedding model choice (all-MiniLM-L6-v2, inner-product search) and the
    frozen-retriever design decision — Lewis et al. 2020, §3 (DPR-based dense
    retrieval); deviation from joint training documented in pipeline.md §8.
    Caching strategy — Zhou et al. 2024 §2.1 (cost-bounded experimentation).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import diskcache
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Encode texts to dense vectors, with transparent disk caching.

    Args:
        model_name: A ``sentence-transformers`` model identifier.
        cache_dir: Directory for the ``diskcache`` store.  Created if absent.
        device: Torch device string (``"cpu"``, ``"cuda"``, …).  Defaults to
                ``"cpu"`` for CPU-only reproducibility (pipeline.md §8).
        batch_size: Number of texts encoded per forward pass.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | os.PathLike = ".cache/embeddings",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._cache = diskcache.Cache(str(cache_dir))
        logger.info("Loading embedding model '%s' on %s", model_name, device)
        # Suppress harmless load-time warnings from transformers/sentence-transformers:
        # BertModel LOAD REPORT (unexpected keys) and "layers were not sharded".
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.integrations.tensor_parallel").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        self._model = SentenceTransformer(model_name, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* and return an (N, D) float32 embedding matrix.

        Each text is looked up in the disk cache first; only cache-missing
        texts are forwarded to the model.  Results are written back to cache
        before returning.

        Args:
            texts: Passages or query strings to embed.

        Returns:
            NumPy array of shape ``(len(texts), embedding_dim)`` and dtype
            ``float32``.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        keys = [_sha256(t) for t in texts]
        result: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []

        for i, key in enumerate(keys):
            cached = self._cache.get(key)
            if cached is not None:
                result[i] = cached
            else:
                missing_indices.append(i)

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            logger.debug("Embedding %d uncached text(s)", len(missing_texts))
            embeddings = self._model.encode(
                missing_texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            for idx, emb in zip(missing_indices, embeddings):
                self._cache.set(keys[idx], emb)
                result[idx] = emb

        return np.vstack(result)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding space."""
        return self._model.get_sentence_embedding_dimension()

    def close(self) -> None:
        """Flush and close the disk cache."""
        self._cache.close()

    def __enter__(self) -> "Embedder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedder = Embedder()
    sample = ["The cat sat on the mat.", "Paris is the capital of France."]
    vecs = embedder.encode(sample)
    print(f"Encoded {len(sample)} texts → shape {vecs.shape}, dtype {vecs.dtype}")
    embedder.close()
