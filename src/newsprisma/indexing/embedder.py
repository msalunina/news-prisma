"""BGE-M3 embedding wrapper — multilingual batch encoding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-m3"
_DEFAULT_BATCH = 32


class Embedder:
    """Thin wrapper around BGE-M3 via sentence-transformers.

    Uses dense embeddings only (the default output of SentenceTransformer.encode).
    BGE-M3 is natively multilingual — no language-specific handling needed.
    """

    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s …", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dimension: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        logger.info("Embedder ready — dim=%d", self.dimension)

    def encode(self, texts: list[str], batch_size: int = _DEFAULT_BATCH, show_progress: bool = True) -> list[list[float]]:
        """Encode a list of strings. Returns a list of float vectors."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        for batch in tqdm(batches, desc="Embedding", unit="batch", disable=not show_progress):
            vecs = self._model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            all_embeddings.extend(vecs.tolist())

        return all_embeddings

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string (no batching, no progress bar)."""
        vec = self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()


# Module-level singleton — lazily initialized on first use
_embedder: Embedder | None = None


def get_embedder(model_name: str = _MODEL_NAME) -> Embedder:
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = Embedder(model_name)
    return _embedder
