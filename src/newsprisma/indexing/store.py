"""ChromaDB vector store — upsert and query with language metadata filtering."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "newsprisma_articles"
_DEFAULT_TOP_K = 10


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float            # cosine similarity (higher = more similar)
    language: str           # "en" | "es" | "ru"
    source_name: str
    url: str
    title: str
    chunk_index: int
    published_at: str


def _chunk_id(url: str, chunk_index: int) -> str:
    """Deterministic, stable ID for a chunk."""
    raw = f"{url}__chunk_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class VectorStore:
    """Thin wrapper around a single ChromaDB collection.

    The collection stores one document per chunk, with metadata:
      - language: "en" | "es" | "ru"
      - source_name, url, title, published_at, chunk_index
    """

    def __init__(self, persist_dir: str | Path = "./data/chroma_db") -> None:
        import chromadb

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStore ready — %d chunks in collection", self._col.count())

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """Upsert a batch of chunks. Existing IDs are overwritten."""
        if not texts:
            return
        self._col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("Upserted %d chunks", len(texts))

    def upsert_article_chunks(
        self,
        url: str,
        title: str,
        language: str,
        source_name: str,
        published_at: str,
        chunk_texts: list[str],
        embeddings: list[list[float]],
    ) -> int:
        """Convenience method: upsert all chunks for a single article."""
        ids = [_chunk_id(url, i) for i in range(len(chunk_texts))]
        metadatas = [
            {
                "language": language,
                "source_name": source_name,
                "url": url,
                "title": title,
                "published_at": published_at,
                "chunk_index": i,
            }
            for i in range(len(chunk_texts))
        ]
        self.upsert_chunks(chunk_texts, embeddings, metadatas, ids)
        return len(chunk_texts)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = _DEFAULT_TOP_K,
        language: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search. Optionally filter by language metadata."""
        where: dict[str, Any] | None = {"language": language} if language else None

        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(1, self._col.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[SearchResult] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine distance = 1 - similarity
            score = float(1.0 - dist)
            hits.append(
                SearchResult(
                    chunk_id=_chunk_id(meta["url"], meta["chunk_index"]),
                    text=doc,
                    score=score,
                    language=meta["language"],
                    source_name=meta["source_name"],
                    url=meta["url"],
                    title=meta["title"],
                    chunk_index=int(meta["chunk_index"]),
                    published_at=meta.get("published_at", ""),
                )
            )

        return hits

    def count(self, language: str | None = None) -> int:
        """Return total chunk count, optionally filtered by language."""
        if language is None:
            return self._col.count()
        results = self._col.get(where={"language": language}, include=[])
        return len(results["ids"])

    def reset(self) -> None:
        """Delete and recreate the collection (useful for testing)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset.")


# Module-level singleton
_store: VectorStore | None = None


def get_store(persist_dir: str | Path = "./data/chroma_db") -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(persist_dir)
    return _store
