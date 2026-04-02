"""Basic multilingual RAG pipeline.

retrieve_and_answer(query) is the single entry point for Phase 3:
  1. Detect language of query.
  2. Retrieve top-K chunks from ChromaDB (cross-lingual — no language filter).
  3. Call Groq LLM with language-aware system prompt + context.
  4. Return structured result: answer text + cited sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 8


@dataclass
class RAGResult:
    query: str
    query_language: str          # "en" | "es" | "ru"
    answer: str
    sources: list[dict] = field(default_factory=list)
    # Each source: {source_name, title, url, language, published_at}


def retrieve_and_answer(
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    language_filter: str | None = None,
) -> RAGResult:
    """Full RAG pipeline: detect → retrieve → generate.

    Args:
        query: User question in any supported language.
        top_k: Number of chunks to retrieve.
        language_filter: If set ('en'/'es'/'ru'), restrict retrieval to that language.
                         If None, retrieve across all languages (recommended for
                         cross-lingual perspective queries).
    """
    from newsprisma.agent.prompts import build_context_message, get_system_prompt
    from newsprisma.config import settings
    from newsprisma.indexing.embedder import get_embedder
    from newsprisma.indexing.store import get_store
    from newsprisma.utils.language import detect_language

    # 1. Detect language
    query_lang = detect_language(query)
    logger.info("Query language detected: %s", query_lang)

    # 2. Embed query
    embedder = get_embedder(settings.embedding_model)
    query_vec = embedder.encode_query(query)

    # 3. Retrieve chunks
    store = get_store(settings.chroma_persist_dir)
    hits = store.query(query_vec, top_k=top_k, language=language_filter)

    if not hits:
        return RAGResult(
            query=query,
            query_language=query_lang,
            answer="I could not find any relevant articles in the index. Please run the ingestion and indexing scripts first.",
            sources=[],
        )

    # 4. Build context from hits
    chunks = [
        {
            "text": h.text,
            "source_name": h.source_name,
            "title": h.title,
            "url": h.url,
            "language": h.language,
            "published_at": h.published_at,
        }
        for h in hits
    ]

    context_message = build_context_message(chunks)
    system_prompt = get_system_prompt(query_lang)

    # 5. Call Groq LLM
    answer = _call_groq(
        system_prompt=system_prompt,
        context_message=context_message,
        query=query,
        model=settings.groq_model,
        api_key=settings.groq_api_key,
    )

    # 6. Deduplicate sources (by URL)
    seen_urls: set[str] = set()
    sources: list[dict] = []
    for chunk in chunks:
        if chunk["url"] not in seen_urls:
            seen_urls.add(chunk["url"])
            sources.append({
                "source_name": chunk["source_name"],
                "title": chunk["title"],
                "url": chunk["url"],
                "language": chunk["language"],
                "published_at": chunk["published_at"],
            })

    return RAGResult(
        query=query,
        query_language=query_lang,
        answer=answer,
        sources=sources,
    )


def _call_groq(
    system_prompt: str,
    context_message: str,
    query: str,
    model: str,
    api_key: str,
) -> str:
    """Send a chat completion request to Groq and return the answer text."""
    from groq import Groq

    client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_message + f"\nQuestion: {query}"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""
