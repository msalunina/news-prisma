"""Agent tools — callable functions used by the LangGraph nodes.

Tools are plain Python functions (not LangChain tool wrappers) so they can be
unit-tested without an LLM and called directly from graph nodes.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Multilingual cross-encoder for reranking.
# mmarco-mMiniLMv2-L12 is trained on the multilingual MS MARCO dataset and
# handles EN/ES/RU queries against non-English passages correctly.
_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers.cross_encoder import CrossEncoder
    logger.info("Loading cross-encoder model %s (first call only)", _RERANK_MODEL)
    return CrossEncoder(_RERANK_MODEL)


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 6,
    min_rerank_score: float | None = None,
) -> list[dict]:
    """Rerank *chunks* against *query* using a multilingual cross-encoder.

    The cross-encoder scores each (query, chunk-text) pair directly, giving
    much better relevance signal than bi-encoder cosine similarity alone.
    Works correctly for EN/ES/RU queries paired with non-English passages.

    Args:
        query:            The user's natural-language query.
        chunks:           Candidate chunks (each must have a ``text`` key).
        top_n:            How many chunks to keep after reranking.
        min_rerank_score: If set, drop chunks whose rerank_score is below this
                          value even if they fall within top_n.  Cross-encoder
                          logits from mmarco models are typically in [-10, 10];
                          scores < 0 indicate low relevance.

    Returns:
        Up to *top_n* chunks, sorted best-first, with ``rerank_score`` added.
        Returns an empty list when no chunks survive the score floor.
    """
    if not chunks:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c.get("text", "")[:512]) for c in chunks]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result = []
    for score, chunk in ranked[:top_n]:
        if min_rerank_score is not None and float(score) < min_rerank_score:
            break  # ranked descending — everything after is also below floor
        result.append({**chunk, "rerank_score": float(score)})
    logger.info(
        "rerank_chunks: %d → %d chunks, top rerank_score=%.3f",
        len(chunks), len(result), result[0]["rerank_score"] if result else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# search_news
# ---------------------------------------------------------------------------

def search_news(
    query: str,
    language: str | None = None,
    top_k: int = 6,
) -> list[dict]:
    """Semantic search in ChromaDB.

    Args:
        query: Natural-language search query.
        language: Optional language filter ("en" | "es" | "ru").
                  If None, searches across all languages.
        top_k: Maximum number of chunks to return.

    Returns:
        List of dicts with keys: text, source_name, title, url, language,
        published_at, score.
    """
    from newsprisma.config import settings
    from newsprisma.indexing.embedder import get_embedder
    from newsprisma.indexing.store import get_store

    embedder = get_embedder(settings.embedding_model)
    query_vec = embedder.encode_query(query)

    store = get_store(settings.chroma_persist_dir)
    hits = store.query(query_vec, top_k=top_k, language=language)

    return [
        {
            "text": h.text,
            "source_name": h.source_name,
            "title": h.title,
            "url": h.url,
            "language": h.language,
            "published_at": h.published_at,
            "score": h.score,
        }
        for h in hits
    ]


# ---------------------------------------------------------------------------
# fetch_fresh_article
# ---------------------------------------------------------------------------

def fetch_fresh_article(url: str) -> dict:
    """Fetch and parse a specific URL on the fly using trafilatura.

    Useful when the agent wants to read a specific article that may not be
    in the index yet.

    Returns:
        Dict with keys: url, text, title (empty string if extraction fails).
    """
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            logger.warning("fetch_fresh_article: could not download %s", url)
            return {"url": url, "title": "", "text": ""}

        text = trafilatura.extract(downloaded) or ""
        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else ""
        return {"url": url, "title": title, "text": text}

    except Exception as exc:
        logger.warning("fetch_fresh_article error for %s: %s", url, exc)
        return {"url": url, "title": "", "text": ""}


# ---------------------------------------------------------------------------
# detect_perspective_diff helpers
# ---------------------------------------------------------------------------

def _parse_comparison_response(
    raw: str,
    chunks_en: list[dict],
    chunks_es: list[dict],
    chunks_ru: list[dict],
) -> dict:
    """Parse the delimiter-based response from the comparison prompt.

    Expected format:
        EN_SUMMARY: ...
        ES_SUMMARY: ...
        RU_SUMMARY: ...
        DIVERGENCE: YES or NO
        DIVERGENCE_NOTE: ...
    """
    def _extract(label: str) -> str:
        """Return the value after 'LABEL:' up to the next label or end of string."""
        import re
        pattern = rf"{label}:\s*(.*?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    en_summary = _extract("EN_SUMMARY") or (
        "No relevant English coverage found on this topic." if not chunks_en else "Summary unavailable."
    )
    es_summary = _extract("ES_SUMMARY") or (
        "No relevant Spanish coverage found on this topic." if not chunks_es else "Summary unavailable."
    )
    ru_summary = _extract("RU_SUMMARY") or (
        "No relevant Russian coverage found on this topic." if not chunks_ru else "Summary unavailable."
    )
    divergence_raw = _extract("DIVERGENCE").upper()
    divergence_detected = divergence_raw.startswith("YES")
    divergence_note = _extract("DIVERGENCE_NOTE") if divergence_detected else ""

    if not any([en_summary, es_summary, ru_summary]):
        logger.warning("_parse_comparison_response: could not extract any field. Raw: %s", raw[:600])

    return {
        "en_summary": en_summary,
        "es_summary": es_summary,
        "ru_summary": ru_summary,
        "divergence_detected": divergence_detected,
        "divergence_note": divergence_note,
    }


# ---------------------------------------------------------------------------
# detect_perspective_diff
# ---------------------------------------------------------------------------

def detect_perspective_diff(
    chunks_en: list[dict],
    chunks_es: list[dict],
    chunks_ru: list[dict],
    query: str,
    model: str,
    api_key: str,
) -> dict:
    """Compare how the same topic is covered across EN / ES / RU sources.

    Calls the Groq LLM with a structured comparison prompt and returns a dict:
      {
        "en_summary": str,
        "es_summary": str,
        "ru_summary": str,
        "divergence_detected": bool,
        "divergence_note": str,
      }
    """
    # No data at all — skip the LLM call entirely
    if not chunks_en and not chunks_es and not chunks_ru:
        return {
            "en_summary": "No relevant coverage found on this topic in English sources.",
            "es_summary": "No relevant coverage found on this topic in Spanish sources.",
            "ru_summary": "No relevant coverage found on this topic in Russian sources.",
            "divergence_detected": False,
            "divergence_note": "",
        }

    from groq import Groq

    def _format_chunks(chunks: list[dict], lang_label: str) -> str:
        if not chunks:
            return f"[No relevant {lang_label} coverage found on this topic]"
        # Deduplicate by URL: keep the highest-scoring chunk per article.
        # Chunks arrive sorted by score (best first), so first-seen wins.
        seen_urls: set[str] = set()
        unique: list[dict] = []
        for c in chunks:
            url = c.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique.append(c)
            if len(unique) == 4:
                break
        parts = []
        for i, c in enumerate(unique, start=1):
            parts.append(f"[{i}] {c.get('source_name', '?')} — {c.get('title', '')}\n{c.get('text', '')[:400]}")
        return "\n\n".join(parts)

    en_block = _format_chunks(chunks_en, "English")
    es_block = _format_chunks(chunks_es, "Spanish")
    ru_block = _format_chunks(chunks_ru, "Russian")

    prompt = f"""You are a cross-lingual media analyst. The user asked: "{query}"

Below are numbered news excerpts from three language ecosystems. Each excerpt is labelled with its source name.

=== ENGLISH SOURCES ===
{en_block}

=== SPANISH SOURCES ===
{es_block}

=== RUSSIAN SOURCES ===
{ru_block}

Your task:
1. Write a summary of how ENGLISH sources cover this topic. Only include articles that are directly relevant to the user's query. Dedicate one sentence to each relevant article. After each claim, cite the source name in square brackets, e.g. [BBC News]. If no English excerpts are relevant to the query, write "No relevant English coverage found on this topic."
2. Write a summary of how SPANISH sources cover this topic. Only include articles that are directly relevant to the user's query. Dedicate one sentence to each relevant article. After each claim, cite the source name in square brackets, e.g. [El País]. If no Spanish excerpts are relevant to the query, write "No relevant Spanish coverage found on this topic."
3. Write a summary of how RUSSIAN sources cover this topic. Only include articles that are directly relevant to the user's query. Dedicate one sentence to each relevant article. After each claim, cite the source name in square brackets, e.g. [Meduza]. If no Russian excerpts are relevant to the query, write "No relevant Russian coverage found on this topic."
4. State whether there is meaningful divergence in framing, emphasis, or facts (YES or NO).
5. If YES, write one sentence explaining the divergence.

Only cite sources that actually appear in the excerpts above. Do not invent source names.

Respond using EXACTLY this format (nothing else before or after):
EN_SUMMARY: <your English summary here>
ES_SUMMARY: <your Spanish summary here>
RU_SUMMARY: <your Russian summary here>
DIVERGENCE: <YES or NO>
DIVERGENCE_NOTE: <one sentence if YES, else leave blank>"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content or ""

    return _parse_comparison_response(raw, chunks_en, chunks_es, chunks_ru)


# ---------------------------------------------------------------------------
# get_source_metadata
# ---------------------------------------------------------------------------

def get_source_metadata(chunks: list[dict]) -> list[dict]:
    """Extract and deduplicate source metadata from a list of retrieved chunks.

    Returns one entry per unique URL:
      {source_name, title, url, language, published_at}
    """
    seen: set[str] = set()
    sources: list[dict] = []
    for c in chunks:
        url = c.get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append({
                "source_name": c.get("source_name", ""),
                "title": c.get("title", ""),
                "url": url,
                "language": c.get("language", ""),
                "published_at": c.get("published_at", ""),
            })
    return sources
