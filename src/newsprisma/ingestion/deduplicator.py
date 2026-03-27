"""Simple deduplication — removes articles with identical URLs or very similar titles."""

from __future__ import annotations

import re
from typing import TypeVar

from newsprisma.ingestion.rss_fetcher import ArticleMetadata

T = TypeVar("T", bound=ArticleMetadata)


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy title comparison."""
    return re.sub(r"[^\w\s]", "", title.lower()).strip()


def deduplicate(articles: list[T]) -> list[T]:
    """Return a deduplicated copy of *articles*.

    Dedup strategy (in order):
    1. Exact URL match — keep first seen.
    2. Normalised title match within the same language — keep first seen.
    """
    seen_urls: set[str] = set()
    seen_titles: set[tuple[str, str]] = set()  # (language, normalised_title)
    result: list[T] = []

    for article in articles:
        url = article.url.rstrip("/")
        norm_title = _normalise_title(article.title)
        title_key = (article.language, norm_title)

        if url in seen_urls:
            continue
        if norm_title and title_key in seen_titles:
            continue

        seen_urls.add(url)
        if norm_title:
            seen_titles.add(title_key)
        result.append(article)

    return result
