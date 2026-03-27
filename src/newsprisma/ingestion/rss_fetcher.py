"""RSS feed fetcher — parses feeds and returns article metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import feedparser
import yaml

logger = logging.getLogger(__name__)

LANG_MAP = {"english": "en", "spanish": "es", "russian": "ru"}


@dataclass
class ArticleMetadata:
    url: str
    title: str
    language: str           # "en" | "es" | "ru"
    source_name: str
    source_origin: str      # country / region of publication
    published_at: datetime | None
    summary: str = ""
    tags: list[str] = field(default_factory=list)


def _parse_date(entry: feedparser.FeedParserDict) -> datetime | None:
    """Best-effort datetime extraction from a feedparser entry."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t = getattr(entry, attr, None)
        if t is not None:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def fetch_feed(source_name: str, rss_url: str, language: str, source_origin: str = "") -> list[ArticleMetadata]:
    """Fetch and parse a single RSS feed. Returns a list of ArticleMetadata."""
    logger.info("Fetching %s (%s) …", source_name, rss_url)
    try:
        feed = feedparser.parse(rss_url)
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", rss_url, exc)
        return []

    if feed.bozo and not feed.entries:
        logger.warning("Malformed feed from %s (bozo=%s)", rss_url, feed.bozo_exception)
        return []

    articles: list[ArticleMetadata] = []
    for entry in feed.entries:
        url = entry.get("link", "")
        title = entry.get("title", "").strip()
        if not url or not title:
            continue

        articles.append(
            ArticleMetadata(
                url=url,
                title=title,
                language=language,
                source_name=source_name,
                source_origin=source_origin,
                published_at=_parse_date(entry),
                summary=entry.get("summary", "").strip(),
                tags=[t.get("term", "") for t in entry.get("tags", [])],
            )
        )

    logger.info("  → %d articles from %s", len(articles), source_name)
    return articles


def fetch_all_sources(sources_yaml: Path) -> Iterator[ArticleMetadata]:
    """Load sources.yaml and yield ArticleMetadata for every source/language combo."""
    with open(sources_yaml) as f:
        config = yaml.safe_load(f)

    for lang_key, sources in config["sources"].items():
        language = LANG_MAP.get(lang_key, lang_key)
        for source in sources:
            articles = fetch_feed(
                source["name"],
                source["rss"],
                language,
                source_origin=source.get("origin", ""),
            )
            yield from articles
