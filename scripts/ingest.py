#!/usr/bin/env python
"""Ingestion script - fetch articles from all RSS sources and save to JSONL.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --max-per-source 10 --skip-parse
    python scripts/ingest.py --output data/snapshots/articles_custom.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newsprisma.config import settings
from newsprisma.ingestion.article_parser import parse_article
from newsprisma.ingestion.deduplicator import deduplicate
from newsprisma.ingestion.rss_fetcher import ArticleMetadata, fetch_all_sources

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def article_to_dict(meta: ArticleMetadata, text: str | None) -> dict:
    return {
        "url": meta.url,
        "title": meta.title,
        "language": meta.language,
        "source_name": meta.source_name,
        "source_origin": meta.source_origin,
        "published_at": meta.published_at.isoformat() if meta.published_at else None,
        "summary": meta.summary,
        "tags": meta.tags,
        "text": text,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest news articles from RSS feeds.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file path (default: data/snapshots/articles_<date>.jsonl)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=settings.max_articles_per_source,
        help="Max articles to keep per source (default from config)",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip full article parsing — save metadata + summary only (much faster)",
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y_%m")
        output_path = settings.snapshots_dir / f"articles_{date_str}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Fetch all RSS feeds ---
    logger.info("Fetching RSS feeds from %s …", settings.sources_yaml)
    raw_articles = list(fetch_all_sources(settings.sources_yaml))
    logger.info("Fetched %d raw articles total", len(raw_articles))

    # --- Step 2: Deduplicate ---
    articles = deduplicate(raw_articles)
    logger.info("After deduplication: %d articles", len(articles))

    # --- Step 3: Cap per source ---
    from collections import defaultdict
    per_source: dict[str, list[ArticleMetadata]] = defaultdict(list)
    for a in articles:
        per_source[a.source_name].append(a)

    capped: list[ArticleMetadata] = []
    for source_name, items in per_source.items():
        kept = items[: args.max_per_source]
        logger.info("  %s: keeping %d / %d articles", source_name, len(kept), len(items))
        capped.extend(kept)

    # Sort by language for a readable output
    capped.sort(key=lambda a: (a.language, a.source_name))
    logger.info("Total articles to process: %d", len(capped))

    # --- Step 4: Parse full text (unless --skip-parse) ---
    written = 0
    skipped = 0
    lang_counts: dict[str, int] = {"en": 0, "es": 0, "ru": 0}

    with open(output_path, "w", encoding="utf-8") as f:
        for i, meta in enumerate(capped, 1):
            text: str | None = None

            if not args.skip_parse:
                parsed = parse_article(meta.url, fallback_title=meta.title)
                if parsed:
                    text = parsed.text
                    if parsed.title:
                        meta.title = parsed.title
                else:
                    skipped += 1
                    logger.debug("[%d/%d] Skipped (no text): %s", i, len(capped), meta.url)
                    # Still save metadata-only record
            else:
                # Use RSS summary as text fallback
                text = meta.summary or None

            record = article_to_dict(meta, text)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            lang_counts[meta.language] = lang_counts.get(meta.language, 0) + 1

            if i % 20 == 0:
                logger.info("Progress: %d / %d", i, len(capped))

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Done! Wrote %d articles to %s", written, output_path)
    logger.info("Language breakdown: EN=%d  ES=%d  RU=%d", lang_counts.get("en", 0), lang_counts.get("es", 0), lang_counts.get("ru", 0))
    if not args.skip_parse:
        logger.info("Articles without full text (metadata only): %d", skipped)


if __name__ == "__main__":
    main()
