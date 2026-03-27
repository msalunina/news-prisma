"""Article parser — fetches a URL and returns clean article text via trafilatura."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import trafilatura
from trafilatura.settings import use_config

logger = logging.getLogger(__name__)

# Trafilatura config: faster, no unnecessary downloads
_TRAF_CONFIG = use_config()
_TRAF_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "20")


@dataclass
class ParsedArticle:
    url: str
    text: str           # clean body text
    title: str          # extracted or fallback from metadata
    author: str
    date: str           # ISO date string as returned by trafilatura, or ""


def parse_article(url: str, fallback_title: str = "") -> ParsedArticle | None:
    """Download and extract clean text from *url*.

    Returns None if trafilatura cannot extract meaningful content
    (e.g. paywalled, JavaScript-only, or network error).
    """
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as exc:
        logger.warning("Network error fetching %s: %s", url, exc)
        return None

    if not downloaded:
        logger.debug("No content downloaded from %s", url)
        return None

    result = trafilatura.extract(
        downloaded,
        config=_TRAF_CONFIG,
        include_comments=False,
        include_tables=False,
        output_format="txt",
        with_metadata=True,
    )

    if not result:
        logger.debug("trafilatura extracted nothing from %s", url)
        return None

    # trafilatura with_metadata=True returns YAML frontmatter + body:
    #   ---
    #   title: ...
    #   author: ...
    #   date: ...
    #   ---
    #   body text ...
    title = fallback_title
    author = ""
    date = ""
    text = ""

    lines = result.splitlines()
    if lines and lines[0] == "---":
        # Find closing ---
        close_idx = next((i for i, l in enumerate(lines[1:], 1) if l == "---"), None)
        if close_idx is not None:
            for line in lines[1:close_idx]:
                if line.startswith("title: "):
                    title = line[len("title: "):].strip() or fallback_title
                elif line.startswith("author: "):
                    author = line[len("author: "):].strip()
                elif line.startswith("date: "):
                    date = line[len("date: "):].strip()
            text = "\n".join(lines[close_idx + 1:]).strip()
        else:
            text = result.strip()
    else:
        text = result.strip()

    if len(text) < 100:
        logger.debug("Extracted text too short (%d chars) from %s", len(text), url)
        return None

    return ParsedArticle(url=url, text=text, title=title, author=author, date=date)
