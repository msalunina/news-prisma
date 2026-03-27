# News Prisma

> *Reveals how the same story looks different through English, Spanish, and Russian media.*

A multilingual agentic RAG system that indexes news from EN/ES/RU sources and answers questions like: *"How is topic X covered differently across these three media ecosystems?"*


Fetches articles from 10 RSS feeds across three languages, extracts clean article text, deduplicates, and saves to a JSONL snapshot.

| Source | Language | Origin |
|---|---|---|
| BBC News | EN | UK |
| Al Jazeera | EN | Qatar |
| The Guardian | EN | UK |
| TASS | EN | Russia (state) |
| El País | ES | Spain |
| El Mundo | ES | Spain |
| BBC Mundo | ES | UK/LatAm |
| Meduza | RU | Latvia (exile) |
| Interfax | RU | Russia (independent) |
| Lenta.ru | RU | Russia |

---

## Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Clone and create a virtual environment

```bash
git clone https://github.com/msalunina/news-prisma.git
cd news-prisma

# with uv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# or with plain pip
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

<!-- ### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys 
``` -->

---

## Running the ingestion pipeline

```bash
python scripts/ingest.py
```

This will:
1. Fetch RSS feeds from all 10 sources (EN/ES/RU)
2. Deduplicate articles by URL and normalised title
3. Download and extract clean article text via [trafilatura](https://trafilatura.readthedocs.io/)
4. Save results to `data/snapshots/articles_YYYY_MM.jsonl`

**Options:**

```bash
# Faster run — saves RSS metadata + summary only, skips full article download
python scripts/ingest.py --skip-parse

# Limit articles per source (default: 50)
python scripts/ingest.py --max-per-source 3

# Custom output path
python scripts/ingest.py --output data/snapshots/my_snapshot.jsonl
```

**Example output:**

```

...

19:27:05 [INFO] newsprisma.ingestion.rss_fetcher — Fetching BBC Mundo (http://www.bbc.co.uk/mundo/index.xml) …
19:27:06 [INFO] newsprisma.ingestion.rss_fetcher —   → 55 articles from BBC Mundo
19:27:06 [INFO] newsprisma.ingestion.rss_fetcher — Fetching Meduza (https://meduza.io/rss/all) …
19:27:06 [INFO] newsprisma.ingestion.rss_fetcher —   → 30 articles from Meduza
19:27:06 [INFO] newsprisma.ingestion.rss_fetcher — Fetching Interfax (https://www.interfax.ru/rss.asp) …
19:27:07 [INFO] newsprisma.ingestion.rss_fetcher —   → 25 articles from Interfax
19:27:07 [INFO] newsprisma.ingestion.rss_fetcher — Fetching Lenta.ru (https://lenta.ru/rss/news) …
19:27:08 [INFO] newsprisma.ingestion.rss_fetcher —   → 200 articles from Lenta.ru
19:27:08 [INFO] ingest — Fetched 693 raw articles total
19:27:08 [INFO] ingest — After deduplication: 690 articles
19:27:08 [INFO] ingest —   BBC News: keeping 3 / 41 articles
19:27:08 [INFO] ingest —   Al Jazeera: keeping 3 / 25 articles
19:27:08 [INFO] ingest —   The Guardian: keeping 3 / 45 articles
19:27:08 [INFO] ingest —   TASS: keeping 3 / 98 articles
19:27:08 [INFO] ingest —   El País: keeping 3 / 144 articles
19:27:08 [INFO] ingest —   El Mundo: keeping 3 / 28 articles
19:27:08 [INFO] ingest —   BBC Mundo: keeping 3 / 55 articles
19:27:08 [INFO] ingest —   Meduza: keeping 3 / 30 articles
19:27:08 [INFO] ingest —   Interfax: keeping 3 / 25 articles
19:27:08 [INFO] ingest —   Lenta.ru: keeping 3 / 199 articles
19:27:08 [INFO] ingest — Total articles to process: 30
19:27:13 [INFO] ingest — Progress: 20 / 30
19:27:15 [INFO] ingest — ============================================================
19:27:15 [INFO] ingest — Done! Wrote 30 articles to data/snapshots/articles_2026_03.jsonl
19:27:15 [INFO] ingest — Language breakdown: EN=12  ES=9  RU=9
19:27:15 [INFO] ingest — Articles without full text (metadata only): 0
```

**JSONL record format:**

```json
{
  "url": "https://www.bbc.com/news/world-...",
  "title": "UN calls for ceasefire",
  "language": "en",
  "source_name": "BBC News",
  "source_origin": "UK",
  "published_at": "2026-03-27T09:00:00+00:00",
  "summary": "The United Nations has called...",
  "tags": ["world", "politics"],
  "text": "Full extracted article body...",
  "ingested_at": "2026-03-27T11:46:43+00:00"
}
```

<!-- ---

## Running the tests

```bash
pytest
```

All tests mock network calls, no internet connection required.

**What the tests cover:**

- `TestFetchFeed` — RSS parsing with a real feedparser against mock XML; verifies metadata extraction, graceful error handling, URL/title filtering, and language tagging
- `TestFetchAllSources` — verifies all three language groups are iterated from `sources.yaml`
- `TestParseArticle` — trafilatura wrapper; covers success path, network errors, empty downloads, short-text rejection, and fallback title logic
- `TestDeduplicate` — URL dedup, normalised-title dedup within a language, cross-language title collision allowed, empty list, order preservation -->

<!-- ---

## Project structure

```
news-prisma/
├── src/newsprisma/
│   ├── config.py                   # Pydantic Settings — loads .env
│   └── ingestion/
│       ├── sources.yaml            # RSS feed registry (10 sources, 3 languages)
│       ├── rss_fetcher.py          # feedparser → list[ArticleMetadata]
│       ├── article_parser.py       # trafilatura: URL → clean text
│       └── deduplicator.py         # URL + normalised-title dedup
├── scripts/
│   └── ingest.py                   # CLI entry point
├── tests/
│   └── test_ingestion.py           # 16 unit tests (all mocked)
├── data/
│   └── snapshots/                  # Output JSONL files (git-ignored)
├── pyproject.toml
└── .env.example
``` -->

---

## Tech stack

| Component | Choice |
|---|---|
| Language | Python 3.11 |
| Package manager | `uv` |
| RSS parsing | `feedparser` |
| Article extraction | `trafilatura` |
| Config | `pydantic-settings` |
<!-- | Testing | `pytest` + `pytest-mock` | -->
