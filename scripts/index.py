"""CLI: Load articles from JSONL snapshot → chunk → embed → store in ChromaDB."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import track

# Allow running as `python scripts/index.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newsprisma.config import settings
from newsprisma.indexing.chunker import chunk_text
from newsprisma.indexing.embedder import Embedder
from newsprisma.indexing.store import VectorStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
console = Console()


def load_articles(path: Path) -> list[dict]:
    articles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def main() -> None:
    # Locate snapshot
    snapshot_dir = settings.snapshots_dir
    candidates = sorted(snapshot_dir.glob("*.jsonl"), reverse=True)
    if not candidates:
        console.print("[red]No JSONL snapshots found in data/snapshots/[/red]")
        sys.exit(1)

    snapshot_path = candidates[0]
    console.print(f"[bold]NewsPrisma — Indexing[/bold]")
    console.print(f"Snapshot : {snapshot_path}")
    console.print(f"ChromaDB : {settings.chroma_persist_dir}")
    console.print(f"Model    : {settings.embedding_model}")
    console.print()

    articles = load_articles(snapshot_path)
    console.print(f"Loaded [bold]{len(articles)}[/bold] articles")

    # Filter to those that have text
    articles_with_text = [a for a in articles if a.get("text", "").strip()]
    skipped = len(articles) - len(articles_with_text)
    if skipped:
        console.print(f"  [yellow]Skipped {skipped} articles without text[/yellow]")

    # Language breakdown
    by_lang: dict[str, int] = {}
    for a in articles_with_text:
        lang = a.get("language", "?")
        by_lang[lang] = by_lang.get(lang, 0) + 1
    for lang, count in sorted(by_lang.items()):
        console.print(f"  {lang}: {count} articles")

    console.print()

    # Load model + store
    embedder = Embedder(settings.embedding_model)
    store = VectorStore(settings.chroma_persist_dir)

    total_chunks = 0
    for article in track(articles_with_text, description="Chunking & indexing…"):
        chunks = chunk_text(article["text"])
        if not chunks:
            continue

        chunk_texts = [c.text for c in chunks]
        embeddings = embedder.encode(chunk_texts, batch_size=settings.embedding_batch_size, show_progress=False)

        n = store.upsert_article_chunks(
            url=article["url"],
            title=article.get("title", ""),
            language=article.get("language", "en"),
            source_name=article.get("source_name", ""),
            published_at=article.get("published_at", ""),
            chunk_texts=chunk_texts,
            embeddings=embeddings,
        )
        total_chunks += n

    console.print()
    console.print(f"[green]Done![/green] Indexed [bold]{total_chunks}[/bold] chunks from {len(articles_with_text)} articles.")
    console.print(f"Collection totals: {store.count()} chunks total")
    for lang in ["en", "es", "ru"]:
        console.print(f"  {lang}: {store.count(lang)} chunks")


if __name__ == "__main__":
    main()
