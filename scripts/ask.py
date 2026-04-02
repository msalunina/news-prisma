#!/usr/bin/env python
"""CLI for the NewsPrisma RAG pipeline.

Usage:
    python scripts/ask.py "What is happening with AI regulation?"
    python scripts/ask.py "¿Qué pasa con la regulación de la IA?" --lang es
    python scripts/ask.py "Что происходит с ИИ?" --top-k 12
    python scripts/ask.py "AI news" --filter-lang en
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Suppress HuggingFace update checks (model is cached locally) and tokenizer fork warnings
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask NewsPrisma a question in any language.")
    parser.add_argument("query", help="Question to ask (any language)")
    parser.add_argument(
        "--top-k", type=int, default=8,
        help="Number of context chunks to retrieve (default: 8)",
    )
    parser.add_argument(
        "--filter-lang", choices=["en", "es", "ru"], default=None,
        help="Restrict retrieval to a single language (default: all languages)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show debug logs")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    from newsprisma.agent.rag import retrieve_and_answer

    console.print(f"\n[bold cyan]NewsPrisma[/bold cyan] — asking: [italic]{args.query}[/italic]\n")

    with console.status("[bold green]Retrieving and generating answer…"):
        result = retrieve_and_answer(
            query=args.query,
            top_k=args.top_k,
            language_filter=args.filter_lang,
        )

    # --- Answer panel ---
    lang_label = {"en": "English", "es": "Spanish", "ru": "Russian"}.get(
        result.query_language, result.query_language
    )
    console.print(Panel(
        Text(result.answer),
        title=f"[bold green]Answer[/bold green]  (query language: {lang_label})",
        border_style="green",
    ))

    # --- Sources table ---
    if result.sources:
        table = Table(title="Sources", show_lines=True)
        table.add_column("Lang", style="bold", width=5)
        table.add_column("Source", style="cyan")
        table.add_column("Title")
        table.add_column("Published", width=12)

        lang_icons = {"en": "EN", "es": "ES", "ru": "RU"}
        for src in result.sources:
            table.add_row(
                lang_icons.get(src["language"], src["language"]),
                src["source_name"],
                src["title"][:80] + ("…" if len(src["title"]) > 80 else ""),
                src["published_at"][:10] if src["published_at"] else "—",
            )
        console.print(table)
    else:
        console.print("[yellow]No sources retrieved.[/yellow]")


if __name__ == "__main__":
    main()
