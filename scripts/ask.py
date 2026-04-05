#!/usr/bin/env python
"""CLI for the NewsPrisma agent.

Usage:
    python scripts/ask.py "What is happening with AI regulation?"
    python scripts/ask.py "¿Qué pasa con la regulación de la IA?"
    python scripts/ask.py "How is climate change covered differently in English vs Spanish news?"
    python scripts/ask.py "Что происходит с ИИ?"
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

    from newsprisma.agent.graph import run_agent

    console.print(f"\n[bold cyan]NewsPrisma[/bold cyan] — asking: [italic]{args.query}[/italic]\n")

    with console.status("[bold green]Running agent…"):
        result = run_agent(args.query)

    lang_label = {"en": "English", "es": "Spanish", "ru": "Russian"}.get(
        result["query_language"], result["query_language"]
    )
    mode_label = "Cross-lingual comparison" if result["mode"] == "compare" else "Single-language"

    # --- Answer panel ---
    console.print(Panel(
        Text(result["answer"]),
        title=f"[bold green]Answer[/bold green]  "
              f"(language: {lang_label} · mode: {mode_label})",
        border_style="green",
    ))

    # --- Divergence warning (compare mode) ---
    diff = result.get("perspective_diff", {})
    if diff.get("divergence_detected"):
        console.print(
            f"[bold yellow]⚠ Divergence detected:[/bold yellow] "
            f"{diff.get('divergence_note', '')}"
        )

    # --- Sources table ---
    sources = result.get("sources", [])
    if sources:
        table = Table(title="Sources", show_lines=True)
        table.add_column("Lang", style="bold", width=5)
        table.add_column("Source", style="cyan")
        table.add_column("Title")
        table.add_column("Published", width=12)

        for src in sources:
            table.add_row(
                src["language"].upper(),
                src["source_name"],
                src["title"][:80] + ("…" if len(src["title"]) > 80 else ""),
                src["published_at"][:10] if src["published_at"] else "—",
            )
        console.print(table)
    else:
        console.print("[yellow]No sources retrieved.[/yellow]")


if __name__ == "__main__":
    main()
