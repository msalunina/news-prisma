#!/usr/bin/env python
"""Dev tool: inspect which articles are selected for LLM calls — no Groq calls made.

Shows the full retrieval pipeline (search → relevance filter → rerank → adaptive filter)
and prints every chunk that would be sent to the LLM.

Usage:
    python scripts/inspect_retrieval.py "What is happening with AI regulation?"
    python scripts/inspect_retrieval.py "compare AI regulation en vs es vs ru" --mode compare
    python scripts/inspect_retrieval.py "Что происходит с ИИ?" --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _print_chunks(console, chunks: list[dict], label: str) -> None:
    from rich.table import Table

    if not chunks:
        console.print(f"  [dim]{label}: no chunks[/dim]")
        return

    table = Table(title=label, show_lines=True, expand=True)
    table.add_column("Score", width=6, style="bold yellow")
    table.add_column("Lang", width=5)
    table.add_column("Source", style="cyan", width=20)
    table.add_column("Title", width=40)
    table.add_column("Published", width=12)
    table.add_column("Chunk preview", width=60)

    for c in chunks:
        score = f"{c.get('score', 0):.3f}"
        lang = c.get("language", "?").upper()
        source = c.get("source_name", "?")
        title = (c.get("title") or "")[:40]
        pub = (c.get("published_at") or "")[:10]
        text = (c.get("text") or "").replace("\n", " ")[:80] + "…"
        table.add_row(score, lang, source, title, pub, text)

    console.print(table)


def run_inspect(query: str, mode: str | None, verbose: bool) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule

    from newsprisma.agent.graph import (
        _FETCH_TOP_K,
        _MIN_RERANK_SCORE,
        _RERANK_TOP_N,
        _filter_by_relevance,
    )
    from newsprisma.agent.tools import rerank_chunks, search_news
    from newsprisma.utils.language import detect_language

    console = Console()

    # --- Detect language and mode ---
    detected_lang = detect_language(query)
    if mode is None:
        from newsprisma.agent.graph import _COMPARE_KEYWORDS
        mode = "compare" if any(kw in query.lower() for kw in _COMPARE_KEYWORDS) else "single"

    console.print(f"\n[bold cyan]NewsPrisma — retrieval inspector[/bold cyan]")
    console.print(f"Query      : [italic]{query}[/italic]")
    console.print(f"Detected   : lang=[bold]{detected_lang}[/bold]  mode=[bold]{mode}[/bold]\n")

    if mode == "single":
        console.print(Rule("[bold]Single-language retrieval[/bold]"))

        raw = search_news(query, language=None, top_k=_FETCH_TOP_K)
        console.print(f"[dim]search_news returned {len(raw)} chunks (top_k={_FETCH_TOP_K})[/dim]")

        filtered = _filter_by_relevance(raw)
        dropped = len(raw) - len(filtered)
        console.print(f"[dim]After relevance floor: {len(filtered)} kept, {dropped} dropped[/dim]")

        reranked = rerank_chunks(
            query, filtered, top_n=_RERANK_TOP_N, min_rerank_score=_MIN_RERANK_SCORE
        )
        console.print(
            f"[dim]After rerank: {len(reranked)} chunks"
            f" (top_n={_RERANK_TOP_N}, min_score={_MIN_RERANK_SCORE})[/dim]\n"
        )

        _print_chunks(console, reranked, f"Chunks → LLM ({len(reranked)} total)")

        # Breakdown by language
        for lang in ("en", "es", "ru"):
            lang_chunks = [c for c in reranked if c.get("language") == lang]
            if lang_chunks:
                console.print(f"  [dim]{lang.upper()}: {len(lang_chunks)} chunk(s)[/dim]")

    else:
        console.print(Rule("[bold]Cross-lingual comparison retrieval[/bold]"))

        results: dict[str, dict] = {}
        for lang in ("en", "es", "ru"):
            raw = search_news(query, language=lang, top_k=_FETCH_TOP_K)
            filtered = _filter_by_relevance(raw)
            reranked = rerank_chunks(
                query, filtered, top_n=_RERANK_TOP_N, min_rerank_score=_MIN_RERANK_SCORE
            )
            results[lang] = {"raw": raw, "filtered": filtered, "reranked": reranked}
            console.print(
                f"[dim]{lang.upper()}: {len(raw)} raw → {len(filtered)} filtered"
                f" → {len(reranked)} reranked[/dim]"
            )

        en_f = results["en"]["reranked"]
        es_f = results["es"]["reranked"]
        ru_f = results["ru"]["reranked"]

        console.print()
        _print_chunks(console, en_f, f"EN chunks → LLM ({len(en_f)})")
        _print_chunks(console, es_f, f"ES chunks → LLM ({len(es_f)})")
        _print_chunks(console, ru_f, f"RU chunks → LLM ({len(ru_f)})")

    console.print(Panel(
        "[bold green]Done.[/bold green] No Groq calls were made.",
        border_style="green",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect retrieval without calling the LLM.")
    parser.add_argument("query", help="Question to ask")
    parser.add_argument(
        "--mode",
        choices=["single", "compare"],
        default=None,
        help="Force retrieval mode (default: auto-detect from query)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show debug logs")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    run_inspect(args.query, args.mode, args.verbose)


if __name__ == "__main__":
    main()
