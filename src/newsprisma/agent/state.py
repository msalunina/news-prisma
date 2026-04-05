"""AgentState — the shared state that flows through the LangGraph state machine."""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # --- Input ---
    query: str                          # Original user query
    query_language: str                 # Detected language: "en" | "es" | "ru"
    mode: Literal["single", "compare"]  # Routing decision

    # --- Retrieved context (one list per language) ---
    # Each entry is a dict with keys: text, source_name, title, url, language, published_at, score
    context_en: list[dict]
    context_es: list[dict]
    context_ru: list[dict]

    # --- Analysis ---
    perspective_diff: dict              # Structured cross-lingual comparison output

    # --- Output ---
    answer: str                         # Final answer text
    sources: list[dict]                 # Cited sources with metadata
