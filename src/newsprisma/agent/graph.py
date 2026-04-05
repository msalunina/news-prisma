"""LangGraph state machine for NewsPrisma.

Graph topology:

    START
      │
      ▼
  [detect_language]        — detect language of user query
      │
      ▼
  [plan_retrieval]         — decide: single-language or cross-lingual compare?
      │
      ├── "single" ──► [retrieve_targeted] ──► [generate_answer] ──► END
      │
      └── "compare" ─► [retrieve_en]
                       [retrieve_es]  (run sequentially — LangGraph adds
                       [retrieve_ru]   parallelism with Send API if needed)
                           │
                           ▼
                    [compare_perspectives]
                           │
                           ▼
                  [generate_comparison_answer] ──► END
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.graph import END, StateGraph

from newsprisma.agent.prompts import build_context_message, get_system_prompt
from newsprisma.agent.rag import _call_groq
from newsprisma.agent.state import AgentState
from newsprisma.agent.tools import (
    detect_perspective_diff,
    get_source_metadata,
    rerank_chunks,
    search_news,
)

logger = logging.getLogger(__name__)

# Relevance thresholds for retrieval and cross-lingual adaptive filtering.
#
# _MIN_RELEVANCE_SCORE: hard floor applied in every retrieve node.
#   Chunks below this are dropped before being stored in state.
#
# _HIGH_CONFIDENCE_THRESHOLD / _ABSENT_THRESHOLD / _MARGINAL_FACTOR:
#   Used by _adaptive_filter (compare mode only) to apply a second, cross-lingual pass.
#   See _adaptive_filter docstring for the three-regime algorithm.
# Lenient hard floor — just cuts true noise; reranker handles fine-grained selection.
_MIN_RELEVANCE_SCORE       = 0.20
_HIGH_CONFIDENCE_THRESHOLD = 0.50
_ABSENT_THRESHOLD          = 0.40
_MARGINAL_FACTOR           = 0.95

# Per-language overrides
_LANG_OVERRIDES = {
    "ru": {
        "min_relevance_score": 0.45, 
        "high_confidence_threshold": 0.46,
        "absent_threshold": 0.40, 
        "marginal_factor": 0.95,
    },
    # "es": {
    #     "min_relevance_score": 0.45, 
    #     "high_confidence_threshold": 0.50, 
    #     "absent_threshold": 0.40, 
    #     "marginal_factor": 0.95,
    # },
}

# Keywords that signal the user wants a cross-lingual comparison
_COMPARE_KEYWORDS = {
    # English
    "compare", "comparison", "differently", "different", "versus", "vs",
    "coverage", "perspective", "perspectives", "english vs", "spanish vs",
    "russian vs", "en vs", "es vs", "ru vs", "across languages",
    "across sources", "media",
    # Spanish
    "comparar", "diferente", "diferencias", "perspectiva", "cobertura",
    # Russian
    "сравнить", "разница", "различия", "перспектива", "освещение",
}


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def node_detect_language(state: AgentState) -> dict:
    """Detect the language of the user's query."""
    from newsprisma.utils.language import detect_language

    lang = detect_language(state["query"])
    logger.info("[detect_language] %r → %s", state["query"][:60], lang)
    return {"query_language": lang}


def node_plan_retrieval(state: AgentState) -> dict:
    """Decide retrieval mode: 'single' or 'compare'."""
    query_lower = state["query"].lower()
    is_compare = any(kw in query_lower for kw in _COMPARE_KEYWORDS)
    mode = "compare" if is_compare else "single"
    logger.info("[plan_retrieval] mode=%s", mode)
    return {"mode": mode}


def _filter_by_relevance(chunks: list[dict]) -> list[dict]:
    """Drop chunks below the minimum relevance floor."""
    return [c for c in chunks if c.get("score", 0.0) >= _MIN_RELEVANCE_SCORE]


def _adaptive_filter(
    chunks_en: list[dict],
    chunks_es: list[dict],
    chunks_ru: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Per-language adaptive relevance filter for compare mode.

    Each language is evaluated against its own top score, so that a language's
    chunks are not penalised or boosted by another language's score distribution
    (which varies due to cross-lingual embedding gaps).

    Per-language regimes:
    - lang_top >= 0.55: well-covered → use absolute floor (_MIN_RELEVANCE_SCORE)
    - 0.40 <= lang_top < 0.55: marginal → threshold = max(0.40, top * 0.90)
    - lang_top < 0.40: topic absent in this language → return []
    """
    def _filter_lang(chunks: list[dict], label: str) -> list[dict]:
        if not chunks:
            return []
        ov = _LANG_OVERRIDES.get(label, {})
        hc  = ov.get("high_confidence_threshold", _HIGH_CONFIDENCE_THRESHOLD)
        ab  = ov.get("absent_threshold",          _ABSENT_THRESHOLD)
        mn  = ov.get("min_relevance_score",       _MIN_RELEVANCE_SCORE)
        mf  = ov.get("marginal_factor",           _MARGINAL_FACTOR)

        lang_top = max(c.get("score", 0.0) for c in chunks)
        if lang_top >= hc:
            threshold = mn
        elif lang_top >= ab:
            threshold = max(ab, lang_top * mf)
        else:
            logger.info("[adaptive_filter] %s absent (top=%.3f < %.2f)", label, lang_top, ab)
            return []
        logger.info("[adaptive_filter] %s top=%.3f → threshold=%.3f", label, lang_top, threshold)
        return [c for c in chunks if c.get("score", 0.0) >= threshold]

    return (
        _filter_lang(chunks_en, "en"),
        _filter_lang(chunks_es, "es"),
        _filter_lang(chunks_ru, "ru"),
    )


# How many candidates to fetch before reranking, and how many to keep after.
_FETCH_TOP_K      = 20
_RERANK_TOP_N     = 6
# No rerank score floor — always keep top-N for maximum recall.
# Off-topic detection is delegated to the LLM in detect_perspective_diff.
_MIN_RERANK_SCORE: float | None = None


def node_retrieve_targeted(state: AgentState) -> dict:
    """Single-language retrieval: search across all languages for the query."""
    query = state["query"]
    candidates = _filter_by_relevance(search_news(query, language=None, top_k=_FETCH_TOP_K))
    chunks = rerank_chunks(
        query, candidates, top_n=_RERANK_TOP_N, min_rerank_score=_MIN_RERANK_SCORE
    )
    logger.info("[retrieve_targeted] %d → %d chunks after rerank", len(candidates), len(chunks))

    en = [c for c in chunks if c["language"] == "en"]
    es = [c for c in chunks if c["language"] == "es"]
    ru = [c for c in chunks if c["language"] == "ru"]
    return {"context_en": en, "context_es": es, "context_ru": ru}


def node_retrieve_en(state: AgentState) -> dict:
    query = state["query"]
    candidates = _filter_by_relevance(search_news(query, language="en", top_k=_FETCH_TOP_K))
    chunks = rerank_chunks(
        query, candidates, top_n=_RERANK_TOP_N, min_rerank_score=_MIN_RERANK_SCORE
    )
    logger.info("[retrieve_en] %d → %d chunks after rerank", len(candidates), len(chunks))
    return {"context_en": chunks}


def node_retrieve_es(state: AgentState) -> dict:
    query = state["query"]
    candidates = _filter_by_relevance(search_news(query, language="es", top_k=_FETCH_TOP_K))
    chunks = rerank_chunks(
        query, candidates, top_n=_RERANK_TOP_N, min_rerank_score=_MIN_RERANK_SCORE
    )
    logger.info("[retrieve_es] %d → %d chunks after rerank", len(candidates), len(chunks))
    return {"context_es": chunks}


def node_retrieve_ru(state: AgentState) -> dict:
    query = state["query"]
    candidates = _filter_by_relevance(search_news(query, language="ru", top_k=_FETCH_TOP_K))
    chunks = rerank_chunks(
        query, candidates, top_n=_RERANK_TOP_N, min_rerank_score=_MIN_RERANK_SCORE
    )
    logger.info("[retrieve_ru] %d → %d chunks after rerank", len(candidates), len(chunks))
    return {"context_ru": chunks}


def node_compare_perspectives(state: AgentState) -> dict:
    """Call detect_perspective_diff to build structured cross-lingual comparison."""
    from newsprisma.config import settings

    diff = detect_perspective_diff(
        chunks_en=state.get("context_en", []),
        chunks_es=state.get("context_es", []),
        chunks_ru=state.get("context_ru", []),
        query=state["query"],
        model=settings.groq_model,
        api_key=settings.groq_api_key,
    )
    logger.info("[compare_perspectives] divergence=%s", diff.get("divergence_detected"))
    return {"perspective_diff": diff}


def node_generate_answer(state: AgentState) -> dict:
    """Generate a grounded answer for single-language mode."""
    from newsprisma.config import settings

    # Combine all retrieved chunks
    all_chunks = (
        state.get("context_en", [])
        + state.get("context_es", [])
        + state.get("context_ru", [])
    )

    if not all_chunks:
        return {
            "answer": "I could not find any relevant articles in the index. Please run the ingestion and indexing scripts first.",
            "sources": [],
        }

    system_prompt = get_system_prompt(state["query_language"])
    context_message = build_context_message(all_chunks)
    answer = _call_groq(
        system_prompt=system_prompt,
        context_message=context_message,
        query=state["query"],
        model=settings.groq_model,
        api_key=settings.groq_api_key,
    )

    sources = get_source_metadata(all_chunks)
    return {"answer": answer, "sources": sources}


def node_generate_comparison_answer(state: AgentState) -> dict:
    """Format the cross-lingual comparison answer directly from the diff.

    The summaries in perspective_diff already contain inline [Source Name] citations
    produced by detect_perspective_diff. We format them into the final answer without
    an extra LLM call to avoid citation hallucinations.
    """
    diff = state.get("perspective_diff", {})

    en_summary = diff.get("en_summary", "No English sources retrieved.")
    es_summary = diff.get("es_summary", "No Spanish sources retrieved.")
    ru_summary = diff.get("ru_summary", "No Russian sources retrieved.")
    divergence = diff.get("divergence_detected", False)
    divergence_note = diff.get("divergence_note", "")

    en_chunks = state.get("context_en", [])
    es_chunks = state.get("context_es", [])
    ru_chunks = state.get("context_ru", [])

    def _article_titles(chunks: list[dict]) -> str:
        """Return a deduplicated list of article titles retrieved for a language."""
        seen: set[str] = set()
        titles: list[str] = []
        for c in chunks:
            url = c.get("url", "")
            if url not in seen:
                seen.add(url)
                title = c.get("title", "").strip()
                source = c.get("source_name", "")
                if title:
                    titles.append(f'"{title}" ({source})')
        return ", ".join(titles) if titles else ""

    def _section(header: str, summary: str, chunks: list[dict]) -> str:
        # titles = _article_titles(chunks)
        # if titles:
        #     return f"**{header}**\n*Articles retrieved: {titles}*\n{summary}"
        return f"**{header}**\n{summary}"

    parts = [
        _section("English media perspective:", en_summary, en_chunks),
        _section("Spanish media perspective:", es_summary, es_chunks),
        _section("Russian media perspective:", ru_summary, ru_chunks),
    ]
    if divergence and divergence_note:
        parts.append(f"**Divergence detected:** {divergence_note}")

    answer = "\n\n".join(parts)

    # Only include sources whose name was actually cited in the summaries.
    cited_names = set(re.findall(r'\[([^\]]+)\]', en_summary + es_summary + ru_summary))
    all_chunks = en_chunks + es_chunks + ru_chunks
    sources = [s for s in get_source_metadata(all_chunks) if s["source_name"] in cited_names]
    if not sources:
        sources = get_source_metadata(all_chunks)
    return {"answer": answer, "sources": sources}


def _lang_name(lang: str) -> str:
    return {"en": "English", "es": "Spanish", "ru": "Russian"}.get(lang, "English")


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_after_plan(state: AgentState) -> str:
    """Return the name of the next node based on the chosen mode."""
    return "single" if state["mode"] == "single" else "compare"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Build and compile the NewsPrisma LangGraph state machine."""
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("detect_language", node_detect_language)
    g.add_node("plan_retrieval", node_plan_retrieval)
    g.add_node("retrieve_targeted", node_retrieve_targeted)
    g.add_node("retrieve_en", node_retrieve_en)
    g.add_node("retrieve_es", node_retrieve_es)
    g.add_node("retrieve_ru", node_retrieve_ru)
    g.add_node("compare_perspectives", node_compare_perspectives)
    g.add_node("generate_answer", node_generate_answer)
    g.add_node("generate_comparison_answer", node_generate_comparison_answer)

    # Entry point
    g.set_entry_point("detect_language")

    # Linear edges
    g.add_edge("detect_language", "plan_retrieval")

    # Conditional routing after plan_retrieval
    g.add_conditional_edges(
        "plan_retrieval",
        route_after_plan,
        {
            "single": "retrieve_targeted",
            "compare": "retrieve_en",
        },
    )

    # Single-language path
    g.add_edge("retrieve_targeted", "generate_answer")
    g.add_edge("generate_answer", END)

    # Cross-lingual path (sequential)
    g.add_edge("retrieve_en", "retrieve_es")
    g.add_edge("retrieve_es", "retrieve_ru")
    g.add_edge("retrieve_ru", "compare_perspectives")
    g.add_edge("compare_perspectives", "generate_comparison_answer")
    g.add_edge("generate_comparison_answer", END)

    return g.compile()


# Module-level compiled graph (lazy singleton)
_graph: Any | None = None


def get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_agent(query: str) -> dict:
    """Run the full agent graph for a query.

    Returns a dict with keys: query, query_language, mode, answer, sources,
    perspective_diff.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "query": query,
        "query_language": "",
        "mode": "single",
        "context_en": [],
        "context_es": [],
        "context_ru": [],
        "perspective_diff": {},
        "answer": "",
        "sources": [],
    }

    final_state = graph.invoke(initial_state)
    return {
        "query": final_state["query"],
        "query_language": final_state["query_language"],
        "mode": final_state["mode"],
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "perspective_diff": final_state.get("perspective_diff", {}),
    }
