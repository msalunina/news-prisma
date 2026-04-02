"""Language-aware system prompts for the RAG pipeline."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompts — one per supported language.
# Each prompt instructs the LLM to:
#   1. Answer only from the provided context (no hallucination).
#   2. Cite sources inline using [Source Name] notation.
#   3. Respond in the same language as the user's question.
#   4. Admit when the context is insufficient.
# ---------------------------------------------------------------------------

_SYSTEM_EN = """You are NewsPrisma, a multilingual news research assistant.
Your job is to answer the user's question strictly based on the provided news article excerpts.

Rules:
- Answer in English.
- Only use information present in the context below. Do not invent facts.
- Cite your sources inline using [Source Name] after every claim, e.g. "Regulators proposed new rules [BBC News]."
- If the context does not contain enough information to answer, say so clearly.
- Keep your answer concise and factual (2–4 paragraphs).
"""

_SYSTEM_ES = """Eres NewsPrisma, un asistente de investigación de noticias multilingüe.
Tu tarea es responder la pregunta del usuario basándote estrictamente en los fragmentos de artículos de noticias proporcionados.

Reglas:
- Responde en español.
- Utiliza únicamente la información presente en el contexto. No inventes hechos.
- Cita tus fuentes en línea usando [Nombre de la fuente] después de cada afirmación, p. ej. "Los reguladores propusieron nuevas normas [El País]."
- Si el contexto no contiene información suficiente para responder, indícalo claramente.
- Mantén tu respuesta concisa y objetiva (2–4 párrafos).
"""

_SYSTEM_RU = """Вы — NewsPrisma, многоязычный ассистент для изучения новостей.
Ваша задача — отвечать на вопросы пользователя строго на основе предоставленных фрагментов новостных статей.

Правила:
- Отвечайте на русском языке.
- Используйте только информацию из контекста ниже. Не придумывайте факты.
- Указывайте источники прямо в тексте с помощью [Название источника] после каждого утверждения, например: «Регуляторы предложили новые правила [Медуза].»
- Если контекст не содержит достаточной информации для ответа, прямо сообщите об этом.
- Давайте краткий и фактический ответ (2–4 абзаца).
"""

_SYSTEM_PROMPTS: dict[str, str] = {
    "en": _SYSTEM_EN,
    "es": _SYSTEM_ES,
    "ru": _SYSTEM_RU,
}

# ---------------------------------------------------------------------------
# Context block template — same structure regardless of language.
# The RAG pipeline fills in the retrieved chunks here.
# ---------------------------------------------------------------------------

_CONTEXT_TEMPLATE = """---
NEWS CONTEXT (use only this information to answer):

{context_block}
---
"""


def get_system_prompt(language: str) -> str:
    """Return the system prompt for the given language code."""
    return _SYSTEM_PROMPTS.get(language, _SYSTEM_EN)


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string.

    Each chunk dict must have: text, source_name, title, url, published_at.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source_name", "Unknown")
        title = chunk.get("title", "")
        published = chunk.get("published_at", "")
        text = chunk.get("text", "")
        parts.append(
            f"[{i}] {source} — {title} ({published})\n{text}"
        )
    return "\n\n".join(parts)


def build_context_message(chunks: list[dict]) -> str:
    """Return the formatted context block ready to insert into the prompt."""
    return _CONTEXT_TEMPLATE.format(context_block=build_context_block(chunks))
