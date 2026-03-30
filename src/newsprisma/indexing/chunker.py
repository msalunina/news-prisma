"""Sentence-aware text chunker.

Splits article text into overlapping chunks that respect sentence boundaries.
Sentence detection uses a simple regex that handles EN/ES/RU adequately without
requiring a language-specific tokenizer (spaCy adds 1 GB of model downloads).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sentence boundary: period/!/?/… followed by whitespace + capital or end-of-string.
# Covers most EN/ES/RU sentences well enough for chunking purposes.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÜÑА-ЯЁ\"\'])")


@dataclass
class Chunk:
    text: str
    chunk_index: int        # 0-based position within the source article
    char_start: int         # character offset in the original text
    char_end: int


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using the boundary regex."""
    parts = _SENT_BOUNDARY.split(text.strip())
    # Remove empties and very short fragments (< 20 chars)
    return [s.strip() for s in parts if len(s.strip()) >= 20]


def chunk_text(
    text: str,
    max_chars: int = 800,
    overlap_sents: int = 1,
) -> list[Chunk]:
    """Split *text* into sentence-aware chunks of at most *max_chars* characters.

    Args:
        text: The raw article body.
        max_chars: Soft upper limit per chunk (a single long sentence may exceed it).
        overlap_sents: Number of sentences from the end of the previous chunk to
            include at the start of the next chunk (context continuity).

    Returns:
        List of Chunk objects in order.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sents: list[str] = []
    current_len = 0
    chunk_idx = 0
    char_cursor = 0

    for sent in sentences:
        sent_len = len(sent)

        # If adding this sentence would exceed the limit (and we already have content),
        # flush the current chunk first.
        if current_sents and current_len + sent_len + 1 > max_chars:
            chunk_text_str = " ".join(current_sents)
            char_start = char_cursor - current_len
            chunks.append(
                Chunk(
                    text=chunk_text_str,
                    chunk_index=chunk_idx,
                    char_start=max(0, char_start),
                    char_end=char_cursor,
                )
            )
            chunk_idx += 1

            # Carry over the last N sentences as overlap
            overlap = current_sents[-overlap_sents:] if overlap_sents else []
            current_sents = overlap
            current_len = sum(len(s) for s in overlap) + max(0, len(overlap) - 1)

        current_sents.append(sent)
        current_len += sent_len + (1 if len(current_sents) > 1 else 0)
        char_cursor += sent_len + 1  # approximate — good enough for metadata

    # Flush remainder
    if current_sents:
        chunk_text_str = " ".join(current_sents)
        char_start = char_cursor - current_len
        chunks.append(
            Chunk(
                text=chunk_text_str,
                chunk_index=chunk_idx,
                char_start=max(0, char_start),
                char_end=char_cursor,
            )
        )

    return chunks
