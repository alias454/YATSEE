"""
Chunking helpers for YATSEE intelligence jobs.
"""

from __future__ import annotations

import re
from typing import List, Set

from yatsee.providers.tokenization import estimate_token_count


def prepare_text_chunk_word(
    text: str,
    max_tokens: int = 2500,
    overlap_tokens: float | None = None,
) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    This preserves the original summarizer behavior of approximating chunk size
    by converting the token budget into a word budget using a fixed heuristic.

    Overlap is applied between adjacent chunks so downstream summarization does
    not lose too much local context at chunk boundaries.

    :param text: Input text to split
    :param max_tokens: Approximate token budget per chunk
    :param overlap_tokens: Optional overlap token count; if omitted, overlap is
        derived dynamically from the current chunk size
    :return: List of overlapping word-based chunks
    """
    words = text.split()
    max_words = int(max_tokens * 0.75)

    # Fast path for short inputs to avoid unnecessary chunking work.
    if len(words) <= max_words:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        current_chunk_words = end - start
        overlap_words = (
            min(int(current_chunk_words * 0.1), 600)
            if overlap_tokens is None
            else int(overlap_tokens * 0.75)
        )

        # Advance by chunk size minus overlap so neighboring chunks retain context.
        start += max_words - overlap_words

    return chunks


def prepare_text_chunk_sentence(
    text: str,
    max_tokens: int = 2500,
    overlap_tokens: float | None = None,
) -> List[str]:
    """
    Split text into sentence-aligned chunks using token estimates.

    Sentence alignment preserves semantic boundaries better than raw word
    slicing, which is important when downstream prompts are trying to preserve
    meeting structure, motions, and speaker intent.

    :param text: Input text
    :param max_tokens: Approximate token budget per chunk
    :param overlap_tokens: Optional overlap token count; if omitted, overlap is
        derived dynamically from the current chunk size
    :return: List of sentence-aligned chunks
    """
    # Fast path when the input already fits into a single chunk.
    if estimate_token_count(text) <= max_tokens:
        return [text]

    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0
    index = 0

    while index < len(sentences):
        sentence = sentences[index]
        sentence_tokens = estimate_token_count(sentence)

        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))

            overlap_target = (
                min(int(current_tokens * 0.1), 600)
                if overlap_tokens is None
                else int(overlap_tokens)
            )

            overlap_chunk: List[str] = []
            overlap_seen = 0

            # Rebuild overlap from the tail of the previous chunk to preserve continuity.
            for existing_sentence in reversed(current_chunk):
                overlap_chunk.insert(0, existing_sentence)
                overlap_seen += estimate_token_count(existing_sentence)
                if overlap_seen >= overlap_target:
                    break

            current_chunk = overlap_chunk
            current_tokens = sum(
                estimate_token_count(existing_sentence)
                for existing_sentence in current_chunk
            )

        current_chunk.append(sentence)
        current_tokens += sentence_tokens
        index += 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def prepare_text_chunk_density(
    text: str,
    max_tokens: int = 2500,
    overlap_tokens: float | None = None,
    density_keywords: Set[str] | None = None,
    density_threshold: int = 25,
    density_exponent: float = 1.0,
) -> List[str]:
    """
    Split text into sentence-aligned chunks with density-aware sizing.

    Passages containing more high-signal keywords are assigned a smaller
    effective chunk size so detail-heavy sections are summarized with tighter
    local context. Overlap is retained between adjacent chunks to reduce loss at
    chunk boundaries.

    :param text: Input text
    :param max_tokens: Approximate token budget per chunk
    :param overlap_tokens: Optional overlap token count; if omitted, overlap is
        derived dynamically from the current chunk size
    :param density_keywords: Keywords that increase content density
    :param density_threshold: Density threshold before chunk shrinking begins
    :param density_exponent: How aggressively chunk size shrinks as density rises
    :return: List of density-aware sentence-aligned chunks
    """
    if estimate_token_count(text) <= max_tokens:
        return [text]

    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    keyword_pattern = None
    if density_keywords:
        escaped_keywords = [re.escape(keyword) for keyword in density_keywords]
        keyword_pattern = re.compile(
            rf"(?<!\w)({'|'.join(escaped_keywords)})(?!\w)",
            re.IGNORECASE,
        )

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0
    accumulated_density = 0

    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        sentence_density = len(keyword_pattern.findall(sentence)) if keyword_pattern else 0

        projected_tokens = current_tokens + sentence_tokens
        projected_density = accumulated_density + sentence_density

        if projected_density == 0:
            scale = 1.0
        else:
            scale = (density_threshold / projected_density) ** density_exponent
            scale = max(scale, 0.1)

        effective_max = max(1, int(max_tokens * scale))

        # Flush before appending the next sentence so chunk boundaries remain
        # stable and the overflow sentence is only introduced once.
        if current_chunk and projected_tokens > effective_max:
            chunks.append(" ".join(current_chunk))

            overlap_target = (
                min(int(current_tokens * 0.1), 600)
                if overlap_tokens is None
                else int(overlap_tokens)
            )

            overlap_chunk: List[str] = []
            overlap_seen = 0

            for existing_sentence in reversed(current_chunk):
                overlap_chunk.insert(0, existing_sentence)
                overlap_seen += estimate_token_count(existing_sentence)
                if overlap_seen >= overlap_target:
                    break

            current_chunk = overlap_chunk
            current_tokens = sum(
                estimate_token_count(existing_sentence)
                for existing_sentence in current_chunk
            )
            accumulated_density = sum(
                len(keyword_pattern.findall(existing_sentence)) if keyword_pattern else 0
                for existing_sentence in current_chunk
            )

        current_chunk.append(sentence)
        current_tokens += sentence_tokens
        accumulated_density += sentence_density

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_text(
    text: str,
    chunk_style: str,
    max_tokens: int,
    density_keywords: Set[str] | None = None,
    density_threshold: int = 25,
) -> List[str]:
    """
    Dispatch text chunking by the configured strategy.

    This function keeps chunk-strategy selection centralized so summary-stage
    orchestration code can stay focused on pass logic rather than chunker
    implementation details.

    :param text: Input text
    :param chunk_style: Chunking strategy: ``word``, ``sentence``, or ``density``
    :param max_tokens: Approximate token budget per chunk
    :param density_keywords: Optional density keywords used by density chunking
    :param density_threshold: Density threshold used by density chunking
    :return: List of text chunks
    """
    if chunk_style == "density":
        return prepare_text_chunk_density(
            text=text,
            max_tokens=max_tokens,
            density_keywords=density_keywords or set(),
            density_threshold=density_threshold,
        )

    if chunk_style == "sentence":
        return prepare_text_chunk_sentence(
            text=text,
            max_tokens=max_tokens,
        )

    return prepare_text_chunk_word(
        text=text,
        max_tokens=max_tokens,
    )