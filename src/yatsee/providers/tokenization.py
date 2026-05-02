"""
Token estimation helpers for provider-aware text generation.

This module provides a generic fallback estimator and a stable seam for adding
provider-specific tokenizers later.
"""

from __future__ import annotations

import re


def estimate_token_count(text: str) -> int:
    """
    Estimate token count using a simple character-based heuristic.

    This is provider-agnostic and intentionally lightweight. It is suitable for
    chunk sizing, rough context budgeting, and approximate pricing when exact
    tokenizer usage is unavailable.

    :param text: Input text
    :return: Estimated token count
    """
    normalized = re.sub(r"\s+", " ", text)
    return max(1, len(normalized) // 4)


def estimate_provider_tokens(
    *,
    provider_name: str,
    model: str,
    text: str,
) -> int:
    """
    Estimate token count for a provider/model pair.

    This currently falls back to the generic heuristic for all providers. It
    exists as the stable abstraction point for future provider-specific
    tokenization support.

    :param provider_name: Provider identifier
    :param model: Model name
    :param text: Input text
    :return: Estimated token count
    """
    # Provider/model-specific tokenizers can be added here later.
    del provider_name
    del model
    return estimate_token_count(text)