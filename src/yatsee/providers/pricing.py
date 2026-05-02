"""
Provider-aware pricing helpers for YATSEE.

This module estimates reference API-equivalent cost from token counts.
It does not represent what local providers actually cost to run.
"""

from __future__ import annotations

from typing import Any


# Curated reference pricing for selected remote API models.
# This is not an exhaustive catalog of all provider models or modalities.
# Update from official provider pricing pages when adding or revising entries.
PRICING_TABLE: dict[str, dict[str, dict[str, float]]] = {
    "openai": {
        "gpt-5.4": {"input_per_million": 2.50, "output_per_million": 15.00},
        "gpt-4.1": {"input_per_million": 2.00, "output_per_million": 8.00},
    },
    "anthropic": {
        "claude-mythos": {"input_per_million": 25.00, "output_per_million": 125.00},
        "claude-opus": {"input_per_million": 5.00, "output_per_million": 25.00},
        "claude-sonnet": {"input_per_million": 3.00, "output_per_million": 15.00},
        "claude-haiku": {"input_per_million": 1.00, "output_per_million": 5.00},
    },
}


def get_pricing(provider_name: str, model: str) -> dict[str, float] | None:
    """
    Return per-million-token pricing metadata for a provider/model.

    Local and CLI-backed providers do not have direct API token pricing, so
    this function returns `None` for those backends. For supported remote
    providers, model matching first attempts an exact normalized lookup and
    then falls back to substring matching against curated entries.

    :param provider_name: Provider identifier
    :param model: Model name
    :return: Pricing metadata or None if unavailable
    """
    provider = provider_name.strip().lower() if provider_name else ""
    normalized_model = model.strip().lower() if model else ""

    if provider in {"ollama", "llamacpp", "codex_cli"}:
        return None

    provider_models = PRICING_TABLE.get(provider)
    if not provider_models:
        return None

    if normalized_model in provider_models:
        return provider_models[normalized_model]

    for known_model, pricing in provider_models.items():
        if known_model in normalized_model:
            return pricing

    return None


def estimate_cost(
    *,
    provider_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """
    Estimate token-based cost in USD.

    This function uses per-million-token pricing metadata from the curated
    pricing table. If pricing is unavailable for the selected provider/model,
    the function returns `None` instead of inventing a value.

    :param provider_name: Provider identifier
    :param model: Model name
    :param input_tokens: Input token count
    :param output_tokens: Output token count
    :return: Estimated USD cost or None if pricing is unavailable
    """
    pricing = get_pricing(provider_name, model)
    if pricing is None:
        return None

    input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
    return input_cost + output_cost


def build_pricing_summary(
    *,
    show_pricing: bool,
    actual_provider: str,
    actual_model: str,
    pricing_provider: str | None,
    pricing_model: str | None,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    """
    Build a normalized pricing summary for reporting.

    This function separates the provider/model actually used for generation
    from the provider/model used for reference pricing. That allows local runs
    to report an estimated hosted API-equivalent cost without pretending the
    hosted provider actually performed the generation.

    :param show_pricing: Whether pricing estimation is enabled
    :param actual_provider: Provider actually used for generation
    :param actual_model: Model actually used for generation
    :param pricing_provider: Optional override provider for reference pricing
    :param pricing_model: Optional override model for reference pricing
    :param input_tokens: Input token count
    :param output_tokens: Output token count
    :return: Structured pricing summary for reporting
    """
    enabled = bool(show_pricing)

    reference_provider = (pricing_provider or actual_provider or "").strip().lower()
    reference_model = (pricing_model or actual_model or "").strip()

    estimated_cost = None
    if enabled and reference_provider and reference_model:
        estimated_cost = estimate_cost(
            provider_name=reference_provider,
            model=reference_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return {
        "enabled": enabled,
        "actual_provider": actual_provider,
        "actual_model": actual_model,
        "reference_provider": reference_provider or None,
        "reference_model": reference_model or None,
        "estimated_cost": estimated_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }