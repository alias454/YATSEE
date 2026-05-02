"""
Registry and lookup helpers for YATSEE LLM providers.
"""

from __future__ import annotations

from . import anthropic, codex_cli, llamacpp, ollama, openai
from .base import ProviderConfigError, TextGenerationProvider


_PROVIDER_MAP: dict[str, TextGenerationProvider] = {
    "ollama": ollama,
    "llamacpp": llamacpp,
    "openai": openai,
    "anthropic": anthropic,
    "codex_cli": codex_cli,
}


def get_provider(provider_name: str) -> TextGenerationProvider:
    """
    Resolve a provider module by configured provider name.

    The registry maps normalized provider identifiers to provider adapter
    modules that implement the shared text-generation contract.

    :param provider_name: Provider identifier
    :return: Provider module implementing `generate_text()`
    :raises ProviderConfigError: If the provider name is missing or unsupported
    """
    normalized = provider_name.strip().lower() if provider_name else ""
    if not normalized:
        raise ProviderConfigError("LLM provider name must be specified.")

    provider = _PROVIDER_MAP.get(normalized)
    if provider is None:
        supported = ", ".join(sorted(_PROVIDER_MAP))
        raise ProviderConfigError(
            f"Unknown LLM provider '{provider_name}'. Supported providers: {supported}"
        )

    if not hasattr(provider, "generate_text"):
        raise ProviderConfigError(
            f"Provider '{provider_name}' does not implement generate_text()."
        )

    return provider