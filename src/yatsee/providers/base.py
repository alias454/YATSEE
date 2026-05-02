"""
Common types and errors for YATSEE text-generation providers.
"""

from __future__ import annotations

from typing import Protocol

import requests


class ProviderError(RuntimeError):
    """
    Base exception for provider failures.
    """


class ProviderConfigError(ProviderError):
    """
    Raised when provider configuration is invalid.
    """


class ProviderRequestError(ProviderError):
    """
    Raised when a provider request or subprocess execution fails.
    """


class ProviderParseError(ProviderError):
    """
    Raised when a provider response cannot be parsed.
    """


class TextGenerationProvider(Protocol):
    """
    Protocol for text-generation providers.

    Providers implement a shared general-purpose text generation interface so
    the calling application can remain agnostic to the underlying transport.
    The prompt may be used for summarization, classification, extraction, or
    any other text-generation task handled by the intelligence stage.
    """

    def generate_text(
        self,
        *,
        session: requests.Session,
        base_url: str,
        model: str,
        prompt: str,
        api_key: str | None = None,
        num_ctx: int = 8192,
        max_output_tokens: int | None = None,
        temperature: float = 0.2,
        timeout: int = 300,
    ) -> str:
        """
        Generate text from the configured provider.

        :param session: Shared HTTP session
        :param base_url: Provider base URL or executable path surrogate
        :param model: Model name or identifier
        :param prompt: Prompt text
        :param api_key: Optional provider API key
        :param num_ctx: Requested context window
        :param max_output_tokens: Optional explicit output token cap
        :param temperature: Sampling temperature
        :param timeout: Request timeout in seconds
        :return: Generated text
        :raises ProviderError: On provider failure
        """
        ...