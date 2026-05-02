"""
Anthropic provider transport.

This implementation targets Anthropic's messages API using a single user message.
"""

from __future__ import annotations

import requests

from .base import (
    ProviderConfigError,
    ProviderParseError,
    ProviderRequestError,
)
from .tokenization import estimate_provider_tokens


def generate_text(
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
    Generate text using Anthropic's messages endpoint.

    :param session: Shared HTTP session
    :param base_url: Anthropic base URL
    :param model: Model name
    :param prompt: Prompt text
    :param api_key: Required API key
    :param num_ctx: Requested context window, used only for output budgeting
    :param max_output_tokens: Optional explicit output token cap
    :param temperature: Sampling temperature
    :param timeout: Request timeout in seconds
    :return: Generated text
    :raises ProviderConfigError: If API key is missing
    :raises ProviderRequestError: On HTTP failures
    :raises ProviderParseError: On response parsing failures
    """
    if not api_key:
        raise ProviderConfigError("Anthropic provider requires an API key.")

    normalized_url = base_url.rstrip("/")

    if max_output_tokens is None:
        prompt_tokens = estimate_provider_tokens(
            provider_name="anthropic",
            model=model,
            text=prompt,
        )
        max_output_tokens = max(1, min(2048, num_ctx - prompt_tokens - 1))

    payload = {
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    try:
        response = session.post(
            f"{normalized_url}/v1/messages",
            json=payload,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ProviderRequestError(f"Anthropic request failed: {exc}") from exc

    try:
        data = response.json()
        content = data.get("content", [])
        if not isinstance(content, list) or not content:
            raise ProviderParseError("Anthropic response contained no content blocks.")

        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                block_text = block.get("text")
                if isinstance(block_text, str):
                    text_parts.append(block_text)

        if not text_parts:
            raise ProviderParseError("Anthropic response contained no text blocks.")

        return "".join(text_parts)
    except ValueError as exc:
        raise ProviderParseError(f"Failed to parse Anthropic JSON response: {exc}") from exc