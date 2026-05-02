"""
OpenAI provider transport.

This implementation uses the Responses API for plain text generation.
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
    Generate text using the OpenAI Responses API.

    :param session: Shared HTTP session
    :param base_url: OpenAI base URL
    :param model: Model name
    :param prompt: Prompt text
    :param api_key: Required API key
    :param num_ctx: Requested context window
    :param max_output_tokens: Optional explicit output token cap
    :param temperature: Sampling temperature
    :param timeout: Request timeout in seconds
    :return: Generated text
    :raises ProviderConfigError: If API key is missing
    :raises ProviderRequestError: On HTTP failures
    :raises ProviderParseError: On response parsing failures
    """
    if not api_key:
        raise ProviderConfigError("OpenAI provider requires an API key.")

    normalized_url = base_url.rstrip("/")

    if max_output_tokens is None:
        prompt_tokens = estimate_provider_tokens(
            provider_name="openai",
            model=model,
            text=prompt,
        )
        max_output_tokens = max(1, min(2048, num_ctx - prompt_tokens - 1))

    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = session.post(
            f"{normalized_url}/v1/responses",
            json=payload,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ProviderRequestError(f"OpenAI request failed: {exc}") from exc

    try:
        data = response.json()

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        output = data.get("output", [])
        text_parts: list[str] = []

        for item in output:
            if not isinstance(item, dict):
                continue

            contents = item.get("content", [])
            if not isinstance(contents, list):
                continue

            for block in contents:
                if not isinstance(block, dict):
                    continue

                if block.get("type") in {"output_text", "text"}:
                    text_value = block.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)

        if text_parts:
            return "".join(text_parts)

        raise ProviderParseError("OpenAI Responses API returned no text output.")

    except ValueError as exc:
        raise ProviderParseError(f"Failed to parse OpenAI JSON response: {exc}") from exc