"""
llama.cpp provider transport.

This implementation targets a llama.cpp server exposing an OpenAI-compatible
`/v1/completions` endpoint.
"""

from __future__ import annotations

import requests

from .base import ProviderParseError, ProviderRequestError
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
    Generate text using a llama.cpp OpenAI-compatible endpoint.

    :param session: Shared HTTP session
    :param base_url: llama.cpp base URL
    :param model: Model name
    :param prompt: Prompt text
    :param api_key: Optional API key
    :param num_ctx: Requested context window
    :param max_output_tokens: Optional explicit output token cap
    :param temperature: Sampling temperature
    :param timeout: Request timeout in seconds
    :return: Generated text
    :raises ProviderRequestError: On HTTP failures
    :raises ProviderParseError: On response parsing failures
    """
    normalized_url = base_url.rstrip("/")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if max_output_tokens is None:
        prompt_tokens = estimate_provider_tokens(
            provider_name="llamacpp",
            model=model,
            text=prompt,
        )
        max_output_tokens = max(1, min(2048, num_ctx - prompt_tokens - 1))

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
    }

    try:
        response = session.post(
            f"{normalized_url}/v1/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ProviderRequestError(f"llama.cpp request failed: {exc}") from exc

    try:
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ProviderParseError("llama.cpp response contained no choices.")

        text = choices[0].get("text")
        if not isinstance(text, str):
            raise ProviderParseError("llama.cpp response missing choices[0].text.")

        return text
    except ValueError as exc:
        raise ProviderParseError(f"Failed to parse llama.cpp JSON response: {exc}") from exc