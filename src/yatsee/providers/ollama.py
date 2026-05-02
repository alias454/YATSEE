"""
Ollama provider transport.
"""

from __future__ import annotations

import json

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
    temperature: float = 0.1,
    timeout: int = 300,
) -> str:
    """
    Generate text using Ollama's `/api/generate` endpoint.

    :param session: Shared HTTP session
    :param base_url: Ollama base URL
    :param model: Model name
    :param prompt: Prompt text
    :param api_key: Optional API key, typically unused for local Ollama
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
            provider_name="ollama",
            model=model,
            text=prompt,
        )
        max_output_tokens = max(1, min(2048, num_ctx - prompt_tokens - 1))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": max_output_tokens,
            "temperature": temperature,
            "seed": 42
        }
    }

    try:
        response = session.post(
            f"{normalized_url}/api/generate",
            json=payload,
            headers=headers,
            stream=True,
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ProviderRequestError(f"Ollama request failed: {exc}") from exc

    parts: list[str] = []

    try:
        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))
            parts.append(data.get("response", ""))

            if data.get("done", False):
                break
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderParseError(f"Failed to parse Ollama streaming response: {exc}") from exc

    output = "".join(parts)
    if not output:
        raise ProviderParseError("Ollama returned empty output.")

    return output