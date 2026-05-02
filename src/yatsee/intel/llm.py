"""
Provider-neutral LLM helper functions for YATSEE intelligence jobs.
"""

from __future__ import annotations

import requests

from yatsee.providers import get_provider
from yatsee.providers.security import validate_model_name, validate_provider_target


def llm_generate_text(
    *,
    session: requests.Session,
    provider_name: str,
    llm_provider_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    num_ctx: int = 8192,
    max_output_tokens: int | None = None,
    allow_remote: bool = False,
    allow_insecure_http: bool = False,
    allow_custom_executable: bool = False,
) -> str:
    """
    Generate text through the configured LLM provider.

    The request is validated before dispatch so model names and provider targets
    are checked consistently at the shared entry point used by the intelligence
    stage.

    :param session: Shared requests session
    :param provider_name: Configured provider identifier
    :param llm_provider_url: Provider base URL or CLI executable target
    :param api_key: Optional provider API key
    :param model: Provider model name
    :param prompt: Prompt text
    :param num_ctx: Requested context window
    :param max_output_tokens: Optional explicit output token cap
    :param allow_remote: Whether remote non-local targets are allowed for local HTTP providers
    :param allow_insecure_http: Whether plain HTTP is allowed for hosted providers
    :param allow_custom_executable: Whether custom CLI executable targets are allowed
    :return: Generated text
    :raises ValueError: If the prompt is empty
    :raises ProviderConfigError: If the model or provider target is invalid
    :raises ProviderError: If provider resolution or execution fails
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty")

    validate_model_name(model)
    validate_provider_target(
        provider_name=provider_name,
        target=llm_provider_url,
        allow_remote=allow_remote,
        allow_insecure_http=allow_insecure_http,
        allow_custom_executable=allow_custom_executable,
    )

    provider = get_provider(provider_name)
    return provider.generate_text(
        session=session,
        base_url=llm_provider_url,
        model=model,
        prompt=prompt,
        api_key=api_key,
        num_ctx=num_ctx,
        max_output_tokens=max_output_tokens,
        temperature=0.2,
        timeout=300,
    )