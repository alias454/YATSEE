"""
Meeting classification helpers for YATSEE intelligence jobs.
"""

from __future__ import annotations

import requests

from yatsee.intel.llm import llm_generate_text


def classify_transcript(
    *,
    session: requests.Session,
    llm_provider: str,
    llm_provider_url: str,
    llm_api_key: str | None,
    model: str,
    prompt: str,
    num_ctx: int,
    allowed_labels: set[str],
    llm_allow_remote: bool = False,
    llm_allow_insecure_http: bool = False,
    llm_allow_custom_executable: bool = False,
) -> str:
    """
    Classify a transcript into one of the configured meeting labels.

    The model response is normalized to lowercase and matched against the
    allowed label set using simple substring checks. If classification is not
    usable or no allowed label is found, the function returns ``general``.

    :param session: Shared requests session
    :param llm_provider: Configured provider name
    :param llm_provider_url: Provider base URL or CLI executable target
    :param llm_api_key: Optional provider API key
    :param model: Model name used for classification
    :param prompt: Classification prompt text
    :param num_ctx: Requested context window
    :param allowed_labels: Set of accepted classification labels
    :param llm_allow_remote: Whether remote non-local targets are allowed for local HTTP providers
    :param llm_allow_insecure_http: Whether plain HTTP is allowed for hosted providers
    :param llm_allow_custom_executable: Whether custom CLI executable targets are allowed
    :return: Matched lowercase label or ``general`` if classification is unavailable or unmatched
    :raises ProviderError: If provider execution fails
    """
    if not allowed_labels or not prompt:
        return "general"

    raw_response = llm_generate_text(
        session=session,
        provider_name=llm_provider,
        llm_provider_url=llm_provider_url,
        api_key=llm_api_key,
        model=model,
        prompt=prompt,
        num_ctx=num_ctx,
        max_output_tokens=64,
        allow_remote=llm_allow_remote,
        allow_insecure_http=llm_allow_insecure_http,
        allow_custom_executable=llm_allow_custom_executable,
    ).strip().lower()

    for allowed_label in sorted(allowed_labels, key=len, reverse=True):
        normalized_label = allowed_label.lower()
        if normalized_label in raw_response:
            return normalized_label

    return "general"