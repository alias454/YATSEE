"""
Security validation helpers for YATSEE providers.
"""

from __future__ import annotations

import ipaddress
import os
import re
from urllib.parse import urlparse

from .base import ProviderConfigError


_CLI_PROVIDERS = {"codex_cli"}

_SAFE_EXECUTABLE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_model_name(model: str) -> None:
    """
    Validate a provider model name.

    :param model: Model name to validate
    :raises ProviderConfigError: If the model name is missing or contains invalid characters
    """
    if not model or not model.strip():
        raise ProviderConfigError("LLM model name must be specified.")

    if any(ord(ch) < 32 for ch in model):
        raise ProviderConfigError("LLM model name contains invalid control characters.")


def validate_provider_target(
    *,
    provider_name: str,
    target: str,
    allow_remote: bool = False,
    allow_insecure_http: bool = False,
    allow_custom_executable: bool = False,
) -> None:
    """
    Validate a provider target before execution.

    This function applies security policy to either an HTTP(S) provider target
    or a CLI executable target. For network targets, transport security and
    locality are validated independently:
    - plain HTTP requires explicit opt-in
    - any off-box target requires explicit opt-in

    :param provider_name: Provider identifier
    :param target: URL or executable target
    :param allow_remote: Whether to allow any off-box provider target
    :param allow_insecure_http: Whether to allow plain HTTP provider targets
    :param allow_custom_executable: Whether to allow custom executable targets for CLI providers
    :raises ProviderConfigError: If the target is missing, malformed, or disallowed
    """
    normalized_provider = provider_name.strip().lower() if provider_name else ""
    normalized_target = target.strip() if target else ""

    if not normalized_provider:
        raise ProviderConfigError("LLM provider name must be specified.")

    if not normalized_target:
        raise ProviderConfigError("LLM provider target must be specified.")

    if normalized_provider in _CLI_PROVIDERS:
        _validate_cli_target(
            executable=normalized_target,
            allow_custom_executable=allow_custom_executable,
        )
        return

    parsed = urlparse(normalized_target)
    scheme = (parsed.scheme or "").lower()
    hostname = parsed.hostname

    if scheme not in {"http", "https"}:
        raise ProviderConfigError(
            f"Provider target for '{normalized_provider}' must use http or https."
        )

    if not hostname:
        raise ProviderConfigError(
            f"Provider target for '{normalized_provider}' must include a hostname."
        )

    # Treat transport security separately from locality. Any plain HTTP target
    # requires explicit opt-in, regardless of provider type.
    if scheme == "http" and not allow_insecure_http:
        raise ProviderConfigError(
            f"Provider '{normalized_provider}' target '{normalized_target}' uses insecure HTTP. "
            "Set llm_allow_insecure_http=true to allow plain HTTP targets."
        )

    # Treat locality strictly: only loopback is local by default. Any off-box
    # host, including private LAN ranges and container/service addresses,
    # requires explicit opt-in.
    is_local = _is_local_host(hostname)
    if not is_local and not allow_remote:
        raise ProviderConfigError(
            f"Provider '{normalized_provider}' target '{normalized_target}' is off-box. "
            "Set llm_allow_remote=true to allow remote targets."
        )


def _validate_cli_target(*, executable: str, allow_custom_executable: bool) -> None:
    """
    Validate a CLI executable target.

    When custom executable targets are not allowed, only the default `codex`
    executable is accepted. When custom executables are allowed, the target must
    still be constrained to a safe basename or an absolute path.

    :param executable: Executable name or path
    :param allow_custom_executable: Whether custom executables are allowed
    :raises ProviderConfigError: If the executable target is invalid or disallowed
    """
    if any(ch in executable for ch in ("\x00", "\n", "\r")):
        raise ProviderConfigError("CLI executable target contains invalid characters.")

    if not allow_custom_executable:
        if executable != "codex":
            raise ProviderConfigError(
                "Custom CLI executables are disabled. Use the default 'codex' target or set "
                "llm_allow_custom_executable=true."
            )
        return

    if executable == "codex":
        return

    if os.path.isabs(executable):
        return

    if _SAFE_EXECUTABLE_RE.fullmatch(executable):
        return

    raise ProviderConfigError(
        "CLI executable target must be 'codex', a safe executable basename, or an absolute path."
    )


def _is_local_host(hostname: str) -> bool:
    """
    Determine whether a hostname resolves to a loopback-only target.

    This helper treats only localhost and loopback IP addresses as local.
    Private LAN addresses, link-local addresses, container hostnames, and other
    non-loopback targets are intentionally treated as remote and require
    explicit opt-in through configuration.

    :param hostname: Hostname from a parsed URL
    :return: True if the hostname is loopback-only, otherwise False
    """
    normalized = hostname.strip().lower()

    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        ip = ipaddress.ip_address(normalized)
        return ip.is_loopback
    except ValueError:
        return False