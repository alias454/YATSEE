"""
Codex CLI provider transport.

This provider executes the local Codex CLI as a subprocess and captures stdout.
It intentionally does not pretend to be an HTTP provider.
"""

from __future__ import annotations

import subprocess

import requests

from .base import (
    ProviderConfigError,
    ProviderParseError,
    ProviderRequestError,
)


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
    Generate text using the local Codex CLI.

    :param session: Shared HTTP session, unused for this provider
    :param base_url: Codex CLI executable path or command name
    :param model: Model name
    :param prompt: Prompt text
    :param api_key: Unused for CLI auth flows
    :param num_ctx: Requested context window, unused unless future CLI flags support it
    :param max_output_tokens: Optional output token cap
    :param temperature: Sampling temperature
    :param timeout: Command timeout in seconds
    :return: Generated text
    :raises ProviderConfigError: If executable or model is missing
    :raises ProviderRequestError: On subprocess failure
    :raises ProviderParseError: On empty output
    """
    del session
    del api_key
    del num_ctx

    executable = base_url.strip() if base_url else "codex"

    if not model:
        raise ProviderConfigError("Codex CLI provider requires a model name.")

    cmd = [
        executable,
        "exec",
        "--model",
        model,
        "--output-last-message",
    ]

    if max_output_tokens is not None:
        cmd.extend(["--max-output-tokens", str(max_output_tokens)])

    cmd.extend(["--temperature", str(temperature)])
    cmd.append(prompt)

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ProviderConfigError(
            f"Codex CLI executable '{executable}' was not found."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ProviderRequestError(f"Codex CLI timed out after {timeout}s.") from exc
    except OSError as exc:
        raise ProviderRequestError(f"Failed to execute Codex CLI: {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise ProviderRequestError(
            f"Codex CLI exited with code {completed.returncode}: {stderr}"
        )

    output = completed.stdout.strip()
    if not output:
        raise ProviderParseError("Codex CLI returned empty output.")

    return output