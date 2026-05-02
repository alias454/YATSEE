"""
Prompt loading and routing helpers for YATSEE intelligence jobs.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import toml

from yatsee.core.errors import ConfigError


DEFAULT_PROMPTS: Dict[str, str] = {
    "overview": (
        "You are an assistant that produces a high-level summary of a meeting transcript.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Summarize the key points, main discussions, and outcomes in clear, concise language."
    ),
    "action_items": (
        "You are an assistant that extracts actionable items, motions, and decisions from a meeting transcript.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "List only the actionable items or decisions in plain language. Skip commentary or filler text."
    ),
}

DEFAULT_PROMPT_ROUTER: Dict[str, Dict[str, str]] = {
    "general": {
        "first": "overview",
        "multi": "overview",
        "final": "action_items",
    }
}


def validate_prompt(
    prompt_name: str | None,
    label: str,
    available_prompts: Dict[str, str],
) -> None:
    """
    Validate that a referenced prompt identifier exists in the loaded prompt set.

    This helper is used for CLI overrides and other direct prompt references so
    invalid identifiers fail with a clear message before generation begins.

    :param prompt_name: Requested prompt name
    :param label: Human-readable label for errors
    :param available_prompts: Prompt dictionary
    :raises ValueError: If the prompt name is invalid
    """
    if not prompt_name:
        return

    if prompt_name not in available_prompts:
        valid_keys = ", ".join(sorted(available_prompts.keys()))
        raise ValueError(
            f"Invalid {label} '{prompt_name}'. Available prompts: {valid_keys}"
        )


def _validate_prompt_bundle(bundle: Dict[str, Any]) -> None:
    """
    Validate the structural requirements of a prompt bundle.

    A valid bundle must contain at least one prompt, a ``general`` route, and
    routing entries that reference prompt identifiers that actually exist in the
    prompt map.

    :param bundle: Loaded prompt bundle dictionary
    :raises ConfigError: If the bundle structure is incomplete or invalid
    """
    prompts = bundle.get("prompts", {})
    prompt_router = bundle.get("prompt_router", {})

    if not prompts:
        raise ConfigError("Prompt bundle does not define any prompts")

    general_route = prompt_router.get("general")
    if not general_route:
        raise ConfigError("Prompt bundle must define a 'general' prompt route")

    for route_name, route in prompt_router.items():
        for required_key in ("first", "multi", "final"):
            if required_key not in route:
                raise ConfigError(
                    f"Prompt route '{route_name}' is missing required key '{required_key}'"
                )

            prompt_id = route[required_key]
            if prompt_id not in prompts:
                raise ConfigError(
                    f"Prompt route '{route_name}.{required_key}' references unknown prompt '{prompt_id}'"
                )


def load_prompt_bundle(entity_cfg: Dict[str, Any], job_profile: str) -> Dict[str, Any]:
    """
    Load prompts and routing metadata for an intelligence job profile.

    Resolution checks for an entity-local prompt file first, then the project
    default prompt file, and finally falls back to the built-in prompt bundle.
    The resulting structure is validated before being returned.

    :param entity_cfg: Merged entity configuration
    :param job_profile: Job profile such as civic or research
    :return: Prompt bundle dictionary
    :raises ConfigError: If prompt loading or validation fails
    """
    prompt_types_path = ""

    entity_prompt_file = os.path.join(
        entity_cfg.get("data_path", ""),
        "prompts",
        job_profile,
        "prompts.toml",
    )
    default_prompt_file = os.path.join("prompts", job_profile, "prompts.toml")

    if os.path.isfile(entity_prompt_file):
        prompt_types_path = entity_prompt_file
    elif os.path.isfile(default_prompt_file):
        prompt_types_path = default_prompt_file

    if not prompt_types_path:
        bundle = {
            "prompts": DEFAULT_PROMPTS,
            "prompt_router": DEFAULT_PROMPT_ROUTER,
            "classifier_prompt": "",
            "classifier_types": {},
            "density_keywords": {},
            "path": "",
            "fallback": True,
        }
        _validate_prompt_bundle(bundle)
        return bundle

    try:
        raw_prompts = toml.load(prompt_types_path)
    except Exception as exc:
        raise ConfigError(
            f"Failed to parse prompt file '{prompt_types_path}': {exc}"
        ) from exc

    raw_prompt_map = raw_prompts.get("prompts", {})
    if not isinstance(raw_prompt_map, dict):
        raise ConfigError(
            f"Prompt file '{prompt_types_path}' has invalid [prompts] structure"
        )

    prompt_texts: Dict[str, str] = {}
    for prompt_id, prompt_entry in raw_prompt_map.items():
        if not isinstance(prompt_entry, dict):
            raise ConfigError(
                f"Prompt '{prompt_id}' in '{prompt_types_path}' must be a table/object"
            )

        prompt_text = prompt_entry.get("text")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ConfigError(
                f"Prompt '{prompt_id}' in '{prompt_types_path}' must define a non-empty text value"
            )

        prompt_texts[prompt_id] = prompt_text

    prompt_router = raw_prompts.get("prompt_router", {})
    if not isinstance(prompt_router, dict):
        raise ConfigError(
            f"Prompt file '{prompt_types_path}' has invalid [prompt_router] structure"
        )

    classifier_prompt_entry = raw_prompts.get("classifier_prompt", {})
    if classifier_prompt_entry and not isinstance(classifier_prompt_entry, dict):
        raise ConfigError(
            f"Prompt file '{prompt_types_path}' has invalid [classifier_prompt] structure"
        )

    classifier_prompt = ""
    if classifier_prompt_entry:
        classifier_prompt = classifier_prompt_entry.get("text", "")
        if classifier_prompt and not isinstance(classifier_prompt, str):
            raise ConfigError(
                f"Prompt file '{prompt_types_path}' classifier_prompt.text must be a string"
            )

    classifier_types = raw_prompts.get("classifier_types", {})
    if classifier_types and not isinstance(classifier_types, dict):
        raise ConfigError(
            f"Prompt file '{prompt_types_path}' has invalid [classifier_types] structure"
        )

    density_keywords = raw_prompts.get("density_keywords", {})
    if density_keywords and not isinstance(density_keywords, dict):
        raise ConfigError(
            f"Prompt file '{prompt_types_path}' has invalid [density_keywords] structure"
        )

    bundle = {
        "prompts": prompt_texts,
        "prompt_router": prompt_router,
        "classifier_prompt": classifier_prompt,
        "classifier_types": classifier_types,
        "density_keywords": density_keywords,
        "path": prompt_types_path,
        "fallback": False,
    }

    _validate_prompt_bundle(bundle)
    return bundle


def resolve_prompt_ids(
    meeting_type: str,
    prompt_lookup: Dict[str, Dict[str, str]],
    first_override: str | None,
    multi_override: str | None,
    final_override: str | None,
) -> Dict[str, str]:
    """
    Resolve the prompt identifiers for the current summarization run.

    Routing first attempts to use the detected meeting type. If no specific
    route exists, the ``general`` route is used. Individual pass prompts may be
    overridden by runtime arguments.

    :param meeting_type: Detected meeting type
    :param prompt_lookup: Prompt router mapping
    :param first_override: Override for the first pass prompt
    :param multi_override: Override for the intermediate pass prompt
    :param final_override: Override for the final synthesis prompt
    :return: Prompt ID mapping
    :raises ConfigError: If prompt routing is incomplete
    """
    fallback_route = prompt_lookup.get("general")
    if not fallback_route:
        raise ConfigError("Prompt routing is missing required 'general' entry")

    route = prompt_lookup.get(meeting_type, fallback_route)

    for required_key in ("first", "multi", "final"):
        if required_key not in route:
            raise ConfigError(
                f"Prompt route for '{meeting_type}' is missing required key '{required_key}'"
            )

    return {
        "first": first_override or route["first"],
        "multi": multi_override or route["multi"],
        "final": final_override or route["final"],
    }