"""
Top-level intelligence stage runner for YATSEE.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

import requests

from yatsee.core.errors import ConfigError
from yatsee.intel.prompts import load_prompt_bundle, validate_prompt
from yatsee.intel.resolve import build_intel_run_config, resolve_intel_paths
from yatsee.intel.summary import summarize_one_transcript

logger = logging.getLogger(__name__)


def extract_context_from_filename(filename: str, extra_note: str | None = None) -> str:
    """
    Derive a lightweight human-readable context string from a transcript filename.

    The extracted value is used to provide prompts with a simple meeting label
    when no richer metadata object is available.

    :param filename: Transcript file path
    :param extra_note: Optional additional context note
    :return: Human-readable context string
    """
    base = os.path.basename(filename)
    base = re.sub(r"^[\w\-]+?\.", "", base)
    base = base.replace(".txt", "")
    base = base.replace("_", " ").replace("-", " ")

    meeting_match = re.match(r"(.*?) (\d{1,2})[ -](\d{1,2})[ -](\d{2,4})", base)
    if meeting_match:
        kind = meeting_match.group(1).strip().title()
        month = int(meeting_match.group(2))
        day = int(meeting_match.group(3))
        year = int(meeting_match.group(4))
        if year < 100:
            year += 2000

        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        context = f"{kind} - {date_str}"
    else:
        context = base.title()

    if extra_note:
        context += f" ({extra_note.strip()})"

    return context


def run_intelligence_stage(args: Any) -> Dict[str, Any]:
    """
    Run the intelligence stage across one or more normalized transcript files.

    This function resolves runtime configuration, loads prompt metadata, applies
    provider and pricing settings, and dispatches each transcript to the
    per-file summarization pipeline.

    :param args: Parsed CLI/runtime arguments
    :return: Intelligence-stage result metadata
    :raises ConfigError: If required provider configuration is missing
    """
    resolved = resolve_intel_paths(args.config, args)
    global_cfg = resolved["global_cfg"]
    entity_cfg = resolved["entity_cfg"]
    system_cfg = global_cfg.get("system", {})

    prompt_bundle = load_prompt_bundle(entity_cfg, args.job_profile)

    if args.print_prompts:
        return {
            "mode": "print_prompts",
            "job_profile": args.job_profile,
            "prompt_file": prompt_bundle["path"],
            "used_fallback_prompts": prompt_bundle["fallback"],
            "prompts": prompt_bundle["prompts"],
        }

    if args.disable_auto_classification:
        validate_prompt(args.first_prompt, "first-prompt", prompt_bundle["prompts"])
        validate_prompt(args.second_prompt, "second-prompt", prompt_bundle["prompts"])
        validate_prompt(args.final_prompt, "final-prompt", prompt_bundle["prompts"])

    llm_provider = str(
        args.llm_provider or system_cfg.get("llm_provider", "ollama")
    ).strip().lower()
    llm_provider_url = args.llm_provider_url or system_cfg.get("llm_provider_url")
    llm_api_key = (
        args.llm_api_key
        if args.llm_api_key is not None
        else system_cfg.get("llm_api_key")
    )

    llm_allow_remote = bool(system_cfg.get("llm_allow_remote", False))
    llm_allow_insecure_http = bool(system_cfg.get("llm_allow_insecure_http", False))
    llm_allow_custom_executable = bool(
        system_cfg.get("llm_allow_custom_executable", False)
    )

    show_pricing = bool(system_cfg.get("show_pricing", False))
    if getattr(args, "show_pricing", False):
        show_pricing = True
    if getattr(args, "no_show_pricing", False):
        show_pricing = False

    pricing_provider = (
        args.pricing_provider
        if getattr(args, "pricing_provider", None) is not None
        else system_cfg.get("pricing_provider")
    )
    pricing_model = (
        args.pricing_model
        if getattr(args, "pricing_model", None) is not None
        else system_cfg.get("pricing_model")
    )

    if not llm_provider:
        raise ConfigError("No llm_provider configured in [system]")

    if not llm_provider_url:
        raise ConfigError("No llm_provider_url configured in [system]")

    run_cfg = build_intel_run_config(args, resolved["max_tokens"])
    run_cfg["show_pricing"] = show_pricing
    run_cfg["pricing_provider"] = pricing_provider
    run_cfg["pricing_model"] = pricing_model

    results: List[Dict[str, Any]] = []

    # A single session keeps connection reuse efficient and avoids ambient
    # environment proxy settings silently altering request paths.
    with requests.Session() as session:
        session.trust_env = False
        session.headers.update({"Content-Type": "application/json"})

        for file_path in resolved["file_list"]:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            context = extract_context_from_filename(file_path, run_cfg["context"])

            with open(file_path, "r", encoding="utf-8") as handle:
                transcript = handle.read()

            result = summarize_one_transcript(
                session=session,
                llm_provider=llm_provider,
                llm_provider_url=llm_provider_url,
                llm_api_key=llm_api_key,
                llm_allow_remote=llm_allow_remote,
                llm_allow_insecure_http=llm_allow_insecure_http,
                llm_allow_custom_executable=llm_allow_custom_executable,
                model=resolved["model"],
                num_ctx=resolved["num_ctx"],
                transcript=transcript,
                base_name=base_name,
                context=context,
                prompt_bundle=prompt_bundle,
                entity_cfg=entity_cfg,
                run_cfg=run_cfg,
                output_dir=resolved["output_dir"],
            )
            results.append(result)

    return {
        "mode": "run",
        "input_dir": resolved["input_dir"],
        "output_dir": resolved["output_dir"],
        "llm_provider": llm_provider,
        "model": resolved["model"],
        "show_pricing": show_pricing,
        "pricing_provider": pricing_provider,
        "pricing_model": pricing_model,
        "processed": len(results),
        "results": results,
        "prompt_file": prompt_bundle["path"],
        "used_fallback_prompts": prompt_bundle["fallback"],
    }