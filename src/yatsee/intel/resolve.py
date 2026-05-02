"""
Resolution helpers for YATSEE intelligence jobs.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from yatsee.core.config import load_entity_config, load_global_config
from yatsee.core.discovery import discover_files
from yatsee.core.errors import ValidationError


def resolve_intel_paths(global_config_path: str, args: Any) -> Dict[str, Any]:
    """
    Resolve configuration, model selection, and runtime input/output paths.

    This function loads global configuration, optionally merges entity-specific
    configuration, selects the summarization model, discovers transcript inputs,
    and derives output and token settings for the intelligence stage.

    :param global_config_path: Path to the global ``yatsee.toml`` file
    :param args: CLI/runtime args namespace
    :return: Dictionary of resolved runtime paths and model settings
    :raises ValidationError: If required configuration or runtime inputs are invalid
    """
    global_cfg = load_global_config(global_config_path)
    entity_cfg: Dict[str, Any] = {}

    if args.entity:
        entity_cfg = load_entity_config(global_cfg, args.entity)
    elif not args.txt_input:
        raise ValidationError("Without --entity, --txt-input must be defined")

    model = (
        args.model
        or entity_cfg.get("summarization_model")
        or global_cfg.get("system", {}).get("default_summarization_model")
    )
    if not model:
        raise ValidationError("No summarization model specified")

    model_map = global_cfg.get("models", {})
    model_key = next((key for key in model_map if key.lower() == model.lower()), None)
    if not model_key:
        raise ValidationError(
            f"Unsupported model '{model}'. Supported: {', '.join(model_map.keys())}"
        )

    input_dir = args.txt_input or os.path.join(entity_cfg.get("data_path"), "normalized")
    if not input_dir:
        raise ValidationError("No input transcript path resolved")

    file_list = discover_files(input_dir, (".txt",))
    file_list = [path for path in file_list if not path.lower().endswith("punct.txt")]
    if not file_list:
        raise ValidationError(f"No transcript files found in: {input_dir}")

    if args.output_dir:
        output_dir = args.output_dir
    elif entity_cfg:
        output_dir = os.path.join(
            entity_cfg["data_path"],
            model_map[model_key]["append_dir"],
        )
    else:
        raise ValidationError(
            "Without --entity, --output-dir must be defined"
        )

    os.makedirs(output_dir, exist_ok=True)

    if args.max_words and not args.max_tokens:
        max_tokens = int(args.max_words / 0.75)
    else:
        max_tokens = args.max_tokens or model_map[model_key].get("max_tokens", 2500)

    num_ctx = model_map[model_key].get("num_ctx", 8192)
    if max_tokens >= num_ctx:
        raise ValidationError(
            f"Chunk size ({max_tokens}) exceeds or collides with model context ({num_ctx})"
        )

    return {
        "global_cfg": global_cfg,
        "entity_cfg": entity_cfg,
        "model": model,
        "model_key": model_key,
        "model_cfg": model_map[model_key],
        "input_dir": input_dir,
        "file_list": file_list,
        "output_dir": output_dir,
        "max_tokens": max_tokens,
        "num_ctx": num_ctx,
    }


def build_intel_run_config(args: Any, max_tokens: int) -> Dict[str, Any]:
    """
    Build a normalized runtime configuration for intelligence-stage execution.

    This separates downstream processing from raw argparse attribute names and
    groups the knobs needed during per-transcript summarization.

    :param args: CLI/runtime args namespace
    :param max_tokens: Resolved max token budget
    :return: Normalized runtime configuration dictionary
    """
    return {
        "job_profile": args.job_profile,
        "chunk_style": args.chunk_style,
        "max_tokens": max_tokens,
        "max_pass": args.max_pass,
        "output_format": args.output_format,
        "disable_auto_classification": args.disable_auto_classification,
        "first_prompt": args.first_prompt,
        "second_prompt": args.second_prompt,
        "final_prompt": args.final_prompt,
        "context": args.context,
        "enable_chunk_writer": args.enable_chunk_writer,
    }