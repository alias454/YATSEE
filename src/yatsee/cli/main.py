"""
Primary CLI entrypoint for YATSEE.

This CLI surface supports:
- configuration management commands
- source fetch orchestration
- audio format stage
- audio transcription stage
- transcript slice stage
- transcript normalize stage
- intelligence-stage orchestration
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from yatsee.audio.format import run_format_stage
from yatsee.audio.transcribe import run_transcribe_stage
from yatsee.config_tools.entity import (
    add_entity,
    list_entities as list_registered_entities,
    purge_entity,
    remove_entity,
)
from yatsee.config_tools.resolve import resolve_config
from yatsee.config_tools.scaffold import build_entity_structure
from yatsee.config_tools.validate import validate_entity_config, validate_global_config
from yatsee.core.config import GLOBAL_CONFIG_PATH, load_global_config
from yatsee.core.errors import YatseeError
from yatsee.intel.runner import run_intelligence_stage
from yatsee.source.fetch import run_source_fetch_for_entity
from yatsee.transcript.normalize import run_normalize_stage
from yatsee.transcript.slice import run_slice_stage


def _parse_inputs(raw: str | None) -> List[str]:
    """
    Parse a comma-separated input list from the CLI.

    :param raw: Raw comma-separated string or None
    :return: Normalized list of lowercase input names
    """
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """
    Build the root CLI parser.

    :return: Fully configured argparse parser
    """
    parser = argparse.ArgumentParser(
        prog="yatsee",
        description="YATSEE command-line interface",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=GLOBAL_CONFIG_PATH,
        help="Path to global yatsee.toml",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ----------------------------
    # config
    # ----------------------------
    config_parser = subparsers.add_parser("config", help="Configuration management commands")
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    entity_parser = config_subparsers.add_parser("entity", help="Manage entity registry")
    entity_subparsers = entity_parser.add_subparsers(dest="entity_command")

    entity_list_parser = entity_subparsers.add_parser("list", help="List registered entities")
    entity_list_parser.set_defaults(handler=handle_config_entity_list)

    entity_add_parser = entity_subparsers.add_parser("add", help="Add an entity to the global registry")
    entity_add_parser.add_argument("--display-name", required=True, help="Human-friendly entity name")
    entity_add_parser.add_argument("--entity", required=True, help="Entity handle")
    entity_add_parser.add_argument("--base", default="", help="Optional namespace/base value")
    entity_add_parser.add_argument("--inputs", default="", help="Comma-separated input types, e.g. youtube")
    entity_add_parser.add_argument("--no-create-dir", action="store_true", help="Do not create the top-level entity directory")
    entity_add_parser.set_defaults(handler=handle_config_entity_add)

    entity_remove_parser = entity_subparsers.add_parser("remove", help="Remove an entity from the global registry only")
    entity_remove_parser.add_argument("--entity", required=True, help="Entity handle")
    entity_remove_parser.set_defaults(handler=handle_config_entity_remove)

    entity_purge_parser = entity_subparsers.add_parser("purge", help="Purge an entity from both the registry and the local filesystem")
    entity_purge_parser.add_argument("--entity", required=True, help="Entity handle")
    entity_purge_parser.add_argument("--dry-run", action="store_true", help="Preview what would be removed without making changes")
    entity_purge_parser.set_defaults(handler=handle_config_entity_purge)

    init_parser = config_subparsers.add_parser("init", help="Create local entity config scaffolds")
    init_parser.add_argument("--entity", help="Initialize only one entity")
    init_parser.set_defaults(handler=handle_config_init)

    validate_parser = config_subparsers.add_parser("validate", help="Validate global and entity config")
    validate_parser.add_argument("--entity", help="Validate a specific entity as well")
    validate_parser.set_defaults(handler=handle_config_validate)

    resolve_parser = config_subparsers.add_parser("resolve", help="Print resolved runtime config")
    resolve_parser.add_argument("--entity", help="Resolve a specific entity")
    resolve_parser.set_defaults(handler=handle_config_resolve)

    # ----------------------------
    # source
    # ----------------------------
    source_parser = subparsers.add_parser("source", help="Source acquisition commands")
    source_subparsers = source_parser.add_subparsers(dest="source_command")

    fetch_parser = source_subparsers.add_parser("fetch", help="Fetch source artifacts for an entity")
    fetch_parser.add_argument("-e", "--entity", required=True, help="Entity handle")
    fetch_parser.add_argument("--source", help="Optional specific source adapter, e.g. youtube")
    fetch_parser.add_argument("-o", "--output-dir", help="Optional download/output directory override")
    fetch_parser.add_argument("--date-after", help="Only include items after YYYYMMDD")
    fetch_parser.add_argument("--date-before", help="Only include items before YYYYMMDD")
    fetch_parser.add_argument("--make-playlist", action="store_true", help="Rebuild playlist cache and exit")
    fetch_parser.set_defaults(handler=handle_source_fetch)

    # ----------------------------
    # audio
    # ----------------------------
    audio_parser = subparsers.add_parser("audio", help="Audio processing commands")
    audio_subparsers = audio_parser.add_subparsers(dest="audio_command")

    format_parser = audio_subparsers.add_parser("format", help="Normalize media into mono 16kHz audio")
    format_parser.add_argument("-e", "--entity", help="Entity handle to process")
    format_parser.add_argument("-i", "--input-dir", help="Direct override path to media input")
    format_parser.add_argument("-o", "--output-dir", help="Directory to save normalized audio")
    format_parser.add_argument("--format", default="flac", choices=["wav", "flac"], help="Output audio format")
    format_parser.add_argument("--create-chunks", action="store_true", help="Split output audio into chunks")
    format_parser.add_argument("--chunk-duration", type=int, default=600, help="Chunk duration in seconds")
    format_parser.add_argument("--chunk-overlap", type=int, default=2, help="Chunk overlap in seconds")
    format_parser.add_argument("--dry-run", action="store_true", help="Preview actions without changing files")
    format_parser.add_argument("--force", action="store_true", help="Reprocess files even if already converted")
    format_parser.set_defaults(handler=handle_audio_format)

    transcribe_parser = audio_subparsers.add_parser("transcribe", help="Transcribe normalized audio to VTT")
    transcribe_parser.add_argument("-e", "--entity", help="Entity handle to process")
    transcribe_parser.add_argument("-i", "--audio-input", help="Audio file or directory")
    transcribe_parser.add_argument("-o", "--output-dir", help="Directory to save transcripts")
    transcribe_parser.add_argument("-g", "--get-chunks", action="store_true", help="Transcribe using audio chunk files")
    transcribe_parser.add_argument("-m", "--model", help="Whisper model size override")
    transcribe_parser.add_argument("--faster", action="store_true", help="Use faster-whisper if installed")
    transcribe_parser.add_argument("-l", "--lang", default="en", help="Language code or 'auto'")
    transcribe_parser.add_argument("-d", "--device", choices=["auto", "cuda", "cpu", "mps"], default="auto", help="Device for model execution")
    transcribe_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    transcribe_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    transcribe_parser.set_defaults(handler=handle_audio_transcribe)

    # ----------------------------
    # transcript
    # ----------------------------
    transcript_parser = subparsers.add_parser("transcript", help="Transcript preparation commands")
    transcript_subparsers = transcript_parser.add_subparsers(dest="transcript_command")

    slice_parser = transcript_subparsers.add_parser("slice", help="Slice VTT transcripts into TXT and JSONL segments")
    slice_parser.add_argument("-e", "--entity", help="Entity handle to process")
    slice_parser.add_argument("-i", "--vtt-input", help="VTT file or directory")
    slice_parser.add_argument("-o", "--output-dir", help="Directory to save transcript outputs")
    slice_parser.add_argument("-m", "--model", help="SentenceTransformer model override")
    slice_parser.add_argument("-g", "--gen-embed", action="store_true", help="Generate JSONL with embeddings")
    slice_parser.add_argument("--max-window", type=float, default=90.0, help="Hard upper limit on segment length")
    slice_parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    slice_parser.add_argument("-d", "--device", choices=["auto", "cuda", "cpu", "mps"], default="auto", help="Device for embedding execution")
    slice_parser.set_defaults(handler=handle_transcript_slice)

    normalize_parser = transcript_subparsers.add_parser("normalize", help="Normalize transcript text")
    normalize_parser.add_argument("-e", "--entity", help="Entity handle to process")
    normalize_parser.add_argument("-i", "--input-path", help="TXT file or directory")
    normalize_parser.add_argument("-o", "--output-dir", help="Directory to save normalized output")
    normalize_parser.add_argument("-m", "--model", help="Transcription model suffix for input path resolution")
    normalize_parser.add_argument("--no-spacy", action="store_true", help="Disable spaCy sentence splitting")
    normalize_parser.add_argument("--deep-clean", action="store_true", help="Enable slightly more aggressive cleanup")
    normalize_parser.add_argument("--preserve-paragraphs", action="store_true", help="Preserve paragraph spacing")
    normalize_parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    normalize_parser.set_defaults(handler=handle_transcript_normalize)

    # ----------------------------
    # intel
    # ----------------------------
    intel_parser = subparsers.add_parser("intel", help="Intelligence-stage commands")
    intel_subparsers = intel_parser.add_subparsers(dest="intel_command")

    intel_run_parser = intel_subparsers.add_parser("run", help="Run multi-pass transcript summarization")
    intel_run_parser.add_argument("-e", "--entity", help="Entity handle to process")
    intel_run_parser.add_argument("-i", "--txt-input", help="Path to a transcript file or directory (.txt)")
    intel_run_parser.add_argument("-o", "--output-dir", help="Directory to save final summaries")
    intel_run_parser.add_argument("-m", "--model", help="LLM model override")
    intel_run_parser.add_argument("--llm-provider", help="LLM provider override (e.g. ollama, llamacpp, openai, anthropic, codex_cli)")
    intel_run_parser.add_argument("--llm-provider-url", help="LLM provider URL or executable target override")
    intel_run_parser.add_argument("--llm-api-key", help="LLM API key override")
    intel_run_parser.add_argument("--show-pricing", action="store_true", help="Estimate reference pricing for the run")
    intel_run_parser.add_argument("--no-show-pricing", action="store_true", help="Disable reference pricing even if enabled in config")
    intel_run_parser.add_argument("--pricing-provider", help="Reference provider for pricing (e.g. openai, anthropic)")
    intel_run_parser.add_argument("--pricing-model", help="Reference model for pricing (e.g. gpt-5.4, claude-sonnet-4)")
    intel_run_parser.add_argument("-f", "--output-format", choices=["markdown", "yaml"], default="markdown", help="Summary output format")
    intel_run_parser.add_argument("-j", "--job-profile", choices=["civic", "research"], default="civic", help="Prompt workflow family")
    intel_run_parser.add_argument("-s", "--chunk-style", choices=["word", "sentence", "density"], default="word", help="Chunk boundary method")
    intel_run_parser.add_argument("-w", "--max-words", type=int, help="Approximate word count threshold for chunking")
    intel_run_parser.add_argument("-t", "--max-tokens", type=int, help="Approximate max tokens per chunk")
    intel_run_parser.add_argument("-p", "--max-pass", type=int, default=3, help="Maximum summarization passes")
    intel_run_parser.add_argument("-d", "--disable-auto-classification", action="store_true", help="Disable automatic meeting classification")
    intel_run_parser.add_argument("--first-prompt", help="Prompt ID for first pass")
    intel_run_parser.add_argument("--second-prompt", help="Prompt ID for multi-pass chunk summaries")
    intel_run_parser.add_argument("--final-prompt", help="Prompt ID for final summary pass")
    intel_run_parser.add_argument("--context", default="", help="Optional human-readable meeting context")
    intel_run_parser.add_argument("--print-prompts", action="store_true", help="Print prompt templates and exit")
    intel_run_parser.add_argument("--enable-chunk-writer", action="store_true", help="Write intermediate chunk summaries for debugging")
    intel_run_parser.set_defaults(handler=handle_intel_run)

    return parser


def handle_config_entity_list(args: argparse.Namespace) -> int:
    global_cfg = load_global_config(args.config)
    entities = list_registered_entities(global_cfg)

    if not entities:
        print("No entities defined.")
        return 0

    print("Registered entities:")
    for entity in entities:
        print(f"- {entity}")
    return 0


def handle_config_entity_add(args: argparse.Namespace) -> int:
    global_cfg = load_global_config(args.config)
    result = add_entity(
        global_cfg=global_cfg,
        config_path=args.config,
        display_name=args.display_name,
        entity=args.entity,
        base=args.base,
        inputs=_parse_inputs(args.inputs),
        create_dir=not args.no_create_dir,
    )
    print(result["message"])
    if result["entity_dir"]:
        print(f"Entity directory: {result['entity_dir']}")
    return 0


def handle_config_entity_remove(args: argparse.Namespace) -> int:
    global_cfg = load_global_config(args.config)
    result = remove_entity(global_cfg=global_cfg, config_path=args.config, entity=args.entity)
    print(result["message"])
    return 0


def handle_config_entity_purge(args: argparse.Namespace) -> int:
    global_cfg = load_global_config(args.config)
    result = purge_entity(
        global_cfg=global_cfg,
        config_path=args.config,
        entity=args.entity,
        dry_run=args.dry_run,
    )

    print(result["message"])
    print(f"Entity: {result['entity']}")
    print(f"Registry entry exists: {result['registry_entry_exists']}")
    print(f"Entity directory: {result['entity_dir']}")
    print(f"Entity directory exists: {result['entity_dir_exists']}")
    print(f"Local config exists: {result['config_exists']}")
    print(f"Contained files: {result['file_count']}")
    print(f"Contained subdirectories: {result['dir_count']}")
    return 0


def handle_config_init(args: argparse.Namespace) -> int:
    global_cfg = load_global_config(args.config)
    messages = build_entity_structure(global_cfg=global_cfg, entity=args.entity)
    for message in messages:
        print(message)
    return 0


def handle_config_validate(args: argparse.Namespace) -> int:
    global_cfg = load_global_config(args.config)

    for message in validate_global_config(global_cfg):
        print(message)

    if args.entity:
        for message in validate_entity_config(global_cfg, args.entity):
            print(message)

    print("Validation passed.")
    return 0


def handle_config_resolve(args: argparse.Namespace) -> int:
    resolved = resolve_config(global_config_path=args.config, entity=args.entity)
    print(json.dumps(resolved, indent=2, sort_keys=True))
    return 0


def handle_source_fetch(args: argparse.Namespace) -> int:
    result = run_source_fetch_for_entity(
        global_config_path=args.config,
        entity=args.entity,
        source_name=args.source,
        output_dir=args.output_dir,
        date_after=args.date_after,
        date_before=args.date_before,
        make_playlist=args.make_playlist,
    )

    print(f"Entity: {result['entity']}")
    print(f"Sources run: {', '.join(result['sources_run'])}")

    for adapter_result in result["results"]:
        print(f"- Source type: {adapter_result['source_type']}")
        print(f"  Output directory: {adapter_result['downloads_dir']}")
        print(f"  Discovered: {adapter_result['discovered']}")
        print(f"  Downloaded: {adapter_result['downloaded']}")
        print(f"  Skipped: {adapter_result['skipped']}")

    for message in result["messages"]:
        print(f"- {message}")

    return 0


def handle_audio_format(args: argparse.Namespace) -> int:
    result = run_format_stage(
        global_config_path=args.config,
        entity=args.entity,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_format=args.format,
        create_chunks=args.create_chunks,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
        dry_run=args.dry_run,
        force=args.force,
    )

    print(f"Input directory: {result['input_dir']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Discovered files: {result['discovered']}")
    print(f"Processed files: {result['processed']}")
    print(f"Skipped files: {result['skipped']}")
    print(f"Chunks created: {result['chunked']}")

    for message in result["messages"]:
        print(f"- {message}")

    return 0


def handle_audio_transcribe(args: argparse.Namespace) -> int:
    result = run_transcribe_stage(
        global_config_path=args.config,
        entity=args.entity,
        audio_input=args.audio_input,
        output_dir=args.output_dir,
        get_chunks=args.get_chunks,
        model_override=args.model,
        faster=args.faster,
        language=args.lang,
        device_arg=args.device,
        verbose=args.verbose,
        quiet=args.quiet,
    )

    print(f"Audio input: {result['audio_input']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Model: {result['model']}")
    print(f"Backend: {result['backend']}")
    print(f"Device: {result['device']}")
    print(f"Discovered files: {result['discovered']}")
    print(f"Processed files: {result['processed']}")
    print(f"Skipped files: {result['skipped']}")

    for message in result["messages"]:
        print(f"- {message}")

    return 0


def handle_transcript_slice(args: argparse.Namespace) -> int:
    result = run_slice_stage(
        global_config_path=args.config,
        entity=args.entity,
        vtt_input=args.vtt_input,
        output_dir=args.output_dir,
        model_override=args.model,
        gen_embed=args.gen_embed,
        max_window=args.max_window,
        force=args.force,
        device_arg=args.device,
    )

    print(f"VTT input: {result['vtt_input']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Embedding model: {result['embedding_model']}")
    print(f"Device: {result['device']}")
    print(f"Discovered files: {result['discovered']}")
    print(f"TXT written: {result['txt_written']}")
    print(f"JSONL written: {result['jsonl_written']}")

    for message in result["messages"]:
        print(f"- {message}")

    return 0


def handle_transcript_normalize(args: argparse.Namespace) -> int:
    result = run_normalize_stage(
        global_config_path=args.config,
        entity=args.entity,
        input_path=args.input_path,
        output_dir=args.output_dir,
        model_override=args.model,
        no_spacy=args.no_spacy,
        deep_clean=args.deep_clean,
        preserve_paragraphs=args.preserve_paragraphs,
        force=args.force,
    )

    print(f"Input path: {result['input_path']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Sentence model: {result['sentence_model']}")
    print(f"Discovered files: {result['discovered']}")
    print(f"Written files: {result['written']}")
    print(f"Skipped files: {result['skipped']}")

    for message in result["messages"]:
        print(f"- {message}")

    return 0


def handle_intel_run(args: argparse.Namespace) -> int:
    result = run_intelligence_stage(args)

    if result.get("mode") == "print_prompts":
        print(f"Job profile: {result['job_profile']}")
        print(f"Prompt file: {result['prompt_file'] or 'inline fallback prompts'}")
        print(f"Used fallback prompts: {result['used_fallback_prompts']}")
        print()

        for key, prompt_text in result["prompts"].items():
            print(f"=== {key} ===")
            print(prompt_text)
            print()
        return 0

    print(f"Input directory: {result['input_dir']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Provider: {result.get('llm_provider', '(unknown)')}")
    print(f"Model: {result['model']}")
    print(f"Processed transcripts: {result['processed']}")
    print(f"Prompt file: {result['prompt_file'] or 'inline fallback prompts'}")
    print(f"Used fallback prompts: {result['used_fallback_prompts']}")
    print(f"Show pricing: {result.get('show_pricing', False)}")

    pricing_provider = result.get("pricing_provider")
    pricing_model = result.get("pricing_model")
    if pricing_provider or pricing_model:
        print(f"Pricing reference: {pricing_provider or '(default provider)'} / {pricing_model or '(default model)'}")

    for item in result["results"]:
        print(f"- {item['base_name']}")
        print(f"  Meeting type: {item['meeting_type']}")
        print(f"  Output: {item['output_path'] or '(no file written)'}")
        print(
            f"  Tokens: in={item['input_tokens']} "
            f"out={item['output_tokens']} total={item['total_tokens']}"
        )

        pricing = item.get("pricing", {})
        if pricing.get("enabled"):
            ref_provider = pricing.get("reference_provider") or "(unknown)"
            ref_model = pricing.get("reference_model") or "(unknown)"
            estimated_cost = pricing.get("estimated_cost")

            if estimated_cost is None:
                print(f"  Reference pricing: unavailable for {ref_provider}/{ref_model}")
            else:
                print(
                    f"  Reference pricing ({ref_provider}/{ref_model}): "
                    f"${estimated_cost:.4f}"
                )

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0

    try:
        return handler(args)
    except YatseeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())