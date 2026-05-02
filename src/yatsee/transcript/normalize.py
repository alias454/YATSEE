"""
Transcript normalization stage for YATSEE.

This module ports the existing transcript normalization behavior behind reusable
functions so the new CLI can invoke it without embedding stage logic directly.

Behavior intentionally mirrors the current standalone script:
- input from normalized TXT-style transcript artifacts or direct override
- mechanical cleanup and sentence shaping
- optional spaCy sentence splitting
- entity-specific replacement rules
- output to normalized/ or direct override
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from yatsee.core.config import load_entity_config, load_global_config
from yatsee.core.discovery import discover_files
from yatsee.core.errors import ConfigError, ValidationError

SUPPORTED_INPUT_EXTENSIONS = (".txt",)


def load_text(path: str) -> str:
    """
    Read a UTF-8 transcript text file.

    :param path: Path to the input text file
    :return: Raw file contents
    :raises ConfigError: If the file cannot be read
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise ConfigError(f"Failed to read transcript '{path}': {exc}") from exc


def write_text(path: str, content: str) -> None:
    """
    Write normalized transcript text to disk.

    :param path: Output file path
    :param content: Normalized text content
    :raises ConfigError: If writing fails
    """
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        raise ConfigError(f"Failed to write normalized transcript '{path}': {exc}") from exc


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph breaks.

    :param text: Raw transcript text
    :return: Text with normalized spacing
    """
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def normalize_common_patterns(text: str) -> str:
    """
    Perform mechanical cleanup for common transcript issues.

    This stage deliberately focuses on low-risk mechanical cleanup rather than
    semantic rewriting.

    :param text: Input transcript text
    :return: Mechanically cleaned transcript text
    """
    # Normalize curly quotes and long dashes into simpler ASCII-ish forms
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", "-").replace("–", "-")

    # Collapse repeated punctuation and malformed spacing around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s\n])", r"\1 \2", text)

    # Normalize spacing around parentheses
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # Collapse accidental repeated words like "the the"
    text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def apply_replacements(text: str, replacements: Dict[str, str]) -> str:
    """
    Apply entity-specific replacement rules.

    Replacements are applied longest-first so more specific phrases win before
    shorter partial matches.

    :param text: Input transcript text
    :param replacements: Mapping of incorrect text to corrected text
    :return: Updated transcript text
    """
    if not replacements:
        return text

    updated = text
    for bad, good in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if not bad:
            continue
        updated = re.sub(re.escape(bad), good, updated, flags=re.IGNORECASE)

    return updated


def sentence_split_basic(text: str) -> List[str]:
    """
    Perform simple regex-based sentence splitting.

    This is the non-spaCy fallback path used when the user disables spaCy or
    when the model is unavailable.

    :param text: Input transcript text
    :return: List of sentence-like strings
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [part.strip() for part in parts if part.strip()]


def sentence_split_spacy(text: str, model_name: str) -> List[str]:
    """
    Perform sentence splitting with spaCy.

    :param text: Input transcript text
    :param model_name: spaCy model name
    :return: List of sentence strings
    :raises RuntimeError: If spaCy or the model cannot be loaded
    """
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is not installed") from exc

    try:
        nlp = spacy.load(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load spaCy model '{model_name}': {exc}") from exc

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def merge_fragment_lines(lines: List[str]) -> List[str]:
    """
    Merge obviously incomplete fragment lines into the previous line when safe.

    This helps reduce ugly sentence-per-line output where a fragment like
    'for the city council' ends up isolated.

    :param lines: Sentence or line list
    :return: Smoothed line list
    """
    if not lines:
        return lines

    merged: List[str] = []
    for line in lines:
        if merged and line and line[0].islower():
            merged[-1] = f"{merged[-1]} {line}".strip()
        else:
            merged.append(line)
    return merged


def deep_clean_text(text: str) -> str:
    """
    Apply slightly more aggressive cleanup for obvious filler and bracket noise.

    This should remain conservative. It is not intended to become semantic
    rewriting.

    :param text: Input transcript text
    :return: Deep-cleaned text
    """
    text = re.sub(r"\[(?:music|applause|laughter|noise)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:music|applause|laughter|noise)\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:um|uh|erm)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_output(lines: List[str], preserve_paragraphs: bool = False) -> str:
    """
    Format normalized lines back into output text.

    :param lines: Normalized line list
    :param preserve_paragraphs: Preserve blank lines between paragraphs if True
    :return: Final output text
    """
    if preserve_paragraphs:
        return "\n".join(lines).strip() + "\n"
    return "\n".join(line for line in lines if line.strip()).strip() + "\n"


def resolve_normalize_paths(
    global_config_path: str,
    entity: str | None,
    input_path: str | None,
    output_dir: str | None,
    model_override: str | None,
) -> Dict[str, Any]:
    """
    Resolve config and filesystem paths for transcript normalization.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Optional entity handle
    :param input_path: Optional transcript input override
    :param output_dir: Optional output override
    :param model_override: Optional transcription model override used for input path resolution
    :return: Dictionary containing resolved config and paths
    :raises ValidationError: If required arguments are missing
    """
    entity_cfg: Dict[str, Any] = {}
    global_cfg = load_global_config(global_config_path)

    if entity:
        entity_cfg = load_entity_config(global_cfg, entity)
    else:
        if not input_path or not output_dir:
            raise ValidationError(
                "Without --entity, both --input-path and --output-dir must be defined"
            )

    data_path = entity_cfg.get("data_path")
    transcription_model = (
        model_override
        or entity_cfg.get("transcription_model")
        or global_cfg.get("system", {}).get("default_transcription_model", "medium")
    )

    resolved_input = input_path or os.path.join(data_path, f"transcripts_{transcription_model}")
    resolved_output = output_dir or os.path.join(data_path, "normalized")

    sentence_model = entity_cfg.get(
        "sentence_model",
        global_cfg.get("system", {}).get("default_sentence_model", "en_core_web_md"),
    )

    return {
        "global_cfg": global_cfg,
        "entity_cfg": entity_cfg,
        "input_path": resolved_input,
        "output_dir": resolved_output,
        "sentence_model": sentence_model,
    }


def run_normalize_stage(
    global_config_path: str,
    entity: str | None = None,
    input_path: str | None = None,
    output_dir: str | None = None,
    model_override: str | None = None,
    no_spacy: bool = False,
    deep_clean: bool = False,
    preserve_paragraphs: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Run the transcript normalization stage.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Optional entity handle
    :param input_path: Optional transcript input override
    :param output_dir: Optional output override
    :param no_spacy: Disable spaCy sentence splitting
    :param deep_clean: Enable slightly more aggressive cleanup
    :param preserve_paragraphs: Preserve paragraph spacing
    :param force: Overwrite existing outputs
    :return: Summary dictionary describing stage results
    """
    resolved = resolve_normalize_paths(
        global_config_path=global_config_path,
        entity=entity,
        input_path=input_path,
        output_dir=output_dir,
        model_override=model_override
    )

    entity_cfg = resolved["entity_cfg"]
    transcript_input = resolved["input_path"]
    output_directory = resolved["output_dir"]
    sentence_model = resolved["sentence_model"]

    input_files = discover_files(transcript_input, SUPPORTED_INPUT_EXTENSIONS)
    if not input_files:
        return {
            "input_path": transcript_input,
            "output_dir": output_directory,
            "sentence_model": sentence_model,
            "discovered": 0,
            "written": 0,
            "skipped": 0,
            "messages": [f"No transcript text files found at {transcript_input}"],
        }

    os.makedirs(output_directory, exist_ok=True)

    replacements = entity_cfg.get("replacements", {}) if isinstance(entity_cfg.get("replacements", {}), dict) else {}

    written = 0
    skipped = 0
    messages: List[str] = []

    for src_path in input_files:
        base_name = os.path.basename(src_path)
        out_path = os.path.join(output_directory, base_name)

        if os.path.abspath(src_path) == os.path.abspath(out_path) and not force:
            skipped += 1
            messages.append(f"Skipped in-place file without --force: {src_path}")
            continue

        if os.path.exists(out_path) and not force:
            skipped += 1
            messages.append(f"Skipped existing normalized transcript: {out_path}")
            continue

        raw_text = load_text(src_path)
        text = normalize_whitespace(raw_text)
        text = normalize_common_patterns(text)

        if deep_clean:
            text = deep_clean_text(text)

        text = apply_replacements(text, replacements)

        try:
            if no_spacy:
                lines = sentence_split_basic(text)
            else:
                lines = sentence_split_spacy(text, sentence_model)
        except RuntimeError as exc:
            messages.append(f"spaCy unavailable for '{src_path}', falling back to regex split: {exc}")
            lines = sentence_split_basic(text)

        lines = merge_fragment_lines(lines)
        final_text = format_output(lines, preserve_paragraphs=preserve_paragraphs)

        write_text(out_path, final_text)
        written += 1
        messages.append(f"Wrote normalized transcript: {out_path}")

    return {
        "input_path": transcript_input,
        "output_dir": output_directory,
        "sentence_model": sentence_model,
        "discovered": len(input_files),
        "written": written,
        "skipped": skipped,
        "messages": messages,
    }