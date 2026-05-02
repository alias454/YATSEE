"""
Output writing helpers for YATSEE intelligence jobs.
"""

from __future__ import annotations

import os
from pathlib import Path

import mdformat
import yaml


def _safe_path_component(value: str) -> str:
    """
    Normalize a filesystem path component into a single safe segment.

    This helper is used when filenames or directory names may be influenced by
    transcript metadata, configuration, or model output. It prevents traversal
    sequences, absolute-path tricks, and malformed control characters from
    affecting where files are written.

    :param value: Raw path component
    :return: Safe single path segment
    :raises ValueError: If the component is empty after normalization
    """
    normalized = (value or "").replace("\\", "/").strip().strip(".")
    normalized = normalized.replace("/", "_").replace("\x00", "")

    if not normalized or normalized in {".", ".."}:
        raise ValueError("Invalid filesystem path component")

    return normalized


def _build_contained_path(root_dir: str, *parts: str) -> str:
    """
    Build a path that is guaranteed to remain within the given root directory.

    All write destinations should be derived through this helper so output files
    cannot escape the intended artifact directory even if an upstream value is
    malformed or unsafe.

    :param root_dir: Intended root directory
    :param parts: Path components to append under the root
    :return: Safe absolute path as a string
    :raises ValueError: If the resolved path escapes the intended root
    """
    root_path = Path(root_dir).resolve()
    candidate = root_path.joinpath(*parts).resolve()

    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside the output directory: {candidate}"
        ) from exc

    return str(candidate)


def write_summary_file(
    summary: str,
    basename: str,
    output_dir: str,
    fmt: str = "yaml",
) -> str:
    """
    Write a summary artifact to disk in Markdown or YAML format.

    Markdown output includes a generated title header based on the source
    basename. YAML output stores the generated text under a single `summary`
    key so downstream readers can consume a stable structure.

    :param summary: Generated summary content
    :param basename: Base filename without extension
    :param output_dir: Directory where the summary file should be written
    :param fmt: Output format, either ``markdown`` or ``yaml``
    :return: Full path to the written summary file
    :raises OSError: If the directory cannot be created or the file cannot be written
    :raises ValueError: If an unsupported format is requested
    """
    if fmt not in {"markdown", "yaml"}:
        raise ValueError(f"Unsupported summary output format: {fmt}")

    extension = "md" if fmt == "markdown" else "yaml"
    safe_basename = _safe_path_component(basename)
    output_path = _build_contained_path(
        output_dir,
        f"{safe_basename}.summary.{extension}",
    )

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        if fmt == "markdown":
            markdown_content = f"# Summary: {basename}\n\n{summary.strip()}\n"
            formatted_markdown = mdformat.text(markdown_content, options={"number": True})
            handle.write(formatted_markdown)
        else:
            yaml.safe_dump({"summary": summary}, handle, sort_keys=False)

    return output_path


def write_chunk_file(
    chunk_text: str,
    output_dir: str,
    meeting_type: str,
    base_name: str,
    pass_num: int,
    chunk_id: int,
) -> str:
    """
    Write one summarized chunk to a structured debug output path.

    Chunk files are grouped by transcript basename, detected meeting type, and
    summarization pass so intermediate model output can be inspected without
    changing final summary behavior.

    :param chunk_text: Summarized chunk text to persist
    :param output_dir: Root output directory for intelligence artifacts
    :param meeting_type: Detected or assigned meeting type label
    :param base_name: Source transcript basename
    :param pass_num: Summarization pass number
    :param chunk_id: One-based chunk number within the pass
    :return: Full path to the written chunk file
    :raises OSError: If the directory cannot be created or the file cannot be written
    """
    safe_base_name = _safe_path_component(base_name)
    safe_meeting_type = _safe_path_component(meeting_type)

    chunk_dir = _build_contained_path(
        output_dir,
        "chunks",
        safe_base_name,
        safe_meeting_type,
        f"pass_{pass_num}",
    )
    chunk_filename = f"{safe_base_name}_part{chunk_id:02d}.txt"
    chunk_path = _build_contained_path(chunk_dir, chunk_filename)

    os.makedirs(chunk_dir, exist_ok=True)

    with open(chunk_path, "w", encoding="utf-8") as handle:
        handle.write(chunk_text)

    return chunk_path