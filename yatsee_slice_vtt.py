#!/usr/bin/env python3
"""
yatsee_slice_vtt.py

Stage 4a of the YATSEE pipeline: slice WebVTT (.vtt) transcripts into
sentence-aware JSONL segments and optional plain text transcripts.

Inputs:
  - WebVTT files (.vtt) located in 'transcripts_<model>/' under the
    entity data path, or any custom directory via --vtt-input.

Outputs:
  - JSONL segments (.segments.jsonl) with:
      * start_time, end_time, duration
      * text_raw (sentence content)
      * segment_index, source_id, video_id
  - Optional plain text transcripts (.txt) with preserved line breaks.

Key Features:
  - Sentence-aware cue consolidation (merges lines lacking terminal punctuation)
  - Optional max-window segment slicing for uniform segment duration
  - Deterministic placeholder video IDs when real IDs are missing
  - Entity-specific config overrides merged with global yatsee.toml
  - Safe defaults, CLI verbosity control, and robust error handling

Dependencies:
  - Python 3 standard libraries: os, sys, json, hashlib, argparse, re, random, string
  - Third-party: toml, webvtt-py

Example Usage:
  ./yatsee_slice_vtt.py -e city_council --vtt-input ./transcripts --create-txt
  ./yatsee_slice_vtt.py --vtt-input ./audio/transcripts --max-window 30 --force
"""

# Standard library
import argparse
import hashlib
import json
import os
import random
import re
import string
import sys
import textwrap
from typing import Any, Dict, List, Optional

# Third-party imports
import toml
import webvtt


def load_global_config(path: str) -> Dict[str, Any]:
    """
    Load the global YATSEE configuration from a TOML file.

    Raises an exception if the file is missing or invalid.
    Ensures downstream code can rely on a complete global config dictionary.

    :param path: Path to the global TOML configuration file
    :return: Parsed configuration as a dictionary
    :raises FileNotFoundError: If the config file does not exist
    :raises ValueError: If TOML parsing fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global configuration file not found: {path}")
    try:
        return toml.load(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse global config '{path}': {exc}") from exc


def load_entity_config(global_cfg: Dict[str, Any], entity: str) -> Dict[str, Any]:
    """
    Load and merge entity-specific configuration with global defaults.

    Handles merging of reserved keys ("settings", "meta") from local config.
    Ensures a flattened dictionary that the pipeline can consume safely.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle to load
    :return: Merged entity configuration dictionary
    :raises KeyError: If entity is missing from global config
    :raises FileNotFoundError: If local entity config is missing
    """
    reserved_keys = {"settings", "meta"}

    entities_cfg = global_cfg.get("entities", {})
    if entity not in entities_cfg:
        raise KeyError(f"Entity '{entity}' not defined in global config")

    system_cfg = global_cfg.get("system", {})
    root_data_dir = os.path.abspath(system_cfg.get("root_data_dir", "./data"))
    local_path = os.path.join(root_data_dir, entity, "config.toml")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local config for entity '{entity}' not found at: {local_path}")

    local_cfg = toml.load(local_path)

    merged = {
        "entity": entity,
        "root_data_dir": root_data_dir,
        **system_cfg,
        **entities_cfg.get(entity, {}),
    }

    for key, value in local_cfg.items():
        if key in reserved_keys:
            merged.update(value)
        else:
            merged[key] = value
    return merged


def discover_files(input_path: str, supported_exts) -> List[str]:
    """
    Collect files from a directory or single file based on allowed extensions.

    :param input_path: Path to directory or file
    :param supported_exts: Supported file extensions tuple (e.g., ".vtt, .txt")
    :return: Sorted list of valid file paths
    :raises FileNotFoundError: If path does not exist
    :raises ValueError: If single file has unsupported extension
    """
    valid_exts = supported_exts
    files: List[str] = []

    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            full = os.path.join(input_path, f)
            if os.path.isfile(full) and f.lower().endswith(valid_exts):
                files.append(full)
        if not files:
            return [] # No files isn't an error
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(valid_exts):
            files.append(input_path)
        else:
            raise ValueError(f"Unsupported file extension: {os.path.splitext(input_path)[1]}")
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

    return sorted(files)


def read_vtt(vtt_path: str) -> List[Dict[str, Any]]:
    """
    Read a VTT file into a list of cues with absolute timestamps.

    Each cue contains:
    - start: start time in seconds
    - end: end time in seconds
    - text: cleaned caption text

    :param vtt_path: Path to .vtt file
    :return: List of cue dictionaries: {"start", "end", "text"}
    """
    cues = []
    try:
        for cap in webvtt.read(vtt_path):
            cues.append({
                "start": round(cap.start_in_seconds, 3),
                "end": round(cap.end_in_seconds, 3),
                "text": cap.text.strip()
            })
    except (webvtt.errors.MalformedFileError, ValueError, OSError) as exc:
        raise ValueError(f"Failed to read VTT '{vtt_path}': {exc}") from exc
    return cues


def consolidate_sentences(cues: List[Dict[str, Any]], max_window: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Merge VTT cues into sentence-bound segments for JSONL and TXT output.

    - Cues without terminal punctuation are merged with next cues.
    - Preserves line breaks in TXT.
    - Flushes buffer if max_window exceeded (seconds).

    :param cues: List of cue dictionaries from read_vtt
    :param max_window: Optional max duration (seconds) per segment
    :return: List of sentence-aware segments
    """
    segments = []
    buffer = []
    seg_start = None

    end_punct = re.compile(r"[.!?…]+$")

    for cue in cues:
        text = cue["text"].replace("\n", " ").strip()
        if not text:
            continue

        if seg_start is None:
            seg_start = cue["start"]
        buffer.append(text)

        seg_end = cue["end"]
        duration = seg_end - seg_start

        # Flush if sentence ends or max_window exceeded
        if end_punct.search(text) or (max_window and duration >= max_window):
            segments.append({
                "start": seg_start,
                "end": seg_end,
                "text_raw": " ".join(buffer)
            })
            buffer = []
            seg_start = None

    if buffer:
        segments.append({
            "start": seg_start,
            "end": cues[-1]["end"],
            "text_raw": " ".join(buffer)
        })

    return segments


def generate_placeholder_id(base_name: Optional[str] = None) -> str:
    """
    Generate a deterministic or random 11-character placeholder ID.

    Used when a YouTube ID is unavailable.

    :param base_name: Optional seed string for deterministic ID
    :return: 11-character alphanumeric string
    """
    if base_name:
        digest = hashlib.sha256(base_name.encode("utf-8")).hexdigest()
        return ''.join(c for c in digest if c.isalnum())[:11]
    return ''.join(random.choices(string.ascii_letters + string.digits, k=11))


def load_video_id_map(id_map_path: str) -> Dict[str, str]:
    """
    Load mapping from lowercase filenames to real YouTube IDs.

    - Only lines of length 11 considered valid.
    - Supports mapping downloaded videos to VTT files.

    :param id_map_path: Path to .downloaded file
    :return: Dictionary mapping lowercase filename -> YouTube ID
    """
    id_map = {}
    if os.path.exists(id_map_path):
        with open(id_map_path, "r", encoding="utf-8") as f:
            for line in f:
                real_id = line.strip()
                if real_id and len(real_id) == 11:
                    id_map[real_id.lower()] = real_id
    return id_map


def write_jsonl(segments: List[Dict[str, Any]], jsonl_path: str, source_id: str, video_id: str) -> None:
    """
    Write sentence-aware segments to JSONL for vector search or timestamp mapping.

    :param segments: List of segment dicts with start/end/text
    :param jsonl_path: Output file path (.jsonl)
    :param source_id: Base source ID for segment identifiers
    :param video_id: YouTube ID or placeholder
    """
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments):
            out = {
                "id": f"{source_id}_{idx:05d}",
                "source": source_id,
                "segment_index": idx,
                "start_time": seg["start"],
                "end_time": seg["end"],
                "duration": round(seg["end"] - seg["start"], 3),
                "text_raw": seg["text_raw"],
                "video_id": video_id
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def write_txt(segments: List[Dict[str, Any]], txt_path: str) -> None:
    """
    Write plain text transcript from sentence-aware segments.

    - Preserves line breaks between sentences.
    - Skips empty segments.

    :param segments: List of sentence segments
    :param txt_path: Output file path (.txt)
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            if seg["text_raw"]:
                f.write(seg["text_raw"] + "\n\n")  # double line break for readability


# These were used on the old script
# May be handy if we do a fixed time sliced cues option
def clean_text(text: str) -> str:
    """
    Normalize spaces and preserve line breaks in captions.

    :param text: Raw VTT caption
    :return: Cleaned text
    """
    return "\n".join(re.sub(r"\s+", " ", l).strip() for l in text.splitlines())


def build_segment(source_id: str, idx: int, buffer: List[str], start: float, window: float, video_id: Optional[str]) -> Dict[str, Any]:
    """
    Build a transcript segment dictionary.

    :param source_id: Canonical source ID
    :param idx: Segment index
    :param buffer: List of cue texts
    :param start: Segment start time in seconds
    :param window: Duration of segment
    :param video_id: Optional YouTube ID
    :return: Segment dictionary
    """
    end = round(start + window, 3)
    seg = {
        "id": f"{source_id}_{idx:05d}",
        "source": source_id,
        "segment_index": idx,
        "start_time": round(start, 3),
        "end_time": end,
        "duration": round(end - start, 3),
        "text_raw": " ".join(buffer).strip()
    }
    if video_id:
        seg["video_id"] = video_id
    return seg


def fixed_slice_cues(cues: List[Dict[str, Any]], source_id: str, window: float, video_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Slice cues into fixed-duration segments.

    - Ignores sentence boundaries, strictly enforces segment duration.
    - Uses build_segment() to construct dictionaries.

    :param cues: List of cue dicts
    :param source_id: Base source ID
    :param window: Segment duration in seconds
    :param video_id: Optional YouTube ID
    :return: List of fixed-duration segment dicts
    """
    segments = []
    buffer = []
    window_start = None
    idx = 0

    for cue in cues:
        if window_start is None:
            window_start = cue["start"]
        if cue["start"] < window_start + window:
            buffer.append(cue["text"])
        else:
            segments.append(build_segment(source_id, idx, buffer, window_start, window, video_id))
            idx += 1
            buffer = [cue["text"]]
            window_start = cue["start"]

    if buffer:
        segments.append(build_segment(source_id, idx, buffer, window_start, window, video_id))
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Slice .vtt transcripts into plain text and JSONL segments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Requirements:
              - Python 3.10+
              - webvtt-py
              - toml (for config parsing)
              - hashlib, json, os, sys (standard library)

            Usage Examples:
              python yatsee_slice_vtt.py -e defined_entity --create-txt
              python yatsee_slice_vtt.py --vtt-input ./transcripts --window 30 --force
              python yatsee_slice_vtt.py -i ./entity/transcripts_small --output-dir ./segments
              python yatsee_slice_vtt.py -e defined_entity --quiet
        """)
    )
    parser.add_argument("-e", "--entity", help="Entity handle to process")
    parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to the global YATSEE configuration file")
    parser.add_argument("-i", "--vtt-input", help="Input file or directory")
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory")
    parser.add_argument("--max-window", type=float, default=90.0, help="Hard upper limit on segment length")
    parser.add_argument("--create-txt", action="store_true", help="Generate plain text transcript")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    # Determine input/output paths
    entity_cfg = {}
    if args.entity:
        # Load entity config
        try:
            global_cfg = load_global_config(args.config)
            entity_cfg = load_entity_config(global_cfg, args.entity)
        except Exception as e:
            print(f"❌ Config load failed: {e}", file=sys.stderr)
            return 1
    else:
        # Require input if no entity is provided
        if not args.vtt_input:
            print("❌ Without --entity, --vtt-input must be defined", file=sys.stderr)
            return 1

    # Determine input/output directory based on entity or CLI override
    input_dir = args.vtt_input or os.path.join(entity_cfg["data_path"], f"transcripts_{entity_cfg.get('transcription_model', 'small')}")
    vtt_files_list = discover_files(input_dir, ".vtt")
    if not vtt_files_list:
        print("↪ No vtt input files found", file=sys.stderr)
        return 0

    # By default, output directory is the same as the input directory
    output_directory = args.vtt_input or os.path.join(entity_cfg["data_path"], f"transcripts_{entity_cfg.get('transcription_model', 'small')}")
    if not os.path.isdir(output_directory):
        print(f"✓ Output directory will be created: {output_directory}", file=sys.stderr)
        os.makedirs(output_directory, exist_ok=True)

    download_tracker = os.path.join(entity_cfg["data_path"], "downloads", ".downloaded")
    id_map = load_video_id_map(download_tracker)

    # -----------------------------------------
    # Make output files from .vtt input
    # -----------------------------------------
    for vtt_file in vtt_files_list:
        base_name = os.path.splitext(os.path.basename(vtt_file))[0]

        # Extract video ID if possible, otherwise generate placeholder
        match = re.match(r"([A-Za-z0-9_-]{11})", base_name)
        video_id = id_map.get(match.group(1).lower()) if match else None
        if not video_id:
            video_id = generate_placeholder_id(base_name)
            if not args.quiet:
                print(f"⚠ Placeholder ID for {base_name}: {video_id}")

        try:
            cues = read_vtt(vtt_file)
            segments = consolidate_sentences(cues, max_window=args.max_window)
            if not cues:
                raise ValueError("No cues found in VTT")
        except (OSError, webvtt.errors.MalformedFileError, ValueError) as e:
            print(f"❌ Failed to read VTT '{vtt_file}': {e}", file=sys.stderr)
            continue

        # Write TXT transcript if requested
        if args.create_txt:
            txt_path = os.path.join(output_directory, f"{base_name}.txt")
            try:
                if not os.path.exists(txt_path) or args.force:
                    write_txt(segments, txt_path)
                    if not args.quiet:
                        print(f"✓ Wrote {txt_path}")
                elif not args.quiet:
                    print(f"ℹ Skipped (exists): {txt_path}")
            except OSError as e:
                print(f"❌ Failed to write TXT '{txt_path}': {e}", file=sys.stderr)

        # Write segment JSONL safely
        jsonl_path = os.path.join(output_directory, f"{base_name}.segments.jsonl")
        try:
            # segments = fixed_slice_cues(cues, base_name, args.window, video_id)
            # if not segments:
            #     raise ValueError("No segments generated")

            if not os.path.exists(jsonl_path) or args.force:
                write_jsonl(segments, jsonl_path, base_name, video_id)
                if not args.quiet:
                    print(f"✓ Wrote {jsonl_path} ({len(segments)} segments)")
            elif not args.quiet:
                print(f"ℹ Skipped (exists): {jsonl_path}")
        except (OSError, ValueError) as e:
            print(f"❌ Failed to generate/write segments for '{vtt_file}': {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
