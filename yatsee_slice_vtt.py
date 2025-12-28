#!/usr/bin/env python3
"""
Processes VTT subtitle files into:
1) Plain text transcripts (.txt)
2) Time-sliced transcript segments (.segments.jsonl)

This script is designed as an early structural step in the pipeline:
- Preserves timestamps for later video linking
- Produces fixed-duration transcript segments for embedding and search
- Avoids linguistic normalization or sentence-level decisions

The output of this script should be treated as a *canonical structural layer*.
Later stages may summarize, embed, or analyze this data, but should not
rewrite timestamps or segmentation.

Requirements:
- Python 3.8+
- webvtt-py

Install with:
  pip install webvtt-py

Usage:
  ./yatsee_vtt_slice.py --vtt-input meeting.vtt --output-dir out/
  ./yatsee_vtt_slice.py --vtt-input transcripts/ --output-dir processed/ --window 30 --force
"""

import os
import sys
import argparse
import json
import re
from typing import List, Optional
import webvtt


def get_files_list(path: str) -> List[str]:
    """
    Resolve a list of .vtt files from a file or directory path.

    This mirrors behavior in other Yatsee scripts:
    - Accepts a single .vtt file
    - Accepts a directory containing multiple .vtt files
    - Fails loudly if nothing usable is found

    :param path: Path to a .vtt file or directory
    :return: List of resolved .vtt file paths
    """
    vtt_files = []

    if os.path.isdir(path):
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path) and filename.lower().endswith(".vtt"):
                vtt_files.append(full_path)

        if not vtt_files:
            raise FileNotFoundError(f"No .vtt files found in directory: {path}")

    elif os.path.isfile(path):
        if path.lower().endswith(".vtt"):
            vtt_files.append(path)
        else:
            raise ValueError(f"Unsupported file extension: {path}")

    else:
        raise FileNotFoundError(f"Input path not found: {path}")

    return vtt_files


def clean_text(text: str) -> str:
    """
    Perform minimal, non-destructive cleanup on caption text.

    Preserves original line breaks from the VTT cues.

    :param text: Raw caption text from VTT
    :return: Cleaned text with line breaks preserved
    """
    # Normalize spaces but preserve existing line breaks
    lines = text.splitlines()
    lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
    return "\n".join(lines)


def read_vtt(vtt_path: str) -> list:
    """
    Load a VTT file and convert cues into a normalized internal structure.

    Each cue is represented as a dict with:
    - start time (seconds)
    - end time (seconds)
    - minimally cleaned text

    :param vtt_path: Path to .vtt file
    :return: List of cue dictionaries
    """
    cues = []

    for cap in webvtt.read(vtt_path):
        cues.append({
            "start": cap.start_in_seconds,
            "end": cap.end_in_seconds,
            "text": clean_text(cap.text),
        })

    return cues


def write_txt(cues: list, output_path: str) -> None:
    """
    Write a plain text transcript from VTT cues.

    This output is intended for:
    - Human inspection
    - Downstream normalization scripts
    - Fallback processing when timestamps are not required

    The ordering strictly follows cue order.
    Original line breaks from the VTT are preserved.

    :param cues: List of normalized cue dictionaries
    :param output_path: Destination .txt file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for cue in cues:
            if cue["text"]:
                # preserve line breaks within each cue
                f.write(cue["text"] + "\n")


def slice_cues(cues: list, source_id: str, window_size: float, video_url: Optional[str] = None,) -> list:
    """
    Slice caption cues into fixed-duration time windows.

    This produces deterministic, time-bounded transcript segments
    suitable for embeddings, search indexing, and video linking.

    Segments are built sequentially based on cue start times.
    No attempt is made to align to sentence or speaker boundaries.

    :param cues: List of normalized cue dictionaries
    :param source_id: Canonical meeting/source identifier
    :param window_size: Window duration in seconds
    :param video_url: Optional base video URL for timestamp links
    :return: List of segment dictionaries
    """
    segments = []
    buffer = []
    window_start = None
    idx = 0

    for cue in cues:
        # Initialize the first window on the first cue
        if window_start is None:
            window_start = cue["start"]

        # If cue falls inside current window, accumulate text
        if cue["start"] < window_start + window_size:
            buffer.append(cue["text"])
        else:
            # Flush the current window and start a new one
            segments.append(
                _build_segment(
                    source_id, idx, buffer, window_start, window_size, video_url
                )
            )
            idx += 1
            buffer = [cue["text"]]
            window_start = cue["start"]

    # Flush any remaining buffered text
    if buffer:
        segments.append(
            _build_segment(
                source_id, idx, buffer, window_start, window_size, video_url
            )
        )

    return segments


def _build_segment(source_id: str, idx: int, buffer: list, start: float, window_size: float, video_url: Optional[str],) -> dict:
    """
    Construct a canonical transcript segment object.

    This function is intentionally isolated so that:
    - Segment schema changes occur in exactly one place
    - IDs remain stable and predictable
    - Future metadata additions do not affect slicing logic

    :param source_id: Canonical meeting/source identifier
    :param idx: Segment index (0-based)
    :param buffer: List of text fragments collected for this window
    :param start: Window start time in seconds
    :param window_size: Window duration in seconds
    :param video_url: Optional base video URL
    :return: Segment dictionary
    """
    start_time = round(start, 3)
    end_time = round(start + window_size, 3)

    seg = {
        "id": f"{source_id}_{idx:05d}",
        "source": source_id,
        "segment_index": idx,
        "start_time": start_time,
        "end_time": end_time,
        "duration": round(end_time - start_time, 3),
        "text_raw": " ".join(buffer).strip(),
    }

    if video_url:
        seg["video_url"] = f"{video_url}&t={int(start_time)}"

    return seg


def main() -> int:
    """
    Command-line entry point.

    Handles argument parsing, file resolution, and batch processing.
    Mirrors conventions used across existing Yatsee scripts to keep
    pipeline behavior predictable.
    """
    parser = argparse.ArgumentParser(
        description="Convert VTT files into plain text and time-sliced transcript segments."
    )
    parser.add_argument("--vtt-input", "-i", required=True, help="Input .vtt file or directory")
    parser.add_argument("--output-dir", "-o", default="processed_vtt", help="Output directory")
    parser.add_argument("--create-txt", action="store_true", help="Generate plain text transcript (.txt) output")
    parser.add_argument("--window", type=float, default=30.0, help="Window size in seconds")
    parser.add_argument("--video-url", help="Base video URL for timestamp links (default: construct from filename)",)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    try:
        vtt_files = get_files_list(args.vtt_input)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    for vtt_file in vtt_files:
        base_name = os.path.splitext(os.path.basename(vtt_file))[0]
        txt_out = os.path.join(args.output_dir, f"{base_name}.txt")
        seg_out = os.path.join(args.output_dir, f"{base_name}.segments.jsonl")

        # Skip work unless explicitly told otherwise
        if not args.force and os.path.exists(txt_out) and os.path.exists(seg_out):
            print(f"↪ Skipping existing: {base_name}")
            continue

        cues = read_vtt(vtt_file)
        if args.create_txt:
            write_txt(cues, txt_out)

        # Determine video URL
        if args.video_url:
            video_base = args.video_url
        else:
            # Default: assume YouTube, use first part of filename as ID
            yt_id = base_name.split(".")[0]
            video_base = f"https://www.youtube.com/watch?v={yt_id}"

        segments = slice_cues(
            cues,
            base_name,
            args.window,
            video_base,
        )

        with open(seg_out, "w", encoding="utf-8") as f:
            for seg in segments:
                if seg["text_raw"]:
                    f.write(json.dumps(seg, ensure_ascii=False) + "\n")

        if args.create_txt:
            print(f"✓ Wrote: {txt_out}")
        print(f"✓ Wrote: {seg_out} ({len(segments)} segments)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
