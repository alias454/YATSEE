"""
Transcript slicing stage for YATSEE.

This module ports the existing VTT slicing behavior behind reusable functions
so the new CLI can invoke it without embedding stage logic directly.

Behavior intentionally mirrors the current standalone script:
- input from transcripts_<model>/ or direct override
- plain text transcript output
- optional JSONL segment output with embeddings
- sentence-aware cue consolidation
- placeholder ID generation when source IDs are unavailable
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
import string
from typing import Any, Dict, List, Optional

import torch
import webvtt
from sentence_transformers import SentenceTransformer

from yatsee.core.config import load_entity_config, load_global_config
from yatsee.core.discovery import discover_files
from yatsee.core.errors import ConfigError, ValidationError

SUPPORTED_INPUT_EXTENSIONS = (".vtt",)


def clear_gpu_cache() -> None:
    """
    Clear PyTorch GPU cache and trigger garbage collection.

    This matters when embeddings are enabled and a GPU-backed
    SentenceTransformer model is used.

    :return: None
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def resolve_device(device_arg: str) -> str:
    """
    Resolve runtime device for SentenceTransformer embedding generation.

    :param device_arg: Requested device: auto, cuda, cpu, mps
    :return: Resolved device name
    """
    if torch.cuda.is_available() and device_arg in ["auto", "cuda"]:
        clear_gpu_cache()
        return "cuda"

    if torch.backends.mps.is_available() and device_arg in ["auto", "mps"]:
        return "mps"

    return "cpu"


def parse_vtt_timestamp(value: str) -> float:
    """
    Convert a WebVTT timestamp into seconds.

    Supports both HH:MM:SS.mmm and MM:SS.mmm variants.

    :param value: WebVTT timestamp string
    :return: Timestamp in seconds
    :raises ValueError: If the timestamp is malformed
    """
    parts = value.strip().split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds

    raise ValueError(f"Malformed VTT timestamp: {value}")


def clean_text(text: str) -> str:
    """
    Normalize spaces while preserving caption line boundaries.

    :param text: Raw VTT caption text
    :return: Cleaned caption text
    """
    return "\n".join(re.sub(r"\s+", " ", line).strip() for line in text.splitlines())


def read_vtt(vtt_path: str) -> List[Dict[str, Any]]:
    """
    Read WebVTT captions into a normalized cue list.

    :param vtt_path: Path to the .vtt file
    :return: List of cue dictionaries
    :raises OSError: If the file cannot be read
    :raises webvtt.errors.MalformedFileError: If the VTT is malformed
    """
    cues: List[Dict[str, Any]] = []

    for cue in webvtt.read(vtt_path):
        cleaned = clean_text(cue.text)
        if not cleaned.strip():
            continue

        cues.append(
            {
                "start": round(parse_vtt_timestamp(cue.start), 3),
                "end": round(parse_vtt_timestamp(cue.end), 3),
                "text": cleaned,
            }
        )

    return cues


def consolidate_sentences(
    cues: List[Dict[str, Any]],
    max_window: float = 90.0,
    min_words: int = 1,
) -> List[Dict[str, Any]]:
    """
    Consolidate VTT cues into sentence-aware transcript segments.

    This preserves the current stage behavior:
    - cues without terminal punctuation continue buffering
    - cues are also hard-bounded by max_window
    - tiny segments can be filtered via min_words

    :param cues: List of normalized cue dictionaries
    :param max_window: Hard upper bound on segment duration
    :param min_words: Minimum word count required to keep a segment
    :return: List of sentence-aware segment dictionaries
    """
    if not cues:
        return []

    segments: List[Dict[str, Any]] = []
    buffer: List[str] = []
    seg_start: Optional[float] = None
    seg_end: Optional[float] = None

    terminal_pattern = re.compile(r"[.!?]['\"]?$")

    for cue in cues:
        if seg_start is None:
            seg_start = cue["start"]

        if seg_end is None:
            seg_end = cue["end"]
        else:
            seg_end = cue["end"]

        text = " ".join(part.strip() for part in cue["text"].splitlines() if part.strip()).strip()
        if text:
            buffer.append(text)

        current_window = cue["end"] - seg_start if seg_start is not None else 0.0
        joined = " ".join(buffer).strip()
        word_count = len(joined.split())

        should_flush = False

        if terminal_pattern.search(text):
            should_flush = True

        if current_window >= max_window:
            should_flush = True

        if should_flush and joined and word_count >= min_words:
            segments.append(
                {
                    "start": round(seg_start, 3),
                    "end": round(seg_end or cue["end"], 3),
                    "text_raw": joined,
                }
            )
            buffer = []
            seg_start = None
            seg_end = None

    if buffer and seg_start is not None and seg_end is not None:
        joined = " ".join(buffer).strip()
        if len(joined.split()) >= min_words:
            segments.append(
                {
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "text_raw": joined,
                }
            )

    return segments


def generate_placeholder_id(base_name: Optional[str] = None) -> str:
    """
    Generate a deterministic placeholder ID when no real source ID is available.

    :param base_name: Optional seed string
    :return: 11-character placeholder ID
    """
    if base_name:
        digest = hashlib.sha256(base_name.encode("utf-8")).hexdigest()
        return "".join(char for char in digest if char.isalnum())[:11]

    return "".join(random.choices(string.ascii_letters + string.digits, k=11))


def load_video_id_map(id_map_path: str) -> Dict[str, str]:
    """
    Load a mapping from lowercase filename prefixes to real source IDs.

    This preserves the current YATSEE behavior of looking up real IDs from
    the downloads/.downloaded tracker when possible.

    :param id_map_path: Path to the .downloaded tracker file
    :return: Dictionary mapping lowercase ID to canonical source ID
    """
    id_map: Dict[str, str] = {}

    if os.path.exists(id_map_path):
        with open(id_map_path, "r", encoding="utf-8") as handle:
            for line in handle:
                real_id = line.strip()
                if real_id and len(real_id) == 11:
                    id_map[real_id.lower()] = real_id

    return id_map


def write_jsonl(
    segments: List[Dict[str, Any]],
    jsonl_path: str,
    source_id: str,
    video_id: str,
    embeddings: Optional[List[List[float]]] = None,
) -> None:
    """
    Write sentence-aware segments to JSONL.

    :param segments: List of segment dictionaries
    :param jsonl_path: Output file path
    :param source_id: Base source identifier
    :param video_id: Real or placeholder video/source ID
    :param embeddings: Optional embedding vectors aligned to segments
    """
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for idx, seg in enumerate(segments):
            out: Dict[str, Any] = {
                "id": f"{source_id}_{idx:05d}",
                "source": source_id,
                "segment_index": idx,
                "start_time": seg["start"],
                "end_time": seg["end"],
                "duration": round(seg["end"] - seg["start"], 3),
                "text_raw": seg["text_raw"],
                "video_id": video_id,
            }

            if embeddings:
                out["embedding"] = embeddings[idx]

            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def write_txt(segments: List[Dict[str, Any]], txt_path: str) -> None:
    """
    Write plain text transcript from sentence-aware segments.

    :param segments: List of sentence-aware segments
    :param txt_path: Output file path
    """
    with open(txt_path, "w", encoding="utf-8") as handle:
        for seg in segments:
            if seg["text_raw"]:
                handle.write(seg["text_raw"] + "\n\n")


def resolve_slice_paths(
    global_config_path: str,
    entity: str | None,
    vtt_input: str | None,
    output_dir: str | None,
    model_override: str | None,
) -> Dict[str, Any]:
    """
    Resolve config and paths for the transcript slice stage.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Optional entity handle
    :param vtt_input: Optional input override
    :param output_dir: Optional output override
    :param model_override: Optional embedding model override
    :return: Resolved config and paths
    :raises ValidationError: If required arguments are missing
    """
    entity_cfg: Dict[str, Any] = {}
    global_cfg = load_global_config(global_config_path)

    if entity:
        entity_cfg = load_entity_config(global_cfg, entity)
    else:
        if not vtt_input or not output_dir:
            raise ValidationError(
                "Without --entity, both --vtt-input and --output-dir must be defined"
            )

    data_path = entity_cfg.get("data_path")
    embedding_model = (
        model_override
        or entity_cfg.get("embedding_model")
        or global_cfg.get("system", {}).get("default_embedding_model", "BAAI/bge-small-en-v1.5")
    )

    resolved_input = vtt_input or os.path.join(
        data_path,
        f"transcripts_{entity_cfg.get('transcription_model', global_cfg.get('system', {}).get('default_transcription_model', 'medium'))}",
    )
    resolved_output = output_dir or os.path.join(data_path, "normalized")

    return {
        "global_cfg": global_cfg,
        "entity_cfg": entity_cfg,
        "vtt_input": resolved_input,
        "output_dir": resolved_output,
        "embedding_model": embedding_model,
    }


def run_slice_stage(
    global_config_path: str,
    entity: str | None = None,
    vtt_input: str | None = None,
    output_dir: str | None = None,
    model_override: str | None = None,
    gen_embed: bool = False,
    max_window: float = 90.0,
    force: bool = False,
    device_arg: str = "auto",
) -> Dict[str, Any]:
    """
    Run the transcript slicing stage.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Optional entity handle
    :param vtt_input: Optional input override
    :param output_dir: Optional output override
    :param model_override: Optional embedding model override
    :param gen_embed: Generate JSONL with embeddings
    :param max_window: Hard upper segment length in seconds
    :param force: Overwrite existing outputs
    :param device_arg: Requested embedding runtime device
    :return: Summary dictionary describing stage results
    """
    resolved = resolve_slice_paths(
        global_config_path=global_config_path,
        entity=entity,
        vtt_input=vtt_input,
        output_dir=output_dir,
        model_override=model_override,
    )

    entity_cfg = resolved["entity_cfg"]
    vtt_input_path = resolved["vtt_input"]
    output_directory = resolved["output_dir"]
    embedding_model = resolved["embedding_model"]

    vtt_files = discover_files(vtt_input_path, SUPPORTED_INPUT_EXTENSIONS)
    if not vtt_files:
        return {
            "vtt_input": vtt_input_path,
            "output_dir": output_directory,
            "embedding_model": embedding_model,
            "discovered": 0,
            "txt_written": 0,
            "jsonl_written": 0,
            "messages": [f"No VTT input files found at {vtt_input_path}"],
        }

    os.makedirs(output_directory, exist_ok=True)

    embedder: Optional[SentenceTransformer] = None
    device = resolve_device(device_arg)

    if gen_embed:
        embedder = SentenceTransformer(embedding_model, device=device)

    id_map_path = ""
    if entity_cfg.get("data_path"):
        id_map_path = os.path.join(entity_cfg["data_path"], "downloads", ".downloaded")
    id_map = load_video_id_map(id_map_path)

    txt_written = 0
    jsonl_written = 0
    messages: List[str] = []

    for vtt_file in vtt_files:
        base_name = os.path.splitext(os.path.basename(vtt_file))[0]

        match = re.match(r"([A-Za-z0-9_-]{11})", base_name)
        video_id = id_map.get(match.group(1).lower()) if match else None
        if not video_id:
            video_id = generate_placeholder_id(base_name)
            messages.append(f"Placeholder ID for {base_name}: {video_id}")

        try:
            cues = read_vtt(vtt_file)
            if not cues:
                raise ValueError("No cues found in VTT")
        except Exception as exc:
            messages.append(f"Failed to read VTT '{vtt_file}': {exc}")
            continue

        txt_path = os.path.join(output_directory, f"{base_name}.txt")
        if not os.path.exists(txt_path) or force:
            txt_segments = consolidate_sentences(cues, max_window=max_window, min_words=1)
            write_txt(txt_segments, txt_path)
            txt_written += 1
            messages.append(f"Wrote TXT transcript: {txt_path}")
        else:
            messages.append(f"Skipped existing TXT transcript: {txt_path}")

        if gen_embed:
            jsonl_path = os.path.join(output_directory, f"{base_name}.segments.jsonl")
            if not os.path.exists(jsonl_path) or force:
                jsonl_segments = consolidate_sentences(cues, max_window=max_window, min_words=15)
                texts = [seg["text_raw"] for seg in jsonl_segments]

                embeddings: List[List[float]] = []
                if embedder and texts:
                    embeddings = embedder.encode(
                        texts,
                        batch_size=32,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).tolist()

                write_jsonl(
                    segments=jsonl_segments,
                    jsonl_path=jsonl_path,
                    source_id=base_name,
                    video_id=video_id,
                    embeddings=embeddings if texts else None,
                )
                jsonl_written += 1
                messages.append(f"Wrote JSONL segments: {jsonl_path}")
            else:
                messages.append(f"Skipped existing JSONL segments: {jsonl_path}")

    if gen_embed:
        clear_gpu_cache()

    return {
        "vtt_input": vtt_input_path,
        "output_dir": output_directory,
        "embedding_model": embedding_model,
        "device": device,
        "discovered": len(vtt_files),
        "txt_written": txt_written,
        "jsonl_written": jsonl_written,
        "messages": messages,
    }