"""
Audio transcription stage for YATSEE.

This module ports the existing transcription stage behind reusable functions so
the new CLI can invoke it without embedding stage logic directly.

Behavior intentionally mirrors the current standalone script:
- global + entity config resolution
- hotword flattening from titles and people aliases
- CPU/CUDA/MPS device selection
- optional faster-whisper backend
- chunk-directory support
- SHA-256 tracker for idempotent reruns
- VTT output in transcripts_<model>/
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import os
import sys
import time
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set

import torch
import torchaudio
from tqdm import tqdm

from yatsee.core.config import load_entity_config, load_global_config
from yatsee.core.discovery import discover_files
from yatsee.core.errors import ValidationError
from yatsee.core.hashing import compute_sha256
from yatsee.core.tracking import append_tracker_value, load_tracker_set

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="torchaudio._backend.utils.info has been deprecated",
)

HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None
SUPPORTED_INPUT_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a")


def load_flat_hotwords(entity_cfg: Dict[str, Any]) -> Optional[str]:
    """
    Flatten entity titles and people aliases into a comma-separated hotwords string.

    :param entity_cfg: Merged configuration dictionary for the entity
    :return: Comma-separated string of hotwords, or None if no hotwords found
    """
    hotwords: Set[str] = set()

    for title_list in entity_cfg.get("titles", {}).values():
        if isinstance(title_list, list):
            hotwords.update([t.strip() for t in title_list if t.strip()])

    for role_dict in entity_cfg.get("people", {}).values():
        for aliases in role_dict.values():
            if isinstance(aliases, list):
                hotwords.update([a.strip() for a in aliases if a.strip()])

    return ", ".join(sorted(hotwords)) if hotwords else None


def get_audio_duration(audio_path: str) -> float:
    """
    Return the duration of an audio file in seconds.

    :param audio_path: Path to the audio file
    :return: Duration in seconds as a float
    :raises RuntimeError: If file cannot be read
    """
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def clear_gpu_cache() -> None:
    """
    Clear PyTorch GPU cache and trigger garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def format_vtt_timestamp(seconds: float) -> str:
    """
    Convert seconds to WebVTT timestamp format HH:MM:SS.mmm.

    :param seconds: Time in seconds
    :return: Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def normalize_segments(segments: list[Any]) -> list[SimpleNamespace]:
    """
    Convert transcription segments to objects with start, end, and text attributes.

    :param segments: List of segment dicts or objects
    :return: List of normalized segment objects
    """
    normalized: list[SimpleNamespace] = []
    for seg in segments:
        if isinstance(seg, dict):
            normalized.append(
                SimpleNamespace(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                )
            )
        else:
            normalized.append(seg)
    return normalized


def resolve_device(device_arg: str, faster: bool) -> tuple[str, bool]:
    """
    Resolve runtime device and fp16 usage.

    :param device_arg: Requested device: auto, cuda, cpu, mps
    :param faster: Whether faster-whisper was requested
    :return: Tuple of (device, use_fp16)
    """
    if torch.cuda.is_available() and device_arg in ["auto", "cuda"]:
        clear_gpu_cache()
        return "cuda", True

    if torch.backends.mps.is_available() and device_arg in ["auto", "mps"]:
        if faster:
            return "cpu", False
        return "mps", False

    return "cpu", False


def load_transcription_model(model: str, device: str, use_fp16: bool, faster: bool) -> tuple[Any, bool]:
    """
    Load the requested transcription backend.

    :param model: Whisper model name
    :param device: Resolved runtime device
    :param use_fp16: Whether fp16 should be enabled
    :param faster: Whether faster-whisper was requested
    :return: Tuple of (model_instance, using_faster_whisper)
    """
    if faster and HAS_FASTER_WHISPER:
        from faster_whisper import WhisperModel

        compute_type = "float16" if use_fp16 else "int8"
        return WhisperModel(model, device=device, compute_type=compute_type), True

    import whisper

    whisper_model = whisper.load_model(model).to(device)
    return whisper_model, False


def resolve_transcribe_paths(
    global_config_path: str,
    entity: str | None,
    audio_input: str | None,
    output_dir: str | None,
    model_override: str | None,
) -> Dict[str, Any]:
    """
    Resolve config and filesystem paths for the transcription stage.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Optional entity handle
    :param audio_input: Optional direct input override
    :param output_dir: Optional direct output override
    :param model_override: Optional model override
    :return: Dictionary of resolved config and paths
    :raises ValidationError: If required arguments are missing
    """
    entity_cfg: Dict[str, Any] = {}
    global_cfg = load_global_config(global_config_path)

    if entity:
        entity_cfg = load_entity_config(global_cfg, entity)
    else:
        if not audio_input or not output_dir:
            raise ValidationError(
                "Without --entity, both --audio-input and --output-dir must be defined"
            )

    model = model_override or entity_cfg.get("transcription_model", "small")
    resolved_input = audio_input or os.path.join(entity_cfg.get("data_path"), "audio")
    resolved_output = output_dir or os.path.join(entity_cfg.get("data_path"), f"transcripts_{model}")

    return {
        "global_cfg": global_cfg,
        "entity_cfg": entity_cfg,
        "audio_input": resolved_input,
        "output_dir": resolved_output,
        "model": model,
    }


def run_transcribe_stage(
    global_config_path: str,
    entity: str | None = None,
    audio_input: str | None = None,
    output_dir: str | None = None,
    get_chunks: bool = False,
    model_override: str | None = None,
    faster: bool = False,
    language: str = "en",
    device_arg: str = "auto",
    verbose: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Run the audio transcription stage.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Optional entity handle
    :param audio_input: Optional input override
    :param output_dir: Optional output override
    :param get_chunks: Whether to use chunk directories when present
    :param model_override: Optional model override
    :param faster: Whether to use faster-whisper
    :param language: Language code or 'auto'
    :param device_arg: Requested runtime device
    :param verbose: Enable verbose transcription output
    :param quiet: Suppress progress output
    :return: Summary dictionary describing stage results
    """
    resolved = resolve_transcribe_paths(
        global_config_path=global_config_path,
        entity=entity,
        audio_input=audio_input,
        output_dir=output_dir,
        model_override=model_override,
    )

    entity_cfg = resolved["entity_cfg"]
    model = resolved["model"]
    audio_input_path = resolved["audio_input"]
    output_directory = resolved["output_dir"]

    audio_file_list = discover_files(audio_input_path, SUPPORTED_INPUT_EXTENSIONS)
    if not audio_file_list:
        return {
            "audio_input": audio_input_path,
            "output_dir": output_directory,
            "model": model,
            "discovered": 0,
            "processed": 0,
            "skipped": 0,
            "messages": [f"No audio input files found at {audio_input_path}"],
        }

    os.makedirs(output_directory, exist_ok=True)

    device, use_fp16 = resolve_device(device_arg, faster)
    whisper_model, use_faster_whisper = load_transcription_model(model, device, use_fp16, faster)

    hotwords = load_flat_hotwords(entity_cfg)
    lang = None if language.lower() == "auto" else language

    hash_tracker = os.path.join(output_directory, ".vtt_hash")
    existing_hashes = load_tracker_set(hash_tracker)

    processed = 0
    skipped = 0
    messages: list[str] = []

    for audio_path in audio_file_list:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vtt_filepath = os.path.join(output_directory, f"{base_name}.vtt")

        file_hash = compute_sha256(audio_path)
        video_id = base_name.split(".", 1)[0]
        hash_key = f"{video_id}:{file_hash}"

        if hash_key in existing_hashes:
            skipped += 1
            messages.append(f"Skipped already transcribed: {audio_path}")
            continue

        audio_directory = audio_input_path
        if os.path.isfile(audio_input_path):
            audio_directory = os.path.dirname(audio_input_path)

        chunk_dir = os.path.join(audio_directory, "chunks", base_name)
        if get_chunks and os.path.isdir(chunk_dir):
            audio_chunks = sorted(
                os.path.join(chunk_dir, entry)
                for entry in os.listdir(chunk_dir)
                if entry.lower().endswith(SUPPORTED_INPUT_EXTENSIONS)
            )
        else:
            audio_chunks = [audio_path]

        try:
            total_duration = get_audio_duration(audio_path)
        except RuntimeError as exc:
            messages.append(f"Failed to read audio info for '{audio_path}': {exc}")
            total_duration = 0.0

        progress_bar = None
        if sys.stdout.isatty() and not verbose and not quiet:
            progress_bar = tqdm(
                total=total_duration,
                unit="sec",
                desc=f"Transcribing {base_name}",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} {unit} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        all_segments: list[Any] = []
        offset = 0.0
        last_emitted_end = 0.0
        start_time = time.time()

        for index, audio_chunk in enumerate(audio_chunks, start=1):
            if progress_bar:
                progress_bar.set_description(f"Transcribing {base_name} | Chunk {index}/{len(audio_chunks)}")
                progress_bar.refresh()

            try:
                if use_faster_whisper:
                    segments, _info = whisper_model.transcribe(
                        audio_chunk,
                        hotwords=hotwords,
                        log_progress=verbose,
                        beam_size=5,
                        language=lang,
                        condition_on_previous_text=False,
                    )
                else:
                    result = whisper_model.transcribe(
                        audio_chunk,
                        initial_prompt=hotwords,
                        verbose=verbose,
                        language=lang,
                        fp16=use_fp16,
                        condition_on_previous_text=False,
                    )
                    segments = result.get("segments", []) if isinstance(result, dict) else []
            except Exception as exc:
                messages.append(f"Error transcribing '{audio_chunk}': {exc}")
                continue

            normalized = normalize_segments(list(segments))
            if not normalized:
                offset = last_emitted_end
                continue

            for seg in normalized:
                seg.start += offset
                seg.end += offset
                all_segments.append(seg)

                if progress_bar:
                    progress_bar.update(seg.end - seg.start)

            offset = normalized[-1].end
            last_emitted_end = offset
            clear_gpu_cache()

        all_segments.sort(key=lambda seg: seg.start)

        with open(vtt_filepath, "w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for seg in all_segments:
                start_ts = format_vtt_timestamp(seg.start)
                end_ts = format_vtt_timestamp(seg.end)
                text = seg.text.strip()
                vtt_file.write(f"{start_ts} --> {end_ts}\n{text}\n\n")

        if progress_bar:
            progress_bar.n = progress_bar.total
            progress_bar.refresh()
            progress_bar.close()

        append_tracker_value(hash_tracker, hash_key)
        processed += 1
        elapsed = round(time.time() - start_time, 2)
        messages.append(
            f"Transcribed successfully: {vtt_filepath} "
            f"(chunks={len(audio_chunks)}, elapsed={elapsed}s, backend={'faster-whisper' if use_faster_whisper else 'whisper'})"
        )

    return {
        "audio_input": audio_input_path,
        "output_dir": output_directory,
        "model": model,
        "device": device,
        "backend": "faster-whisper" if use_faster_whisper else "whisper",
        "discovered": len(audio_file_list),
        "processed": processed,
        "skipped": skipped,
        "messages": messages,
    }