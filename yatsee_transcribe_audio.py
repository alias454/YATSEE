#!/usr/bin/env python3

# Copyright (C) 2026 Alias454
# Licensed under the AGPLv3. See LICENSE file for details.
"""
yatsee_transcribe_audio.py

Stage 3 of YATSEE: Transcribe audio files to WebVTT (.vtt) using Whisper or faster-whisper.

Purpose:
  - Convert audio (.mp3, .wav, .flac, .m4a) into text transcripts for downstream
    processing, search, and indexing.
  - Supports both single audio files and chunked audio for improved model performance.

Input/Output:
  - Input: Audio files under 'audio/' or user-specified directory.
           Optional chunked files under 'audio/chunks/<file_basename>/'
  - Output: Transcripts (.vtt) stored in 'transcripts_<model>/' under entity data
            path or specified output directory. Overlapping or near-overlapping
            segments are merged for clarity.

Key Features:
  - Automatically skips already-transcribed files using SHA-256 hash tracking
  - Supports hotwords from titles and people aliases in entity config
  - Verbose or quiet output modes
  - CPU, CUDA, and Apple MPS device support with automatic fallback
  - Deterministic output contained within entity's data path

Dependencies:
  - whisper or faster-whisper
  - torch, torchaudio
  - toml (for configuration)
  - tqdm (for progress bars)

Usage Examples:
  ./yatsee_transcribe_audio.py -e entity_handle --audio-input ./audio --model small
  ./yatsee_transcribe_audio.py --audio-input ./downloads/video.mp4 --faster --lang en
  ./yatsee_transcribe_audio.py -i ./single_file.mp3 -d cpu --lang es
  ./yatsee_transcribe_audio.py --audio-input ./audio_folder -o ./transcripts/

Design Notes:
  - Loads flat entity configuration by merging global yatsee.toml with local entity config
  - Cleanly merges overlapping segments to maintain VTT quality
  - Tracks processed files via SHA-256 hashes to prevent redundant transcription
  - Supports chunked audio to handle long recordings efficiently
  - Modular functions for config, hashing, file discovery, hotword flattening,
    segment normalization, and VTT writing
"""

# Standard library
import argparse
import gc
import hashlib
import importlib.util
import logging
import os
import sys
import textwrap
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set

# Third-party imports
import torch
import torchaudio
import toml
from tqdm import tqdm

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, message="torchaudio._backend.utils.info has been deprecated")

# Check if faster-whisper is installed
HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None

SUPPORTED_INPUT_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a")


def logger_setup(cfg: dict) -> logging.Logger:
    """
    Configure the root logger using settings from a configuration dictionary.

    Looks for the following keys in `cfg`:
      - "log_level": Logging level as a string (e.g., "INFO", "DEBUG"). Defaults to "INFO".
      - "log_format": Logging format string. Defaults to "%(asctime)s %(levelname)s %(name)s: %(message)s".

    Initializes basic logging configuration and returns a logger instance
    for the calling module.

    :param cfg: Dictionary containing logging configuration
    :return: Configured logger instance
    """
    log_level = cfg.get("log_level", "INFO")
    log_format = cfg.get("log_format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(format=log_format, level=log_level)
    return logging.getLogger(__name__)


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


def load_flat_hotwords(entity_cfg: Dict[str, Any]) -> Optional[str]:
    """
    Flatten entity titles and people aliases into a comma-separated hotwords string.

    :param entity_cfg: Merged configuration dictionary for the entity
    :return: Comma-separated string of hotwords, or None if no hotwords found
    """
    hotwords: Set[str] = set()

    # Titles
    for title_list in entity_cfg.get("titles", {}).values():
        if isinstance(title_list, list):
            hotwords.update([t.strip() for t in title_list if t.strip()])

    # People aliases
    for role_dict in entity_cfg.get("people", {}).values():
        for aliases in role_dict.values():
            if isinstance(aliases, list):
                hotwords.update([a.strip() for a in aliases if a.strip()])

    return ", ".join(sorted(hotwords)) if hotwords else None


def get_audio_duration(audio_path: str) -> float:
    """
    Return the duration of an audio file in seconds.

    Uses torchaudio to read file metadata. Returns 0.0 if file cannot be read.

    :param audio_path: Path to the audio file
    :return: Duration in seconds as a float
    :raises RuntimeError: If file cannot be read
    """
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def discover_files(input_path: str, supported_exts) -> List[str]:
    """
    Recursively collect audio files from a directory or single file.

    Filters by allowed extensions and returns a sorted list of file paths.

    :param input_path: Directory or single audio file path
    :param supported_exts: Tuple of supported file extensions (e.g., '.mp3', '.wav')
    :return: Sorted list of valid audio file paths
    :raises FileNotFoundError: If path does not exist
    :raises ValueError: If file extension is unsupported
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


def compute_sha256(path: str) -> str:
    """
    Compute the SHA-256 hash of a file.

    Used to track already-transcribed audio and prevent redundant processing.

    :param path: Path to the file
    :return: Hexadecimal SHA-256 hash string
    :raises RuntimeError: If file cannot be read
    """
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Failed to hash file: {path}") from exc


def load_tracked_hashes(tracker_path: str) -> Set[str]:
    """
    Load SHA-256 hashes from a tracker file.

    Returns an empty set if the tracker file does not exist.

    :param tracker_path: Path to '.vtt_hash' file
    :return: Set of SHA-256 hash strings
    """
    if not os.path.exists(tracker_path):
        return set()
    try:
        with open(tracker_path, "r", encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}
    except OSError:
        return set()


def clear_gpu_cache() -> None:
    """
    Clear PyTorch GPU cache and trigger garbage collection.

    Useful to reduce memory pressure when transcribing large audio files on CUDA devices.
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


def normalize_segments(segments):
    """
    Convert transcription segments to objects with 'start', 'end', and 'text' attributes.

    Handles both dict-based segments and objects returned by different Whisper models.

    :param segments: List of segment dicts or objects
    :return: List of SimpleNamespace objects with 'start', 'end', 'text'
    """
    normalized = []
    for seg in segments:
        if isinstance(seg, dict):
            normalized.append(SimpleNamespace(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"]
            ))
        else:
            normalized.append(seg)  # already an object
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to WebVTT (.vtt) using whisper or faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Requirements:
              - Python 3.10+
              - torch (PyTorch)
              - whisper (https://github.com/openai/whisper)
              - faster-whisper (optional, https://github.com/guillaumekln/faster-whisper)
              - ffmpeg (required for audio decoding)

            Usage Examples:
              python yatsee_transcribe_audio.py -e defined_entity
              python yatsee_transcribe_audio.py --audio_input ./audio_files --model small --faster
              python yatsee_transcribe_audio.py -i ./single_file.mp3 -d cpu --lang es
              python yatsee_transcribe_audio.py --audio_input ./audio_folder -o ./transcripts/
        """),
    )
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device for model execution: 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon Support, 'cpu' for compatibility. Default is 'auto'"
    )
    parser.add_argument("-e", "--entity", help="Entity handle to process")
    parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to the global YATSEE configuration file")
    parser.add_argument("-i", "--audio-input", help="Audio file or directory (Defaults to ./audio)")
    parser.add_argument("-o", "--output-dir", help="Directory to save transcripts")
    parser.add_argument("-g", "--get-chunks", action="store_true", help="Transcribe using audio chunk files")
    parser.add_argument("-m", "--model", help="Whisper model size (overrides entity/system defaults: small, medium, turbo etc.c)")
    parser.add_argument("--faster", action="store_true", help="Use faster-whisper if installed")
    parser.add_argument("-l", "--lang", default="en", help="Language code or 'auto'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    # Verbose output flag
    verbose = args.verbose and not args.quiet

    # Determine input/output paths
    entity_cfg = {}
    if args.entity:
        # Load entity config
        try:
            global_cfg = load_global_config(args.config)
            entity_cfg = load_entity_config(global_cfg, args.entity)
        except Exception as e:
            logging.error("Config load failed: %s", e)
            return 1
    else:
        # Require both input/output if no entity is provided
        if not args.audio_input or not args.output_dir:
            logging.error("Without --entity, both --audio-input and --output-dir must be defined")
            return 1

    # Set up custom logger
    logger = logger_setup(global_cfg.get("system", {}))

    # Adjust logger level for quiet mode
    if args.quiet:
        logger.setLevel(logging.WARNING)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    elif verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    model = args.model or entity_cfg.get("transcription_model", "small")

    # -----------------------------------------
    # Validate device for model inference
    # -----------------------------------------
    if torch.cuda.is_available() and args.device in ["auto", "cuda"]:
        logger.debug("Cleared GPU cache")
        clear_gpu_cache()
        device = "cuda"
        use_fp16 = True  # Use fp16 for performance on GPU
    elif torch.backends.mps.is_available() and args.device in ["auto", "mps"]:
        if args.faster:
            # Handle the Faster-Whisper Edge Case
            logger.warning("Faster-Whisper does not support MPS (Metal) yet. Using CPU (optimized for ARM with faster-whisper).")
            device = "cpu"
            use_fp16 = False
        else:
            # Apple Silicon Support
            device = "mps"
            use_fp16 = False
    else:
        if args.device == "cuda":
            logger.warning("CUDA requested but not available, falling back to CPU.")
        if args.device == "mps":
            logger.warning("MPS requested but not available, falling back to CPU.")
        device = "cpu"
        use_fp16 = False

    #-----------------------------------------
    # Load the model: faster-whisper or whisper
    #-----------------------------------------
    if args.faster and HAS_FASTER_WHISPER:
        # Import faster-whisper
        from faster_whisper import WhisperModel
        logger.info("Using faster-whisper model '%s' on device '%s'", model, device)

        compute_type = "float16" if use_fp16 else "int8"
        whisper_model = WhisperModel(model, device=device, compute_type=compute_type)
        use_faster_whisper = True
    else:
        # Import openai whisper
        import whisper
        from whisper.utils import get_writer

        if args.faster and not HAS_FASTER_WHISPER:
            logger.warning("faster-whisper requested but not installed, falling back to standard whisper.")
        logger.info("Using standard whisper model '%s' on device '%s'", model, device)

        whisper_model = whisper.load_model(model).to(device)
        use_faster_whisper = False

    # Use audio_input if specified; else fall back to entity data_path
    audio_input = args.audio_input or os.path.join(entity_cfg.get("data_path"), "audio")
    audio_file_list = discover_files(audio_input, SUPPORTED_INPUT_EXTENSIONS)
    if not audio_file_list:
        logger.info("No audio input files found at %s", audio_input)
        return 0

    output_directory = args.output_dir or os.path.join(entity_cfg.get("data_path"), f"transcripts_{model}")
    if not os.path.isdir(output_directory):
        logger.info("Output directory will be created: %s", output_directory)
        os.makedirs(output_directory, exist_ok=True)

    # Prepare hotwords
    hotwords = load_flat_hotwords(entity_cfg)

    lang = None if args.lang.lower() == "auto" else args.lang

    hash_tracker = os.path.join(output_directory, ".vtt_hash")
    existing_hashes = load_tracked_hashes(hash_tracker)

    logger.info("Found %d audio file(s) to transcribe", len(audio_file_list))

    #-----------------------------------------
    # Start main transcription processing loop
    #-----------------------------------------
    for audio_path in audio_file_list:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Construct output VTT filename
        vtt_filepath = os.path.join(output_directory, f"{base_name}.vtt")

        # Keep track of transcription time per file
        start_time = time.time()

        # File tracking support
        file_hash = compute_sha256(audio_path)
        video_id = base_name.split(".", 1)[0]
        hash_key = f"{video_id}:{file_hash}"
        if hash_key in existing_hashes:
            logger.info("Skipping already-transcribed file: %s", audio_path)
            continue

        # Chunking support
        audio_directory = audio_input
        if os.path.isfile(audio_input):
            audio_directory = os.path.dirname(audio_input)

        chunk_dir = os.path.join(audio_directory, "chunks", base_name)
        if args.get_chunks and os.path.isdir(chunk_dir):
            logger.info("Using chunk directory: %s", chunk_dir)
            audio_chunks = sorted(
                os.path.join(chunk_dir, f)
                for f in os.listdir(chunk_dir)
                if f.lower().endswith(SUPPORTED_INPUT_EXTENSIONS)
            )
        else:
            audio_chunks = [audio_path]

        # Total duration for progress bar (seconds)
        try:
            total_duration = get_audio_duration(audio_path)
        except RuntimeError as e:
            logger.warning("Failed to read audio info for '%s': %s", audio_path, e)
            total_duration = 0.0

        # If verbose then faster whisper will output its own verbose steam
        progress_bar = None
        if sys.stdout.isatty() and not verbose:
            progress_bar = tqdm(
                total=total_duration,
                unit="sec",
                desc=f"Transcribing {base_name}",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} {unit} [{elapsed}<{remaining}, {rate_fmt}]"
            )

        all_segments = []
        offset = 0.0
        last_emitted_end = 0.0

        # Transcribe each chunk sequentially
        for i, audio_chunk in enumerate(audio_chunks, start=1):
            if progress_bar:
                progress_bar.set_description(f"Transcribing {base_name} | Chunk {i}/{len(audio_chunks)}")
                progress_bar.refresh()

            # -----------------------------------------
            # Transcribe with faster-whisper
            # -----------------------------------------
            if use_faster_whisper:
                try:
                    segments, _info = whisper_model.transcribe(audio_chunk, hotwords=hotwords, log_progress=verbose, beam_size=5, language=lang, condition_on_previous_text=False)
                    # print("Detected language '%s' with probability %f" % (_info.language, _info.language_probability))
                except Exception:
                    logger.error("Error transcribing '%s' with faster-whisper", audio_chunk, exc_info=True)
                    continue
            else:
                # -----------------------------------------
                # Transcribe with openai whisper
                # -----------------------------------------
                try:
                    result = whisper_model.transcribe(audio_chunk, initial_prompt=hotwords, verbose=verbose, language=lang, fp16=use_fp16, condition_on_previous_text=False)
                    # Extract segments safely
                    segments = result.get("segments", []) if isinstance(result, dict) else getattr(result, "segments", []) or []
                except Exception:
                    logger.error("Error transcribing '%s'", audio_chunk, exc_info=True)
                    continue

            # normalize
            segments = normalize_segments(segments)

            if not segments:
                # No speech detected, still advance offset conservatively
                offset = last_emitted_end
                continue

            # Shift timestamps and append
            for seg in segments:
                seg.start += offset
                seg.end += offset
                all_segments.append(seg)

                if progress_bar:
                    progress_bar.update(seg.end - seg.start)

            # Update offset only once
            offset = segments[-1].end
            last_emitted_end = offset

            clear_gpu_cache()

        # Write a single merged VTT
        all_segments.sort(key=lambda s: s.start)

        with open(vtt_filepath, "w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for seg in all_segments:
                start_ts = format_vtt_timestamp(seg.start)
                end_ts = format_vtt_timestamp(seg.end)
                text = seg.text.strip()
                vtt_file.write(f"{start_ts} --> {end_ts}\n{text}\n\n")
                if verbose:
                    logger.info("[%s --> %s] %s", start_ts, end_ts, text)

        if progress_bar:
            # We cheat a little to make the user feel warm and fuzzy
            progress_bar.n = progress_bar.total  # Force progress to 100%
            progress_bar.refresh()  # Refresh to show updated bar
            progress_bar.close()

        duration = time.time() - start_time
        hours, remainder = divmod(int(duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("Transcript saved to: %s in %s", vtt_filepath, f"{f'{hours}h ' if hours else ''}{minutes}m {seconds}s")

        # Update hash tracker
        with open(hash_tracker, "a", encoding="utf-8") as hf:
            hf.write(hash_key + "\n")
        existing_hashes.add(hash_key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
