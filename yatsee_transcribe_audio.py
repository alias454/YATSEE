#!/usr/bin/env python3
"""
yatsee_transcribe_audio.py

Stage 3 of YATSEE: Transcribe audio files to WebVTT using Whisper or faster-whisper.

Input/Output:
  - Input: Audio files (.mp3, .wav, .flac, .m4a) in 'audio/' or specified path under
    entity handle.
    Supports optional chunked audio processing under 'audio/chunks/<file_basename>/'.
  - Output: Transcripts (.vtt) written to 'transcripts_<model>/' under the entity's
    data path or specified output directory. Overlapping or dangling segments are
    merged for clarity.

Dependencies:
  - whisper or faster-whisper
  - torch
  - torchaudio
  - toml (for config parsing)
  - tqdm

Usage Examples:
  ./yatsee_transcribe_audio.py -e entity_handle --audio_input ./audio --model small
  ./yatsee_transcribe_audio.py --audio_input ./downloads/video.mp4 --faster --lang en
  ./yatsee_transcribe_audio.py -i ./single_file.mp3 -d cpu --lang es
  ./yatsee_transcribe_audio.py --audio_input ./audio_folder -o ./transcripts/

Features / Design Notes:
  - Loads flat entity configuration by merging global yatsee.toml with local entity config.
  - Automatically skips already-transcribed files using SHA-256 hash tracking.
  - Supports hotwords extracted from titles and people aliases in entity configs.
  - Single progress bar per file, updated across all chunks for consistent user experience.
  - Cleanly merges overlapping or near-overlapping segments to maintain VTT quality.
  - Works on CPU, CUDA, or Apple MPS, with fallbacks for faster-whisper limitations.
  - Outputs are deterministic, contained within the entity's data path.
"""

import hashlib
import argparse
import os
import sys
import time
import textwrap
import importlib.util

import warnings
import gc
import toml
import torch
import torchaudio
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Set
from types import SimpleNamespace

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, message="torchaudio._backend.utils.info has been deprecated")

# Check if faster-whisper is installed
HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None

SUPPORTED_INPUT_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a")


def load_global_config(path: str) -> Dict[str, Any]:
    """
    Load the global YATSEE configuration file.

    :param path: Path to the global TOML config file
    :return: Parsed global configuration as a dictionary
    :raises FileNotFoundError: If the file does not exist
    :raises ValueError: If the TOML cannot be parsed
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

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle to load (e.g., 'us_ca_fresno_city_council')
    :return: Merged entity configuration dictionary
    :raises KeyError: If entity is not defined in global config
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
    Flatten titles and people aliases into a single comma-separated string.

    :param entity_cfg: Merged entity config
    :return: Comma-separated hotwords string or None if empty
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
    Calculate the duration of an audio file in seconds.

    :param audio_path: Path to the audio file.
    :return: Duration of the audio in seconds as a float.
    """
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except RuntimeError as e:
        print(f"⚠️ Failed to read audio info for '{audio_path}': {e}", file=sys.stderr)
        return 0.0


def discover_files(input_path: str, supported_exts) -> List[str]:
    """
    Collect files from a directory or single file based on allowed extensions.

    :param input_path: Path to directory or file
    :param supported_exts: Supported file extensions tuple
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
    Compute SHA-256 hash of a file for tracking processed media.

    :param path: Absolute path to file
    :return: Hex-encoded SHA-256 hash
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
    Load SHA-256 hashes of already converted audio to prevent redundant work.

    :param tracker_path: Path to .converted tracker file
    :return: Set of SHA-256 hashes
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
    Clear PyTorch CUDA cache and invoke garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def format_vtt_timestamp(seconds: float) -> str:
    """
    Convert seconds to WebVTT timestamp format HH:MM:SS.mmm.

    :param seconds: Time in seconds
    :return: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def normalize_segments(segments):
    """
    Convert segments to objects with 'start', 'end', 'text' attributes.
    Works with dicts or objects.
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
    """Main entry point for transcribing audio files to VTT format."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to WebVTT (.vtt) using whisper or faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Requirements:
              - Python 3.9+
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
    parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to global yatsee.toml")
    parser.add_argument("-i", "--audio-input", help="Audio file or directory (Defaults to ./audio)")
    parser.add_argument("-o", "--output-dir", help="Directory to save transcripts")
    parser.add_argument("-g", "--get-chunks", action="store_true", help="Transcribe using audio chunk files")
    parser.add_argument("-m", "--model", help="Whisper model size (overrides entity/system defaults)")
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
            print(f"❌ Config load failed: {e}", file=sys.stderr)
            return 1
    else:
        # Require both input/output if no entity is provided
        if not args.audio_input or not args.output_dir:
            print("❌ Without --entity, both --audio-input and --output-dir must be defined", file=sys.stderr)
            return 1

    model = args.model or entity_cfg.get("transcription_model", "small")

    # -----------------------------------------
    # Validate device for model inference
    # -----------------------------------------
    if torch.cuda.is_available() and args.device in ["auto", "cuda"]:
        clear_gpu_cache()
        device = "cuda"
        use_fp16 = True  # Use fp16 for performance on GPU
    elif torch.backends.mps.is_available() and args.device in ["auto", "mps"]:
        if args.faster:
            # Handle the Faster-Whisper Edge Case
            print("⚠️ Faster-Whisper does not support MPS (Metal) yet. Using CPU (optimized for ARM with faster-whisper).")
            device = "cpu"
            use_fp16 = False
        else:
            # Apple Silicon Support
            device = "mps"
            use_fp16 = False
    else:
        if args.device == "cuda":
            print("⚠️ CUDA requested but not available, falling back to CPU.", file=sys.stderr)
        if args.device == "mps":
            print("⚠️ MPS requested but not available, falling back to CPU.", file=sys.stderr)
        device = "cpu"
        use_fp16 = False

    #-----------------------------------------
    # Load the model: faster-whisper or whisper
    #-----------------------------------------
    if args.faster and HAS_FASTER_WHISPER:
        # Import faster-whisper
        from faster_whisper import WhisperModel
        print(f"🚀 Using faster-whisper model '{model}' on device '{device}'...")

        compute_type = "float16" if use_fp16 else "int8"
        whisper_model = WhisperModel(model, device=device, compute_type=compute_type)
        use_faster_whisper = True
    else:
        # Import openai whisper
        import whisper
        from whisper.utils import get_writer

        if args.faster and not HAS_FASTER_WHISPER:
            print("⚠️ faster-whisper requested but not installed, falling back to standard whisper.", file=sys.stderr)
        print(f"🐢 Using standard whisper model '{model}' on device '{device}'...")

        whisper_model = whisper.load_model(model).to(device)
        use_faster_whisper = False

    # Use audio_input if specified; else fall back to entity data_path
    audio_directory = args.audio_input or os.path.join(entity_cfg.get("data_path"), "audio")
    audio_file_list = discover_files(audio_directory, SUPPORTED_INPUT_EXTENSIONS)
    if not audio_file_list:
        print("↪ No audio input files found", file=sys.stderr)
        return 0

    output_directory = args.output_dir or os.path.join(entity_cfg.get("data_path"), f"transcripts_{model}")
    if not os.path.isdir(output_directory):
        print(f"✓ Output directory will be created: {output_directory}", file=sys.stderr)
        os.makedirs(output_directory, exist_ok=True)

    # Prepare hotwords
    hotwords = load_flat_hotwords(entity_cfg)

    lang = None if args.lang.lower() == "auto" else args.lang

    hash_tracker = os.path.join(output_directory, ".vtt_hash")
    existing_hashes = load_tracked_hashes(hash_tracker)

    print(f"🔍 Found {len(audio_file_list)} audio file(s) to transcribe.\n")

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
            print(f"✅ Skipping already-transcribed file: {audio_path}")
            continue

        # Chunking support
        chunk_dir = os.path.join(audio_directory, "chunks", base_name)
        if args.get_chunks and os.path.isdir(chunk_dir):
            print(f"Using chunk directory: {chunk_dir}")
            audio_chunks = sorted(
                os.path.join(chunk_dir, f)
                for f in os.listdir(chunk_dir)
                if f.lower().endswith(SUPPORTED_INPUT_EXTENSIONS)
            )
        else:
            audio_chunks = [audio_path]

        # Total duration for progress bar (seconds)
        total_duration = get_audio_duration(audio_path)
        progress_bar = None
        if not verbose:
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
                    segments, _ = whisper_model.transcribe(audio_chunk, hotwords=hotwords, beam_size=5, language=lang, condition_on_previous_text=False)
                except Exception as err:
                    print(f"❌ Error transcribing '{audio_chunk}' with faster-whisper: {err}", file=sys.stderr)
                    continue
            else:
                # -----------------------------------------
                # Transcribe with openai whisper
                # -----------------------------------------
                try:
                    result = whisper_model.transcribe(audio_chunk, initial_prompt=hotwords, verbose=verbose, language=lang, fp16=use_fp16, condition_on_previous_text=False)
                    # Extract segments safely
                    segments = result.get("segments", []) if isinstance(result, dict) else getattr(result, "segments", []) or []
                except Exception as err:
                    print(f"❌ Error transcribing '{audio_chunk}' with whisper: {err}", file=sys.stderr)
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
                    print(f"[{start_ts} --> {end_ts}] {text}", flush=True)

        if progress_bar:
            # We cheat a little to make the user feel warm and fuzzy
            progress_bar.n = progress_bar.total  # Force progress to 100%
            progress_bar.refresh()  # Refresh to show updated bar
            progress_bar.close()

        duration = time.time() - start_time
        print(f"✅ Transcript saved to: {vtt_filepath} in {duration:.1f}s")

        # Update hash tracker
        with open(hash_tracker, "a", encoding="utf-8") as hf:
            hf.write(hash_key + "\n")
        existing_hashes.add(hash_key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
