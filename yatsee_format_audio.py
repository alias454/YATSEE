#!/usr/bin/env python3
"""
yatsee_format_audio.py

Stage 2 of YATSEE: Normalize downloaded media into mono 16kHz WAV or FLAC
for Whisper-style transcription models.

Purpose:
  - Converts raw downloaded media to a transcription-ready format.
  - Optionally splits long audio into sequential chunks with overlap.
  - Tracks processed files via SHA-256 to prevent redundant work.

Input/Output:
  - Input: Media files (.mp4, .webm, .m4a) located in 'downloads/'
    under the entity_handle data directory or a direct input path
  - Output: Normalized audio (.wav or .flac) in 'audio/' under entity_handle
  - Output (optional): Sequential chunks in 'audio/chunks/<base_name>/'

Key Features:
  - Dry-run mode to preview operations without writing files
  - Force mode to reprocess already converted files
  - Automatic creation of output directories if missing
  - Uses external ffmpeg and ffprobe tools for conversion and duration checks
  - Safe handling of invalid files and unexpected exceptions

Dependencies:
  - ffmpeg
  - ffprobe
  - toml (for global/entity configuration parsing)

Usage Examples:
  ./yatsee_format_audio.py -e entity_handle --create-chunks --force
  ./yatsee_format_audio.py --data-path ./data/entity_handle --format wav

Design Notes:
  - Resolves entity_handle from global yatsee.toml and optional local config
  - Tracks processed files via SHA-256 hashes
  - Modular design: config loading, file discovery, hashing, conversion, and chunking
  - Preserves original media; chunks are additional outputs
  - Defaults used when config entries are missing or incomplete
"""

# Standard library
import argparse
import hashlib
import os
import subprocess
import sys
import textwrap
from typing import Any, Dict, List, Set

# Third-party imports
import toml

SUPPORTED_INPUT_EXTENSIONS = (".m4a", ".mp4", ".webm")


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


def compute_sha256(path: str) -> str:
    """
    Compute the SHA-256 hash of a file for tracking processed audio.

    Useful for deduplication and preventing redundant conversions.

    :param path: Absolute path to the input file
    :return: Hex-encoded SHA-256 hash string
    :raises RuntimeError: If the file cannot be read
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
    Load previously converted audio file hashes to avoid reprocessing.

    :param tracker_path: Path to the '.converted' tracker file
    :return: Set of SHA-256 hash strings
    """
    if not os.path.exists(tracker_path):
        return set()
    try:
        with open(tracker_path, "r", encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}
    except OSError:
        return set()


def discover_files(input_path: str, supported_exts) -> List[str]:
    """
    Collect valid audio files from a directory or a single file.

    - Recursively checks only the specified directory or single file
    - Filters by allowed file extensions
    - Returns sorted list for predictable processing order

    :param input_path: Path to a directory or single file
    :param supported_exts: Tuple of allowed file extensions (e.g., '.mp4', '.m4a')
    :return: Sorted list of file paths matching the extensions
    :raises FileNotFoundError: If the path does not exist
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
            raise ValueError(f"Unsupported audio extension: {os.path.splitext(input_path)[1]}")
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

    return sorted(files)


def get_audio_duration(input_file: str)-> tuple[bool, float | None, str]:
    """
    Determine the duration of an audio file in seconds using ffprobe.

    Does not load the audio into memory; relies on external ffprobe call.

    :param input_file: Path to the FLAC or WAV audio file
    :return: Tuple(success: bool, duration: float | None, message: str)
        - success: True if duration was obtained
        - duration: Total audio length in seconds if successful, else None
        - message: Status or error description
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())

        return True, duration, f"Duration: {duration}s"
    except subprocess.CalledProcessError as e:
        return False, None, f"ffprobe failed for {input_file}: {e.stderr.strip()}"
    except Exception as e:
        return False, None, f"Unexpected error for {input_file}: {e}"


def chunk_audio_file(input_file: str, output_dir: str, total_duration: float, chunk_duration: int = 600, overlap: int = 2) -> tuple[bool, list[str], str]:
    """
    Split a long audio file into sequential smaller chunks.

    - Output files are named sequentially with zero-padded indices
    - Optional overlap prevents cutting words or phrases at boundaries
    - Does not modify the original input file

    :param input_file: Path to the source audio file
    :param output_dir: Directory where chunk files will be written
    :param total_duration: Total length of the audio in seconds
    :param chunk_duration: Duration of each chunk in seconds (default 600)
    :param overlap: Overlap in seconds between consecutive chunks (default 2)
    :return: Tuple(success: bool, chunks: List[str], message: str)
        - success: True if chunking succeeded
        - chunks: List of created chunk file paths
        - message: Status or error description
    :raises RuntimeError: If ffmpeg fails to generate chunks
    """
    chunks = []
    start = 0
    idx = 0

    # Iterate over the audio, producing chunks until the end
    try:
        while start < total_duration:
            actual_duration = min(chunk_duration, total_duration - start)
            out_file = os.path.join(output_dir, f"{idx:03d}.flac")
            chunks.append(out_file)

            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_file,
                "-ss", str(max(0, start)),
                "-t", str(actual_duration),
                "-c", "copy",
                out_file
            ]

            # Run ffmpeg safely
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            idx += 1
            start += chunk_duration - overlap

        return True, chunks, f"Created {len(chunks)} chunks in {output_dir}"
    except subprocess.CalledProcessError as e:
        return False, chunks, f"ffmpeg failed for chunk {idx}: {e.stderr.decode(errors='ignore')}"
    except Exception as e:
        return False, chunks, f"Unexpected error during chunking: {e}"


def format_audio(input_src: str, output_path: str, file_format: str = "flac") -> tuple[bool, str]:
    """
    Convert media files to mono 16kHz audio for transcription.

    - Supports WAV or FLAC output
    - Returns success status and message; caller decides further action
    - Does not log or raise exceptions; handles errors internally

    :param input_src: Path to the input media file
    :param output_path: Full path for the normalized output file
    :param file_format: Desired audio format: 'wav' or 'flac'
    :return: Tuple(success: bool, message: str)
        - success: True if conversion succeeded
        - message: Status description
    """
    if file_format not in {"wav", "flac"}:
        return False, f"Unsupported format: {file_format}"

    codec = "pcm_s16le" if file_format == "wav" else "flac"

    cmd = [
        "ffmpeg", "-y", "-vn",
        "-i", input_src,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        "-c:a", codec,
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True, f"Converted successfully: {output_path}"
    except subprocess.CalledProcessError as e:
        return False, f"ffmpeg failed for {input_src}: {e.stderr.decode(errors='ignore')}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Format and optionally chunk audio files for YATSEE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Requirements:
              - Python 3.10+
              - ffmpeg installed and available in PATH
              - Supported input audio formats: mp3, wav, flac, etc.

            Usage Examples:
              python yatsee_format_audio.py -e defined_entity
              python yatsee_format_audio.py --input-dir ./raw_audio --format wav
              python yatsee_format_audio.py -e defined_entity --create-chunks --chunk-duration 300
              python yatsee_format_audio.py --dry-run
              python yatsee_format_audio.py -i ./raw_audio -o ./formatted_audio --force
        """)
    )
    parser.add_argument("-e", "--entity", help="Entity handle to process")
    parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to the global YATSEE configuration file")
    parser.add_argument("-i", "--input-dir", help="Direct override path to entity data directory")
    parser.add_argument("-o", "--output-dir", help="Directory to save audio")
    parser.add_argument("--format", default="flac", choices=["wav", "flac"], help="Output audio format")
    parser.add_argument("--create-chunks", action="store_true", help="Split output audio into chunks")
    parser.add_argument("--chunk-duration", type=int, default=600, help="Chunk duration in seconds")
    parser.add_argument("--chunk-overlap", type=int, default=2, help="Chunk overlap in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without changing files")
    parser.add_argument("--force", action="store_true", help="Reprocess files even if already converted")
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
        # Require both input/output if no entity is provided
        if not args.input_dir or not args.output_dir:
            print("❌ Without --entity, both --input-dir and --output-dir must be defined", file=sys.stderr)
            return 1

    # Use input_dir if specified; else fall back to entity data_path
    downloads_path = args.input_dir or os.path.join(entity_cfg.get("data_path"), "downloads")
    input_files = discover_files(downloads_path, SUPPORTED_INPUT_EXTENSIONS)
    if not input_files:
        print("↪ No input files found", file=sys.stderr)
        return 0

    print(f"🔍 Discovered {len(input_files)} files to convert.")

    audio_out = args.output_dir or os.path.join(entity_cfg.get("data_path"), "audio")
    if not os.path.isdir(audio_out) and not args.dry_run:
        print(f"✓ Output directory will be created: {audio_out}", file=sys.stderr)
        os.makedirs(audio_out, exist_ok=True)

    # Transcription processes see a quality improvement when large audio files are chunked
    chunk_root_dir = os.path.join(audio_out, "chunks")
    if not os.path.isdir(chunk_root_dir) and args.create_chunks:
        print(f"✓ Chunk root directory will be created: {chunk_root_dir}", file=sys.stderr)
        os.makedirs(chunk_root_dir, exist_ok=True)

    converted_hashes: Set[str] = set()
    hash_tracker = os.path.join(audio_out, ".converted")
    if not args.force:
        converted_hashes = load_tracked_hashes(hash_tracker)

    # -----------------------------------------
    # Process new audio file outputs
    # -----------------------------------------
    for idx, src_path in enumerate(input_files, start=1):
        file_hash = compute_sha256(src_path)

        if file_hash in converted_hashes and not args.force:
            print(f"[{idx}/{len(input_files)}] Skipping already converted: {src_path}")
            continue

        base_name = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(audio_out, f"{base_name}.{args.format}")

        print(f"[{idx}/{len(input_files)}] Converting: {src_path} → {out_path}")

        if args.dry_run:
            print("  ↪ dry-run: ffmpeg not executed, state not written")
            continue

        try:
            # -----------------------------------------
            # Convert audio to flac or wav
            # -----------------------------------------
            format_success, format_msg = format_audio(input_src=src_path, output_path=out_path, file_format=args.format)
            # out_path is the converted FLAC/WAV
            file_path = out_path

            # -----------------------------------------
            # Output chunks from single audio file
            # -----------------------------------------
            if format_success and args.create_chunks:
                # Ensure output directory exists for chunks
                chunk_out_path = os.path.join(chunk_root_dir, f"{base_name}")
                if not os.path.isdir(chunk_out_path):
                    print(f"✓ Chunk file directory will be created: {chunk_out_path}", file=sys.stderr)
                    os.makedirs(chunk_out_path, exist_ok=True)

                success, total_duration, msg = get_audio_duration(out_path)
                if not success or total_duration is None:
                    print(f"❌ {msg}", file=sys.stderr)
                    continue

                chunk_success, chunks, chunk_msg = chunk_audio_file(
                    input_file=file_path,
                    output_dir=chunk_out_path,
                    total_duration=total_duration,
                    chunk_duration=args.chunk_duration,
                    overlap=args.chunk_overlap
                )
                if chunk_success:
                    print(f"✓ Created {len(chunks)} chunk files", file=sys.stderr)
                else:
                    print(f"❌ Chunking failed: {chunk_msg}", file=sys.stderr)

            if not format_success:
                print(f"❌ Conversion failed: {src_path}, skipping", file=sys.stderr)
                continue

            # Record successful conversion
            with open(hash_tracker, "a", encoding="utf-8") as tracker:
                tracker.write(file_hash + "\n")

            print(f"✓ {format_msg}")
        except Exception as e:
            # Only log unexpected exceptions, don't stop the whole script
            print(f"❌ Unexpected error converting {src_path}: {e}", file=sys.stderr)
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
