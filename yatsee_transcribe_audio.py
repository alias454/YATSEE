#!/usr/bin/env python3
"""
Transcribes audio files to WebVTT (.vtt) using OpenAI's Whisper or faster-whisper.

Requirements:
- Python 3.8+
- torch (PyTorch)
- whisper (https://github.com/openai/whisper)
- faster-whisper (optional, https://github.com/guillaumekln/faster-whisper)
- ffmpeg (required for audio decoding)

Installation suggestions:

  # Python packages:
  pip install torch
  pip install --upgrade git+https://github.com/openai/whisper.git
  pip install faster-whisper  # optional, for better performance

  # FFmpeg (system package, choose one):
  sudo dnf install ffmpeg      # Fedora/RHEL
  sudo apt-get install ffmpeg  # Debian/Ubuntu
  brew install ffmpeg          # macOS or use apt / choco / etc.

Usage examples:
  ./yatsee_transcribe_audio.py --audio_input ./audio_files --model small
  ./yatsee_transcribe_audio.py --audio_input ./single_file.mp3 --faster --lang es

Note:
- CUDA GPU is used by default if available.
- Use `--device cpu` to force CPU processing.

TODO: Refactor to use a central print/log function for consistent output control.
TODO: Consider logging to file for batch jobs or audits.
TODO: Optionally support direct .srt conversion if format variation needed.
"""

import argparse
import os
import sys
import time
import toml
import textwrap
import torch
import torchaudio
import importlib.util
from tqdm import tqdm

# Check if faster-whisper is installed
HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None


def get_audio_duration(audio_path):
    """
    Calculate the duration of an audio file in seconds.

    :param audio_path: Path to the audio file.
    :return: Duration of the audio in seconds as a float.
    """
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def load_flat_hotwords_str(config: str, entity_path: str,) -> str | None:
    """
    Load unweighted flat hotwords from a TOML config for a specific entity.
    Includes titles, aliases, and division names (e.g., wards or parishes).

    :param config: Loaded TOML config using a specific entity (e.g., city_council)
    :param entity_path: Dot-separated path to the entity section
                        (e.g. "country.US.state.IL.city_council").
    :return: Comma-separated hotwords string or None if entity not found.
    """
    # Traverse to the target entity
    section = config
    for part in entity_path.split("."):
        section = section.get(part)
        if section is None:
            return None

    hotwords = set()

    # Add courtesy and role titles
    titles_section = section.get("titles", {})
    for title_group in titles_section.values():
        hotwords.update([t.strip() for t in title_group if t.strip()])

    # Add all person aliases
    people_section = section.get("people", {})
    for role_group in people_section.values():
        for _, alias_list in role_group.items():
            hotwords.update([a.strip() for a in alias_list if a.strip()])

    # Add division names if defined
    # divisions = section.get("divisions", {})
    # division_names = divisions.get("names", [])
    # hotwords.update([d.strip() for d in division_names if d.strip()])

    # Return as sorted, comma-separated string
    return ", ".join(sorted(hotwords))


def format_vtt_timestamp(seconds: float) -> str:
    """
    Format seconds into a WebVTT timestamp string (HH:MM:SS.mmm).

    :param seconds: Timestamp in seconds
    :return: Formatted timestamp string for VTT
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def gather_audio_files(input_path: str) -> list[str]:
    """
    Collect a list of audio files from a directory or single file path.

    :param input_path: Path to audio file or directory
    :return: List of valid audio file paths
    :raises FileNotFoundError: If no valid files found
    :raises ValueError: If unsupported file extension encountered
    """
    valid_extensions = (".mp3", ".wav", ".flac", ".m4a")
    audio_files = []

    if os.path.isdir(input_path):
        # Collect all audio files in directory
        for filename in os.listdir(input_path):
            full_path = os.path.join(input_path, filename)
            if os.path.isfile(full_path) and filename.lower().endswith(valid_extensions):
                audio_files.append(full_path)
        if not audio_files:
            raise FileNotFoundError(f"No valid audio files found in directory: {input_path}")
    elif os.path.isfile(input_path):
        # Single file provided
        if input_path.lower().endswith(valid_extensions):
            audio_files.append(input_path)
        else:
            raise ValueError(f"Unsupported audio file extension: {os.path.splitext(input_path)[1]}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return audio_files


def main() -> int:
    """Main entry point for transcribing audio files to VTT format."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to WebVTT (.vtt) using whisper or faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Requirements:
              - Python 3.8+
              - torch (PyTorch)
              - whisper (https://github.com/openai/whisper)
              - faster-whisper (optional, https://github.com/guillaumekln/faster-whisper)
              - ffmpeg (required for audio decoding)

            Usage Examples:
              ./yatsee_transcribe_audio.py --audio_input ./audio_files --model small --faster
              ./yatsee_transcribe_audio.py -i ./single_file.mp3 -d cpu --lang es
              ./yatsee_transcribe_audio.py --audio_input ./audio_folder -o ./transcripts/
        """),
    )
    parser.add_argument(
        "-i", "--audio_input",
        type=str,
        required=True,
        help="Path to audio file or directory containing audio files to transcribe."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)."
    )
    parser.add_argument(
        "-l", "--lang",
        type=str,
        default="en",
        help="Language spoken in audio (e.g. en, es, fr). Default is English."
    )
    parser.add_argument(
        "-d", "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for model execution: 'cuda' for GPU (default if available), 'cpu' for compatibility."
    )
    parser.add_argument(
        "--faster",
        action="store_true",
        help="Use faster-whisper for transcription if installed."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        help="Directory to save .vtt transcript files (default: 'transcripts_<model>')."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress (default)."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output (overrides --verbose)."
    )
    args = parser.parse_args()

    # Get the base and entity keys
    config_file = "yatsee.toml"
    config = toml.load(config_file)
    base = config.get("base", "")
    entity = config.get("entity", "")

    full_key = base + entity

    hotwords = load_flat_hotwords_str(config_file, full_key) or None

    # Verbose output flag
    verbose = args.verbose and not args.quiet

    # Validate and prepare device for model inference
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda"
        use_fp16 = True  # Use fp16 for performance on GPU
    else:
        if args.device == "cuda":
            print("⚠️ CUDA requested but not available, falling back to CPU.", file=sys.stderr)
        device = "cpu"
        use_fp16 = False

    # Collect audio files from input path
    try:
        audio_file_list = gather_audio_files(args.audio_input)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    # Determine output directory, default to model-based folder
    output_directory = args.output_dir or f"transcripts_{args.model}"
    os.makedirs(output_directory, exist_ok=True)

    # Load the model: choose faster-whisper or standard whisper
    if args.faster and HAS_FASTER_WHISPER:
        # Import faster-whisper
        from faster_whisper import WhisperModel
        print(f"🚀 Using faster-whisper model '{args.model}' on device '{device}'...")

        compute_type = "float16" if use_fp16 else "int8"
        whisper_model = WhisperModel(args.model, device=device, compute_type=compute_type)
        use_faster_whisper = True
    else:
        # Import openai whisper
        import whisper
        from whisper.utils import get_writer

        if args.faster and not HAS_FASTER_WHISPER:
            print("⚠️ faster-whisper requested but not installed, falling back to standard whisper.", file=sys.stderr)
        print(f"🐢 Using standard whisper model '{args.model}' on device '{device}'...")

        whisper_model = whisper.load_model(args.model).to(device)
        use_faster_whisper = False

    print(f"🔍 Found {len(audio_file_list)} audio file(s) to transcribe.\n")

    # Transcribe files using selected model
    for audio_path in audio_file_list:
        # Keep track of transcription time per file
        start_time = time.time()

        # Construct output VTT filename
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vtt_filepath = os.path.join(output_directory, f"{base_name}.vtt")

        if use_faster_whisper:
            # Transcribe with faster-whisper
            print(f"🚀 Transcribing '{audio_path}' with faster-whisper...")
            try:
                segments, _ = whisper_model.transcribe(audio_path, hotwords=hotwords, beam_size=5, language=args.lang)

                # Not really necessary but this gives a similar output to standard whisper
                progress_bar = None
                if not verbose:
                    audio_duration = get_audio_duration(audio_path)
                    progress_bar = tqdm(
                        total=audio_duration,
                        unit="sec",
                        desc="Transcribing",
                        dynamic_ncols=True,
                        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} {unit} [{elapsed}<{remaining}, {rate_fmt}]"
                    )

                with open(vtt_filepath, "w", encoding="utf-8") as vtt_file:
                    vtt_file.write("WEBVTT\n\n")
                    for segment in segments:
                        start_ts = format_vtt_timestamp(segment.start)
                        end_ts = format_vtt_timestamp(segment.end)
                        text = segment.text.strip()
                        vtt_file.write(f"{start_ts} --> {end_ts}\n{text}\n\n")
                        if verbose:
                            print(f"[{start_ts} --> {end_ts}] {text}", flush=True)
                        elif progress_bar:
                            # Display progress bar as the chunks are written
                            progress_bar.update(segment.end - segment.start)

                if progress_bar:
                    # We cheat a little to make the user feel warm and fuzzy
                    progress_bar.n = progress_bar.total  # Force progress to 100%
                    progress_bar.refresh()  # Refresh to show updated bar
                    progress_bar.close()

                # Keep track of transcription time per file
                duration = time.time() - start_time
                print(f"✅ Transcript saved to: {vtt_filepath} in {duration:.1f}s")
            except Exception as err:
                print(f"❌ Error transcribing '{audio_path}' with faster-whisper: {err}", file=sys.stderr)
        else:
            # Transcribe with OpenAI whisper
            print(f"🐢 Transcribing '{audio_path}' with standard whisper...")
            try:
                result = whisper_model.transcribe(audio_path, initial_prompt=hotwords, verbose=verbose, language=args.lang, fp16=use_fp16)
                writer = get_writer("vtt", output_directory)
                writer(result, audio_path, {"output_filename": base_name})

                # Keep track of transcription time per file
                duration = time.time() - start_time
                print(f"✅ Transcript saved to: {vtt_filepath} in {duration:.1f}s")
            except Exception as err:
                print(f"❌ Error transcribing '{audio_path}' with whisper: {err}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
