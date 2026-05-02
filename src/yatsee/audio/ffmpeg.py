"""
Low-level ffmpeg and ffprobe helpers for YATSEE audio processing.

These helpers isolate external process execution so the higher-level audio
format stage remains focused on pipeline behavior instead of shell details.
"""

from __future__ import annotations

import subprocess
from typing import Optional, Tuple


def get_audio_duration(input_file: str) -> Tuple[bool, Optional[float], str]:
    """
    Determine the duration of an audio file in seconds using ffprobe.

    :param input_file: Path to the FLAC or WAV audio file
    :return: Tuple(success, duration, message)
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return True, duration, f"Duration: {duration}s"
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        return False, None, f"ffprobe failed for {input_file}: {stderr}"
    except Exception as exc:
        return False, None, f"Unexpected error for {input_file}: {exc}"


def format_audio(input_src: str, output_path: str, file_format: str = "flac") -> Tuple[bool, str]:
    """
    Convert media files to mono 16kHz audio for transcription.

    :param input_src: Path to the input media file
    :param output_path: Full path for the normalized output file
    :param file_format: Desired audio format: 'wav' or 'flac'
    :return: Tuple(success, message)
    """
    if file_format not in {"wav", "flac"}:
        return False, f"Unsupported format: {file_format}"

    codec = "pcm_s16le" if file_format == "wav" else "flac"

    cmd = [
        "ffmpeg",
        "-y",
        "-vn",
        "-i",
        input_src,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        "-c:a",
        codec,
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True, f"Converted successfully: {output_path}"
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else str(exc)
        return False, f"ffmpeg failed for {input_src}: {stderr}"


def chunk_audio_file(
    input_file: str,
    output_dir: str,
    total_duration: float,
    chunk_duration: int = 600,
    overlap: int = 2,
) -> Tuple[bool, list[str], str]:
    """
    Split a long audio file into sequential smaller chunks.

    :param input_file: Path to the source audio file
    :param output_dir: Directory where chunk files will be written
    :param total_duration: Total length of the audio in seconds
    :param chunk_duration: Duration of each chunk in seconds
    :param overlap: Overlap in seconds between consecutive chunks
    :return: Tuple(success, chunks, message)
    """
    chunks: list[str] = []
    start = 0
    idx = 0

    try:
        while start < total_duration:
            actual_duration = min(chunk_duration, total_duration - start)
            out_file = f"{output_dir}/{idx:03d}.flac"
            chunks.append(out_file)

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_file,
                "-ss",
                str(max(0, start)),
                "-t",
                str(actual_duration),
                "-c",
                "copy",
                out_file,
            ]

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            idx += 1
            start += chunk_duration - overlap

        return True, chunks, f"Created {len(chunks)} chunks in {output_dir}"
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else str(exc)
        return False, chunks, f"ffmpeg failed for chunk {idx}: {stderr}"
    except Exception as exc:
        return False, chunks, f"Unexpected error during chunking: {exc}"