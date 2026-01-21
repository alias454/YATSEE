#!/usr/bin/env python3
"""
yatsee_download_audio.py

Layer 1 Download: Fetch audio from YouTube sources defined in YATSEE entity
configurations, producing raw audio files for transcript generation.

Input/Output:
  - Input: YouTube channel/playlist paths from entity_handle config (Layer 1 Download)
  - Output: MP3 audio files in 'downloads/' under entity_handle path

Dependencies:
  - yt-dlp for YouTube extraction
  - toml for global + entity config parsing
  - argparse, os, sys for CLI and filesystem handling
  - FFmpeg (via yt-dlp postprocessor) for audio extraction

Usage Examples:
  ./yatsee_download_audio.py -e entity_handle
  ./yatsee_download_audio.py -e entity_handle --date-after 20230101 --dry-run

Design Notes:
  - Strictly follows global + flattened entity configs; fails loudly on missing keys
  - Merges global defaults with local entity overrides without heuristics
  - Supports optional date filtering and dry-run mode
  - Modular functions for config loading, YouTube resolution, and download execution
  - Outputs audio ready for Layer 2 transcription and punctuation polishing

Idempotency:
  - Tracker file '.downloaded' ensures files aren't re-downloaded
  - Playlist cache '.playlist_ids.json' prevents repeated YouTube queries
"""

# Standard library
import argparse
import logging
import os
import random
import re
import sys
import textwrap
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Third-party imports
import json
import toml
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError


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

    # Merge: system -> global entity -> local reserved -> local
    merged = {**system_cfg, **entities_cfg[entity], "entity": entity, "root_data_dir": root_data_dir}
    for key, value in local_cfg.items():
        if key in reserved_keys:
            merged.update(value)
        else:
            merged[key] = value
    return merged


def clean_downloaded_file(video_id: str, output_dir: str) -> str:
    """
    Normalize and safely rename a downloaded audio file.

    - Sanitizes filenames to lowercase alphanumeric with safe separators
    - Avoids overwriting existing files
    - Resolves collisions with numeric suffixes

    :param video_id: Prefix of the downloaded file to locate
    :param output_dir: Directory where downloaded files reside
    :return: Sanitized filename in output_dir, or empty string if not found
    :raises OSError: On filesystem failures (e.g., rename, scan)
    """
    # Scan directory for first file starting with video_id
    original = None
    with os.scandir(output_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.startswith(video_id):
                original = entry.name
                break

    if not original:
        # No file found for this video ID
        return ""

    # Normalize filename: lowercase, safe chars only, collapse multiple underscores
    cleaned = original.lower()
    cleaned = re.sub(r"[\"'‘’“”⧸]", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9._-]", "_", cleaned)
    cleaned = re.sub(r"_-_", "_", cleaned)
    cleaned = re.sub(r"__+", "_", cleaned)

    # Rename only if the sanitized filename differs
    if cleaned == original:
        return cleaned

    original_path = os.path.join(output_dir, original)
    cleaned_path = os.path.join(output_dir, cleaned)

    if not os.path.exists(cleaned_path):
        os.rename(original_path, cleaned_path)
        return cleaned

    # Resolve collisions
    base, ext = os.path.splitext(cleaned)
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        candidate_path = os.path.join(output_dir, candidate)
        if not os.path.exists(candidate_path):
            os.rename(original_path, candidate_path)
            return candidate
        counter += 1


def load_cached_playlist(path: str, max_age_seconds: int) -> Optional[list[str]]:
    """
    Load a playlist cache file if it exists and is recent enough.

    Returns None if the cache is missing, invalid, or expired.

    Args:
        path: Path to the cache file.
        max_age_seconds: Maximum allowed cache age in seconds.

    Returns:
        A list of video IDs if the cache is valid, otherwise None.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    fetched_at = payload.get("fetched_at")
    video_ids = payload.get("video_ids")

    if not isinstance(fetched_at, (int, float)):
        return None

    if not isinstance(video_ids, list):
        return None

    if time.time() - fetched_at > max_age_seconds:
        return None

    return [str(vid) for vid in video_ids if isinstance(vid, (str, int))]


def save_cached_playlist(path: str, channel: str, video_ids: list[str]) -> None:
    """
    Save a list of video IDs to a cache file with a timestamp.

    Ensures parent directories exist before writing.

    :param path: Path to the cache file
    :param channel: YouTube channel slug
    :param video_ids: List of video IDs to store
    """
    payload = {
        "fetched_at": int(time.time()),
        "channel": channel,
        "video_ids": [str(vid) for vid in video_ids if vid],
    }

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save cache file '{path}': {e}") from e


def get_video_ids(youtube_path: str) -> List[str]:
    """
    Extract video IDs from a YouTube channel or playlist in flat mode.

    Uses yt-dlp extract_flat to avoid downloading content.

    :param youtube_path: Channel or playlist path (e.g., @handle/streams)
    :return: List of video IDs
    """
    url = f"https://www.youtube.com/{youtube_path.lstrip('/')}"
    opts = {
        "quiet": True,
        "extract_flat": True,  # exactly like --flat-playlist
        "skip_download": True,  # don’t even try to download anything
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return [e["id"] for e in info.get("entries", []) if e and "id" in e]


def get_video_upload_date(video_id: str, js_runtime: str = "deno") -> Optional[str]:
    """
    Get the upload date of a YouTube video in YYYYMMDD format.

    Returns None if the date cannot be determined.

    :param js_runtime: Use specific app for JS challenges
    :param video_id: YouTube video ID
    :return: Upload date string or None
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
        "no_warnings": True,
        "js_runtimes": {
            js_runtime: {}
        },
    }

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except (DownloadError, ExtractorError):
        return None
    except OSError:
        # subprocess / runtime / filesystem weirdness
        return None

    upload_date = info.get("upload_date")
    if isinstance(upload_date, str):
        return upload_date

    return None


def get_tracked_ids(tracker_file: str, dry_run: bool = False) -> set[str]:
    """
    Load previously processed video IDs from tracker file.

    Ensures idempotent downloads by skipping already downloaded videos.
    Creates the tracker file if missing and not in dry-run mode.

    :param tracker_file: Path to the .downloaded tracker file
    :param dry_run: Do not create tracker file if True
    :return: Set of tracked video IDs
    """
    tracked_ids = set()

    if os.path.exists(tracker_file):
        with open(tracker_file, "r", encoding="utf-8") as f:
            tracked_ids = {line.strip() for line in f if line.strip()}
    else:
        if not dry_run:
            # Ensure the file exists for later appends
            open(tracker_file, "a", encoding="utf-8").close()

    return tracked_ids


def download_audio(video_id: str, output_dir: str, js_runtime: str = "deno", dry_run: bool = False) -> Dict[str, Any]:
    """
    Download a single YouTube video's audio using yt-dlp with behavior identical
    to the reference shell script invocation.

    Design constraints:
    - Explicitly forces Node.js for JS execution (--js-runtimes node)
    - Uses a fixed desktop Chrome user-agent
    - Applies identical retry, fragment retry, throttling, and sleep behavior
    - Performs no directory creation beyond validating the output path
    - Returns structured status instead of printing
    - Raises exceptions for the caller to handle

    # :param max_retries: Total retry attempts for failed downloads
    # :param fragment_retries: Retry attempts per media fragment
    # :param concurrent_fragments: Number of fragments downloaded in parallel
    # :param min_sleep: Minimum sleep interval between requests (seconds)
    # :param max_sleep: Maximum sleep interval between requests (seconds)
    # :param throttled_rate: Bandwidth throttle rate (e.g. "500K")
    # :param user_agent: Explicit user-agent string to send with requests

    :param video_id: YouTube video ID to download
    :param output_dir: Directory where audio files will be written
    :param js_runtime: Use specific app for JS challenges
    :param dry_run: If True, resolves configuration but performs no download
    :return: Dictionary describing the download result
    :raises RuntimeError: If the download fails
    """
    # Throttling and retry logic
    min_sleep = 3  # Wait at least 3 seconds between downloads.
    max_sleep = 7  # Adds jitter — randomizes the wait time between 3 and 7 seconds to mimic human behavior.
    max_retries = 3  # Limits total retries for a failed download to 3 attempts.
    fragment_retries = 2  # Controls how many times to retry individual video fragments
    concurrent_fragments = 1  # Only download one video chunk at a time (slower but safer)
    throttled_rate = "500K"  # Caps download bandwidth at 500 KB/s. Adjust based on network speed and how nice you want to be.

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"

    # Construct canonical YouTube watch URL
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Output template must match shell script semantics
    output_template = os.path.join(output_dir, "%(id)s.%(title)s.%(ext)s")

    # yt-dlp configuration mirroring CLI flags exactly
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,

        # Retry and fragment behavior
        "retries": max_retries,
        "fragment_retries": fragment_retries,
        "concurrent_fragments": concurrent_fragments,

        # Throttling and pacing
        "sleep_interval": min_sleep,
        "max_sleep_interval": max_sleep,
        "throttled_rate": throttled_rate,

        # Critical parity flags
        "js_runtimes": {
            js_runtime: {}
        },
        "user_agent": user_agent,

        # Audio extraction
        # "postprocessors": [{
        #     "key": "FFmpegExtractAudio",
        #     "preferredcodec": "flac",
        # }],

        # Silence yt-dlp; caller owns logging
        "quiet": False,
        "no_warnings": False,
    }

    # Randomized politeness delay matches shell script intent
    time.sleep(random.uniform(min_sleep, max_sleep))

    if dry_run:
        return {
            "video_id": video_id,
            "url": url,
            "status": "dry-run",
            "output_template": output_template
        }

    # Execute download
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise RuntimeError(f"yt-dlp download failed for {video_id}: {exc}") from exc

    return {
        "status": "downloaded",
        "output_dir": output_dir
    }


def off_peak_warning() -> bool:
    """
    Check whether the current local hour is within the off-peak window (1am–6am).

    Returns:
        bool: True if within off-peak hours, False otherwise.
    """
    current_hour = datetime.now().hour
    return 1 <= current_hour <= 6


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download YouTube audio for YATSEE processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Requirements:
              - Python 3.10+
              - yt-dlp installed and available in PATH
              - ffmpeg installed for audio extraction
              - Output format: MP3

            Usage Examples:
              python yatsee_download_youtube.py -e defined_entity
              python yatsee_download_youtube.py --youtube-path @channel/streams --output-dir ./audio
              python yatsee_download_youtube.py -e defined_entity --date-after 20260101
              python yatsee_download_youtube.py --dry-run
              python yatsee_download_youtube.py -e defined_entity --make-playlist
        """)
    )
    parser.add_argument("-e", "--entity", required=True, help="Entity handle")
    parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to global config")
    parser.add_argument("-o", "--output-dir", help="Output directory for MP3 audio")
    parser.add_argument("--js-runtime", choices=["deno", "node", "quickjs", "quickjs-ng", "bun"], help="Specify the JS runtime for yt-dlp (default: deno)")
    parser.add_argument("--youtube-path", help="YouTube channel/playlist path (e.g. @handle/streams)")
    parser.add_argument("--date-after", default="", help="Only include videos after YYYYMMDD (e.g. 20250601)")
    parser.add_argument("--date-before", default="", help="Only include videos before YYYYMMDD (e.g. 20251231)")
    parser.add_argument("--dry-run", action="store_true", help="Resolve URLs without downloading")
    parser.add_argument("--make-playlist", action="store_true", help="Create a playlist cache file and exit")
    args = parser.parse_args()

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
        if not args.output_dir:
            logging.error("Without --entity, --output-dir must be defined")
            return 1

    # Set up custom logger
    logger = logger_setup(global_cfg.get("system", {}))

    # Warn if running outside off-peak
    # if not off_peak_warning():
    #     logger.warning("Current hour (%s) is outside off-peak window (1am–6am).",datetime.now().hour)

    # Use output_dir if specified; else fall back to entity data_path
    data_path = args.output_dir or entity_cfg.get("data_path")
    if not data_path:
        logger.error("No valid data path found")
        return 1

    # Ensure output directory exists
    output_dir = args.output_dir or os.path.join(data_path, "downloads")
    if not os.path.isdir(output_dir) and not args.dry_run:
        logger.info("Output directory will be created: %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Read existing tracker file for idempotent file management
    tracker_file = os.path.join(output_dir, ".downloaded")
    tracked_ids = get_tracked_ids(tracker_file, dry_run=args.dry_run)

    # Pick a JS runtime https://github.com/yt-dlp/yt-dlp/issues/15012
    js_runtime = (
            args.js_runtime
            or entity_cfg.get("js_runtime")
            or global_cfg.get("system", {}).get("default_js_runtime", "deno")
    )

    # YouTube Path
    youtube_source = args.youtube_path or entity_cfg.get("sources", {}).get("youtube")
    if isinstance(youtube_source, dict):
        youtube_path = youtube_source.get("youtube_path")
    else:
        youtube_path = youtube_source
    if not youtube_path:
        logger.error("Missing YouTube source path")
        return 1

    # Create cache file for video ID
    # or 2 * 24 * 60 * 60, or 3 days, you’re an adult
    CACHE_MAX_AGE = 24 * 60 * 60  # 24 hours

    cache_file = os.path.join(output_dir, ".playlist_ids.json")
    video_ids = load_cached_playlist(cache_file, CACHE_MAX_AGE)
    if video_ids is None:
        logger.info("Playlist cache missing, invalid, or expired. Rebuilding.")
        video_ids = get_video_ids(youtube_path)
        save_cached_playlist(cache_file, youtube_path, video_ids)

    logger.info("Downloaded %d of %d resolved video IDs from %s",len(tracked_ids), len(video_ids), youtube_path)
    if args.dry_run:
        logger.info("Dry run enabled; exiting before download operations")
        return 0
    if args.make_playlist:
        logger.info("Playlist cache created for %s; exiting", youtube_path)
        return 0

    # DATEAFTER =${2: -$(date - d "-90 day" + %Y %m %d)}
    #     DATEAFTER = 20241231
    date_after = args.date_after
    if not date_after:
        date_after = (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")

    # DATEBEFORE =${3: -$(date - d "+1 day" + %Y %m %d)}
    #     DATEBEFORE = 20260101
    date_before = args.date_before
    if not date_before:
        date_before = (datetime.today() + timedelta(days=1)).strftime("%Y%m%d")

    # Loop through the list of IDs to process
    for video_id in video_ids:
        if video_id in tracked_ids:
            logger.debug("Skipping %s (already processed)", video_id)
            continue

        upload_date = get_video_upload_date(video_id, js_runtime=js_runtime)
        if not upload_date:
            logger.warning("Skipping %s (no upload date)", video_id)
            continue

        if date_before and upload_date > date_before:
            logger.warning("Skipping %s (upload date in the future: %s)",video_id, upload_date)
            continue

        if date_after and upload_date < date_after:
            logger.info("Reached cutoff date with %s (%s); stopping",video_id, upload_date)
            break

        # We made it here so we can download the audio
        result = download_audio(video_id, output_dir, js_runtime=js_runtime, dry_run=args.dry_run)
        if result.get("status") == "downloaded":
            cleaned_file = clean_downloaded_file(video_id, output_dir)
            if cleaned_file:
                with open(tracker_file, "a", encoding="utf-8") as fh:
                    fh.write(video_id + "\n")
                    logger.info("Downloaded and cleaned: %s", cleaned_file)
            else:
                logger.warning("Downloaded file for %s not found after download", video_id)
        else:
            logger.error("Download failed: %s", result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
