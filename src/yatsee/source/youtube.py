"""
YouTube source adapter for YATSEE.

This module provides the first source adapter behind the new source fetch
command family. It intentionally keeps the first package migration narrower
than the legacy standalone downloader while preserving the important behavior:

- entity-driven config resolution
- source path lookup from local config
- output into downloads/
- tracker-based idempotency
- optional date filtering
- playlist cache reuse
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError

from yatsee.core.config import load_entity_config, load_global_config
from yatsee.core.errors import ConfigError, ValidationError
from yatsee.core.tracking import append_tracker_value, load_tracker_set


def clean_downloaded_file(video_id: str, output_dir: str) -> str:
    """
    Normalize and safely rename a downloaded file.

    :param video_id: Prefix of the downloaded file to locate
    :param output_dir: Directory where downloaded files reside
    :return: Sanitized filename in output_dir, or empty string if not found
    :raises ConfigError: On filesystem failures
    """
    original = None
    try:
        with os.scandir(output_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.startswith(video_id):
                    original = entry.name
                    break
    except OSError as exc:
        raise ConfigError(f"Failed to scan output directory '{output_dir}': {exc}") from exc

    if not original:
        return ""

    cleaned = original.lower()
    cleaned = re.sub(r"[\"'‘’“”⧸]", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9._-]", "_", cleaned)
    cleaned = re.sub(r"_-_", "_", cleaned)
    cleaned = re.sub(r"__+", "_", cleaned)

    if cleaned == original:
        return cleaned

    original_path = os.path.join(output_dir, original)
    cleaned_path = os.path.join(output_dir, cleaned)

    try:
        if not os.path.exists(cleaned_path):
            os.rename(original_path, cleaned_path)
            return cleaned

        base, ext = os.path.splitext(cleaned)
        counter = 1
        while True:
            candidate = f"{base}_{counter}{ext}"
            candidate_path = os.path.join(output_dir, candidate)
            if not os.path.exists(candidate_path):
                os.rename(original_path, candidate_path)
                return candidate
            counter += 1
    except OSError as exc:
        raise ConfigError(f"Failed to rename downloaded file '{original_path}': {exc}") from exc


def load_cached_playlist(path: str, max_age_seconds: int) -> Optional[List[str]]:
    """
    Load a playlist cache file if it exists and is recent enough.

    :param path: Path to the cache file
    :param max_age_seconds: Maximum allowed cache age in seconds
    :return: List of video IDs if cache is valid, otherwise None
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
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

    return [str(video_id) for video_id in video_ids if isinstance(video_id, (str, int))]


def save_cached_playlist(path: str, channel: str, video_ids: List[str]) -> None:
    """
    Save playlist video IDs to a cache file.

    :param path: Path to the cache file
    :param channel: Source channel or playlist identifier
    :param video_ids: List of discovered video IDs
    :raises ConfigError: If writing fails
    """
    payload = {
        "fetched_at": int(time.time()),
        "channel": channel,
        "video_ids": [str(video_id) for video_id in video_ids if video_id],
    }

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:
        raise ConfigError(f"Failed to save playlist cache '{path}': {exc}") from exc


def get_video_ids(youtube_path: str) -> List[str]:
    """
    Extract video IDs from a YouTube channel or playlist in flat mode.

    :param youtube_path: Channel or playlist path
    :return: List of video IDs
    """
    url = f"https://www.youtube.com/{youtube_path.lstrip('/')}"
    opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
    }

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return [entry["id"] for entry in info.get("entries", []) if entry and "id" in entry]


def get_video_upload_date(video_id: str, js_runtime: str = "deno") -> Optional[str]:
    """
    Get a video's upload date in YYYYMMDD format.

    :param video_id: YouTube video ID
    :param js_runtime: Selected JS runtime for yt-dlp
    :return: Upload date string or None
    :raises RuntimeError: On yt-dlp/runtime failures
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
            upload_date = info.get("upload_date")
            if isinstance(upload_date, str):
                return upload_date
            return None
    except (DownloadError, ExtractorError) as exc:
        raise RuntimeError(f"CRITICAL: {video_id}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"SYSTEM_ERROR: {exc}") from exc


def download_audio(video_id: str, output_dir: str, js_runtime: str = "deno") -> Dict[str, Any]:
    """
    Download a single YouTube video's best audio stream.

    :param video_id: YouTube video ID
    :param output_dir: Output directory for downloads
    :param js_runtime: JS runtime for yt-dlp
    :return: Result dictionary
    :raises RuntimeError: If download fails
    """
    min_sleep = 3
    max_sleep = 7
    max_retries = 3
    fragment_retries = 2
    concurrent_fragments = 1
    throttled_rate = "500K"

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = os.path.join(output_dir, "%(id)s.%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "retries": max_retries,
        "fragment_retries": fragment_retries,
        "concurrent_fragments": concurrent_fragments,
        "sleep_interval": min_sleep,
        "max_sleep_interval": max_sleep,
        "throttled_rate": throttled_rate,
        "js_runtimes": {
            js_runtime: {}
        },
        "user_agent": user_agent,
        "quiet": False,
        "no_warnings": False,
    }

    time.sleep(random.uniform(min_sleep, max_sleep))

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except (DownloadError, ExtractorError) as exc:
        raise RuntimeError(f"CRITICAL: {video_id}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"SYSTEM_ERROR: {video_id}: {exc}") from exc

    return {
        "status": "downloaded",
        "video_id": video_id,
        "output_dir": output_dir,
    }


def resolve_youtube_fetch_paths(
    global_config_path: str,
    entity: str,
    output_dir: str | None,
    youtube_path_override: str | None,
    js_runtime_override: str | None,
) -> Dict[str, Any]:
    """
    Resolve config and paths for the YouTube source adapter.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Entity handle
    :param output_dir: Optional output directory override
    :param youtube_path_override: Optional direct YouTube path override
    :param js_runtime_override: Optional JS runtime override
    :return: Resolved config and paths
    :raises ValidationError: If required source config is missing
    """
    global_cfg = load_global_config(global_config_path)
    entity_cfg = load_entity_config(global_cfg, entity)

    data_path = entity_cfg.get("data_path")
    if not data_path:
        raise ValidationError(f"Entity '{entity}' does not define a data_path.")

    resolved_output_dir = output_dir or os.path.join(data_path, "downloads")
    os.makedirs(resolved_output_dir, exist_ok=True)

    youtube_source = entity_cfg.get("sources", {}).get("youtube")
    if isinstance(youtube_source, dict):
        youtube_path = youtube_path_override or youtube_source.get("youtube_path")
        enabled = bool(youtube_source.get("enabled", True))
    else:
        youtube_path = youtube_path_override or youtube_source
        enabled = True

    if not enabled:
        raise ValidationError(f"YouTube source is disabled for entity '{entity}'.")

    if not youtube_path:
        raise ValidationError(f"Missing YouTube source path for entity '{entity}'.")

    js_runtime = (
        js_runtime_override
        or entity_cfg.get("js_runtime")
        or global_cfg.get("system", {}).get("default_js_runtime", "deno")
    )

    return {
        "global_cfg": global_cfg,
        "entity_cfg": entity_cfg,
        "youtube_path": youtube_path,
        "js_runtime": js_runtime,
        "output_dir": resolved_output_dir,
    }


def run_youtube_fetch(
    global_config_path: str,
    entity: str,
    output_dir: str | None = None,
    youtube_path_override: str | None = None,
    js_runtime_override: str | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
    make_playlist: bool = False,
) -> Dict[str, Any]:
    """
    Run the YouTube fetch adapter for an entity.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Entity handle
    :param output_dir: Optional output override
    :param youtube_path_override: Optional YouTube path override
    :param js_runtime_override: Optional JS runtime override
    :param date_after: Optional lower date bound in YYYYMMDD
    :param date_before: Optional upper date bound in YYYYMMDD
    :param make_playlist: Rebuild playlist cache and stop
    :return: Summary dictionary
    """
    resolved = resolve_youtube_fetch_paths(
        global_config_path=global_config_path,
        entity=entity,
        output_dir=output_dir,
        youtube_path_override=youtube_path_override,
        js_runtime_override=js_runtime_override,
    )

    youtube_path = resolved["youtube_path"]
    js_runtime = resolved["js_runtime"]
    downloads_dir = resolved["output_dir"]

    tracker_file = os.path.join(downloads_dir, ".downloaded")
    tracked_ids = load_tracker_set(tracker_file)

    cache_max_age = 24 * 60 * 60
    cache_file = os.path.join(downloads_dir, ".playlist_ids.json")
    video_ids = load_cached_playlist(cache_file, cache_max_age)

    messages: List[str] = []

    if video_ids is None:
        messages.append("Playlist cache missing, invalid, or expired. Rebuilding.")
        video_ids = get_video_ids(youtube_path)
        save_cached_playlist(cache_file, youtube_path, video_ids)
    else:
        messages.append("Using cached playlist IDs.")

    if make_playlist:
        messages.append(f"Playlist cache created for {youtube_path}; exiting.")
        return {
            "entity": entity,
            "source_type": "youtube",
            "youtube_path": youtube_path,
            "downloads_dir": downloads_dir,
            "discovered": len(video_ids),
            "downloaded": 0,
            "skipped": 0,
            "messages": messages,
        }

    resolved_date_after = date_after or (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")
    resolved_date_before = date_before or (datetime.today() + timedelta(days=1)).strftime("%Y%m%d")

    downloaded = 0
    skipped = 0

    for video_id in video_ids:
        if video_id in tracked_ids:
            skipped += 1
            messages.append(f"Skipped already downloaded: {video_id}")
            continue

        try:
            upload_date = get_video_upload_date(video_id, js_runtime=js_runtime)
            if not upload_date:
                skipped += 1
                messages.append(f"Skipped {video_id}: no upload date")
                continue

            if resolved_date_before and upload_date > resolved_date_before:
                skipped += 1
                messages.append(f"Skipped {video_id}: upload date in the future ({upload_date})")
                continue

            if resolved_date_after and upload_date < resolved_date_after:
                messages.append(f"Reached cutoff date with {video_id} ({upload_date}); stopping.")
                break

            result = download_audio(video_id, downloads_dir, js_runtime=js_runtime)
            cleaned = clean_downloaded_file(video_id, downloads_dir)
            append_tracker_value(tracker_file, video_id)

            downloaded += 1
            messages.append(
                f"Downloaded {video_id}"
                + (f" -> {cleaned}" if cleaned else "")
            )
        except RuntimeError as exc:
            if "429" in str(exc):
                raise ConfigError("HTTP 429 detected during YouTube fetch; stopping.")
            raise ConfigError(str(exc)) from exc

    return {
        "entity": entity,
        "source_type": "youtube",
        "youtube_path": youtube_path,
        "downloads_dir": downloads_dir,
        "discovered": len(video_ids),
        "downloaded": downloaded,
        "skipped": skipped,
        "messages": messages,
    }