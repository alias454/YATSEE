#!/bin/bash
#
# process_download_audio.sh — Download audio from YouTube channel streams
#
# Usage:
#   DATEAFTER and DATEBEFORE specify the range of video publish dates to process,
#   formatted as YYYYMMDD. Defaults:
#     DATEAFTER = 20241231
#     DATEBEFORE = 20260101
#
#   Example:
#     ./yatsee_download_audio.sh https://youtube.com/<channel>/streams 20250101 20250501
#
# What it does:
#   - Fetches video IDs from the given YouTube channel within the date range.
#   - Downloads best-quality audio for each video.
#   - Cleans up audio filenames.
#   - Tracks processed video IDs to avoid re-downloading.
#
# Politeness improvements:
#   - Random sleep between 3-7 seconds to avoid hammering servers.
#   - Bandwidth throttling using --throttled-rate (example: 500KB/s).
#   - Retry with exponential backoff (up to 3 retries).
#   - Warns if run outside off-peak hours (1am-6am).
#
# Requirements:
#   - yt-dlp (https://github.com/yt-dlp/yt-dlp)
#   - sed, find, bash 4+ (for associative arrays and features)
#
# Make sure to run this script from a directory where you want audio/ and transcripts/ folders created.
# Outputs:
#   - Audio saved under ./downloads/
#   - Processed video IDs logged in ./downloads/.downloaded
#
# TODO: Add a --quiet or --no-progress option to suppress yt-dlp output if desired.
# TODO: Support subtitle download if available on the video platform.
# TODO: Validate audio format post-download (ffprobe or similar).
# TODO: Add better error handling for failed downloads or conversions.

set -euo pipefail

# Function to clean a filename by replacing/removing unwanted chars
clean_filename() {
  local filename="${1}"
  echo "${filename}" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E '
      s/[\"'\''’‘”“]/_/g;    # Normalize all types of quotes/apostrophes to underscore
      s/⧸/–/g;               # Special slashed character to dash or underscore
      s/[^a-z0-9._-]/_/g;    # Replace all other non-safe characters
      s/_-_/_/g;             # Normalize sequences like "_-_" to "_"
      s/__+/_/g;             # Collapse multiple underscores into one
    '
}

# Simple off-peak hour check (1am-6am local time)
# current_hour=$(date +%H)
# if (( current_hour < 1 || current_hour > 6 )); then
#   echo "Warning: Current hour ($current_hour) outside off-peak window (1am-6am)."
#   echo "Consider running during off-peak hours to reduce load."
# fi

CHANNEL_URL=${1:-"https://www.youtube.com/@cityofAnyTown/streams"}
downloads='./downloads'
id_tracker="${downloads}/.downloaded"

# Date range for videos to process
DATEAFTER=${2:-20241231}
DATEBEFORE=${3:-$(date -d "+1 day" +%Y%m%d)}

# Sleep and retry configs
MIN_SLEEP=3           # Wait at least 3 seconds between downloads.
MAX_SLEEP=7           # Adds jitter — randomizes the wait time between 3 and 7 seconds to mimic human behavior.
MAX_RETRIES=3         # Limits total retries for a failed download to 3 attempts.
FRAGMENT_RETRIES=2    # Controls how many times to retry individual video fragments
CONCURRENT_RETRIES=1  # Only download one video chunk at a time (slower but safer)
THROTTLED_RATE="500K" # Caps download bandwidth at 500 KB/s. Adjust based on network speed and how nice you want to be.

# Make sure audio exists
mkdir -p "${downloads}"

# Create the file for tracking ids
touch "${id_tracker}"

echo "Fetching list of video IDs from the channel..."
video_ids=$(yt-dlp --flat-playlist --get-id "${CHANNEL_URL}")

for audio_id in ${video_ids}; do
    # Skip if already downloaded.
  if [[ -f "${id_tracker}" ]] && grep -Fxq -- "${audio_id}" "${id_tracker}"; then
    echo "✓ Skipping ${audio_id} (already processed)"
    continue
  fi

  # Need to do this since alternatives suck even worse
  upload_date=$(yt-dlp --print "%(upload_date)s" "https://www.youtube.com/watch?v=${audio_id}" 2>/dev/null)

  if [[ -z "${upload_date}" ]]; then
    echo "⚠️ Skipping ${audio_id} (no upload date)"
    continue
  fi

  # Shouldn't happen but just in case
  if [[ "${upload_date}" -gt "${DATEBEFORE}" ]]; then
    echo "⚠️  Skipping ${audio_id} (upload date is in the future: ${upload_date})"
    continue
  fi

  # Stop downloading items outside of the given range --dateafter "${DATEAFTER}" --datebefore "${DATEBEFORE}"
  if [[ "${upload_date}" -lt "${DATEAFTER}" ]]; then
    echo "🛑 Reached cutoff date with ${audio_id} (${upload_date}), stopping."
    break
  fi

  echo "→ Processing ${audio_id}..."

  # For mp3 files -- transcription works better when converted to mono wav from bestaudio
  # yt-dlp \
    # --extract-audio \
    # --audio-format mp3 \
    # --audio-quality 0 \
    # --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36" \
    # -o "${downloads}/%(id)s.%(title)s.%(ext)s" "https://www.youtube.com/watch?v=${audio_id}"
  yt-dlp \
    --retries "${MAX_RETRIES}" \
    --fragment-retries "${FRAGMENT_RETRIES}" \
    --concurrent-fragments "${CONCURRENT_RETRIES}" \
    --sleep-interval "${MIN_SLEEP}" \
    --max-sleep-interval "${MAX_SLEEP}" \
    --throttled-rate "${THROTTLED_RATE}" \
    --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36" \
    -f bestaudio \
    -o "${downloads}/%(id)s.%(title)s.%(ext)s" "https://www.youtube.com/watch?v=${audio_id}"

  # Find the audio file for this video
  original_audio=$(find "${downloads}/" -type f -name "${audio_id}.*" | head -n 1)
  if [[ -z "${original_audio}" ]]; then
    echo "⚠️ No file found for ${audio_id}, skipping renaming."
    continue
  fi
  file_name=$(basename "${original_audio}")
  cleaned_audio="${downloads}/$(clean_filename "${file_name}")"

  if [[ "${original_audio}" != "${cleaned_audio}" ]]; then
    mv "${original_audio}" "${cleaned_audio}"
    echo "Renamed audio file: ${original_audio} → ${cleaned_audio}"
  fi

  echo "${audio_id}" >> "${id_tracker}"
  echo "✓ Fetched ${audio_id}"

  # exit the loop after one run for testing
  # break
done
