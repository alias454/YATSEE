#!/bin/bash
#
# yatsee_format_audio.sh — Convert audio/video to mono 16kHz WAV or FLAC for Whisper
#
# Usage:
#   Run from project root directory. Assumes audio files are in ./downloads/ will output converted audio to ./audio/
#   Tracks processed files in ./audio/.converted to avoid duplicate conversions.
#
# Converts:
#   - .mp4 and .webm → .wav or .flac (mono, 16kHz)
#
# Requirements:
#   - ffmpeg
#   - bash 4+ (for mapfile and arrays)
#
# TODO: Add --force flag to overwrite existing .wav files without prompting.
# TODO: Add a --quiet mode to suppress ffmpeg output.
# TODO: Verify successful conversion with exit code checks or file existence validation.

set -euo pipefail
shopt -s nullglob

downloads='./downloads'
audio_dir='./audio'
id_tracker="${audio_dir}/.converted"

# Determine output format from first argument or default to flac
output_format="${1:-flac}"

# Set ffmpeg codec and extension based on output format
case "$output_format" in
  wav)
    codec="pcm_s16le"
    ext="wav"
    ;;
  flac)
    codec="flac"
    ext="flac"
    ;;
  *)
    echo "Unsupported output format: $output_format"
    exit 1
    ;;
esac

# Make sure audio_dir exists
mkdir -p "${audio_dir}"

# Create the file for tracking ids
touch "${id_tracker}"

force=false
# Uncomment to force reconversion
# if [[ "${1:-}" == "--force" ]]; then
#   force=true
#   echo "⚠️  Force mode enabled: all files will be reprocessed"
# fi

# Function: Get sha256 hash of source audio file
get_file_hash() {
  sha256sum "${1}" | awk '{print $1}'
}

# Gather .mp4 and .webm files from downloads
mapfile -t audio_files < <(find "${downloads}" -type f \( -iname "*.m4a" -o -iname "*.mp4" -o -iname "*.webm" \))

total=${#audio_files[@]}
i=0
for audio_file in "${audio_files[@]}"; do
  file_hash=$(get_file_hash "${audio_file}")
  already_seen=$(grep -c "^${file_hash}$" "${id_tracker}" || true)

  # Skip if already processed
  if [[ $already_seen -gt 0 && $force == false ]]; then
    echo "✓ Skipping ${audio_file} (already converted)"
    continue
  fi

  # Prepare names and paths
  file_name="$(basename "${audio_file}")"
  out_file="${audio_dir}/${file_name%.*}.${ext}"

  ((++i))
  echo "[${i}/${total}] Converting: ${audio_file} → ${out_file}"
  # Run ffmpeg command to convert to FLAC or WAV
  if ffmpeg -y -vn -i "${audio_file}" -ar 16000 -ac 1 -sample_fmt s16 -c:a "${codec}" "${out_file}"; then
    echo "✓ Created ${ext^^}: ${out_file}"

    if [[ $already_seen -eq 0 ]]; then
      echo "${file_hash}" >> "${id_tracker}"
    fi
  else
    echo "❌ Failed to convert: ${audio_file}" >&2
  fi
done
