#!/bin/bash
#
# yatsee_vtt_to_txt.sh — Convert VTT subtitle files to plain text
#
# Usage:
#   ./yatsee_vtt_to_txt.sh [model] [--output_dir DIR] [--vtt_input DIR] [--force]
#
# Arguments:
#   model:         Whisper model used (small, medium, large). Defaults to "small"
#   --output_dir DIR:  Output directory for .txt files (default: same as input folder)
#   --vtt_input  DIR: Input directory for .vtt files (default: ./transcripts_${model})
#   --force:       Force processing even if file hash is unchanged
#
# Input:
#   - VTT files in ./transcripts_${model}/*.vtt (or overridden with --vtt_input)
# Output:
#   - TXT files in the output directory, one per VTT
#
# TODO: Add a --quiet flag to suppress routine output, keeping only errors.
# TODO: Add a --verbose flag for extra debug details during processing.
# TODO: Convert this to a function-based shell structure for easier sourcing or re-use.
# TODO: Optionally support input/output of file lists instead of scanning a directory.
# TODO: Add --dry-run mode to preview which files would be processed without modifying anything.

set -euo pipefail
shopt -s nullglob

# Defaults
model="small"
force=false
vtt_input=""
output_dir=""

print_help() {
  cat <<EOF
Usage: $0 [--model MODEL] [--output_dir DIR] [--vtt_input DIR] [--force] [--help]

Arguments:
  -m, --model:          Whisper model used (small, medium, large). Defaults to "small"
  -i, --vtt-input DIR   Input directory for .vtt files (default: ./transcripts_{model})
  -o, --output-dir DIR  Output directory for .txt files (default: same as input folder)
  --force               Force processing even if file hash is unchanged
  --help                Show this help message and exit

Description:
  Converts VTT subtitle files into plain text transcripts by stripping timestamps
  and merging lines into readable paragraphs. Uses SHA256 hashes to skip unchanged files.

Examples:
  $0 --model small
  $0 -m large --vtt-input ./transcripts_large --output-dir ./txt_out --force

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      print_help
      exit 0
      ;;
    --force)
      force=true
      shift 1
      ;;
    --output-dir|-o)
      output_dir="$2"
      shift 2
      ;;
    --vtt-input|-i)
      vtt_input="$2"
      shift 2
      ;;
    --model|-m)
      model="$2"
      shift 2
      ;;
    *)
      echo "❌ Unknown argument: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

# Handle directories and files
transcript_dir="./transcripts_${model}"
if [ -n "$vtt_input" ]; then
  transcript_dir="$vtt_input"
fi
mkdir -p "$transcript_dir"

if [ -z "$output_dir" ]; then
  output_dir="$transcript_dir"
fi
mkdir -p "$output_dir"

vtt_files=("${transcript_dir}"/*.vtt)
if [ ${#vtt_files[@]} -eq 0 ]; then
  echo "No VTT files found in ${transcript_dir}. Exiting."
  exit 0
fi

# Create the file for tracking ids
id_tracker="${output_dir}/.vtt_hashes_${model}"
touch "${id_tracker}"

# Function: Get sha256 hash of VTT file
get_file_hash() {
  sha256sum "${1}" | awk '{print $1}'
}

# Function to convert a VTT file to plain TXT
convert_vtt_to_txt() {
  local vtt_file="${1}"
  local txt_file="${2}"

  awk '
    BEGIN { first_line=1 }
    /^WEBVTT/ || /^[0-9]+$/ || /^[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}/ || /-->/ { next }
    /^[[:space:]]*$/ {
      if (!first_line) print ""
      next
    }
    {
      if (!first_line) printf " "
      printf "%s", $0
      first_line=0
    }
    END { print "" }
  ' "${vtt_file}" | sed 's/  */ /g; s/ *$//' > "${txt_file}"
}

# Process VTT files
total=${#vtt_files[@]}
i=0
for original_vtt in "${vtt_files[@]}"; do
  file_hash=$(get_file_hash "$original_vtt")
  already_seen=$(grep -c "^${file_hash}$" "${id_tracker}" || true)

  if [[ $already_seen -gt 0 && $force == false ]]; then
    echo "↪ Skipping unchanged: ${original_vtt}"
    continue
  fi

  # Prepare names and paths
  base_name="$(basename "${original_vtt}" .vtt)"
  txt_file="${output_dir}/${base_name}.txt"

  ((++i))
  echo "[${i}/${total}] Processing: ${original_vtt}"
  convert_vtt_to_txt "${original_vtt}" "$txt_file"
  echo "✓ Created TXT: ${txt_file}"

  if [[ $already_seen -eq 0 ]]; then
    echo "${file_hash}" >> "${id_tracker}"
  fi
done
