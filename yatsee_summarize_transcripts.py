#!/usr/bin/env python3
"""
yatsee_summarize_transcripts.py

Stage 5 of the YATSEE pipeline: Summarize transcripts from civic meetings
(city council, committee, town hall, etc.) using a local LLM via Ollama.
Chunk summaries are kept in memory only; no intermediate files are written
to disk.

Inputs:
  - Single transcript file or directory of `.txt` files
  - Optional human-readable meeting context via --context to guide summarization
  - Local Ollama model pulled via `ollama pull` (e.g., llama3, mistral)

Outputs:
  - Final merged summary written to --output-dir (default: ./summary/)
  - Output formats supported: Markdown (default) or YAML
  - Intermediate chunk summaries remain in memory and are not written to disk

Key Features:
  - Automatic classification of meeting type using transcript content
    and optional filename heuristics
  - Multi-pass summarization pipeline with configurable depth (--max-pass)
  - Modular prompt system for different summary styles:
      * overview, action_items, detailed, more_detailed, most_detailed,
        final_pass_detailed
      * Dynamic selection of prompt based on meeting type
      * Optional manual overrides (--first-prompt, --second-prompt, --final-prompt)
  - Memory-first design prioritizing privacy and performance
  - Handles long transcripts via in-memory chunking with automatic merging
  - Emphasis on structured civic dialogue: motions, votes, decisions, speaker intent
  - Gender-neutral and consistent summary style

Arguments:
  -i / --input-dir           Directory or file path of transcripts to summarize
  --model                    Local model to use (e.g., llama3, mistral)
  --context                  Optional human-readable meeting context
  --output-dir               Directory to save final summary (default: ./summary/)
  --output-format            'markdown' (default) or 'yaml'
  --first-prompt             Optional manual prompt ID for first summarization
                             pass (used if auto-classification is disabled)
  --second-prompt            Optional manual prompt ID for second pass
                             (used if auto-classification is disabled)
  --final-prompt             Optional manual prompt ID for final pass
                             (used if auto-classification is disabled)
  --max-words                Approximate word count per memory chunk (default: 3500)
  --max-pass                 Maximum summarization passes (default: 3)
  --disable-auto-classification  Disable automatic prompt selection

Dependencies:
  - Python 3.10+
  - Ollama (local API endpoint)
  - Locally pulled LLMs: llama3, mistral, gemma, etc.
  - Standard libraries: os, sys, re, argparse, textwrap, logging
  - Optional: YAML library for output if using --output-format yaml

Example Usage:
  python yatsee_summarize_transcripts.py --model llama3 \
      -i council_meeting_2025_06_01 \
      --context "City Council Meeting - June 2025"

  python yatsee_summarize_transcripts.py --model mistral \
      -i firehall_meeting_2025_05 \
      --context "Fire Hall Proposal Discussion" \
      --output-format markdown

  python yatsee_summarize_transcripts.py --model gemma:2b \
      -i finance_committee_2025_05 \
      --disable-auto-classification \
      --first-prompt overview \
      --second-prompt detailed \
      --final-prompt final_pass_detailed
"""

# Standard library
import argparse
import json
import logging
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Set

# Third-party imports
import requests
import toml
import yaml


def classify_meeting(session: requests.Session, llm_provider_url: str, model: str, prompt: str, allowed_labels: set[str]) -> str:
    """
    Classify the type of a meeting transcript using a local Ollama LLM.

    The function sends a single request to the Ollama server with the provided
    prompt and model, retrieves the JSON response, and validates the label
    against allowed meeting types.

    :param session: A preconfigured requests.Session object for efficient HTTP reuse.
    :param llm_provider_url: URL of the LLM provider server.
    :param model: Name of the model to use (e.g., 'llama3').
    :param prompt: Prompt text containing context and transcript snippet.
    :param allowed_labels: Set of accepted meeting types that can be returned.
    :return: Lowercase meeting type label (e.g., 'city_council') or 'general' if no match.
    :raises RuntimeError: If the HTTP request fails or JSON decoding fails.
    """
    # In case we don't identify a clear label type or have a prompt bail out
    if not allowed_labels or not prompt:
        return "general"

    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = session.post(f"{llm_provider_url}/api/generate", json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        raw = data.get("response", "").strip().lower()

        for label in allowed_labels:
            if label in raw:
                return label

        # Always return something if classifications fails
        return "general"

    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Error during meeting classification request: {e}")


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


def discover_files(input_path: str, supported_exts, exclude_suffix: str = None) -> List[str]:
    """
    Discover files in a directory or single file path matching allowed extensions.

    Optionally excludes files ending with a specified suffix.

    :param input_path: Path to directory or single file
    :param supported_exts: Allowed file extensions (e.g., '.txt' or ('.txt', '.out'))
    :param exclude_suffix: Optional suffix to exclude from results
    :return: Sorted list of file paths matching criteria
    :raises FileNotFoundError: If path does not exist
    :raises ValueError: If a single file does not match supported extensions
    """
    files: List[str] = []

    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            full = os.path.join(input_path, f)
            if (
                os.path.isfile(full)
                and f.lower().endswith(supported_exts)
                and (exclude_suffix is None or not f.lower().endswith(exclude_suffix))
            ):
                files.append(full)
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(supported_exts) and (
            exclude_suffix is None or not input_path.lower().endswith(exclude_suffix)
        ):
            files.append(input_path)
        else:
            raise ValueError(f"Unsupported file extension or excluded: {os.path.basename(input_path)}")
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

    return sorted(files)


def extract_context_from_filename(filename, extra_note=None):
    """
    Extract a human-readable context description from a transcript filename.

    Parses date patterns and normalizes separators, producing strings like
    'City Council — 2025-06-15'. Optionally appends an extra note in parentheses.

    :param filename: Transcript file path
    :param extra_note: Optional string to append for additional context
    :return: Human-friendly context string
    """
    # Just the basename (strip path)
    base = os.path.basename(filename)

    # Remove hash-like prefix if present
    base = re.sub(r"^[\w\-]+?\.", "", base)

    # Remove extension
    base = base.replace(".txt", "")

    # Normalize separators
    base = base.replace("_", " ").replace("-", " ")

    # Try to extract components
    meeting_match = re.match(r"(.*?) (\d{1,2})[ -](\d{1,2})[ -](\d{2,4})", base)
    if meeting_match:
        kind = meeting_match.group(1).strip().title()
        month = int(meeting_match.group(2))
        day = int(meeting_match.group(3))
        year = int(meeting_match.group(4))
        if year < 100:
            year += 2000

        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        context = f"{kind} — {date_str}"
    else:
        context = base.title()

    if extra_note:
        context += f" ({extra_note.strip()})"

    return context


def estimate_token_count(text: str) -> int:
    """
    Estimate the token count of a string for LLM usage.

    Uses an average ratio of 0.75 words per token as heuristic.

    :param text: Input text
    :return: Estimated token count (integer)
    """
    return int(len(text.split()) / 0.75)


def calculate_cost_openai_gpt4o(input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of a GPT-4o call on OpenAI based on token usage.

    This is a simple helper to quickly approximate pricing for planning or
    logging purposes. It assumes linear costs and does not account for any
    tiered or promotional pricing.

    Pricing:
      - $0.005 per 1,000 input tokens
      - $0.015 per 1,000 output tokens

    :param input_tokens: Number of tokens sent to the model (prompt)
    :param output_tokens: Number of tokens generated by the model (completion)
    :return: Estimated cost in USD, rounded to 4 decimal places
    """
    input_cost = input_tokens / 1000 * 0.005
    output_cost = output_tokens / 1000 * 0.015
    return round(input_cost + output_cost, 4)


def prepare_text_chunk(text: str, max_tokens: int = 2500, overlap_tokens: int | None = None) -> List[str]:
    """
    Split a large text into overlapping token-based chunks for LLM processing.

    Useful when the text exceeds the model context limit. Overlapping chunks
    preserve continuity so that summaries or embeddings capture context across
    boundaries. Sliding window size is determined by max_tokens and optional
    overlap.

    :param text: Input text string to chunk
    :param max_tokens: Maximum number of tokens per chunk
    :param overlap_tokens: Number of tokens to repeat from previous chunk;
        defaults to min(10% of max_tokens, 800)
    :return: List of string chunks, each within token limits, preserving overlap
    """
    if overlap_tokens is None:
        overlap_tokens = min(int(max_tokens * 0.1), 800)  # max 800 tokens overlap

    tokens = text.split()
    total_tokens = len(tokens)

    if total_tokens <= max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)

        if end == total_tokens:
            break

        # Slide window by max_tokens - overlap_tokens to keep overlap
        start += max_tokens - overlap_tokens
        if start < 0:
            start = 0

    return chunks


def prepare_text_chunk_sentence(text: str, max_tokens: int = 3000) -> list[str]:
    """
    Split text into sentence-aligned chunks while staying under a token limit.

    This is preferred when semantic coherence matters — sentence boundaries
    are respected to avoid splitting mid-thought. The function first estimates
    token count; if text fits in one chunk, returns immediately.

    :param text: Input transcript text
    :param max_tokens: Approximate maximum tokens per chunk
    :return: List of sentence-aligned chunks ready for model consumption
    """
    # Fast path: skip chunking if the text fits in one chunk
    if int(len(text.split()) / 0.75) <= max_tokens:
        return [text]

    # Split text into sentences
    sentences = re.split(r'(?<=[.?!])\s+', text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize_transcript(session: requests.Session, llm_provider_url: str, model: str, prompt: str = "detailed", num_ctx: int = 8192) -> str:
    """
    Generate a summary of a transcript using a local Ollama LLM in streaming mode.

    Uses a requests.Session to post a prompt to the model, streaming output
    line by line. Useful for long transcripts where memory or latency matters.
    Stops reading the stream once the 'done' flag is reached.

    :param session: Active requests.Session to reuse connections efficiently
    :param llm_provider_url: URL of the LLM provider server
    :param model: Name of the Ollama model to use (e.g., 'llama3')
    :param prompt: Prompt text or template to drive summarization
    :param num_ctx: Maximum context length the model can handle (used in options)
    :return: Generated summary text
    :raises RuntimeError: If the HTTP request fails or streaming JSON is malformed
    """
    # payload = {"model": model, "prompt": prompt}
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.2,
        "options": {
            "num_ctx": num_ctx
        }
    }

    try:
        response = session.post(f"{llm_provider_url}/api/generate", json=payload, stream=True)
        response.raise_for_status()

        summary = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                summary += data.get("response", "")
                if data.get("done", False):
                    break
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON from Ollama stream: {e}")
        return summary

    except requests.RequestException as e:
        raise RuntimeError(f"Error during Ollama request: {e}")


def write_summary_file(summary: str, basename: str, output_dir: str, fmt: str = "yaml") -> str:
    """
    Write a summary to disk in YAML or Markdown format.

    Creates directories if missing. Returns the full path to the written file.

    :param summary: The summary content to save
    :param basename: Base filename without extension
    :param output_dir: Directory to store the file
    :param fmt: 'yaml' or 'markdown'
    :return: Full path of the written file
    :raises OSError, IOError: If the file cannot be written
    """
    filename = f"{basename}.summary.{'md' if fmt == 'markdown' else 'yaml'}"
    out_path = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        if fmt == "markdown":
            f.write(f"# Summary: {basename}\n\n{summary.strip()}\n")
        else:
            yaml.safe_dump({"summary": summary}, f, sort_keys=False)

    return out_path


def write_chunk_files(chunks: List[str], output_dir: str, meeting_type: str, base_name: str) -> List[str]:
    """
    Save text chunks to structured directories by meeting type and base name.

    Each chunk gets its own file with a numbered suffix to preserve order.
    Directory is automatically created if it does not exist. Returns a list
    of all written file paths.

    :param chunks: List of text chunks to save
    :param output_dir: Root directory for chunk storage
    :param meeting_type: Subdirectory label, e.g., 'city_council'
    :param base_name: Base filename prefix for all chunk files
    :return: List of full paths of all chunk files
    :raises OSError, IOError: If any chunk file cannot be written
    """
    chunks_path = os.path.join(output_dir, "chunks", meeting_type, base_name)
    os.makedirs(chunks_path, exist_ok=True)

    written_files = []
    for idx, chunk_text in enumerate(chunks):
        chunk_filename = f"{base_name}_part{idx:02d}.txt"
        chunk_path = os.path.join(chunks_path, chunk_filename)
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk_text)
        written_files.append(chunk_path)

    return written_files


def generate_known_speakers_context(speaker_matches: dict) -> List[str] | str:
    """
    Generate a human-readable summary of known speakers in a transcript chunk.

    Useful for adding context to chunks before indexing or summarizing.
    Highlights canonical names detected and mentions when multiple name
    variants are present.

    :param speaker_matches: Mapping of canonical speaker ID to set of matched names
    :return: Markdown-friendly string describing detected speakers, or empty string
    """
    if not speaker_matches:
        return ""

    descriptive_mentions = []
    ambiguous_hits = []

    for canonical_name, matches in speaker_matches.items():
        sorted_matches = sorted(matches, key=len, reverse=True)
        best_variant = sorted_matches[0]
        descriptive_mentions.append(best_variant)

        # Check if multiple name forms were used
        if len(matches) > 1:
            ambiguous_hits.append((canonical_name, sorted_matches))

    mentions_string = ", ".join(descriptive_mentions)
    context_lines = [f"*Speakers detected in this segment include: {mentions_string}."]

    if ambiguous_hits:
        context_lines.append("Some individuals were referenced using multiple name variants, e.g.:")
        for name, variants in ambiguous_hits:
            readable = name.replace("_", " ")
            context_lines.append(f"  - {readable}: {', '.join(variants)}")

    context_lines.append("Mentions of partial or ambiguous names may refer to multiple individuals.*")
    return "\n".join(context_lines)


def scan_transcript_for_names(transcript_text: str, name_permutations: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """
    Identify mentions of known individuals in a transcript using name and title variants.

    This function searches the transcript for any variant of a canonical
    name (including titles, abbreviations, and alternative spellings) and
    returns all matches. Word boundaries are respected to avoid false
    positives (e.g., "Rob" won't match inside "Robotics"). Useful for
    tagging speakers or preparing context for summaries and embeddings.

    :param transcript_text: Raw transcript text to scan
    :param name_permutations: Dictionary mapping canonical person keys to lists
        of known name/title variants
    :return: Dictionary mapping canonical names to sets of matched strings
    """
    found_matches = {}

    # Normalize the transcript once (for case-insensitive matching)
    lower_transcript = transcript_text.lower()

    for person_key, permutations in name_permutations.items():
        matches = set()

        for name_variant in permutations:
            # Simple word boundary match to avoid partial hits (e.g., "Rob" inside "Robotics")
            pattern = r'\b' + re.escape(name_variant.lower()) + r'\b'
            if re.search(pattern, lower_transcript):
                matches.add(name_variant)

        if matches:
            found_matches[person_key] = matches

    return found_matches


def build_name_permutations(data_config: dict) -> dict:
    """
    Generate comprehensive name and title permutations for known individuals.

    This prepares a lookup table for transcript scanning, expanding each
    canonical person's aliases to include:
      - First name, last name, full name
      - Known aliases from configuration
      - Titles combined with last or full names
      - Rephrased multi-word title variants (e.g., "Information Technology Director")

    The resulting dictionary can then be used to detect all possible references
    to a person across transcripts, improving speaker matching and context generation.

    :param data_config: Dictionary containing 'people' (canonical names -> aliases)
        and 'titles' (role -> list of titles)
    :return: Dictionary mapping canonical person keys to sorted list of name/title variants
    :raises KeyError: If 'people' or 'titles' keys are missing from data_config
    """
    people_by_role = data_config.get("people", {})
    titles_by_role = data_config.get("titles", {})

    name_permutations = {}

    for role_key, people in people_by_role.items():
        titles_for_role = titles_by_role.get(role_key, [])

        for person_key, known_aliases in people.items():
            # Extract full name from the key (e.g., 'Rachel_Sampson')
            try:
                first_name, last_name = person_key.split("_", 1)
            except ValueError:
                # Handle unexpected name formats gracefully
                first_name = person_key
                last_name = ""
            full_name = f"{first_name} {last_name}".strip()

            # Start with known aliases + inferred names
            permutations = set(known_aliases)
            permutations.update({first_name, last_name, full_name})

            # Add title permutations
            for title in titles_for_role:
                # Add standard "Title Lastname" and "Title Fullname" forms
                permutations.add(f"{title} {last_name}".strip())
                permutations.add(f"{title} {full_name}".strip())

                # If the title contains multiple words (e.g., "Information Technology"), include rephrased forms
                if " " in title:
                    permutations.add(f"{title} Director {last_name}".strip())
                    permutations.add(f"{title} Director {full_name}".strip())

            name_permutations[person_key] = sorted(permutations)

    return name_permutations


def validate_prompt(prompt_name: str | None, label: str, available_prompts: dict) -> None:
    """
    Validate that a requested prompt exists in the loaded prompt set.

    This function performs validation only and has no side effects:
    - No printing
    - No exiting
    - No mutation of global state

    The caller (typically main()) is responsible for catching exceptions
    and presenting user-facing error messages.

    :param prompt_name: The prompt identifier provided via CLI (may be None)
    :param label: Human-readable label used for error context (e.g. "first prompt")
    :param available_prompts: Dictionary of available prompt templates
    :raises ValueError: If the prompt name is defined but not found
    """
    if not prompt_name:
        # No prompt explicitly requested; nothing to validate
        return

    if prompt_name not in available_prompts:
        valid_keys = ", ".join(sorted(available_prompts.keys()))
        raise ValueError(
            f"Invalid {label} '{prompt_name}'. "
            f"Available prompts: {valid_keys}"
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Requirements:
          - Python 3.10+
          - Ollama CLI installed and models pulled locally
          - Input transcripts must be .txt or .out files

        Features:
          - Automatic classification of meeting type (city_council, finance_committee, etc.)
          - Dynamic prompt selection per meeting type
          - Manual prompt override for custom workflows
          - Recursive multi-pass summarization for long transcripts (default: 3 passes)
          - Optional disabling of auto-classification
          - Output in YAML or Markdown
          - Fully local and privacy-respecting

        Example usage:
          python yatsee_summarize_transcripts.py -e defined_entity
          python yatsee_summarize_transcripts.py --model llama3:latest -i normalized/ --context "City Council - June 2025"
          python yatsee_summarize_transcripts.py -m mistral:latest -i transcripts/ -o summaries/ --output-format markdown
          python yatsee_summarize_transcripts.py -m gemma:2b -i finance/ --disable-auto-classification --first-prompt detailed
    """)
)
    parser.add_argument("-e", "--entity", help="Entity handle to process")
    parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to global yatsee.toml")
    parser.add_argument("-i", "--txt-input", help="Path to a transcript file or directory (supports .txt or .out)")
    parser.add_argument("-o", "--output-dir", help="Directory to save the summary output (Default: summary)")
    parser.add_argument("-m", "--model", help="Local Ollama model name (e.g. 'llama3:latest', 'mistral:latest', 'mistral-nemo:latest', gemma:2b)")
    parser.add_argument("-f", "--output-format", choices=["markdown", "yaml"], default="markdown", help="Summary output format (Default: markdown)")
    parser.add_argument("-j", "--job-type", choices=["summary", "research"], default="summary", help="Define job type to select prompt workflow (default: summary)")
    parser.add_argument("-w", "--max-words", type=int, help="Word count threshold for chunking transcript (default: 3500)")
    parser.add_argument("-t", "--max-tokens", type=int, help="Approximate max tokens per chunk (Default: 2500)")
    parser.add_argument("-p", "--max-pass", type=int, default=3, help="Max iterations for multi-pass summarization (default: 3)")
    parser.add_argument("-d", "--disable-auto-classification", action="store_true", help="Disable automatic meeting type classification. Requires manual prompt overrides")
    parser.add_argument("--first-prompt", help="Prompt type for first pass (only used when auto-classification is disabled)")
    parser.add_argument("--second-prompt", help="Prompt type for second pass of chunk summaries (only used when auto-classification is disabled)")
    parser.add_argument("--final-prompt", help="Prompt type for final pass summarization (only used when auto-classification is disabled)")
    parser.add_argument("--context", default="", help="Optional context string to guide summarization (e.g., 'City Council - June 2025')")
    parser.add_argument("--print-prompts", action="store_true", help="Print all prompt templates for job type and exit")
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
        # Require input if no entity is provided
        if not args.txt_input:
            logging.error("Without --entity, --txt-input must be defined")
            return 1

    # Set up custom logger
    logger = logger_setup(global_cfg.get("system", {}))

    # ----------------------------
    # Load PROMPTS
    # ----------------------------
    prompt_types_path = ""
    entity_prompt_file = os.path.join(entity_cfg.get("data_path"), "prompts", args.job_type, "prompts.toml")
    default_prompt_file = os.path.join("prompts", args.job_type, "prompts.toml")

    if os.path.isfile(entity_prompt_file):
        prompt_types_path = entity_prompt_file
    elif os.path.isfile(default_prompt_file):
        prompt_types_path = default_prompt_file

    if prompt_types_path:
        # Load prompts and prompt routing
        _prompts = toml.load(prompt_types_path)
        PROMPTS = {k: v["text"] for k, v in _prompts.get("prompts", {}).items()}
        PROMPT_LOOKUP = _prompts.get("prompt_router", {})
        CLASSIFIER_PROMPT = _prompts.get("classifier_prompt", {}).get("text", "")
        classifier_types = _prompts.get("classifier_types", {})
    else:
        # fallback: inline general defaults
        logger.warning("No prompts found for job type '%s', using inline general defaults", args.job_type)
        PROMPTS = {
            "overview": (
                "You are an assistant that produces a high-level summary of a meeting transcript.\n\n"
                "Context: {context}\n\n"
                "Transcript:\n{text}\n\n"
                "Summarize the key points, main discussions, and outcomes in clear, concise language."
            ),
            "action_items": (
                "You are an assistant that extracts actionable items, motions, and decisions from a meeting transcript.\n\n"
                "Context: {context}\n\n"
                "Transcript:\n{text}\n\n"
                "List only the actionable items or decisions in plain language. Skip commentary or filler text."
            )
        }
        PROMPT_LOOKUP = {
            "general": {"first": "overview", "multi": "overview", "final": "action_items"}
        }
        CLASSIFIER_PROMPT = ""
        classifier_types = {}

    if args.disable_auto_classification:
        validate_prompt(args.first_prompt, "first-prompt", PROMPTS)
        validate_prompt(args.second_prompt, "second-prompt", PROMPTS)
        validate_prompt(args.final_prompt, "final-prompt", PROMPTS)

    # Handle --print-prompts early and exit
    if args.print_prompts:
        print(F"Available {args.job_type} prompt templates:\n")
        for key, prompt_text in PROMPTS.items():
            print(f"=== {key} ===\n{prompt_text}\n{'-'*40}\n")
        return 0

    # --- Flatten hotwords / build name permutations ---
    known_speakers_permutations = build_name_permutations(entity_cfg)

    # --- Resolve model ---
    # Priority: CLI arg > entity config > global config
    model = (
        args.model
        or entity_cfg.get("summarization_model")
        or global_cfg.get("system", {}).get("default_summarization_model")
    )
    if not model:
        logger.error("No model specified. Use --model(-m) or set it in entity/global config.")
        return 1

    # supported_models = ["mistral:latest", "mistral-nemo:latest", "llama3:latest", "gemma:2b", "qwen2.5:7b-instruct-q4_k_m"]
    # model_map is already loaded from config
    model_map = global_cfg.get("models", {})

    # Normalize lookup for user input (case-insensitive)
    model_key = next((m for m in model_map if m.lower() == model.lower()), None)
    if not model_key:
        map_keys = ', '.join(model_map.keys())
        logger.error("Unsupported model '%s'. Supported: %s",model, map_keys)
        return 1

    # --- Resolve input path ---
    input_dir = args.txt_input or os.path.join(entity_cfg.get("data_path"), "normalized")
    file_list = discover_files(input_dir, ".txt", "punct.txt")
    if not input_dir:
        logger.error("No input file or directory specified. Use --txt-input(-i) or set it in the config.")
        return 0

    # --- Resolve output path ---
    model_cfg = model_map[model]

    output_dir = args.output_dir or os.path.join(entity_cfg["data_path"], model_cfg["append_dir"])
    if not os.path.isdir(output_dir):
        logger.info("Output directory will be created: %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Now model_key matches the exact key in config
    max_tokens = args.max_tokens or model_map[model_key].get("max_tokens", 3500)
    num_ctx = model_map[model_key].get("num_ctx", 8192)

    # Open up a request object and setup the client connection
    # Create a global or shared session object once
    llm_session = requests.Session()

    # Optionally configure session headers, retries, timeouts here
    llm_session.headers.update({"Content-Type": "application/json"})

    llm_provider_url = global_cfg.get("system", {}).get("llm_provider_url")

    # ----------------------------
    # Process files for summerizer jobs
    # ----------------------------
    for file_path in file_list:
        # Extract a friendly context string (e.g., meeting name and date) from the filename
        context = extract_context_from_filename(file_path, args.context)

        # Initialize counters for input and output tokens (used to estimate API cost)
        token_usage = 0
        output_token_usage = 0

        logger.info("Using model: %s", model)
        # Base name of the input file
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            # Read the full transcript text from file
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read()

            meeting_type = "general"
            if not args.disable_auto_classification:
                # Try to auto classify the meeting type based off a small snippet of text
                classifier_prompt = CLASSIFIER_PROMPT.format(context=context, text=transcript[:2000])
                meeting_type = classify_meeting(
                    session=llm_session,
                    llm_provider_url=llm_provider_url,
                    model=model,
                    prompt=classifier_prompt,
                    allowed_labels=set(classifier_types.get("allowed", []))
                )

                # track token usage
                token_usage += estimate_token_count(classifier_prompt)
                output_token_usage += estimate_token_count(meeting_type)
                logger.info("Auto-detected meeting type: %s", meeting_type)

            first_pass_id = args.first_prompt or PROMPT_LOOKUP.get(meeting_type, PROMPT_LOOKUP["general"])["first"]
            multi_pass_id = args.second_prompt or PROMPT_LOOKUP.get(meeting_type, PROMPT_LOOKUP["general"])["multi"]
            final_pass_id = args.final_prompt or PROMPT_LOOKUP.get(meeting_type, PROMPT_LOOKUP["general"])["final"]

            logger.info("Processing transcript: %s", base_name)

            # Set pass count values
            max_pass = args.max_pass

            # Loop to dynamically summarize text, possibly recursively summarizing summaries
            pass_num = 1  # Current pass counter
            while pass_num <= max_pass:
                # Select prompt type: first pass uses 'prompt_type_first', subsequent use 'prompt_type_second'
                prompt_type = first_pass_id if pass_num == 1 else multi_pass_id

                # Writing chunk files
                enable_chunk_writer = False
                if enable_chunk_writer:
                    # Prepare transcript for RAG if chunk writer enabled
                    # This will use half the max tokens per chunk with about 16.7% overlap
                    chunks = prepare_text_chunk(transcript, int(args.max_tokens / 2), int(args.max_tokens / 6))
                    try:
                        chunk_files = write_chunk_files(chunks, output_dir, meeting_type, base_name)
                        logger.info("Wrote %d chunks to %s", len(chunk_files), os.path.join(output_dir, "chunks", meeting_type))
                    except Exception as e:
                        logger.error("Error writing chunk files: %s", e)

                chunks_only = False
                if chunks_only:
                    break  # exit the while loop

                # Prepare transcript for summarization by chunking contents if necessary
                # If text too large to summarize in one call, break into chunks and summarize each chunk
                chunks = prepare_text_chunk(transcript, max_tokens=max_tokens)
                chunk_summaries = []

                count = 0
                for chunk in chunks:
                    count += 1
                    logger.info("Pass %d processing chunk %d/%d with prompt [%s]", pass_num, count, len(chunks), prompt_type)
                    # Inject context back into the chunk when operating on large files
                    # The context is in the file but with each chunk the llm looses context so add some back
                    speaker_matches = scan_transcript_for_names(chunk, known_speakers_permutations)
                    speaker_context = generate_known_speakers_context(speaker_matches)
                    chunk = speaker_context + "\n\n" + chunk

                    # for person, hits in speaker_matches.items():
                    #     print(f"{person}: {sorted(hits)}")
                    # print("\n\n")

                    # print(speaker_context)
                    # print("\n")

                    # Summarize each chunk using the main prompt
                    prompt_template = PROMPTS.get(prompt_type, PROMPTS["detailed"])
                    chunk_prompt = prompt_template.format(context=context or "No context provided.", text=chunk)

                    chunk_summary = summarize_transcript(llm_session, llm_provider_url, model, chunk_prompt, num_ctx)
                    chunk_summaries.append(chunk_summary)

                    # Update token usage counts
                    token_usage += estimate_token_count(chunk)
                    output_token_usage += estimate_token_count(chunk_summary)
                # End for loop

                # Combine chunk summaries into a single string for final summarization
                summary = "\n\n".join(chunk_summaries)

                # If summary is small enough or this is the last pass, finalize and exit loop
                if pass_num >= 2:
                    if estimate_token_count(summary) <= max_tokens or len(chunk_summaries) == 1 or pass_num == max_pass:
                        logger.info("Processing final summary with prompt [%s]", final_pass_id)
                        prompt_template = PROMPTS.get(final_pass_id, PROMPTS["detailed"])
                        final_prompt = prompt_template.format(context=context or "No context provided.", text=summary)

                        transcript = summarize_transcript(llm_session, llm_provider_url, model, final_prompt, num_ctx)

                        # Update token usage counts
                        token_usage += estimate_token_count(transcript)
                        output_token_usage += estimate_token_count(transcript)
                        break

                # Increment pass number and repeat loop
                transcript = summary
                pass_num += 1
            # End while loop

            chunks_only = False
            if chunks_only:
                continue # continue to next file

            # Calculate total tokens used and estimate API cost
            total_tokens = token_usage + output_token_usage
            estimated_cost = calculate_cost_openai_gpt4o(token_usage, output_token_usage)

            # Print token usage and estimated cost for transparency
            logger.info("Token usage: tokIn=%d tokOut=%d total=%d", token_usage, output_token_usage, total_tokens)
            logger.info("Estimated GPT-4o API cost: $%.4f", estimated_cost)

            # ----------------------------
            # Write output files per job type
            # ----------------------------
            if args.job_type == "summary":
                try:
                    # After all passes or break condition, final_summary is stored in transcript variable
                    final_summary = transcript

                    summary_file = write_summary_file(final_summary, base_name, output_dir, args.output_format)
                    logger.debug("\nFinal Summary:\n")
                    logger.debug(final_summary)
                    logger.info("Final summary written to: %s\n", summary_file)
                except Exception as e:
                    logger.error("Failed to write summary file: %s", e)
            elif args.job_type == "research":
                try:
                    pass
                    # print(summary)
                    # After all passes or break condition, final_facts stored as json
                    # final_facts = aggregate_facts_from_chunks(chunk_summaries)
                    #
                    # fact_file = write_facts_file(final_facts, base_name, output_dir)
                    # logger.debug("\nFinal Summary:\n")
                    # logger.debug(final_facts)
                    # logger.info("Research facts written to: %s\n", fact_file)
                except Exception as e:
                    logger.error("Failed to write research facts file: %s", e)

        except Exception as e:
            # Catch and report any errors reading file or processing
            logger.exception("Error processing file '%s': %s", file_path, e)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())