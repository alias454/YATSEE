#!/usr/bin/env python3
"""
yatsee_normalize_structure.py

Stage 4c of the YATSEE pipeline: Normalize and clean polished transcripts
into sentence-per-line files ready for summarization, embedding, or indexing.

Inputs:
  - ..txt files from 'normalized/' under the entity data path (Layer 2 Transform)
  - Custom input directories or files via --input-dir

Outputs:
  - Cleaned .txt files with one sentence per line in the 'normalized/' directory
    under the same entity handle
  - Optional paragraph preservation and deep cleaning

Key Features:
  - Robust text normalization:
      * Collapses character and phrase repetitions
      * Removes filler words and bracketed content (optional deep clean)
      * Corrects punctuation, spacing, and capitalization
      * Preserves numbers, acronyms, and entity names
  - Sentence splitting using spaCy, with optional paragraph preservation
  - Limits consecutive and inline repetitions
  - Entity-specific replacements and cascading configuration from global + local TOML
  - Safe defaults, CLI verbosity, and force-overwrite support

Dependencies:
  - Python 3 standard libraries: os, sys, re, argparse, typing, logging
  - Third-party: spaCy, toml

Example Usage:
  ./yatsee_normalize_structure.py -e defined_entity
  ./yatsee_normalize_structure.py -i ./normalized -o ./normalized --deep-clean
  ./yatsee_normalize_structure.py -e entity_handle --no-spacy --preserve-paragraphs
"""

# Standard library
import argparse
import logging
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

# Third-party imports
import spacy
import toml


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
    Collect files from a directory or single file based on allowed extensions.

    :param input_path: Path to directory or file
    :param supported_exts: Supported file extensions tuple (e.g., ".vtt, .txt")
    :param exclude_suffix: Optional suffix to exclude from results
    :return: Sorted list of valid file paths
    :raises FileNotFoundError: If path does not exist
    :raises ValueError: If single file has unsupported extension
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


def collapse_inline(line: str, inline_max: int = 5) -> str:
    """
    Collapse repeated inline phrases within a single line.

    - Prevents excessive repetition from transcript artifacts.
    - Uses regex to identify repeated phrases up to 5 words.

    :param line: Input text line.
    :param inline_max: Maximum allowed repetitions.
    :return: Line with repetitions collapsed.
    """
    pattern = re.compile(r'\b((?:\w+\s+){0,4}\w+)(?:\s+\1){' + str(inline_max) + r',}', flags=re.IGNORECASE)
    return pattern.sub(lambda m: (' ' + m.group(1)) * inline_max, line)


def limit_repetitions(text: str, inline_max: int = 2, line_max: int = 1) -> str:
    """
    Collapse repeated lines and inline phrases across the text.

    - Reduces transcript noise by limiting inline and consecutive line repetition.
    - Case-insensitive and ignores non-alphanumeric differences when comparing lines.

    :param text: Multi-line input text.
    :param inline_max: Max inline phrase repetitions.
    :param line_max: Max consecutive identical lines.
    :return: Cleaned text.
    """
    pattern = re.compile(r'\b((?:\w+\s+){0,4}\w+)(?:\s+\1){' + str(inline_max) + r',}', flags=re.IGNORECASE)

    result_lines = []
    prev_line_key = None
    consecutive_count = 0

    for line in text.splitlines():
        # Use the pre-compiled pattern
        processed = pattern.sub(lambda m: (' ' + m.group(1)) * inline_max, line.strip())

        current_key = re.sub(r'[^a-z0-9]', '', processed.lower())

        if not current_key:
            result_lines.append('')
            prev_line_key = None
            consecutive_count = 0
            continue

        if current_key == prev_line_key:
            consecutive_count += 1
        else:
            consecutive_count = 1

        if consecutive_count <= line_max:
            result_lines.append(processed)

        prev_line_key = current_key

    return '\n'.join(result_lines)


def merge_incomplete_sentences(text: str) -> str:
    """
    Merge consecutive lines in a transcript that do not end with sentence-ending punctuation.

    - Fixes over-splitting caused by spaCy or other sentence segmenters.
    - Preserves paragraph breaks (double newlines) while merging incomplete lines.
    - Trims extra whitespace and ignores empty lines.

    :param text: Multi-line transcript string.
    :return: Transcript with incomplete lines merged into full sentences.
    """
    import re

    end_punct = re.compile(r"[.!?…]$")
    paragraphs = re.split(r'\n\s*\n', text)  # preserve paragraph boundaries
    merged_paragraphs = []

    for para in paragraphs:
        lines = [l.strip() for l in para.splitlines() if l.strip()]
        buffer = []
        merged_lines = []

        for line in lines:
            buffer.append(line)
            if end_punct.search(line):
                merged_lines.append(" ".join(buffer))
                buffer = []

        if buffer:
            merged_lines.append(" ".join(buffer))

        merged_paragraphs.append("\n".join(merged_lines))

    return "\n\n".join(merged_paragraphs) + "\n"


def capitalize_sentences(text: str, preserve_entities: Optional[List[str]] = None) -> str:
    """
    Capitalize the first alphabetical character of each sentence while
    preserving acronyms, entity names, and proper nouns.

    :param text: Sentence-separated text
    :param preserve_entities: List of words/phrases not to modify
    :return: Capitalized text
    """
    preserve_entities = preserve_entities or []
    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    capitalized = []

    for sentence in sentences:
        stripped = sentence.lstrip()
        if not stripped:
            continue

        # Skip if sentence starts with a preserved entity
        if any(stripped.lower().startswith(ent.lower()) for ent in preserve_entities):
            capitalized.append(sentence)
            continue

        match = re.search(r'[A-Za-z]', stripped)
        if match:
            idx = sentence.index(match.group(0))
            fixed = sentence[:idx] + sentence[idx].upper() + sentence[idx + 1:]
            capitalized.append(fixed)
        else:
            capitalized.append(sentence)

    return ' '.join(capitalized)


def normalize_text(text: str, deep: bool = False, preserve_entities: Optional[List[str]] = None) -> str:
    """
    Robust transcript normalization.

    Features:
    - Standardizes time formats (e.g., '4: 16 p.m.' -> '4:16 PM')
    - Fixes split numerics and large numbers (e.g., '3 . 5' -> '3.5', '1, 000' -> '1,000')
    - Collapses character stutters (sooooo -> soo)
    - Collapses short word/phrase repetitions
    - Normalizes punctuation and removes "dragging" whitespace before commas/dots
    - Preserves numbers, acronyms, and entity names
    - Fixes spacing, punctuation, and capitalization for 'I' and 'U.S.'
    - Removes filler words and bracketed content (optional deep clean)

    :param text: Raw transcript text
    :param deep: Enable deep cleaning (filler words, brackets)
    :param preserve_entities: List of words/phrases to protect from changes
    :return: Normalized text
    """
    preserve_entities = preserve_entities or []
    placeholders = {f"__ENTITY{i}__": name for i, name in enumerate(preserve_entities)}

    # --- Protect entity names temporarily ---
    for ph, name in placeholders.items():
        text = re.sub(r'\b' + re.escape(name) + r'\b', ph, text, flags=re.IGNORECASE)

    # --- Structural & Whitespace Cleanup ---
    text = re.sub(r'[\s\r\n\u00A0]+', ' ', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Collapse char stutters
    text = re.sub(r'\b([A-Za-z])(?:[\s,]+\1){2,}\b', r'\1', text)  # Collapse word stutters
    text = re.sub(r'\b((?:\w+\s+){0,3}\w+)(?:\s+\1){2,}', r'\1', text, flags=re.IGNORECASE)

    # --- Punctuation Standardization ---
    # Enforce "Space After Punctuation" globally.
    # Fixes "March 21,2025" -> "March 21, 2025".
    # However, it temporarily breaks times ("4:16" -> "4: 16") and numbers ("1,000" -> "1, 000").
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'([?!])\1+', r'\1', text)
    text = re.sub(r'\s*\.\s*\.\s*\.', ' ... ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before
    text = re.sub(r'([.,!?;:])(?=\S)', r'\1 ', text)  # Add space after

    # --- Time Normalization ---
    # Unified Regex to handle:
    # 1. Separated: "4: 16 p. M.", "4.30 a. m."
    # 2. Mashed:    "416 p. m."
    # 3. Hour Only: "10 a. M."

    # Regex Logic:
    # \b(\d{1,2})           -> Group 1: Hour
    # (?:                   -> Start Non-capturing group for Minutes
    #   (?:\s*[:.]\s*|\s+)? ->   Inner Non-capturing: Separator (Colon/Dot/Space) is OPTIONAL
    #   (\d{2})             ->   Group 2: Minutes (Must be 2 digits if they exist)
    # )?                    -> End Minutes group (Whole group is OPTIONAL)
    # \s*([AaPp])           -> Group 3: A/P
    # \.?\s*[Mm]...         -> M with trailing dot checks
    text = re.sub(
        r'\b(\d{1,2})(?:(?:\s*[:.]\s*|\s+)?(\d{2}))?\s*([AaPp])\.?\s*[Mm](?:\.(?!\w)|\b)',
        lambda m: f"{m.group(1)}:{m.group(2)} {m.group(3).upper()}M" if m.group(2) else f"{m.group(1)} {m.group(3).upper()}M",
        text
    )

    # --- Post-Time Cleanup ---
    # The Time logic eats trailing dots (from "p.m."), but Step 3 might have left a space
    # before the next comma. E.g., "4:16 PM ,". This snaps it back to "4:16 PM,".
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # --- Numeric Re-assembly (The "Gluer") ---
    # Re-glue Decimals/Ratios: "3 . 5" -> "3.5"
    text = re.sub(r'(\d+)\s*([.:])\s*(\d+)', r'\1\2\3', text)

    # Re-glue Large Numbers: "2, 525, 000" -> "2,525,000"
    # CRITICAL: We use (?!\d) to ensure the second group is EXACTLY 3 digits.
    # This prevents merging dates like "March 21, 2025" because 2025 is 4 digits.
    # We run this twice to handle chained numbers (Millions/Billions).
    large_num_regex = r'(\d{1,3})(?:,\s+|\s+,|\s+)(\d{3})(?!\d)'
    text = re.sub(large_num_regex, r'\1,\2', text)
    text = re.sub(large_num_regex, r'\1,\2', text)

    # --- Content-specific fixes ---
    text = re.sub(r'\bi\b', 'I', text)
    text = re.sub(r'\$\s+(\d)', r'$\1', text)  # "$ 50" -> "$50"
    text = re.sub(r'(\d)\s+%', r'\1%', text)  # "50 %" -> "50%"
    text = re.sub(r'\b(u)\.\s*(s)\.\b', 'U.S.', text, flags=re.IGNORECASE)

    # --- Deep cleaning ---
    if deep:
        filler_words = r'\b(um|uh|erm|you know|like|so|well|ah|oh|mm|hmm|okay)\b'
        text = re.sub(filler_words, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'\s{2,}', ' ', text)

    # --- Restore entity placeholders ---
    for ph, name in placeholders.items():
        text = text.replace(ph, name)

    return text.strip()


def split_sentences(text: str, model: spacy.language.Language) -> List[str]:
    """
    Split text into sentences using a spaCy model.

    :param text: Input text.
    :param model: Loaded spaCy model for sentence segmentation.
    :return: List of sentence strings.
    """
    doc = model(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def process_text_to_sentences(
        text: str,
        model: Optional[spacy.language.Language] = None,
        use_spacy: bool = True,
        preserve_paragraphs: bool = False,
        trim_whitespace: bool = True
) -> str:
    """
    Convert text into one sentence per line with optional paragraph preservation.

    Features:
    - Uses spaCy for sentence segmentation if available.
    - Preserves paragraph breaks with blank lines if requested.
    - Optionally trims excessive leading/trailing whitespace per line.
    - Falls back to simple newline splitting if spaCy is disabled or unavailable.

    :param text: Input text.
    :param model: Optional spaCy model for splitting.
    :param use_spacy: Enable spaCy-based sentence splitting.
    :param preserve_paragraphs: Keep paragraph separation as blank lines.
    :param trim_whitespace: Strip leading/trailing spaces from sentences.
    :return: Text split into one sentence per line.
    """
    text = text.strip()

    if use_spacy and model:
        if preserve_paragraphs:
            paragraphs = re.split(r'\n\s*\n', text)
            processed_paragraphs = []
            for p in paragraphs:
                sentences = [s.strip() for s in split_sentences(p, model) if s.strip()]
                if trim_whitespace:
                    sentences = [s.strip() for s in sentences]
                processed_paragraphs.append("\n".join(sentences))
            return "\n\n".join(processed_paragraphs) + "\n"
        else:
            sentences = [s.strip() for s in split_sentences(text, model) if s.strip()]
            return "\n".join(sentences) + "\n"
    else:
        # Fallback: split on existing line breaks
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines) + "\n"


def apply_replacements(text: str, replacements: Dict[str, str]) -> str:
    """
    Apply flattened entity-specific replacements to transcript text.

    - Matches whole words case-insensitively.
    - Sorted by longest key first to avoid partial matches overwriting longer phrases.

    :param text: Input text.
    :param replacements: Dictionary mapping bad -> correct strings.
    :return: Text with replacements applied.
    """
    for bad, good in sorted(replacements.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(bad) + r'\b', re.IGNORECASE)
        text = pattern.sub(good, text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize transcripts and split into sentence-per-line files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Requirements:
              - Python 3.10+
              - spaCy (for sentence splitting)
              - toml (for config parsing)

            Usage Examples:
              python yatsee_normalize_structure.py -e defined_entity
              python yatsee_normalize_structure.py --input-dir ./normalized --output-dir ./normalized_out
              python yatsee_normalize_structure.py -i ./normalized/file.txt --deep-clean
        """)
    )
    parser.add_argument("-e", "--entity", help="Entity handle to process")
    parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to the global YATSEE configuration file")
    parser.add_argument("-i", "--input-dir", type=str, help="Input file or directory")
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory")
    parser.add_argument("-m", "--model", type=str, help="spaCy model to use (e.g. en_core_web_md)")
    parser.add_argument("--no-spacy", action="store_true", help="Disable spaCy sentence splitting")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--deep-clean", action="store_true", help="Enable deep cleaning")
    parser.add_argument("--preserve-paragraphs", action="store_true", help="Keep paragraph breaks")
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
        if not args.input_dir:
            logging.error("Without --entity, --input-dir must be defined")
            return 1

    # Set up custom logger
    logger = logger_setup(global_cfg.get("system", {}))

    # Determine input/output directory based on entity or CLI override
    append_dir = args.input_dir or entity_cfg.get("transcription_model", "small")
    input_dir = args.input_dir or os.path.join(entity_cfg["data_path"], f'transcripts_{append_dir}')
    file_list = discover_files(input_dir, ".txt")
    if not file_list:
        logger.info("No .txt input files found")
        return 0

    # By default, output directory is the normalized directory
    output_directory = args.output_dir or os.path.join(entity_cfg["data_path"], 'normalized')
    if not os.path.isdir(output_directory):
        logger.info("Output directory will be created: %s", output_directory)
        os.makedirs(output_directory, exist_ok=True)

    # ----------------------------
    # Load spaCy model if enabled
    # ----------------------------
    spacy_model = None
    if not args.no_spacy:
        spacy_model_name = (
            args.model
            or entity_cfg.get("sentence_model")
            or global_cfg.get("system", {}).get("default_sentence_model", "en_core_web_sm")
        )
        if not spacy_model_name:
            logger.error("No spaCy model specified from CLI, entity config, or system config.")
            return 1

        try:
            spacy_model = spacy.load(spacy_model_name)
            logger.info("Using spaCy model: %s", spacy_model_name)
        except OSError:
            logger.error("spaCy model '%s' not found. Install with: python -m spacy download %s",spacy_model_name, spacy_model_name)
            return 1

    # Load replacements from flattened entity config
    replacements = entity_cfg.get("replacements", {})

    # Process each file
    for file_path in file_list:
        filename = os.path.basename(file_path)
        output_file = os.path.join(output_directory, filename)

        if os.path.isfile(output_file) and not args.force:
            logger.info("Skipping existing file: %s", output_file)
            continue

        # Read, normalize, split, limit repetitions, and apply replacements
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            logger.debug("Read %d characters from %s", len(raw_text), filename)

        normalized = normalize_text(raw_text, deep=args.deep_clean)
        normalized = capitalize_sentences(normalized)
        processed = process_text_to_sentences(normalized, spacy_model, not args.no_spacy, args.preserve_paragraphs)
        processed = merge_incomplete_sentences(processed)
        processed = limit_repetitions(processed, inline_max=2, line_max=1)
        processed = apply_replacements(processed, replacements)

        # Write output
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(processed)
            logger.info("Processed file: %s", output_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
