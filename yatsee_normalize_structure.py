#!/usr/bin/env python3
"""
Splits transcript text files into individual sentences using spaCy and lightly cleans the text.

This script is designed for post-transcription cleanup and segmentation:
- It assumes that raw transcripts have already been cleaned of timestamps and speaker labels.
- It performs light text normalization to fix spacing and punctuation issues.
- It uses spaCy to segment text into natural sentences for further processing.

Requirements:
- Python 3.8+
- spaCy
- Download the spaCy model with:
    python -m spacy download en_core_web_sm

Usage:
  ./yatsee_normalize_structure.py --txt-input input.txt --output-dir output_dir/
  ./yatsee_normalize_structure.py --txt-input transcripts/ --output-dir split_transcripts/ [--force] [--no-spacy] [--deep-clean] [--preserve-paragraphs]

Arguments:
  --txt-input, -i      Input .txt file or directory of .txt files (required).
  --output-dir, -o     Output directory where processed files will be saved (default: ./normalized).
  --no-spacy           Disable spaCy sentence segmentation; fallback to splitting at line breaks.
  --force              Overwrite existing output files if they exist.
  --deep-clean         Apply deeper cleanup such as filler word removal, repeated character trimming, etc.
  --preserve-paragraphs Keep paragraphs separated by blank lines, splitting sentences inside paragraphs.

Notes:
- When input is a directory, all .txt files will be processed.
- Output files will be saved in the specified output directory with the same filenames as inputs.
- SpaCy model 'en_core_web_sm' is required unless --no_spacy is specified.
- Output files contain one sentence per line, after basic or deep normalization.
- When --preserve_paragraphs is used, paragraphs are separated by a blank line in output.

Examples:
  ./yatsee_normalize_structure.py -i transcript.txt -o split_sentences/
  ./yatsee_normalize_structure.py -i transcripts_folder/ -o split_sentences/ --force
  ./yatsee_normalize_structure.py -i transcripts/ -o split/ --no_spacy --deep-clean --preserve-paragraphs
"""

import os
import sys
import argparse
import spacy
import re
from collections import defaultdict
from typing import Optional, List


def filter_file(path_list: list[str], ext: str = ".txt") -> list[str]:
    """
    Filters only txt files and returns them as a list

    :param path_list: Base name of the meeting file (no extension)
    :param ext: File extension to filter on
    :return: List of resolved file paths
    """
    files = []

    for path in path_list:
        if path.lower().endswith(ext):
            files.append(path)

    return files


def get_files_list(path: str) -> list[str]:
    """
    Collect a list of .txt and .out files from a directory or single file path.

    :param path: Path to .txt or .out files, or directory
    :return: List of valid txt file paths
    :raises FileNotFoundError: If no valid files found
    :raises ValueError: If unsupported file extension encountered
    """
    valid_extensions = (".txt", ".out")
    txt_files = []

    if os.path.isdir(path):
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path) and filename.lower().endswith(valid_extensions):
                txt_files.append(full_path)
        if not txt_files:
            raise FileNotFoundError(f"No valid .txt files found in directory: {path}")
    elif os.path.isfile(path):
        if path.lower().endswith(valid_extensions):
            txt_files.append(path)
        else:
            raise ValueError(f"Unsupported file extension: {os.path.splitext(path)[1]}")
    else:
        raise FileNotFoundError(f"Input path not found: {path}")

    return txt_files


def load_spacy_model() -> spacy.language.Language:
    """
    Load the spaCy English language model.

    :return: spaCy language model instance.
    """
    return spacy.load("en_core_web_sm")


def limit_repetitions(text: str, max_repeats: int = 5) -> str:
    """
    Limits repeated words/phrases within a line and across lines.

    - Collapses inline phrase repetition (e.g., 'thank you thank you ...')
    - Limits full-line repetitions (e.g., 'thank you.' repeated on new lines)

    :param text: Multi-line transcript text
    :param max_repeats: Maximum allowed repetitions (default: 5)
    :return: Cleaned text with reduced repetition
    """
    # Step 1: Collapse inline repeated phrases
    def collapse_inline_repeats(line):
        # Detect 2+ consecutive repeated phrases (1–5 words), collapse to max_repeats
        pattern = re.compile(r'\b((?:\w+\s+){0,4}\w+)(?:\s+\1){' + str(max_repeats) + r',}', flags=re.IGNORECASE)
        def replacer(match):
            phrase = match.group(1)
            return (' ' + phrase) * max_repeats
        return pattern.sub(replacer, line)

    # Step 2: Limit repeated full lines
    line_counts = defaultdict(int)
    result_lines = []

    for line in text.splitlines():
        collapsed_line = collapse_inline_repeats(line.strip())
        if not collapsed_line:
            result_lines.append('')
            continue
        line_counts[collapsed_line] += 1
        if line_counts[collapsed_line] <= max_repeats:
            result_lines.append(collapsed_line)

    return '\n'.join(result_lines)


def capitalize_sentences(text: str) -> str:
    """
    Capitalizes the first letter of each sentence in the input text.

    Sentences are detected by splitting on sentence-ending punctuation
    followed by whitespace (including newlines).

    :param text: Raw input text possibly containing multiple sentences.
    :return: Text with each sentence's first character capitalized,
             preserving the rest of the sentence as-is.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    capitalized = []

    for s in sentences:
        s = s.strip()
        if s:
            # Capitalize first character, keep the rest unchanged
            s = s[0].upper() + s[1:]
            capitalized.append(s)

    # Join sentences back with a space separating them
    return ' '.join(capitalized)


def normalize_text(text: str, deep: bool = False) -> str:
    """
    Cleanup of transcript text using regular expressions.
    Light cleaning (always applied):
    Cleanup of transcript text using regular expressions.
    Light cleaning (always applied):
    - Removes carriage returns
    - Trims leading and trailing spaces
    - Collapses multiple whitespace characters
    - Normalizes punctuation spacing
    - Capitalizes standalone 'i'
    - Fixes number formatting (e.g., "$ 5,000" → "$5,000")
    - Normalizes ellipses and excessive punctuation
    - Fixes acronyms like "u.s bank" → "U.S. Bank"

    Deep cleaning (if enabled):
    - Removes filler words
    - Strips content in [[double brackets]]
    - Re-collapses whitespace
    - Trims excessive repeated characters
    """
    # --- Light Cleaning ---
    # 1. Remove carriage returns
    text = text.replace('\r', '')

    # 2. Trim leading/trailing spaces per line
    text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)

    # 3. Normalize ellipses (fix multiple dots with spaces)
    text = re.sub(r'(\.\s*){2,}\.', '...', text)

    # 4. Fix multiple dots on acronyms (U.S... → U.S.)
    text = re.sub(r'(\b[A-Z]{1,3})\.+(?=\s|$)', r'\1.', text)

    # 5. Add space after punctuation except dot (avoid messing decimals)
    text = re.sub(r'([,!?;:])(\S)', r'\1 \2', text)

    # 6. Collapse multiple whitespace characters to single space
    text = re.sub(r'[\s\u00A0]+', ' ', text)

    # 7. Remove space before punctuation
    text = re.sub(r' ([.,!?;:])', r'\1', text)

    # 8. Capitalize standalone 'i'
    text = re.sub(r'\bi\b', 'I', text)

    # 9. Number formatting quirks fixes
    text = re.sub(r'(?<=\d),\s(?=\d{3}\b)', r',', text)  # 1, 000 → 1,000
    text = re.sub(r'(\d)\s+\.(\d)', r'\1.\2', text)  # 10. 05 → 10.05
    text = re.sub(r'\$\s(?=\d)', r'$', text)  # $ 5,000 → $5,000
    text = re.sub(r'(?<=\d)\s%(?!\S)', r'%', text)  # 10 % → 10%

    # 10. Fix excessive punctuation (like multiple exclamation marks)
    text = re.sub(r'([!?])\1{2,}', r'\1\1', text)  # "!!!" → "!!"

    if deep:
        text = re.sub(r'\b(um+|uh+|erm+|you know|like|so|well|ah|oh|mm|hmm|okay)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'\s{2,}', ' ', text)

        # Normalize multiple dots to ellipsis, in case not caught above
        text = re.sub(r'\.\.+', '...', text)

        # Trim repeated characters (e.g. "heyyy" → "heyy")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Collapse multiple spaces again after removals
        text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


def split_sentences(text: str, model: spacy.language.Language) -> List[str]:
    """
    Segment text into natural sentences using spaCy.

    :param text: Normalized input text
    :param model: Loaded spaCy model
    :return: List of individual sentence strings
    """
    doc = model(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def process_text_to_sentences(text: str, model: Optional[spacy.language.Language], use_spacy: bool = True, preserve_paragraphs: bool = False) -> str:
    """
    Process and segment transcript text into cleaned sentence lines.

    :param text: Raw input text
    :param model: spaCy language model
    :param use_spacy: Whether to use spaCy sentence splitting
    :param preserve_paragraphs: Keep blank lines between paragraph blocks
    :return: Processed text with one sentence per line (paragraphs separated if applicable)
    """
    if use_spacy and model:
        if preserve_paragraphs:
            paragraphs = re.split(r'\n\s*\n', text.strip())
            processed_paragraphs = []
            for para in paragraphs:
                sentences = split_sentences(para, model)
                processed_paragraphs.append("\n".join(sentences))
            return "\n\n".join(processed_paragraphs).strip() + "\n"
        else:
            sentences = split_sentences(text, model)
            return "\n".join(sentences).strip() + "\n"
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines).strip() + "\n"


def main() -> int:
    """
    Main entry point for command-line execution.
    Handles argument parsing, directory checks, and batch processing logic.

    :return: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Split transcript text files into individual sentences.",
    )
    parser.add_argument(
        "--txt-input",
        "-i",
        required=True,
        help="Input .txt file or directory",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="normalized",
        help="Output directory for processed files (default: ./normalized)",
    )
    parser.add_argument(
        "--no-spacy",
        action="store_true",
        help="Disable spaCy sentence segmentation (use line breaks instead)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--deep-clean",
        action="store_true",
        help="Enable deeper text normalization (remove fillers, brackets, etc.)",
    )
    parser.add_argument(
        "--preserve-paragraphs",
        action="store_true",
        help="Keep paragraphs separated by blank lines, splitting sentences inside paragraphs",
    )
    args = parser.parse_args()

    # Collect txt files from input path
    if not args.txt_input:
        print("❌ No input file or directory specified. Use --txt-input to set one.", file=sys.stderr)
        return 1

    try:
        file_list = get_files_list(args.txt_input)
        file_list = filter_file(file_list)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    # Determine output directory, default to the ./summary directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    use_spacy = not args.no_spacy
    spacy_model = None
    if use_spacy:
        try:
            spacy_model = load_spacy_model()
        except OSError:
            print(
                "❌ spaCy model not found. Install with: python -m spacy download en_core_web_sm",
                file=sys.stderr,
            )
            return 1

    for file_path in file_list:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")

        if os.path.isfile(output_file) and not args.force:
            print(f"↪ Skipping existing: {output_file}")
            continue

        # Get first pass normalized text using simple regex tools
        with open(file_path, "r", encoding="utf-8") as infile:
            raw_text = infile.read()

            # Perform cleaning and capitalization if required
            normalized_text = normalize_text(raw_text, deep=args.deep_clean)
            normalized_text = limit_repetitions(normalized_text, max_repeats=5)
            normalized_text = capitalize_sentences(normalized_text)

        processed_text = process_text_to_sentences(normalized_text, spacy_model, use_spacy, args.preserve_paragraphs)
        # write file out to filesystem
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(processed_text)
            print(f"✓ Wrote: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
