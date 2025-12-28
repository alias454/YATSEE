#!/usr/bin/env python3
"""
yatsee_polish_transcript.py

THE TRANSFORM LAYER.
Restores punctuation and capitalization to raw ASR text using a BERT-based model.

Input:  Raw .txt files from 'transcripts/' (Layer 1 - Canonical)
Output: .punct.txt files to 'normalized/' (Layer 2 - Transform)

Usage:
  ./yatsee_polish_transcript.py -i ./transcripts -o ./normalized -d auto
"""

import os
import sys
import argparse
import torch
import textwrap
from glob import glob
from tqdm import tqdm
from deepmultilingualpunctuation import PunctuationModel

# Default SOTA model for punctuation
DEFAULT_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"


def main():
    parser = argparse.ArgumentParser(
        description="Restore punctuation in raw text files using Deep Learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Usage Examples:
              ./yatsee_polish_transcript.py -i ./transcripts -o ./normalized
              # Force CPU if VRAM is full
              ./yatsee_polish_transcript.py -i ./transcripts -o ./normalized -d cpu
        """)
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw .txt files (Layer 1)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Directory to save .punct.txt files (Layer 2)."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device for model execution (default: auto)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in output directory."
    )

    args = parser.parse_args()

    # --- DEVICE DETECTION LOGIC ---
    # If user explicitly requests CPU, we must hide the GPU to prevent OOM
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("⚡ Device: CPU (Forced via args)")

    elif torch.cuda.is_available() and args.device in ["auto", "cuda"]:
        # NVIDIA GPU
        torch.cuda.empty_cache()
        print("⚡ Device: NVIDIA CUDA (FP16 enabled)")

    elif torch.backends.mps.is_available() and args.device in ["auto", "mps"]:
        # Apple Silicon
        print("⚡ Device: APPLE METAL (MPS)")

    else:
        # Fallback
        if args.device == "cuda":
            print("⚠️ CUDA requested but not available, falling back to CPU.", file=sys.stderr)
        if args.device == "mps":
            print("⚠️ MPS requested but not available, falling back to CPU.", file=sys.stderr)
        print("⚡ Device: CPU")

    # --- LOAD MODEL ---
    print(f"📥 Loading Model: {args.model}...")
    try:
        # The library wrapper handles loading.
        # Note: We do NOT manually move the model to device here anymore to avoid AttributesErrors.
        # The library uses torch.cuda.is_available() internally.
        model = PunctuationModel(model=args.model)

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # --- PREPARE DIRECTORIES ---
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # --- FIND FILES ---
    input_files = glob(os.path.join(args.input_dir, "*.txt"))
    if not input_files:
        print(f"❌ No .txt files found in {args.input_dir}")
        return 1

    print(f"🧹 Processing {len(input_files)} transcripts...")

    # --- PROCESS LOOP ---
    success_count = 0

    for filepath in tqdm(input_files, desc="Polishing"):
        filename = os.path.basename(filepath)
        # Use the .punct.txt extension to indicate this is Step C (Punctuated)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.punct.txt"
        save_path = os.path.join(args.output_dir, output_filename)

        # Skip logic
        if os.path.exists(save_path) and not args.force:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()

            if not raw_text.strip():
                continue

            # INFERENCE
            clean_text = model.restore_punctuation(raw_text)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(clean_text)

            success_count += 1

        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")

    print(f"\n✨ Done. {success_count} transcripts polished.")
    print(f"📂 Output: {os.path.abspath(args.output_dir)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())