# **YATSEE** -- Yet Another Tool for Speech Extraction & Enrichment

YATSEE is a local-first, end-to-end data pipeline designed to systematically refine raw meeting audio into a clean, searchable, and auditable intelligence layer. It automates the tedious work of downloading, transcribing, and normalizing unstructured conversations.

# Table of Contents

- [Stage 1 – Audio Intake & Download](#stage-1--audio-intake--download)
- [Stage 2 – Audio Normalization & Chunking](#stage-2--audio-normalization--chunking)
- [Stage 3 – Transcription & Text Normalization](#stage-3--transcription--text-normalization)
- [Stage 4a – Slicing Transcripts into Segments](#stage-4a--slicing-transcripts-into-segments)
- [Stage 4b – Transcript Normalization & Structure](#stage-4b--transcript-normalization--structure)
- [Stage 5 – Summarization & Multi-Pass Processing](#stage-5--summarization--multi-pass-processing)
- [Optional Utilities – YATSEE Sound Recorder](#optional-utilities-yatsee-sound-recorder)

---

## Stage 1 – Audio Intake & Download

### Purpose:
Fetch audio from YouTube sources defined in YATSEE entity configurations, producing raw audio files ready for transcription. This is the first step in the pipeline—the “funnel entry”—where your content comes in, no matter the source.

### Input:
YouTube channels or playlists specified in your entity configuration.

### Output:
Downloaded audio files stored in downloads/ under the entity path.

### How it Works:
YATSEE reads your global and entity-specific configuration. Missing keys fail loudly, so you know immediately if something’s misconfigured.
Videos are resolved via yt-dlp. FFmpeg handles the audio extraction and conversion to MP3.
Optional date filters allow you to only download content after or before a certain date.

### The system is idempotent:
.downloaded tracker file prevents re-downloading the same content.
.playlist_ids.json caches playlist resolution to avoid repeated YouTube queries.

### Why This Stage Matters:
Centralizes your audio collection from multiple sources.
Ensures consistency: everything that moves to Stage 2 is in the same format.
Avoids duplication and wasted bandwidth with built-in caching and tracking.

### User Options / Argparse Flags:
-e / --entity: Which entity to process (required).  
-c / --config: Path to global configuration (default yatsee.toml).  
-o / --output-dir: Override output location for MP3s.  
--js-runtime: Choose JavaScript runtime for yt-dlp (deno, node, quickjs, etc.).  
--youtube-path: Specific channel or playlist to download.  
--date-after / --date-before: Filter videos by date.  
--dry-run: Resolve URLs without downloading.  
--make-playlist: Generate a playlist cache file and exit.  

### Usage Examples:
python yatsee_download_audio.py -e defined_entity    
python yatsee_download_audio.py -e defined_entity --date-after 20260101 --dry-run  
python yatsee_download_audio.py --youtube-path @channel/streams --output-dir ./audio  
python yatsee_download_audio.py -e defined_entity --make-playlist

### Design Notes:
Modular structure separates config loading, YouTube resolution, and download execution.
Ensures everything entering Stage 2 (formatting) is named correctly and ready to be converted to FLAC 16Khz audio.
Allows partial workflows: if you only want to download audio, you can stop here.

---

## Stage 2 – Audio Formatting & Chunking

### Purpose:
Convert raw downloaded audio into a format suitable for AI transcription, optionally splitting long recordings into manageable chunks. This stage ensures your audio is ready for Whisper-style transcription models while keeping your workflow efficient and reproducible.

### Input:
Raw media files from downloads/ (MP3, WAV, FLAC, MP4, M4A, WEBM) under the entity path or a specified directory.

### Output:
Formatted audio in audio/ under the entity path (.wav or .flac).
Optional sequential chunks stored in audio/chunks/<base_name>/.

### How it Works:
YATSEE reads the global and entity-specific configuration to determine formatting preferences.
Files are converted using ffmpeg/ffprobe, normalized to mono, 16kHz.
Optionally, long audio files are split into sequential chunks with configurable overlap to prevent sentence truncation.
SHA-256 tracking prevents reprocessing files unnecessarily.
Dry-run mode previews operations without writing files; force mode reprocesses even if files were already converted.
Original media is preserved; chunks are additional outputs.

### Why This Stage Matters:
Standardizes audio across multiple sources for consistent transcription quality.
Avoids memory overload by chunking long recordings.
Supports experimentation: different chunk lengths or overlaps can improve transcription fidelity and downstream summarization.
Idempotent design ensures efficiency in repeated runs.

### User Options / Argparse Flags:
-e / --entity: Entity handle to process.  
-c / --config: Global YATSEE configuration path (default: yatsee.toml).  
-i / --input-dir: Direct override for source files.  
-o / --output-dir: Directory to save normalized audio.  
--format: Output format (wav or flac).  
--create-chunks: Enable splitting audio into chunks.  
--chunk-duration: Length of each chunk in seconds (default 600).  
--chunk-overlap: Seconds of overlap between chunks (default 2).  
--dry-run: Preview actions without changing files.  
--force: Reprocess files even if already converted.

### Usage Examples:
python yatsee_format_audio.py -e defined_entity    
python yatsee_format_audio.py --input-dir ./raw_audio --format wav  
python yatsee_format_audio.py -e defined_entity --create-chunks --chunk-duration 300  
python yatsee_format_audio.py --dry-run  
python yatsee_format_audio.py -i ./raw_audio -o ./formatted_audio --force  

### Design Notes:
Modular design separates configuration, file discovery, hashing, conversion, and chunking.
Default values ensure smooth operation even with partial configuration.
Designed to integrate seamlessly with Stage 3 (Transcription & Cleaning).

---

## Stage 3 – Transcription

### Purpose:
Convert your formatted audio into structured text for search, indexing, and summarization. Stage 3 ensures that raw speech becomes usable transcripts while maintaining quality and context, even for long or chunked recordings.

### Input:
Audio files from audio/ or a user-specified file or directory.

#### Optional:
Chunked audio files in audio/chunks/<file_basename>/.

### Output:
WebVTT (.vtt) transcripts stored in transcripts_<model>/ under the entity path or a specified directory.
Overlapping or near-overlapping segments are merged for readability.

### How it Works:
YATSEE loads the merged entity configuration (global + local) for settings like hotwords, people aliases, and transcription preferences.
Audio is processed using Whisper or faster-whisper, with optional GPU (CUDA), Apple MPS, or CPU execution. Automatic fallback ensures smooth operation.
Chunked audio is transcribed sequentially to handle long recordings efficiently.
Overlapping segments are merged deterministically, producing clean, high-quality VTT files.
SHA-256 hashing tracks already-transcribed files, avoiding redundant work.
Verbose or quiet output modes give you control over logging detail.

### Why This Stage Matters:
Transforms raw audio into structured text for downstream tasks like summarization, action-item extraction, or semantic search.
Handles chunked audio and long recordings without losing context or clarity.
Supports language selection and hotword emphasis for better accuracy in specialized domains.
Ensures reproducibility and determinism for entities, which is critical for civic or research applications.

### User Options / Argparse Flags:
-e / --entity: Entity handle to process.  
-c / --config: Global configuration path (yatsee.toml).  
-i / --audio-input: Single file or folder to transcribe (default ./audio).  
-o / --output-dir: Directory to save transcripts.  
-g / --get-chunks: Transcribe chunked audio files.  
-m / --model: Whisper model size (small, medium, large, turbo, etc.).  
--faster: Use faster-whisper if installed.  
-l / --lang: Language code or auto (default en).  
-d / --device: Execution device (auto, cuda, cpu, mps).  
-v / --verbose: Verbose output.  
-q / --quiet: Suppress verbose output.  

### Usage Examples:
python yatsee_transcribe_audio.py -e defined_entity    
python yatsee_transcribe_audio.py --audio-input ./audio --model small --faster  
python yatsee_transcribe_audio.py -i ./single_file.mp3 -d cpu --lang es  
python yatsee_transcribe_audio.py --audio-input ./audio_folder -o ./transcripts/  

### Design Notes:
Modular design separates configuration loading, file discovery, chunk handling, hotword flattening, segment normalization, and VTT writing.
Deterministic merging ensures repeated runs produce identical outputs.
Fully local operation; no cloud services required.
Integrates seamlessly with Stage 4 (Cleaning & Normalization).

---

## Stage 4a – Slicing Transcripts into Segments

### Purpose:
Stage 4a transforms your VTT transcripts into structured, sentence-aware segments for embedding, indexing, or semantic search. This step ensures that each segment is contextually coherent, timestamped, and ready for downstream AI processing.

### Input:
WebVTT files (.vtt) from Stage 3, typically in transcripts_<model>/.
Optional custom directories via --vtt-input.

### Output:
Plain text transcripts (.txt) preserving line breaks.
Optional JSONL segments (.segments.jsonl) containing:
  start_time, end_time, duration
  text_raw (sentence content)
  segment_index, source_id, video_id

### How it Works:
Reads the VTT file and consolidates lines into sentence-aware cues, merging lines that lack terminal punctuation.
Optionally applies max-window slicing to enforce uniform segment lengths.
Generates deterministic placeholder video IDs when real IDs are missing, ensuring reproducibility.
Respects entity-specific configuration overrides, merging them with global yatsee.toml.
CLI options allow control over verbosity, output location, device selection, and embedding generation.

### Why This Stage Matters:
Produces granular, structured segments suitable for semantic indexing, search, or multi-pass summarization.
Maintains context while preparing data for embedding models.
Offers flexibility: generate plain text only, JSONL only, or JSONL with embeddings.
Deterministic outputs allow consistent pipelines across multiple runs or entities.

### User Options / Argparse Flags:
-e / --entity: Entity handle to process.  
-c / --config: Global configuration path (yatsee.toml).  
-i / --vtt-input: VTT file or folder to slice.  
-o / --output-dir: Directory for segment output.  
-m / --model: SentenceTransformer model name for embedding generation.  
-g / --gen-embed: Generate JSONL segments with embeddings and timestamps.  
--max-window: Hard upper limit (seconds) on segment duration.  
--force: Overwrite existing outputs.  
-d / --device: Execution device (auto, cuda, cpu, mps).  
--verbose: Verbose logging.  
--quiet: Suppress output.  

### Usage Examples:
python yatsee_slice_vtt.py -e defined_entity --gen-embed  
python yatsee_slice_vtt.py --vtt-input ./transcripts --max-window 30 --force  
python yatsee_slice_vtt.py -i ./entity/transcripts_small --output-dir ./segments  
python yatsee_slice_vtt.py -e defined_entity --quiet

### Design Notes:
Modular functions handle config loading, VTT parsing, sentence consolidation, JSONL generation, and optional embeddings.
Ensures deterministic segment IDs for consistent indexing.
Can be run independently on a single transcript or batch-processed for entire entities.
Fully local; no cloud dependencies are required.

---

## Stage 4b – Transcript Normalization & Structure

### Purpose:
Stage 4b (Normalization) takes transcripts and converts them into clean, sentence-per-line files, making them ready for summarization, embedding, or semantic indexing. This step ensures that your text is consistent, readable, and free from filler noise or formatting artifacts.

### Input:
Plain .txt files from previous slicing steps, typically in transcripts_<modlel>/ under the entity data path.
Custom input directories or individual files via --input-dir.

### Output:
Cleaned .txt files with one sentence per line, saved under normalized/ in the entity path.
Optional paragraph preservation or deep cleaning depending on flags.

### How it Works:
Sentence Splitting: Uses spaCy to break text into individual sentences. Can be disabled if you prefer a simple line-by-line approach.

### Text Normalization:
Collapses repeated characters and phrases.
Removes filler words and optional bracketed content.
Corrects punctuation, spacing, and capitalization.
Preserves numbers, acronyms, and entity names.
Optional Deep Cleaning: Removes noise that may interfere with embeddings or AI summarization.

### Paragraph Preservation: Maintains paragraph structure if needed for context-sensitive tasks.
Config Overrides: Entity-specific replacements cascade from global + local TOML configs for deterministic results.

### Why This Stage Matters:
Produces high-quality, AI-ready text without the clutter from raw transcripts.
Ensures embeddings, semantic search, or summarization tasks work reliably.
Provides consistent outputs across different entities and runs.
Flexible: choose light cleaning, deep cleaning, or paragraph-preserving workflows depending on your use case.

### User Options / Argparse Flags:
-e / --entity: Entity handle to process.  
-c / --config: Global configuration path (yatsee.toml).  
-i / --input-dir: Input file or directory.  
-o / --output-dir: Directory to save normalized files.  
-m / --model: spaCy model to use (en_core_web_md, etc.).  
--no-spacy: Disable spaCy sentence splitting.  
--force: Overwrite existing files.  
--deep-clean: Enable advanced cleaning of filler content and formatting noise.  
--preserve-paragraphs: Keep paragraph breaks intact.

### Usage Examples:
python yatsee_normalize_structure.py -e defined_entity  
python yatsee_normalize_structure.py --input-dir ./normalized --output-dir ./normalized_out  
python yatsee_normalize_structure.py -i ./normalized/file.txt --deep-clean  
python yatsee_normalize_structure.py -e entity_handle --no-spacy --preserve-paragraphs

### Design Notes:
Modular functions handle config loading, file discovery, sentence splitting, cleaning, and writing outputs.
Deterministic outputs allow repeated runs without accidental divergence.
Fully local; no cloud dependencies required.
Supports light or heavy cleaning depending on downstream AI tasks.

---

## Stage 5 – Summarization & Multi-Pass Processing

### Purpose:
Stage 5 takes normalized transcripts and produces structured, high-level summaries of meetings, including city councils, committees, town halls, or any spoken-word civic events. This stage uses a local LLM via Ollama, keeping all intermediate chunk summaries in memory for privacy and performance.

### Input:
Single transcript file or a directory of .txt files (output of Stage 4b).
Optional human-readable context (--context) to guide summarization.
Local Ollama models (e.g., llama3, mistral, gemma) pulled via ollama pull.

### Output:
Final merged summary written to --output-dir (default: ./summary/).
Supports Markdown (default) or YAML format.
Intermediate chunk summaries exist in memory only and are not written to disk unless debugging flags are used.

### How it Works:
Automatic Meeting Classification:
Determines meeting type using transcript content and optional filename hints.
Selects prompt workflows dynamically based on type (overview, action items, detailed, etc.).

### Multi-Pass Summarization:
Summarization happens in iterative passes (--max-pass) for context depth.
Users can override auto-classification and supply manual prompts for each pass.
Chunking of transcripts can be done by word, sentence, or density, depending on desired granularity.

### Prompt Modularity:
Different prompt types for different summary styles:
overview, action_items, detailed, more_detailed, most_detailed, final_pass_detailed.
Dynamic selection allows LLM to adapt output based on meeting type.
Optional manual overrides for full control (--first-prompt, --second-prompt, --final-prompt).

### Memory-First Design:
Prioritizes local computation and privacy.
Handles long transcripts via in-memory chunking and automatic merging.

### Structured Civic Focus:
Extracts motions, votes, decisions, speaker intent, and other civic-specific elements.
Maintains gender-neutral and consistent summary style for all outputs.

### User Options / Argparse Flags:
-e / --entity: Entity handle to process.  
-c / --config: Path to global yatsee.toml.  
-i / --txt-input: Transcript file or directory to summarize.  
-o / --output-dir: Directory for final summaries (default: summary/).  
-m / --model: Local Ollama model (e.g., llama3:latest, mistral:latest).  
-f / --output-format: Output format (markdown default, or yaml).  
-j / --job-type: Defines prompt workflow (summary default, research optional).  
-s / --chunk-style: Chunking method (word, sentence, density).  
-w / --max-words: Word threshold per chunk (default 3500).  
-t / --max-tokens: Approximate max tokens per chunk (overrides --max-words).  
-p / --max-pass: Maximum summarization iterations (default 3).  
-d / --disable-auto-classification: Turn off auto prompt selection; manual prompts required.  
--first-prompt, --second-prompt, --final-prompt: Manual prompt overrides per pass.  
--context: Optional meeting context to guide summarization.  
--print-prompts: Display all prompt templates and exit.  
--enable-chunk-writer: Write intermediate chunks to disk for debugging.  

### Usage Examples:
python yatsee_summarize_transcripts.py -e defined_entity  
python yatsee_summarize_transcripts.py --model llama3 -i council_meeting_2025_06_01 --context "City Council - June 2025"  
python yatsee_summarize_transcripts.py --model mistral -i firehall_meeting_2025_05 --context "Fire Hall Proposal Discussion" --output-format markdown  
python yatsee_summarize_transcripts.py --model gemma -i finance_committee_2025_05 --disable-auto-classification --first-prompt overview --second-prompt detailed --final-prompt final_pass_detailed

### Design Notes:
Modular functions handle config, chunking, prompt selection, and multi-pass summarization.
In-memory workflow ensures fast processing without polluting disk storage.
Designed to extract structured civic intelligence while remaining flexible for manual overrides or custom prompts.
Fully local, no cloud APIs required, supporting privacy-sensitive use cases.

---

## Optional Utilities (YATSEE Sound Recorder)

A lightweight, cross-platform utility for real-time audio capture of meetings. Designed to integrate seamlessly with the YATSEE pipeline, it produces high-fidelity FLAC files while preserving every word spoken.

### System Requirements:
Windows 10/11, macOS, or Linux
Python 3.10+ with libraries: customtkinter, sounddevice, soundfile, numpy

### OS-Level Microphone Tips:
Background hiss often comes from analog gain or mic boost. Keep "Mic Boost" / "Input Boost" at 0.0 dB for clean recordings.
Defaults to 24-bit recording for maximum digital headroom.
Windows-Specific Notes (Shure MV7 / USB mics):
Enable "Allow desktop apps to access your microphone" in Privacy settings.
For stability, select the device ending in (WASAPI).
Disable "Allow applications to take exclusive control" if recording fails.
Auto-Level in Shure MOTIV software is recommended for variable speaker distances.

### Production Deployment:
To bundle as a standalone executable:

pip install pyinstaller  
pyinstaller --noconsole --onefile yatsee_sound_recorder.py

### Key Features:
Threaded Queue: Ensures continuous audio capture without gaps.
Real-time Disk Write: Streams directly to disk to prevent data loss.
Modern Folder Picker: Dark-mode browser bypassing legacy dialogs.
FLAC Encoding: Lossless audio at ~50% the file size of WAV.

### Potential Future Features:
Silence detection to skip writing empty audio.
Voice-triggered start/stop via offline hotwords.
Rolling buffer to capture audio before triggers.
Expandable voice commands (save, discard, rename clips).