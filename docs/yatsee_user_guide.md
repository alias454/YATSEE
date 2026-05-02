# YATSEE: User Guide

YATSEE is a local-first audio extraction and processing pipeline designed to systematically refine raw meeting audio into clean, searchable, and auditable artifacts. It automates the work of acquiring source media, preparing audio, transcribing speech, normalizing text, and generating higher-level intelligence from long-form public recordings.

# Table of Contents

- [Stage 1 – Audio Intake & Download](#stage-1--audio-intake--download)
- [Stage 2 – Audio Formatting & Chunking](#stage-2--audio-formatting--chunking)
- [Stage 3 – Transcription](#stage-3--transcription)
- [Stage 4a – Slicing Transcripts into Segments](#stage-4a--slicing-transcripts-into-segments)
- [Stage 4b – Transcript Normalization & Structure](#stage-4b--transcript-normalization--structure)
- [Stage 5 – Intelligence, Summarization & Provider-Based LLM Processing](#stage-5--intelligence-summarization--provider-based-llm-processing)

---

## Stage 1 – Audio Intake & Download

### Purpose:
Fetch source media for a configured entity and place it into the pipeline as raw downloadable input. This is the funnel entry for audio-first processing and is typically used for YouTube-backed source acquisition.

### Input:
Configured entity source definitions or an explicitly provided source path, such as a YouTube channel or playlist.

### Output:
Downloaded source media stored in `downloads/` under the entity data path, or in a user-specified output directory.

### How it Works:
YATSEE loads the global configuration and the selected entity configuration, merges them, and resolves the enabled source settings for the run. For YouTube-based acquisition, media is resolved using `yt-dlp`. Optional date filters allow selective fetching of newer or older recordings. Playlist caching can be generated independently to avoid repeated upstream resolution work.

### The system is idempotent:
- `.downloaded` tracking prevents re-fetching the same source items repeatedly.
- Playlist cache files can be reused to reduce repeated source discovery work.
- Repeated fetch runs skip already-known media where possible.

### Why This Stage Matters:
- Centralizes source acquisition into a predictable entry point.
- Keeps input handling consistent before audio formatting begins.
- Reduces duplicate downloads and wasted bandwidth.
- Lets you stop after source acquisition if that is all you need.

### User Options / CLI Flags:
- `-e` / `--entity`: Which entity to process.
- `--source`: Optional specific source adapter, such as `youtube`.
- `-c` / `--config`: Path to global configuration file. Default: `yatsee.toml`.
- `-o` / `--output-dir`: Override output directory for fetched media.
- `--date-after` / `--date-before`: Filter source media by date.
- `--make-playlist`: Rebuild playlist cache and exit.

### Usage Examples:
```bash
yatsee source fetch -e example_entity
yatsee source fetch -e example_entity --date-after 20260101
yatsee source fetch -e example_entity --source youtube --output-dir ./downloads
yatsee source fetch -e example_entity --make-playlist
```

### Design Notes:
- Source acquisition is separated from later audio formatting and transcription stages.
- This stage is intended to be repeatable and safe to rerun.
- The primary workflow is entity-driven, but targeted overrides remain possible when needed.

---

## Stage 2 – Audio Formatting & Chunking

### Purpose:
Convert raw source media into transcription-ready audio, optionally splitting long recordings into manageable chunks. This stage prepares media for ASR workflows by enforcing a consistent audio format.

### Input:
Source files from `downloads/` or a user-provided file or directory.

### Output:
Normalized audio written to `audio/` under the entity path, typically as `.wav` or `.flac`. Optional chunks written to `audio/chunks/<base_name>/`.

### How it Works:
YATSEE discovers input files, resolves formatting preferences from configuration and CLI options, and converts supported media using `ffmpeg` and `ffprobe`. Audio is normalized to mono, 16 kHz output suitable for transcription. Long recordings can be split into sequential chunks with overlap to reduce context loss near chunk boundaries. Tracking prevents unnecessary repeated conversion work unless forced.

### Why This Stage Matters:
- Standardizes audio for more predictable transcription quality.
- Makes long recordings easier to process on modest hardware.
- Allows chunk sizing and overlap tuning without changing source media.
- Preserves source media while producing clean derived artifacts.

### User Options / CLI Flags:
- `-e` / `--entity`: Entity handle to process.
- `-c` / `--config`: Global configuration path. Default: `yatsee.toml`.
- `-i` / `--input-dir`: Input file or directory override.
- `-o` / `--output-dir`: Directory to save normalized audio.
- `--format`: Output format, either `wav` or `flac`.
- `--create-chunks`: Enable chunk generation.
- `--chunk-duration`: Chunk length in seconds. Default: `600`.
- `--chunk-overlap`: Overlap in seconds between chunks. Default: `2`.
- `--dry-run`: Preview actions without writing files.
- `--force`: Reprocess files even if already converted.

### Usage Examples:
```bash
yatsee audio format --entity example_entity
yatsee audio format --input-dir ./raw_audio --format wav
yatsee audio format --entity example_entity --create-chunks --chunk-duration 300
yatsee audio format --entity example_entity --dry-run
yatsee audio format --input-dir ./raw_audio --output-dir ./formatted_audio --force
```

### Design Notes:
- Formatting, discovery, hashing, and chunking are separate concerns internally.
- This stage is intended to feed transcription cleanly, not to alter the semantic content of the recording.
- The output contract is deliberately stable so downstream stages can consume it predictably.

---

## Stage 3 – Transcription

### Purpose:
Convert formatted audio into structured transcripts for search, normalization, summarization, and downstream analysis.

### Input:
Audio files from `audio/`, chunk directories under `audio/chunks/`, or a user-specified input file or directory.

### Output:
WebVTT transcript files stored in `transcripts_<model>/` under the entity data path, or in a user-specified output directory.

### How it Works:
YATSEE loads merged configuration, resolves transcription settings, and processes audio using Whisper or faster-whisper. Execution can target CUDA, CPU, or Apple MPS depending on environment and configuration. Chunked audio can be transcribed sequentially for long recordings. Transcript segments are normalized and near-overlapping segments are merged for readability and consistency. Tracking prevents redundant transcription of unchanged inputs.

### Why This Stage Matters:
- Transforms audio into machine-usable text.
- Supports long recordings without requiring the entire source to fit comfortably in memory.
- Allows model, language, and device selection per run.
- Provides the base text artifacts required by slicing, normalization, and summarization.

### User Options / CLI Flags:
- `-e` / `--entity`: Entity handle to process.
- `-c` / `--config`: Global configuration path. Default: `yatsee.toml`.
- `-i` / `--audio-input`: Audio file or directory to transcribe.
- `-o` / `--output-dir`: Directory to save transcripts.
- `-g` / `--get-chunks`: Transcribe chunked audio.
- `-m` / `--model`: Transcription model override.
- `--faster`: Use faster-whisper if available.
- `-l` / `--lang`: Language code or `auto`. Default: `en`.
- `-d` / `--device`: Execution device such as `auto`, `cuda`, `cpu`, or `mps`.
- `-v` / `--verbose`: Verbose output.
- `-q` / `--quiet`: Reduced output.

### Usage Examples:
```bash
yatsee audio transcribe --entity example_entity
yatsee audio transcribe --audio-input ./audio --model small --faster
yatsee audio transcribe --audio-input ./single_file.mp3 --device cpu --lang es
yatsee audio transcribe --audio-input ./audio_folder --output-dir ./transcripts
```

### Design Notes:
- Transcription is kept separate from text cleanup and summarization.
- Segment normalization is deterministic to keep reruns stable.
- The primary transcript artifact is VTT because it preserves timing while remaining readable.

---

## Stage 4a – Slicing Transcripts into Segments

### Purpose:
Transform VTT transcripts into structured, sentence-aware segments for downstream embedding, indexing, or semantic analysis. This stage also produces plain text transcript output that other stages can consume.

### Input:
WebVTT transcript files from `transcripts_<model>/` or a user-specified file or directory.

### Output:
- Plain text transcript files (`.txt`)
- Optional `.segments.jsonl` files containing structured segment records
- Optional segment embeddings when enabled

### How it Works:
YATSEE reads transcript cues and consolidates them into sentence-aware units. Lines lacking terminal punctuation can be merged to avoid fragmented text. Optional max-window constraints can force upper bounds on segment duration. When embedding generation is enabled, segment records include timestamps and embeddings suitable for downstream retrieval workflows.

### Why This Stage Matters:
- Produces structured transcript fragments that are easier to embed and search.
- Maintains timestamp alignment for later auditability.
- Provides plain text output for normalization and summarization.
- Makes later retrieval-oriented workflows more coherent than raw VTT alone.

### User Options / CLI Flags:
- `-e` / `--entity`: Entity handle to process.
- `-c` / `--config`: Global configuration path. Default: `yatsee.toml`.
- `-i` / `--vtt-input`: VTT file or folder to slice.
- `-o` / `--output-dir`: Directory for sliced outputs.
- `-m` / `--model`: Embedding model override when generating embeddings.
- `-g` / `--gen-embed`: Generate JSONL segments with embeddings.
- `--max-window`: Hard upper bound on segment length. Default: `90.0`.
- `--force`: Overwrite existing outputs.
- `-d` / `--device`: Execution device such as `auto`, `cuda`, `cpu`, or `mps`.

### Usage Examples:
```bash
yatsee transcript slice --entity example_entity --gen-embed
yatsee transcript slice --vtt-input ./transcripts --max-window 30 --force
yatsee transcript slice --vtt-input ./entity/transcripts_small --output-dir ./segments
yatsee transcript slice --entity example_entity --device cpu
```

### Design Notes:
- This stage bridges timing-preserving transcript artifacts and retrieval-friendly structured text.
- Deterministic identifiers and structured segment metadata help downstream indexing remain stable.
- It can be run independently on one transcript or over a full entity batch.

---

## Stage 4b – Transcript Normalization & Structure

### Purpose:
Convert transcript text into cleaner, more consistent sentence-per-line or paragraph-preserving output for summarization, embeddings, or semantic search.

### Input:
Plain `.txt` transcript files, typically derived from Stage 4a and stored under `transcripts_<model>/` or another user-specified input location.

### Output:
Normalized `.txt` files written to `normalized/` under the entity data path, or a user-specified output directory.

### How it Works:
YATSEE loads the merged configuration, resolves normalization preferences, and cleans transcript text using sentence splitting and replacement rules. spaCy can be used for sentence-aware processing, or disabled when a simpler line-preserving workflow is preferred. Normalization can include punctuation cleanup, filler reduction, spacing cleanup, and application of configured replacements. Optional deep cleaning or paragraph preservation modes support different downstream use cases.

### Why This Stage Matters:
- Produces cleaner text for LLM summarization and retrieval workflows.
- Reduces ASR clutter and recurring transcription artifacts.
- Applies entity-specific replacements consistently across runs.
- Gives downstream consumers a more stable and usable text artifact than raw transcript output.

### User Options / CLI Flags:
- `-e` / `--entity`: Entity handle to process.
- `-c` / `--config`: Global configuration path. Default: `yatsee.toml`.
- `-i` / `--input-path`: Input file or directory.
- `-o` / `--output-dir`: Directory to save normalized files.
- `-m` / `--model`: Transcription model suffix override for input path resolution.
- `--no-spacy`: Disable spaCy sentence splitting.
- `--deep-clean`: Enable more aggressive cleanup.
- `--preserve-paragraphs`: Preserve paragraph structure.
- `--force`: Overwrite existing outputs.

### Usage Examples:
```bash
yatsee transcript normalize --entity example_entity
yatsee transcript normalize --input-path ./transcripts_small --output-dir ./normalized_out
yatsee transcript normalize --input-path ./transcripts_small/file.txt --deep-clean
yatsee transcript normalize --entity example_entity --no-spacy --preserve-paragraphs
```

### Design Notes:
- This stage is about text quality and consistency, not semantic interpretation.
- Replacement rules from configuration are often as important as model quality for clean downstream output.
- It supports both lighter cleanup and heavier cleanup depending on the next consumer.

---

## Stage 5 – Intelligence, Summarization & Provider-Based LLM Processing

### Purpose:
Generate structured summaries and other higher-level intelligence artifacts from transcript text using a configurable provider layer rather than a single hardcoded model runtime.

### Input:
Single transcript file or a directory of `.txt` files, typically from normalized transcript output or another prepared text source. Optional human-readable context, prompt configuration, provider selection, provider security settings, and pricing-reference configuration may also be supplied.

### Output:
Final summaries written to the configured or specified output directory, typically in Markdown and optionally YAML. Optional intermediate chunk summaries can also be written for debugging.

### How it Works:
YATSEE classifies transcript content when appropriate, resolves prompt routing, and processes transcripts through a multi-pass summarization workflow. Transcripts are chunked based on word, sentence, or density-aware strategies depending on configuration and input size. Chunk-level summaries are refined across passes until a final structured report is produced.

The intelligence stage now uses a provider abstraction. That means YATSEE can generate text through multiple backends, including local runtimes and hosted APIs, without changing the summarization workflow itself.

Supported provider patterns currently include:
- local HTTP runtimes such as Ollama
- local OpenAI-compatible runtimes such as llama.cpp
- hosted APIs such as OpenAI and Anthropic
- CLI-backed providers such as `codex_cli`

YATSEE also applies provider-target hardening before execution. That means local-first provider choices remain the default posture unless you explicitly relax those controls in configuration.

YATSEE can also optionally calculate reference pricing for a run. This is useful when the actual work is performed locally but you want to estimate what the same token volume would have cost on a hosted provider.

### Why This Stage Matters:
- Converts long-form transcripts into usable high-level artifacts.
- Preserves important civic details such as motions, votes, decisions, and follow-up items.
- Supports multiple prompt workflows without changing core pipeline structure.
- Decouples summarization logic from any single model runtime.
- Makes local-vs-hosted cost comparison possible without changing actual execution.

### Provider Security Defaults:
By default, YATSEE keeps the provider layer locked down unless you explicitly opt in through config.

That means:
- remote non-local targets for local HTTP providers are blocked unless `llm_allow_remote = true`
- insecure HTTP for hosted providers is blocked unless `llm_allow_insecure_http = true`
- custom CLI executable targets are blocked unless `llm_allow_custom_executable = true`

If those settings are omitted from config, YATSEE still behaves as if they are `false`.

### User Options / CLI Flags:
- `-e` / `--entity`: Entity handle to process.
- `-c` / `--config`: Path to global `yatsee.toml`.
- `-i` / `--txt-input`: Transcript file or directory to summarize.
- `-o` / `--output-dir`: Directory for final summaries.
- `-m` / `--model`: LLM model override.
- `--llm-provider`: Provider override such as `ollama`, `llamacpp`, `openai`, `anthropic`, or `codex_cli`.
- `--llm-provider-url`: Provider URL or executable target override.
- `--llm-api-key`: Provider API key override.
- `--show-pricing`: Enable reference pricing for the run.
- `--no-show-pricing`: Disable reference pricing even if enabled in config.
- `--pricing-provider`: Reference provider used for pricing estimation.
- `--pricing-model`: Reference model used for pricing estimation.
- `-f` / `--output-format`: Output format, either `markdown` or `yaml`. Default: `markdown`.
- `-j` / `--job-profile`: Prompt workflow, either `civic` or `research`. Default: `civic`.
- `-s` / `--chunk-style`: Chunking method, one of `word`, `sentence`, or `density`. Default: `word`.
- `-w` / `--max-words`: Approximate word threshold per chunk.
- `-t` / `--max-tokens`: Approximate token threshold per chunk.
- `-p` / `--max-pass`: Maximum number of summarization passes. Default: `3`.
- `-d` / `--disable-auto-classification`: Disable automatic prompt routing.
- `--first-prompt`, `--second-prompt`, `--final-prompt`: Manual prompt overrides.
- `--context`: Optional context to guide summarization.
- `--print-prompts`: Display resolved prompts and exit.
- `--enable-chunk-writer`: Write intermediate chunk outputs for debugging.

### Usage Examples:
```bash
yatsee intel run -e example_entity

yatsee intel run -e example_entity --model llama3:latest --llm-provider ollama --llm-provider-url http://localhost:11434

yatsee intel run --txt-input council_meeting_2025_06_01.txt --context "City Council - June 2025" --model mistral-nemo:latest --llm-provider llamacpp --llm-provider-url http://localhost:8080

yatsee intel run --txt-input finance_committee_2025_05.txt --llm-provider codex_cli --llm-provider-url codex --model gpt-5.4 --show-pricing --pricing-provider openai --pricing-model gpt-5.4

yatsee intel run --txt-input firehall_meeting_2025_05.txt --llm-provider anthropic --llm-provider-url https://api.anthropic.com --llm-api-key "$ANTHROPIC_API_KEY" --model claude-opus-4.1
```

### Design Notes:
- Summarization is intentionally multi-pass because long civic transcripts routinely exceed comfortable single-pass context windows.
- Prompt orchestration, provider execution, pricing estimation, chunking, and output writing are separate concerns internally.
- This stage is designed to extract durable, structured intelligence rather than produce a generic free-form recap.
- Reference pricing is only an estimate. It depends on the configured pricing table and estimated token counts unless exact provider usage metadata is available.
- Provider hardening settings are intentionally config-driven rather than casually exposed as one-run CLI switches.