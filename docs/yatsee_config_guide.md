# YATSEE: Configuration & Orchestration Guide

YATSEE runs on a single config file that controls how the system interprets data. Most of the file is already wired with sane defaults. The parts that matter are the sections that define names, places, roles, and recurring issues specific to their city or county.

If these fields stay generic, the model stays generic. When you fill them in with real local data, the system stops guessing and starts identifying people and topics correctly across all meeting transcripts.

This document walks through each section of the config file, explains what it does, and points out exactly where you need to plug in your own information. Everything else can be left alone unless you know what you’re doing.

## Global System Configuration (yatsee.toml)

### 1. System

The `[system]` block controls global behavior for logging, file paths, and model provider access.

#### **root_data_dir**
- Path where all generated data, cached artifacts, and model outputs will be stored.
- Change this if you want YATSEE to write data somewhere else (for example, a mounted network volume).

#### **log_level**
- Controls how noisy the logs are.
- Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
- Use `DEBUG` only when troubleshooting.

#### **log_format**
- Standard Python logging format string.
- Most users will never need to change this unless integrating with structured logging.

#### **llm_provider_url**
- Base URL for the model runtime (Ollama or compatible).
- You must update this if your models run on a remote box, a different port, or behind a proxy.

---

### 2. Default Models

YATSEE uses multiple specialized subsystems (summarization, embeddings, transcription, etc.). Each has a default model defined here.

#### **default_js_runtime**
- Required for YouTube challenge negotiation using `yt-dlp`.
- Typically `"deno"` or `"node"`.
- Change only if you know which runtime your system supports.

#### **default_summarization_model**
- The model used to generate long-form summaries.
- Must match a key under `[models]`.

#### **default_transcription_model**
- Whisper or faster-whisper model name used for audio transcription.

#### **default_sentence_model**
- SpaCy model used for downstream sentence-level operations.

#### **default_embedding_model**
- HuggingFace-compatible embedding model for vectorization.
- Update this if you prefer a different embedding approach or multilingual support.

---

### 3. Model Definitions

The `[models]` block defines per-model runtime configuration.

Each key corresponds to an actual model name used by your LLM provider.  
Each model entry includes:

- **append_dir**  
  Subdirectory automatically created under `root_data_dir` to store output for that model.

- **max_tokens**  
  Target chunk size for summarization. Should be roughly half of the model's context window. Adjust this if you are not seeing the quality you expect from your summaries.

- **num_ctx**  
  Maximum supported context window for the model.

You only need to edit these if:
- You add new models,
- You want different chunking behavior,
- You reorganize output directories.

---

### 4. Entity Definitions

Entities tell YATSEE *what* to track, *where* it belongs in the hierarchy, and *which inputs* it consumes.

Each entry under `[entities.<name>]` contains:

#### **display_name**
- Human-readable name shown in logs or UI.

#### **base**
- The logical classification path for data organization.  
  Example format: `country.US.state.IL.` or `media.youtube.channel.`

#### **entity**
- Canonical name used for file paths and identifiers.

#### **inputs**
- List of enabled input processors(Only YouTube supported for now).  
  Typical value: `["youtube"]`.

Modify or add entities when you want YATSEE to track new channels, regions, or content categories.

---

### Summary of What *Must* Be Edited

Most users only need to touch:

1. **llm_provider_url**  
   Point this at wherever your models actually run.

2. **default models**  
   If you prefer different summarization or embedding models.

3. **entities**  
   Add or remove these depending on what you're ingesting.

Everything else can be left at its defaults unless you're doing advanced tuning or debugging.


## Localized Entity Config (config.toml)

This section of the configuration controls how a specific entity (in this case, a City Council) is defined, what sources it pulls from, and how names, titles, divisions, and replacement rules are handled. These settings are meant to be edited when creating or customizing a new entity.

---

### 1. `[settings]`

The `[settings]` block defines the high-level identity and behavior of the entity.

#### **entity_type**
Describes what kind of organization this is.  
Example: `"city_council"`, `"school_board"`, `"committee"`.

#### **entity_level**
Specifies the jurisdiction level  
Examples: `"city"`, `"county"`, `"state"`.

#### **location**
Human-readable description of where the entity is located. This is used for metadata and disambiguation.

#### **data_path**
Directory path where all processed output, transcripts, JSON metadata, and summaries will be stored for this entity(Derived on setup leave it be).

#### **js_runtime**
JavaScript runtime used by `yt-dlp` during YouTube challenge negotiation.  
Common values: `"node"` or `"deno"`.

#### **summarization_model**
Model override used for summarization tasks. Must match a model defined in the global `[models]` section.

#### **transcription_model**
Whisper or faster-whisper model override used for audio transcription.

#### **notes**
Free-form description of the entity. Use this for URLs, contact info, or other context.

---

### 2. `[sources.youtube]`

Controls how YouTube ingestion works for this entity.

#### **youtube_path**
The YouTube channel or user path to monitor for livestreams and videos.

#### **enabled**
Set to `true` to activate YouTube ingestion for this entity.

---

### 3. `[divisions]`

Defines the political or administrative subdivisions represented within the entity.  
This is intentionally generic so you can use `"wards"`, `"districts"`, `"precincts"`, `"parishes"`, etc.

#### **type**
Name of the division type.

#### **names**
List of division names exactly as you want them to appear in downstream processing.

---

### 4. `[titles]`

Canonical title groups used to classify people based on what appears in transcripts.  
These lists should contain **single pieces of a title**, not full names.

Examples:
- `"Mayor"`, not `"Mayor Sam Jones"`.
- `"Chief"`, `"Fire Chief"`, etc.

These help with hotword detection during transcription and improve role inference.

Title groups included:
- **mayor**
- **city_manager**
- **city_clerk**
- **alderperson**
- **directors**
- **staff**
- **third_parties**

Each list can be expanded or modified depending on local terminology.

---

### 5. `[people.<role>]`

Defines individuals grouped by their role.  
Each entry lists all the name fragments or variants that may appear in speech.

#### Format:
```
[people.role]
  Identifier = ["Name part 1", "Name part 2", ...]
```

Guidelines:
- Use underscores in identifiers (e.g., `Sam_Jones`).
- Include first names, last names, nicknames, and common mis-hearings.
- Avoid full phrases like `"Mayor Sam Jones"` to prevent bloated hotword lists.

Roles included:
- **mayor**
- **city_manager**
- **city_clerk**
- **alderperson**
- **staff**
- **directors**

---

### 6. `[replacements]`

A dictionary of known transcription errors and their correct forms.

#### How it works:
- Keys: common Automated Speech Recognition(ASR) mis-hearings or bad spellings found in transcripts.
- Values: intended spelling or entity name.
- Applied after transcription to standardize text during the normalization process.

Examples:
- `"Free Fork"` → `"Freeport"`
- `"Fair Gram"` → `"Fehr Graham"`
- `"Schadl"` → `"Shadle"`

Add as many corrections as needed as you observe new error patterns over time.

Note: Using a higher quality model for ASR can reduce errors significantly but none of them are perfect.

---

### Summary of What You Should Edit

When creating or maintaining an entity:

1. **[settings]**  
   Update location, data path, models, and notes.

2. **[sources.youtube]**  
   Set the correct channel path and enable or disable ingestion.

3. **[divisions]**  
   Match the jurisdiction's actual structure.

4. **[titles]** and **[people]**  
   Expand or adjust based on who appears in your meetings.

5. **[replacements]**  
   Add corrections as new transcription errors show up.
