# YATSEE: Configuration & Orchestration Guide

YATSEE uses a layered configuration model to control how the system resolves defaults, identifies entities, selects models, chooses LLM providers, applies provider security policy, and processes source material across the CLI.

Most of the configuration can stay at sensible defaults. The parts that matter most are the sections that define names, places, roles, sources, provider settings, security settings, and recurring transcript issues specific to the entity you are processing.

If these fields stay generic, the system stays generic. When you fill them in with real local data, YATSEE stops guessing and starts identifying people, places, and recurring topics more accurately across transcripts and downstream artifacts.

This document explains the global configuration model, entity-local overrides, and the parts of the config you are most likely to customize.

## Configuration Model

YATSEE follows this general pattern:

1. load global `yatsee.toml`
2. load entity-local `config.toml`
3. merge entity-local settings over global defaults
4. cache the merged result in memory
5. resolve stage behavior from the merged configuration

This model supports the primary command families:

- `yatsee config ...`
- `yatsee source fetch ...`
- `yatsee audio format ...`
- `yatsee audio transcribe ...`
- `yatsee transcript slice ...`
- `yatsee transcript normalize ...`
- `yatsee intel run ...`

---

## Global System Configuration (`yatsee.toml`)

The global config defines system-wide defaults, runtime settings, provider selection, provider security controls, pricing references, model selection, and entity registration.

### 1. `[system]`

The `[system]` block controls global behavior for logging, file paths, LLM provider access, provider security policy, and reference pricing behavior.

#### `root_data_dir`
- Base path where generated data, cached artifacts, and model outputs are stored.
- Change this if you want YATSEE to write pipeline data somewhere else, such as a mounted disk or dedicated volume.

#### `log_level`
- Controls logging verbosity.
- Typical values: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
- Use `DEBUG` only when troubleshooting.

#### `log_format`
- Standard Python logging format string.
- Most users will never need to change this unless integrating with another logging convention.

#### `llm_provider`
- Name of the LLM backend used for intelligence-stage generation.
- Typical values include:
  - `"ollama"`
  - `"llamacpp"`
  - `"openai"`
  - `"anthropic"`
  - `"codex_cli"`
- This selects how YATSEE talks to the backend, not just which model name it passes.

#### `llm_provider_url`
- Generic target for the configured provider.
- For HTTP providers, this is usually a base URL.
- For CLI-backed providers such as `codex_cli`, this can be the executable or command target.
- Examples:
  - `"http://localhost:11434"`
  - `"http://192.168.2.58:11434"`
  - `"http://localhost:8080"`
  - `"codex"`

#### `llm_api_key`
- Optional API key for hosted providers.
- Usually empty for local providers such as Ollama or llama.cpp.
- Typically required for remote providers such as OpenAI or Anthropic.

#### `llm_allow_remote`
- Controls whether local HTTP providers such as Ollama or llama.cpp are allowed to target non-local hosts.
- Default behavior is effectively `false` when the setting is omitted.
- This is intentionally conservative so local-first deployments do not silently drift into remote execution.

#### `llm_allow_insecure_http`
- Controls whether hosted providers such as OpenAI or Anthropic may use plain HTTP instead of HTTPS.
- Default behavior is effectively `false` when the setting is omitted.
- In normal use, this should remain disabled.

#### `llm_allow_custom_executable`
- Controls whether CLI-backed providers such as `codex_cli` may use a custom executable target instead of the default command.
- Default behavior is effectively `false` when the setting is omitted.
- This is intentionally guarded because executable selection is a security boundary.

#### `show_pricing`
- Enables reference pricing output during intelligence runs.
- This is useful when you run local providers but want to estimate what the same job would have cost on a hosted API.

#### `pricing_provider`
- Provider used for reference pricing estimation.
- This can be different from the actual runtime provider.
- Example:
  - actual provider: `"ollama"`
  - pricing provider: `"openai"`

#### `pricing_model`
- Model used for reference pricing estimation.
- This is paired with `pricing_provider`.
- Example:
  - `"gpt-5.4"`
  - `"claude-opus-4.1"`

### Example `[system]`

```toml
[system]
root_data_dir = "./data"
log_level = "INFO"
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"

llm_provider = "ollama"
llm_provider_url = "http://192.168.2.58:11434"
llm_api_key = ""

# Security hardening for provider targets
llm_allow_remote = false
llm_allow_insecure_http = false
llm_allow_custom_executable = false

# Optional reference pricing for estimated API-equivalent cost reporting, even when running locally.
show_pricing = true
pricing_provider = "openai"
pricing_model = "gpt-5.4"
```

### How provider, security, and pricing settings work together

YATSEE separates:

- the provider actually used to generate output
- the security policy that controls which targets are allowed
- the provider/model used for reference pricing

Example:

- `llm_provider = "ollama"`
- `llm_provider_url = "http://192.168.2.58:11434"`
- `llm_allow_remote = false`
- `pricing_provider = "openai"`
- `pricing_model = "gpt-5.4"`

In that configuration:

- YATSEE intends to run summarization through Ollama
- the configured Ollama target must still satisfy the provider security policy
- if the Ollama target is non-local and `llm_allow_remote = false`, the run is rejected
- if the run succeeds, YATSEE reports an estimated OpenAI-equivalent cost using the configured pricing reference

If `show_pricing` is disabled, no pricing estimate is reported.

If `show_pricing` is enabled but the selected pricing provider/model does not have a pricing entry, pricing remains unavailable rather than inventing a fake value.

### Security defaults by omission

These hardening settings default to safe behavior when omitted from config:

- `llm_allow_remote` → effectively `false`
- `llm_allow_insecure_http` → effectively `false`
- `llm_allow_custom_executable` → effectively `false`

That means secure behavior is the default even if these keys are not present in `yatsee.toml`.

---

### 2. Default Models

YATSEE uses multiple specialized subsystems, including transcription, summarization, sentence processing, and embeddings. Default models are defined globally and can be overridden per entity.

#### `default_js_runtime`
- JavaScript runtime used for YouTube challenge negotiation with `yt-dlp`.
- Typical values: `"node"` or `"deno"`.

#### `default_summarization_model`
- Default model used for long-form summarization and related intelligence stages.
- This is the model name passed to the configured LLM provider.

#### `default_transcription_model`
- Whisper or faster-whisper model name used for audio transcription by default.

#### `default_sentence_model`
- spaCy model used for sentence-level downstream operations such as segmentation or normalization-related processing.

#### `default_embedding_model`
- Embedding model used when vectorization or semantic retrieval workflows are enabled.

---

### 3. `[models]`

The `[models]` block defines per-model runtime configuration.

Each key corresponds to a model identifier used by your selected runtime or pipeline configuration.

A typical model entry may include:

#### `append_dir`
- Subdirectory created under the entity data path for outputs associated with that model.

#### `max_tokens`
- Target chunk size for summarization or related LLM operations.
- In practice, this is usually set below the total model context window.

#### `num_ctx`
- Maximum supported context window for the model.

You only need to edit this section when:

- adding new models
- changing summarization chunking behavior
- changing output directory conventions
- tuning context-related runtime behavior

---

### 4. `[entities.<name>]`

Entities tell YATSEE what to process, how to organize it, and which sources or behaviors to enable.

Each entry under `[entities.<name>]` typically contains fields like:

#### `display_name`
- Human-readable name shown in logs or interfaces.

#### `base`
- Logical classification path for data organization.

#### `entity`
- Canonical identifier used for paths and internal resolution.

#### `inputs`
- Enabled input processors or source types for that entity.

Modify or add entities when you want YATSEE to process new municipalities, organizations, channels, or source groups.

---

## Global Configuration: What Usually Needs Editing

Most users only need to change:

1. **`llm_provider` and `llm_provider_url`**  
   Point YATSEE at the backend you actually use.

2. **`llm_api_key`**  
   Set this only when using hosted providers that require authentication.

3. **hardening settings**  
   Leave these at their safe defaults unless you intentionally need remote local-runtime targets, insecure hosted HTTP, or a custom CLI executable.

4. **reference pricing options**  
   Enable and tune these only if you want “what this would have cost” reporting.

5. **default models**  
   Change these if you prefer different transcription, summarization, or embedding models.

6. **entities**  
   Add or remove entity definitions depending on what you want YATSEE to process.

Everything else can usually remain at defaults unless you are tuning behavior or troubleshooting.

---

## Localized Entity Configuration (`config.toml`)

The entity-local `config.toml` controls how one specific entity is defined, what sources it uses, and how titles, people, divisions, and transcript cleanup rules are handled.

This is the configuration layer you are most likely to edit when creating or maintaining a new entity.

---

### 1. `[settings]`

The `[settings]` block defines the high-level identity and behavior of the entity.

#### `entity_type`
- Describes the kind of organization.
- Examples: `"city_council"`, `"school_board"`, `"committee"`.

#### `entity_level`
- Describes the jurisdiction or organizational level.
- Examples: `"city"`, `"county"`, `"state"`.

#### `location`
- Human-readable description of where the entity is located.
- Used for metadata and disambiguation.

#### `data_path`
- Directory path where processed output, transcripts, metadata, and summaries are stored for this entity.
- In most cases this is derived during setup and should be left alone.

#### `js_runtime`
- JavaScript runtime used by `yt-dlp` during source fetching workflows.
- Common values: `"node"` or `"deno"`.

#### `summarization_model`
- Per-entity override for summarization or intelligence stages.
- Must match a valid configured model.

#### `transcription_model`
- Per-entity override for transcription.

#### `notes`
- Free-form description of the entity.
- Useful for URLs, contact info, source context, or operational notes.

---

### 2. `[sources.youtube]`

Controls YouTube ingestion for the entity.

#### `youtube_path`
- Channel, user, or playlist path to monitor or fetch from.

#### `enabled`
- Set to `true` to enable YouTube ingestion for the entity.

If additional source types are added in the future, they should follow a similar source-specific configuration pattern.

---

### 3. `[divisions]`

Defines the political or administrative subdivisions represented within the entity.

This is intentionally generic so it can support terms such as:

- wards
- districts
- precincts
- parishes

#### `type`
- Name of the division type.

#### `names`
- Ordered list of division names exactly as you want them to appear in downstream processing.

---

### 4. `[titles]`

Canonical title groups used to classify people based on transcript content.

These lists should contain title fragments or role terms, not full names.

Examples:
- `"Mayor"`, not `"Mayor Sam Jones"`
- `"Chief"`
- `"Fire Chief"`

These help with:
- hotword selection
- role inference
- transcript interpretation
- downstream summarization context

Common title groups may include:

- `mayor`
- `city_manager`
- `city_clerk`
- `alderperson`
- `directors`
- `staff`
- `third_parties`

Adjust these to match local terminology.

---

### 5. `[people.<role>]`

Defines individuals grouped by role.

Each entry should list the name fragments or variants that may appear in speech or transcription.

#### Format

```toml
[people.role]
Sam_Jones = ["Sam", "Jones", "Samuel"]
```

Guidelines:

- use underscores in identifiers, for example `Sam_Jones`
- include first names, last names, nicknames, and common transcription variants
- avoid full phrases like `"Mayor Sam Jones"` unless there is a specific reason to include them

Typical role groups may include:

- `mayor`
- `city_manager`
- `city_clerk`
- `alderperson`
- `staff`
- `directors`

---

### 6. `[replacements]`

A dictionary of known transcription errors and their corrected forms.

#### How it works

- keys are common ASR mis-hearings, bad spellings, or recurring mistakes
- values are the intended corrected forms
- replacements are applied during normalization-related cleanup

Examples:

- `"Free Fork"` → `"Freeport"`
- `"Fair Gram"` → `"Fehr Graham"`
- `"Schadl"` → `"Shadle"`

Add corrections over time as you observe recurring error patterns.

Better ASR models can reduce these issues, but they do not eliminate them.

---

## CLI Overrides for `yatsee intel run`

The intelligence stage supports direct CLI overrides for provider and pricing behavior.

Common flags include:

- `--llm-provider`
- `--llm-provider-url`
- `--llm-api-key`
- `--show-pricing`
- `--no-show-pricing`
- `--pricing-provider`
- `--pricing-model`

These override the matching `[system]` settings for a single run without changing the underlying config file.

The provider hardening settings are intentionally config-driven rather than casually exposed as one-off CLI switches. This keeps security-sensitive behavior slightly harder to weaken by accident.

These overrides are useful when you want to:

- test a different backend
- temporarily point at another runtime
- compare local execution against a hosted pricing reference
- run one-off experiments without editing other parts of `yatsee.toml`

---

## Entity Configuration: What Usually Needs Editing

When creating or maintaining an entity, these sections are the ones that typically matter most:

1. **`[settings]`**  
   Update location, models, notes, and entity behavior.

2. **`[sources.youtube]`**  
   Set the correct channel or playlist path and enable or disable ingestion.

3. **`[divisions]`**  
   Match the actual local structure.

4. **`[titles]` and `[people]`**  
   Expand and refine these based on who appears regularly in the source material.

5. **`[replacements]`**  
   Add new cleanup rules as you discover transcription mistakes.

---

## Operational Notes

A few practical rules are worth keeping in mind:

- keep entity data specific and concrete
- avoid bloated title or people lists
- prefer stable role labels and identifiers
- use replacements for recurring transcription failures instead of trying to solve everything with model changes alone
- treat configuration as part of output quality, not just setup overhead
- keep provider targets local unless you intentionally want remote execution
- leave hosted providers on HTTPS unless you have a very specific reason not to
- treat custom CLI executable targets as an intentional opt-in, not a casual default

Good configuration materially improves:

- transcript quality
- role inference
- summary accuracy
- entity disambiguation
- search and retrieval usefulness

---

## Final Guidance

Most users do not need to tune every field.

In practice, the highest-value work is:

- define the right entity
- point it at the right sources
- set the right providers and models
- keep the provider hardening defaults in place unless you have a specific reason to change them
- add real local names, titles, and replacements
- optionally enable reference pricing if you want savings visibility for local runs

Once those are in place, YATSEE has much better context for turning audio-first source material into reliable downstream artifacts.