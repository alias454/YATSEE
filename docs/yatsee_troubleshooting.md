# YATSEE: Troubleshooting

YATSEE is local-first, so performance and failure modes depend heavily on the local machine, installed tools, selected provider, provider security policy, and model configuration.

Some failures come from the core pipeline itself. Others come from the chosen transcription backend, LLM provider, local runtime, hosted API, CLI integration, or provider hardening settings.

## Common Issues

### LLM provider is unavailable
If `yatsee intel run` fails with a connection error, timeout, or empty response:

- confirm the configured `llm_provider` is correct
- confirm `llm_provider_url` points to the right target
- confirm the requested model is available for that provider
- confirm the provider is reachable from the machine running YATSEE

Examples:
- Ollama not running on the expected port
- llama.cpp server not running or not exposing the expected endpoint
- `codex_cli` not installed or not on `PATH`
- remote API base URL is wrong

### Hosted API authentication failure
If OpenAI or Anthropic requests fail immediately:

- confirm `llm_api_key` is set when using hosted providers
- confirm the key is valid and active
- confirm the selected provider matches the key you supplied
- check whether the API endpoint requires additional headers or account access

Typical symptoms:
- HTTP 401 Unauthorized
- HTTP 403 Forbidden
- model access denied
- request rejected before generation begins

### Provider target blocked by hardening settings
If a provider target looks valid but YATSEE rejects it before execution:

- confirm whether the target is local or remote
- confirm whether the provider is a hosted API or a local runtime
- review the following settings in `yatsee.toml`:
  - `llm_allow_remote`
  - `llm_allow_insecure_http`
  - `llm_allow_custom_executable`

Typical cases:
- remote Ollama or llama.cpp target blocked because `llm_allow_remote = false`
- hosted provider blocked because the target uses `http://` and `llm_allow_insecure_http = false`
- custom `codex_cli` executable blocked because `llm_allow_custom_executable = false`

### Local model runtime is unavailable
If summarization fails while using a local provider:

- confirm the local runtime is running
- confirm the requested model is installed locally
- confirm the configured provider URL is correct
- confirm the machine hosting the runtime is reachable if using a remote local server on the LAN
- confirm the remote target is actually allowed by your hardening settings

Typical examples:
- Ollama not started
- model not pulled yet
- wrong host or port
- firewall or network issue between machines
- remote target blocked by default security settings

### `codex_cli` execution failure
If `codex_cli` is selected and summarization fails:

- confirm the `codex` executable is installed
- confirm it is available on `PATH`, or use an explicit command target if custom executables are enabled
- confirm the CLI is authenticated
- confirm the CLI supports the flags and invocation style expected by the current provider adapter
- confirm `llm_allow_custom_executable` is enabled if you are not using the default `codex` command

Typical symptoms:
- executable not found
- subprocess exit code not equal to zero
- empty output returned
- auth prompt or interactive behavior where non-interactive output was expected
- custom executable rejected before subprocess execution

### GPU memory exhaustion
If transcription or local summarization fails with out-of-memory errors:

- reduce the transcription model size
- reduce summarization chunk size or context size
- avoid running multiple GPU-heavy stages at the same time
- retry on CPU if needed
- check whether another local process is already using the GPU

Typical symptoms:
- CUDA out of memory
- Metal or MPS allocation failure
- provider crash or sudden termination during inference

### Unexpectedly slow transcription
If transcription is much slower than expected:

- confirm the intended device is actually being used
- check whether a previous process is still using the GPU
- verify that the environment has the expected dependencies installed
- confirm chunking is configured sensibly for the hardware

### Unexpectedly slow summarization
If the intelligence stage is much slower than expected:

- confirm which provider is actually being used
- confirm the selected model is appropriate for the hardware
- confirm context and chunk sizes are not too large for the runtime
- verify that the local provider is not silently falling back to CPU
- compare performance across providers if both Ollama and llama.cpp are available

### `ffmpeg` is missing or not on `PATH`
If audio formatting fails with command-not-found errors:

- confirm `ffmpeg` is installed
- confirm it is available on the system `PATH`
- restart the shell after installation if necessary

### Transcript quality is poor
If downstream summaries are weak because transcript quality is poor:

- confirm the transcription model is appropriate for the audio quality
- confirm the language setting is correct
- review whether chunking introduced avoidable fragmentation
- add or improve `[replacements]` entries for recurring ASR failures
- confirm entity-specific names, titles, and people lists are current

### Pricing output is missing
If pricing does not appear when expected:

- confirm `show_pricing = true` or `--show-pricing` was set
- confirm the selected `pricing_provider` and `pricing_model` exist in the pricing table
- confirm you are not expecting local providers to have native API pricing
- remember that pricing is reference pricing, not actual local compute cost

Typical cases:
- pricing disabled in config
- model missing from the pricing table
- local provider selected without a pricing reference override
- pricing available in theory but no matching table entry exists

### Pricing output looks inaccurate
If the estimated cost feels wrong:

- confirm the reference pricing model is the one you intended
- confirm the pricing table values are current
- remember that token counts may still be heuristic rather than exact
- confirm the estimated cost is being compared against the right hosted provider

This estimate is best treated as directional unless exact provider usage metadata is available.

---

## Debugging Checklist

Before reporting an issue, confirm:

- the virtual environment is active
- the expected Python dependencies are installed
- `ffmpeg` is available on `PATH`
- the configured provider is correct
- the provider target is reachable
- the requested model exists for that provider
- API credentials are set when using hosted providers
- the selected device and model size make sense for the machine
- the relevant config values in `yatsee.toml` and entity `config.toml` match what you think they are

For intelligence-stage issues, also confirm:

- `llm_provider`
- `llm_provider_url`
- `llm_api_key` if applicable
- `llm_allow_remote`
- `llm_allow_insecure_http`
- `llm_allow_custom_executable`
- `show_pricing`
- `pricing_provider`
- `pricing_model`

---

## Useful Isolation Steps

When something fails, narrow the blast radius first.

### Test one stage at a time
Instead of rerunning the entire pipeline, test only the failing stage.

Examples:
```bash
yatsee audio format --entity example_entity
yatsee audio transcribe --entity example_entity
yatsee transcript normalize --entity example_entity
yatsee intel run -e example_entity
```

### Test one transcript instead of a directory
Point the stage at a single known file so you can separate content issues from batch-processing issues.

### Print prompts without running inference
If the issue may be prompt-related:

```bash
yatsee intel run -e example_entity --print-prompts
```

### Write chunk outputs for inspection
If final summaries are weak or inconsistent:

```bash
yatsee intel run -e example_entity --enable-chunk-writer
```

This helps determine whether the issue begins at:
- chunking
- prompt routing
- intermediate summarization
- final synthesis

### Override provider settings for one run
If you need to compare backends without editing most of config:

```bash
yatsee intel run -e example_entity --llm-provider ollama --llm-provider-url http://localhost:11434
yatsee intel run -e example_entity --llm-provider llamacpp --llm-provider-url http://localhost:8080
```

Keep in mind that provider selection can be overridden from the CLI, but the provider hardening settings remain config-driven.

### Verify whether the failure is policy or transport
If the run fails before contacting the provider:

- check whether the target is being blocked by hardening settings
- compare the target type against the provider:
  - local HTTP provider
  - hosted provider
  - CLI-backed provider
- verify whether the failure is a security-policy rejection or a real transport/runtime failure

---

## Reporting Issues

When reporting a failure, include:

- operating system
- CPU, GPU, and RAM details
- the exact command that was run
- the active provider and model
- the full terminal error output
- whether the issue is reproducible
- whether the issue affects one file or all files
- whether the problem is specific to one provider or backend

For provider-hardening issues, include:

- `llm_provider`
- `llm_provider_url`
- `llm_allow_remote`
- `llm_allow_insecure_http`
- `llm_allow_custom_executable`

For pricing issues, include:

- `show_pricing`
- `pricing_provider`
- `pricing_model`
- the reported token counts
- the reported estimated cost