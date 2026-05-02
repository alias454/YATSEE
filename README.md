# YATSEE Audio Extraction Pipeline

## Yet Another Tool for Speech Extraction & Enrichment

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

YATSEE is a local-first, end-to-end data pipeline designed to systematically refine raw meeting audio into clean, searchable, and auditable intelligence. It automates the tedious work of downloading, transcribing, normalizing, and generating higher-level intelligence from unstructured conversations.

This is a local-first, privacy-respecting toolkit for anyone who wants to turn public noise into actionable intelligence. Local runtimes are the default and preferred path, while optional external providers are supported when needed.

## Why This Exists

Public records are often public in name only. Civic business is frequently buried in four-hour livestreams and jargon-filled transcripts that are technically accessible but functionally opaque. The barrier to entry for an interested citizen is hours of time and dealing with complex jargon.

YATSEE solves that by using a carefully tuned local-first LLM workflow to transform that wall of text into a high-signal summary. YATSEE can be set to extract specific votes, contracts, and policy debates that allow you to find what you are interested in fast. It's a tool for creating clarity and accountability that modern civic discourse requires.

## Demo

![Demo video](./docs/assets/yatsee_demo.gif)

---

## Documentation
 - [YATSEE Pipeline Overview](./docs/yatsee_overview.md)
 - [YATSEE Configuration Guide](./docs/yatsee_config_guide.md)
 - [YATSEE User Guide](./docs/yatsee_user_guide.md)
 - [YATSEE Prompt Orchestration](./docs/yatsee_prompt_orchestration.md)
 - [YATSEE Troubleshooting](./docs/yatsee_troubleshooting.md)

---

## Command families

YATSEE provides these primary command families:

- `yatsee config ...`
- `yatsee source fetch ...`
- `yatsee audio format ...`
- `yatsee audio transcribe ...`
- `yatsee transcript slice ...`
- `yatsee transcript normalize ...`
- `yatsee intel run ...`

## Installation

### System tools

Required or commonly needed tools:

- `ffmpeg`
- `yt-dlp` for source fetching workflows

### Clone the repository

```bash
git clone https://github.com/YATSEE-Labs/YATSEE.git
cd yatsee
```

### Bootstrap a local environment

Linux/macOS:

```bash
./scripts/setup.sh
```

Windows PowerShell:

```powershell
.\scripts\setup.ps1
```

These setup scripts are convenience helpers for local development.

### Install directly from `pyproject.toml`

Create and activate a virtual environment.

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the package in editable mode:

```bash
pip install --upgrade pip
pip install -e .
```

Install common optional functionality:

```bash
pip install -e .[full]
```

Or install only the extra sets you need:

```bash
pip install -e .[transcript]
pip install -e .[intelligence]
pip install -e .[llamacpp]
pip install -e .[index]
pip install -e .[ui]
```

## Requirements

### Minimum system requirements

YATSEE can run on modest local hardware, but a baseline system is still required.

Minimum practical requirements:

- **CPU:** modern 64-bit multi-core processor
- **RAM:** 16 GB recommended
- **Storage:** sufficient free disk space for source media, intermediate audio, transcripts, and derived artifacts
- **Python:** 3.11 or newer
- **OS:** Linux or macOS recommended
- **Required tools:** `ffmpeg`
- **Optional tools:** `yt-dlp` for source fetching workflows

A GPU is **not required**, but some stages, especially transcription, may be substantially slower on CPU-only systems.

### Tested systems

The pipeline was developed and tested on the following Linux system:

- **CPU:** Intel Core i7-10750H (6 cores / 12 threads, up to 5.0 GHz)
- **RAM:** 32 GB DDR4
- **GPU:** NVIDIA GeForce RTX 2060 (6 GB VRAM, CUDA 12.8)
- **Storage:** NVMe SSD
- **OS:** Fedora Linux
- **Shell:** Bash
- **Python:** 3.11 or newer

Additional testing was performed on Apple Silicon:

- **Model:** Mac Mini (M4 Base)
- **CPU:** Apple M4 (10 cores / 4 performance cores, up to 120 GB/s memory bandwidth)
- **RAM:** 16 GB
- **Storage:** NVMe SSD
- **OS:** macOS Sonoma / Sequoia
- **Shell:** ZSH
- **Python:** 3.11 or newer

### Performance notes

- GPU acceleration was used for Whisper / faster-whisper on Linux with CUDA 12.8 and NVIDIA driver 570.144.
- CPU-only operation is supported, but transcription will be substantially slower.
- Apple Silicon is viable, but transcription performance was notably slower than the tested Linux CUDA system.
- `faster-whisper` has limited or no practical MPS support in some environments, so Apple Silicon paths may fall back to CPU behavior depending on setup.

> Windows support is not a primary tested platform at this time.

## Quick start

### 1. Create a local runtime config

`yatsee.conf` is the configuration template. Copy it to `yatsee.toml` to create your local runtime config, then edit `yatsee.toml` for your environment and entities.

```bash
cp yatsee.conf yatsee.toml
```

Edit `yatsee.toml` and define at least one entity such as `example_entity`.

For the intelligence stage, set at least:

- `llm_provider`
- `llm_provider_url`

Optional provider and pricing settings include:

- `llm_api_key`
- `show_pricing`
- `pricing_provider`
- `pricing_model`

Optional provider hardening settings include:

- `llm_allow_remote`
- `llm_allow_insecure_http`
- `llm_allow_custom_executable`

If omitted, these hardening settings default to safe behavior.

### 2. Inspect the CLI

```bash
yatsee --help
yatsee config --help
```

If the console script is not yet available in your environment, use:

```bash
python -m yatsee.cli.main --help
```

### 3. Validate configuration

```bash
yatsee config entity list
yatsee config validate
yatsee config resolve --entity example_entity
```

### 4. Fetch source media

```bash
yatsee source fetch -e example_entity --make-playlist
```

### 5. Process later stages as needed

```bash
yatsee audio format --entity example_entity --dry-run
yatsee audio transcribe --entity example_entity
yatsee transcript slice --entity example_entity
yatsee transcript normalize --entity example_entity
yatsee intel run -e example_entity --print-prompts
```

### 6. Run the intelligence stage

Local-first default example:

```bash
yatsee intel run -e example_entity --model llama3:latest --llm-provider ollama --llm-provider-url http://localhost:11434
```

Alternative provider example:

```bash
yatsee intel run -e example_entity --model mistral-nemo:latest --llm-provider llamacpp --llm-provider-url http://localhost:8080
```

## Pipeline model

At a high level, YATSEE processes recordings through staged boundaries:

1. source acquisition
2. audio formatting
3. transcription
4. transcript slicing
5. transcript normalization
6. higher-level intelligence and summarization

The broader workflow can also support embeddings and semantic retrieval over the resulting artifacts.

## Configuration model

YATSEE follows a layered configuration strategy:

1. load global `yatsee.toml`
2. load entity-local `config.toml`
3. merge entity-local settings over global defaults
4. cache merged config in memory
5. resolve stage behavior from the merged result

This configuration pattern is central to the system design.

For intelligence-stage providers, YATSEE also applies a small security policy layer by default:

- remote non-local targets for local HTTP providers are blocked unless explicitly allowed
- insecure HTTP for hosted providers is blocked unless explicitly allowed
- custom CLI executable targets are blocked unless explicitly allowed

These settings are intentionally config-driven so they take a little more effort to relax.

## Search and indexing

YATSEE can be used alongside indexing and search workflows built on top of its outputs, including normalized transcripts, summaries, and other derived artifacts.

## License

YATSEE is open-source software licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

### What this means:
- **Freedom:** You are free to use, modify, and distribute this software.
- **Open Source:** If you modify YATSEE and share it, or let users interact with your modified version over a network, you must provide the corresponding source under the AGPLv3.

**Commercial Licensing:**
If you wish to use YATSEE in a proprietary product or closed-source commercial environment, please contact admin <at> alias454 <dot> com for a commercial license.