# YATSEE Audio Transcription Pipeline

**YATSEE** -- Yet Another Tool for Speech Extraction & Enrichment

YATSEE is a local-first, end-to-end data pipeline designed to systematically refine raw meeting audio into clean, searchable, and auditable intelligence. It automates the tedious work of downloading, transcribing, and normalizing unstructured conversations.

This is a local-first, privacy-respecting toolkit for anyone who wants to turn public noise into actionable intelligence.

## Why This Exists

Public records are often public in name only. Civic business is frequently buried in four-hour livestreams and jargon-filled transcripts that are technically accessible but functionally opaque. The barrier to entry for an interested citizen is hours of time and dealing with complex jargon.

YATSEE solves that by using a carefully tuned local LLM to transform that wall of text into a high-signal summary. YATSEE can be set to extract specific votes, contracts, and policy debates that allow you to find what you are interested in fast. It's a tool for creating clarity and accountability that modern civic discourse requires.

## Demo

![Demo video](./docs/assets/yatsee_demo.gif)

---

## Documentation
 - [YATSEE Pipeline Overview](./docs/yatsee_overview.md)
 - [YATSEE Configuration Guide](./docs/yatsee_config_guide.md)
 - [YATSEE User Guide](./docs/yatsee_user_guide.md)
 - [YATSEE Prompt Orchestration](./docs/yatsee_prompt_orchestration.md)

All modules are fully documented using standard Python docstrings.

To browse module documentation use `pydoc` locally:

```bash
pydoc ./yatsee_summarize_transcripts.py
```

---

## 🚀 Quick Start (Plug-and-Play)

Follow these steps to get YATSEE running.

---

### 1. **Clone the Repository**
```bash
git clone https://github.com/alias454/yatsee.git
cd yatsee
```

---

### 2. **Edit Initial Configuration**
```bash
# Copy the template to create your local config file
cp yatsee.conf yatsee.toml
```
- Open `yatsee.toml` in any text editor.
- Add at least **one entity** with the required fields:
  - `entity` (unique identifier)

---

### 3. **Run the Setup Script**
```bash
chmod +x setup.sh
./setup.sh
```
- Installs Python dependencies
- Downloads NLP models (`spaCy`, etc.)
- Checks for GPU (CUDA/MPS) and warns if only CPU is available

---

### 4. **Activate the Python Environment**
```bash
source .venv/bin/activate
```
> Python ≥3.10 recommended. CPU works, but GPU/MPS accelerates transcription.

---

### 5. **Run the Config Builder**
```bash
python yatsee_build_config.py --create
```
- Uses the entity info in `yatsee.toml` to:
  - Create the main directory(default ./data) for the pipeline
  - Initialize per-entity pipeline configs
  - `sources.youtube.youtube_path` (YouTube channel/playlist)
  - Any optional data structures like titles, people, replacements etc.
- This is the **minimum viable entity** needed for the downloader.
- **Important:** Run this after `setup.sh` and after adding at least one entity.

---

### 6. **Run the Processing Pipeline**
```bash
see Script Summary below
```
- Processes audio/video in `downloads/`
- Converts to `.flac`/`.wav` in `audio/`
- Generates transcripts, normalizes text, and produces summaries
- All scripts are modular: you can run them individually or as a pipeline

---

### 7. **Launch the Demo Search UI**
```bash
streamlit run yatsee_search_demo.py -- -e entity_name_configured
```
- Provides semantic and structured search over transcripts and summaries

---

### ⚠️ Notes for New Users
- `entity` is a unique key identifier for all scripts. Keep it consistent.
- Each pipeline stage ensures directories exist for output; do **not** manually create them.
- Optional: You can edit additional pipeline settings (like per-entity hotwords or divisions) in the generated config.

---

## Requirements

This pipeline was developed and tested on the following setup:
  - **CPU:** Intel Core i7-10750H (6 cores / 12 threads, up to 5.0 GHz)
  - **RAM:** 32 GB DDR4
  - **GPU:** NVIDIA GeForce RTX 2060 (6 GB VRAM, CUDA 12.8)
  - **Storage:** NVMe SSD
  - **OS:** Fedora Linux
  - **Shell:** Bash
  - **Python:** 3.10 or newer

Additional testing was performed on Apple Silicon (macOS):
  - **Model:** Mac Mini (M4 Base)
  - **CPU:** Apple M4 (10 cores / 4 performance cores, up to 120GB/s memory bandwidth)
  - **RAM:** 16 GB
  - **Storage:** NVMe SSD
  - **OS:** macOS Sonoma / Sequoia
  - **Shell:** ZSH
  - **Python:** 3.9 or newer

GPU acceleration was enabled for Whisper / faster-whisper using CUDA 12.8 and NVIDIA driver 570.144 on Linux. However, faster whisper has limited/no support for mps.

Note: Audio transcription was much slower on the MAC than on Linux. it's doable but it's much slower.

Note: The pipeline works on CPU-only systems without a GPU. However, transcription (especially with Whisper or faster-whisper) will be much slower compared to systems with CUDA-enabled GPU acceleration or MPS.

> ⚠️ Not tested on Windows. Use at your own risk on `Windows` platforms.

---
Manual Installation (If not using setup.sh)

If you cannot use the setup script, ensure you have `ffmpeg` and `yt-dlp` installed via your package manager, then install the Python requirements:

### Required CLI Tools

- `yt-dlp` – Download livestream audio from YouTube
- `ffmpeg` – Convert audio to `.flac` or `.wav` format

### Required Python Packages

- `toml`             Needed for reading the toml config
- `requests`         Needed for interacting with ollama API if installed
- `torch`            Required for Whisper and model inference (with or without CUDA)
- `pyyaml`           YAML output support (for summaries)
- `whisper`          Audio transcription (standard)
- `spacy`            Sentence segmentation + text cleanup
  - Model: en_core_web_sm (or larger)

### Optional:

- `faster-whisper`   Audio transcription (optional)

### Optional: Summarization with Local LLM

  - ollama  Run local LLMs for summarization

---

# Setup

### Install ffmpeg for your platform

macOS (Homebrew):
```bash
brew install ffmpeg
```

Fedora:
```bash
sudo dnf install ffmpeg
```

Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Install Python Packages

You can use pip to install the core requirements:

Install:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Whisper/faster-whisper

On first run, it will download a model (e.g., base, medium). Ensure you have enough RAM.

### Install ollama(Optional)

Used for generating markdown or YAML summaries from transcripts.

install:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

See https://ollama.com for supported models and system requirements.
