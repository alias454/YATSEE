# YATSEE Audio Transcription Pipeline

**YATSEE** -- Yet Another Tool for Speech Extraction & Enrichment

YATSEE is a local-first, end-to-end data pipeline designed to systematically refine raw meeting audio into a clean, searchable, and auditable intelligence layer. It automates the tedious work of downloading, transcribing, and normalizing unstructured conversations.

This is a local-first, privacy-respecting toolkit for anyone who wants to turn public noise into actionable intelligence.

## Why This Exists

Public records are often public in name only. Civic business is frequently buried in four-hour livestreams and jargon-filled transcripts that are technically accessible but functionally opaque. The barrier to entry for an interested citizen is hours of time and complex jargon.

YATSEE solves that by using a carefully tuned local LLM to transform that wall of text into a high-signal summary—extracting the specific votes, contracts, and policy debates that matter. It's a tool for creating the clarity and accountability that modern civic discourse requires, with or without the government's help.

---

[▶ Demo video](yatsee_demo.mp4)

---

## Documentation

All modules are fully documented using standard Python docstrings.

To browse documentation locally:

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

## 📌 YATSEE Audio Pipeline Overview

A modular pipeline for extracting, enriching, and summarizing civic meeting audio data.

### Pipeline Flow Overview
1. `downloads/` → raw video/audio  
2. `audio/` → converted `.wav` or `.flac`  
3. `transcripts_<model>/` → `.vtt` + flat `.txt`  
4. `normalized/` → cleaned, structured `.txt`  
5. `summary/` → `.md` or `.yaml` summaries

YATSEE is designed as a collection of independent tools. While they work best as a unified pipeline, each script can be run standalone as long as the input data matches the Interface Contract.

### 1. Automated Download
- **Script:** `yatsee_download_audio.py`
- **Input:** YouTube URL (bestaudio)
- **Output:** `.mp4` or `.webm` to `downloads/`
- **Tool:** `yt-dlp`
  - **Purpose:** Archive livestream audio for local processing 

### 2. Convert Audio
- **Script:** `yatsee_format_audio.py`
- **Input:** `.mp4` or `.webm` from `downloads/`
- **Output:** `.wav` or `.flac` to `audio/`
- **Tool:** `ffmpeg`
- **Format Settings:**
  - WAV: `-ar 16000 -ac 1 -sample_fmt s16 -c:a pcm_s16le`
  - FLAC `-ar 16000 -ac 1 -sample_fmt s16 -c:a flac`
- **Notes:**
  - Supports **chunked output** for long audio
  - Optional overlap between chunks to prevent cutting phrases

### 3. Transcribe Audio
- **Script:** `yatsee_transcribe_audio.py`
- **Input:** `.flac` from `audio/`
- **Output:** `.vtt` to `transcripts_<model>/`
- **Tool:** `whisper` or `faster-whisper`
- **Notes:**
  - Supports **stitching chunked audio** back into a single transcript
  - Accepts model selection: `small`, `medium`, `large`, etc.
  - Faster-whisper improves performance if installed

### 4. Clean and Normalize
#### a. Strip Timestamps
- **Script:** `yatsee_slice_vtt.py`
- **Input:** `.vtt` from `transcripts_<model>/`
- **Output:** `.txt` (timestamp-free) to same folder

#### b. JSONL Segmentation and Text Compression
- **Script:** `yatsee_slice_vtt.py`
- **Input:** `.vtt` from `transcripts_<model>/`
- **Output:** `.jsonl` to same folder
  - **Purpose:** JSONL segments (sliced transcript for embeddings/search).

#### c. Text Normalization
- **Script:** `yatsee_normalize_structure.py`
- **Input:** `.punct.txt` from `normalized/`
- **Output:** `.norm.txt` to `normalized/`
- **Tool:** `spaCy`
  - **Purpose:** Segment text into readable sentences and normalize punctuation/spacing.

### 5. Summarize Transcripts using AI (Local LLM)
- **Script:** `yatsee_summarize_transcripts.py`
- **Input:** `.txt` from `normalized/`
- **Output:** `.md` or `.yaml` to `summary/`
- **Tool:** `ollama`
- **Notes:**
  - Supports short and long-form summaries
  - Optional YAML output (e.g., vote logs, action items, discussion summaries)

All scripts are modular and can be run independently or as part of an automated workflow.

### Index and Search _(In Development)_
#### Index Data
- **Script:** `yatsee_index_data.py`
- **Input:** `.txt` from `normalized/`
- **Input:** `.md` from `summary/`
- **Output:** `embeddings` to `yatsee_db/`
- **Tool:** `ChromaDB`
- **Notes:**
  - Generate embeddings from raw transcripts and summaries into a searchable civic intelligence database.
  - **Vector Search (Semantic):** Uses ChromaDB with the `BAAI/bge-small-en-v1.5` model to allow for fuzzy, concept-based queries (e.g., "Find discussions about road repairs").

#### Search _(In Development)_
- **Script:** `yatsee_index_data.py`
- **Input:** `.txt` from `normalized/`
- **Input:** `.md` from `summary/`
- **Input:** `embeddings` from `yatsee_db/`
- **Tool:** `Streamlit and ChromaDB`
- **Notes:**
  - **UI:** A simple web interface built with Streamlit to provide an overview of the generated transcripts and summaries.
  - Planned **Graph Search (Relational):** Extract structured data (Votes, Contracts, Appointments) into a knowledge graph to trace connections between people and money.

---

## 📁 Filesystem Layout
```
data/
└── <entity_handle>/
    ├── downloads/                ← Raw input (audio/video)
    ├── audio/                    ← Converted 16kHz mono files
    ├── transcripts_<model>/      ← VTTs + initial flat .txt files
    ├── normalized/               ← Cleaned + structured output (spaCy)
    ├── summary/                  ← Generated meeting summaries (.md/.yaml)
    ├── yatsee_db/                ← Vector database files (ChromaDB)
    ├── prompts/                  ← Optional default prompt overrides(created by user)
    └── conf.toml                 ← Localized entity config
```

---

## Config file routing/load order
```
Global TOML
    |
    +--> Entity handle
            |
            +--> Local config (hotwords, divisions, data_path)
                    |
                    +--> Pipeline stage (downloads, audio, transcripts)
```

---

## Prompt override layout example:
```
./prompts/                      # default prompts for all entities
  └── research/
      └── prompts.toml          # default prompts & routing for 'research' job type

./data/
  └── defined_entity/           # entity-specific data
      └── prompts/
          └── research/
              └── prompts.toml  # full override for defined_entity 'research' job type

./data/
  └── generic_entity/           # another entity with no override
      └── prompts/
          └── research/
              # no file, falls back to default in prompts/research/prompts.toml

**Behavior**:

  - Loader first checks `data/<entity>/prompts/<job_type>/prompts.toml`.  
  - If found → full override of defaults.  
  - If not found → fall back to `prompts/<job_type>/prompts.toml`.
```

---

## ⚙️ Requirements & Setup

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

### Install CLI tools for your platform

macOS (Homebrew):
```bash
brew install yt-dlp ffmpeg
```

Fedora:
```bash
sudo dnf install yt-dlp ffmpeg
```

Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
sudo chmod a+rx /usr/local/bin/yt-dlp
```

### Install Python Packages

You can use pip to install the core requirements:

Install:
```bash
pip install torch torchaudio tqdm
pip install --upgrade git+https://github.com/openai/whisper.git

pip install toml pyyaml spacy
python -m spacy download en_core_web_sm
```

### Install faster-whisper (Optional for faster transcription)

Install:
```bash
pip install faster-whisper  # optional, for better performance
```

On first run, it will download a model (e.g., base, medium). Ensure you have enough RAM.

### Install ollama(Optional)

Used for generating markdown or YAML summaries from transcripts.

install:
```bash
pip install requests

curl -fsSL https://ollama.com/install.sh | sh
```

See https://ollama.com for supported models and system requirements.

---

## 🚀 Running the Pipeline

Run each script in sequence or independently as needed:
All scripts accept the -e (entity) flag to route data to the correct folders defined in yatsee.toml.

### 📄 Script Summary

| Script                                                | Purpose                                       |
|-------------------------------------------------------|-----------------------------------------------|
| `python3 yatsee_download_audio.py -e <entity>`        | Download audio from YouTube URLs              |
| `python3 yatsee_format_audio.py -e <entity>`          | Convert downloaded files to `.flac` or `.wav` |
| `python3 yatsee_transcribe_audio.py -e <entity>`      | Transcribe audio files to `.vtt`              |
| `python3 yatsee_slice_vtt.py -e <entity>`             | Slice and segment `.vtt` files                |
| `python3 yatsee_normalize_structure.py -e <entity>`   | Clean and normalize text structure            |
| `python3 yatsee_summarize_transcripts.py -e <entity>` | Generate summaries from cleaned transcripts   |
| `python3 yatsee_index_data.py -e <entity>`            | Vectorize and index embeddings                |
| `streamlit run yatsee_search_demo.py -- -e <entity>`  | Search summaries and transcripts              |
