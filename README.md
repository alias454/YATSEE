# YATSEE Audio Transcription Pipeline

**YATSEE** -- Yet Another Tool for Speech Extraction & Enrichment

YATSEE is a local-first, end-to-end data pipeline designed to systematically refine raw meeting audio into a clean, searchable, and auditable intelligence layer. It automates the tedious work of downloading, transcribing, and normalizing unstructured conversations.

This is a local-first, privacy-respecting toolkit for anyone who wants to turn public noise into actionable intelligence.

## Why This Exists

Public records are often public in name only. Civic business is frequently buried in four-hour livestreams and jargon-filled transcripts that are technically accessible but functionally opaque. The barrier to entry for an interested citizen is hours of time and complex jargon.

YATSEE solves that by using a carefully tuned local LLM to transform that wall of text into a high-signal summary—extracting the specific votes, contracts, and policy debates that matter. It's a tool for creating the clarity and accountability that modern civic discourse requires, with or without the government's help.

---

## 📌 YATSEE Audio Pipeline Overview

A modular pipeline for extracting, enriching, and summarizing civic meeting audio data.

Scripts are modular, with clear input/output expectations.

### 1. Automated Download
- **Script:** `yatsee_download_audio.sh`
- **Input:** YouTube URL (bestaudio)
- **Output:** `.mp4` or `.webm` to `downloads/`
- **Tool:** `yt-dlp`
  - **Purpose:** Archive livestream audio for local processing 

### 2. Convert Audio
- **Script:** `yatsee_format_audio.sh`
- **Input:** `.mp4` or `.webm` from `downloads/`
- **Output:** `.wav` or `.flac` to `audio/`
- **Tool:** `ffmpeg`
- **Format Settings:**
  - WAV: `-ar 16000 -ac 1 -sample_fmt s16 -c:a pcm_s16le`
  - FLAC `-ar 16000 -ac 1 -sample_fmt s16 -c:a flac`

### 3. Transcribe Audio
- **Script:** `yatsee_transcribe_audio.py`
- **Input:** `.flac` from `audio/`
- **Output:** `.vtt` to `transcripts_<model>/`
- **Tool:** `whisper` or `faster-whisper`
- **Notes:**
  - Supports `faster-whisper` if installed
  - Accepts model selection: `small`, `medium`, `large`, etc.
  - Outputs to `transcripts_<model>/` by default

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

#### c. Sentence Segmentation
- **Script:** `yatsee_polish_transcript.py`
- **Input:** `.txt` from `transcripts_<model>/`
- **Output:** `.punct.txt` to `normalized/`
- **Tool:** `deep multilingual punctuation model`
  - **Purpose:** Deep learning punctuation.

#### d. Text Normalization
- **Script:** `yatsee_normalize_structure.py`
- **Input:** `.punct.txt` from `normalized/`
- **Output:** `.txt` to `normalized/`
- **Tool:** `spaCy`
  - **Purpose:** Segment text into readable sentences and normalize punctuation/spacing.

### 5. Summarize Transcripts using AI (Local LLM)
- **Script:** `yatsee_summarize_transcripts.py`
- **Input:** `.out` or `.txt` from `normalized/`
- **Output:** `.md` or `.yaml` to `summary/`
- **Tool:** `ollama`
- **Notes:**
  - Supports short and long-form summaries
  - Optional YAML output (e.g., vote logs, action items, discussion summaries)

All scripts are modular and can be run independently or as part of an automated workflow.

### Index & Search _(In Development)_
- **Goal:** Turn the generated summaries and raw transcripts into a searchable civic intelligence database.
- **Vector Search (Semantic):** Use ChromaDB with the `nomic-embed-text` model to allow for fuzzy, concept-based queries (e.g., "Find discussions about road repairs").
- **Graph Search (Relational):** Extract structured data (Votes, Contracts, Appointments) into a knowledge graph to trace connections between people and money.
- **UI:** A simple web interface built with Streamlit to provide an overview of the city's operations.

---

## 📁 Filesystem Layout

    transcripts_pipeline/
    │
    ├── downloads/                ← raw input (audio/video)
    ├── audio/                    ← post-conversion (.wav/.flac) files
    ├── transcripts_<model>/      ← VTTs + initial flat .txt files
    │   ├── meeting.vtt
    │   └── meeting.txt           ← basic timestamp removal only
    │
    ├── normalized/               ← cleaned + structured output
    │   └── meeting.txt           ← just structure normalization (spaCy)
    │
    ├── summary/                  ← generated meeting summaries (.yaml/.md) files
    │   └── summary.md

---

## ⚙️ Requirements & Setup

This pipeline was developed and tested on the following setup:
  - **CPU:** Intel Core i7-10750H (6 cores / 12 threads, up to 5.0 GHz)
  - **RAM:** 32 GB DDR4
  - **GPU:** NVIDIA GeForce RTX 2060 (6 GB VRAM, CUDA 12.8)
  - **Storage:** NVMe SSD
  - **OS:** Fedora Linux
  - **Shell:** Bash
  - **Python:** 3.8 or newer

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

> ⚠️ Not test on Windows. Use at your own risk on `Windows` platforms.

---

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

### 1. Download Audio
```bash
./yatsee_download_audio.sh https://youtube.com/some-url
```

### 2. Convert to Audio Format
```bash
./yatsee_format_audio.sh
```

### 3. Transcribe Audio
```bash
python yatsee_transcribe_audio.py --audio_input ./audio --model medium --faster
```

### 4. Clean Text

**Slice and segment vtt files for embeddings/search or Strip timestamps for txt**
```bash
python yatsee_slice_vtt.py --vtt-input transcripts_<model> --output-dir transcripts_<model> --window 30
```

**Normalize structure**
```bash
# Install
  python -m spacy download en_core_web_sm
  
python yatsee_normalize_structure.py -i transcripts_medium/
```

### 5. Optional: Summarize Transcripts

YATSEE includes an optional script for generating summaries using a local LLM via the ollama tool. This script demonstrates one possible downstream use of the normalized transcripts.

```bash
# Requires ollama and a pulled model
python3 yatsee_summarize_transcripts.py -i normalized/ -m your_pulled_model
```

### 📄 Script Summary

| Script                            | Purpose                                       |
|-----------------------------------|-----------------------------------------------|
| `yatsee_download_audio.sh`        | Download audio from YouTube URLs              |
| `yatsee_format_audio.sh`          | Convert downloaded files to `.flac` or `.wav` |
| `yatsee_transcribe_audio.py`      | Transcribe audio files to `.vtt`              |
| `yatsee_slice_vtt.py `            | Slice and segment `.vtt` files                |
| `yatsee_normalize_structure.py`   | Clean and normalize text structure            |
| `yatsee_summarize_transcripts.py` | Generate summaries from cleaned transcripts   |
