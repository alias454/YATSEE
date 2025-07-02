# YATSEE Audio Transcription Pipeline

**YATSEE** -- Yet Another Tool for Speech Extraction & Enrichment

This project provides a modular pipeline for downloading, transcribing, cleaning, and preparing audio/video recordings into readable, structured text. It’s designed with a local-first philosophy, giving you full control over your data at every stage.

## Why This Exists

Capturing meetings, podcasts, or public hearings is easy. Turning them into something useful and searchable? Not so much.

Raw transcripts are often messy: broken sentence structure, inconsistent formatting, and poor readability. YATSEE steps through a clean, repeatable pipeline that transforms **raw audio** into **usable, structured text** that is ready for humans or machines.

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
- **Script:** `yatsee_vtt_to_txt.sh`
- **Input:** `.vtt` from `transcripts_<model>/`
- **Output:** `.txt` (timestamp-free) to same folder

#### b. Sentence Segmentation and Text Normalization
- **Script:** `yatsee_normalize_structure.py`
- **Input:** `.txt` from `transcripts_<model>/`
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

### Index & Search _(planned)_
- **Goal:** Store transcripts/summaries in a fast, queryable format (likely vector store or DB)
- **Tools:** TBD

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

GPU acceleration was enabled for Whisper / faster-whisper using CUDA 12.8 and NVIDIA driver 570.144.

Note: The pipeline works on CPU-only systems without a GPU. However, transcription (especially with Whisper or faster-whisper) will be much slower compared to systems with CUDA-enabled GPU acceleration.

> ⚠️ Not tested on `macOS` or `Windows`. Use at your own risk on those platforms.

---

### Required CLI Tools

- `yt-dlp` – Download livestream audio from YouTube
- `ffmpeg` – Convert audio to `.flac` or `.wav` format

### Required Python Packages

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

pip install pyyaml spacy
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
./yatsee_format_audio.sh downloads/
```

### 3. Transcribe Audio
```bash
python yatsee_transcribe_audio.py --input_dir audio/ --model base
```

### 4. Clean Text

**Strip timestamps**
```bash
./yatsee_vtt_to_txt.sh transcripts_base/
```

**Normalize structure**
```bash
python yatsee_normalize_structure.py --input_dir transcripts_base/
```

### 5. Optional: Summarize Transcripts
```bash
python yatsee_summarize_transcripts.py --input_dir normalized/
```

### 📄 Script Summary

| Script                            | Purpose                                       |
|-----------------------------------|-----------------------------------------------|
| `yatsee_download_audio.sh`        | Download audio from YouTube URLs              |
| `yatsee_format_audio.sh`          | Convert downloaded files to `.flac` or `.wav` |
| `yatsee_transcribe_audio.py`      | Transcribe audio files to `.vtt`              |
| `yatsee_vtt_to_txt.sh`            | Strip timestamps from `.vtt` files            |
| `yatsee_normalize_structure.py`   | Clean and normalize text structure            |
| `yatsee_summarize_transcripts.py` | Generate summaries from cleaned transcripts   |
