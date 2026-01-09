#!/usr/bin/env bash
# yatsee_setup.sh
# Fully hardened pre-flight setup for YATSEE pipeline
# Ensures Python, virtualenv, dependencies, GPU/MPS, spaCy, and directories are ready

set -euo pipefail

echo "🔧 Starting YATSEE pre-flight setup..."

# ----------------------------
# 1. Check Python version (>=3.10)
# ----------------------------
PYTHON_REQUIRED="3.10"
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$(printf '%s\n' "$PYTHON_REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1)" != "$PYTHON_REQUIRED" ]]; then
    echo "❌ Python $PYTHON_REQUIRED+ is required. Found $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION OK"

# ----------------------------
# 2. Check venv module
# ----------------------------
if ! python3 -m venv --help &>/dev/null; then
    echo "❌ Python venv module is not available"
    exit 1
fi
echo "✅ venv module available"

# ----------------------------
# 3. Create and activate virtualenv
# ----------------------------
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "⚡ Virtual environment activated"
echo "💡 Remember to source $VENV_DIR/bin/activate in future terminals"

# ----------------------------
# 4. Upgrade pip and install requirements
# ----------------------------
pip install --upgrade pip
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️ requirements.txt not found, skipping pip install"
fi

# ----------------------------
# 5. Check essential CLI tools
# ----------------------------
for tool in ffmpeg yt-dlp; do
    if ! command -v "$tool" &>/dev/null; then
        echo "❌ $tool is not installed. Please install it via your package manager."
        echo "   Linux: sudo apt install $tool  |  macOS: brew install $tool"
    else
        echo "✅ $tool found: $(command -v "$tool")"
    fi
done

# ----------------------------
# 6. Check GPU / MPS / CPU
# ----------------------------
echo "🔍 Detecting compute device..."
GPU_AVAILABLE=false
MPS_AVAILABLE=false
DEVICE="CPU"

# CUDA GPU check
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_AVAILABLE=true
    DEVICE="CUDA GPU"
fi

# Apple MPS check
if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -iq 'True'; then
    MPS_AVAILABLE=true
    DEVICE="Apple MPS"
fi

if [[ "$GPU_AVAILABLE" == false && "$MPS_AVAILABLE" == false ]]; then
    echo "⚠️ No GPU/MPS detected. CPU fallback will be used (slower performance)."
    read -p "Do you want to continue anyway? [y/N]: " choice
    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        echo "🚫 Setup aborted due to lack of GPU/MPS"
        exit 1
    fi
fi
echo "✅ Using device: $DEVICE"

# ----------------------------
# 7. Install spaCy model
# ----------------------------
SPACY_MODEL="en_core_web_md"
if ! python3 -m spacy validate | grep -q "$SPACY_MODEL"; then
    echo "⚡ Installing spaCy model $SPACY_MODEL..."
    python3 -m spacy download "$SPACY_MODEL" --force
fi
echo "✅ spaCy model $SPACY_MODEL ready"

# ----------------------------
# 8. Create default directories
# ----------------------------
echo "🔧 Creating default pipeline directories..."
BASE_DIRS=("downloads" "audio" "transcripts_medium" "normalized" "summary" "data")
for d in "${BASE_DIRS[@]}"; do
    mkdir -p "$d"
done
echo "✅ Base directories created: ${BASE_DIRS[*]}"

# Suggest entity-specific directory scaffolding
echo "💡 Entity directories will be created by yatsee_config_builder.py after adding entities in yatsee.toml"

# ----------------------------
# 9. Final summary
# ----------------------------
echo "🖥️  System summary:"
echo "   Python version: $PYTHON_VERSION"
echo "   Virtualenv: $VENV_DIR"
echo "   CLI tools: ffmpeg ($(command -v ffmpeg || echo 'missing')), yt-dlp ($(command -v yt-dlp || echo 'missing'))"
echo "   Processing device: $DEVICE"
echo "   spaCy model: $SPACY_MODEL"

echo "🎉 YATSEE pre-flight setup complete!"
echo "Next step: run ./yatsee_config_builder.py to create entity configs and directories"
