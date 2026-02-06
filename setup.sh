#!/usr/bin/env bash
# yatsee_setup.sh
# Pre-flight setup for YATSEE pipeline
# Ensures Python, virtualenv, dependencies, GPU/MPS, spaCy, etc. are ready

set -euo pipefail

echo "🔧 Starting YATSEE pre-flight setup..."

# ----------------------------
# Check Python version (>=3.10)
# ----------------------------
PYTHON_REQUIRED="3.10"
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$(printf '%s\n' "$PYTHON_REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1)" != "$PYTHON_REQUIRED" ]]; then
    echo "❌ Python $PYTHON_REQUIRED+ is required. Found $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION OK"

# ----------------------------
# Check venv module
# ----------------------------
if ! python3 -m venv --help &>/dev/null; then
    echo "❌ Python venv module is not available"
    exit 1
fi
echo "✅ venv module available"

# ----------------------------
# Check essential CLI tools
# ----------------------------
tools=("ffmpeg" "yt-dlp")
for tool in "${tools[@]}"; do
    if ! command -v "$tool" &>/dev/null; then
        echo "❌ $tool is not installed. Please install it via your package manager."
        echo "   Linux: sudo apt install $tool  |  macOS: brew install $tool"
    else
        echo "✅ $tool found: $(command -v "$tool")"
    fi
done

# ----------------------------
# Create and activate virtualenv
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
# Upgrade pip and install requirements
# ----------------------------
INSTALL_PERFORMED=false
read -p "Do you want to upgrade pip and install requirements? [y/N]: " choice
if [[ "$choice" =~ ^[Yy]$ ]]; then
    pip install --upgrade pip
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        echo "✅ Python dependencies installed"
        INSTALL_PERFORMED=true
    else
        echo "⚠️ requirements.txt not found, skipping"
    fi
else
    echo "⏭️ Skipping installation. (Note: Some checks may be skipped if requirements aren't setup)"
fi

# ----------------------------
# Check GPU / MPS (Only if torch is available)
# ----------------------------
DEVICE="Unknown (Requirements skipped)"

# We check if torch is actually 'importable' before running detection logic
if python3 -c "import torch" &>/dev/null; then
    echo "🔍 Detecting compute device..."
    GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    MPS_AVAILABLE=$(python3 -c "import torch; print(torch.backends.mps.is_available())")

    if [[ "$GPU_AVAILABLE" == "True" ]]; then
        DEVICE="CUDA GPU"
    elif [[ "$MPS_AVAILABLE" == "True" ]]; then
        DEVICE="Apple MPS"
    else
        DEVICE="CPU"
        echo "⚠️ No GPU/MPS detected. CPU fallback will be used."
    fi
    echo "✅ Using device: $DEVICE"
fi

# ----------------------------
# Install spaCy model (Only if spacy is available)
# ----------------------------
SPACY_MODEL="en_core_web_md"
if python3 -c "import spacy" &>/dev/null; then
    if ! python3 -m spacy validate | grep -q "$SPACY_MODEL"; then
        echo "⚡ Installing spaCy model $SPACY_MODEL..."
        python3 -m spacy download "$SPACY_MODEL" --force
    fi
    echo "✅ spaCy model $SPACY_MODEL ready"
else
    echo "⏭️ spaCy not installed; skipping model check."
fi

# ----------------------------
# Create default directories
# ----------------------------
# Suggest entity-specific directory scaffolding
echo "💡 Entity directories will be created by yatsee_config_builder.py after adding entities in yatsee.toml"

# ----------------------------
# Final summary
# ----------------------------
echo "🖥️  System summary:"
echo "   Python version: $PYTHON_VERSION"
echo "   Virtualenv: $VENV_DIR"
echo "   CLI tools: ffmpeg ($(command -v ffmpeg || echo 'missing')), yt-dlp ($(command -v yt-dlp || echo 'missing'))"
echo "   Processing device: $DEVICE"
echo "   spaCy model: $SPACY_MODEL"

echo "🎉 YATSEE pre-flight setup complete!"
echo "Next step: run ./yatsee_config_builder.py to create entity configs and directories"
