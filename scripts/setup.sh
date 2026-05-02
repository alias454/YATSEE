#!/usr/bin/env bash
# yatsee_setup.sh
#
# Bootstrap helper for local YATSEE development.
#
# This script is intentionally a convenience wrapper around:
#   - Python version checks
#   - virtualenv creation
#   - editable package install from pyproject.toml
#   - stage-related system tool checks
#   - optional spaCy model install
#
# It is not the source of truth for Python dependencies. pyproject.toml is.

set -euo pipefail

echo "Starting YATSEE bootstrap..."

PYTHON_REQUIRED="3.11"
VENV_DIR=".venv"
DEFAULT_INSTALL_TARGET="full"
DEFAULT_SPACY_MODEL="en_core_web_md"

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

print_header() {
    printf '\n== %s ==\n' "$1"
}

python_version_ok() {
    local found_version
    found_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    if [[ "$(printf '%s\n' "$PYTHON_REQUIRED" "$found_version" | sort -V | head -n1)" != "$PYTHON_REQUIRED" ]]; then
        echo "ERROR: Python ${PYTHON_REQUIRED}+ is required. Found ${found_version}."
        return 1
    fi

    echo "Python ${found_version} OK"
}

check_python_and_venv() {
    print_header "Python checks"

    if ! have_cmd python3; then
        echo "ERROR: python3 is not installed or not on PATH."
        exit 1
    fi

    python_version_ok

    if ! python3 -m venv --help >/dev/null 2>&1; then
        echo "ERROR: Python venv module is not available."
        exit 1
    fi

    echo "venv module available"
}

check_repo_files() {
    print_header "Repository checks"

    if [[ ! -f "pyproject.toml" ]]; then
        echo "ERROR: pyproject.toml not found. Run this from the YATSEE repository root."
        exit 1
    fi

    echo "Found pyproject.toml"
}

check_system_tools() {
    print_header "System tool checks"

    local missing=0

    if have_cmd ffmpeg; then
        echo "ffmpeg found: $(command -v ffmpeg)"
    else
        echo "WARNING: ffmpeg not found."
        echo "  Needed for audio formatting/transcoding workflows."
        echo "  Linux: install via your package manager"
        echo "  macOS: brew install ffmpeg"
        missing=1
    fi

    if have_cmd yt-dlp; then
        echo "yt-dlp found: $(command -v yt-dlp)"
    else
        echo "WARNING: yt-dlp not found."
        echo "  Needed for source fetching workflows."
        echo "  Linux: install via your package manager or pipx"
        echo "  macOS: brew install yt-dlp"
        missing=1
    fi

    if [[ "$missing" -eq 0 ]]; then
        echo "Required external tools for common workflows are present"
    else
        echo "One or more optional-but-important system tools are missing"
    fi
}

create_and_activate_venv() {
    print_header "Virtual environment"

    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
        echo "Created virtual environment at $VENV_DIR"
    else
        echo "Using existing virtual environment at $VENV_DIR"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "Activated virtual environment"
    echo "Remember to run: source $VENV_DIR/bin/activate"
}

install_package() {
    print_header "Package install"

    local install_choice install_target pip_spec

    echo "Install targets:"
    echo "  1) base          -> pip install -e ."
    echo "  2) transcript    -> pip install -e .[transcript]"
    echo "  3) intelligence  -> pip install -e .[intelligence]"
    echo "  4) llamacpp      -> pip install -e .[llamacpp]"
    echo "  5) index         -> pip install -e .[index]"
    echo "  6) ui            -> pip install -e .[ui]"
    echo "  7) full          -> pip install -e .[full]"
    echo "  8) skip install"

    read -r -p "Choose install target [7]: " install_choice
    install_choice="${install_choice:-7}"

    case "$install_choice" in
        1) install_target="base" ;;
        2) install_target="transcript" ;;
        3) install_target="intelligence" ;;
        4) install_target="llamacpp" ;;
        5) install_target="index" ;;
        6) install_target="ui" ;;
        7) install_target="$DEFAULT_INSTALL_TARGET" ;;
        8)
            echo "Skipping package install"
            return 0
            ;;
        *)
            echo "Unknown choice: $install_choice"
            exit 1
            ;;
    esac

    python -m pip install --upgrade pip

    if [[ "$install_target" == "base" ]]; then
        pip_spec="-e ."
    else
        pip_spec="-e .[${install_target}]"
    fi

    echo "Installing with: pip install ${pip_spec}"
    python -m pip install ${pip_spec}
    echo "Package install complete"
}

check_compute_device() {
    print_header "Compute device"

    if ! python -c "import torch" >/dev/null 2>&1; then
        echo "torch not installed in this environment; skipping CUDA/MPS detection"
        return 0
    fi

    python <<'PY'
import torch

device = "CPU"
if torch.cuda.is_available():
    device = "CUDA GPU"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "Apple MPS"

print(f"Detected processing device: {device}")
PY
}

maybe_install_spacy_model() {
    print_header "spaCy model"

    if ! python -c "import spacy" >/dev/null 2>&1; then
        echo "spaCy not installed in this environment; skipping model install"
        return 0
    fi

    read -r -p "Check/install spaCy model ${DEFAULT_SPACY_MODEL}? [Y/n]: " spacy_choice
    spacy_choice="${spacy_choice:-Y}"

    if [[ ! "$spacy_choice" =~ ^[Yy]$ ]]; then
        echo "Skipping spaCy model check"
        return 0
    fi

    if python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('${DEFAULT_SPACY_MODEL}') else 1)"; then
        echo "spaCy model ${DEFAULT_SPACY_MODEL} already available"
    else
        echo "Installing spaCy model ${DEFAULT_SPACY_MODEL}..."
        python -m spacy download "${DEFAULT_SPACY_MODEL}"
        echo "spaCy model ${DEFAULT_SPACY_MODEL} installed"
    fi
}

show_next_steps() {
    print_header "Next steps"

    cat <<'EOF'
Common commands:

  yatsee --help
  yatsee config --help
  yatsee config entity list
  yatsee config validate

If you are using editable install mode and the console script is not available yet, use:

  python -m yatsee.cli.main --help

Notes:
  - pyproject.toml is the source of truth for Python dependencies
  - ffmpeg is needed for audio formatting/transcoding
  - yt-dlp is needed for source fetching
  - transcript/index workflows may require spaCy models and additional extras
  - intel workflows now use provider settings such as llm_provider and llm_provider_url in yatsee.toml
EOF
}

show_summary() {
    print_header "Summary"

    echo "Bootstrap complete"
    echo "  Python required : ${PYTHON_REQUIRED}+"
    echo "  Virtualenv      : ${VENV_DIR}"
    echo "  pyproject.toml  : present"
    echo "  ffmpeg          : $(command -v ffmpeg || echo 'missing')"
    echo "  yt-dlp          : $(command -v yt-dlp || echo 'missing')"
}

main() {
    check_python_and_venv
    check_repo_files
    check_system_tools
    create_and_activate_venv
    install_package
    check_compute_device
    maybe_install_spacy_model
    show_summary
    show_next_steps
}

main "$@"