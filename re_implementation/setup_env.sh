#!/usr/bin/env bash
# ============================================================
# setup_env.sh — Create a Python environment for the project.
#
# Usage:  bash setup_env.sh
#
# • If conda is available it creates a prefixed conda env at
#   ./.conda_env and installs the requirements there.
# • Otherwise it falls back to a standard Python venv.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
CONDA_PREFIX_DIR="$SCRIPT_DIR/.conda_env"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== Seq2Seq Re-implementation — Environment Setup ==="

# ----------------------------------------------------------
# Try conda first
# ----------------------------------------------------------
if command -v conda &>/dev/null; then
    echo "[INFO] conda found — creating prefixed conda environment at $CONDA_PREFIX_DIR"
    conda create --prefix "$CONDA_PREFIX_DIR" python=3.11 -y
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_PREFIX_DIR"
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
    echo ""
    echo "[DONE] Conda env ready.  Activate with:"
    echo "  conda activate $CONDA_PREFIX_DIR"
else
    echo "[INFO] conda not found — falling back to python venv"
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
    echo ""
    echo "[DONE] Venv ready.  Activate with:"
    echo "  source $VENV_DIR/bin/activate"
fi

# Download NLTK data needed for BLEU
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo ""
echo "Environment setup complete."
