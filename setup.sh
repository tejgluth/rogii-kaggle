#!/bin/bash
# One-command setup for the ROGII competition environment
# Run on DGX Spark after cloning the repo

set -e

echo "======================================================"
echo " ROGII Kaggle Competition Environment Setup"
echo "======================================================"

ENV_NAME="rogii"
PYTHON_VERSION="3.11"

# --- Conda environment ---
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing Python dependencies..."
pip install -r requirements.txt

# --- RAPIDS (GPU) ---
echo "Attempting RAPIDS install (CUDA/DGX Spark)..."
pip install cudf-cu12 cuml-cu12 \
    --extra-index-url=https://pypi.nvidia.com \
    --quiet 2>/dev/null \
    && echo "RAPIDS installed (GPU enabled)" \
    || echo "RAPIDS not available — CPU fallback will be used"

# --- Codex CLI ---
echo ""
echo "Installing Codex CLI..."
if ! command -v node &> /dev/null; then
    echo "Node.js not found. Install Node.js 18+ from https://nodejs.org"
else
    npm install -g @openai/codex 2>/dev/null \
        && echo "Codex CLI installed: $(codex --version 2>/dev/null)" \
        || echo "Could not install Codex CLI. Run: npm install -g @openai/codex"
fi

# --- .env ---
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env — only fill in KAGGLE_USERNAME and KAGGLE_KEY"
fi

# --- experiment log ---
python -c "
import csv, pathlib
p = pathlib.Path('experiments/experiment_log.csv')
if p.stat().st_size < 10:
    print('Experiment log ready.')
" 2>/dev/null || true

chmod +x scripts/download_data.sh

echo ""
echo "======================================================"
echo " Setup complete. Next steps:"
echo ""
echo "  1.  conda activate rogii"
echo ""
echo "  2.  Authenticate Claude Code (Claude Pro subscription):"
echo "      claude          <- will open browser login on first run"
echo ""
echo "  3.  Authenticate Codex CLI (ChatGPT Plus subscription):"
echo "      codex auth      <- will open browser login"
echo ""
echo "  4.  Add Kaggle credentials:"
echo "      Edit .env  OR  place kaggle.json in ~/.kaggle/"
echo ""
echo "  5.  Download data:"
echo "      bash scripts/download_data.sh"
echo ""
echo "  6.  Start the system:"
echo "      claude          <- reads CLAUDE.md and begins"
echo "======================================================"
