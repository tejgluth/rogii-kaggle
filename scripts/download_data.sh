#!/bin/bash
# Download competition data using Kaggle API
# Prerequisites: kaggle CLI installed and ~/.kaggle/kaggle.json configured

set -e

COMPETITION="rogii-wellbore-geology-prediction"
DATA_DIR="data/raw"

# Local no-admin setup: prefer a repo virtualenv if one exists.
if [ -d ".venv/bin" ]; then
    export PATH="$PWD/.venv/bin:$PATH"
fi

# Kaggle CLI accepts KAGGLE_USERNAME/KAGGLE_KEY from the environment.
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

echo "Downloading competition data: $COMPETITION"
echo "Destination: $DATA_DIR"

# Check kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found"
    echo "Install: pip install kaggle"
    echo "Auth:    place kaggle.json in ~/.kaggle/"
    exit 1
fi

mkdir -p "$DATA_DIR"

# Download
kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

# Unzip
cd "$DATA_DIR"
for f in *.zip; do
    echo "Unzipping $f..."
    unzip -q "$f"
done

echo ""
echo "Download complete. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
