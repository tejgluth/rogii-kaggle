#!/usr/bin/env bash
set -euo pipefail
cd "/Users/speakeasy/Projects/rogii-kaggle/local_runs/exp116_fast_cache"
ROGII_DATASET_PATH="/Users/speakeasy/Projects/rogii-kaggle/data/rogii-wellbore-geology-prediction" \
EXP116_LOCAL_PROFILE=fast \
EXP116_N_SPLITS=3 \
EXP116_LGB_COUNT=1 \
EXP116_CB_COUNT=1 \
EXP116_MAX_ESTIMATORS=3500 \
EXP116_OPTUNA_TRIALS=120 \
EXP116_RESID_ROUNDS=400 \
"/Users/speakeasy/Projects/rogii-kaggle/.venv/bin/python" "/Users/speakeasy/Projects/rogii-kaggle/scripts/run_exp116_local_train_cache.py"
cd "/Users/speakeasy/Projects/rogii-kaggle"
".venv/bin/python" "scripts/package_upload_exp116_cache.py" \
  --run-dir "local_runs/exp116_fast_cache" \
  --package-dir "kaggle_datasets/rogii-exp116-fast-model-cache" \
  --dataset-id "tejasgluth/rogii-exp116-fast-model-cache" \
  --title "ROGII exp116 fast model cache" \
  --upload
