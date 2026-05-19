#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tejas/Desktop/rogii-kaggle"
PROMPT="$ROOT/prompts/orchestrator_continue.md"
LOG_DIR="$ROOT/logs/codex_orchestrator"
LOCK_FILE="$ROOT/.codex_orchestrator.lock"
CODEX_BIN="/home/tejas/.nvm/versions/node/v22.22.3/bin/codex"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${STAMP}.log"
LAST_FILE="$LOG_DIR/last_message_${STAMP}.txt"

mkdir -p "$LOG_DIR"
cd "$ROOT"

{
  echo "[$(date --iso-8601=seconds)] scheduled Codex orchestrator launch"
  echo "root=$ROOT"
  echo "prompt=$PROMPT"
} >> "$LOG_FILE"

if ! flock -n "$LOCK_FILE" true; then
  echo "[$(date --iso-8601=seconds)] previous scheduled orchestrator still running; skipping" >> "$LOG_FILE"
  exit 0
fi

flock "$LOCK_FILE" "$CODEX_BIN" exec \
  --cd "$ROOT" \
  --sandbox danger-full-access \
  --search \
  -m gpt-5.5 \
  -c 'model_reasoning_effort="xhigh"' \
  --output-last-message "$LAST_FILE" \
  - < "$PROMPT" >> "$LOG_FILE" 2>&1

echo "[$(date --iso-8601=seconds)] scheduled Codex orchestrator finished" >> "$LOG_FILE"
