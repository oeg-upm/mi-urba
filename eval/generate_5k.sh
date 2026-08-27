#!/usr/bin/env bash
# Generate the 5K diverse dataset: 500 complaints × 10 selected models.
# Uses cli.py directly, one model at a time, workers=1, resume supported.
#
# Usage: bash eval/generate_5k.sh
# Stop:  pkill -9 -f "miurba-generator/cli.py"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEN="$ROOT/miurba-generator"
#VENV="$GEN/.venv/bin/python"
VENV="python3"
CONFIG="$ROOT/eval/config.eval.yaml"
OUT="$ROOT/eval/results/dataset-5k"

MODELS=(
  gen_deepseek-r1-distill-qwen-7b
  gen_phi-4
  gen_qwen3-0.6b
  gen_pantheon-rp-8b-llama3
  gen_mistral-7b-v0.3-base
  gen_deepseek-r1-distill-llama-8b
  gen_granite-3.3-2b
  gen_olmo-2-13b-instruct
  gen_falcon3-7b-instruct
  gen_glm-4.7-flash
)

N=500
SEED=42
WORKERS=2
TIMEOUT=900
MAX_RETRIES=3
RETRY_DELAY=60

mkdir -p "$OUT"

i=0
for M in "${MODELS[@]}"; do
  i=$((i + 1))
  RAW="$OUT/$M/raw/complaints.jsonl"
  if [[ -f "$RAW" ]]; then
    DONE=$(wc -l < "$RAW")
  else
    DONE=0
  fi
  echo "[$(date '+%H:%M:%S')] [$i/${#MODELS[@]}] $M ($DONE/$N done)"

  if [[ "$DONE" -ge "$N" ]]; then
    echo "  SKIP (already complete)"
    continue
  fi

  "$VENV" "$GEN/cli.py" generate \
    -n "$N" \
    --model "$M" \
    --config "$CONFIG" \
    --output-dir "$OUT/$M" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --max-retries "$MAX_RETRIES" \
    --retry-delay "$RETRY_DELAY"

  echo "[$(date '+%H:%M:%S')] [$i/${#MODELS[@]}] $M DONE"
  sleep 20  # cooldown between models
done

echo "=== 5K GENERATION COMPLETE ==="
