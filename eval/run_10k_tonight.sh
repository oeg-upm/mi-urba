#!/usr/bin/env bash
# Schedule the 10K dataset generation for tonight at 00:00.
# Usage: nohup bash eval/run_10k_tonight.sh &
set -u
ROOT="/backup/pc1/data/miUrba2/mi-urba"
TARGET="08:00"  # placeholder, computed below

# seconds until next 00:00 local
now=$(date +%s)
target=$(date -d "tomorrow 00:00" +%s)
delta=$((target - now))
echo "[$(date '+%F %T')] sleeping ${delta}s until $(date -d @$target '+%F %T') before starting 10K generation"

sleep "$delta"

echo "[$(date '+%F %T')] launching generate_10k_dataset.py"
cd "$ROOT"
exec python3 eval/generate_10k_dataset.py