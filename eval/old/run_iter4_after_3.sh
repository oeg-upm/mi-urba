#!/bin/bash
# miUrba — Run iteration 4 after iteration 3 completes
# Waits for iter_3 to finish, then applies iter_4 prompts and runs iteration 4.
# Usage: nohup bash eval/run_iter4_after_3.sh 250 &

N=${1:-250}
ROOT_DIR="/mnt/backup/pc1/data/miUrba2/mi-urba"
cd "$ROOT_DIR"

# Wait for iter_3 to complete (check for its completion marker in log)
echo "[$(date '+%H:%M:%S')] Waiting for iter_3 to complete..."
while true; do
    if grep -q "ITERATION 3 COMPLETE" eval/results/iter_3_run.log 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] iter_3 complete!"
        break
    fi
    if ! pgrep -f "run_iteration.sh 3" > /dev/null 2>&1; then
        # Check if it actually completed or died
        if grep -q "ITERATION 3 COMPLETE" eval/results/iter_3_run.log 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] iter_3 complete!"
            break
        else
            echo "[$(date '+%H:%M:%S')] iter_3 process died without completion marker. Check log."
            break
        fi
    fi
    sleep 60
done

# Apply iter_4 prompts
echo "[$(date '+%H:%M:%S')] Applying iter_4 prompt variant..."
bash eval/apply_prompt_variant.sh 4

# Run iter_4
echo "[$(date '+%H:%M:%S')] Starting iter_4..."
bash eval/run_iteration.sh 4 "$N" > eval/results/iter_4_run.log 2>&1
echo "[$(date '+%H:%M:%S')] iter_4 finished."

# Restore iter_2 prompts as current
echo "[$(date '+%H:%M:%S')] Restoring iter_2 prompts as current..."
bash eval/apply_prompt_variant.sh 2
echo "[$(date '+%H:%M:%S')] Done."