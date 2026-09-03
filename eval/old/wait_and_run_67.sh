#!/bin/bash
# miUrba — Wait for inference server to recover, then run iterations 6 and 7
# Polls the server with a small model until it responds, then launches iterations.
#
# Usage: nohup bash eval/wait_and_run_67.sh [N] &

N=${1:-250}
ROOT_DIR="/mnt/backup/pc1/data/miUrba2/mi-urba"
cd "$ROOT_DIR"

LOG="$ROOT_DIR/eval/results/wait_and_run_67.log"
ENDPOINT="http://192.168.10.11:9292"

echo "=== $(date '+%H:%M:%S') Waiting for inference server to recover ===" | tee "$LOG"

# Poll with a small/fast model (qwen3-4b-instruct-awq) until it responds
while true; do
    response=$(curl -s --max-time 60 "$ENDPOINT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3-4b-instruct-awq","messages":[{"role":"user","content":"Di OK"}],"max_tokens":5,"enable_thinking":false,"chat_template_kwargs":{"enable_thinking":false}}' 2>&1)
    if echo "$response" | grep -q '"choices"'; then
        echo "=== $(date '+%H:%M:%S') Server recovered! ===" | tee -a "$LOG"
        break
    fi
    echo "[$(date '+%H:%M:%S')] Server still down: $(echo "$response" | head -c 100)" | tee -a "$LOG"
    sleep 120
done

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S') Running iterations 6 and 7 ===" | tee -a "$LOG"

for ITER in 6 7; do
    echo "" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"
    echo "  APPLYING PROMPT VARIANT FOR ITERATION $ITER" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"
    bash eval/apply_prompt_variant.sh "$ITER" 2>&1 | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "  RUNNING ITERATION $ITER (N=$N)" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"

    LOG_ITER="$ROOT_DIR/eval/results/iter_${ITER}_run.log"
    bash eval/run_iteration.sh "$ITER" "$N" > "$LOG_ITER" 2>&1
    RC=$?
    echo "Iteration $ITER finished (exit code $RC)." | tee -a "$LOG"
    tail -5 "$LOG_ITER" | tee -a "$LOG"
done

# Restore iter2 prompts as current
echo "" | tee -a "$LOG"
echo "  Restoring iter_2 prompts as current" | tee -a "$LOG"
bash eval/apply_prompt_variant.sh 2 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S') ITERATIONS 6, 7 COMPLETE ===" | tee -a "$LOG"