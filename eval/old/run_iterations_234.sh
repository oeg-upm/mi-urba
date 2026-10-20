#!/bin/bash
# miUrba — Run iterations 2, 3, 4 in sequence with different prompt variants
# Each iteration: apply prompt variant → generate → extract → evaluate
#
# Usage: bash eval/run_iterations_234.sh [N]
#   N = complaints per generator (default 250)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

N=${1:-250}
LOG_ALL="$ROOT_DIR/eval/results/iterations_234.log"

echo "=== Starting iterations 2, 3, 4 with N=$N ===" | tee "$LOG_ALL"

for ITER in 2 3 4; do
    echo "" | tee -a "$LOG_ALL"
    echo "================================================================" | tee -a "$LOG_ALL"
    echo "  APPLYING PROMPT VARIANT FOR ITERATION $ITER" | tee -a "$LOG_ALL"
    echo "================================================================" | tee -a "$LOG_ALL"
    bash eval/apply_prompt_variant.sh "$ITER" | tee -a "$LOG_ALL"

    echo "" | tee -a "$LOG_ALL"
    echo "  RUNNING ITERATION $ITER (N=$N)" | tee -a "$LOG_ALL"
    echo "================================================================" | tee -a "$LOG_ALL"

    LOG_ITER="$ROOT_DIR/eval/results/iter_${ITER}_run.log"
    bash eval/run_iteration.sh "$ITER" "$N" 2>&1 | tee -a "$LOG_ALL" "$LOG_ITER"

    echo "Iteration $ITER complete." | tee -a "$LOG_ALL"
done

echo "" | tee -a "$LOG_ALL"
echo "=== ALL ITERATIONS (2, 3, 4) COMPLETE ===" | tee -a "$LOG_ALL"