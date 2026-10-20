#!/bin/bash
# miUrba — Run iterations 5, 6, 7 in sequence with different prompt variants
# Each iteration: apply prompt variant → generate → extract → evaluate
#
# Variants:
#   iter 5: "Combinación ganadora" — iter2 + risks de iter3 + subcat primera mención de iter4
#   iter 6: "Estructura guiada" — guía de estructura del texto (ubicación→problema→elementos→urgencia)
#   iter 7: "Minimalista + few-shot" — instrucciones simplificadas + principio una sola causa
#
# Usage: nohup bash eval/run_iterations_567.sh [N] &
#   N = complaints per generator (default 250)

N=${1:-250}
ROOT_DIR="/mnt/backup/pc1/data/miUrba2/mi-urba"
cd "$ROOT_DIR"

LOG_ALL="$ROOT_DIR/eval/results/iterations_567.log"

echo "=== Starting iterations 5, 6, 7 with N=$N ===" | tee "$LOG_ALL"

for ITER in 5 6 7; do
    echo "" | tee -a "$LOG_ALL"
    echo "================================================================" | tee -a "$LOG_ALL"
    echo "  APPLYING PROMPT VARIANT FOR ITERATION $ITER" | tee -a "$LOG_ALL"
    echo "================================================================" | tee -a "$LOG_ALL"
    bash eval/apply_prompt_variant.sh "$ITER" 2>&1 | tee -a "$LOG_ALL"

    echo "" | tee -a "$LOG_ALL"
    echo "  RUNNING ITERATION $ITER (N=$N)" | tee -a "$LOG_ALL"
    echo "================================================================" | tee -a "$LOG_ALL"

    LOG_ITER="$ROOT_DIR/eval/results/iter_${ITER}_run.log"
    bash eval/run_iteration.sh "$ITER" "$N" > "$LOG_ITER" 2>&1
    RC=$?
    echo "Iteration $ITER finished (exit code $RC)." | tee -a "$LOG_ALL"

    # Show last few lines of the iteration log
    tail -5 "$LOG_ITER" | tee -a "$LOG_ALL"
done

# Restore iter2 prompts as current
echo "" | tee -a "$LOG_ALL"
echo "  Restoring iter_2 prompts as current" | tee -a "$LOG_ALL"
bash eval/apply_prompt_variant.sh 2 2>&1 | tee -a "$LOG_ALL"

echo "" | tee -a "$LOG_ALL"
echo "=== ITERATIONS 5, 6, 7 COMPLETE ===" | tee -a "$LOG_ALL"