#!/bin/bash
# Generate 1000 complaints with each vLLM model, 32 workers, sequential
set -e

cd /home/dani/miUrba/mi-urba

MODELS=(
    "gen_qwen3_5_27b_awq"
    "gen_qwen3_6_35b_awq"
    "gen_gemma4_12b_awq"
    "gen_qwen3_5_4b_awq"
    "gen_qwen3_4b_instr_awq"
)

N=1000
WORKERS=32
SEED=42

for model in "${MODELS[@]}"; do
    outdir="eval/results/dataset-bench/${model}"
    logfile="${outdir}/log.txt"

    echo "=================================================="
    echo "[$(date '+%H:%M:%S')] Generando con $model → $outdir"
    echo "=================================================="

    rm -rf "$outdir"
    mkdir -p "$outdir"

    cd miurba-generator
    .venv/bin/python cli.py generate \
        -n $N \
        --model "$model" \
        --config ../eval/config.eval.yaml \
        --output-dir "../$outdir" \
        --no-resume \
        --seed $SEED \
        --workers $WORKERS \
        --timeout 300 \
        > "../$logfile" 2>&1 || true
    cd ..

    n=$(grep -c . "${outdir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
    failed=$(grep -c "Generation failed\|Worker error\|429\|Error" "$logfile" 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] $model: $n/$N generadas, $failed errores"
    echo ""
done

echo "=================================================="
echo "[$(date '+%H:%M:%S')] TODOS LOS MODELOS COMPLETADOS"
echo "=================================================="

# Summary
echo ""
echo "=== RESUMEN ==="
for model in "${MODELS[@]}"; do
    outdir="eval/results/dataset-bench/${model}"
    logfile="${outdir}/log.txt"
    n=$(grep -c . "${outdir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
    gt=$(grep -c . "${outdir}/raw/ground_truth.jsonl" 2>/dev/null || echo 0)
    echo "  ${model}: ${n} complaints, ${gt} ground truth"
done