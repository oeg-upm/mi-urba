#!/bin/bash
# Generate 1000 complaints per model with warmup + 32 workers
# For each model: send warmup request, wait up to 15min for load, then generate

cd /home/dani/miUrba/mi-urba

MODELS=(
    "gen_qwen3_6_35b_awq"
    "gen_gemma4_12b_awq"
    "gen_qwen3_5_4b_awq"
    "gen_qwen3_4b_instr_awq"
)

MODEL_IDS=(
    "qwen3.6-35b-awq"
    "gemma4-12b-awq"
    "qwen3.5-4b-awq"
    "qwen3-4b-instruct-awq"
)

N=1000
WORKERS=32
SEED=42
WARMUP_TIMEOUT=900  # 15 min

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    model_id="${MODEL_IDS[$i]}"
    outdir="eval/results/dataset-bench/${model}"
    logfile="${outdir}/log.txt"

    echo "=================================================="
    echo "[$(date '+%H:%M:%S')] WARMUP: $model ($model_id)"
    echo "=================================================="

    # Step 1: Warmup — send a simple request and wait up to 15 min
    warmup_ok=0
    for attempt in $(seq 1 $((WARMUP_TIMEOUT / 20))); do
        response=$(curl -s --max-time 60 http://192.168.10.11:9292/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model_id}\",\"messages\":[{\"role\":\"user\",\"content\":\"Responde solo: OK\"}],\"max_tokens\":10,\"enable_thinking\":false,\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>&1)
        
        if echo "$response" | grep -q "choices"; then
            echo "[$(date '+%H:%M:%S')] Modelo cargado OK (intento $attempt)"
            warmup_ok=1
            break
        fi
        echo "[$(date '+%H:%M:%S')] Esperando... (intento $attempt)"
        sleep 20
    done

    if [ "$warmup_ok" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ERROR: $model no cargó en 15 min. Saltando."
        echo ""
        continue
    fi

    # Step 2: Generate
    echo "[$(date '+%H:%M:%S')] Generando 1000 quejas con $model (32 workers)..."
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
    echo "[$(date '+%H:%M:%S')] $model: $n/$N generadas"
    echo ""
    sleep 10
done

echo "=================================================="
echo "[$(date '+%H:%M:%S')] COMPLETADO"
echo "=================================================="
echo ""
echo "=== RESUMEN FINAL ==="
ALL_MODELS=(
    "gen_gemma4_31b_awq" "gen_gemma4_26b_awq" "gen_qwen3_5_27b_awq"
    "gen_qwen3_6_35b_awq" "gen_gemma4_12b_awq" "gen_qwen3_5_4b_awq" "gen_qwen3_4b_instr_awq"
)
for m in "${ALL_MODELS[@]}"; do
    outdir="eval/results/dataset-bench/${m}"
    n=$(grep -c . "${outdir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
    echo "  ${m}: ${n}/1000"
done