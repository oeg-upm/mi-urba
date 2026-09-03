#!/bin/bash
# Recover missing complaints with --resume
cd /home/dani/miUrba/mi-urba

MODELS=(
    "gen_gemma4_31b_awq|gemma4-31b-awq|12000"
    "gen_qwen3_6_35b_awq|qwen3.6-35b-awq|10000"
    "gen_qwen3_5_27b_awq|qwen3.5-27b-awq|8000"
    "gen_gemma4_26b_awq|gemma4-26b-awq|8000"
    "gen_gemma4_12b_awq|gemma4-12b-awq|6000"
    "gen_qwen3_5_4b_awq|qwen3.5-4b-awq|3000"
    "gen_qwen3_4b_instr_awq|qwen3-4b-instruct-awq|3000"
)

WORKERS=32
SEED=42
WARMUP_TIMEOUT=900

BASEDIR="eval/results/dataset-50k"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_key model_id count <<< "$entry"
    outdir="${BASEDIR}/${model_key}"
    logfile="${outdir}/log_retry.txt"

    existing=$(grep -c . "${outdir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
    missing=$((count - existing))

    if [ "$missing" -le 0 ]; then
        echo "[$(date '+%H:%M:%S')] $model_key: SKIP ($existing/$count complete)"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] $model_key: $missing missing, recovering..."

    # Warmup
    warmup_ok=0
    for attempt in $(seq 1 $((WARMUP_TIMEOUT / 20))); do
        response=$(curl -s --max-time 120 http://192.168.10.11:9292/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model_id}\",\"messages\":[{\"role\":\"user\",\"content\":\"Di OK\"}],\"max_tokens\":5,\"enable_thinking\":false,\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>&1)
        if echo "$response" | grep -q '"choices"'; then
            echo "[$(date '+%H:%M:%S')] $model_id loaded (attempt $attempt)"
            warmup_ok=1
            break
        fi
        sleep 20
    done

    if [ "$warmup_ok" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] $model_id FAILED to load. Skipping."
        continue
    fi

    # Generate with --resume (only fills missing IDs)
    cd miurba-generator
    .venv/bin/python cli.py generate \
        -n $count \
        --model "$model_key" \
        --config ../eval/config.eval.yaml \
        --output-dir "../$outdir" \
        --seed $SEED \
        --workers $WORKERS \
        --timeout 300 \
        > "../$logfile" 2>&1 || true
    cd ..

    n=$(grep -c . "${outdir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] $model_key: $n/$count"
    sleep 5
done

echo ""
echo "=== RESUMEN FINAL ==="
total=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_key model_id count <<< "$entry"
    n=$(grep -c . "${BASEDIR}/${model_key}/raw/complaints.jsonl" 2>/dev/null || echo 0)
    total=$((total + n))
    printf "  %-35s %6d / %6d\n" "$model_key" "$n" "$count"
done
printf "  %-35s %6d / %6d\n" "TOTAL" "$total" "50000"