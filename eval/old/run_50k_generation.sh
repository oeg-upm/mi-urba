#!/bin/bash
# Generate 50K complaints distributed across 7 vLLM models
# Safeguards:
#   - Warmup per model (wait up to 15min for load)
#   - --resume: if interrupted, rerun and it continues where it left off
#   - Incremental file writes (each complaint appended immediately)
#   - Per-model logs + progress tracking
#   - Skip models that already have their full quota

cd /home/dani/miUrba/mi-urba

# Model → (config_key, model_id, count)
# Total: 50,000
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
WARMUP_TIMEOUT=900  # 15 min

BASEDIR="eval/results/dataset-50k"
PROGRESS_FILE="${BASEDIR}/progress.txt"
mkdir -p "$BASEDIR"

# Initialize progress file if not exists
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "# miUrba 50K dataset generation progress" > "$PROGRESS_FILE"
    echo "# Started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$PROGRESS_FILE"
    echo "# Format: model|count_generated|count_target|status|timestamp" >> "$PROGRESS_FILE"
fi

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_key model_id count <<< "$entry"
    outdir="${BASEDIR}/${model_key}"
    logfile="${outdir}/log.txt"

    # Check if already complete
    if [ -f "${outdir}/raw/complaints.jsonl" ]; then
        existing=$(grep -c . "${outdir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
        if [ "$existing" -ge "$count" ]; then
            echo "[$(date '+%H:%M:%S')] $model_key: SKIP (already $existing/$count)"
            continue
        fi
        echo "[$(date '+%H:%M:%S')] $model_key: RESUME ($existing/$count already generated)"
    else
        echo "[$(date '+%H:%M:%S')] $model_key: START ($count target)"
    fi

    # ── Step 1: Warmup ──
    echo "[$(date '+%H:%M:%S')] WARMUP $model_id..."
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
        echo "${model_key}|0|${count}|WARMUP_FAILED|$(date '+%Y-%m-%d_%H:%M:%S')" >> "$PROGRESS_FILE"
        continue
    fi

    # ── Step 2: Generate (with --resume for crash recovery) ──
    mkdir -p "$outdir"
    echo "[$(date '+%H:%M:%S')] Generating $count complaints with $WORKERS workers..."
    
    cd miurba-generator
    .venv/bin/python cli.py generate \
        -n $count \
        --model "$model_key" \
        --config ../eval/config.eval.yaml \
        --output-dir "../$outdir" \
        --seed $SEED \
        --workers $WORKERS \
        --timeout 300 \
        >> "../$logfile" 2>&1 || true
    cd ..

    # ── Step 3: Record progress ──
    n=$(grep -c . "${outdir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
    status="COMPLETE"
    if [ "$n" -lt "$count" ]; then
        status="PARTIAL"
    fi
    echo "${model_key}|${n}|${count}|${status}|$(date '+%Y-%m-%d_%H:%M:%S')" >> "$PROGRESS_FILE"
    echo "[$(date '+%H:%M:%S')] $model_key: $n/$count ($status)"
    echo ""
    sleep 10
done

# ── Final summary ──
echo "=================================================="
echo "[$(date '+%H:%M:%S')] ALL MODELS DONE"
echo "=================================================="
echo ""
echo "=== PROGRESS FILE ==="
cat "$PROGRESS_FILE"
echo ""
echo "=== DATASET SIZES ==="
total=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_key model_id count <<< "$entry"
    n=$(grep -c . "${BASEDIR}/${model_key}/raw/complaints.jsonl" 2>/dev/null || echo 0)
    total=$((total + n))
    printf "  %-35s %6d / %6d\n" "$model_key" "$n" "$count"
done
printf "  %-35s %6d / %6d\n" "TOTAL" "$total" "50000"