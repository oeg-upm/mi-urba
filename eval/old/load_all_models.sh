#!/bin/bash
# Load each vLLM model, measure load time (no timeout)

cd /home/dani/miUrba/mi-urba

MODELS=(
    "qwen3.6-35b-awq"
    "gemma4-31b-awq"
    "gemma4-26b-awq"
    "qwen3.5-27b-awq"
    "gemma4-12b-awq"
    "qwen3.5-4b-awq"
    "qwen3-4b-instruct-awq"
)

echo "Modelo                    | Load time"
echo "--------------------------|----------"

RESULTS_FILE="eval/results/dataset-bench/load_times.txt"
> "$RESULTS_FILE"

for model_id in "${MODELS[@]}"; do
    echo -n "$model_id" | tee -a "$RESULTS_FILE"
    start=$(date +%s)
    
    # Keep trying until we get a valid response
    while true; do
        response=$(curl -s --max-time 600 http://192.168.10.11:9292/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model_id}\",\"messages\":[{\"role\":\"user\",\"content\":\"Di OK\"}],\"max_tokens\":5,\"enable_thinking\":false,\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>&1)
        
        if echo "$response" | grep -q '"choices"'; then
            end=$(date +%s)
            elapsed=$((end - start))
            mins=$((elapsed / 60))
            secs=$((elapsed % 60))
            printf " | %dm %ds\n" "$mins" "$secs" | tee -a "$RESULTS_FILE"
            break
        fi
        echo -n "." >&2
        sleep 10
    done
done

echo ""
echo "=== Tabla de load times ==="
cat "$RESULTS_FILE"