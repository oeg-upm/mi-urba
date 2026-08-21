#!/bin/bash
# miUrba — Iteration runner
# Generates 1000 complaints per generator model (4 models),
# extracts with 4 extractor models on each dataset,
# evaluates all 16 combinations.
#
# Usage: bash eval/run_iteration.sh <iteration_number>

set -e
# Resolve repo root relative to this script (works from any CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

ITER=${1:-1}
N=${2:-1000}
WORKERS=32
SEED=42
WARMUP_TIMEOUT=900

GEN_MODELS=(
    "gen_gemma4_31b_awq|local_llama/gemma4-31b-awq"
    "gen_qwen3_6_35b_awq|local_llama/qwen3.6-35b-awq"
    "gen_gemma4_12b_awq|local_llama/gemma4-12b-awq"
    "gen_qwen3_4b_instr_awq|local_llama/qwen3-4b-instruct-awq"
)

EXT_MODELS=(
    "extract_gemma4_12b|local_llama/gemma4-12b-awq"
)

BASEDIR="eval/results/iter_${ITER}"
mkdir -p "$BASEDIR"

# ── Helper: warmup a model ──
# Retries until the model responds. Distinguishes server-down (curl fails to
# connect) from model-loading (curl succeeds but returns error/empty).
# Waits up to WARMUP_TIMEOUT seconds, with growing backoff when the server is
# unreachable (likely crashed / restarting).
warmup() {
    local model_id=$1
    local max_wait=${WARMUP_TIMEOUT}
    local elapsed=0
    local attempt=0
    local server_down_streak=0
    while [ "$elapsed" -lt "$max_wait" ]; do
        attempt=$((attempt + 1))
        # curl returns non-zero on connection failure / timeout
        response=$(curl -s --max-time 60 http://192.168.10.1:4000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${LLAMA_GATEWAY_API_KEY}" \
            -d "{\"model\":\"${model_id}\",\"messages\":[{\"role\":\"user\",\"content\":\"Di OK\"}],\"max_tokens\":5,\"enable_thinking\":false,\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>&1)
        curl_rc=$?
        if echo "$response" | grep -q '"choices"'; then
            echo "  [$(date '+%H:%M:%S')] $model_id loaded (attempt $attempt)"
            return 0
        fi
        if [ "$curl_rc" -ne 0 ]; then
            # curl failed to connect / timed out → server likely down
            server_down_streak=$((server_down_streak + 1))
            # Grow backoff when server is down: 20s → 60s after repeated failures
            wait=20; [ "$server_down_streak" -ge 3 ] && wait=60
            [ "$server_down_streak" -ge 6 ] && wait=120
            echo "  [$(date '+%H:%M:%S')] $model_id: server unreachable (attempt $attempt, streak $server_down_streak) — retry in ${wait}s"
            sleep "$wait"
            elapsed=$((elapsed + wait))
        else
            # curl connected but got error JSON (model loading) → short wait
            server_down_streak=0
            sleep 20
            elapsed=$((elapsed + 20))
        fi
    done
    echo "  [$(date '+%H:%M:%S')] $model_id FAILED to load after ${elapsed}s"
    return 1
}

# ═══════════════════════════════════════════════════════════════
# PHASE 1: GENERATE
# ═══════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════"
echo "  ITERATION $ITER — PHASE 1: GENERATION"
echo "═══════════════════════════════════════════════════"

for entry in "${GEN_MODELS[@]}"; do
    IFS='|' read -r gen_key gen_id <<< "$entry"
    outdir="${BASEDIR}/gen_${gen_key}"
    logfile="${outdir}/gen_log.txt"
    
    # Skip if already complete
    if [ -f "${outdir}/raw/complaints.jsonl" ]; then
        n=$(grep -c . "${outdir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
        if [ "$n" -ge "$N" ]; then
            echo "[$(date '+%H:%M:%S')] $gen_key: SKIP ($n/$N complete)"
            continue
        fi
    fi
    
    echo "[$(date '+%H:%M:%S')] $gen_key: warmup..."
    warmup "$gen_id" || continue
    
    mkdir -p "$outdir"
    echo "[$(date '+%H:%M:%S')] $gen_key: generating $N complaints..."
    
    cd miurba-generator
    .venv/bin/python cli.py generate \
        -n $N --model "$gen_key" --config ../eval/config.eval.yaml \
        --output-dir "../$outdir" --seed $SEED --workers $WORKERS --timeout 300 \
        > "../$logfile" 2>&1 || true
    cd ..
    
    n=$(grep -c . "${outdir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] $gen_key: $n/$N generated"
    sleep 5
done

# ═══════════════════════════════════════════════════════════════
# PHASE 2: EXTRACT
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ITERATION $ITER — PHASE 2: EXTRACTION"
echo "═══════════════════════════════════════════════════"

for gen_entry in "${GEN_MODELS[@]}"; do
    IFS='|' read -r gen_key gen_id <<< "$gen_entry"
    gen_dir="${BASEDIR}/gen_${gen_key}"
    
    if [ ! -f "${gen_dir}/raw/complaints.jsonl" ]; then
        echo "  SKIP $gen_key (no complaints)"
        continue
    fi
    
    for ext_entry in "${EXT_MODELS[@]}"; do
        IFS='|' read -r ext_key ext_id <<< "$ext_entry"
        ext_dir="${gen_dir}/ext_${ext_key}"
        ext_file="${ext_dir}/annotated/extractions.jsonl"
        
        # Skip if already complete
        if [ -f "$ext_file" ]; then
            n=$(grep -c . "$ext_file" 2>/dev/null || echo 0)
            gen_n=$(grep -c . "${gen_dir}/raw/complaints.jsonl" 2>/dev/null || echo 0)
            if [ "$n" -ge "$gen_n" ]; then
                echo "[$(date '+%H:%M:%S')] $gen_key × $ext_key: SKIP ($n done)"
                continue
            fi
        fi
        
        echo "[$(date '+%H:%M:%S')] $gen_key × $ext_key: warmup..."
        warmup "$ext_id" || continue
        
        mkdir -p "$ext_dir/annotated"
        echo "[$(date '+%H:%M:%S')] $gen_key × $ext_key: extracting..."
        
        cd miurba-extractor
        .venv/bin/python cli.py extract \
            --config ../eval/config.eval.yaml --model "$ext_key" \
            --raw-dir "../${gen_dir}/raw" --annotated-dir "$ext_dir/annotated" \
            --timeout 300 --max-retries 3 --workers $WORKERS \
            > "../${ext_dir}/ext_log.txt" 2>&1 || true
        cd ..
        
        # Fix path if extractor wrote to wrong location
        if [ ! -f "$ext_file" ] && [ -f "miurba-extractor/${ext_file}" ]; then
            mkdir -p "$ext_dir/annotated"
            mv "miurba-extractor/${ext_file}" "$ext_file"
            rm -rf "miurba-extractor/eval"
        fi
        
        n=$(grep -c . "$ext_file" 2>/dev/null || echo 0)
        echo "[$(date '+%H:%M:%S')] $gen_key × $ext_key: $n extracted"
        sleep 5
    done
done

# ═══════════════════════════════════════════════════════════════
# PHASE 3: EVALUATE
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ITERATION $ITER — PHASE 3: EVALUATION"
echo "═══════════════════════════════════════════════════"

cd "$ROOT_DIR"
python3 << PYEOF
import sys, json, os
sys.path.insert(0, 'eval')
from evaluate import evaluate, generate_report

ITER = ${ITER}
BASEDIR = f"eval/results/iter_{ITER}"

gen_keys = ["gen_gemma4_31b_awq", "gen_qwen3_6_35b_awq", "gen_gemma4_12b_awq", "gen_qwen3_4b_instr_awq"]
ext_keys = ["extract_gemma4_12b"]

results = []
matrix = {}

for gen_key in gen_keys:
    gen_dir = f"{BASEDIR}/gen_{gen_key}"
    gt_file = f"{gen_dir}/raw/ground_truth.jsonl"
    if not os.path.exists(gt_file):
        continue
    
    for ext_key in ext_keys:
        ext_file = f"{gen_dir}/ext_{ext_key}/annotated/extractions.jsonl"
        if not os.path.exists(ext_file):
            continue
        
        name = f"{gen_key.replace('gen_','')} × {ext_key.replace('extract_','')}"
        result = evaluate(gt_file, ext_file, name)
        results.append(result)
        
        cm = result['complaint_metrics']
        fm = result['field_metrics']
        matrix[(gen_key, ext_key)] = {
            'f1': cm['f1'], 'precision': cm['precision'], 'recall': cm['recall'],
            'em': cm['exact_match_rate'], 'pred_per_q': cm['avg_pred_extractions'],
            'urgency': fm['urgency']['accuracy'],
            'elem_p': fm['affected_elements_precision']['accuracy'],
            'elem_r': fm['affected_elements_recall']['accuracy'],
            'fp': result['match_types'].get('fp', 0),
        }

# Print F1 matrix
print("\n=== F1 MATRIX (generator × extractor) ===\n")
print(f"{'Generator ↓ / Extractor →':<30s}", end="")
for ek in ext_keys:
    print(f" {ek.replace('extract_',''):>15s}", end="")
print()
print("-" * 90)
for gk in gen_keys:
    print(f"{gk.replace('gen_',''):<30s}", end="")
    for ek in ext_keys:
        v = matrix.get((gk, ek), {}).get('f1', 0)
        print(f" {v:>15.4f}", end="")
    print()

# Print key metrics
print("\n=== KEY METRICS (all 16 combinations) ===\n")
print(f"{'Combination':<55s} {'F1':>6s} {'P':>6s} {'R':>6s} {'EM':>6s} {'Urg':>6s} {'ElP':>6s} {'ElR':>6s} {'FP':>5s} {'Ext/q':>6s}")
print("-" * 110)
for r in results:
    cm = r['complaint_metrics']
    fm = r['field_metrics']
    mt = r['match_types']
    print(f"{r['model']:<55s} {cm['f1']:>6.3f} {cm['precision']:>6.3f} {cm['recall']:>6.3f} {cm['exact_match_rate']:>6.3f} {fm['urgency']['accuracy']:>6.3f} {fm['affected_elements_precision']['accuracy']:>6.3f} {fm['affected_elements_recall']['accuracy']:>6.3f} {mt.get('fp',0):>5d} {cm['avg_pred_extractions']:>6.2f}")

# Save metrics
metrics_file = f"{BASEDIR}/metrics.json"
with open(metrics_file, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Save report
report_file = f"{BASEDIR}/report.md"
generate_report(results, report_file)

print(f"\nMetrics: {metrics_file}")
print(f"Report: {report_file}")
PYEOF

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ITERATION $ITER COMPLETE"
echo "═══════════════════════════════════════════════════"