#!/usr/bin/env bash
# Giant cross-perplexity matrix over the llama-swap gateway (gpus-vm).
# Builds the model list from the gateway's registered -xppl variants,
# excluding architectures removed from modern llama.cpp (mixtral).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GATEWAY="${GATEWAY:-http://192.168.10.11:9292}"
N_SCENARIOS="${N_SCENARIOS:-3}"
REQ_TIMEOUT="${REQ_TIMEOUT:-900}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-4000}"
OUT="${OUT:-eval/results/diversity/cross_perplexity_gateway_full.json}"
CACHE="${CACHE:-eval/results/diversity/xppl_gen_cache_gateway_full.json}"

# model list: every -xppl variant except removed-arch (mixtral) ones
MODELS=$(curl -sf "$GATEWAY/v1/models" | python3 -c "
import json, sys
ids = [m['id'] for m in json.load(sys.stdin)['data']]
excl = {'mixtral-8x7b-instruct-xppl', 'nous-hermes-2-mixtral-8x7b-xppl'}
print(' '.join(sorted(i for i in ids if i.endswith('-xppl') and i not in excl)))
")
N=$(echo $MODELS | wc -w)
echo "models: $N | scenarios: $N_SCENARIOS | req_timeout: $REQ_TIMEOUT"
echo "output: $OUT"

exec python3 -u eval/cross_perplexity_unified.py \
  --models $MODELS \
  --n-scenarios "$N_SCENARIOS" \
  --gateway "$GATEWAY" \
  --req-timeout "$REQ_TIMEOUT" \
  --max-gen-tokens "$MAX_GEN_TOKENS" \
  --output "$OUT" \
  --gen-cache "$CACHE" \
  "$@"
