#!/usr/bin/env bash
# Cross-perplexity unified (consolidates v1-v4):
#   - content-only NLL, temp=0, neutral penalties (diagonal ~= 1)
#   - prompt_logprobs via patched llama.cpp (1 req/text), fallback incremental
# Requires the patched server (see eval/llama_cpp_prompt_logprobs.patch):
#   server bin: /home/dani/llama.cpp-xppl/build/bin/llama-server
# Results: eval/results/diversity/cross_perplexity_unified.json
#          eval/results/diversity/xppl_gen_cache_unified.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 -u eval/cross_perplexity_unified.py \
  --models qwen3-0.6b llama-3.2-1b-instruct gemma-2-2b-it falcon3-1b-instruct \
  --n-scenarios 5 "$@"
