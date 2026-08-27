#!/usr/bin/env bash
# Cross-perplexity v4: patched llama.cpp with prompt_logprobs, 1 request per text.
# Requires the patched server (see eval/llama_cpp_prompt_logprobs.patch):
#   server bin: /home/dani/llama.cpp-xppl/build/bin/llama-server
# Results: eval/results/diversity/cross_perplexity_v4.json
#          eval/results/diversity/xppl_gen_cache_v4.json
set -euo pipefail

cd /home/dani/xppl
python3 -u eval/cross_perplexity_v4.py \
  --models qwen3-0.6b llama-3.2-1b-instruct gemma-2-2b-it falcon3-1b-instruct \
  --n-scenarios 5 "$@"