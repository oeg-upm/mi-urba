# eval/old/ — archived scripts

One-off and superseded evaluation scripts, kept for provenance and possible
re-runs. Nothing in the active pipeline depends on them.

Moved here (repo cleanup, Sep 2026). Two adjustments were applied so they
still run from their new location:

- `cross_perplexity{,_v2,_v3,_v4}.py`: repo root is now `parents[2]`.
- (2026-10) `env_utils.py` eliminado: el loader de API key está inlineado en cada script.
- Probes/measurement scripts: API-key loading is inlined (repo `.env`, env vars win).

## Contents

| Script | Why superseded / one-off |
|---|---|
| `cross_perplexity{,_v2,_v3,_v4}.py` | Exploratory iterations; consolidated into `eval/cross_perplexity_unified.py` (its docstring documents which decision comes from each version) |
| `test_all_generators.py`, `test_remaining_generators.py` | First sweep probing local_llama gateway models as generators |
| `test_retest_{generators,problems,isolated,final}.py` | Follow-up retests of weak/failed/new models |
| `measure_load_times.py`, `measure_load_remaining.py` | One-off model load-time measurements |
| `load_all_models.sh` | One-off warmup/loader for all gateway models |
| `run_50k_generation.sh` | 50K dataset generation (dataset marked "do not use" in TASK_ITER.md) |
| `run_bench_generation{,_v2}.sh` | One-off benchmark generation runs |
| `recover_missing.sh` | One-off recovery of missing 50K complaints |
| `run_iterations_{234,34,567}.sh`, `run_iter4_after_3.sh`, `wait_and_run_67.sh` | Batch launchers for specific prompt-iteration rounds; the generic runner `eval/run_iteration.sh` remains active |
