# E2 · Datasets a escala

Corpora de quejas sintéticas con ground truth a distintas escalas, generados con el
prompt de producción (iter 6 en los posteriores a su adopción).

- Doc extendido: `../../docs/experimento_datasets.md`

## Variantes

| Variante | Estado | Ubicación |
|---|---|---|
| `dataset/` (854 quejas, 3 generadores) | ✅ en repo | `../../results/dataset/` |
| `dataset-5k/` (500 × 12 modelos, selección iter 1) | ✅ en repo | `../../results/dataset-5k/` |
| `dataset-10k/` (~1000 × 10 modelos, selección iter 2) | ⚠️ parcial (warmup timeouts) | `../../results/dataset-10k/` |
| `dataset-50k/` | ❌ eliminado (housekeeping 2026-10; "do not use") | — |
| `dataset-bench/` | ❌ eliminado (housekeeping 2026-10; one-off) | — |

## Mapa de artefactos

| Tipo | Ruta |
|---|---|
| Código | `./generate_5k_dataset.py`, `./generate_10k_dataset.py`, `./generate_5k.sh`, `./run_10k_tonight.sh`, `./extract_eval_5k.py` |
| Exploración | `./explore_10k.py` → `../../results/dataset-10k_exploration.json`, `explore_10k.log` |
| Logs | `../../results/dataset-5k.log`, `../../results/dataset-10k_scheduled.log`, `dataset-10k/generation_10k.log` |

Reproducibilidad: procedimiento determinista (seed, escenarios presampleados) pero
dependiente de modelos del gateway y horas de GPU.
