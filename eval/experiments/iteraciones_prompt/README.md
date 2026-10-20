# E5 · Iteraciones del prompt de generación

Optimización del prompt del generador usando la extracción como métrica objetiva.
8 rondas (iter 1–8), matriz 4 generadores AWQ × 4 extractores. **Ganadora: iter 6**
(F1 0.973, FP 15), adoptada en producción (commit `7257145`).

- Doc extendido: `../../docs/experimento_iteraciones_prompt.md`

## Mapa de artefactos

| Tipo | Ruta | Nota |
|---|---|---|
| Código | `./run_iteration.sh` | runner genérico (generate × extract × evaluate) |
| Código | `./apply_prompt_variant.sh` | aplica `prompts_iter<N>.py.bak` sobre prompts.py |
| Código (archivado) | `mi-urba-eval-old/run_iterations_*.sh` (fuera del repo), `mi-urba-eval-old/run_iter4_after_3.sh` (fuera del repo), `mi-urba-eval-old/wait_and_run_67.sh` (fuera del repo) | lanzadores one-off |
| Resultados | `../../results/iter_{1..8}/report.md` + `metrics.json` | evidencia por ronda |
| Datos crudos | `../../results/iter_*/gen_*/, ext_*/` | ⚠️ gitineados (efímeros); candidatos a poda |
| Logs | `../../results/iter_*_run.log`, `iter_4_wait.log`, `iterations_567.log`, `iter_*/master_log.txt` | |

## Reproducibilidad recuperada (housekeeping 2026-10)

Las variantes `prompts_iter{2..7}.py.bak` (generador y extractor) se han recuperado de
un checkout hermano (`/backup/pc1/data/miUrba2/mi-urba/`, idénticas a las de
`/home/dani/xppl/`) y committeadas en el repo. Verificación: `prompts_iter6.py.bak`
del generador es byte-idéntico al `prompts.py` de producción. El experimento vuelve a
ser reproducible.
