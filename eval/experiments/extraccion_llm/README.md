# E4 · Extracción LLM vs ground truth

¿Recupera el extractor la estructura (categoría, urgencia, ubicación, riesgos, elementos)
que el generador puso en la queja? Diseño, matching N≠M y métricas por campo.

- Diseño original: `../../README.md`
- Doc extendido: `../../docs/experimento_extraccion_llm.md`

## Mapa de artefactos

| Tipo | Ruta | Nota |
|---|---|---|
| Código | `./run_experiment.py`, `./evaluate.py`, `./config.eval.yaml` | generate → extract → evaluate |
| Datos | `../../results/dataset/` | corpus 854 quejas + GT |
| Datos | `../../results/extractions/<extractor>/annotated/extractions.jsonl` | 10 extractores: kimi-k2.6, kimi-k2.7-code, qwen3.5-397b, qwen3.6-35b, glm-5.1, gemma4-31b, minimax-m3{,-a,-b}, qwen3_local |
| Resultados | `../../results/metrics.json`, `../../results/report.md` | ⚠️ gitineados por el .gitignore actual — candidato a versionar |
| Logs | `../../results/extract-all.log`, `extractions/*/log.txt` | |

Reproducibilidad: fases 1–2 (generar/extraer) dependen de gateway/APIs; fase 3
(`--evaluate`) es determinista sobre los JSONL persistidos.
