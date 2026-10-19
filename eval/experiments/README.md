# eval/experiments — Experimentos identificados

Una carpeta por experimento. E1 (selección) y E3 (caracterización) son **fases del mismo
experimento** — dos iteraciones de la pregunta "¿qué generadores usar?" — y por eso
comparten carpeta.

Los scripts y configs de cada experimento **viven dentro de su carpeta** (rutas
ajustadas tras el traslado); los datos permanecen en `eval/results/` (los scripts los
referencian y el `.gitignore` se configura por rutas). Cada README documenta el mapa
código↔datos↔resultados del experimento. El loader de API key está inlineado en cada script que habla con el gateway (sin módulo compartido).

| Carpeta | Experimento | Fases |
|---|---|---|
| `seleccion_modelos/` | ¿Qué generadores usar? — selección (iter 1 TF-IDF) y caracterización por fingerprints (iter 2 embeddings+linajes) | E1 + E3 |
| `datasets/` | Corpora a escala: 854, 5K, 10K (50K y bench eliminados en housekeeping) | E2 |
| `extraccion_llm/` | Fidelidad del extractor LLM frente a ground truth | E4 |
| `iteraciones_prompt/` | Optimización del prompt del generador en 8 rondas (ganadora: iter 6) | E5 |

Cadena causal: selección → datasets → extracción → iteraciones del prompt (adopta
iter 6) → caracterización multidimensional → re-selección por linajes → dataset 10K.

Documentación extendida por experimento en `eval/docs/`.
