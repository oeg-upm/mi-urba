# Experimento de iteraciones del prompt de generación

> Fuente: `eval/experiments/iteraciones_prompt/run_iteration.sh`, `eval/experiments/iteraciones_prompt/apply_prompt_variant.sh`,
> `mi-urba-eval-old/run_iterations_*.sh` (fuera del repo). Resultados: `eval/results/iter_{1..8}/`
> (report.md, metrics.json, logs). Resultado final aplicado a producción:
> commit `7257145` ("optimized prompts — structured guidance + risk/urgency mapping").

---

## 1. La pregunta

El experimento de extracción LLM (`experimento_extraccion_llm.md`) identificó qué
campos fallan al recuperar la estructura desde el texto. La pregunta de este
experimento es la inversa y práctica:

> **¿Qué variante del prompt del generador produce textos de los que el extractor
> recupera mejor la estructura original?**

Es un proceso de **optimización del prompt del generador usando la extracción como
métrica objetiva**: no se evalúa si el texto "suena bien", sino si conserva la
información estructural (categoría, urgencia, riesgos, elementos) tras pasar por el
extractor.

## 2. Diseño

### 2.1 Matriz generadores × extractores

Cada iteración cruza **4 generadores** con **4 extractores** (16 combinaciones):

| Generadores (AWQ, cuantizados) | Extractores |
|---|---|
| gemma4-31b-awq | gemma4-31b |
| qwen3.6-35b-awq | qwen3.6-35b |
| gemma4-12b-awq | gemma4-12b |
| qwen3-4b-instruct-awq | qwen3-4b |

Cruzar con varios extractores controla que la mejora venga del texto y no de la
simpatía de un extractor concreto por el estilo de un generador. En iter_8 la matriz
cambia: gemma4-12b pasa a ser el extractor principal (4 generadores × 1 extractor).

### 2.2 Procedimiento por iteración

`bash eval/experiments/iteraciones_prompt/run_iteration.sh <N>` ejecuta, por iteración:

1. **Aplicar la variante de prompt** (`apply_prompt_variant.sh <iter>`): copia
   `prompts_iter<N>.py.bak` sobre `miurba-generator/src/prompts.py` (y el equivalente
   del extractor si la iteración también toca el prompt de extracción). Tras el bloque
   de iteraciones se restaura la variante elegida como production (iter 2 fue el
   "current" durante las rondas 5-7).
2. **Generar** N quejas por generador (iter 1: 1000/generador; iter 2-8: 250 nuevas
   por ronda), workers=32, seed=42, warmup con backoff 20s→60s→120s.
3. **Extraer** con cada extractor.
4. **Evaluar** las 16 combinaciones → `metrics.json` + `report.md` en
   `eval/results/iter_<N>/`.

Se generan **quejas nuevas en cada ronda** (no se reusan), de modo que la mejora no
pueda ser sobreajuste a un dataset fijo.

## 3. Las variantes de prompt

| Iter | Foco de la variante | N quejas |
|------|---------------------|----------|
| 1 | Baseline (prompt de producción vigente) | 999/generador |
| 2 | Urgencia con palabras clave; subcategoría explícita; formato de elements | 250 |
| 3 | Reducir falsos positivos + mejorar risks con mapeo explícito | 250 |
| 4 | Urgencia al inicio + subcategoría en la primera mención | 250 |
| 5-7 | Variantes consecutivas (lanzadas en bloque por `run_iterations_567.sh`) | 250 |
| 6 | **Ganadora**: guía estructurada + mapeo de riesgos/urgencia | 250 |
| 8 | Validación cruzada con extractor principal gemma4-12b | 250 |

## 4. Resultados

Resumen de F1 de la mejor combinación por iteración (datos completos en
`iter_<N>/report.md`):

| Iter | Mejor F1 | Combinación | Observación |
|------|----------|-------------|-------------|
| 1 | 0.929 | gemma4_31b × gemma4_12b | baseline |
| 2 | 0.939 | gemma4_31b × gemma4_12b | +0.01 sobre baseline |
| 3 | 0.931 | gemma4_31b × qwen3_6_35b | ≈ baseline |
| 4 | 0.940 | gemma4_31b × gemma4_12b | ≈ iter 2 |
| 5 | 0.934 | gemma4_31b × gemma4_12b | ≈ |
| **6** | **0.973** | gemma4_31b × gemma4_12b | **salto claro; exact match 0.944; FP 15** |
| 7 | 0.949 | gemma4_31b × gemma4_12b | retrocede algo respecto a 6 |
| 8 | 0.968 | gemma4_31b × gemma4_12b | valida iter 6 con extractor principal cambiado |

Patrones:

- El generador **gemma4_31b_awq** gana en todas las rondas: produce los textos más
  extraíbles del cuarteto.
- **Iter 6 es el ganador inequívoco** (+0.034 F1 sobre la mejor anterior, +0.016 exact
  match). Su prompt se adoptó como producción (commit `7257145`) y es el que usa
  `generate_diversity_dataset.py` ("prompt de producción iter 6").
- La progresión no es monotónica (7 < 6): la variante 6 acertó el equilibrio entre
  fidelidad estructural y naturalidad del texto; la 7 sobre-corrigió.

## 5. Notas operativas

- Logs completos por iteración: `results/iter_<N>/master_log.txt` y `iter_N_run.log`.
- Las rondas se lanzaron en bloques con scripts one-off (`mi-urba-eval-old/run_iterations_234.sh`,
  `mi-urba-eval-old/run_iterations_34.sh`, `run_iter4_after_3.sh`, `wait_and_run_67.sh` (fuera del repo) —
  este último esperaba a que el servidor de inferencia se recuperase).
- Las variantes `prompts_iter{2..7}.py.bak` (generador y extractor) están versionadas
  desde el housekeeping de 2026-10, recuperadas de un checkout hermano; verificado que
  `prompts_iter6.py.bak` del generador es byte-idéntico al `prompts.py` de producción.

## 6. Conexión con otros experimentos

- **Input**: los fallos por campo detectados en `experimento_extraccion_llm.md`.
- **Output hacia datasets**: el prompt iter 6 es el que generan todos los corpus
  posteriores (diversidad, 5K, 10K).
- **Output hacia selección**: los 4 generadores AWQ de esta matriz provienen de la
  primera selección de generadores; la segunda selección (ver
  `experimento_seleccion_generadores.md`) reemplaza este cuarteto.
