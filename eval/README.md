# eval/ — Evaluación experimental de miUrba

Este documento es el **hilo conductor**: por qué existe cada experimento, en qué orden
se ejecutaron, qué respondieron y qué queda abierto. El detalle de cada uno está en
`experiments/<experimento>/README.md` (mapa código↔datos) y `docs/` (narrativa
extendida).

---

## 1. El sistema y su bucle de evaluación

miUrba genera **quejas vecinales sintéticas** a partir de un escenario estructurado
(categoría, urgencia, ubicación, riesgos, elementos) y después **extrae** esa
estructura desde el texto con un LLM. Todo el esfuerzo de evaluación gira alrededor de
una pregunta central:

> **¿Conserva el sistema la información estructural de la queja de extremo a extremo?**

Eso se descompone en cuatro sub-preguntas, y cada experimento ataca una:

1. ¿Qué modelos pueden escribir las quejas? → **selección de generadores**
2. ¿Cómo de bien extrae el extractor la estructura? → **evaluación extractor vs GT**
3. ¿Cómo hacemos que el texto conserve mejor la estructura? → **iteraciones del prompt**
4. ¿Los generadores son genuinamente distintos entre sí? → **caracterización**

## 2. Cadena causal (orden real de ejecución)

```
Cribado de 89 modelos del gateway (ago 2026)
        │
        ▼
E1a. SELECCIÓN iter 1 (TF-IDF) ──► 10 generadores ──► dataset-5k
        │
        ▼
E4. EXTRACCIÓN VLM vs GT (854 quejas) ──► identifica campos que fallan
        │
        ▼
E5. ITERACIONES DEL PROMPT (iter 1-8) ──► gana iter 6 (F1 0.973) ──► prompt de producción
        │
        ▼
E3. CARACTERIZACIÓN multidimensional (4 familias de métricas, 87 generadores)
        │
        ▼
E1b. SELECCIÓN iter 2 (embeddings 7 jueces + 1 por linaje) ──► 10 nuevos generadores
        │                                                          (solo 1 repite)
        ▼
     dataset-10k (parcial)
```

## 3. Los experimentos

| # | Experimento | Pregunta | Resultado principal | Carpeta |
|---|---|---|-----|---------|
| E1+E3 | Selección y caracterización de generadores | ¿Qué generadores usar? (2 iteraciones: TF-IDF → fingerprints) | 10 seleccionados por linajes con embeddings; dos ejes ortogónicos estilo/semántica | `experiments/seleccion_modelos/` |
| E2 | Datasets a escala | ¿Con qué corpus evaluamos? | 854 + 5K completos; 10K parcial; 50K descartado | `experiments/datasets/` |
| E4 | Extracción LLM vs GT | ¿Fidelidad del extractor? | P/R/F1 por campo y por par generador×extractor; categoría ≫ subcategoría; urgencia sobreestimada | `experiments/extraccion_llm/` |
| E5 | Iteraciones del prompt | ¿Qué prompt conserva mejor la estructura? | iter 6: F1 0.973, exact match 0.944 → producción | `experiments/iteraciones_prompt/` |

## 4. Timeline

| Fecha | Hito |
|---|---|
| ~21 ago 2026 | Runner de iteraciones; variantes iter 2-4 |
| 26 ago 2026 | Cribado 89 modelos + selección TF-IDF + corpus de diversidad |
| 10 ago–sep | dataset-50k (descartado) y datasets 5k |
| 3 sep 2026 | Embeddings 3 jueces; cross-PPL gateway con llama.cpp parcheado |
| sep 2026 | Estilometría + rigidez; iteraciones 5-8; **iter 6 adoptada** (prompt de producción vigente) |
| 7-9 sep 2026 | dataset-10k (parcial); auditoría de afirmaciones (`AFIRMACIONES_*`, en `experiments/seleccion_modelos/`) |
| sep 2026 | Informe de similaridad (87 generadores, 4 métricas) |
| 19-20 oct 2026 | Housekeeping: reorganización por experimentos, recuperación de variantes de prompt |

## 5. Gaps identificados (al hacer este ejercicio)

### Gaps de diseño no ejecutado
- **G1 · Imágenes**: la hipótesis 5 de E4 (extraer con fotos generadas por FLUX) nunca
  se ejecutó. El extractor multimodal se eligió para eso; quedó aplazado indefinidamente.
- **G2 · Fidelidad del generador no medida directamente**: no hay ningún experimento
  que puntúe "¿el texto refleja el escenario GT?" con juez LLM/humano. E4 lo mide al
  revés (extractor→GT) y E5 usa esa medida como proxy. Candidato a experimento E7,
  reusando `diversity/gen_*/` y `dataset-10k/`.
- **G3 · Selección iter 2 sin validación downstream**: los 10 modelos elegidos con la
  métrica buena (embeddings+linajes) alimentaron el dataset-10k, pero este quedó
  incompleto (warmup timeouts) — la segunda selección nunca se validó end-to-end como
  sí lo estuvo la primera (5k → E4).

### Gaps de cobertura
- **G4 · Todo es sintético**: ningún run sobre quejas reales ni validación humana de
  naturalidad. La validez ecológica del pipeline está sin medir.
- **G5 · Extracción de un solo problema dominante**: el GT tiene ~1.17 problemas/queja;
  los multi-problema (~15% del generador) apenas tienen peso en las métricas.
- **G6 · Evaluator-dependence**: E4 usa mayormente un extractor por texto; los pares
  generador×extractor cubiertos varían entre `metrics.json` (raíz) y los `iter_*/`,
  sin una matriz completa común.

### Gaps de infraestructura/proceco
- **G7 · 50K abandonado sin diagnóstico escrito** (marcado "do not use" en TASK_ITER,
  fichero que ya no existe; ver C6 en `AFIRMACIONES_VERIFICACION.md`).
- **G8 · Recolección dependiente del gateway**: los runs grandes (10K, cross-PPL full)
  no son reproducibles sin la VM de GPU con los modelos exactos de sep-2026 cargables.
- **G9 · E4 `metrics.json`/`report.md` fueron gitineados históricamente**: ahora se
  versionan, pero no hay snapshot firmado de cuándo se generaron exactamente.

## 6. Convenciones

- Los datos viven en `results/`; los scripts y configs de cada experimento en su
  carpeta de `experiments/`; la narrativa extendida en `docs/`.
- Los logs de ejecución se archivan **fuera del repo**: `../mi-urba-eval-logs/`.
- Los scripts archivados (superseded) están en `../mi-urba-eval-old/` (fuera del repo).
- Semillas fijas y configs versionadas donde aplica; ver el README de cada experimento
  para su reproducibilidad exacta.
