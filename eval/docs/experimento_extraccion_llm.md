# Experimento de extracción LLM vs ground truth

> Fuente: `eval/README.md` (diseño), `eval/experiments/extraccion_llm/run_experiment.py`, `eval/experiments/extraccion_llm/evaluate.py`,
> `eval/experiments/extraccion_llm/config.eval.yaml`. Resultados: `eval/results/metrics.json`, `eval/results/report.md`,
> `eval/results/extractions/<modelo>/annotated/extractions.jsonl`.

---

## 1. La pregunta

El sistema genera quejas vecinales sintéticas a partir de un **escenario conocido**:
categoría, urgencia, ubicación, riesgos, elementos afectados. Sobre el texto de la queja
actúa un **extractor** (LLM) que intenta recuperar esa estructura. La pregunta del
experimento es:

> **¿Cuánto se parece lo que el extractor recupera del texto a la ground truth con la
> que se generó el texto?**

Esto es un test de **fidelidad del canal completo**: generador (estructura → texto) y
extractor (texto → estructura). Si la extracción fuese perfecta, el par generador+extractor
conserva toda la información estructural.

## 2. Hipótesis a validar

1. La **categoría principal** se recupera bien (está explícita o casi en el texto).
2. La **subcategoría** es más difícil (el vecino no usa terminología técnica).
3. La **urgencia tiende a sobreestimarse** (el vecino exagera respecto al GT).
4. El extractor detecta **problemas adicionales** no contemplados en el GT
   (falsos positivos) porque lee matices mencionados de pasada.
5. Las **imágenes** mejorarían la extracción de elementos y riesgos (diferido: FLUX es
   lento, ~3 min/imagen; se plantea como experimento separado).
6. El **modelo de generación afecta** la extracción: textos más realistas (kimi) son más
   fáciles de extraer que textos más rígidos (glm).

## 3. Glosario

| Término | Definición |
|---|---|
| **GT (ground truth)** | Estructura de 1..N problemas con la que se generó la queja: categoría principal/subcategoría, urgencia, ubicación (zona + detalle), riesgos, elementos afectados |
| **Extracción** | Lo que el LLM recupera del texto, con el mismo esquema |
| **Matching** | Emparejamiento entre extracciones del GT (N) y del extractor (M); N ≠ M en general |
| **Falso negativo** | Extracción del GT sin pareja → penaliza recall |
| **Falso positivo** | Extracción del extractor sin pareja → penaliza precision |

## 4. Protocolo

### 4.1 Fases

```
FASE 1 — Generar   3 modelos × 30 quejas = 90 quejas → complaints.jsonl + ground_truth.jsonl
FASE 2 — Extraer   LLM sobre cada queja → extractions.jsonl
FASE 3 — Evaluar   matching + métricas → metrics.json + report.md
```

Ejecución: `python eval/experiments/extraccion_llm/run_experiment.py [--generate|--extract|--evaluate]`.
Factores: generador ∈ {kimi-k2.6, qwen3_local, glm-5.1}; extractor único (kimi-k2.6);
sin imágenes. Seed fija (42) y config versionada (`config.eval.yaml`) para reproducibilidad.
Cada run guarda logs completos del generador y del extractor.

### 4.2 Matching GT vs extracción

No hay alineación 1:1: el GT trae N problemas y el extractor puede producir M ≠ N.

1. Emparejar por **categoría principal** (GT `fontanería` ↔ pred `fontanería`).
2. Si varias extracciones comparten categoría principal, desempatar por **subcategoría**,
   y en última instancia por similitud global.
3. GT sin pareja → **falso negativo**. Predicción sin pareja → **falso positivo**.

### 4.3 Métricas por campo (sobre extracciones emparejadas)

| Campo | Métrica | Criterio |
|---|---|---|
| category.principal | exact match | GT == pred |
| category.subcategory | exact match | GT == pred |
| category completa | exact match | principal + subcategory |
| urgency | exact match | GT == pred |
| location.zone | exact match | GT == pred |
| location.detail | substring | texto libre: GT ⊆ pred o pred ⊆ GT |
| risks | Jaccard | \|GT ∩ pred\| / \|GT ∪ pred\| |
| affected_elements | precision/recall | match por (principal, subcategory) |

### 4.4 Métricas por queja y agregadas

- **Precision**: extracciones correctas / totales del extractor.
- **Recall**: extracciones correctas / totales del GT.
- **F1** y **exact match** (todos los campos correctos en todas las extracciones).
- Todo lo anterior se agrega **por modelo de generación**, para comparar qué generador
  produce textos más extraíbles (hipótesis 6).

## 5. Resultados

### 5.1 Run principal (854 quejas con GT)

`results/metrics.json` + `results/report.md` agregan por par generador×extractor
(incluye self-eval `qwen3_local`). Los números completos están en el report.

### 5.2 Extracciones por extractor (results/extractions/)

| Extractor | Variante |
|---|---|
| kimi-k2.6, kimi-k2.7-code | gateway |
| qwen3.5-397b, qwen3.6-35b | gateway |
| glm-5.1, gemma4-31b | gateway |
| minimax-m3 (+ variantes a, b) | gateway |
| qwen3_local | local |

Cada uno deja `annotated/extractions.jsonl` en su directorio.

## 6. Verificación de hipótesis (síntesis)

Los patrones confirmados en los reports: categorías principales casi perfectas
(>0.95 accuracy), subcategoría como el campo más difícil, ligera sobreestimación de
urgencia, y presencia consistente de falsos positivos (el extractor extrae ~1.3-1.6
problemas por queja frente a ~1.17 del GT — la hipótesis 4).

## 7. Reproducibilidad

- Seed fija en generación (`seed_base: 42`).
- `config.eval.yaml` versionado en el repo (merge de `config.yaml` +
  `config.neuralwatt.yaml` + llama-swap).
- Logs completos por run.
- `results/` se commitea (no está en .gitignore).

## 8. Conexión con otros experimentos

- El dataset de 854 quejas de este experimento es una variante temprana de la línea de
  datasets (ver `experimento_datasets.md`).
- Las conclusiones aquí (qué campos fallan y por qué) son el input directo de las
  **iteraciones del prompt** (`experimento_iteraciones_prompt.md`).
