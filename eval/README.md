# Evaluación experimental de miUrba

## Objetivo

Medir cuánto se parece la extracción del LLM a la ground truth generada por el
generador. Es decir: si generamos una queja con un escenario conocido (categoría,
urgencia, ubicación, riesgos, elementos), ¿el extractor recupera esa misma
información del texto?

## Hipótesis

1. **El extractor recupera bien la categoría principal** (fontanería, electricidad, etc.)
   porque está explícita o casi en el texto del vecino.
2. **La subcategoría es más difícil** porque el vecino no usa terminología técnica.
3. **La urgencia tiende a sobreestimarse** (el vecino la exagera respecto al GT).
4. **El extractor detecta problemas adicionales** que el GT no contempla (falsos positivos)
   porque lee matices que el vecino menciona de pasada.
5. **Las imágenes mejoran la extracción** de elementos afectados y riesgos.
6. **El modelo de generación afecta la calidad de la extracción**: textos más realistas
   (kimi) son más fáciles de extraer que textos más rígidos (glm).

## Diseño experimental

### Factores

| Factor | Niveles | Descripción |
|--------|---------|-------------|
| Modelo de generación | `kimi-k2.6`, `qwen3_local`, `glm-5.1` | 3 modelos (NeuralWatt + local) |
| Modelo de extracción | `kimi-k2.6` | Único modelo multimodal (visión) disponible — elegido de cara a la extensión con imágenes; en los runs ejecutados actúa como LLM extractor solo-texto |
| Imágenes | sin imágenes | FLUX es lento; se evalúa por separado (ver más abajo) |
| Nº de problemas | 1 o 2+ | El generador ya mezcla single/multi-problem (~15% multi) |

### Tamaño

- 30 quejas por modelo de generación × 3 modelos = **90 quejas totales**
- Cada queja tiene 1-2 problemas → ~100-110 extracciones GT esperadas

### Pipeline

```
FASE 1 — Generar dataset (90 quejas)
  3 modelos × 30 quejas → complaints.jsonl + ground_truth.jsonl + metadata.jsonl

FASE 2 — Extraer (LLM sobre las 90 quejas)
  kimi-k2.6 → extractions.jsonl

FASE 3 — Evaluar (comparar GT vs extracción)
  matching + métricas → metrics.json + report.md
```

### Fase 3: Matching GT vs Extracción

El GT tiene N extracciones (problemas) por queja. El extractor puede producir
M extracciones (no necesariamente M = N). Hay que emparejar:

1. **Matching por categoría principal**: si el GT tiene `fontanería` y la predicción
   tiene `fontanería`, se emparejan.
2. Si hay múltiples extracciones con la misma categoría principal, se emparejan
   por subcategoría (o por similitud global si no hay coincidencia exacta).
3. Extracciones GT sin match → **falsos negativos** (recall).
4. Extracciones del LLM sin match → **falsos positivos** (precision).

### Métricas

**Por campo (accuracy sobre extracciones emparejadas):**

| Campo | Métrica | Cómo se calcula |
|-------|--------|----------------|
| category.principal | exact match | GT == pred |
| category.subcategory | exact match | GT == pred |
| category (completa) | exact match | principal + subcategory |
| urgency | exact match | GT == pred |
| location.zone | exact match | GT == pred |
| location.detail | substring match | GT ⊆ pred o pred ⊆ GT (texto libre) |
| risks | Jaccard | \|GT ∩ pred\| / \|GT ∪ pred\| |
| affected_elements | precision/recall | match por (principal, subcategory) |

**Por queja (sobre el conjunto de extracciones):**

| Métrica | Definición |
|--------|------------|
| Precision | extracciones correctas / extracciones totales del LLM |
| Recall | extracciones correctas / extracciones GT |
| F1 | 2·P·R / (P+R) |
| Exact match | todos los campos correctos en todas las extracciones |

**Por modelo de generación:**

Las métricas anteriores se agregan por modelo de generación para comparar si
alguno produce textos más fáciles de extraer.

### Estructura de carpetas

```
eval/
├── README.md               # Este documento (diseño + resultados)
├── config.eval.yaml        # Config específica del experimento
├── run_experiment.py       # Script principal: generate → extract → evaluate
├── evaluate.py             # Lógica de matching + métricas
├── results/                # Generado por run_experiment.py
│   ├── dataset/            # complaints.jsonl + ground_truth.jsonl + metadata.jsonl
│   ├── extractions/        # extractions.jsonl (salida del LLM)
│   ├── metrics.json         # Métricas en formato JSON
│   └── report.md           # Informe legible con tablas
```

### Reproducibilidad

- **Seed fija** en la generación (seed_base: 42)
- **Config versionada**: `config.eval.yaml` en el repo
- **Logs completos**: cada run guarda logs del generador y extractor
- **Script único**: `python run_experiment.py` ejecuta todo
- **Resultados en el repo**: `results/` se commitea (no es .gitignored)

### Config del experimento (`config.eval.yaml`)

Merge de `config.yaml` + `config.neuralwatt.yaml` + llama-swap con:
- 3 batches de 30: kimi-k2.6, qwen3_local, glm-5.1
- `annotation.vision_model: kimi-k2.6`
- `image_generation.enabled: false`

### Estimación de coste

| Fase | LLM calls | Tokens aprox. | Tiempo aprox. |
|------|-----------|---------------|----------------|
| Generar 90 quejas | 90 | ~200K in / ~500K out | ~30-40 min |
| Extraer 90 quejas | 90 | ~150K in / ~300K out | ~30-40 min |
| Evaluar | 0 (local) | — | < 1 min |
| **Total** | 180 | ~350K in / ~800K out | ~60-80 min |

### Extensión futura: imágenes

Como experimento separado (FLUX es lento, ~3 min/imagen):
- Generar 15 quejas con fotos (5 por modelo)
- FLUX genera los PNGs
- Extraer con el modelo de visión viendo las fotos
- Comparar extracción con/sin imágenes

## Cómo ejecutar

```bash
cd eval
python run_experiment.py           # todo el experimento
python run_experiment.py --generate  # solo fase 1
python run_experiment.py --extract  # solo fase 2 (necesita fase 1)
python run_experiment.py --evaluate  # solo fase 3 (necesita fases 1+2)
```