# Evaluación del extractor — Resultados

**Dataset:** 250 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| gemma4_31b_awq × gemma4_12b | 0.9573 | 0.988 | 0.9677 | 0.924 | 1.17 | 1.24 |
| qwen3_6_35b_awq × gemma4_12b | 0.8782 | 0.9435 | 0.897 | 0.8008 | 1.15 | 1.33 |
| gemma4_12b_awq × gemma4_12b | 0.9392 | 0.9819 | 0.9526 | 0.8992 | 1.17 | 1.27 |
| qwen3_4b_instr_awq × gemma4_12b | 0.8132 | 0.8651 | 0.8195 | 0.6907 | 1.18 | 1.33 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| gemma4_31b_awq × gemma4_12b | 1.0 | 0.8507 | 0.8507 | 0.8681 | 0.9549 | 0.6806 | 0.7708 | 0.6528 | 0.8304 |
| qwen3_6_35b_awq × gemma4_12b | 1.0 | 0.708 | 0.708 | 0.756 | 0.94 | 0.652 | 0.7057 | 0.616 | 0.7647 |
| gemma4_12b_awq × gemma4_12b | 1.0 | 0.8451 | 0.8451 | 0.8838 | 0.9718 | 0.7254 | 0.7864 | 0.605 | 0.7817 |
| qwen3_4b_instr_awq × gemma4_12b | 1.0 | 0.6061 | 0.6061 | 0.3203 | 0.9221 | 0.5368 | 0.5711 | 0.5368 | 0.7042 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| gemma4_31b_awq × gemma4_12b | 245 | 43 | 5 | 22 |
| qwen3_6_35b_awq × gemma4_12b | 177 | 73 | 21 | 64 |
| gemma4_12b_awq × gemma4_12b | 240 | 44 | 6 | 30 |
| qwen3_4b_instr_awq × gemma4_12b | 140 | 91 | 47 | 83 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados