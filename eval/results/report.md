# Evaluación del extractor — Resultados

**Dataset:** 99 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| kimi-k2.6 | 0.8779 | 0.936 | 0.8955 | 0.8081 | 1.2 | 1.46 |
| kimi-k2.7-code | 0.8468 | 0.9461 | 0.876 | 0.7475 | 1.2 | 1.69 |
| qwen3.5-397b | 0.8648 | 0.915 | 0.8809 | 0.7959 | 1.18 | 1.43 |
| qwen3.6-35b | 0.8805 | 0.9074 | 0.8886 | 0.8283 | 1.2 | 1.33 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| kimi-k2.6 | 1.0 | 0.8482 | 0.8482 | 0.3393 | 0.875 | 0.6161 | 0.3028 | 0.0 | 0.0 |
| kimi-k2.7-code | 1.0 | 0.8761 | 0.8761 | 0.354 | 0.8938 | 0.6018 | 0.2434 | 0.0 | 0.0 |
| qwen3.5-397b | 1.0 | 0.8411 | 0.8411 | 0.2617 | 0.8598 | 0.5981 | 0.2765 | 0.0 | 0.0 |
| qwen3.6-35b | 1.0 | 0.8224 | 0.8224 | 0.3645 | 0.8598 | 0.5888 | 0.2407 | 0.0 | 0.0 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| kimi-k2.6 | 95 | 17 | 7 | 33 |
| kimi-k2.7-code | 99 | 14 | 6 | 54 |
| qwen3.5-397b | 90 | 17 | 9 | 33 |
| qwen3.6-35b | 88 | 19 | 12 | 25 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados