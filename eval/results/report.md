# Evaluación del extractor — Resultados

**Dataset:** 100 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| kimi-k2.6 | 0.6845 | 0.9 | 0.7555 | 0.46 | 1.25 | 2.16 |
| kimi-k2.7-code | 0.668 | 0.91 | 0.7432 | 0.44 | 1.25 | 2.51 |
| qwen3.5-397b | 0.6695 | 0.9 | 0.7443 | 0.43 | 1.25 | 2.16 |
| qwen3.6-35b | 0.7283 | 0.9017 | 0.7842 | 0.55 | 1.25 | 1.93 |
| minimax-m3 | 0.6652 | 0.93 | 0.7469 | 0.42 | 1.25 | 2.46 |
| gemma4-31b | 0.5967 | 0.8833 | 0.6835 | 0.34 | 1.25 | 2.46 |
| glm-5.1 | 0.6602 | 0.8967 | 0.7347 | 0.43 | 1.25 | 2.28 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| kimi-k2.6 | 1.0 | 0.7257 | 0.7257 | 0.2743 | 0.8584 | 0.6106 | 0.3853 | 0.5199 | 0.5221 |
| kimi-k2.7-code | 1.0 | 0.7456 | 0.7456 | 0.307 | 0.8772 | 0.5877 | 0.3735 | 0.4817 | 0.5132 |
| qwen3.5-397b | 1.0 | 0.6637 | 0.6637 | 0.2389 | 0.8584 | 0.5929 | 0.4049 | 0.5133 | 0.5029 |
| qwen3.6-35b | 1.0 | 0.6875 | 0.6875 | 0.3393 | 0.8304 | 0.625 | 0.4152 | 0.5662 | 0.5714 |
| minimax-m3 | 1.0 | 0.6897 | 0.6897 | 0.3276 | 0.8793 | 0.6121 | 0.3772 | 0.4368 | 0.4914 |
| gemma4-31b | 1.0 | 0.7364 | 0.7364 | 0.3091 | 0.8636 | 0.5727 | 0.4455 | 0.497 | 0.453 |
| glm-5.1 | 1.0 | 0.7411 | 0.7411 | 0.3036 | 0.8571 | 0.5893 | 0.4568 | 0.5074 | 0.4911 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| kimi-k2.6 | 82 | 31 | 12 | 103 |
| kimi-k2.7-code | 85 | 29 | 11 | 137 |
| qwen3.5-397b | 75 | 38 | 12 | 103 |
| qwen3.6-35b | 77 | 35 | 13 | 81 |
| minimax-m3 | 80 | 36 | 9 | 130 |
| gemma4-31b | 81 | 29 | 15 | 136 |
| glm-5.1 | 83 | 29 | 13 | 116 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados