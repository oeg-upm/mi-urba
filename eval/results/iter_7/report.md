# Evaluación del extractor — Resultados

**Dataset:** 250 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| gemma4_31b_awq × gemma4_31b | 0.9263 | 0.9947 | 0.9482 | 0.856 | 1.17 | 1.37 |
| gemma4_31b_awq × qwen3_6_35b | 0.9049 | 0.9521 | 0.9199 | 0.8543 | 1.17 | 1.39 |
| gemma4_31b_awq × gemma4_12b | 0.9337 | 0.982 | 0.9491 | 0.884 | 1.17 | 1.32 |
| gemma4_31b_awq × qwen3_4b | 0.8684 | 0.9607 | 0.8972 | 0.78 | 1.17 | 1.56 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| gemma4_31b_awq × gemma4_31b | 1.0 | 0.8419 | 0.8419 | 0.8041 | 0.9381 | 0.6632 | 0.7646 | 0.6578 | 0.8677 |
| gemma4_31b_awq × qwen3_6_35b | 1.0 | 0.7365 | 0.7365 | 0.7329 | 0.9278 | 0.639 | 0.7022 | 0.7057 | 0.8081 |
| gemma4_31b_awq × gemma4_12b | 1.0 | 0.8293 | 0.8293 | 0.8223 | 0.9512 | 0.6551 | 0.6527 | 0.6359 | 0.8281 |
| gemma4_31b_awq × qwen3_4b | 1.0 | 0.7589 | 0.7589 | 0.4468 | 0.7624 | 0.5177 | 0.6578 | 0.5115 | 0.5361 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| gemma4_31b_awq × gemma4_31b | 245 | 46 | 2 | 51 |
| gemma4_31b_awq × qwen3_6_35b | 204 | 73 | 13 | 67 |
| gemma4_31b_awq × gemma4_12b | 238 | 49 | 6 | 42 |
| gemma4_31b_awq × qwen3_4b | 214 | 68 | 11 | 108 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados