# Evaluación del extractor — Resultados

**Dataset:** 250 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| gemma4_31b_awq × gemma4_31b | 0.878 | 0.9747 | 0.9084 | 0.788 | 1.17 | 1.51 |
| gemma4_31b_awq × qwen3_6_35b | 0.8938 | 0.9281 | 0.9039 | 0.8548 | 1.17 | 1.34 |
| gemma4_31b_awq × gemma4_12b | 0.9107 | 0.984 | 0.9339 | 0.844 | 1.17 | 1.38 |
| gemma4_31b_awq × qwen3_4b | 0.907 | 0.97 | 0.9275 | 0.84 | 1.17 | 1.41 |
| qwen3_6_35b_awq × gemma4_31b | 0.8012 | 0.9561 | 0.8485 | 0.6447 | 1.14 | 1.63 |
| qwen3_6_35b_awq × qwen3_6_35b | 0.8319 | 0.8981 | 0.8519 | 0.7411 | 1.14 | 1.42 |
| qwen3_6_35b_awq × gemma4_12b | 0.8472 | 0.9408 | 0.8763 | 0.7456 | 1.14 | 1.43 |
| qwen3_6_35b_awq × qwen3_4b | 0.8319 | 0.9306 | 0.8622 | 0.7281 | 1.14 | 1.5 |
| gemma4_12b_awq × gemma4_31b | 0.877 | 0.9899 | 0.9135 | 0.7621 | 1.17 | 1.5 |
| gemma4_12b_awq × gemma4_12b | 0.914 | 0.9839 | 0.9373 | 0.8387 | 1.17 | 1.35 |
| gemma4_12b_awq × qwen3_4b | 0.9087 | 0.957 | 0.9236 | 0.8468 | 1.17 | 1.41 |
| qwen3_4b_instr_awq × gemma4_31b | 0.7545 | 0.8946 | 0.7934 | 0.5676 | 1.15 | 1.62 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| gemma4_31b_awq × gemma4_31b | 1.0 | 0.8986 | 0.8986 | 0.8601 | 0.951 | 0.6503 | 0.8479 | 0.6856 | 0.8042 |
| gemma4_31b_awq × qwen3_6_35b | 1.0 | 0.7491 | 0.7491 | 0.7528 | 0.9151 | 0.6347 | 0.7478 | 0.7411 | 0.8235 |
| gemma4_31b_awq × gemma4_12b | 1.0 | 0.8547 | 0.8547 | 0.8408 | 0.955 | 0.6505 | 0.7768 | 0.6758 | 0.8264 |
| gemma4_31b_awq × qwen3_4b | 1.0 | 0.7649 | 0.7649 | 0.4982 | 0.7474 | 0.5228 | 0.6667 | 0.5729 | 0.6064 |
| qwen3_6_35b_awq × gemma4_31b | 1.0 | 0.7379 | 0.7379 | 0.7339 | 0.9395 | 0.6008 | 0.7608 | 0.6418 | 0.711 |
| qwen3_6_35b_awq × qwen3_6_35b | 1.0 | 0.6272 | 0.6272 | 0.7281 | 0.8904 | 0.614 | 0.7076 | 0.6784 | 0.7361 |
| qwen3_6_35b_awq × gemma4_12b | 1.0 | 0.7284 | 0.7284 | 0.7737 | 0.9342 | 0.6296 | 0.7157 | 0.6217 | 0.7497 |
| qwen3_6_35b_awq × qwen3_4b | 1.0 | 0.6598 | 0.6598 | 0.4066 | 0.7012 | 0.4606 | 0.6515 | 0.5207 | 0.527 |
| gemma4_12b_awq × gemma4_31b | 1.0 | 0.899 | 0.899 | 0.8084 | 0.9477 | 0.7178 | 0.8153 | 0.6844 | 0.7753 |
| gemma4_12b_awq × gemma4_12b | 1.0 | 0.8462 | 0.8462 | 0.8322 | 0.9615 | 0.7378 | 0.7529 | 0.6681 | 0.7931 |
| gemma4_12b_awq × qwen3_4b | 1.0 | 0.7935 | 0.7935 | 0.5 | 0.7681 | 0.5435 | 0.663 | 0.564 | 0.5779 |
| qwen3_4b_instr_awq × gemma4_31b | 1.0 | 0.6543 | 0.6543 | 0.2819 | 0.883 | 0.5266 | 0.582 | 0.5696 | 0.6809 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| gemma4_31b_awq × gemma4_31b | 257 | 29 | 7 | 92 |
| gemma4_31b_awq × qwen3_6_35b | 203 | 68 | 20 | 61 |
| gemma4_31b_awq × gemma4_12b | 247 | 42 | 4 | 55 |
| gemma4_31b_awq × qwen3_4b | 218 | 67 | 8 | 68 |
| qwen3_6_35b_awq × gemma4_31b | 183 | 65 | 13 | 123 |
| qwen3_6_35b_awq × qwen3_6_35b | 143 | 85 | 28 | 89 |
| qwen3_6_35b_awq × gemma4_12b | 177 | 66 | 18 | 83 |
| qwen3_6_35b_awq × qwen3_4b | 159 | 82 | 20 | 102 |
| gemma4_12b_awq × gemma4_31b | 258 | 29 | 3 | 85 |
| gemma4_12b_awq × gemma4_12b | 242 | 44 | 4 | 49 |
| gemma4_12b_awq × qwen3_4b | 219 | 57 | 14 | 74 |
| qwen3_4b_instr_awq × gemma4_31b | 123 | 65 | 25 | 112 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados