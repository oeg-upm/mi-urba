# Evaluación del extractor — Resultados

**Dataset:** 250 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| gemma4_31b_awq × gemma4_31b | 0.8873 | 0.9847 | 0.9182 | 0.792 | 1.17 | 1.44 |
| gemma4_31b_awq × qwen3_6_35b | 0.9156 | 0.9652 | 0.9314 | 0.8648 | 1.17 | 1.36 |
| gemma4_31b_awq × gemma4_12b | 0.8857 | 0.982 | 0.9165 | 0.792 | 1.17 | 1.47 |
| gemma4_31b_awq × qwen3_4b | 0.9073 | 0.9707 | 0.9282 | 0.84 | 1.17 | 1.43 |
| qwen3_6_35b_awq × gemma4_31b | 0.8024 | 0.9476 | 0.847 | 0.6504 | 1.14 | 1.63 |
| qwen3_6_35b_awq × qwen3_6_35b | 0.8217 | 0.9096 | 0.8479 | 0.7175 | 1.14 | 1.51 |
| qwen3_6_35b_awq × gemma4_12b | 0.8197 | 0.9381 | 0.8562 | 0.6858 | 1.14 | 1.54 |
| qwen3_6_35b_awq × qwen3_4b | 0.8204 | 0.9196 | 0.8506 | 0.708 | 1.14 | 1.57 |
| gemma4_12b_awq × gemma4_31b | 0.8983 | 0.9847 | 0.9259 | 0.812 | 1.17 | 1.42 |
| gemma4_12b_awq × qwen3_6_35b | 0.9042 | 0.9483 | 0.9175 | 0.8629 | 1.17 | 1.38 |
| gemma4_12b_awq × gemma4_12b | 0.915 | 0.982 | 0.9361 | 0.852 | 1.17 | 1.38 |
| gemma4_12b_awq × qwen3_4b | 0.9213 | 0.9613 | 0.9339 | 0.88 | 1.17 | 1.34 |
| qwen3_4b_instr_awq × gemma4_31b | 0.7055 | 0.891 | 0.7594 | 0.4872 | 1.17 | 1.75 |
| qwen3_4b_instr_awq × qwen3_6_35b | 0.7453 | 0.8467 | 0.7718 | 0.5911 | 1.17 | 1.56 |
| qwen3_4b_instr_awq × gemma4_12b | 0.7375 | 0.8932 | 0.7802 | 0.547 | 1.17 | 1.65 |
| qwen3_4b_instr_awq × qwen3_4b | 0.708 | 0.834 | 0.7423 | 0.5342 | 1.17 | 1.67 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| gemma4_31b_awq × gemma4_31b | 1.0 | 0.8681 | 0.8681 | 0.8056 | 0.9479 | 0.6562 | 0.8333 | 0.7104 | 0.86 |
| gemma4_31b_awq × qwen3_6_35b | 1.0 | 0.7309 | 0.7309 | 0.7345 | 0.8982 | 0.6145 | 0.7503 | 0.7361 | 0.8024 |
| gemma4_31b_awq × gemma4_12b | 1.0 | 0.8502 | 0.8502 | 0.8676 | 0.9443 | 0.6446 | 0.7666 | 0.6201 | 0.7276 |
| gemma4_31b_awq × qwen3_4b | 1.0 | 0.7465 | 0.7465 | 0.5 | 0.7148 | 0.493 | 0.6802 | 0.5921 | 0.5974 |
| qwen3_6_35b_awq × gemma4_31b | 1.0 | 0.7562 | 0.7562 | 0.6983 | 0.9504 | 0.5992 | 0.7369 | 0.6484 | 0.7314 |
| qwen3_6_35b_awq × qwen3_6_35b | 1.0 | 0.607 | 0.607 | 0.6376 | 0.917 | 0.5459 | 0.6849 | 0.6718 | 0.73 |
| qwen3_6_35b_awq × gemma4_12b | 1.0 | 0.725 | 0.725 | 0.75 | 0.9458 | 0.5958 | 0.701 | 0.6069 | 0.6896 |
| qwen3_6_35b_awq × qwen3_4b | 1.0 | 0.6441 | 0.6441 | 0.3856 | 0.6695 | 0.428 | 0.6215 | 0.5459 | 0.5367 |
| gemma4_12b_awq × gemma4_31b | 1.0 | 0.8676 | 0.8676 | 0.7875 | 0.9686 | 0.7456 | 0.8142 | 0.7003 | 0.8235 |
| gemma4_12b_awq × qwen3_6_35b | 1.0 | 0.7572 | 0.7572 | 0.7464 | 0.9312 | 0.7355 | 0.7494 | 0.7147 | 0.7784 |
| gemma4_12b_awq × gemma4_12b | 1.0 | 0.8362 | 0.8362 | 0.8537 | 0.9582 | 0.7317 | 0.761 | 0.6736 | 0.7863 |
| gemma4_12b_awq × qwen3_4b | 1.0 | 0.7133 | 0.7133 | 0.5305 | 0.7168 | 0.552 | 0.6559 | 0.5899 | 0.6039 |
| qwen3_4b_instr_awq × gemma4_31b | 1.0 | 0.6102 | 0.6102 | 0.3178 | 0.9025 | 0.5381 | 0.5717 | 0.5706 | 0.6653 |
| qwen3_4b_instr_awq × qwen3_6_35b | 1.0 | 0.5463 | 0.5463 | 0.3657 | 0.838 | 0.5046 | 0.5525 | 0.6003 | 0.7153 |
| qwen3_4b_instr_awq × gemma4_12b | 1.0 | 0.6043 | 0.6043 | 0.3447 | 0.9191 | 0.5106 | 0.5809 | 0.5457 | 0.6525 |
| qwen3_4b_instr_awq × qwen3_4b | 1.0 | 0.5753 | 0.5753 | 0.1598 | 0.6164 | 0.347 | 0.5898 | 0.489 | 0.4924 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| gemma4_31b_awq × gemma4_31b | 250 | 38 | 5 | 72 |
| gemma4_31b_awq × qwen3_6_35b | 201 | 74 | 10 | 56 |
| gemma4_31b_awq × gemma4_12b | 244 | 43 | 6 | 80 |
| gemma4_31b_awq × qwen3_4b | 212 | 72 | 9 | 73 |
| qwen3_6_35b_awq × gemma4_31b | 183 | 59 | 16 | 126 |
| qwen3_6_35b_awq × qwen3_6_35b | 139 | 90 | 25 | 107 |
| qwen3_6_35b_awq × gemma4_12b | 174 | 66 | 18 | 109 |
| qwen3_6_35b_awq × qwen3_4b | 152 | 84 | 22 | 119 |
| gemma4_12b_awq × gemma4_31b | 249 | 38 | 6 | 69 |
| gemma4_12b_awq × qwen3_6_35b | 209 | 67 | 15 | 67 |
| gemma4_12b_awq × gemma4_12b | 240 | 47 | 6 | 57 |
| gemma4_12b_awq × qwen3_4b | 199 | 80 | 14 | 56 |
| qwen3_4b_instr_awq × gemma4_31b | 144 | 92 | 37 | 173 |
| qwen3_4b_instr_awq × qwen3_6_35b | 118 | 98 | 47 | 135 |
| qwen3_4b_instr_awq × gemma4_12b | 142 | 93 | 38 | 151 |
| qwen3_4b_instr_awq × qwen3_4b | 126 | 93 | 54 | 172 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados