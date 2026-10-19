# Evaluación del extractor — Resultados

**Dataset:** 250 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| gemma4_31b_awq × gemma4_31b | 0.8513 | 0.98 | 0.8923 | 0.728 | 1.17 | 1.57 |
| gemma4_31b_awq × qwen3_6_35b | 0.9208 | 0.9569 | 0.9326 | 0.8833 | 1.18 | 1.32 |
| gemma4_31b_awq × gemma4_12b | 0.9177 | 0.988 | 0.9401 | 0.852 | 1.17 | 1.38 |
| gemma4_31b_awq × qwen3_4b | 0.8691 | 0.9487 | 0.8946 | 0.784 | 1.17 | 1.49 |
| qwen3_6_35b_awq × gemma4_31b | 0.8056 | 0.9551 | 0.8518 | 0.6408 | 1.17 | 1.69 |
| qwen3_6_35b_awq × qwen3_6_35b | 0.8572 | 0.9081 | 0.8687 | 0.7692 | 1.18 | 1.45 |
| qwen3_6_35b_awq × gemma4_12b | 0.8612 | 0.9531 | 0.8868 | 0.7429 | 1.17 | 1.44 |
| qwen3_6_35b_awq × qwen3_4b | 0.8376 | 0.9415 | 0.8658 | 0.702 | 1.17 | 1.59 |
| gemma4_12b_awq × gemma4_31b | 0.8975 | 0.9966 | 0.9278 | 0.7959 | 1.14 | 1.41 |
| gemma4_12b_awq × qwen3_6_35b | 0.8992 | 0.9433 | 0.9123 | 0.8443 | 1.18 | 1.35 |
| gemma4_12b_awq × gemma4_12b | 0.9304 | 0.99 | 0.9489 | 0.8675 | 1.17 | 1.33 |
| gemma4_12b_awq × qwen3_4b | 0.8973 | 0.9632 | 0.917 | 0.8233 | 1.17 | 1.45 |
| qwen3_4b_instr_awq × gemma4_31b | 0.6803 | 0.881 | 0.7372 | 0.4327 | 1.17 | 1.78 |
| qwen3_4b_instr_awq × qwen3_6_35b | 0.7826 | 0.8745 | 0.802 | 0.6213 | 1.17 | 1.51 |
| qwen3_4b_instr_awq × gemma4_12b | 0.7344 | 0.8755 | 0.7705 | 0.5306 | 1.17 | 1.58 |
| qwen3_4b_instr_awq × qwen3_4b | 0.7102 | 0.8429 | 0.7435 | 0.5184 | 1.17 | 1.72 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| gemma4_31b_awq × gemma4_31b | 1.0 | 0.8924 | 0.8924 | 0.7535 | 0.941 | 0.6458 | 0.6808 | 0.6892 | 0.7853 |
| gemma4_31b_awq × qwen3_6_35b | 1.0 | 0.7831 | 0.7831 | 0.7096 | 0.9007 | 0.625 | 0.5426 | 0.7454 | 0.8211 |
| gemma4_31b_awq × gemma4_12b | 1.0 | 0.8655 | 0.8655 | 0.7483 | 0.9414 | 0.6414 | 0.5853 | 0.652 | 0.796 |
| gemma4_31b_awq × qwen3_4b | 1.0 | 0.759 | 0.759 | 0.4209 | 0.7194 | 0.5036 | 0.5731 | 0.5905 | 0.6223 |
| qwen3_6_35b_awq × gemma4_31b | 1.0 | 0.7491 | 0.7491 | 0.583 | 0.9336 | 0.5793 | 0.5861 | 0.6046 | 0.6796 |
| qwen3_6_35b_awq × qwen3_6_35b | 1.0 | 0.6518 | 0.6518 | 0.5749 | 0.8988 | 0.5709 | 0.5118 | 0.6337 | 0.6977 |
| qwen3_6_35b_awq × gemma4_12b | 1.0 | 0.7212 | 0.7212 | 0.6134 | 0.9405 | 0.5948 | 0.5638 | 0.5998 | 0.7138 |
| qwen3_6_35b_awq × qwen3_4b | 1.0 | 0.6377 | 0.6377 | 0.4226 | 0.7094 | 0.4604 | 0.5286 | 0.4824 | 0.5006 |
| gemma4_12b_awq × gemma4_31b | 1.0 | 0.9273 | 0.9273 | 0.7455 | 0.9455 | 0.697 | 0.6596 | 0.6833 | 0.8111 |
| gemma4_12b_awq × qwen3_6_35b | 1.0 | 0.7881 | 0.7881 | 0.7286 | 0.9554 | 0.6617 | 0.5328 | 0.7032 | 0.8042 |
| gemma4_12b_awq × gemma4_12b | 1.0 | 0.8636 | 0.8636 | 0.7657 | 0.9615 | 0.6853 | 0.5705 | 0.6573 | 0.7972 |
| gemma4_12b_awq × qwen3_4b | 1.0 | 0.7885 | 0.7885 | 0.448 | 0.7312 | 0.4982 | 0.5541 | 0.5505 | 0.5974 |
| qwen3_4b_instr_awq × gemma4_31b | 1.0 | 0.6157 | 0.6157 | 0.3223 | 0.9008 | 0.5207 | 0.553 | 0.5742 | 0.6722 |
| qwen3_4b_instr_awq × qwen3_6_35b | 1.0 | 0.5584 | 0.5584 | 0.4156 | 0.8874 | 0.5325 | 0.4596 | 0.5617 | 0.7042 |
| qwen3_4b_instr_awq × gemma4_12b | 1.0 | 0.6125 | 0.6125 | 0.3125 | 0.925 | 0.55 | 0.4819 | 0.5109 | 0.6285 |
| qwen3_4b_instr_awq × qwen3_4b | 1.0 | 0.5517 | 0.5517 | 0.2371 | 0.6724 | 0.431 | 0.4831 | 0.4542 | 0.4389 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| gemma4_31b_awq × gemma4_31b | 257 | 31 | 5 | 105 |
| gemma4_31b_awq × qwen3_6_35b | 213 | 59 | 11 | 45 |
| gemma4_31b_awq × gemma4_12b | 251 | 39 | 3 | 55 |
| gemma4_31b_awq × qwen3_4b | 211 | 67 | 15 | 95 |
| qwen3_6_35b_awq × gemma4_31b | 203 | 68 | 16 | 142 |
| qwen3_6_35b_awq × qwen3_6_35b | 161 | 86 | 29 | 92 |
| qwen3_6_35b_awq × gemma4_12b | 194 | 75 | 18 | 84 |
| qwen3_6_35b_awq × qwen3_4b | 169 | 96 | 22 | 125 |
| gemma4_12b_awq × gemma4_31b | 153 | 12 | 2 | 43 |
| gemma4_12b_awq × qwen3_6_35b | 212 | 57 | 18 | 60 |
| gemma4_12b_awq × gemma4_12b | 247 | 39 | 6 | 44 |
| gemma4_12b_awq × qwen3_4b | 220 | 59 | 13 | 83 |
| qwen3_4b_instr_awq × gemma4_31b | 149 | 93 | 45 | 193 |
| qwen3_4b_instr_awq × qwen3_6_35b | 129 | 102 | 45 | 123 |
| qwen3_4b_instr_awq × gemma4_12b | 147 | 93 | 47 | 148 |
| qwen3_4b_instr_awq × qwen3_4b | 128 | 104 | 55 | 190 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados