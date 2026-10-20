# Evaluación del extractor — Resultados

**Dataset:** 250 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| gemma4_31b_awq × gemma4_31b | 0.9477 | 0.9907 | 0.9614 | 0.904 | 1.17 | 1.28 |
| gemma4_31b_awq × qwen3_6_35b | 0.9136 | 0.9399 | 0.9217 | 0.8866 | 1.17 | 1.27 |
| gemma4_31b_awq × gemma4_12b | 0.966 | 0.988 | 0.9733 | 0.944 | 1.17 | 1.22 |
| gemma4_31b_awq × qwen3_4b | 0.912 | 0.9753 | 0.9324 | 0.844 | 1.17 | 1.4 |
| qwen3_6_35b_awq × gemma4_31b | 0.8828 | 0.9574 | 0.9056 | 0.8009 | 1.13 | 1.31 |
| qwen3_6_35b_awq × qwen3_6_35b | 0.8969 | 0.9276 | 0.9064 | 0.8553 | 1.14 | 1.26 |
| qwen3_6_35b_awq × gemma4_12b | 0.8813 | 0.9423 | 0.9001 | 0.8095 | 1.13 | 1.29 |
| qwen3_6_35b_awq × qwen3_4b | 0.8514 | 0.9235 | 0.8733 | 0.7749 | 1.13 | 1.4 |
| gemma4_12b_awq × gemma4_31b | 0.9378 | 0.9839 | 0.9519 | 0.8835 | 1.17 | 1.28 |
| gemma4_12b_awq × qwen3_6_35b | 0.9187 | 0.946 | 0.9271 | 0.8785 | 1.17 | 1.26 |
| gemma4_12b_awq × gemma4_12b | 0.9404 | 0.9799 | 0.9527 | 0.8916 | 1.17 | 1.26 |
| gemma4_12b_awq × qwen3_4b | 0.9291 | 0.9672 | 0.9409 | 0.8755 | 1.17 | 1.33 |
| qwen3_4b_instr_awq × gemma4_31b | 0.8063 | 0.865 | 0.8166 | 0.6793 | 1.18 | 1.32 |
| qwen3_4b_instr_awq × qwen3_6_35b | 0.7888 | 0.8077 | 0.7852 | 0.6838 | 1.18 | 1.27 |
| qwen3_4b_instr_awq × gemma4_12b | 0.7954 | 0.8544 | 0.8055 | 0.654 | 1.18 | 1.31 |
| qwen3_4b_instr_awq × qwen3_4b | 0.7693 | 0.8495 | 0.7868 | 0.6245 | 1.18 | 1.48 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| gemma4_31b_awq × gemma4_31b | 1.0 | 0.91 | 0.91 | 0.8339 | 0.955 | 0.6817 | 0.8743 | 0.7376 | 0.9279 |
| gemma4_31b_awq × qwen3_6_35b | 1.0 | 0.781 | 0.781 | 0.7737 | 0.9015 | 0.6423 | 0.7579 | 0.7825 | 0.871 |
| gemma4_31b_awq × gemma4_12b | 1.0 | 0.8793 | 0.8793 | 0.8621 | 0.9621 | 0.6793 | 0.7761 | 0.6681 | 0.8494 |
| gemma4_31b_awq × qwen3_4b | 1.0 | 0.7887 | 0.7887 | 0.5634 | 0.7641 | 0.5176 | 0.6843 | 0.6115 | 0.6273 |
| qwen3_6_35b_awq × gemma4_31b | 1.0 | 0.7621 | 0.7621 | 0.7177 | 0.9435 | 0.6452 | 0.7957 | 0.6831 | 0.8723 |
| qwen3_6_35b_awq × qwen3_6_35b | 1.0 | 0.6292 | 0.6292 | 0.675 | 0.8875 | 0.6208 | 0.7083 | 0.6802 | 0.775 |
| qwen3_6_35b_awq × gemma4_12b | 1.0 | 0.7317 | 0.7317 | 0.7764 | 0.9431 | 0.6423 | 0.7121 | 0.6413 | 0.8015 |
| qwen3_6_35b_awq × qwen3_4b | 1.0 | 0.6833 | 0.6833 | 0.4167 | 0.7125 | 0.5083 | 0.6542 | 0.5342 | 0.5382 |
| gemma4_12b_awq × gemma4_31b | 1.0 | 0.8877 | 0.8877 | 0.814 | 0.9719 | 0.7228 | 0.8374 | 0.7108 | 0.8819 |
| gemma4_12b_awq × qwen3_6_35b | 1.0 | 0.7518 | 0.7518 | 0.7299 | 0.9526 | 0.7044 | 0.7603 | 0.7245 | 0.8358 |
| gemma4_12b_awq × gemma4_12b | 1.0 | 0.8486 | 0.8486 | 0.8627 | 0.9789 | 0.7007 | 0.7653 | 0.6235 | 0.7887 |
| gemma4_12b_awq × qwen3_4b | 1.0 | 0.7722 | 0.7722 | 0.4982 | 0.758 | 0.5374 | 0.6661 | 0.545 | 0.586 |
| qwen3_4b_instr_awq × gemma4_31b | 1.0 | 0.6364 | 0.6364 | 0.2987 | 0.8874 | 0.5671 | 0.6209 | 0.6298 | 0.8218 |
| qwen3_4b_instr_awq × qwen3_6_35b | 1.0 | 0.5302 | 0.5302 | 0.3953 | 0.8558 | 0.5395 | 0.6233 | 0.6033 | 0.7783 |
| qwen3_4b_instr_awq × gemma4_12b | 1.0 | 0.6316 | 0.6316 | 0.3289 | 0.886 | 0.5395 | 0.6184 | 0.5414 | 0.7142 |
| qwen3_4b_instr_awq × qwen3_4b | 1.0 | 0.6211 | 0.6211 | 0.1674 | 0.6652 | 0.3568 | 0.6017 | 0.4284 | 0.4508 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| gemma4_31b_awq × gemma4_31b | 263 | 26 | 4 | 30 |
| gemma4_31b_awq × qwen3_6_35b | 214 | 60 | 16 | 40 |
| gemma4_31b_awq × gemma4_12b | 255 | 35 | 3 | 15 |
| gemma4_31b_awq × qwen3_4b | 224 | 60 | 9 | 65 |
| qwen3_6_35b_awq × gemma4_31b | 189 | 59 | 14 | 55 |
| qwen3_6_35b_awq × qwen3_6_35b | 151 | 89 | 19 | 48 |
| qwen3_6_35b_awq × gemma4_12b | 180 | 66 | 16 | 52 |
| qwen3_6_35b_awq × qwen3_4b | 164 | 76 | 22 | 83 |
| gemma4_12b_awq × gemma4_31b | 253 | 32 | 7 | 34 |
| gemma4_12b_awq × qwen3_6_35b | 206 | 68 | 16 | 38 |
| gemma4_12b_awq × gemma4_12b | 241 | 43 | 8 | 29 |
| gemma4_12b_awq × qwen3_4b | 217 | 64 | 11 | 51 |
| qwen3_4b_instr_awq × gemma4_31b | 147 | 84 | 48 | 82 |
| qwen3_4b_instr_awq × qwen3_6_35b | 114 | 101 | 61 | 83 |
| qwen3_4b_instr_awq × gemma4_12b | 144 | 84 | 51 | 82 |
| qwen3_4b_instr_awq × qwen3_4b | 141 | 86 | 52 | 123 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados