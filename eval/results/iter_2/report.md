# Evaluación del extractor — Resultados

**Dataset:** 250 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| gemma4_31b_awq × gemma4_31b | 0.869 | 0.9907 | 0.9077 | 0.752 | 1.17 | 1.55 |
| gemma4_31b_awq × qwen3_6_35b | 0.8905 | 0.9558 | 0.9108 | 0.8163 | 1.18 | 1.44 |
| gemma4_31b_awq × gemma4_12b | 0.9133 | 0.992 | 0.9393 | 0.832 | 1.17 | 1.39 |
| gemma4_31b_awq × qwen3_4b | 0.8907 | 0.9587 | 0.913 | 0.816 | 1.17 | 1.44 |
| qwen3_6_35b_awq × gemma4_31b | 0.7965 | 0.9602 | 0.8496 | 0.6283 | 1.16 | 1.66 |
| qwen3_6_35b_awq × qwen3_6_35b | 0.8492 | 0.9208 | 0.8721 | 0.7602 | 1.16 | 1.47 |
| qwen3_6_35b_awq × gemma4_12b | 0.8392 | 0.9491 | 0.8753 | 0.7212 | 1.16 | 1.45 |
| qwen3_6_35b_awq × qwen3_4b | 0.7946 | 0.8982 | 0.8264 | 0.6858 | 1.16 | 1.58 |
| gemma4_12b_awq × gemma4_31b | 0.8812 | 0.99 | 0.9163 | 0.7631 | 1.17 | 1.51 |
| gemma4_12b_awq × qwen3_6_35b | 0.9136 | 0.9558 | 0.9264 | 0.8653 | 1.17 | 1.38 |
| gemma4_12b_awq × gemma4_12b | 0.916 | 0.9859 | 0.9379 | 0.8434 | 1.17 | 1.37 |
| gemma4_12b_awq × qwen3_4b | 0.8873 | 0.9665 | 0.9121 | 0.7992 | 1.17 | 1.49 |
| qwen3_4b_instr_awq × gemma4_31b | 0.6942 | 0.8808 | 0.7483 | 0.4854 | 1.18 | 1.75 |
| qwen3_4b_instr_awq × qwen3_6_35b | 0.7737 | 0.8714 | 0.7991 | 0.6164 | 1.17 | 1.51 |
| qwen3_4b_instr_awq × gemma4_12b | 0.7462 | 0.8724 | 0.7836 | 0.5649 | 1.18 | 1.54 |
| qwen3_4b_instr_awq × qwen3_4b | 0.7384 | 0.8536 | 0.7703 | 0.5774 | 1.18 | 1.62 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| gemma4_31b_awq × gemma4_31b | 1.0 | 0.8828 | 0.8828 | 0.8069 | 0.9483 | 0.6483 | 0.6437 | 0.6603 | 0.7856 |
| gemma4_31b_awq × qwen3_6_35b | 1.0 | 0.7636 | 0.7636 | 0.7855 | 0.8945 | 0.6182 | 0.5233 | 0.683 | 0.7709 |
| gemma4_31b_awq × gemma4_12b | 1.0 | 0.8616 | 0.8616 | 0.8235 | 0.9412 | 0.6367 | 0.5568 | 0.5949 | 0.7249 |
| gemma4_31b_awq × qwen3_4b | 1.0 | 0.7643 | 0.7643 | 0.4643 | 0.7357 | 0.5071 | 0.5676 | 0.5423 | 0.5863 |
| qwen3_6_35b_awq × gemma4_31b | 1.0 | 0.7659 | 0.7659 | 0.6587 | 0.9524 | 0.5754 | 0.6138 | 0.6081 | 0.7083 |
| qwen3_6_35b_awq × qwen3_6_35b | 1.0 | 0.6203 | 0.6203 | 0.6498 | 0.8987 | 0.5907 | 0.4986 | 0.6354 | 0.7391 |
| qwen3_6_35b_awq × gemma4_12b | 1.0 | 0.7045 | 0.7045 | 0.7126 | 0.9352 | 0.5951 | 0.5378 | 0.5551 | 0.6903 |
| qwen3_6_35b_awq × qwen3_4b | 1.0 | 0.6596 | 0.6596 | 0.417 | 0.7277 | 0.4979 | 0.5323 | 0.4755 | 0.5021 |
| gemma4_12b_awq × gemma4_31b | 1.0 | 0.8641 | 0.8641 | 0.7805 | 0.9617 | 0.6969 | 0.6487 | 0.6365 | 0.7625 |
| gemma4_12b_awq × qwen3_6_35b | 1.0 | 0.7509 | 0.7509 | 0.8132 | 0.9451 | 0.707 | 0.54 | 0.7063 | 0.7973 |
| gemma4_12b_awq × gemma4_12b | 1.0 | 0.8345 | 0.8345 | 0.8169 | 0.9683 | 0.6937 | 0.5704 | 0.5616 | 0.6919 |
| gemma4_12b_awq × qwen3_4b | 1.0 | 0.7798 | 0.7798 | 0.491 | 0.7401 | 0.5487 | 0.5511 | 0.5123 | 0.5403 |
| qwen3_4b_instr_awq × gemma4_31b | 1.0 | 0.6067 | 0.6067 | 0.3305 | 0.9079 | 0.5356 | 0.5052 | 0.5738 | 0.6792 |
| qwen3_4b_instr_awq × qwen3_6_35b | 1.0 | 0.5088 | 0.5088 | 0.3991 | 0.8904 | 0.5219 | 0.4336 | 0.5819 | 0.6981 |
| qwen3_4b_instr_awq × gemma4_12b | 1.0 | 0.5975 | 0.5975 | 0.3263 | 0.9195 | 0.5212 | 0.4809 | 0.4954 | 0.589 |
| qwen3_4b_instr_awq × qwen3_4b | 1.0 | 0.5923 | 0.5923 | 0.2189 | 0.6395 | 0.3433 | 0.4764 | 0.4162 | 0.4185 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| gemma4_31b_awq × gemma4_31b | 256 | 34 | 3 | 97 |
| gemma4_31b_awq × qwen3_6_35b | 210 | 65 | 13 | 77 |
| gemma4_31b_awq × gemma4_12b | 249 | 40 | 4 | 58 |
| gemma4_31b_awq × qwen3_4b | 214 | 66 | 13 | 81 |
| qwen3_6_35b_awq × gemma4_31b | 193 | 59 | 10 | 124 |
| qwen3_6_35b_awq × qwen3_6_35b | 147 | 90 | 20 | 87 |
| qwen3_6_35b_awq × gemma4_12b | 174 | 73 | 15 | 81 |
| qwen3_6_35b_awq × qwen3_4b | 155 | 80 | 27 | 123 |
| gemma4_12b_awq × gemma4_31b | 248 | 39 | 4 | 89 |
| gemma4_12b_awq × qwen3_6_35b | 205 | 68 | 14 | 66 |
| gemma4_12b_awq × gemma4_12b | 237 | 47 | 7 | 57 |
| gemma4_12b_awq × qwen3_4b | 216 | 61 | 14 | 94 |
| qwen3_4b_instr_awq × gemma4_31b | 145 | 94 | 42 | 180 |
| qwen3_4b_instr_awq × qwen3_6_35b | 116 | 112 | 44 | 123 |
| qwen3_4b_instr_awq × gemma4_12b | 141 | 95 | 45 | 131 |
| qwen3_4b_instr_awq × qwen3_4b | 138 | 95 | 48 | 153 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados