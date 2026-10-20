# Evaluación del extractor — Resultados

**Dataset:** 999 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| gemma4_31b_awq × gemma4_31b | 0.875 | 0.9716 | 0.9056 | 0.7838 | 1.18 | 1.45 |
| gemma4_31b_awq × qwen3_6_35b | 0.8878 | 0.9423 | 0.9053 | 0.8229 | 1.18 | 1.42 |
| gemma4_31b_awq × gemma4_12b | 0.9094 | 0.9708 | 0.9291 | 0.8468 | 1.18 | 1.34 |
| gemma4_31b_awq × qwen3_4b | 0.8614 | 0.9601 | 0.8923 | 0.7598 | 1.18 | 1.59 |
| qwen3_6_35b_awq × gemma4_31b | 0.8115 | 0.9541 | 0.8558 | 0.6647 | 1.16 | 1.57 |
| qwen3_6_35b_awq × qwen3_6_35b | 0.8326 | 0.9118 | 0.8575 | 0.733 | 1.16 | 1.48 |
| qwen3_6_35b_awq × gemma4_12b | 0.8466 | 0.946 | 0.8778 | 0.7304 | 1.16 | 1.43 |
| qwen3_6_35b_awq × qwen3_4b | 0.7935 | 0.9311 | 0.8363 | 0.6521 | 1.16 | 1.74 |
| gemma4_12b_awq × gemma4_31b | 0.884 | 0.9776 | 0.9124 | 0.7841 | 1.18 | 1.45 |
| gemma4_12b_awq × qwen3_6_35b | 0.8934 | 0.9455 | 0.9091 | 0.8244 | 1.18 | 1.48 |
| gemma4_12b_awq × gemma4_12b | 0.9203 | 0.9798 | 0.9381 | 0.8504 | 1.18 | 1.35 |
| gemma4_12b_awq × qwen3_4b | 0.8692 | 0.9672 | 0.8996 | 0.7631 | 1.18 | 1.61 |
| qwen3_4b_instr_awq × gemma4_31b | 0.6975 | 0.8982 | 0.755 | 0.4751 | 1.18 | 1.78 |
| qwen3_4b_instr_awq × qwen3_6_35b | 0.7276 | 0.8561 | 0.7627 | 0.5516 | 1.17 | 1.65 |
| qwen3_4b_instr_awq × gemma4_12b | 0.7266 | 0.8815 | 0.7709 | 0.5386 | 1.18 | 1.62 |
| qwen3_4b_instr_awq × qwen3_4b | 0.6982 | 0.8924 | 0.7541 | 0.4815 | 1.18 | 2.02 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| gemma4_31b_awq × gemma4_31b | 1.0 | 0.8316 | 0.8316 | 0.3763 | 0.943 | 0.6956 | 0.6415 | 0.6686 | 0.8493 |
| gemma4_31b_awq × qwen3_6_35b | 1.0 | 0.7021 | 0.7021 | 0.3666 | 0.9019 | 0.6792 | 0.5166 | 0.7118 | 0.8005 |
| gemma4_31b_awq × gemma4_12b | 1.0 | 0.7787 | 0.7787 | 0.3933 | 0.9506 | 0.6861 | 0.5499 | 0.7113 | 0.8842 |
| gemma4_31b_awq × qwen3_4b | 1.0 | 0.7487 | 0.7487 | 0.2862 | 0.7397 | 0.5572 | 0.5757 | 0.5971 | 0.5856 |
| qwen3_6_35b_awq × gemma4_31b | 1.0 | 0.7071 | 0.7071 | 0.3625 | 0.9178 | 0.6122 | 0.5574 | 0.6156 | 0.7631 |
| qwen3_6_35b_awq × qwen3_6_35b | 1.0 | 0.6018 | 0.6018 | 0.3233 | 0.8736 | 0.5895 | 0.4655 | 0.6098 | 0.703 |
| qwen3_6_35b_awq × gemma4_12b | 1.0 | 0.6758 | 0.6758 | 0.375 | 0.9216 | 0.6059 | 0.4812 | 0.6516 | 0.8116 |
| qwen3_6_35b_awq × qwen3_4b | 1.0 | 0.6379 | 0.6379 | 0.2899 | 0.7209 | 0.4537 | 0.4981 | 0.5266 | 0.4996 |
| gemma4_12b_awq × gemma4_31b | 1.0 | 0.8263 | 0.8263 | 0.3166 | 0.9462 | 0.709 | 0.6012 | 0.6521 | 0.8204 |
| gemma4_12b_awq × qwen3_6_35b | 1.0 | 0.7021 | 0.7021 | 0.2869 | 0.9258 | 0.6966 | 0.5011 | 0.6485 | 0.7395 |
| gemma4_12b_awq × gemma4_12b | 1.0 | 0.7705 | 0.7705 | 0.3142 | 0.9559 | 0.7167 | 0.5285 | 0.6931 | 0.8594 |
| gemma4_12b_awq × qwen3_4b | 1.0 | 0.7527 | 0.7527 | 0.2321 | 0.7679 | 0.5473 | 0.5426 | 0.5538 | 0.546 |
| qwen3_4b_instr_awq × gemma4_31b | 1.0 | 0.6047 | 0.6047 | 0.2972 | 0.8793 | 0.5191 | 0.4578 | 0.543 | 0.6806 |
| qwen3_4b_instr_awq × qwen3_6_35b | 1.0 | 0.5291 | 0.5291 | 0.2573 | 0.8199 | 0.5089 | 0.4097 | 0.5191 | 0.627 |
| qwen3_4b_instr_awq × gemma4_12b | 1.0 | 0.5747 | 0.5747 | 0.2779 | 0.9 | 0.5179 | 0.4308 | 0.5618 | 0.7046 |
| qwen3_4b_instr_awq × qwen3_4b | 1.0 | 0.5942 | 0.5942 | 0.2508 | 0.6441 | 0.358 | 0.4527 | 0.4741 | 0.4393 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| gemma4_31b_awq × gemma4_31b | 948 | 192 | 35 | 311 |
| gemma4_31b_awq × qwen3_6_35b | 766 | 325 | 70 | 313 |
| gemma4_31b_awq × gemma4_12b | 883 | 251 | 41 | 207 |
| gemma4_31b_awq × qwen3_4b | 837 | 281 | 57 | 470 |
| qwen3_6_35b_awq × gemma4_31b | 671 | 278 | 59 | 414 |
| qwen3_6_35b_awq × qwen3_6_35b | 538 | 356 | 99 | 369 |
| qwen3_6_35b_awq × gemma4_12b | 638 | 306 | 64 | 294 |
| qwen3_6_35b_awq × qwen3_4b | 592 | 336 | 80 | 579 |
| gemma4_12b_awq × gemma4_31b | 937 | 197 | 38 | 314 |
| gemma4_12b_awq × qwen3_6_35b | 766 | 325 | 70 | 364 |
| gemma4_12b_awq × gemma4_12b | 873 | 260 | 39 | 207 |
| gemma4_12b_awq × qwen3_4b | 843 | 277 | 52 | 481 |
| qwen3_4b_instr_awq × gemma4_31b | 586 | 383 | 142 | 711 |
| qwen3_4b_instr_awq × qwen3_6_35b | 473 | 421 | 184 | 626 |
| qwen3_4b_instr_awq × gemma4_12b | 546 | 404 | 161 | 581 |
| qwen3_4b_instr_awq × qwen3_4b | 571 | 390 | 150 | 947 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados