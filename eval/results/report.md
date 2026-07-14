# Evaluación del extractor — Resultados

**Dataset:** 99 quejas con ground truth

## Métricas por modelo

| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |
|--------|-----------|--------|----|-------------|--------------------|---------------------|
| kimi-k2.6 | 0.8847 | 0.936 | 0.9006 | 0.8182 | 1.2 | 1.47 |
| kimi-k2.7-code | 0.8771 | 0.9327 | 0.8936 | 0.8081 | 1.2 | 1.58 |
| qwen3.5-397b | 0.846 | 0.9007 | 0.8635 | 0.7677 | 1.2 | 1.38 |
| qwen3.6-35b | 0.8779 | 0.9125 | 0.8888 | 0.8182 | 1.2 | 1.3 |
| minimax-m3 | 0.8636 | 0.9428 | 0.8886 | 0.7677 | 1.2 | 1.58 |
| gemma4-31b | 0.851 | 0.9327 | 0.877 | 0.7475 | 1.2 | 1.58 |
| glm-5.1 | 0.8636 | 0.9108 | 0.8785 | 0.798 | 1.2 | 1.43 |

## Accuracy por campo (sobre extracciones emparejadas)

| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |
|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|
| kimi-k2.6 | 1.0 | 0.8839 | 0.8839 | 0.375 | 0.8661 | 0.6161 | 0.3103 | 0.0662 | 0.0521 |
| kimi-k2.7-code | 1.0 | 0.8829 | 0.8829 | 0.3694 | 0.8649 | 0.5946 | 0.292 | 0.0731 | 0.0601 |
| qwen3.5-397b | 1.0 | 0.8318 | 0.8318 | 0.271 | 0.8318 | 0.6075 | 0.2601 | 0.0844 | 0.0592 |
| qwen3.6-35b | 1.0 | 0.8333 | 0.8333 | 0.3519 | 0.8519 | 0.6296 | 0.2315 | 0.0694 | 0.0448 |
| minimax-m3 | 1.0 | 0.8839 | 0.8839 | 0.3304 | 0.8571 | 0.5893 | 0.2741 | 0.0908 | 0.0595 |
| gemma4-31b | 1.0 | 0.9009 | 0.9009 | 0.3964 | 0.8649 | 0.5946 | 0.3026 | 0.0593 | 0.048 |
| glm-5.1 | 1.0 | 0.8981 | 0.8981 | 0.2963 | 0.8796 | 0.6019 | 0.2793 | 0.0895 | 0.0586 |

## Tipos de match

| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |
|--------|-------|---------|-------------------|------------------|
| kimi-k2.6 | 99 | 13 | 7 | 34 |
| kimi-k2.7-code | 98 | 13 | 8 | 45 |
| qwen3.5-397b | 89 | 18 | 12 | 30 |
| qwen3.6-35b | 90 | 18 | 11 | 21 |
| minimax-m3 | 99 | 13 | 7 | 44 |
| gemma4-31b | 100 | 11 | 8 | 45 |
| glm-5.1 | 97 | 11 | 11 | 34 |

## Notas

- **Exact**: categoría principal + subcategoría coinciden
- **Partial**: solo categoría principal coincide
- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó
- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT
- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)
- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados