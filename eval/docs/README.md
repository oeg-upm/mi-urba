# eval/docs — Documentación de experimentos

| Documento | Experimento |
|---|---|
| `familia1_cross_perplexity.md` | Cross-perplexity entre generadores: conceptos, glosario, protocolo y métricas derivadas (KLD, Jeffreys, dH) |
| `experimento_seleccion_generadores.md` | Selección de los 10 generadores: cribado de 89 modelos y dos iteraciones (TF-IDF → embeddings + linajes) |
| `experimento_caracterizacion_fingerprints.md` | Caracterización multidimensional de los 87 generadores: embeddings, estilometría y rigidez (familias 2–4) |
| `experimento_extraccion_llm.md` | Evaluación del extractor LLM contra ground truth (matching, métricas por campo) |
| `experimento_iteraciones_prompt.md` | Optimización del prompt del generador en 8 rondas usando la extracción como métrica (ganadora: iter 6) |
| `experimento_datasets.md` | Corpora generados a escala: 854, 5K, 10K, 50K (marcado "do not use") y benchmark |

Cadena causal entre experimentos: selección de generadores → datasets → evaluación del
extractor → iteraciones del prompt (adopta iter 6) → caracterización multidimensional →
re-selección por linajes → dataset 10K.
