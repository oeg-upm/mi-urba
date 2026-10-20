# E1+E3 · Selección y caracterización de generadores

Un experimento con dos iteraciones sobre la misma pregunta: **¿qué modelos del gateway
usamos como generadores de quejas?** La primera iteración seleccionó 10 modelos con
TF-IDF; la caracterización multidimensional (cross-perplexity, embeddings, estilometría,
rigidez) permitió re-seleccionar con una métrica mejor y restricción de linajes.

- Doc extendido: `../../docs/experimento_seleccion_generadores.md`
- Familia 1 (cross-perplexity): `../../docs/familia1_cross_perplexity.md`
- Familias 2–4: `../../docs/experimento_caracterizacion_fingerprints.md`

## Mapa de artefactos

| Tipo | Ruta | Nota |
|---|---|---|
| Código | `./generate_diversity_dataset.py` | generación 20 quejas × modelo (corpus común) |
| Código | `./analyze_diversity.py`, `./select_models.py` | iteración 1 (TF-IDF + filtros) |
| Código | `./cross_perplexity_unified.py`, `./run_cross_perplexity.sh`, `./run_gateway_matrix.sh` | familia 1 |
| Código | `./embed_similarity.py`, `./emb_endpoint.py` | familia 2 |
| Código | `./stylometry_analysis.py`, `./rigidity_probe.py`, `./final_analysis.py`, `./analyze_cross_perplexity.py` | familias 3, 4 y consolidación |
| Infra | `./llama_cpp_prompt_logprobs.patch`, `./llama_cpp_xppl.Dockerfile` | servidor `-xppl` |
| Corpus | `../../results/diversity/gen_<model>/complaints.jsonl` (81 modelos × 20), `scenarios.json` | insumo de las 4 familias |
| Datos | `../../results/diversity/{similarity_matrix,model_quality,selected_10,selected_10_10k,models,fast_models,chat_templates}.json` | iteración 1 y cribado |
| Datos | `../../results/diversity/cross_perplexity_gateway{,_big,_full}.json`, `xppl_gen_cache_gateway*.json` | familia 1 |
| Datos | `../../results/diversity/embedding_similarity{,_7j}.json`, `mds_voces_embeddings{,_7j}.png` | familia 2 |
| Datos | `../../results/diversity/stylometry_analysis.json`, `stylometry_mds_e1.png` | familia 3 |
| Datos | `../../results/diversity/rigidity_probe.json` | familia 4 |
| Informe | `../../results/diversity/final_numbers.json`, `fig_correlations_4metrics.png`, `fig_rigidity_vs_distinctiveness.png`, `llm_similarity_report.tex/.pdf` | consolidación |
| Logs | `../../results/diversity/generation.log`; `../../results/{gen_all_models,gen_retest_*,load_times}.log` | generación y cribado |
| Código archivado | `/backup/pc1/data/miUrba2/mi-urba-eval-old/` (archivo histórico, fuera del repo) (v1–v4 cross-perplexity, tests, load-times) | cross-perplexity v1–v4, tests de generadores, load-times |

Reproducibilidad: análisis determinista sobre los JSONs (100% re-ejecutable); la
recolecta requiere el gateway con los modelos `-xppl` y el parche de llama.cpp.
