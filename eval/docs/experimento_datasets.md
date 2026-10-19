# Experimento de datasets a escala

> Fuentes: `eval/experiments/datasets/generate_5k_dataset.py`, `eval/experiments/datasets/generate_10k_dataset.py`,
> `eval/experiments/seleccion_modelos/generate_diversity_dataset.py`, `eval/experiments/datasets/explore_10k.py`, `eval/experiments/datasets/generate_5k.sh`,
> `eval/experiments/datasets/run_10k_tonight.sh`, `eval/experiments/datasets/extract_eval_5k.py`.
> Resultados: `eval/results/dataset{,-5k,-10k,-50k,-bench}/`, logs `*.log`.

---

## 1. Objetivo

Producir corpus de quejas vecinales sintéticas a distintas escalas, con **ground truth
estructural completa** (categorías, urgencia, riesgos, elementos), para alimentar la
evaluación del extractor y los experimentos de selección. Cada dataset es una variante
de la misma receta que cambia en escala y en qué generadores participan.

## 2. Receta común

Todos los datasets comparten el mismo pipeline:

1. **Presamplear escenarios** con seed fija: problema(s) + comunidad vecinal + persona
   + modificador. El escenario es la estructura de la queja (y su ground truth).
2. **Generar** el texto con cada modelo mediante su prompt de producción
   (workers paralelos, warmup con reintentos, logs por modelo).
3. **Persistir** por modelo: `complaints.jsonl` (texto), `ground_truth.jsonl`
   (estructura del escenario), `metadata.jsonl` (modelo, escenario, tiempos).

Diferencias entre variantes: número de quejas por modelo, qué modelos participan
(y por tanto qué prompt de producción tenían vigente) y destino del corpus.

## 3. Variantes

### 3.1 `dataset/` — 854 quejas (baseline de evaluación)

- **Modelos**: kimi-k2.6, qwen3_local, glm-5.1 (3 generadores del diseño original).
- **Propósito**: corpus principal del experimento de extracción LLM
  (`experimento_extraccion_llm.md`).
- **Resultado**: 854 quejas con GT; `metrics.json`/`report.md` en `results/` se calcularon
  sobre él.

### 3.2 `dataset-5k/` — 500 quejas × 10 modelos

- **Modelos**: los 10 de la **primera selección** (TF-IDF): deepseek-r1-distill-qwen-7b,
  phi-4, qwen3-0.6b, pantheon-rp-8b-llama3, mistral-7b-v0.3-base,
  deepseek-r1-distill-llama-8b, granite-3.3-2b, olmo-2-13b-instruct,
  falcon3-7b-instruct, glm-4.7-flash.
- **Ejecución**: `generate_5k_dataset.py` (workers=8) tras `generate_5k.sh`; log en
  `dataset-5k.log`. Diseñado para ejecutarse cuando el backend se recuperase
  (llama-swap estaba caído en el momento del diseño).
- **Extracción asociada**: `extract_eval_5k.py` — extracción con gemma4-12b
  (output restringido) + evaluación.

### 3.3 `dataset-10k/` — ~1000 quejas × 10 modelos

- **Modelos**: los 10 de la **segunda selección** (embeddings 7 jueces, un representante
  por linaje): qwen3.5-122b-a10b, devstral-small-2507, agentica-deepcoder-14b,
  gpt-oss-120b, muse-glimmer-30b, gemma-3-12b-it, glm-4.5-air, olmo-2-13b-instruct,
  falcon3-10b-instruct, openchat-3.5-0106.
- **Pre-exploración**: `explore_10k.py` lanza 10 generaciones por modelo en cola única
  con el prompt real de producción, y estima latencia media/queja y ETA de las 1000 →
  `dataset-10k_exploration.json`. Sirvió para decidir viabilidad y orden antes de
  comprometer horas de GPU.
- **Ejecución**: `run_10k_tonight.sh` (lote nocturno); `generation_10k.log`.

### 3.4 `dataset-50k/` — 50K quejas × 7 modelos AWQ

- **Modelos**: gemma4 12b/26b/31b AWQ, qwen3-4b-instr, qwen3.5-4b, qwen3.5-27b,
  qwen3.6-35b (todos AWQ = rápidos en la GPU local).
- **Estado**: ⚠️ marcado "do not use" en TASK_ITER — se detectaron problemas de calidad
  en la generación masiva. Conservado con sus logs de recuperación
  (`master_log.txt`, `recover_log*.txt`, `progress.txt`, script `mi-urba-eval-old/recover_missing.sh` (fuera del repo))
  y con `consolidated/` y `sample_1000/` para inspección.

### 3.5 `dataset-bench/` — benchmark one-off

Generación puntual con `mi-urba-eval-old/run_bench_generation{,_v2}.sh` (fuera del repo). Directorio vacío en el
estado actual; el valor fue la prueba del procedimiento, no el corpus.

## 4. Decisiones transversales

| Decisión | Motivo |
|---|---|
| Seed fija y escenarios presampleados | mismos estímulos entre modelos → comparabilidad |
| Un directorio por modelo (`gen_<model>/`) | generación incremental y reanudable por modelo |
| Log por modelo + log maestro | trazabilidad de fallos parciales sin perder el conjunto |
| Prompts de producción reales | los corpora miden la diversidad/calidad del sistema, no de un prompt artificial |
| Reanudación/recuperación (50K) | runs nocturnos largos con servidor inestable: `recover_missing.sh` regenera solo las quejas que faltan |

## 5. Qué dataset usar para qué

| Necesitas | Usa |
|---|---|
| Evaluar extractor con histórico de resultados | `dataset/` (854) |
| Corpus multi-generador pequeño, primera selección | `dataset-5k/` |
| Corpus multi-generador, selección por linajes | `dataset-10k/` |
| Nada (calidad comprometida) | `dataset-50k/` — solo inspección |

## 6. Conexión con otros experimentos

- El prompt de producción usado en 5K/10K/diversidad es el de **iter 6**
  (`experimento_iteraciones_prompt.md`).
- Los modelos de cada dataset son el output de las dos selecciones descritas en
  `experimento_seleccion_generadores.md`.
- El `dataset/` de 854 es el que evalúa el extractor (`experimento_extraccion_llm.md`).
