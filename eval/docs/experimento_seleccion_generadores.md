# Experimento de selección de generadores (dos iteraciones)

> Fuentes: `eval/experiments/seleccion_modelos/select_models.py`, `eval/experiments/seleccion_modelos/analyze_diversity.py`,
> `eval/experiments/seleccion_modelos/generate_diversity_dataset.py`, `mi-urba-eval-old/measure_load_times.py` (fuera del repo),
> `mi-urba-eval-old/test_all_generators.py` (fuera del repo), commit `9b8d95c` (P4 prep).
> Resultados: `eval/results/diversity/{models,fast_models,selected_10,selected_10_10k}.json`,
> `similarity_matrix.json`, `model_quality.json`, `load_times.log`, `gen_all_models.log`.

---

## 1. La pregunta

El generador de quejas puede usar cualquier modelo del gateway `local_llama` (llama-swap
sobre una VM de GPU). Pero no cualquier modelo sirve, y usar muchos a la vez no es
operativo. La pregunta que este experimento resuelve, en dos iteraciones:

> **¿Qué modelos del gateway usamos como generadores de quejas, y cuáles son los 10
> que más valor aportan?**

Ambas iteraciones responden lo mismo con la misma estructura —filtrar por calidad,
medir distancia entre modelos, elegir los más distintos— y difieren en la **métrica de
distancia**, que es justamente lo que la caracterización multidimensional
(`experimento_caracterizacion_fingerprints.md`) puso a punto entre una y otra.

## 2. Glosario

| Término | Definición |
|---|---|
| **Gateway** | llama-swap en la VM de GPU: expone decenas de modelos tras un endpoint común, cargando uno a la vez |
| **Distintividad** | distancia media de un modelo a todos los demás en la matriz de similaridad; los modelos redundantes sobran |
| **Greedy max-min** | algoritmo de selección: empieza por el modelo más lejano del resto y en cada paso añade el que maximiza su distancia mínima a los ya elegidos → cubre el espacio de estilos con k modelos |
| **Linaje** | familia real de un modelo (p. ej. las 6 cuantizaciones de qwen3.5-122b son el mismo modelo); dos cuantizaciones del mismo linaje no aportan diversidad real |
| **Texto garbled** | texto corrupto (caracteres rotos, repetición degenerada), típico de modelos mal cuantizados |

## 3. Fase 0 — Cribado masivo (¿quién puede generar?)

Antes de elegir, hubo que saber qué modelos del gateway funcionaban como generadores:

1. **Test de generación** (`mi-urba-eval-old/test_all_generators.py` + retests, fuera del repo):
   probe de los 89 modelos registrados con el prompt real. Resultado: 88 OK, 1 fallido.
2. **Medición de coste** (`mi-urba-eval-old/measure_load_times.py` y `load_all_models.sh`, fuera del repo): para cada
   modelo, tiempo de **load** (cargar pesos en GPU), **warm** (primera inferencia) y
   **gen** (generar una queja). Filtro de operatividad: load < 5 min, warm < 30 s →
   **81 modelos fast** (`fast_models.json`).
3. Logs: `gen_all_models.log`, `load_times.log`.

## 4. Generación del corpus de comparación

Con los modelos válidos se genera el corpus sobre el que se medirá todo:

- **20 quejas por modelo** a partir de **20 escenarios presampleados con seed fija**
  (`scenarios.json`): todos los modelos reciben exactamente los mismos estímulos.
- Secuencial por modelo (una petición cada vez) con 30 s de enfriamiento entre modelos
  para que llama-swap libere el anterior.
- Prompt de producción real (iter 6 tras su adopción).
- Salida: `diversity/gen_<model>/complaints.jsonl` (81 directorios) + `generation.log`.

## 5. Iteración 1 — Selección con TF-IDF (26 ago 2026)

### 5.1 Métrica de distancia

**TF-IDF + coseno** (`analyze_diversity.py`, tokenizer español propio: minúsculas,
sin puntuación, palabras > 2 letras): centroide TF-IDF por modelo (media de sus 20
textos) y coseno entre centroides → matriz 81×81 (`similarity_matrix.json`).
Añade métricas de calidad por modelo (`model_quality.json`).

### 5.2 Filtro de calidad (`select_models.py`)

Barrera mínima para ser seleccionable — se mide sobre los 20 textos de cada modelo:

| Filtro | Umbral | Qué evita |
|---|---|---|
| Longitud media | 100–1500 chars | textos telegráficos o divagantes |
| Diversidad léxica | mínimo de vocabulario | repetición |
| Garbled | < 2% de caracteres no-ASCII-no-español | modelos mal cuantizados |
| Dominio | ≥ 60% de textos mencionan el dominio (fuga/agua/tubería/garaje…) | modelos que ignoran el escenario |

### 5.3 Selección

Greedy max-min sobre la matriz TF-IDF, entre los modelos que pasan el filtro →
`selected_10.json`: deepseek-r1-distill-qwen-7b, phi-4, qwen3-0.6b,
pantheon-rp-8b-llama3, mistral-7b-v0.3-base, deepseek-r1-distill-llama-8b,
granite-3.3-2b, olmo-2-13b-instruct, falcon3-7b-instruct, glm-4.7-flash.

### 5.4 Límite detectado

TF-IDF mide **solapamiento léxico**, no estilo ni semántica. Dos modelos pueden usar
las mismas palabras (fuga, vecino, comunidad — el dominio fuerza vocabulario común)
y aun así escribir de forma muy distinta, y viceversa. La caracterización
multidimensional (cross-PPL, embeddings, estilometría) mostró que la estructura real
del espacio es de **linajes** y de un eje estilo/semántica que TF-IDF no captura.

## 6. Iteración 2 — Selección con embeddings y linajes (post-caracterización)

Repite el mismo procedimiento con tres cambios informados por la caracterización:

1. **Métrica de distancia**: la matriz **fused de 7 jueces de embedding** (z-scored,
   ver `experimento_caracterizacion_fingerprints.md`) en vez de TF-IDF.
2. **Restricción de linaje**: un solo representante por linaje real (las 6 variantes de
   qwen3.5-122b compiten entre sí; elegida una, las demás quedan excluidas).
3. **Barrera de tamaño**: ≥ 7B — los modelos pequeños aportaban diversidad pero
   penalizaban calidad de texto.

Selección: greedy max-min + filtro de sanidad + ≥7B + un-representante-por-linaje →
`selected_10_10k.json`: qwen3.5-122b-a10b, devstral-small-2507, agentica-deepcoder-14b,
gpt-oss-120b, muse-glimmer-30b, gemma-3-12b-it, glm-4.5-air, olmo-2-13b-instruct,
falcon3-10b-instruct, openchat-3.5-0106.

## 7. Las dos selecciones en contexto

| | Iteración 1 | Iteración 2 |
|---|---|---|
| Fecha | 26 ago 2026 | post-caracterización (sep) |
| Distancia | TF-IDF coseno | embeddings 7 jueces (fused) |
| Restricciones | filtro de calidad | filtro + ≥7B + 1 por linaje |
| Conocimiento previo | ninguno | matriz fused + estructura de linajes |
| Qué alimentó | dataset-5k | dataset-10k |
| Fichero | `selected_10.json` | `selected_10_10k.json` |

Solo 1 de los 10 (olmo-2-13b) sobrevive de una selección a la otra: la métrica cambia
tanto la respuesta que los conjuntos son casi disjuntos. Es el mejor argumento de que
la elección de la métrica de distancia **es** la decisión central del experimento.

## 8. Conexión con otros experimentos

- La **caracterización multidimensional** (`experimento_caracterizacion_fingerprints.md`)
  es a la vez el medio de la iteración 2 y un resultado en sí mismo.
- Los corpora generados con cada selección están en
  `experimento_datasets.md` (5K y 10K).
- Los 4 generadores AWQ de `experimento_iteraciones_prompt.md` son del entorno de la
  primera iteración.
