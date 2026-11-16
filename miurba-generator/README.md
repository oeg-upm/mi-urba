# miurba-generator

Genera datasets sintéticos de quejas vecinales: para cada registro muestrea un
escenario (categoría, urgencia, ubicación, riesgos, elementos afectados,
persona, comunidad) y pide a un LLM el texto del vecino (`user_text`) y un
`timestamp`. Opcionalmente genera las fotos adjuntas con FLUX.2.

## Uso

Todos los parámetros se leen del config compartido. Único argumento:
`--config` (por defecto, `../config.yaml`). El nivel de log se controla con
`logging.level`.

```bash
python cli.py generate --config ../config.yaml   # generar según el plan
python cli.py merge    --config ../config.yaml   # fusionar salidas por modelo
```

`generate` recorre el plan (`generation.batches` o `generation.models` +
`num_records`), escribe la salida de cada modelo en
`<default_output_dir>/<modelo>/` y fusiona todo al final en
`<default_output_dir>/<paths.raw_subdir>/`.

`configs/` contiene 5 configs de ejemplo (campaña de corpus, smoke multi-modelo,
smoke de iteración rápida, variante paramétrica y solo-merge). Todas son
autocontenidas y referencian la taxonomía con `taxonomy: taxonomy.yaml`; los
endpoints LAN de llama-swap son ilustrativos. Se usan con
`python cli.py generate --config configs/<f>.yaml` o vía
`docker-compose-example.yaml`.

## Salida

| Fichero | Contenido |
|---|---|
| `<out>/<modelo>/raw/complaints.jsonl` | `id`, `user_text`, `user_pics`, `timestamp` |
| `<out>/<modelo>/raw/ground_truth.jsonl` | extracciones esperadas (`complaint_id`, `extractions`) |
| `<out>/<modelo>/raw/metadata.jsonl` | persona, modifier, comunidad, `image_prompts` |
| `<out>/<modelo>/images/*.png` | fotos (solo con `image_generation.enabled: true`) |

## Configuración

Secciones del config que consume. Obligatorias para ejecutar:
`endpoints`, `models`, un plan en `generation` y el vocabulario de dominio
de `taxonomy.yaml` (referenciado con `taxonomy:` en el config:
`categories`, `affected_elements`, `location_zones`, `risks`,
`urgency_levels`, `communities`, `personas`, `modifiers`, `modifier_prob`,
`weights`, `urgency_weights`, `category_risks`, `category_elements`,
`category_zones`, `risk_expressions`, `urgency_expressions` y
`element_subcategories`).
Opcionales con defaults: `paths`, `image_generation`, `logging` y el resto de
claves de `generation`. Detalle y valores por defecto comentados en
`config.yaml` y `taxonomy.yaml`.

### `generation:`

| Clave | Función | Default |
|---|---|---|
| `default_output_dir` | Base de salida | `data` |
| `batches` | Plan `[{model, count}, ...]` | `[]` |
| `models` + `num_records` | Plan alternativo: lista de modelos | `null` |
| `start_id` | Primer ID (secuencial en el plan) | `1` |
| `seed` | Semilla de muestreo. Determinista solo con `workers: 1` | `null` |
| `resume` | Saltar registros ya generados | `true` |
| `timeout`, `max_retries`, `retry_delay` | Cliente LLM | `300`, `3`, `5.0` |
| `model_load_wait` | Paciencia (s) con modelos en frío; `null` = off | `null` |
| `workers` | Generación en paralelo | `1` |
| `validate` | Validación en 2ª pasada (juez LLM) | `false` |
| `max_validation_retries` | Regeneraciones si el juez rechaza | `3` |
| `validator` | Modelo juez; `null` = el propio generador | `null` |

### `models:` y `endpoints:`

```yaml
models:
  qwen3.5:
    endpoint: llama_swap
    model: qwen3.5-122b-a10b
    temperature: 0.85
    extra_params:              # claves verbatim al payload (opcional)
      enable_thinking: false

endpoints:
  llama_swap:
    base_url: http://llama-swap.local:9292
    connection_backoff: true   # backoff generoso ante caídas (opcional)
  neuralwatt:
    base_url: https://api.neuralwatt.com/v1
    api_key: ${NEURALWATT_API_KEY}   # secreto: resolver desde .env
```

### `image_generation:` (opcional, requiere GPU)

`enabled: false` por defecto. Requiere `pip install -r requirements-images.txt`
y un host con GPUs. Si se activa sin GPU o dependencias, el pipeline continúa
sin imágenes. CLI standalone: `flux2_imagegen.py --metadata data/raw/metadata.jsonl`.

## Validación con juez (opcional)

Con `validate: true`, tras generar cada texto un modelo juez comprueba que
alinea con el escenario (categoría, subcategoría, urgencia, elementos,
riesgos, ubicación). Si no alinea, regenera hasta `max_validation_retries`
veces y conserva el último texto. El juez fallido o degradado se refleja en el
resumen del run (`validation: ok/rejected/degraded`).

## Tests

```bash
python -m pytest tests/
```

Sin red. Ejecuta pytest dentro de este directorio (no desde la raíz del repo).
