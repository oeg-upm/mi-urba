# miurba-generator

Genera datasets sintéticos de quejas vecinales: para cada registro muestrea un
escenario (categoría, urgencia, ubicación, riesgos, elementos afectados,
persona, comunidad) y pide a un LLM el texto del vecino (`user_text`) y un
`timestamp`.

## Uso

Todos los parámetros se leen de un config. Único argumento: `--config`
(por defecto, `config/config.yaml`). El nivel de log se controla con
`logging.level`.

```bash
python cli.py generate                # genera según el plan de config/config.yaml
python cli.py merge                   # fusiona salidas por modelo
python cli.py generate --config config/config-smoke.yaml
```

`generate` recorre el plan (`generation.batches` o `generation.models` +
`num_records`), escribe la salida de cada modelo en
`<default_output_dir>/<modelo>/` y fusiona todo al final en
`<default_output_dir>/<paths.raw_subdir>/`.

### `config/`

| Fichero | Uso |
|---|---|
| `config.yaml` | Default: endpoint genérico vía `OPENAI_API_*`, plan pequeño |
| `config-smoke.yaml` | Iteración rápida: 1 generador × 50 con juez, resume: false |
| `config-corpus.yaml` | Campaña de corpus: N generadores × M intentos, juez, resume |
| `config-merge.yaml` | Solo `merge` (no genera) |

Todas referencian la taxonomía compartida con
`taxonomy: ../../schemas/taxonomy.yaml`. La configuración de ejemplo usa las
variables genéricas `OPENAI_API_ENDPOINT`, `OPENAI_API_MODEL` y
`OPENAI_API_KEY` (en `miurba-generator/.env`, copia `.env.example`); los ids
`your-*-model` de los ejemplos smoke/corpus son placeholders a sustituir.

## Salida

| Fichero | Contenido |
|---|---|
| `<out>/<modelo>/raw/complaints.jsonl` | `id`, `user_text`, `timestamp` |
| `<out>/<modelo>/raw/ground_truth.jsonl` | extracciones esperadas (`complaint_id`, `extractions`) |
| `<out>/<modelo>/raw/metadata.jsonl` | `complaint_id`, persona, modifier, comunidad, procedencia |
| `<out>/<modelo>/raw/rejected.jsonl` | registros descartados con motivo y etapa (auditoría) |
| `<out>/<modelo>/raw/conversations.jsonl` | transcripción completa por registro (auditoría) |
| `<out>/<modelo>/manifest.json` | manifiesto de la campaña: versiones, semilla y política de muestreo, procedencia y una entrada por pasada con sus recuentos |

`rejected.jsonl` y `conversations.jsonl` se escriben **por modelo** y no se
fusionan en `raw/`: son material de auditoría, no parte del corpus.

`merge` deduplica por identificador (gana la primera aparición), escribe de
forma atómica (un merge fallido no trunca el corpus anterior), avisa si los tres
ficheros difieren en longitud y termina con código `2` si no encuentra ningún
directorio de modelo que fusionar.

Cada registro se valida contra `schemas/complaint_schema.json`,
`schemas/extraction_schema.json` y `schemas/metadata_schema.json` antes de
escribirse; si no conforma, se descarta y queda en `rejected.jsonl` con el
error del schema (`stage: contract`). El `timestamp` se normaliza a ISO 8601 y
un valor ausente o no parseable implica descarte.

`metadata.jsonl` incluye `provenance`: modelo (clave y id), semilla, hash del
config efectivo, huellas SHA-256 de las plantillas de prompt empleadas y la
marca de generación.

## Configuración

Secciones del config que consume. Obligatorias para ejecutar:
`endpoints`, `models`, un plan en `generation` y el vocabulario de dominio
de `schemas/taxonomy.yaml` (referenciado con `taxonomy:` en el config:
`categories`, `affected_elements`, `location_zones`, `risks`,
`urgency_levels`, `communities`, `personas`, `modifiers`, `modifier_prob`,
`weights`, `urgency_weights`, `category_risks`, `category_elements`,
`category_zones`, `risk_expressions`, `urgency_expressions` y
`element_subcategories`).
Opcionales con defaults: `paths`, `logging` y el resto de
claves de `generation`. Detalle y valores por defecto comentados en
`config/config.yaml` y `schemas/taxonomy.yaml`.

### `generation:`

| Clave | Función | Default |
|---|---|---|
| `default_output_dir` | Base de salida | `data` |
| `batches` | Plan `[{model, count}, ...]` | `[]` |
| `models` + `num_records` | Plan alternativo: lista de modelos | `null` |
| `start_id` | Primer ID (secuencial en el plan) | `1` |
| `seed` | Semilla de muestreo. Cada registro deriva su propio flujo de `Random(seed:id)`: el escenario no depende del orden, de `workers` ni de los reintentos. Reproduce escenario y perfil, **no** el texto del modelo | `null` |
| `resume` | Saltar los registros ya completos. Un registro presente en `complaints` pero no en `ground_truth`/`metadata` (escritura interrumpida) se elimina de los tres ficheros y se regenera | `true` |
| `timeout`, `max_retries`, `retry_delay` | Cliente LLM | `300`, `3`, `5.0` |
| `model_load_wait` | Paciencia (s) con modelos en frío; `null` = off | `null` |
| `workers` | Generación en paralelo | `1` |
| `validate` | Validación en 2ª pasada (juez LLM) | `false` |
| `max_validation_retries` | Regeneraciones si el juez rechaza | `3` |
| `validator` | Modelo juez (**obligatorio** si `validate: true`); tiene que ser distinto del generador | — |

### `models:` y `endpoints:`

```yaml
models:
  generator:
    endpoint: openai
    model: ${OPENAI_API_MODEL}
    temperature: 0.85
    extra_params:              # claves verbatim al payload (opcional)
      enable_thinking: false

endpoints:
  openai:
    base_url: ${OPENAI_API_ENDPOINT}
    api_key: ${OPENAI_API_KEY}        # secreto: resolver desde .env
    connection_backoff: true          # backoff generoso ante caídas (opcional)
```

### Validación con juez (opcional)

Con `validate: true`, tras generar cada texto un modelo juez comprueba que
alinea con el escenario (categoría, subcategoría, urgencia, elementos,
riesgos, ubicación). Si no alinea, regenera hasta `max_validation_retries`
veces; si se agota el presupuesto el registro **se descarta** (no se escribe
nunca) y queda auditado en `rejected.jsonl` con el motivo. Una reanudación
posterior lo reintenta; con `seed` fijada el reintento reproduce el mismo
escenario (el flujo se deriva de `seed:id`), así que para cambiar el escenario
hay que cambiar el identificador o generar sin semilla. El juez fallido o
degradado se refleja en el resumen del run (`validation: ok/rejected/degraded`)
y **nunca cuenta como aprobación** (fail-closed).

## Tests

```bash
python -m pytest tests/
```

Sin red. Ejecuta pytest dentro de este directorio (no desde la raíz del repo).
