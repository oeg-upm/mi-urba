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
| `<out>/<modelo>/raw/metadata.jsonl` | `complaint_id`, `persona`, `modifier`, `community` |
| `<out>/<modelo>/raw/rejected.jsonl` | registros descartados con motivo y etapa (auditoría) |
| `<out>/<modelo>/raw/conversations.jsonl` | transcripción completa por registro (auditoría) |

`rejected.jsonl` y `conversations.jsonl` se escriben **por modelo** y no se
fusionan en `raw/`: son material de auditoría, no parte del corpus.

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
| `seed` | Semilla de muestreo. Determinista solo con `workers: 1` | `null` |
| `resume` | Saltar registros ya generados | `true` |
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
posterior lo reintenta con un escenario nuevo. El juez fallido o degradado se
refleja en el resumen del run (`validation: ok/rejected/degraded`) y **nunca
cuenta como aprobación** (fail-closed).

## Tests

```bash
python -m pytest tests/
```

Sin red. Ejecuta pytest dentro de este directorio (no desde la raíz del repo).
