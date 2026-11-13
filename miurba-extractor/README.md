# miurba-extractor

Extrae información estructurada de quejas vecinales (texto + fotos adjuntas)
con un VLM. Lee el `complaints.jsonl` producido por `miurba-generator`, usa un
modelo de visión para describir las fotos y produce
`extractions.jsonl` conforme a `schemas/extraction_schema.json`.

## Uso

Todos los parámetros se leen del config compartido. Único argumento:
`--config` (por defecto, `../config.yaml`).

```bash
python cli.py extract --config ../config.yaml
```

Entrada: `<raw_dir>/<complaints_file>` (producido por el generador) y los PNG
en `<images_dir>`. Salida: `<annotated_dir>/<extractions_file>`.

## Salida

`extractions.jsonl` — un registro por queja:

```json
{
  "complaint_id": "Q0001",
  "image_descriptions": ["caption de cada foto, en orden"],
  "extractions": [
    {
      "category": {"principal": "...", "subcategory": "..."},
      "affected_elements": [{"principal": "...", "subcategory": "..."}],
      "location": {"zone": "...", "detail": "..."},
      "risks": ["..."],
      "urgency": "..."
    }
  ]
}
```

Todo registro se valida contra `schemas/extraction_schema.json` (con los
enums sincronizados desde `config.yaml`) independientemente del modo de
muestreo del modelo. Un registro inválido cuenta como fallo y no se escribe.

## Dos modos de extracción

- **Single-call** (`stage_split: false`): el VLM ve texto y fotos en una
  llamada y produce descripciones + extracción.
- **Stage-split** (`stage_split: true`): dos llamadas — un modelo de visión
  (`extraction.vision_model`) describe las fotos (schema
  `image_description_schema.json`) y un modelo de texto extrae la estructura.
  Requiere fotos y `vision_model`; sin ellos cae automáticamente a single-call.

## Configuración

Secciones consumidas: `extraction`, `models`, `endpoints`, `paths`,
`logging` y la taxonomía (`categories`, `affected_elements`,
`location_zones`, `risks`, `urgency_levels`). Defaults comentados en
`config.yaml`.

### `extraction:`

| Clave | Función | Default |
|---|---|---|
| `model` | **Obligatorio.** Modelo VLM (clave de `models:`) | — |
| `temperature` | Temperatura de extracción | `0.2` |
| `default_output_dir` | Base de rutas; `null` → `generation.default_output_dir` | `null` |
| `raw_dir` / `annotated_dir` / `images_dir` | Directorios explícitos | `null` → `<base>/<subdir de paths:>` |
| `resume` | Saltar quejas ya extraídas | `true` |
| `timeout`, `max_retries`, `retry_delay` | Cliente LLM | `300`, `3`, `5.0` |
| `model_load_wait` | Paciencia (s) con modelos en frío; `null` = off | `null` |
| `workers` | Extracción en paralelo | `1` |
| `stage_split` | Pipeline de dos llamadas | `false` |
| `vision_model` | Modelo de visión para stage-split | `null` |

### `models:` — salida estructurada

`models.<nombre>.structured_output` declara lo que el **servidor** soporta:

| Modo | Comportamiento |
|---|---|
| `null` | Generación libre |
| `json_object` | `response_format: json_object` — JSON válido garantizado, sin schema |
| `json_schema` | `response_format: json_schema` — el servidor fuerza el schema completo (vLLM+XGrammar, llama.cpp) |

Con `json_schema`, `cli.py` carga el schema y lo envía en cada petición.
Recuerda: la validación final en local se aplica en todos los modos.

## Requisitos del endpoint

Cualquier endpoint OpenAI-compatible. Para modelos multimodales, el servidor
debe aceptar mensajes con `image_url` (data URL base64). `json_schema`
requiere soporte de structured outputs en el servidor.

## Tests

```bash
python -m pytest tests/
```

Sin red. Ejecuta pytest dentro de este directorio (no desde la raíz del repo).
