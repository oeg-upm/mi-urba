# miurba-extractor

Extrae información estructurada de quejas vecinales (texto) con un LLM.
Lee el `complaints.jsonl` producido por `miurba-generator` y produce
`extractions.jsonl` conforme a `schemas/extraction_schema.json`.

## Uso

Todos los parámetros se leen de un config. Único argumento: `--config`
(por defecto, `config/config.yaml`).

```bash
python cli.py extract                          # usa config/config.yaml
python cli.py extract --config config/config-jsonschema.yaml
```

Entrada: `<raw_dir>/<complaints_file>` (producido por el generador — apunta
`extraction.raw_dir` a su salida). Salida:
`<annotated_dir>/<extractions_file>`.

### `config/`

| Fichero | Uso |
|---|---|
| `config.yaml` | Default: endpoint genérico vía `OPENAI_API_*` |
| `config-jsonschema.yaml` | Ejemplo con `structured_output: json_schema` (vLLM) |

Ambas referencian `taxonomy: ../../schemas/taxonomy.yaml`. Los secretos van
en `miurba-extractor/.env` (copia `.env.example`), referenciados como `${VAR}`.

## Salida

`extractions.jsonl` — un registro por queja:

```json
{
  "complaint_id": "Q0001",
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
enums sincronizados desde `taxonomy.yaml`) independientemente del modo de
muestreo del modelo. Un registro inválido cuenta como fallo y no se escribe.

## Configuración

Secciones consumidas: `extraction`, `models`, `endpoints`, `paths`,
`logging` y la taxonomía de `schemas/taxonomy.yaml` (`categories`,
`affected_elements`, `location_zones`, `risks`, `urgency_levels`). Defaults
comentados en `config/config.yaml`.

### `extraction:`

| Clave | Función | Default |
|---|---|---|
| `model` | **Obligatorio.** Modelo extractor (clave de `models:`) | — |
| `temperature` | Temperatura de extracción | `0.2` |
| `default_output_dir` | Base de rutas; `null` → subdirectorios de `paths:` del propio extractor | `null` |
| `raw_dir` / `annotated_dir` | Directorios explícitos | `null` → `<base>/<subdir de paths:>` |
| `resume` | Saltar quejas ya extraídas | `true` |
| `timeout`, `max_retries`, `retry_delay` | Cliente LLM | `300`, `3`, `5.0` |
| `model_load_wait` | Paciencia (s) con modelos en frío; `null` = off | `null` |
| `workers` | Extracción en paralelo | `1` |

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

Cualquier endpoint OpenAI-compatible. `json_schema` requiere soporte de
structured outputs en el servidor.

## Tests

```bash
python -m pytest tests/
```

Sin red. Ejecuta pytest dentro de este directorio (no desde la raíz del repo).
