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
| `temperature` | Temperatura de extracción; si no se declara, se usa la del modelo (`models.<nombre>.temperature`) | `0.2` |
| `default_output_dir` | Base de rutas; `null` → `generation.default_output_dir` | `null` |
| `raw_dir` / `annotated_dir` | Directorios explícitos | `null` → `<base>/<subdir de paths:>` |
| `resume` | Saltar quejas ya extraídas y podar las extracciones cuyo `complaint_id` ya no existe en el corpus de entrada | `true` |
| `timeout`, `max_retries`, `retry_delay` | Cliente LLM | `300`, `3`, `5.0` |
| `model_load_wait` | Paciencia (s) con modelos en frío; `null` = off | `null` |
| `workers` | Extracción en paralelo | `1` |
| `constrain_vocabulary` | Restringir categoría/subcategoría y elementos afectados al vocabulario de la taxonomía (modo cerrado) | `false` |

### Modo cerrado y modo abierto

El schema con el que se valida localmente siempre impone los enums de `risks`,
`urgency_levels` y `location_zones`. Los campos de clasificación (categoría y
elementos afectados) son libres por defecto (modo abierto); con
`constrain_vocabulary: true` el schema los restringe a la taxonomía (modo
cerrado), de modo que el servidor y la validación local rechazan cualquier
etiqueta que no exista en el vocabulario.

### `models:` — salida estructurada

`models.<nombre>.structured_output` declara lo que el **servidor** soporta:

| Modo | Comportamiento |
|---|---|
| `null` | Generación libre |
| `json_object` | `response_format: json_object` — JSON válido garantizado, sin schema |
| `json_schema` | `response_format: json_schema` — el servidor fuerza el schema completo (vLLM+XGrammar, llama.cpp) |

Con `json_schema`, `cli.py` carga el schema y lo envía en cada petición.
Recuerda: la validación final en local se aplica en todos los modos.

## Rechazos y reanudación

Los registros que fallan (JSON inválido, salida no conforme al schema, caída
del modelo) se cuentan como fallidos y quedan auditados en `rejected.jsonl`,
con su motivo. Un fallo inesperado en un registro no aborta la ejecución.

Al reanudar, las extracciones cuyo `complaint_id` ya no existe en el corpus de
entrada se descartan; si el corpus de entrada está vacío no se poda nada.

Limitación conocida: la reanudación se basa en el identificador. Si un registro
se regenera con otro texto manteniendo el mismo `id`, la extracción anterior se
conserva y queda desalineada. El corpus no guarda una huella del texto de
origen; hasta que la haya, regenerar el corpus exige vaciar `extractions.jsonl`
(`resume: false`).

Códigos de salida: `2` si falta el fichero de entrada, `1` si todos los
registros pendientes fallaron, `0` en cualquier otro caso (incluida una
reanudación en la que ya estaba todo extraído).

## Requisitos del endpoint

Cualquier endpoint OpenAI-compatible. `json_schema` requiere soporte de
structured outputs en el servidor.

## Tests

```bash
python -m pytest tests/
```

Sin red. Ejecuta pytest dentro de este directorio (no desde la raíz del repo).
