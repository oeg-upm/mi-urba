# miurba-generator

Generador de **datasets sintéticos de quejas/incidencias** reportadas por
vecinos de una comunidad de vecinos. Genera el **texto del vecino** y, de forma
**opcional**, las **imágenes** (fotos) adjuntas.

## Qué hace

Para cada queja:

1. Muestrea un escenario (categoría, urgencia, ubicación, riesgos, elementos
   afectados), una comunidad, un canal, una persona y un modificador de estilo.
2. Pide a un LLM (OpenAI-compatible) que genere `user_text` + `timestamp`.
   **El LLM NO describe las imágenes**: describir las fotos es tarea del
   extractor.
3. Si la generación de imágenes está activada (`image_generation.enabled` en la
   config compartida), construye prompts de imagen a partir del **scenario**
   (no de descripciones textuales) y genera los PNG con FLUX.2-dev (multi-GPU).

## Salida

- `data/raw/complaints.jsonl` — la queja (`id, user_text, user_pics, timestamp`).
  `user_pics` son nombres `.png`.
- `data/raw/ground_truth.jsonl` — extracciones esperadas (complaint_id +
  `extractions`), conforme a `extraction_schema.json`.
- `data/raw/metadata.jsonl` — persona, modifier, comunidad y `image_prompts`
  (los prompts usados para generar las imágenes).
- `data/images/*.png` — las fotos reales (solo si la generación de imágenes
  está activada).

El contrato de salida es `schemas/complaint_schema.json` (en la raíz del repo).

## Uso (config-only)

Todos los parámetros se leen del config compartido (`../config.yaml`, o el
que se indique con `--config`). No hay flags de parámetros.

```bash
python cli.py generate --config ../config.yaml  # generar según generation.batches/models
python cli.py merge    --config ../config.yaml  # fusionar salidas por modelo
```

Secciones del config que gobiernan la generación (ver defaults comentados
en el propio `config.yaml`):

| Clave (`generation:`) | Función |
|---|---|
| `default_output_dir` | Base de salida; cada modelo escribe en `<dir>/<model>/` |
| `batches` | Plan `[{model, count}, ...]` |
| `models` + `num_records` | Plan alternativo: lista explícita de modelos |
| `start_id` | Primer ID (secuencial a lo largo del plan) |
| `seed` | Semilla de muestreo (`null` = no determinista) |
| `resume` | Saltar registros ya generados |
| `timeout`, `max_retries`, `retry_delay` | Cliente LLM |
| `validate` | Validación en 2ª pasada (juez LLM verifica alineación texto/escenario) |
| `max_validation_retries` | Regeneraciones máximas si el juez rechaza (se conserva el último texto) |
| `validator` | Modelo juez (clave de `models:`); `null` = el propio generador. Recomendado: modelo distinto |
| `model_load_wait` | Paciencia (s) con modelos en frío: 5xx en la ventana no consumen reintentos. `null` = off |
| `workers` | Generación en paralelo (ThreadPool); 1 = secuencial. Appends thread-safe |

Además: `paths:` (nombres de subdirs/ficheros) e `image_generation:`
(`enabled: true/false`).

Tests: `python -m pytest tests/` (no requieren red).

> **Nota**: el prompt de generación es *scenario-faithful* — el texto debe
> mencionar los elementos afectados, riesgos y expresión de urgencia del
> escenario (expresiones configurables en `risk_expressions:` y
> `urgency_expressions:` del config). Los datasets generados con versiones
> anteriores del prompt no son comparables con los nuevos.

## Generación de imágenes (opcional)

Se controla con la sección `image_generation` de la config compartida
(`enabled: false` por defecto). Requiere GPU + FLUX.2 y las dependencias extra
de `requirements-images.txt`:

```bash
pip install -r requirements-images.txt
```

Activar solo en un host con GPUs. Si se activa sin GPU/deps, el pipeline lo
detecta y **continúa sin imágenes** (no falla). También hay un CLI standalone
`flux2_imagegen.py` (`--metadata`, `--test`, 12 incidencias) que reutiliza el
mismo backend.

## Configuración

Toda la configuración se lee de la **config compartida** en la raíz del
repositorio (`../config.yaml`). Secciones usadas: `endpoints`, `models`,
`generation`, `paths`, `logging`, `categories`, `affected_elements`,
`location_zones`, `risks`, `urgency_levels`, `risk_expressions`,
`urgency_expressions`, `urgency_weights`, `category_elements`, `category_zones`,
`category_risks`, `communities`, `personas`, `weights`, `modifiers`,
`modifier_prob`, `image_generation`. La sección `annotation` no la usa (es del
extractor).

### Parámetros por modelo y endpoint

- `models.<nombre>.extra_params` — claves provider-specific que se pasan
  verbatim al payload (p. ej. `enable_thinking: false`,
  `chat_template_kwargs: {...}`). Sin él, el payload es limpio
  (`model`/`messages`/`temperature`).
- `endpoints.<nombre>.connection_backoff: true` — backoff generoso ante
  caídas del servidor (`ConnectionError`/`Timeout`): espera
  `max(retry_delay, 20s) × intento`. Default `false` (lineal).
