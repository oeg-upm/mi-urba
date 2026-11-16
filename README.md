# miUrba

Generación sintética y extracción estructurada de quejas vecinales.
Dos herramientas independientes sobre una configuración compartida:

- **`miurba-generator/`** — genera datasets sintéticos de quejas: texto del
  vecino (LLM), ground truth estructurado y, opcionalmente, imágenes (FLUX.2).
- **`miurba-extractor/`** — extrae estructura de quejas reales o sintéticas
  (texto + fotos) con un VLM, conforme a un JSON Schema.

## Flujo de datos

```
miurba-generator                          miurba-extractor
  escenario ─► LLM ─► user_text             user_text + PNGs ─► VLM
  escenario ─► FLUX ─► PNGs (opcional)                            │
       ▼                                                          ▼
  complaints.jsonl + ground_truth.jsonl + metadata.jsonl    extractions.jsonl
       └────────────── entrada del extractor ──────────────┘
```

El nexo entre ambas es el formato de datos (`schemas/` y los PNG
referenciados por `user_pics`), no código.

## Estructura

```
mi-urba/
├── config.yaml                  # configuración de ejecución (obligatoria)
├── taxonomy.yaml                # vocabulario de dominio: taxonomías, personas, pesos
├── schemas/                     # contratos de datos
│   ├── complaint_schema.json
│   ├── extraction_schema.json
│   └── image_description_schema.json
├── docker-compose.yml           # ejecución dockerizada
├── scripts/docker_smoke.sh      # smoke test end-to-end
├── miurba-generator/            # CLI: generate, merge
└── miurba-extractor/            # CLI: extract
```

## Requisitos

- Python 3.10+ (probado con 3.12)
- Un endpoint LLM OpenAI-compatible (vLLM, llama.cpp/llama-swap, Ollama…)
- Opcional: GPU + FLUX.2 para generación de imágenes

## Inicio rápido

```bash
# 1. Configura el endpoint y los modelos en config.yaml (ver abajo).
#    Los secretos van en .env (copia .env.example) y se referencian como ${VAR}.

# 2. Generar quejas sintéticas
cd miurba-generator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py generate --config ../config.yaml

# 3. Extraer estructura de las quejas generadas
cd ../miurba-extractor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py extract --config ../config.yaml
```

Salidas por defecto: `<default_output_dir>/<modelo>/{raw,images}` en el
generador y `<default_output_dir>/annotated/extractions.jsonl` en el extractor.

## Docker

```bash
docker compose build
docker compose run --rm miurba-generator generate --config /config/config.test.yaml
docker compose run --rm miurba-extractor extract  --config /config/config.test.yaml
```

Monta `config.yaml`, `schemas/`, los directorios de datos y `logs/`
(activa `logging.file: /logs/run.log` para persistir el log del run).
`scripts/docker_smoke.sh` ejecuta la verificación completa.

## Configuración

Un único `config.yaml` compartido en la raíz. Detalle completo de claves,
defaults y comentarios en el propio fichero y en los README de cada
subproyecto. Resumen:

| Sección | Fichero | Obligatoria | Uso |
|---|---|---|---|
| `endpoints` | config.yaml | sí | URLs base y `api_key` (referenciada como `${VAR}` desde `.env`) |
| `models` | config.yaml | sí | Modelos por nombre: `endpoint`, `model`, `temperature`, `structured_output` (extractor), `extra_params` |
| Taxonomía (`categories`, `risks`, `location_zones`, `personas`, `communities`, `weights`, …) | taxonomy.yaml | sí (generador) | Dominio del muestreo y del prompt; se referencia con `taxonomy:` en el config. Claves de taxonomía en config.yaml → error fail-fast |
| `risk_expressions`, `urgency_expressions` | taxonomy.yaml | sí (generador) | Expresiones obligatorias en el texto generado; cobertura validada al arrancar |
| `generation` | no | Plan de batches, seed, workers, retries, validación con juez |
| `extraction` | no (salvo `model`, obligatorio) | Modelo VLM, rutas, workers, retries |
| `paths`, `logging` | no | Nombres de ficheros y log de run |
| `image_generation` | no | FLUX.2 (`enabled: false` por defecto; requiere GPU) |

Variantes típicas: `config.local.yaml` / `config.test.yaml` (gitignored,
con IPs de LAN); se seleccionan con `--config`.

## Testing

Ejecuta `pytest` **dentro** de cada subproyecto (los paquetes `src` son
independientes y colisionan si se lanzan desde la raíz):

```bash
cd miurba-generator && python -m pytest tests/
cd miurba-extractor && python -m pytest tests/
```

No requieren red.
