# miUrba

Repositorio con **dos tareas independientes** sobre quejas/incidencias
reportadas por vecinos de comunidades:

1. **`miurba-generator/`** — generador de **datasets sintéticos** de quejas
   vecinales. Genera el texto del vecino y, opcionalmente, las **imágenes**
   (fotos) adjuntas (FLUX.2-dev, multi-GPU, activable por config).
2. **`miurba-extractor/`** — **extractor / summarizador** con un **VLM** que ve
   el texto y las fotos, describe las imágenes y produce la extracción
   estructurada conforme a un JSON schema.

Cada proyecto es **totalmente independiente**: su propio código (`src/`), CLI,
`requirements.txt` y `data/`. No comparten paquetes ni módulos.

## Lo único compartido

- **Configuración:** un único `config.yaml` en la raíz (+ variantes
  `config.local.yaml` y `config.neuralwatt.yaml`). Secciones destacadas:
  `image_generation` (generador, opcional) y `annotation.vision_model`
  (extractor, modelo vision).
- **Schemas:** `schemas/` en la raíz define los contratos de datos:
  - `complaint_schema.json` — salida del generador / entrada del extractor.
  - `extraction_schema.json` — salida del extractor (incluye
    `image_descriptions`, que produce el extractor, no el generador).

## Flujo entre los dos proyectos

```
miurba-generator                                  miurba-extractor
  scenario ──► LLM ──► user_text + timestamp        user_text + PNGs ──► VLM
  scenario ──► FLUX ──► PNGs (opcional)                                   │
      │                                                                   ▼
      ▼                                                  image_descriptions + extractions.jsonl
  complaints.jsonl + scenarios.jsonl                    (conforme a extraction_schema.json)
   (user_pics = nombres .png)
```

- El generador produce `complaints.jsonl` + `scenarios.jsonl` (+ los PNG si la
  generación de imágenes está activada). **No** produce `image_descriptions`.
- El extractor consume `complaints.jsonl` + los PNG; **él** describe las
  imágenes (VLM) y extrae la estructura.

El nexo es **el formato de datos** (`complaint_schema.json` y los ficheros
`.png` referenciados por `user_pics`), no código.

## Estructura

```
mi-urba/
├── config.yaml                  # COMPARTIDA
├── config.local.yaml
├── config.neuralwatt.yaml
├── schemas/
│   ├── complaint_schema.json
│   └── extraction_schema.json
├── miurba-generator/            # TAREA 1 (independiente)
│   ├── cli.py  src/  data/  requirements.txt  requirements-images.txt  README.md
│   ├── flux2_imagegen.py        # CLI standalone de imágenes (FLUX.2)
│   └── test_models.py           # auxiliar: benchmark de modelos
└── miurba-extractor/            # TAREA 2 (independiente, VLM)
    ├── cli.py  src/  data/  requirements.txt  README.md
```

## Inicio rápido

```bash
# Generador (sin imágenes)
cd miurba-generator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py generate -n 10

# Generador con imágenes (requiere GPU + FLUX.2)
pip install -r requirements-images.txt
# en config.yaml: image_generation.enabled: true
python cli.py generate -n 10

# Extractor (VLM)
cd ../miurba-extractor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../miurba-generator/data/raw/{complaints,scenarios}.jsonl data/raw/
cp -r ../miurba-generator/data/images data/        # si se generaron imágenes
python cli.py extract
```
