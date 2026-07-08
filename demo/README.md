# Demo de miUrba

Esta demo muestra el uso de **todas las partes** del proyecto miUrba end-to-end.

## Qué demuestra

| Fase | Proyecto | Qué hace | Comando |
|------|----------|----------|---------|
| 1 | `miurba-generator` | Genera 3 quejas sintéticas (texto de vecino + metadatos) | `cli.py generate -n 3` |
| 2 | `miurba-extractor` | Extrae estructura + describe imágenes con VLM | `cli.py extract` |
| 3 | `miurba-generator` (bonus) | Generación de imágenes con FLUX.2-dev | `flux2_imagegen.py` |

## Requisitos

- Python 3.10+
- Ollama corriendo en `localhost:11434` con el modelo `kimi-k2.6:cloud`
- (Opcional) 2 GPUs NVIDIA para la fase de imágenes FLUX.2

## Uso rápido

```bash
# Ver datos pre-generados sin necesidad de LLM (offline)
bash demo/run_demo.sh --sample

# Demo completa (genera + extrae, necesita Ollama)
bash demo/run_demo.sh

# Solo fase 1: generador
bash demo/run_demo.sh --generate

# Solo fase 2: extractor (necesita datos del generador)
bash demo/run_demo.sh --extract

# Solo bonus: cómo funciona la generación de imágenes
bash demo/run_demo.sh --flux
```

## Estructura

```
demo/
├── README.md              # Este documento
├── run_demo.sh            # Script principal de la demo
├── config.demo.yaml       # Config reducida (3 registros, 1 modelo)
└── sample_data/           # Datos pre-generados para demo offline
    ├── complaints.jsonl   # 3 quejas de muestra
    ├── scenarios.jsonl    # 3 escenarios (ground truth)
    └── extractions.jsonl  # 3 extracciones estructuradas + descripciones de imágenes
```

## Flujo de la demo

```
FASE 1: miurba-generator                    FASE 2: miurba-extractor
  scenario → LLM → user_text + timestamp      user_text + PNGs → VLM
                                              (texto + fotos reales)
       │                                           │
       ▼                                           ▼
  complaints.jsonl                      extractions.jsonl
  scenarios.jsonl                       (image_descriptions + extractions)
```

### Fase 1 — Generador

```bash
cd miurba-generator
python cli.py generate -n 3 --config ../demo/config.demo.yaml
```

Produce:
- `data/raw/complaints.jsonl` — texto del vecino, canal, timestamp, fotos referenciadas
- `data/raw/scenarios.jsonl` — ground truth: problemas, comunidad, persona, image_prompts

### Fase 2 — Extractor

```bash
cd miurba-extractor
# Copiar datos del generador
cp ../miurba-generator/data/raw/{complaints,scenarios}.jsonl data/raw/

# Extraer
python cli.py extract --config ../demo/config.demo.yaml
```

Produce:
- `data/annotated/extractions.jsonl` — `{complaint_id, image_descriptions, extractions}`

### Bonus — FLUX.2 (imágenes)

```bash
cd miurba-generator
pip install -r requirements-images.txt

# Test rápido
python flux2_imagegen.py --test

# Generar desde scenarios
python flux2_imagegen.py --scenarios data/raw/scenarios.jsonl --limit 10

# 12 incidencias de demostración
python flux2_imagegen.py
```

## Datos de muestra

Los archivos en `sample_data/` son registros reales del dataset generado, listos
para inspeccionar sin ejecutar el pipeline:

- **`complaints.jsonl`** — 3 quejas con diferentes canales (email, WhatsApp, app),
  personas (inmobiliario, conflictivo, pasivo-agresiva) y tipos de problema
  (cubierta, fachada, jardinería).
- **`scenarios.jsonl`** — Ground truth de cada queja: categoría, subcategoría,
  urgencia, ubicación, riesgos, elementos afectados, comunidad, persona.
- **`extractions.jsonl`** — Extracción estructurada producida por el VLM, incluyendo
  `image_descriptions` (descripciones de las fotos) y `extractions` (categoría,
  elementos, ubicación, riesgos, urgencia).

## Config de demo

`config.demo.yaml` es una copia de `config.yaml` con:
- `generation.batches`: 1 modelo (kimi-k2.6), 3 registros
- `image_generation.enabled`: false (sin FLUX por defecto)
- Resto de configuración idéntica (categorías, comunidades, personas, etc.)