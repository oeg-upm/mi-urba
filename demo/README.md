# Demo de miUrba

Demo **real** del pipeline end-to-end: genera quejas sintéticas con un LLM y
extrae estructura con un VLM, todo con llamadas reales a la API.

## Qué hace

```
FASE 1: miurba-generator                     FASE 2: miurba-extractor
  LLM → user_text + timestamp                   VLM → estructura + descripciones
       │                                              │
       ▼                                              ▼
  complaints.jsonl                           extractions.jsonl
  scenarios.jsonl (ground truth)             (category, elements, location, risks, urgency)
```

## Requisitos

- Python 3.10+
- Acceso a `https://api.neuralwatt.com/v1` (API key en `config.demo.yaml`)
- Acceso a `http://192.168.10.11:9292` (llama-swap local, para `qwen3_local`)

## Endpoints y modelos configurados

| Modelo | Endpoint | Tipo | Notas |
|--------|----------|------|-------|
| `kimi-k2.6` | NeuralWatt | LLM + VLM | Multimodal, puede ver imágenes |
| `kimi-k2.5` | NeuralWatt | LLM | |
| `qwen3.5-397b` | NeuralWatt | LLM | |
| `glm-5.1` | NeuralWatt | LLM | |
| `glm-5.2` | NeuralWatt | LLM | |
| `qwen3_local` | llama-swap | LLM | `qwen3.6-27b-q6-k-xl-mtp-2` (local GPU) |

Los batches de la demo usan `kimi-k2.6` (3 registros) + `qwen3_local` (3 registros).
La extracción usa `kimi-k2.6` como VLM (necesita capacidades multimodales).

## Uso rápido (script)

```bash
# Ejecutar el pipeline completo (llamadas reales al LLM, ~2 min)
bash demo/run_demo.sh

# Ver la salida pre-generada sin ejecutar el LLM
bash demo/run_demo.sh --show
```

## Uso manual (comandos python)

### Setup

```bash
# Crear virtualenvs e instalar dependencias
cd miurba-generator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ../miurba-extractor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Fase 1 — Generador

```bash
cd miurba-generator
source .venv/bin/activate

# Generar 3 quejas sintéticas (LLM: kimi-k2.6 via NeuralWatt)
python cli.py generate -n 3 --config ../demo/config.demo.yaml --output-dir ../demo/output

# Generar con el modelo local (qwen3.6-27b via llama-swap)
python cli.py generate -n 10 --config ../demo/config.demo.yaml --model qwen3_local

# Generar con otro modelo de NeuralWatt
python cli.py generate -n 10 --config ../demo/config.demo.yaml --model glm-5.1

# Generar por lotes multi-modelo (usa generation.batches del config:
#   kimi-k2.6 × 3 + qwen3_local × 3)
python cli.py generate-multi --config ../demo/config.demo.yaml

# Fusionar salidas de generate-multi en data/raw/
python cli.py merge --config ../demo/config.demo.yaml

# Generar con imágenes (requiere GPU + FLUX.2)
#   1. Activar en config: image_generation.enabled: true
#   2. pip install -r requirements-images.txt
python cli.py generate -n 10 --config ../demo/config.demo.yaml

# CLI standalone de imágenes FLUX.2
python flux2_imagegen.py --test                          # test rápido (1 imagen)
python flux2_imagegen.py --scenarios data/raw/scenarios.jsonl --limit 10  # desde scenarios
python flux2_imagegen.py                                # 12 incidencias demo
```

**Salida del generador:**
- `data/raw/complaints.jsonl` — `{id, user_text, user_pics, timestamp}`
- `data/raw/scenarios.jsonl` — ground truth: `{complaint_id, extractions, image_descriptions, persona, modifier, community, num_pics, image_prompts}`
- `data/images/*.png` — fotos reales (solo si FLUX activado)

### Fase 2 — Extractor

```bash
cd miurba-extractor
source .venv/bin/activate

# Copiar datos del generador
mkdir -p data/raw data/images
cp ../miurba-generator/data/raw/{complaints,scenarios}.jsonl data/raw/
cp -r ../miurba-generator/data/images/* data/images/   # si hay imágenes

# Extraer estructura + descripciones de imágenes (VLM: kimi-k2.6 multimodal)
python cli.py extract --config ../demo/config.demo.yaml

# Especificar rutas explícitas
python cli.py extract --config ../demo/config.demo.yaml \
  --raw-dir data/raw \
  --images-dir data/images \
  --annotated-dir data/annotated

# Usar otro modelo VLM
python cli.py extract --config ../demo/config.demo.yaml --model glm-5.1

# Sin resume (empezar de cero)
python cli.py extract --config ../demo/config.demo.yaml --no-resume
```

**Salida del extractor:**
- `data/annotated/extractions.jsonl` — `{complaint_id, image_descriptions, extractions}`
  - `image_descriptions`: descripción de cada foto por el VLM (vacío si no hay fotos)
  - `extractions[]`: `{category, affected_elements, location, risks, urgency}` por cada problema

### Flujo completo manual

```bash
# 1. Generar
cd miurba-generator && source .venv/bin/activate
python cli.py generate -n 3 --config ../demo/config.demo.yaml --output-dir /tmp/miurba-demo

# 2. Extraer
cd ../miurba-extractor && source .venv/bin/activate
mkdir -p data/raw && cp /tmp/miurba-demo/raw/complaints.jsonl data/raw/
python cli.py extract --config ../demo/config.demo.yaml --raw-dir data/raw --annotated-dir /tmp/miurba-demo/annotated

# 3. Ver resultados
python -c "import json; [print(json.dumps(json.loads(l), indent=2, ensure_ascii=False)) for l in open('/tmp/miurba-demo/annotated/extractions.jsonl') if l.strip()]"
```

## Salida

La demo produce en `demo/output/`:

| Fichero | Contenido |
|---------|-----------|
| `raw/complaints.jsonl` | Texto del vecino, timestamp, fotos referenciadas |
| `raw/scenarios.jsonl` | Ground truth: extracciones esperadas + metadata de generación |
| `annotated/extractions.jsonl` | Extracción VLM: category, elements, location, risks, urgency + image_descriptions |

## Config

`config.demo.yaml` es un config completo (merge de `config.yaml` +
`config.neuralwatt.yaml` + endpoint llama-swap) con:
- 2 endpoints: NeuralWatt (cloud) + llama-swap (local GPU)
- Batches: `kimi-k2.6` × 3 + `qwen3_local` × 3
- `vision_model: kimi-k2.6` (multimodal, puede ver imágenes)
- `image_generation.enabled: false` (sin FLUX)

## Datos pre-generados

`sample_output/` contiene la salida de una ejecución real para ver sin LLM:

- **Q0001**: Vecino egoísta reporta fachada + carpintería. El VLM extrae 2 problemas.
- **Q0002**: Vecino hipocondríaco con modificador latinoamericano reporta fontanería.
  El VLM identifica fontanería urgente + problemas adicionales.