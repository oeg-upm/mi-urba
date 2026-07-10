# Demo de miUrba

Demo **real** del pipeline end-to-end: genera quejas sintéticas con un LLM y
extrae estructura con un VLM, todo con llamadas reales a la API.

## Qué hace

```
FASE 1: miurba-generator                     FASE 2: miurba-extractor
  LLM (kimi-k2.6) → user_text + timestamp       VLM (kimi-k2.6) → estructura + descripciones
       │                                              │
       ▼                                              ▼
  complaints.jsonl                         extractions.jsonl
  scenarios.jsonl                          (category, elements, location, risks, urgency)
```

## Requisitos

- Python 3.10+
- Acceso a `https://api.neuralwatt.com/v1` (API key en `config.demo.yaml`)

## Uso

```bash
# Ejecutar el pipeline completo (llamadas reales al LLM, ~2 min)
bash demo/run_demo.sh

# Ver la salida pre-generada sin ejecutar el LLM
bash demo/run_demo.sh --show
```

## Salida

La demo produce en `demo/output/`:

| Fichero | Contenido |
|---------|-----------|
| `raw/complaints.jsonl` | Texto del vecino, canal, timestamp, fotos referenciadas |
| `raw/scenarios.jsonl` | Ground truth: problemas, comunidad, persona, modifier |
| `annotated/extractions.jsonl` | Extracción VLM: category, elements, location, risks, urgency + image_descriptions |

## Config

`config.demo.yaml` es un config completo (merge de `config.yaml` +
`config.neuralwatt.yaml`) con:
- 3 registros, modelo `kimi-k2.6`
- `vision_model: kimi-k2.6` (multimodal, puede ver imágenes)
- `image_generation.enabled: false` (sin FLUX)

## Datos pre-generados

`sample_output/` contiene la salida de una ejecución real para ver sin LLM:

- **Q0001**: Vecino técnico (email) reporta carpintería en terraza comunitaria.
  El VLM extrae 4 problemas distintos de un solo texto.
- **Q0002**: Propietario tacaño (telefono, exceso de emojis) reporta fuga de
  agua caliente. El VLM identifica fontanería urgente.
- **Q0003**: Inquilino reciente (WhatsApp) con foto no disponible. El VLM
  marca la foto como ausente y extrae del texto.