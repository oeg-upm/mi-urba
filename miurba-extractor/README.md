# miurba-extractor

Extractor / **summarizador** de quejas vecinales con un **VLM** (modelo
multimodal vision). Transforma una queja no estructurada en una extracción
estructurada conforme a un JSON schema, **describiendo él mismo las fotos**.

## Qué hace

Recibe el texto del vecino y las **fotos reales** (PNG) adjuntadas, y usa un
VLM (vision-language model) que **ve las imágenes** para, en una sola llamada:

1. Describir cada foto → `image_descriptions`.
2. Extraer la información estructurada → `extractions` (categoría, elementos
   afectados, ubicación, riesgos, urgencia), considerando texto e imágenes.

**Describir las imágenes es tarea del extractor.** El generador ya no produce
descripciones.

## Entrada

Consume la **salida del proyecto `miurba-generator`**:

- `data/raw/complaints.jsonl` — la queja (`id, user_text, user_pics, channel, …`).
- `data/images/*.png` — las fotos referenciadas por `user_pics`.

Contrato de entrada: `schemas/complaint_schema.json` (raíz del repo).
Si una foto no existe, se marca como no disponible y se extrae del texto.

## Salida

- `data/annotated/extractions.jsonl` — `{complaint_id, image_descriptions,
  extractions}`.
- Contrato de salida: `schemas/extraction_schema.json` (raíz del repo).

## Uso

```bash
python cli.py extract --model <vlm>                 # modelo vision
python cli.py extract --raw-dir data/raw --images-dir data/images --annotated-dir data/annotated
```

## Configuración

Config compartida en la raíz (`../config.yaml`). El modelo vision se resuelve
como: `--model` > `annotation.vision_model` > `annotation.model`. **El modelo
debe ser multimodal** (capaz de ver imágenes). Secciones usadas: `endpoints`,
`models`, `annotation`, `paths`, `categories`, `location_zones`, `risks`,
`urgency_levels`.
