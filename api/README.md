# miUrba API

Servidor FastAPI que expone las dos tareas del proyecto (generación y
extracción) via REST, con convenciones estilo OpenAI.

## Endpoints

| Método | Path | Descripción |
|--------|------|-------------|
| GET | `/v1/health` | Health check |
| GET | `/v1/models` | Lista modelos disponibles (formato OpenAI) |
| POST | `/v1/generate` | Genera quejas sintéticas con LLM |
| POST | `/v1/extract` | Extrae estructura de quejas con VLM |
| POST | `/v1/pipeline` | Generate + extract en un solo call |
| GET | `/docs` | Swagger UI (interactivo) |

## Despliegue con Docker

```bash
# 1. Construir y arrancar
docker compose up --build

# 2. La API está en http://localhost:8000
#    Swagger UI: http://localhost:8000/docs
```

### Config con tu API key

El contenedor incluye `config.yaml` (sin secrets). Para usar tu propio endpoint
con API key, monta tu config:

```yaml
# docker-compose.yml
volumes:
  - ./config.neuralwatt.yaml:/app/config.yaml:ro
```

O usa una variable de entorno:

```bash
docker run -e MIURBA_CONFIG=config.neuralwatt.yaml -p 8000:8000 miurba-api
```

## Uso local (sin Docker)

```bash
# Instalar deps
pip install -r api/requirements.txt
pip install -r miurba-generator/requirements.txt
pip install -r miurba-extractor/requirements.txt

# Arrancar (usa config.yaml por defecto)
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# O con un config específico
MIURBA_CONFIG=config.neuralwatt.yaml python -m uvicorn api.server:app --port 8000
```

## Ejemplos

### Listar modelos

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "kimi-k2.6", "object": "model", "owned_by": "neuralwatt", "capabilities": {"vision": true}},
    {"id": "glm-5.1", "object": "model", "owned_by": "neuralwatt", "capabilities": {"vision": false}}
  ]
}
```

### Generar quejas

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "kimi-k2.6", "n": 3}'
```

```json
{
  "object": "generation",
  "id": "gen_abc123",
  "model": "kimi-k2.6",
  "complaints": [
    {"id": "Q0001", "user_text": "Estimado administrador...", "channel": "email", ...}
  ],
  "scenarios": [
    {"id": "Q0001", "problems": [...], "community": {...}, ...}
  ],
  "stats": {"generated": 3}
}
```

### Extraer estructura

```bash
curl -X POST http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "complaints": [
      {
        "id": "TEST001",
        "user_text": "Se ha roto una tubería en el garaje, hay agua por todas partes. Muy urgente.",
        "user_pics": [],
        "channel": "whatsapp",
        "timestamp": "2024-01-01T10:00:00+01:00"
      }
    ]
  }'
```

```json
{
  "object": "extraction",
  "id": "ext_def456",
  "model": "kimi-k2.6",
  "extractions": [
    {
      "complaint_id": "TEST001",
      "image_descriptions": [],
      "extractions": [
        {
          "category": {"principal": "fontanería", "subcategory": "rotura_tubería"},
          "affected_elements": [{"principal": "tubería", "subcategory": "desagüe"}],
          "location": {"zone": "garaje", "detail": "interior del garaje"},
          "risks": ["inundación"],
          "urgency": "urgente"
        }
      ]
    }
  ]
}
```

### Extraer con imágenes

```bash
curl -X POST http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "complaints": [
      {
        "id": "Q0001",
        "user_text": "Mira qué desperfecto en la fachada...",
        "user_pics": ["Q0001_1.png"],
        "channel": "whatsapp",
        "timestamp": "2024-01-01T10:00:00+01:00"
      }
    ],
    "images": {
      "Q0001_1.png": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  }'
```

### Pipeline completo (generate + extract)

```bash
curl -X POST http://localhost:8000/v1/pipeline \
  -H "Content-Type: application/json" \
  -d '{"generate_model": "kimi-k2.6", "n": 3}'
```

```json
{
  "object": "pipeline",
  "id": "pipe_ghi789",
  "generate_model": "kimi-k2.6",
  "extract_model": "kimi-k2.6",
  "complaints": [...],
  "scenarios": [...],
  "extractions": [...]
}
```

## Arquitectura

```
Cliente ──HTTP──► FastAPI (api/server.py)
                    │
                    ├─ POST /v1/generate  ──► subprocess: miurba-generator/cli.py
                    ├─ POST /v1/extract   ──► subprocess: miurba-extractor/cli.py
                    └─ POST /v1/pipeline  ──► generate + extract secuencial
                    │
                    └── LLM (NeuralWatt / Ollama) ◄── config.yaml
```

El servidor orquesta llamadas subprocess a los CLI existentes, pasando datos
via ficheros temporales JSONL. Así no hay conflictos de imports entre los dos
subproyectos (que comparten nombres de módulo) y se reutiliza código probado.