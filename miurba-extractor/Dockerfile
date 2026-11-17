# miurba-extractor — container for the VLM extraction service
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY cli.py ./
COPY src ./src
COPY prompts ./prompts
COPY config ./config

# src/config.py resolves the default config at /app/config/config.yaml
# (baked examples above; override with mounts) and the shared schemas +
# taxonomy at /schemas (mount ../schemas from the repo root).

RUN useradd -m appuser
USER appuser

# Mounts:
#   -v ./config:/app/config:ro
#   -v ../schemas:/schemas:ro
#   -v ./data:/data
#   -v ./logs:/logs   (config → logging.file: /logs/run.log)
ENTRYPOINT ["python", "cli.py"]
