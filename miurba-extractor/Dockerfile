# miurba-extractor — container for the VLM extraction service
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY cli.py ./
COPY src ./src

# src/config.py resolves the shared config at /config.yaml and schemas
# at /schemas (repo-root convention). Do NOT bake them: mount instead.

RUN useradd -m appuser
USER appuser

# Mounts (root-level, repo-root convention):
#   -v ./config.yaml:/config.yaml:ro
#   -v ./schemas:/schemas:ro
#   -v ./data:/data
#   -v ./logs:/logs   (config.yaml → logging.file: /logs/run.log)
ENTRYPOINT ["python", "cli.py"]
