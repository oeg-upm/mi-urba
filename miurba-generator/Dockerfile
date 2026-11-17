# miurba-generator — container for the synthetic-complaint generator
FROM python:3.12-slim

WORKDIR /app

# System deps: none beyond stdlib (requests/pyyaml/fastapi-free project)
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Project code (src/ + cli.py + example configs). Tests and data are NOT baked in.
COPY cli.py ./
COPY src ./src
COPY prompts ./prompts
COPY config ./config

# Non-root user
RUN useradd -m appuser
USER appuser

# Defaults resolve to /app/config/config.yaml (baked examples above; override
# with mounts) and the shared taxonomy at /schemas/taxonomy.yaml (mount it):
#   -v ./config:/app/config:ro
#   -v ../schemas:/schemas:ro
#   -v ./data:/data
#   -v ./logs:/logs   (config → logging.file: /logs/run.log)
ENTRYPOINT ["python", "cli.py"]
