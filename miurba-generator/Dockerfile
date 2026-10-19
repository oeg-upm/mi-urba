# miurba-generator — container for the synthetic-complaint generator
FROM python:3.12-slim

WORKDIR /app

# System deps: none beyond stdlib (requests/pyyaml/fastapi-free project)
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Project code (src/ + cli.py). Tests and data are NOT baked in.
COPY cli.py ./
COPY src ./src

# Non-root user
RUN useradd -m appuser
USER appuser

# The client resolves the shared config at /config.yaml (repo-root
# convention). Config, data and logs come from mounts:
#   -v ./config.yaml:/config.yaml:ro
#   -v ./data:/data
#   -v ./logs:/logs   (config.yaml → logging.file: /logs/run.log)
ENTRYPOINT ["python", "cli.py"]
