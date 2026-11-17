# miurba-generator — container for the synthetic-complaint generator
FROM python:3.12-slim

WORKDIR /app

# System deps: none beyond stdlib (requests/pyyaml/fastapi-free project)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Project code (src/ + cli.py + example configs). Tests and data are NOT baked in.
COPY cli.py ./
COPY src ./src
COPY prompts ./prompts
COPY config ./config

# Non-root user
RUN useradd -m appuser
USER appuser

# Defaults resolve to /app/config/config.yaml (mounted over the baked
# examples) and the shared taxonomy at /schemas/taxonomy.yaml (mounted from
# ../schemas). Data and logs persist via ./data:/app/data and ./logs:/app/logs;
# .env (OPENAI_API_*) is passed by docker compose via env_file.
ENTRYPOINT ["python", "cli.py"]
