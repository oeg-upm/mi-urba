# miurba-extractor — container for the extraction service
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY cli.py ./
COPY src ./src
COPY prompts ./prompts
COPY config ./config

# src/config.py resolves the default config at /app/config/config.yaml
# (mounted over the baked examples), the shared schemas + taxonomy at
# /schemas (mounted from ../schemas), and data/logs via /app/data and
# /app/logs. .env (OPENAI_API_*) is passed by docker compose via env_file.

RUN useradd -m appuser
USER appuser

ENTRYPOINT ["python", "cli.py"]
