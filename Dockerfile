FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install all dependencies in one layer
COPY api/requirements.txt /tmp/api-requirements.txt
COPY miurba-generator/requirements.txt /tmp/gen-requirements.txt
COPY miurba-extractor/requirements.txt /tmp/ext-requirements.txt

RUN pip install --no-cache-dir \
    -r /tmp/api-requirements.txt \
    -r /tmp/gen-requirements.txt \
    -r /tmp/ext-requirements.txt

# Copy project code (no secrets — config.yaml has no API keys)
COPY config.yaml ./
COPY schemas/ schemas/
COPY miurba-generator/ miurba-generator/
COPY miurba-extractor/ miurba-extractor/
COPY api/ api/

# Config path (override with -e MIURBA_CONFIG=... or volume mount)
# Users provide their own config with API keys via volume mount.
ENV MIURBA_CONFIG=config.yaml
ENV MIURBA_HOST=0.0.0.0
ENV MIURBA_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:8000/v1/health || exit 1

CMD ["sh", "-c", "uvicorn api.server:app --host ${MIURBA_HOST} --port ${MIURBA_PORT}"]