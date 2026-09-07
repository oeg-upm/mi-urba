# miurba-generator — standalone container (text-only generation)
#
# Build:  docker build -t miurba-generator -f miurba-generator/Dockerfile .
# Run:    docker run --rm --network host \
#           -v ./miurba-generator/data:/app/miurba-generator/data \
#           -v ./config.yaml:/app/config.yaml:ro \
#           miurba-generator generate -n 5
#
# NOTE: FLUX.2 image generation (requirements-images.txt) is NOT included:
# it requires CUDA + multi-GPU and stays host-only.

FROM python:3.10-slim

WORKDIR /app

COPY miurba-generator/requirements.txt /tmp/gen-requirements.txt
RUN pip install --no-cache-dir -r /tmp/gen-requirements.txt

# Shared config + schemas live at repo root; cli.py resolves ../config.yaml
COPY config.yaml ./
COPY schemas/ schemas/
COPY miurba-generator/ miurba-generator/

WORKDIR /app/miurba-generator

ENTRYPOINT ["python", "cli.py"]