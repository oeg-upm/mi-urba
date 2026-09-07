# miurba-extractor — standalone container (VLM extraction)
#
# Build:  docker build -t miurba-extractor -f miurba-extractor/Dockerfile .
# Run:    docker run --rm --network host \
#           -v ./miurba-extractor/data:/app/miurba-extractor/data \
#           -v ./config.yaml:/app/config.yaml:ro \
#           miurba-extractor extract --raw-dir data/raw --images-dir data/images --annotated-dir data/annotated

FROM python:3.10-slim

WORKDIR /app

COPY miurba-extractor/requirements.txt /tmp/ext-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ext-requirements.txt

COPY config.yaml ./
COPY schemas/ schemas/
COPY miurba-extractor/ miurba-extractor/

WORKDIR /app/miurba-extractor

ENTRYPOINT ["python", "cli.py"]