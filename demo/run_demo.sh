#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# miUrba — Demo REAL del pipeline completo
#
# Ejecuta las dos fases con llamadas reales a un LLM:
#   1. miurba-generator: genera quejas sintéticas (kimi-k2.6 + qwen3_local)
#   2. miurba-extractor: extrae estructura + describe imágenes con VLM
#
# Requisitos:
#   - Python 3.10+
#   - Acceso a https://api.neuralwatt.com/v1 (API key en config.demo.yaml)
#   - Acceso a http://192.168.10.11:9292 (llama-swap local para qwen3_local)
#
# Uso:
#   bash demo/run_demo.sh             # ejecuta generador + extractor
#   bash demo/run_demo.sh --show      # solo muestra datos pre-generados (sin LLM)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$SCRIPT_DIR/config.demo.yaml"
OUT="$SCRIPT_DIR/output"

C='\033[36m'  # cyan
B='\033[1m'   # bold
DIM='\033[2m' # dim
G='\033[32m'  # green
R='\033[0m'   # reset

banner() {
    echo -e "\n${C}${B}═══════════════════════════════════════════════════════════${R}"
    echo -e "${C}${B}  $1${R}"
    echo -e "${C}${B}═══════════════════════════════════════════════════════════${R}\n"
}

show_jsonl() {
    local file="$1" label="$2"
    echo -e "\n${B}${label}${R} ${DIM}($(wc -l < "$file") registros)${R}"
    echo -e "${DIM}──────────────────────────────────────────────────────────${R}"
    python3 -c "
import json
with open('$file') as f:
    for line in f:
        if line.strip():
            print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
            print()
"
}

MODE="${1:-run}"

# ── Show pre-generated output ──────────────────────────────────────────────
if [ "$MODE" = "--show" ]; then
    banner "miUrba — Salida pre-generada (sin LLM)"
    show_jsonl "$SCRIPT_DIR/sample_output/complaints.jsonl" "FASE 1 — complaints.jsonl"
    show_jsonl "$SCRIPT_DIR/sample_output/scenarios.jsonl" "FASE 1 — scenarios.jsonl"
    show_jsonl "$SCRIPT_DIR/sample_output/extractions.jsonl" "FASE 2 — extractions.jsonl"
    exit 0
fi

# ── Run real pipeline ───────────────────────────────────────────────────────
banner "miUrba — Demo REAL (pipeline end-to-end)"
echo -e "  LLMs: kimi-k2.6 (NeuralWatt) + qwen3_local (llama-swap)"
echo -e "  6 quejas · sin imágenes (FLUX desactivado)"
echo ""

rm -rf "$OUT"
mkdir -p "$OUT"

# ── Fase 1: Generador ───────────────────────────────────────────────────────
banner "FASE 1 — miurba-generator"
echo -e "  Genera quejas sintéticas con un LLM real.\n"

cd "$REPO_ROOT/miurba-generator"

if [ ! -d .venv ]; then
    echo -e "${DIM}Creando venv...${R}"
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
fi

echo -e "${G}▶ python cli.py generate -n 3 --config ../demo/config.demo.yaml${R}\n"
.venv/bin/python cli.py generate -n 3 --config "$CONFIG" --output-dir "$OUT" --no-resume

show_jsonl "$OUT/raw/complaints.jsonl" "complaints.jsonl"
show_jsonl "$OUT/raw/scenarios.jsonl" "scenarios.jsonl"

# ── Fase 2: Extractor ──────────────────────────────────────────────────────
banner "FASE 2 — miurba-extractor"
echo -e "  Extrae estructura con VLM (kimi-k2.6, multimodal).\n"

cd "$REPO_ROOT/miurba-extractor"

if [ ! -d .venv ]; then
    echo -e "${DIM}Creando venv...${R}"
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
fi

mkdir -p data/raw data/images
cp "$OUT/raw/complaints.jsonl" data/raw/
cp "$OUT/raw/scenarios.jsonl" data/raw/

echo -e "${G}▶ python cli.py extract --config ../demo/config.demo.yaml${R}\n"
.venv/bin/python cli.py extract --config "$CONFIG" \
    --raw-dir data/raw \
    --images-dir data/images \
    --annotated-dir "$OUT/annotated"

show_jsonl "$OUT/annotated/extractions.jsonl" "extractions.jsonl"

# ── Done ────────────────────────────────────────────────────────────────────
banner "Demo completada"
echo -e "  Salida: $OUT/"
echo -e "  ${DIM}raw/complaints.jsonl    — texto del vecino (6 quejas)${R}"
echo -e "  ${DIM}raw/scenarios.jsonl     — ground truth (3 escenarios)${R}"
echo -e "  ${DIM}annotated/extractions.jsonl — estructura extraída (VLM)${R}"
echo ""
echo -e "  Para ver sin re-ejecutar: ${C}bash demo/run_demo.sh --show${R}"