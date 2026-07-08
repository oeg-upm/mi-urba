#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# miUrba — Demo completa del pipeline
#
# Ejecuta las dos fases del proyecto end-to-end:
#   1. miurba-generator: genera 3 quejas sintéticas (texto + metadatos)
#   2. miurba-extractor: extrae estructura + descripciones de imágenes (VLM)
#
# Requisitos:
#   - Python 3.10+
#   - Ollama corriendo en localhost:11434 (o el endpoint configurado)
#   - Modelo kimi-k2.6:cloud disponible en Ollama
#
# Uso:
#   cd mi-urba
#   bash demo/run_demo.sh             # demo completa (genera + extrae)
#   bash demo/run_demo.sh --sample    # solo muestra datos pre-generados
#   bash demo/run_demo.sh --generate  # solo fase 1 (generador)
#   bash demo/run_demo.sh --extract   # solo fase 2 (extractor, necesita datos)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO_DIR="$SCRIPT_DIR"
DEMO_DATA="$DEMO_DIR/demo_data"
SAMPLE_DATA="$DEMO_DIR/sample_data"

# Colors
B='\033[1m'
DIM='\033[2m'
R='\033[0m'
G='\033[32m'
Y='\033[33m'
C='\033[36m'

banner() {
    echo -e "\n${C}${B}═══════════════════════════════════════════════════════════════${R}"
    echo -e "${C}${B}  $1${R}"
    echo -e "${C}${B}═══════════════════════════════════════════════════════════════${R}\n"
}

step() {
    echo -e "\n${G}▶ $1${R}"
}

note() {
    echo -e "${DIM}  $1${R}"
}

# ── Helpers ────────────────────────────────────────────────────────────────

check_ollama() {
    if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e "${Y}⚠  Ollama no responde en localhost:11434${R}"
        echo -e "${DIM}  La demo necesita un endpoint LLM. Opciones:${R}"
        echo -e "${DIM}    1. Arrancar Ollama: ollama serve${R}"
        echo -e "${DIM}    2. Usar --sample para ver datos pre-generados${R}"
        return 1
    fi
    return 0
}

show_jsonl() {
    local file="$1"
    local label="$2"
    local max="${3:-1}"
    echo -e "\n${B}$label${R} (${C}$(wc -l < "$file") registros${R})"
    echo -e "${DIM}───────────────────────────────────────────────────────────${R}"
    python3 -c "
import json, sys
with open('$file') as f:
    for i, line in enumerate(f):
        if i >= $max: break
        obj = json.loads(line)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        if i < $max - 1:
            print()
"
}

# ── Phases ─────────────────────────────────────────────────────────────────

phase_generate() {
    banner "FASE 1 — miurba-generator"
    echo -e "  Genera quejas sintéticas de vecinos usando un LLM."
    echo -e "  ${DIM}3 registros · modelo kimi-k2.6 · sin imágenes (FLUX desactivado)${R}\n"

    check_ollama || return 1

    step "Configurando entorno del generador"
    cd "$REPO_ROOT/miurba-generator"

    if [ ! -d .venv ]; then
        note "Creando virtualenv..."
        python3 -m venv .venv
        .venv/bin/pip install -q -r requirements.txt
    fi
    note "Usando: $(.venv/bin/python --version)"

    step "Ejecutando: python cli.py generate -n 3 --config ../demo/config.demo.yaml"
    .venv/bin/python cli.py generate \
        -n 3 \
        --config "$DEMO_DIR/config.demo.yaml" \
        --output-dir "$DEMO_DATA"

    step "Salida del generador"
    show_jsonl "$DEMO_DATA/raw/complaints.jsonl" "complaints.jsonl" 3
    show_jsonl "$DEMO_DATA/raw/scenarios.jsonl" "scenarios.jsonl" 1
}

phase_extract() {
    banner "FASE 2 — miurba-extractor"
    echo -e "  Extrae estructura + describe imágenes usando un VLM."
    echo -e "  ${DIM}modelo kimi-k2.6 (multimodal) · texto + fotos → JSON estructurado${R}\n"

    check_ollama || return 1

    step "Configurando entorno del extractor"
    cd "$REPO_ROOT/miurba-extractor"

    if [ ! -d .venv ]; then
        note "Creando virtualenv..."
        python3 -m venv .venv
        .venv/bin/pip install -q -r requirements.txt
    fi
    note "Usando: $(.venv/bin/python --version)"

    # Copy data from generator to extractor
    step "Copiando datos del generador al extractor"
    mkdir -p data/raw data/images
    if [ -f "$DEMO_DATA/raw/complaints.jsonl" ]; then
        cp "$DEMO_DATA/raw/complaints.jsonl" data/raw/
        note "complaints.jsonl → data/raw/"
    fi
    if [ -f "$DEMO_DATA/raw/scenarios.jsonl" ]; then
        cp "$DEMO_DATA/raw/scenarios.jsonl" data/raw/
        note "scenarios.jsonl → data/raw/"
    fi

    step "Ejecutando: python cli.py extract --config ../demo/config.demo.yaml"
    .venv/bin/python cli.py extract \
        --config "$DEMO_DIR/config.demo.yaml" \
        --raw-dir data/raw \
        --images-dir data/images \
        --annotated-dir "$DEMO_DATA/annotated"

    step "Salida del extractor"
    show_jsonl "$DEMO_DATA/annotated/extractions.jsonl" "extractions.jsonl" 3
}

phase_sample() {
    banner "DEMO (datos pre-generados)"
    echo -e "  Muestra el resultado completo del pipeline sin necesidad de LLM.\n"

    step "Fase 1 — Salida del generador"
    show_jsonl "$SAMPLE_DATA/complaints.jsonl" "complaints.jsonl" 3
    show_jsonl "$SAMPLE_DATA/scenarios.jsonl" "scenarios.jsonl" 1

    step "Fase 2 — Salida del extractor"
    show_jsonl "$SAMPLE_DATA/extractions.jsonl" "extractions.jsonl" 3

    step "Esquema de extracción (extraction_schema.json)"
    python3 -c "
import json
with open('$REPO_ROOT/schemas/extraction_schema.json') as f:
    schema = json.load(f)
# Show the structure compactly
print(json.dumps({
    'required': schema['required'],
    'properties': list(schema['properties'].keys()),
    'problemExtraction': {
        'required': schema['\$defs']['problemExtraction']['required'],
        'properties': list(schema['\$defs']['problemExtraction']['properties'].keys()),
    }
}, indent=2))
"
}

phase_flux() {
    banner "BONUS — Generación de imágenes (FLUX.2-dev)"
    echo -e "  El generador puede producir fotos sintéticas con FLUX.2-dev."
    echo -e "  ${DIM}Requiere 2 GPUs + dependencias extra (requirements-images.txt)${R}\n"

    echo -e "  ${B}1. Activar en config:${R}"
    echo -e "${DIM}     image_generation:${R}"
    echo -e "${DIM}       enabled: true${R}"
    echo -e ""

    echo -e "  ${B}2. Instalar dependencias:${R}"
    echo -e "${DIM}     cd miurba-generator${R}"
    echo -e "${DIM}     pip install -r requirements-images.txt${R}"
    echo -e ""

    echo -e "  ${B}3. Generar con imágenes integradas:${R}"
    echo -e "${DIM}     python cli.py generate -n 10   # imágenes activadas por config${R}"
    echo -e ""

    echo -e "  ${B}4. O usar el CLI standalone:${R}"
    echo -e "${DIM}     # Test rápido (1 imagen)${R}"
    echo -e "${DIM}     python flux2_imagegen.py --test${R}"
    echo -e ""
    echo -e "${DIM}     # Generar desde scenarios.jsonl${R}"
    echo -e "${DIM}     python flux2_imagegen.py --scenarios data/raw/scenarios.jsonl --limit 10${R}"
    echo -e ""
    echo -e "${DIM}     # 12 incidencias de demostración con prompts fijos${R}"
    echo -e "${DIM}     python flux2_imagegen.py${R}"
    echo -e ""

    echo -e "  ${B}5. Ejemplo de prompt de imagen generado:${R}"
    python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT/miurba-generator')
from src.prompts import build_image_prompts
problems = [{
    'category': 'fontanería',
    'subcategory': 'rotura_tubería',
    'urgency': 'urgente',
    'location_zone': 'garaje',
    'location_detail': 'plaza 23',
    'affected_elements': [{'principal': 'tubería', 'subcategory': 'desagüe'}]
}]
community = {'type': 'bloque_pisos', 'num_viviendas': 48, 'year': 1985}
prompts = build_image_prompts(problems, community, 2)
for i, p in enumerate(prompts, 1):
    print(f'  Foto {i}: {p}')
"
}

# ── Main ───────────────────────────────────────────────────────────────────

MODE="all"
if [ "${1:-}" = "--sample" ]; then MODE="sample"; fi
if [ "${1:-}" = "--generate" ]; then MODE="generate"; fi
if [ "${1:-}" = "--extract" ]; then MODE="extract"; fi
if [ "${1:-}" = "--flux" ]; then MODE="flux"; fi

banner "miUrba — Demo del pipeline completo"
echo -e "  Generador de quejas sintéticas + Extractor con VLM"
echo -e "  ${DIM}Repo: $(basename "$REPO_ROOT")${R}"
echo -e "  ${DIM}Modo: $MODE${R}"

mkdir -p "$DEMO_DATA"

case "$MODE" in
    all)
        phase_generate
        phase_extract
        phase_flux
        ;;
    generate)
        phase_generate
        ;;
    extract)
        phase_extract
        ;;
    sample)
        phase_sample
        ;;
    flux)
        phase_flux
        ;;
esac

banner "Demo completada"
echo -e "  ${DIM}Datos de la demo: $DEMO_DATA${R}"
echo -e "  ${DIM}Datos de muestra: $SAMPLE_DATA${R}"
echo -e ""
echo -e "  Para ver los datos pre-generados sin LLM:"
echo -e "  ${C}bash demo/run_demo.sh --sample${R}"
echo -e ""
echo -e "  Para ver cómo funciona la generación de imágenes:"
echo -e "  ${C}bash demo/run_demo.sh --flux${R}"