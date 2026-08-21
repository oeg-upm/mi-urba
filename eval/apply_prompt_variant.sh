#!/bin/bash
# miUrba — Apply prompt variant for an iteration
# Copies the variant .bak file into prompts.py before running an iteration.
#
# Usage: bash eval/apply_prompt_variant.sh <iteration_number>
#
# Variants:
#   iter 2: prompts_iter2.py.bak (urgencia palabras clave, subcat explícita, formato elements)
#   iter 3: prompts_iter3.py.bak (foco: reducir FP + mejorar risks con mapeo explícito)
#   iter 4: prompts_iter4.py.bak (foco: urgencia al inicio + subcat primera mención)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ITER=${1:?Usage: apply_prompt_variant.sh <iteration_number>}

GEN_PROMPT="$ROOT_DIR/miurba-generator/src/prompts.py"
EXT_PROMPT="$ROOT_DIR/miurba-extractor/src/prompts.py"

GEN_VARIANT="$ROOT_DIR/miurba-generator/src/prompts_iter${ITER}.py.bak"
EXT_VARIANT="$ROOT_DIR/miurba-extractor/src/prompts_iter${ITER}.py.bak"

if [ ! -f "$GEN_VARIANT" ]; then
    echo "ERROR: Generator variant not found: $GEN_VARIANT"
    exit 1
fi
if [ ! -f "$EXT_VARIANT" ]; then
    echo "ERROR: Extractor variant not found: $EXT_VARIANT"
    exit 1
fi

cp "$GEN_VARIANT" "$GEN_PROMPT"
cp "$EXT_VARIANT" "$EXT_PROMPT"

echo "Applied prompt variant iter_${ITER} to:"
echo "  $GEN_PROMPT"
echo "  $EXT_PROMPT"