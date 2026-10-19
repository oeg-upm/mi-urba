#!/usr/bin/env bash
# docker_smoke.sh — full smoke test of the dockerized miUrba pipeline.
#
# Usage:  bash scripts/docker_smoke.sh
# Requires: docker, docker compose, and a live endpoint (config.test.yaml).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0; FAIL=0
ok()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
fail() { echo "  ❌ $1"; FAIL=$((FAIL+1)); }

ENDPOINT="${MIURBA_SMOKE_ENDPOINT:-http://192.168.10.11:9292}"

echo "── 0. Precondiciones"
curl -s --max-time 5 "$ENDPOINT/v1/models" >/dev/null \
  && ok "endpoint $ENDPOINT vivo" || { fail "endpoint caído — abortando"; exit 1; }
[ -f .env ] && echo "  (usando .env para secretos)" || cp .env.example .env
rm -rf "$ROOT/data_test" "$ROOT/logs"   # smoke limpio: sin resume de runs anteriores

echo "── 1. Build de imágenes"
docker compose build -q && ok "imágenes construidas" || fail "build"

echo "── 2. Config visible dentro del contenedor"
docker compose run --rm --no-deps --entrypoint python miurba-generator \
  -c "import yaml; c=yaml.safe_load(open('/config.yaml')); assert 'generation' in c; assert 'logging' in c" \
  && ok "config.yaml montado en /config.yaml" || fail "config mount"

echo "── 3. Generación dockerizada (3 quejas, workers=3, juez externo)"
if ! GEN_OUT=$(docker compose run --rm miurba-generator \
  generate --config /config/config.test.yaml 2>&1); then
  fail "generate: docker compose run terminó con error"; echo "$GEN_OUT" | tail -5; GEN_OUT="";
fi
EXPECTED=$(python3 -c "import yaml; print(sum(b['count'] for b in yaml.safe_load(open('config.test.yaml'))['generation']['batches']))")
echo "$GEN_OUT" | grep -q "gen=$EXPECTED" && ok "generate: $EXPECTED/$EXPECTED" || { fail "generate"; echo "$GEN_OUT" | tail -5; }
echo "$GEN_OUT" | grep -q "errors absorbed" && echo "  ⚠ errores absorbidos (revisar logs)" || ok "sin errores absorbidos"

echo "── 4. Extracción dockerizada (stage-split + json_schema)"
if ! EXT_OUT=$(docker compose run --rm miurba-extractor \
  extract --config /config/config.test.yaml 2>&1); then
  fail "extract: docker compose run terminó con error"; echo "$EXT_OUT" | tail -5; EXT_OUT="";
fi
echo "$EXT_OUT" | grep -qE "Extracted:\s+$EXPECTED" && ok "extract: $EXPECTED/$EXPECTED" || { fail "extract"; echo "$EXT_OUT" | tail -5; }

echo "── 5. Secretos: \${VAR} expandida desde .env (fail-fast si falta)"
docker compose run --rm --no-deps -e TEST_SECRET=sk-x --entrypoint python miurba-generator \
  -c "
import sys; sys.path.insert(0,'/app')
from src.config import load_config, _expand_vars
assert _expand_vars({'k': '\${TEST_SECRET}'})['k'] == 'sk-x'
print('ok')" >/dev/null && ok "\${VAR} se expande en el contenedor" || fail "\${VAR} expansion"

echo "── 6. Log de run escrito en el volumen montado"
LOG_FILE="$ROOT/logs/run.log"
if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
  ok "logs/run.log existe y no está vacío ($(wc -l < "$LOG_FILE") líneas)"
  grep -qE 'API call failed|Generated|Extracting' "$LOG_FILE" \
    && ok "el log contiene eventos del run" || fail "log sin contenido esperado"
else
  fail "logging.file no escribió en ./logs/"
fi

echo ""
echo "══════════════════════════════════"
echo "Resultado: $PASS pasados, $FAIL fallidos"
[ "$FAIL" -eq 0 ]
