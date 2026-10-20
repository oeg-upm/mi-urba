#!/usr/bin/env python3
"""Test ALL local_llama models as complaint generators.

For each model, send a minimal generation prompt with a 900s (15min) timeout
to allow for model loading. Log OK/WEAK/FAIL + load time + sample text.
Writes results incrementally to gen_all_models.log.
"""
from pathlib import Path
import os
import requests, json, time, sys, os

def _env_path():
    return Path(__file__).resolve().parents[2] / ".env"


def load_env() -> None:
    """Populate os.environ from the repo-root .env (existing vars win)."""
    env = _env_path()
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key(name: str = "LLAMA_GATEWAY_API_KEY") -> str:
    """Return the API key for `name`, from env vars or the repo .env file."""
    load_env()
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}: set it in {_env_path()} (see .env.example) "
            "or export it before running."
        )
    return value


ENDPOINT = "http://192.168.10.1:4000/v1/chat/completions"
KEY = get_api_key()
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
LOG = "/mnt/backup/pc1/data/miUrba2/mi-urba/eval/results/gen_all_models.log"

SYSTEM = ("Eres un vecino de una comunidad de vecinos. Escribe lo que un vecino "
          "REAL escribiría para reportar un problema de mantenimiento. "
          "RESPONDE SOLO con JSON válido, sin markdown.")
USER = """PROBLEMA A REPORTAR:
  1. fontanería/fugas en garaje, plaza 23 (urgencia: urgente)

INSTRUCCIONES:
Escribe el user_text como lo expresaría este vecino. Es urgente, no puede esperar.
RESPONDE SOLO con JSON: {"user_text": "...", "timestamp": "2026-08-23T10:00:00Z"}"""

# Get model list
try:
    r = requests.get("http://192.168.10.1:4000/v1/models", headers=HEADERS, timeout=30)
    all_models = [m["id"] for m in r.json().get("data", [])]
except Exception as e:
    print(f"FATAL: cannot list models: {e}", flush=True)
    sys.exit(1)

# Only local_llama/, skip embed models (they can't do chat completion)
candidates = [m for m in all_models
              if m.startswith("local_llama/") and "embed" not in m.lower()]
candidates.sort()

print(f"=== Testing {len(candidates)} local_llama models (15min timeout each) ===", flush=True)
print(f"=== Results log: {LOG} ===", flush=True)

results = {"OK": [], "WEAK": [], "FAIL": [], "ERR": []}

for i, model in enumerate(candidates, 1):
    tag = model.replace("local_llama/", "")
    payload = {"model": model, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ], "max_tokens": 250, "temperature": 0.85}
    try:
        t0 = time.time()
        r = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=900)
        d = r.json()
        dt = time.time() - t0
        if "choices" in d:
            content = d["choices"][0]["message"]["content"]
            c = content
            if c.startswith("```"):
                c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
            try:
                parsed = json.loads(c.strip())
                txt = parsed.get("user_text", "")
                ok = any(w in txt.lower() for w in ["fuga", "agua", "garaj", "plaza", "tuber", "urgent"]) and len(txt) > 30
                mark = "OK  " if ok else "WEAK"
                results["OK" if ok else "WEAK"].append(model)
                line = f"[{i:3d}/{len(candidates)}] {mark} {tag:30s} ({dt:6.1f}s): {txt[:80]}"
            except Exception as pe:
                results["FAIL"].append(model)
                line = f"[{i:3d}/{len(candidates)}] FAIL {tag:30s} ({dt:6.1f}s): JSON parse: {pe}. {content[:40]}"
        else:
            results["ERR"].append(model)
            line = f"[{i:3d}/{len(candidates)}] ERR  {tag:30s} ({dt:6.1f}s): {str(d.get('error', d))[:50]}"
    except Exception as e:
        results["ERR"].append(model)
        dt = time.time() - t0
        line = f"[{i:3d}/{len(candidates)}] ERR  {tag:30s} ({dt:6.1f}s): {str(e)[:50]}"

    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# Summary
print("\n" + "=" * 70, flush=True)
print(f"SUMMARY: OK={len(results['OK'])} WEAK={len(results['WEAK'])} FAIL={len(results['FAIL'])} ERR={len(results['ERR'])}", flush=True)
print(f"OK models ({len(results['OK'])}):", flush=True)
for m in results["OK"]:
    print(f"  {m}", flush=True)
with open(LOG, "a") as f:
    f.write(f"\nSUMMARY: OK={len(results['OK'])} WEAK={len(results['WEAK'])} FAIL={len(results['FAIL'])} ERR={len(results['ERR'])}\n")
    f.write("OK models:\n")
    for m in results["OK"]:
        f.write(f"  {m}\n")
print("=== DONE ===", flush=True)