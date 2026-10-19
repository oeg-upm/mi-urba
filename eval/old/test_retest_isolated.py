#!/usr/bin/env python3
"""Retest problem models changing ONE variable at a time.

For each FAIL model, test two variants separately:
  A: max_tokens=2000 (no thinking change) — was it a truncation issue?
  B: enable_thinking=false (max_tokens=250) — was it thinking active?
For never-tested models: 15min timeout, max_tokens=400.

Strictly sequential with 30s cooldown between requests to avoid
overlapping model loads in llama-swap (which would cause both to fail).
"""
from pathlib import Path
import os
import requests, json, time, re

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
LOG = "/mnt/backup/pc1/data/miUrba2/mi-urba/eval/results/gen_retest_isolated.log"
COOLDOWN = 30  # seconds between requests

SYSTEM = ("Eres un vecino de una comunidad de vecinos. Escribe lo que un vecino "
          "REAL escribiría para reportar un problema de mantenimiento. "
          "RESPONDE SOLO con JSON válido, sin markdown.")
USER = """PROBLEMA A REPORTAR:
  1. fontanería/fugas en garaje, plaza 23 (urgencia: urgente)

INSTRUCCIONES:
Escribe el user_text como lo expresaría este vecino. Es urgente, no puede esperar.
RESPONDE SOLO con JSON: {"user_text": "...", "timestamp": "2026-08-23T10:00:00Z"}"""

# Determine which models need retest
main_log = "/mnt/backup/pc1/data/miUrba2/mi-urba/eval/results/gen_all_models.log"
fail_models = set()
for line in open(main_log):
    m = re.match(r'\[\s*\d+/\d+\]\s+(OK  |WEAK|FAIL|ERR  )\s+(\S+)\s+\(', line)
    if not m: continue
    if m.group(1) == "FAIL": fail_models.add("local_llama/"+m.group(2))

# Never tested
r = requests.get("http://192.168.10.1:4000/v1/models", headers=HEADERS, timeout=30)
all_models = [m["id"] for m in r.json().get("data", [])]
candidates = [m for m in sorted(all_models) if m.startswith("local_llama/") and "embed" not in m.lower()]
tested = set()
for line in open(main_log):
    m = re.search(r'\]\s+(?:OK  |WEAK|FAIL|ERR  )\s+(\S+)\s+\(', line)
    if m: tested.add("local_llama/"+m.group(1))
never_tested = [m for m in candidates if m not in tested]

print(f"=== Isolated retest: {len(fail_models)} FAIL models (2 variants A/B) + {len(never_tested)} never-tested (15min) ===", flush=True)
print(f"=== Cooldown {COOLDOWN}s between requests ===", flush=True)

with open(LOG, "w") as f:
    f.write(f"=== Isolated retest ===\n")

def run_one(model, variant, max_tokens, enable_thinking, timeout):
    tag = model.replace("local_llama/", "")
    payload = {"model": model, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ], "temperature": 0.85, "max_tokens": max_tokens}
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    try:
        t0 = time.time()
        r = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=timeout)
        d = r.json()
        dt = time.time() - t0
        if "choices" in d:
            content = d["choices"][0]["message"]["content"]
            c = content
            if c.startswith("```"): c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
            try:
                parsed = json.loads(c.strip())
            except json.JSONDecodeError:
                jm = re.search(r'\{.*\}', c, re.DOTALL)
                if jm:
                    try: parsed = json.loads(jm.group(0))
                    except: parsed = {}
                else: parsed = {}
            txt = parsed.get("user_text", "")
            ok = any(w in txt.lower() for w in ["fuga","agua","garaj","plaza","tuber","urgent"]) and len(txt) > 30
            mark = "OK  " if ok else ("WEAK" if txt else "FAIL")
            line = f"{mark} {tag:30s} {variant:12s} ({dt:6.1f}s): {txt[:80]}"
        else:
            line = f"ERR  {tag:30s} {variant:12s} ({dt:6.1f}s): {str(d.get('error',d))[:50]}"
    except Exception as e:
        dt = time.time() - t0
        line = f"ERR  {tag:30s} {variant:12s} ({dt:6.1f}s): {str(e)[:50]}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    return dt

# Phase 1: FAIL models, variant A (max_tokens=2000, thinking unchanged)
print("\n=== PHASE 1: FAIL models, variant A (max_tokens=2000) ===", flush=True)
for i, model in enumerate(sorted(fail_models), 1):
    print(f"  [{i}/{len(fail_models)}]", end=" ", flush=True)
    run_one(model, "A_mt2000", max_tokens=2000, enable_thinking=None, timeout=300)
    time.sleep(COOLDOWN)

# Phase 2: FAIL models, variant B (enable_thinking=false, max_tokens=250)
print("\n=== PHASE 2: FAIL models, variant B (enable_thinking=false) ===", flush=True)
for i, model in enumerate(sorted(fail_models), 1):
    print(f"  [{i}/{len(fail_models)}]", end=" ", flush=True)
    run_one(model, "B_nothink", max_tokens=250, enable_thinking=False, timeout=300)
    time.sleep(COOLDOWN)

# Phase 3: never-tested, 15min timeout
print(f"\n=== PHASE 3: never-tested models (15min timeout) ===", flush=True)
for i, model in enumerate(sorted(never_tested), 1):
    print(f"  [{i}/{len(never_tested)}]", end=" ", flush=True)
    run_one(model, "C_15min", max_tokens=400, enable_thinking=False, timeout=900)
    time.sleep(COOLDOWN)

print("\n=== ISOLATED RETEST DONE ===", flush=True)