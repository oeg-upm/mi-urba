#!/usr/bin/env python3
"""Measure load+warm time for the 20 remaining models that weren't measured
due to a backend crash. Appends to the existing load_times.log."""
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


EP = "http://192.168.10.1:4000/v1/chat/completions"
KEY = get_api_key()
H = {"Content-Type":"application/json","Authorization":f"Bearer {KEY}"}
LOG = "/mnt/backup/pc1/data/miUrba2/mi-urba/eval/results/load_times.log"
FAST = "/mnt/backup/pc1/data/miUrba2/mi-urba/eval/results/fast_models.json"

SYS = ("Eres un vecino de una comunidad de vecinos. Escribe lo que un vecino "
       "REAL escribiría para reportar un problema de mantenimiento. "
       "RESPONDE SOLO con JSON válido, sin markdown.")
USR = ('PROBLEMA A REPORTAR:\n  1. fontanería/fugas en garaje, plaza 23 (urgencia: urgente)\n\n'
       'INSTRUCCIONES:\nEscribe el user_text como lo expresaría este vecino. Es urgente, no puede esperar.\n'
       'RESPONDE SOLO con JSON: {"user_text": "...", "timestamp": "2026-08-23T10:00:00Z"}')

# Find missing models
done = set()
for line in open(LOG):
    m = re.match(r'\[\s*\d+/\d+\]\s+(?:FAST|slow|LOAD_|WARM_)\s+(\S+)', line)
    if m: done.add("local_llama/"+m.group(1))

tested = {}; rank = {"OK  ":0,"WEAK":1,"FAIL":2,"ERR  ":3}
for logf in ["eval/results/gen_all_models.log","eval/results/gen_retest_problems.log","eval/results/gen_retest_isolated.log","eval/results/gen_retest_final.log"]:
    try:
        for line in open(logf):
            m = re.match(r'(?:\[\s*\d+/\d+\]\s+)?(OK  |WEAK|FAIL|ERR  )\s+(\S+)\s+', line)
            if not m: continue
            mark, model = m.groups(); full = "local_llama/"+model
            if full not in tested or rank[mark] < tested[full]: tested[full] = rank[mark]
    except FileNotFoundError: pass
ok_models = sorted(m for m,r in tested.items() if r==0)
missing = [m for m in ok_models if m not in done]
print(f"=== Measuring {len(missing)} missing models ===", flush=True)

results = []
for i, model in enumerate(missing, 1):
    tag = model.replace("local_llama/","")
    extra = {"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
    # Phase 1: cold load (max 15min)
    t0 = time.time()
    try:
        r = requests.post(EP, headers=H, json={
            "model": model, "messages": [{"role":"user","content":"Di OK"}],
            "max_tokens": 5, "temperature": 0, **extra
        }, timeout=900)
        d = r.json()
        load_s = time.time() - t0
        if "choices" not in d:
            line = f"[{i:3d}/{len(missing)}] LOAD_ERR {tag:30s} ({load_s:6.1f}s): {str(d.get('error',d))[:40]}"
            print(line, flush=True)
            with open(LOG,"a") as f: f.write(line+"\n")
            time.sleep(30); continue
    except Exception as e:
        load_s = time.time() - t0
        line = f"[{i:3d}/{len(missing)}] LOAD_TIMEOUT {tag:30s} ({load_s:6.1f}s)"
        print(line, flush=True)
        with open(LOG,"a") as f: f.write(line+"\n")
        time.sleep(30); continue
    # Phase 2: warm latency
    t0 = time.time()
    try:
        r = requests.post(EP, headers=H, json={
            "model": model, "messages": [{"role":"user","content":"Di OK"}],
            "max_tokens": 5, "temperature": 0, **extra
        }, timeout=60)
        d = r.json()
        warm_s = time.time() - t0
        if "choices" not in d:
            line = f"[{i:3d}/{len(missing)}] WARM_ERR {tag:30s} load={load_s:6.1f}s warm={warm_s:6.1f}s"
            print(line, flush=True)
            with open(LOG,"a") as f: f.write(line+"\n")
            time.sleep(30); continue
    except Exception:
        warm_s = time.time() - t0
        line = f"[{i:3d}/{len(missing)}] WARM_TIMEOUT {tag:30s} load={load_s:6.1f}s warm={warm_s:6.1f}s"
        print(line, flush=True)
        with open(LOG,"a") as f: f.write(line+"\n")
        time.sleep(30); continue
    # Phase 3: gen
    t0 = time.time(); gen_s = None
    try:
        r = requests.post(EP, headers=H, json={
            "model": model,
            "messages": [{"role":"system","content":SYS},{"role":"user","content":USR}],
            "max_tokens": 300, "temperature": 0.85, **extra
        }, timeout=60)
        d = r.json(); gen_s = time.time() - t0
        if "choices" not in d: gen_s = None
    except Exception: gen_s = time.time() - t0
    qualifies = (load_s < 300) and (warm_s < 30)
    mark = "FAST" if qualifies else "slow"
    line = f"[{i:3d}/{len(missing)}] {mark} {tag:30s} load={load_s:6.1f}s warm={warm_s:5.1f}s gen={gen_s if gen_s else 0:5.1f}s"
    print(line, flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")
    if qualifies:
        results.append({"model": model, "load_s": load_s, "warm_s": warm_s, "gen_s": gen_s})
    time.sleep(30)

print(f"\n=== {len(results)} new fast models ===", flush=True)
# Merge with existing fast_models.json
existing = []
try:
    existing = json.load(open(FAST))
except: pass
all_fast = existing + results
# Dedup by model
seen = set(); merged = []
for r in all_fast:
    if r["model"] not in seen:
        seen.add(r["model"]); merged.append(r)
with open(FAST, "w") as f:
    json.dump(merged, f, indent=2)
print(f"=== Total fast models: {len(merged)} saved to {FAST} ===", flush=True)
print("=== MISSING MEASURE DONE ===", flush=True)