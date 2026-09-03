#!/usr/bin/env python3
"""Measure load time + warm inference latency for each OK model.

For each model:
1. First request with 900s (15min) timeout — measures LOAD time (cold start).
2. Second request (model already loaded) with 60s timeout — measures WARM latency.
3. A third request (real generation prompt, max_tokens=300) — measures generation latency.

Keep only models that:
- load in < 5 min (300s)
- warm-respond in < 30s

Strictly sequential, 30s cooldown between models.
Writes load_times.log + a JSON list of qualifying models to fast_models.json.
"""
import requests, json, time, re

EP = "http://192.168.10.1:4000/v1/chat/completions"
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # eval/env_utils
from env_utils import get_api_key
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

# Load OK models (same logic as the diversity generator)
def load_ok_models():
    tested = {}
    rank = {"OK  ":0, "WEAK":1, "FAIL":2, "ERR  ":3}
    logs = ["eval/results/gen_all_models.log",
            "eval/results/gen_retest_problems.log",
            "eval/results/gen_retest_isolated.log",
            "eval/results/gen_retest_final.log"]
    base = "/mnt/backup/pc1/data/miUrba2/mi-urba"
    for logf in logs:
        p = base + "/" + logf
        try:
            for line in open(p):
                m = re.match(r'(?:\[\s*\d+/\d+\]\s+)?(OK  |WEAK|FAIL|ERR  )\s+(\S+)\s+', line)
                if not m: continue
                mark, model = m.groups()
                full = "local_llama/" + model
                if full not in tested or rank[mark] < tested[full]:
                    tested[full] = rank[mark]
        except FileNotFoundError:
            pass
    return sorted(m for m, r in tested.items() if r == 0)

models = load_ok_models()
print(f"=== Measuring load + warm latency for {len(models)} models ===", flush=True)
with open(LOG, "w") as f:
    f.write(f"=== {len(models)} models ===\n")

results = []  # (model, load_s, warm_s, gen_s)
for i, model in enumerate(models, 1):
    tag = model.replace("local_llama/", "")
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
            line = f"[{i:3d}/{len(models)}] {tag:30s} LOAD_ERR ({load_s:6.1f}s): {str(d.get('error',d))[:40]}"
            print(line, flush=True)
            with open(LOG,"a") as f: f.write(line+"\n")
            time.sleep(30); continue
    except Exception as e:
        load_s = time.time() - t0
        line = f"[{i:3d}/{len(models)}] {tag:30s} LOAD_TIMEOUT ({load_s:6.1f}s)"
        print(line, flush=True)
        with open(LOG,"a") as f: f.write(line+"\n")
        time.sleep(30); continue

    # Phase 2: warm latency (simple prompt, max 60s)
    t0 = time.time()
    try:
        r = requests.post(EP, headers=H, json={
            "model": model, "messages": [{"role":"user","content":"Di OK"}],
            "max_tokens": 5, "temperature": 0, **extra
        }, timeout=60)
        d = r.json()
        warm_s = time.time() - t0
        if "choices" not in d:
            line = f"[{i:3d}/{len(models)}] {tag:30s} WARM_ERR load={load_s:6.1f}s warm={warm_s:6.1f}s"
            print(line, flush=True)
            with open(LOG,"a") as f: f.write(line+"\n")
            time.sleep(30); continue
    except Exception as e:
        warm_s = time.time() - t0
        line = f"[{i:3d}/{len(models)}] {tag:30s} WARM_TIMEOUT load={load_s:6.1f}s warm={warm_s:6.1f}s"
        print(line, flush=True)
        with open(LOG,"a") as f: f.write(line+"\n")
        time.sleep(30); continue

    # Phase 3: real generation (max_tokens=300, max 60s)
    t0 = time.time()
    gen_s = None
    try:
        r = requests.post(EP, headers=H, json={
            "model": model,
            "messages": [{"role":"system","content":SYS},{"role":"user","content":USR}],
            "max_tokens": 300, "temperature": 0.85, **extra
        }, timeout=60)
        d = r.json()
        gen_s = time.time() - t0
        if "choices" not in d:
            gen_s = None
    except Exception:
        gen_s = time.time() - t0

    # Qualify: load < 300s AND warm < 30s
    qualifies = (load_s < 300) and (warm_s < 30)
    mark = "FAST" if qualifies else "slow"
    line = f"[{i:3d}/{len(models)}] {mark} {tag:30s} load={load_s:6.1f}s warm={warm_s:5.1f}s gen={gen_s if gen_s else 0:5.1f}s"
    print(line, flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")
    if qualifies:
        results.append({"model": model, "load_s": load_s, "warm_s": warm_s, "gen_s": gen_s})
    time.sleep(30)

# Save fast models list
with open(FAST, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n=== {len(results)} fast models (load<5min, warm<30s) saved to {FAST} ===", flush=True)
for r in results:
    print(f"  {r['model']:35s} load={r['load_s']:.1f}s warm={r['warm_s']:.1f}s", flush=True)
print("=== MEASURE DONE ===", flush=True)