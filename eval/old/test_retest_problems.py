#!/usr/bin/env python3
"""Retest problem models: JSON-FAIL with max_tokens=2000 + enable_thinking=false,
and never-tested (timeout) with 15min timeout.

Uses a separate log so the main gen_all_models.log stays clean."""
import requests, json, time, re

ENDPOINT = "http://192.168.10.1:4000/v1/chat/completions"
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # eval/env_utils
from env_utils import get_api_key
KEY = get_api_key()
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
LOG = "/mnt/backup/pc1/data/miUrba2/mi-urba/eval/results/gen_retest_problems.log"

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
ok_models = set()
fail_models = set()
for line in open(main_log):
    m = re.match(r'\[\s*\d+/\d+\]\s+(OK  |WEAK|FAIL|ERR  )\s+(\S+)\s+\(', line)
    if not m: continue
    mark, model = m.groups()
    if mark in ("OK  ", "WEAK"): ok_models.add("local_llama/"+model)
    elif mark == "FAIL": fail_models.add("local_llama/"+model)

# Never tested (from full model list)
r = requests.get("http://192.168.10.1:4000/v1/models", headers=HEADERS, timeout=30)
all_models = [m["id"] for m in r.json().get("data", [])]
candidates = [m for m in sorted(all_models) if m.startswith("local_llama/") and "embed" not in m.lower()]
tested = set()
for line in open(main_log):
    m = re.search(r'\]\s+(?:OK  |WEAK|FAIL|ERR  )\s+(\S+)\s+\(', line)
    if m: tested.add("local_llama/"+m.group(1))
never_tested = [m for m in candidates if m not in tested]

# Build retest list: FAIL models (max_tokens=2000 + no_think) + never tested (15min timeout)
# Tag each with its mode
retest_list = []
for m in sorted(fail_models):
    retest_list.append((m, "fail"))
for m in sorted(never_tested):
    retest_list.append((m, "timeout"))

print(f"=== Retesting {len(retest_list)} problem models ===", flush=True)
print(f"  FAIL (max_tokens=2000 + enable_thinking=false): {len(fail_models)}", flush=True)
print(f"  never-tested (15min timeout): {len(never_tested)}", flush=True)

with open(LOG, "w") as f:
    f.write(f"=== Retest {len(retest_list)} problem models ===\n")

for i, (model, mode) in enumerate(retest_list, 1):
    tag = model.replace("local_llama/", "")
    # Mode-specific params
    payload = {"model": model, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ], "temperature": 0.85}
    if mode == "fail":
        payload["max_tokens"] = 2000  # room for thinking output
        payload["enable_thinking"] = False
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        timeout = 300
    else:  # timeout — give 15min for model loading
        payload["max_tokens"] = 400
        timeout = 900

    try:
        t0 = time.time()
        r = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=timeout)
        d = r.json()
        dt = time.time() - t0
        if "choices" in d:
            content = d["choices"][0]["message"]["content"]
            c = content
            if c.startswith("```"): c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
            # Try to extract JSON even if there's thinking prefix
            try:
                parsed = json.loads(c.strip())
            except json.JSONDecodeError:
                # Try to find JSON object in the content (after thinking)
                import re as _re
                jm = _re.search(r'\{.*\}', c, _re.DOTALL)
                if jm:
                    try: parsed = json.loads(jm.group(0))
                    except: parsed = {}
                else: parsed = {}
            txt = parsed.get("user_text", "")
            ok = any(w in txt.lower() for w in ["fuga","agua","garaj","plaza","tuber","urgent"]) and len(txt) > 30
            mark = "OK  " if ok else ("WEAK" if txt else "FAIL")
            line = f"[{i:3d}/{len(retest_list)}] {mark} {tag:30s} ({dt:6.1f}s, {mode:7s}): {txt[:80]}"
        else:
            line = f"[{i:3d}/{len(retest_list)}] ERR  {tag:30s} ({dt:6.1f}s, {mode:7s}): {str(d.get('error',d))[:50]}"
    except Exception as e:
        dt = time.time() - t0
        line = f"[{i:3d}/{len(retest_list)}] ERR  {tag:30s} ({dt:6.1f}s, {mode:7s}): {str(e)[:50]}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
print("=== RETEST PROBLEMS DONE ===", flush=True)