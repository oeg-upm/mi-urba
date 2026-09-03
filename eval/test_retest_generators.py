#!/usr/bin/env python3
"""Test the remaining local_llama generators that were skipped due to backend crash."""
import requests, json, time, re

ENDPOINT = "http://192.168.10.1:4000/v1/chat/completions"
from env_utils import get_api_key
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

# Already done
done = set()
for line in open(LOG):
    m = re.search(r'\]\s+(?:OK  |WEAK|FAIL|ERR  )\s+(\S+)', line)
    if m: done.add("local_llama/" + m.group(1))

r = requests.get("http://192.168.10.1:4000/v1/models", headers=HEADERS, timeout=30)
all_models = [m["id"] for m in r.json().get("data", [])]
# Only local_llama, not embed, not already done
candidates = [m for m in sorted(all_models)
              if m.startswith("local_llama/") and "embed" not in m.lower() and m not in done]

print(f"=== Testing {len(candidates)} remaining models (5min timeout) ===", flush=True)

for i, model in enumerate(candidates, 1):
    tag = model.replace("local_llama/", "")
    payload = {"model": model, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ], "max_tokens": 250, "temperature": 0.85}
    try:
        t0 = time.time()
        r = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=300)
        d = r.json()
        dt = time.time() - t0
        if "choices" in d:
            content = d["choices"][0]["message"]["content"]
            c = content
            if c.startswith("```"): c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
            try:
                parsed = json.loads(c.strip())
                txt = parsed.get("user_text", "")
                ok = any(w in txt.lower() for w in ["fuga","agua","garaj","plaza","tuber","urgent"]) and len(txt) > 30
                mark = "OK  " if ok else "WEAK"
                line = f"[{i:3d}/{len(candidates)}] {mark} {tag:30s} ({dt:6.1f}s): {txt[:80]}"
            except Exception as pe:
                line = f"[{i:3d}/{len(candidates)}] FAIL {tag:30s} ({dt:6.1f}s): JSON parse: {pe}. {content[:40]}"
        else:
            line = f"[{i:3d}/{len(candidates)}] ERR  {tag:30s} ({dt:6.1f}s): {str(d.get('error',d))[:50]}"
    except Exception as e:
        dt = time.time() - t0
        line = f"[{i:3d}/{len(candidates)}] ERR  {tag:30s} ({dt:6.1f}s): {str(e)[:50]}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
print("=== RETEST DONE ===", flush=True)