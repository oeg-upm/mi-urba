#!/usr/bin/env python3
"""Final retest:
1. 4 stubborn models (phi-4-reasoning-plus let think; 3 timeouts with 15min)
2. Any NEW models in the endpoint not yet tested in any previous log.

Method for all: max_tokens=2000, enable_thinking=false (except phi-4-reasoning-plus
which gets think-free), 15min timeout. Strictly sequential, 30s cooldown.
"""
import requests, json, time, re

EP = "http://192.168.10.1:4000/v1/chat/completions"
from env_utils import get_api_key
KEY = get_api_key()
H = {"Content-Type":"application/json","Authorization":f"Bearer {KEY}"}
LOG = "/mnt/backup/pc1/data/miUrba2/mi-urba/eval/results/gen_retest_final.log"

SYS = ("Eres un vecino de una comunidad de vecinos. Escribe lo que un vecino "
       "REAL escribiría para reportar un problema de mantenimiento. "
       "RESPONDE SOLO con JSON válido, sin markdown.")
USR = ('PROBLEMA A REPORTAR:\n  1. fontanería/fugas en garaje, plaza 23 (urgencia: urgente)\n\n'
       'INSTRUCCIONES:\nEscribe el user_text como lo expresaría este vecino. Es urgente, no puede esperar.\n'
       'RESPONDE SOLO con JSON: {"user_text": "...", "timestamp": "2026-08-23T10:00:00Z"}')

# Collect every model already tested (across all logs)
tested = set()
for logf in ["eval/results/gen_all_models.log",
             "eval/results/gen_retest_problems.log",
             "eval/results/gen_retest_isolated.log"]:
    try:
        for line in open(logf):
            m = re.search(r'\]\s+(?:OK  |WEAK|FAIL|ERR  )\s+(\S+)\s+\(|^(?:OK  |WEAK|FAIL|ERR  )\s+(\S+)\s+\S+\s+\(', line)
            if m:
                model = m.group(1) or m.group(2)
                tested.add("local_llama/" + model)
    except FileNotFoundError:
        pass

# Current endpoint model list
r = requests.get("http://192.168.10.1:4000/v1/models", headers=H, timeout=30)
all_models = [m["id"] for m in r.json().get("data", [])]
locals_ = [m for m in sorted(all_models) if m.startswith("local_llama/") and "embed" not in m.lower()]

new_models = [m for m in locals_ if m not in tested]
print(f"=== Already tested: {len(tested & set(locals_))} | New (untested): {len(new_models)} ===", flush=True)
for m in new_models:
    print(f"  NEW: {m}", flush=True)

# Build case list
cases = []
# 4 stubborn
cases.append(("local_llama/phi-4-reasoning-plus", 4000, None, 600, "think_free"))
cases.append(("local_llama/deepseek-v4-flash-iq2", 2000, False, 900, "15min"))
cases.append(("local_llama/falcon-7b-instruct",   2000, False, 900, "15min"))
cases.append(("local_llama/mixtral-8x7b-instruct",2000, False, 900, "15min"))
# New models: max_tokens=2000, no_think, 15min timeout
for m in new_models:
    cases.append((m, 2000, False, 900, "new"))

with open(LOG, "w") as f:
    f.write(f"=== Final retest: 4 stubborn + {len(new_models)} new ===\n")
print(f"\n=== Retesting {len(cases)} models (4 stubborn + {len(new_models)} new) ===", flush=True)

for model, mt, think, timeout, label in cases:
    tag = model.replace("local_llama/","")
    payload = {"model":model,"messages":[{"role":"system","content":SYS},{"role":"user","content":USR}],
               "temperature":0.85,"max_tokens":mt}
    if think is not None:
        payload["enable_thinking"] = think
        payload["chat_template_kwargs"] = {"enable_thinking": think}
    try:
        t0=time.time()
        r=requests.post(EP,headers=H,json=payload,timeout=timeout)
        d=r.json()
        dt=time.time()-t0
        if "choices" in d:
            c=d["choices"][0]["message"]["content"]
            fin = d["choices"][0].get("finish_reason")
            cs = c
            if cs.startswith("```"): cs = "\n".join(l for l in cs.split("\n") if not l.strip().startswith("```"))
            parsed = {}
            try: parsed = json.loads(cs.strip())
            except json.JSONDecodeError:
                jm = re.search(r'\{.*\}', cs, re.DOTALL)
                if jm:
                    try: parsed = json.loads(jm.group(0))
                    except: pass
            txt = parsed.get("user_text","")
            ok = any(w in txt.lower() for w in ["fuga","agua","garaj","plaza","tuber","urgent"]) and len(txt) > 30
            mark = "OK  " if ok else ("WEAK" if txt else "FAIL")
            line = f"{mark} {tag:28s} {label:11s} ({dt:6.1f}s, finish={fin}): {txt[:80]}"
            if not txt:
                line += f" | RAW: {repr(c)[:200]}"
        else:
            line = f"ERR  {tag:28s} {label:11s} ({dt:6.1f}s): {str(d.get('error',d))[:60]}"
    except Exception as e:
        dt=time.time()-t0
        line = f"ERR  {tag:28s} {label:11s} ({dt:6.1f}s): {str(e)[:60]}"
    print(line, flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")
    time.sleep(30)
print("=== FINAL RETEST DONE ===", flush=True)