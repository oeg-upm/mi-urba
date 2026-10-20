#!/usr/bin/env python3
"""Exploración del 10K: 10 generaciones por cada uno de los 10 modelos,
1 petición a la vez, usando el PROMPT REAL del generador (build_generation_prompt).

Estima por modelo: latencia media/queja y ETA de sus 1000 quejas.
Salida: eval/results/dataset-10k_exploration.json
"""
import os
import sys as _sys
from pathlib import Path as _Path

def _env_path():
    return Path(__file__).resolve().parents[3] / ".env"


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
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "miurba-generator"))
from src.prompts import build_generation_prompt  # noqa: E402
import yaml

URL = "http://192.168.10.1:4000/v1/chat/completions"
H = {"Content-Type": "application/json",
     "Authorization": f"Bearer {get_api_key()}"}
CFG = yaml.safe_load(open(ROOT / "experiments" / "extraccion_llm" / "config.eval.yaml"))

MODELS = [
    "gen_muse-glimmer-30b",
    "gen_glm-4.5-air",
]
N_GEN = 10

import requests

def warmup(model_id, max_wait=360):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            r = requests.post(URL, headers=H, timeout=90, json={
                "model": f"local_llama/{model_id}",
                "messages": [{"role": "user", "content": "Di OK"}],
                "max_tokens": 5, "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False}})
            if r.status_code == 200 and "choices" in r.json():
                return time.time() - t0
        except requests.RequestException:
            pass
        time.sleep(10)
    return None

results = {}
_mf = Path("/out") / "dataset-10k_exploration_partial.json"
if _mf.exists():
    results = json.loads(_mf.read_text())
for genkey in MODELS:
    name = genkey[len("gen_"):]
    mc = CFG["models"][genkey]
    t_w = warmup(name)
    if t_w is None:
        results[name] = {"warmup": None, "status": "WARMUP_FAIL"}
        print(f"{name:30s} WARMUP FAIL", flush=True)
        continue

    ep = CFG["endpoints"][mc["endpoint"] if (mc := CFG["models"][genkey]) else "litellm"]
    base = mc.get("temperature", 0.85)
    extra = mc.get("extra_params", {}) or {}

    times, tokens, ok_count = [], [], 0
    for i in range(N_GEN):
        # real production prompt with a sampled-style scenario (deterministic sample)
        problems = [{"category": "fontanería", "subcategory": "fugas",
                     "location_zone": "garaje", "location_detail": "plaza 23",
                     "urgency": "urgente", "risks": ["inundación"],
                     "affected_elements": [{"principal": "fontanería", "subcategory": "batería"}]}]
        msgs = build_generation_prompt(
            record_id=f"QEXP{i+1:03d}",
            problems=problems,
            community={"name": "Residencial Las Acacias", "type": "bloque_pisos",
                       "num_viviendas": 40, "year": 1998, "has_elevator": True,
                       "has_pool": False, "has_garage": True},
            num_pics=0,
            persona={"profile": "Vecino observado y formal, escribe con cortesía.",
                     "style": "formal"},
            modifier={"description": "ninguno", "style_override": "ninguno"},
        )
        payload = {"model": mc["model"], "temperature": base,
                   "max_tokens": extra.get("max_tokens", 10000),
                   "messages": msgs, **{k: v for k, v in extra.items() if k != "max_tokens"}}
        t0 = time.time()
        try:
            r = requests.post(URL, json=payload, headers=H, timeout=600)
            dt = time.time() - t0
            d = r.json()
            if "choices" in d:
                ok_count += 1
                tokens.append(d.get("usage", {}).get("completion_tokens", 0))
                times.append(dt)
        except Exception:
            times.append(float("nan"))
        time.sleep(2)

    valid = [t for t in times if t == t]
    m = sum(valid) / len(valid) if valid else float("nan")
    eta_h = m * 1000 / 3600 if valid else None
    results[name] = {"warmup_s": round(t_w, 1) if (t_w := t_w) else t_w,
                     "ok": ok_count, "mean_s": round(m, 1),
                     "mean_tokens": round(sum(tokens) / len(tokens)) if tokens else 0,
                     "eta_1000_h": round(eta_h, 2) if eta_h else None}
    print(f"{name:30s} ok={ok_count}/10  mean={m:.1f}s  ETA/1000={eta_h and round(eta_h,2)}h", flush=True)
    time.sleep(5)  # settle: no in-flight requests cross the model switch

total_h = sum(r["eta_1000_h"] or 0 for r in results.values() if isinstance(r, dict) and r.get("eta_1000_h"))
print(f"\nTOTAL estimado dataset completo: {total_h:.1f}h (secuencial, 1 worker)", flush=True)
results["_total_eta_h"] = round(total_h, 1)
out = Path("/out") / "dataset-10k_exploration_partial.json" if Path("/out").exists() else ROOT / "eval" / "results" / "dataset-10k_exploration.json"
out.write_text(json.dumps(results, indent=1))
# merge into final JSON (repo side) if writable, else save partial
final = ROOT / "eval" / "results" / "dataset-10k_exploration.json"
try:
    final.write_text(json.dumps(results, indent=1))
    print(f"guardado -> {final}")
except OSError:
    out.write_text(json.dumps(results, indent=1))
    print(f"guardado -> {out}")