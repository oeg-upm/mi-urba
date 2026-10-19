#!/usr/bin/env python3
"""E7: juez de fidelidad generador→escenario (ver experiments/fidelidad_generador/README.md).

Para cada queja muestreada, envía al juez (GT JSON + texto) y recibe un veredicto JSON
estructurado: completeness por problema, invenciones, urgency/risks match, naturalness.

Salidas:
    eval/results/fidelidad/judgments_<modeltag>.jsonl   (un juicio por línea)
    eval/results/fidelidad/fidelity_summary.json        (métricas por generador)

Ejemplo:
    python3 eval/experiments/fidelidad_generador/judge_fidelity.py \
        --dataset eval/results/dataset-10k --n-per-model 100
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
JUDGE_URL = "http://192.168.10.11:9292/v1/chat/completions"
JUDGE_MODEL = "qwen3.8-27b-awq"
OUT = ROOT / "eval" / "results" / "fidelidad"

PROMPT = """Eres un evaluador exhaustivo. Te doy la DESCRIPCIÓN ESTRUCTURADA de una queja
vecinal (la que se usó para generarla) y el TEXTO final escrito. Evalúa la fidelidad del
texto a la descripción. No uses conocimiento externo: juzga solo contra la descripción.

DESCRIPCIÓN (GT):
{gt}

TEXTO:
{text}

Responde SOLO con JSON válido:
{{
  "problems": [
    {{"gt_index": 0, "present": true, "distorted": false,
      "urgency": "exact|softened|exaggerated|missing"}},
    ...
  ],
  "invented": ["descripción de cada problema/riesgo/elemento inventado"],
  "risks_ok": true,
  "naturalness": 4,
  "comments": "máx 1 frase"
}}
- "problems": UNA entrada por cada problema de la descripción GT, en orden.
- "present": false si el problema no se menciona en absoluto.
- "distorted": true si se menciona pero con datos incorrectos (categoria/elementos/ubicacion).
- "invented": problemas o riesgos del texto que NO están en la descripción.
- "risks_ok": false solo si un riesgo GT falta o se añade uno inventado relevante.
- "naturalness": 1-5, ¿parece una queja vecinal real escrita por un vecino?
"""


def get_api_key() -> str:
    v = os.environ.get("LLAMA_GATEWAY_API_KEY", "").strip()
    if not v:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, val = line.partition("=")
                    os.environ.setdefault(k.strip(), val.strip().strip('"').strip("'"))
            v = os.environ.get("LLAMA_GATEWAY_API_KEY", "").strip()
    return v


def judge(gt: dict, text: str, key: str, temp: float, retries: int = 3) -> dict | None:
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(
            gt=json.dumps(gt, ensure_ascii=False, indent=1), text=text)}],
        "temperature": temp, "max_tokens": 1500,
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    for attempt in range(retries):
        try:
            r = requests.post(JUDGE_URL, json=payload, headers=headers, timeout=600)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                raise ValueError("no JSON block in judge output")
            return json.loads(m.group(0))
        except Exception as e:
            print(f"    judge attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    return None


def load_pairs(dataset: Path, model_dir: str, n: int, seed: int):
    """Devuelve [(complaint_id, gt_problems, text)] muestreados con seed."""
    d = dataset / model_dir / "raw"
    gts = {json.loads(l)["complaint_id"]: json.loads(l)
           for l in open(d / "ground_truth.jsonl") if l.strip()}
    pairs = []
    for l in open(d / "complaints.jsonl"):
        if not l.strip():
            continue
        c = json.loads(l)
        gt = gts.get(c["id"])
        if gt and (c.get("user_text") or "").strip():
            pairs.append((c["id"], gt["extractions"], c["user_text"]))
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    return pairs[:n]


def metrics(j: dict | None, n_gt_problems: int) -> dict:
    if not j:
        return {"ok": False}
    probs = j.get("problems", [])
    present = sum(1 for p in probs if p.get("present"))
    return {
        "ok": True,
        "completeness": present / n_gt_problems if n_gt_problems else 1.0,
        "distorted": any(p.get("distorted") for p in probs),
        "hallucination": bool(j.get("invented")),
        "urgency_exact": all(p.get("urgency") == "exact" for p in probs) if probs else False,
        "risks_ok": bool(j.get("risks_ok")),
        "naturalness": j.get("naturalness"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "eval" / "results" / "dataset-10k"))
    ap.add_argument("--n-per-model", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="*", help="subconjunto de gen_<tag> a evaluar")
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    key = get_api_key()
    dataset = Path(args.dataset)
    OUT.mkdir(parents=True, exist_ok=True)
    model_dirs = args.models or sorted(
        d.name for d in dataset.iterdir() if d.is_dir() and d.name.startswith("gen_"))

    summary = {}
    for md in model_dirs:
        tag = md.removeprefix("gen_")
        pairs = load_pairs(dataset, md, args.n_per_model, args.seed)
        out_path = OUT / f"judgments_{tag}.jsonl"
        done = set()
        if out_path.exists():  # resume
            for l in open(out_path):
                if l.strip():
                    done.add(json.loads(l)["complaint_id"])
        rows, ms = [], []
        print(f"[{tag}] {len(pairs)} quejas ({len(done)} ya juzgadas)", flush=True)
        for cid, gt_problems, text in pairs:
            if cid in done:
                continue
            j = judge(gt_problems, text, key, args.temp)
            row = {"model": tag, "complaint_id": cid, "n_gt_problems": len(gt_problems),
                   "judgment": j, "metrics": metrics(j, len(gt_problems))}
            rows.append(row)
            with open(out_path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row["metrics"]["ok"]:
                ms.append(row["metrics"])
            time.sleep(1)
        if not ms:
            summary[tag] = {"error": "sin juicios válidos"}
            continue
        summary[tag] = {
            "n": len(ms),
            "completeness_rate": round(sum(m["completeness"] for m in ms) / len(ms), 4),
            "hallucination_rate": round(sum(m["hallucination"] for m in ms) / len(ms), 4),
            "urgency_accuracy": round(sum(m["urgency_exact"] for m in ms) / len(ms), 4),
            "risks_accuracy": round(sum(m["risks_ok"] for m in ms) / len(ms), 4),
            "naturalness_mean": round(
                sum(m["naturalness"] for m in ms if isinstance(m["naturalness"], (int, float)))
                / max(1, sum(1 for m in ms if isinstance(m["naturalness"], (int, float)))), 3),
        }
        print(f"[{tag}] -> {summary[tag]}", flush=True)

    out_s = OUT / "fidelity_summary.json"
    existing = json.load(open(out_s)) if out_s.exists() else {}
    existing.update(summary)
    json.dump(existing, open(out_s, "w"), ensure_ascii=False, indent=1)
    print(f"summary -> {out_s}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
