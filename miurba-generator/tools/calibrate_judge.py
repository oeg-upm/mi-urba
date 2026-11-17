#!/usr/bin/env python3
"""Calibración del juez de validación contra etiquetas previas del corpus.

Usa el prompt REAL del generador (`_build_validation_prompt`) y el parser REAL
(`validate_complaint`), de forma que lo que se mide aquí es exactamente lo que
filtrará la campaña.

Etiquetas de referencia: severidad del gold1k (S0 = apta, S2/S3 = defectuosa).

Uso:
  python tools/calibrate_judge.py --n-per-class 20 \
      --judge mistral-small-3.2-24b --judge qwen3.8-27b-vllm
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = Path(os.environ.get("MIURBA_EVAL_DIR", ROOT / "eval"))
CORPUS = EVAL / "results/dataset-50k"
SEVERITY = Path(os.environ.get("MIURBA_SEVERITY_FILE", EVAL / "gold1k_results.jsonl"))
MODELS = ["gemma-3-12b-awq", "gpt-oss-20b", "olmo-3.1-32b-awq",
          "qwen3.6-27b-awq", "muse-glimmer-30b-awq"]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import LLMClient            # noqa: E402
from src.pipeline import validate_complaint  # noqa: E402


def gt_to_scenario(extractions):
    """GT (formato extractor) -> escenario (formato generador)."""
    out = []
    for e in extractions:
        cat = e.get("category") or {}
        loc = e.get("location") or {}
        out.append({
            "category": cat.get("principal"),
            "subcategory": cat.get("subcategory"),
            "location_zone": loc.get("zone"),
            "location_detail": loc.get("detail") or "",
            "urgency": e.get("urgency"),
            "affected_elements": e.get("affected_elements") or [],
            "risks": e.get("risks") or ["ninguno"],
        })
    return out


def load_data():
    texts, gts = {}, {}
    for model in MODELS:
        for line in open(CORPUS / model / "raw/complaints.jsonl"):
            r = json.loads(line)
            texts[r["id"]] = (model, r["user_text"])
        for line in open(CORPUS / model / "raw/ground_truth.jsonl"):
            r = json.loads(line)
            gts[r["complaint_id"]] = r["extractions"]
    sev = {}
    for line in open(SEVERITY):
        r = json.loads(line)
        if r.get("error") is None and r.get("severity"):
            sev[r["id"]] = r["severity"]
    return texts, gts, sev


def pick(sev, n_per_class, seed=7):
    rnd = random.Random(seed)
    good = defaultdict(list)
    bad = defaultdict(list)
    for cid, s in sev.items():
        model = cid and None
        if s in ("S0", "S1"):
            good[cid] = s
        elif s in ("S2", "S3"):
            bad[cid] = s
    # estratificado por modelo usando el rango de ids
    def by_model(ids):
        buckets = defaultdict(list)
        for cid in ids:
            buckets[(int(cid[1:]) - 1) // 10000].append(cid)
        return buckets
    picked = []
    for label, pool in (("apta", good), ("defectuosa", bad)):
        buckets = by_model(pool)
        per_bucket = max(1, n_per_class // len(MODELS))
        for b in sorted(buckets):
            picked += [(cid, label, pool[cid]) for cid in rnd.sample(buckets[b], min(per_bucket, len(buckets[b])))]
    rnd.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--judge", action="append", default=[])
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    ap.add_argument("--out", default=str(EVAL / "judge_calibration.json"))
    args = ap.parse_args()
    judges = args.judge or ["mistral-small-3.2-24b"]

    texts, gts, sev = load_data()
    sample = pick(sev, args.n_per_class)
    print(f"muestra: {len(sample)} registros "
          f"({sum(1 for _, l, _ in sample if l == 'apta')} aptas / "
          f"{sum(1 for _, l, _ in sample if l == 'defectuosa')} defectuosas)")

    results = {"sample": [], "judges": {}}
    for cid, label, severity in sample:
        model, text = texts[cid]
        results["sample"].append({"id": cid, "model": model, "label": label,
                                  "severity": severity})

    for judge_name in judges:
        client = LLMClient(base_url=args.base_url, model=judge_name, temperature=0.0,
                           timeout=300, max_retries=4, retry_delay=5.0,
                           model_load_wait=180)
        conf = Counter()
        rows = []
        for cid, label, severity in sample:
            model, text = texts[cid]
            scenario = gt_to_scenario(gts[cid])
            try:
                is_valid, issues, degraded, _raw = validate_complaint(client, text, scenario)
            except Exception as e:  # a judge crash must not kill the calibration
                is_valid, issues, degraded = False, [f"crash:{type(e).__name__}"], True
            verdict = "apta" if (is_valid and not degraded) else "defectuosa"
            conf[(label, verdict)] += 1
            rows.append({"id": cid, "model": model, "label": label, "severity": severity,
                         "verdict": verdict, "issues": issues[:6], "degraded": degraded})
            print(f"  [{judge_name}] {cid} {severity} -> {verdict} {issues[:2]}", flush=True)
        tp = conf[("apta", "apta")]
        fn = conf[("apta", "defectuosa")]
        tn = conf[("defectuosa", "defectuosa")]
        fp = conf[("defectuosa", "apta")]
        n = max(sum(conf.values()), 1)
        results["judges"][judge_name] = {
            "confusion": {"apta_aceptada": tp, "apta_rechazada": fn,
                          "defectuosa_rechazada": tn, "defectuosa_aceptada": fp},
            "false_accept_rate": round(fp / max(fp + tn, 1), 3),
            "false_reject_rate": round(fn / max(fn + tp, 1), 3),
            "accuracy": round((tp + tn) / n, 3),
            "rows": rows,
        }
        print(f"\n== {judge_name}: exactitud {(tp + tn) / n:.1%} · "
              f"falsos aptos {fp}/{fp + tn} · falsos rechazos {fn}/{fn + tp}\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"guardado -> {args.out}")


if __name__ == "__main__":
    main()
