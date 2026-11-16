#!/usr/bin/env python3
"""Evalúa un prompt/modelo de juez contra el conjunto congelado de casos.

Uso típico (una vuelta de iteración cuesta minutos, no una campaña):

    python tools/judge_bench.py --tag v5-baseline --model qwen3.5-9b-awq
    # editar prompts/validate_record.md y volver a lanzar con otro --tag

Requiere que `tools/build_judge_bench.py` haya creado `eval/judge_bench/` y que
los anotadores hayan dejado sus etiquetas en `eval/judge_bench/labels/`.
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

ROOT = Path("/backup/pc1/data/miUrba2/mi-urba")
BENCH = ROOT / "eval/judge_bench"
sys.path.insert(0, str(ROOT / "miurba-generator"))

from src.llm import LLMClient              # noqa: E402
from src.pipeline import validate_complaint  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_gold(annotators: list[str]) -> tuple[dict, dict]:
    """Devuelve (gold, agreement): gold[case_id] = True/False, solo acuerdos.

    Si los anotadores no coinciden, el caso queda fuera (y se cuenta aparte).
    """
    per_annotator = {}
    for name in annotators:
        labels = {}
        for f in sorted((BENCH / "labels").glob(f"{name}_batch_*.jsonl")):
            for rec in load_jsonl(f):
                labels[rec["case_id"]] = bool(rec.get("apto"))
        per_annotator[name] = labels

    common = set.intersection(*(set(v) for v in per_annotator.values())) if per_annotator else set()
    gold, agreement = {}, {"n_common": len(common), "agree": 0, "disagree": 0}
    for case_id in common:
        values = {per_annotator[n][case_id] for n in per_annotator}
        if len(values) == 1:
            gold[case_id] = values.pop()
            agreement["agree"] += 1
        else:
            agreement["disagree"] += 1
    return gold, agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="nombre de esta vuelta (para el historial)")
    ap.add_argument("--model", default="qwen3.5-9b-awq")
    ap.add_argument("--base-url", default="http://192.168.10.100:9292")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--annotators", nargs="*", default=["deepseek", "glm"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump-verdicts", action="store_true",
                    help="guardar el veredicto de cada caso (para combinar jueces)")
    args = ap.parse_args()

    cases = load_jsonl(BENCH / "cases.jsonl")
    gold, agreement = load_gold(args.annotators)
    cases = [c for c in cases if c["case_id"] in gold]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("no hay casos con etiqueta consensuada todavía")

    client = LLMClient(base_url=args.base_url, model=args.model, temperature=args.temperature,
                       timeout=300, max_retries=3, retry_delay=5.0, model_load_wait=180,
                       extra_params={"max_tokens": args.max_tokens, "enable_thinking": False,
                                     "chat_template_kwargs": {"enable_thinking": False}})

    results, lock = [], threading.Lock()

    def run(case):
        try:
            valid, issues, degraded, raw = validate_complaint(
                client, case["text"], case["scenario"])
            verdict = bool(valid and not degraded)
            kinds = [d.get("kind") for d in (raw or {}).get("_diagnostics", [])]
        except Exception as e:  # noqa: BLE001
            verdict, issues, kinds = False, [f"crash:{type(e).__name__}"], ["crash"]
        with lock:
            results.append({"case_id": case["case_id"], "model": case["model"],
                            "gold": gold[case["case_id"]], "judge": verdict,
                            "issues": list(issues)[:5], "kinds": kinds})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, cases))

    tp = sum(1 for r in results if not r["gold"] and not r["judge"])   # defecto detectado
    fn = sum(1 for r in results if not r["gold"] and r["judge"])       # defecto colado
    tn = sum(1 for r in results if r["gold"] and r["judge"])           # apto aceptado
    fp = sum(1 for r in results if r["gold"] and not r["judge"])       # apto rechazado
    n = len(results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    summary = {
        "tag": args.tag, "model": args.model, "temperature": args.temperature,
        "n": n, "agreement": agreement,
        "accuracy": round((tp + tn) / n, 4),
        "defect_precision": round(precision, 4),
        "defect_recall": round(recall, 4),
        "defect_f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "leaked_defects": fn, "false_rejects": fp,
        "judge_failure_kinds": dict(Counter(k for r in results for k in r["kinds"] if k)),
        "by_model": {},
        "disagreements": [r for r in results if r["gold"] != r["judge"]][:60],
    }
    for model in sorted({r["model"] for r in results}):
        sub = [r for r in results if r["model"] == model]
        ok = sum(1 for r in sub if r["gold"] == r["judge"])
        summary["by_model"][model] = {"n": len(sub), "accuracy": round(ok / len(sub), 4),
                                      "leaked": sum(1 for r in sub if not r["gold"] and r["judge"]),
                                      "false_rejects": sum(1 for r in sub if r["gold"] and not r["judge"])}

    if args.dump_verdicts:
        verdicts_path = BENCH / "results" / f"{args.tag}.verdicts.jsonl"
        with open(verdicts_path, "w", encoding="utf-8") as f:
            for r in sorted(results, key=lambda x: x["case_id"]):
                f.write(json.dumps({"case_id": r["case_id"], "model": r["model"],
                                    "gold": r["gold"], "judge": r["judge"]},
                                   ensure_ascii=False) + "\n")

    out = BENCH / "results" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n== {args.tag} ({args.model}, T={args.temperature}) sobre {n} casos "
          f"(acuerdo entre anotadores: {agreement['agree']}/{agreement['n_common']})")
    print(f"   exactitud {summary['accuracy']:.1%} | defectos: P {precision:.3f} R {recall:.3f} "
          f"F1 {summary['defect_f1']:.3f}")
    print(f"   defectos colados {fn} | aptos rechazados {fp}")
    print(f"   fallos del juez: {summary['judge_failure_kinds']}")
    for model, s in summary["by_model"].items():
        print(f"   {model:22s} n={s['n']:3d} acc {s['accuracy']:.1%} colados {s['leaked']:2d} "
              f"rechazados {s['false_rejects']:2d}")
    print(f"\n   -> {out}")


if __name__ == "__main__":
    main()
