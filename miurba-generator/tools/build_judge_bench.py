#!/usr/bin/env python3
"""Build a frozen, blind judge benchmark from already-generated records.

Input: the raw outputs of previous campaigns (conversations.jsonl holds every
candidate text the judge saw, with the scenario that was requested).

Output (under $MIURBA_EVAL_DIR/judge_bench/, default `eval/judge_bench/`):
  cases.jsonl      one case per line: {case_id, model, scenario, text}. NOTHING
                   about the judge verdict is included: annotators must be blind.
  labels_hidden.jsonl  the judge's own verdict and the campaign it came from, keyed
                   by case_id, for later analysis only (never given to annotators).
  batches/batch_NN.jsonl  ready-to-annotate chunks.

Stratification: `--per-model` cases per generator, split evenly between texts the
judge accepted and texts it rejected, so both error rates can be measured.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = Path(os.environ.get("MIURBA_EVAL_DIR", ROOT / "eval"))
DEFAULT_RUNS = ["smoke-5x100", "smoke-5x100-t025", "smoke-5x100-t050"]
MODELS = ["gemma-3-12b-awq", "olmo-3.1-32b-awq", "qwen3.6-27b-awq",
          "muse-glimmer-30b-awq", "gpt-oss-20b"]


def load_turns(runs):
    """Collect (model, scenario, text, accepted, campaign) for every judge input."""
    pool = []
    for run in runs:
        for model in MODELS:
            path = EVAL / "results" / run / model / "raw/conversations.jsonl"
            if not path.exists():
                continue
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                for turn in rec["turns"]:
                    text = (turn.get("user_text") or "").strip()
                    if not text:
                        continue
                    pool.append({
                        "model": model,
                        "scenario": rec["scenario"],
                        "text": text,
                        "accepted": turn.get("failure") is None,
                        "campaign": run,
                        "record_id": rec["id"],
                        "attempt": turn.get("attempt"),
                    })
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--per-model", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    pool = load_turns(args.runs)
    by_model = defaultdict(lambda: {"ok": [], "bad": []})
    seen = set()
    for case in pool:
        key = (case["model"], case["record_id"], case["attempt"], case["text"][:80])
        if key in seen:
            continue
        seen.add(key)
        by_model[case["model"]]["ok" if case["accepted"] else "bad"].append(case)

    half = args.per_model // 2
    cases, hidden = [], []
    for model in MODELS:
        for bucket in ("ok", "bad"):
            items = by_model[model][bucket]
            rnd.shuffle(items)
            for case in items[:half]:
                cases.append(case)
    rnd.shuffle(cases)

    out_dir = EVAL / "judge_bench"
    (out_dir / "batches").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cases.jsonl", "w", encoding="utf-8") as fc, \
         open(out_dir / "labels_hidden.jsonl", "w", encoding="utf-8") as fh:
        for i, case in enumerate(cases, 1):
            case_id = f"C{i:04d}"
            fc.write(json.dumps({"case_id": case_id, "model": case["model"],
                                 "scenario": case["scenario"], "text": case["text"]},
                                ensure_ascii=False) + "\n")
            fh.write(json.dumps({"case_id": case_id, "judge_accepted": case["accepted"],
                                 "campaign": case["campaign"], "model": case["model"],
                                 "record_id": case["record_id"],
                                 "attempt": case["attempt"]}, ensure_ascii=False) + "\n")

    batch_dir = out_dir / "batches"
    for old in batch_dir.glob("batch_*.jsonl"):
        old.unlink()
    chunks = [cases[i:i + args.batch_size] for i in range(0, len(cases), args.batch_size)]
    for n, chunk in enumerate(chunks, 1):
        with open(batch_dir / f"batch_{n:02d}.jsonl", "w", encoding="utf-8") as f:
            for offset, case in enumerate(chunk):
                case_id = f"C{cases.index(case) + 1:04d}"
                f.write(json.dumps({"case_id": case_id, "model": case["model"],
                                    "scenario": case["scenario"], "text": case["text"]},
                                   ensure_ascii=False) + "\n")

    print(f"casos: {len(cases)} | lotes: {len(chunks)} de {args.batch_size}")
    for model in MODELS:
        ok = sum(1 for c in cases if c["model"] == model and c["accepted"])
        bad = sum(1 for c in cases if c["model"] == model and not c["accepted"])
        print(f"  {model:22s} aceptados por el juez {ok:3d} | rechazados {bad:3d}")


if __name__ == "__main__":
    main()
