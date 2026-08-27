#!/usr/bin/env python3
"""Extract + evaluate the 5K dataset with gemma4-12b (constrained decoding).

For each of the 10 generators:
1. Extract all its complaints with extract_gemma4_12b (constrained decoding)
2. Evaluate extractions vs ground_truth → metrics

Outputs per-generator metrics + a consolidated report.
"""
import subprocess, sys, os, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "miurba-extractor"
GEN = ROOT / "miurba-generator"
CONFIG = ROOT / "eval" / "config.eval.yaml"
DATASET = ROOT / "eval" / "results" / "dataset-5k"
EVAL = ROOT / "eval"

selected = [
    "gen_deepseek-r1-distill-qwen-7b",
    "gen_phi-4",
    "gen_qwen3-0.6b",
    "gen_pantheon-rp-8b-llama3",
    "gen_mistral-7b-v0.3-base",
    "gen_deepseek-r1-distill-llama-8b",
    "gen_granite-3.3-2b",
    "gen_olmo-2-13b-instruct",
    "gen_falcon3-7b-instruct",
    "gen_glm-4.7-flash",
]

DATASET.mkdir(parents=True, exist_ok=True)
log = open(DATASET / "extract_eval_5k.log", "a")

# Phase 1: extract each generator's complaints
for i, genkey in enumerate(selected, 1):
    gen_dir = DATASET / genkey
    raw = gen_dir / "raw"
    ext_dir = gen_dir / "ext_extract_gemma4_12b" / "annotated"
    ext_file = ext_dir / "extractions.jsonl"
    n_comp = sum(1 for _ in open(raw / "complaints.jsonl")) if (raw / "complaints.jsonl").exists() else 0
    n_ext = sum(1 for _ in open(ext_file)) if ext_file.exists() else 0
    line = f"[{i:2d}/{len(selected)}] EXTRACT {genkey} ({n_comp} complaints, {n_ext} extracted)"
    print(line, flush=True); log.write(line+"\n"); log.flush()
    if n_comp == 0:
        line = "  SKIP (no complaints generated yet)"
        print(line, flush=True); log.write(line+"\n"); log.flush(); continue
    if n_ext >= n_comp:
        line = "  SKIP (already extracted)"
        print(line, flush=True); log.write(line+"\n"); log.flush(); continue
    cmd = [
        str(EXT / ".venv" / "bin" / "python"), str(EXT / "cli.py"), "extract",
        "--config", str(CONFIG), "--model", "extract_gemma4_12b",
        "--raw-dir", str(raw), "--annotated-dir", str(ext_dir),
        "--timeout", "900", "--max-retries", "3", "--retry-delay", "5", "--workers", "1",
    ]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=99000, cwd=str(EXT))
        dt = time.time() - t0
        n_after = sum(1 for _ in open(ext_file)) if ext_file.exists() else 0
        line = f"  DONE {n_after}/{n_comp} in {dt:.0f}s"
        if r.returncode != 0:
            line += f" (rc={r.returncode})"
            log.write(r.stderr[-500:]+"\n"); log.flush()
        print(line, flush=True); log.write(line+"\n"); log.flush()
    except subprocess.TimeoutExpired:
        line = f"  TIMEOUT after {time.time()-t0:.0f}s"
        print(line, flush=True); log.write(line+"\n"); log.flush()
    time.sleep(20)

# Phase 2: evaluate each
print("\n=== Phase 2: Evaluation ===", flush=True)
sys.path.insert(0, str(EVAL))
from evaluate import evaluate

results = []
for genkey in selected:
    gen_dir = DATASET / genkey
    gt_file = gen_dir / "raw" / "ground_truth.jsonl"
    ext_file = gen_dir / "ext_extract_gemma4_12b" / "annotated" / "extractions.jsonl"
    if not gt_file.exists() or not ext_file.exists():
        print(f"  SKIP {genkey} (missing gt or extractions)", flush=True); continue
    name = genkey
    r = evaluate(str(gt_file), str(ext_file), name)
    results.append(r)

# Save metrics + report
with open(DATASET / "metrics.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n=== 5K EVAL RESULTS ===", flush=True)
print(f"{'Generator':<40}{'F1':>6}{'P':>6}{'R':>6}{'EM':>6}{'Urg':>6}{'Sub':>6}{'Rsk':>6}{'FP':>5}", flush=True)
print("-"*92, flush=True)
for r in sorted(results, key=lambda x: -x['complaint_metrics']['f1']):
    cm = r['complaint_metrics']; fm = r['field_metrics']; mt = r['match_types']
    print(f"{r['model']:<40}{cm['f1']:>6.3f}{cm['precision']:>6.3f}{cm['recall']:>6.3f}{cm['exact_match_rate']:>6.3f}{fm['urgency']['accuracy']:>6.3f}{fm['category_subcategory']['accuracy']:>6.3f}{fm['risks_jaccard']['accuracy']:>6.3f}{mt.get('fp',0):>5d}", flush=True)

log.close()
print("=== 5K EXTRACT+EVAL COMPLETE ===", flush=True)