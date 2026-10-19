#!/usr/bin/env python3
"""Generate the 5K diverse dataset: 500 complaints × 10 selected models.

Uses the production generator pipeline (miurba-generator/cli.py generate) per
model, with --resume so it can be restarted. Writes to eval/results/dataset-5k/.
"""
import subprocess, sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "miurba-generator"
CONFIG = ROOT / "experiments" / "extraccion_llm" / "config.eval.yaml"
OUTDIR = ROOT / "eval" / "results" / "dataset-5k"
N = 500
WORKERS = 8

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

OUTDIR.mkdir(parents=True, exist_ok=True)
log = open(OUTDIR / "generation_5k.log", "a")

for i, genkey in enumerate(selected, 1):
    modeldir = OUTDIR / genkey
    n_done = 0
    cf = modeldir / "raw" / "complaints.jsonl"
    if cf.exists():
        n_done = sum(1 for _ in open(cf))
    line = f"[{i:2d}/{len(selected)}] {genkey} ({n_done}/{N} done)"
    print(line, flush=True); log.write(line+"\n"); log.flush()
    if n_done >= N:
        line = f"  SKIP (already complete)"
        print(line, flush=True); log.write(line+"\n"); log.flush(); continue
    t0 = time.time()
    # Call the generator CLI
    cmd = [
        str(GEN / ".venv" / "bin" / "python"), str(GEN / "cli.py"), "generate",
        "-n", str(N), "--model", genkey, "--config", str(CONFIG),
        "--output-dir", str(modeldir), "--seed", "42", "--workers", str(WORKERS),
        "--timeout", "300",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=86400, cwd=str(GEN))
        dt = time.time() - t0
        n_after = sum(1 for _ in open(cf)) if cf.exists() else 0
        line = f"  DONE {n_after}/{N} in {dt:.0f}s"
        if r.returncode != 0:
            line += f" (rc={r.returncode})"
            log.write(r.stderr[-500:]+"\n"); log.flush()
        print(line, flush=True); log.write(line+"\n"); log.flush()
    except subprocess.TimeoutExpired:
        line = f"  TIMEOUT after {time.time()-t0:.0f}s"
        print(line, flush=True); log.write(line+"\n"); log.flush()
    time.sleep(20)

log.close()
print("=== 5K GENERATION COMPLETE ===", flush=True)