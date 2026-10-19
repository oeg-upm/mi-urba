#!/usr/bin/env python3
"""Generate the 10K dataset: 1000 complaints × 10 selected models.

Model selection: eval/results/diversity/selected_10_10k.json
(greedy max-min over 7-judge fused embedding distance, sanity filter,
>=7B, one per lineage — Sep 2026).

Uses the production generator pipeline (miurba-generator/cli.py generate)
with --resume so it can be restarted. Writes to eval/results/dataset-10k/.
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
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "miurba-generator"
CONFIG = ROOT / "experiments" / "extraccion_llm" / "config.eval.yaml"
OUTDIR = ROOT / "eval" / "results" / "dataset-10k"
N = 1000
WORKERS = 1

# Order updated 10:07: only models with exploration ETA < 2h per 1000
# (agentica 10.15h and muse-glimmer 7.43h excluded; see
# eval/results/dataset-10k_exploration.json). Total: ~12.6h.
selected = [
    "gen_devstral-small-2507",
    "gen_qwen3.5-122b-a10b",
]

OUTDIR.mkdir(parents=True, exist_ok=True)
log = open(OUTDIR / "generation_10k.log", "a")


def logline(s: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {s}"
    print(line, flush=True); log.write(line + "\n"); log.flush()


def warmup(model_id: str, max_wait: int = 360) -> bool:
    """Wait until the model responds (distinguish server-down from loading)."""
    import requests
    url = "http://192.168.10.1:4000/v1/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {get_api_key()}"}
    payload = {
        "model": f"local_llama/{model_id}",
        "messages": [{"role": "user", "content": "Di OK"}],
        "max_tokens": 5,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    server_down = 0
    while time.time() - t0 < max_wait:
        try:
            r = requests.post(url, json=payload, timeout=90, headers=headers)
            if r.status_code == 200 and "choices" in r.json():
                logline(f"  warmup OK ({model_id}) in {time.time()-t0:.0f}s")
                return True
            # connected but error JSON → model loading
            server_down = 0
            logline(f"  warmup probe: HTTP {r.status_code} ({model_id})")
        except requests.RequestException:
            server_down += 1
            if server_down >= 5:
                logline(f"  warmup FAIL: gateway unreachable ({model_id})")
                return False
        time.sleep(15)
    logline(f"  warmup TIMEOUT after {max_wait}s ({model_id})")
    return False


def model_id_of(genkey: str) -> str:
    """Strip the gen_ prefix: gen_foo -> foo."""
    return genkey[len("gen_"):]


# Pass-loop: repeat the selected list until every model reaches N
def _count(cf: Path) -> int:
    return sum(1 for _ in open(cf)) if cf.exists() else 0

for _pass in range(1, 21):  # up to 5 passes, safety
    incomplete = [gk for gk in selected if _count(OUTDIR / gk / "raw" / "complaints.jsonl") < N]
    if not incomplete:
        logline(f"ALL MODELS COMPLETE (pass {_pass})")
        break
    logline(f"--- pass {_pass}: {len(incomplete)} incomplete: {incomplete}")
    for i, genkey in enumerate(selected, 1):
        modeldir = OUTDIR / genkey
        cf = modeldir / "raw" / "complaints.jsonl"
        n_done = _count(cf)
        logline(f"[p{_pass} {i:2d}/{len(selected)}] {genkey} ({n_done}/{N} done)")
        if n_done >= N:
            logline("  SKIP (already complete)")
            continue
        if not warmup(model_id_of(genkey)):
            logline("  ABORTED for this model (warmup failed)")
            continue
        t0 = time.time()
        cmd = [
            "docker", "run", "--rm", "--network", "host",
            "-v", f"{CONFIG}:/app/config.eval.yaml:ro",
            "-v", f"{modeldir}:/data",
            "miurba-generator", "generate",
            "-n", str(N), "--model", genkey, "--config", "/app/config.eval.yaml",
            "--output-dir", "/data", "--seed", str(42 + _pass), "--workers", str(WORKERS),
            "--timeout", "300",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
            dt = time.time() - t0
            n_after = _count(cf)
            line = f"  DONE {n_after}/{N} in {dt:.0f}s"
            if r.returncode != 0:
                line += f" (rc={r.returncode})"
                log.write(r.stderr[-500:] + "\n"); log.flush()
            logline(line)
        except subprocess.TimeoutExpired:
            logline(f"  TIMEOUT after {time.time() - t0:.0f}s")

log.close()
print("=== 10K GENERATION COMPLETE ===", flush=True)