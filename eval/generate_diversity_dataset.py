#!/usr/bin/env python3
"""Generate the SAME N scenarios with every OK model, to measure diversity.

1. Presample N=20 scenarios (problems + community + persona + modifier) with a
   fixed seed → identical inputs for all models.
2. For each of the 88 OK models: load, generate the 20 complaints (sequential),
   save to eval/results/diversity/gen_<model>/raw/complaints.jsonl
3. Strictly sequential per model (one chat at a time), 30s cooldown between
   models to let llama-swap free the previous model.

Uses the EXACT same generation prompt as the production generator (iter 6), so
the diversity we measure is the real production diversity.
"""
import json, logging, os, random, sys, time, re, gc
from pathlib import Path

# Add generator to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "miurba-generator"))
from src import config as cfg
from src.llm import LLMClient
from src.prompts import build_generation_prompt
from src.sampling import (sample_scenario, sample_community, sample_persona,
                          sample_modifier, sample_photo_count, should_be_multi_problem,
                          should_have_photos)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ENDPOINT = "http://192.168.10.1:4000"
from env_utils import get_api_key
KEY = get_api_key()
N = 20
SEED = 42
OUTROOT = Path(__file__).resolve().parent.parent / "eval" / "results" / "diversity"

# OK models (from the test results). Loaded from gen_all_models.log + retests.
def load_ok_models():
    """Load FAST models (load<5min, warm<30s) from fast_models.json."""
    import json as _json
    base = Path(__file__).resolve().parent.parent
    p = base / "eval" / "results" / "fast_models.json"
    if p.exists():
        data = _json.load(open(p))
        return sorted(m["model"] for m in data)
    # fallback: all OK from logs
    tested = {}
    rank = {"OK  ":0, "WEAK":1, "FAIL":2, "ERR  ":3}
    logs = ["eval/results/gen_all_models.log","eval/results/gen_retest_problems.log","eval/results/gen_retest_isolated.log","eval/results/gen_retest_final.log"]
    for logf in logs:
        pp = base / logf
        if not pp.exists(): continue
        for line in open(pp):
            m = re.match(r'(?:\[\s*\d+/\d+\]\s+)?(OK  |WEAK|FAIL|ERR  )\s+(\S+)\s+', line)
            if not m: continue
            mark, model = m.groups(); full = "local_llama/" + model
            if full not in tested or rank[mark] < tested[full]: tested[full] = rank[mark]
    return sorted(m for m, r in tested.items() if r == 0)

# 1. Presample N identical scenarios
def presample_scenarios(n, seed):
    random.seed(seed)
    scenarios = []
    for i in range(n):
        community = sample_community()
        persona = sample_persona()
        modifier = sample_modifier()
        if should_be_multi_problem():
            n_problems = random.choices([2, 3], weights=[0.75, 0.25], k=1)[0]
        else:
            n_problems = 1
        problems = [sample_scenario() for _ in range(n_problems)]
        has_photos = should_have_photos()
        num_pics = sample_photo_count(has_photos)
        scenarios.append({
            "record_id": f"Q{i+1:04d}",
            "problems": problems,
            "community": community,
            "persona": persona,
            "modifier": modifier,
            "num_pics": num_pics,
        })
    return scenarios

def generate_for_model(model_id, scenarios, outdir):
    """Generate the N scenarios with one model. Sequential (one chat at a time)."""
    client = LLMClient(
        base_url=ENDPOINT,
        model=model_id,
        temperature=0.85,
        timeout=300,  # 5min per request; backoff handles server crashes
        api_key=KEY,
        extra_params={"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}},
    )
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "complaints.jsonl"
    # Resume: skip already done
    done = set()
    if outpath.exists():
        for line in open(outpath):
            if line.strip():
                done.add(json.loads(line).get("id", ""))
    with open(outpath, "a", encoding="utf-8") as f:
        for sc in scenarios:
            if sc["record_id"] in done:
                continue
            messages = build_generation_prompt(
                record_id=sc["record_id"],
                problems=sc["problems"],
                community=sc["community"],
                num_pics=sc["num_pics"],
                persona=sc["persona"],
                modifier=sc["modifier"],
            )
            try:
                response = client.chat(messages)
                content = client.extract_content(response)
                if not content:
                    logger.warning(f"[{model_id}/{sc['record_id']}] empty content")
                    continue
                # try parse JSON; some models append special tokens after the JSON
                parsed = None
                try:
                    parsed = json.loads(content.strip())
                except json.JSONDecodeError:
                    c = content
                    if c.startswith("```"):
                        c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
                    try:
                        parsed = json.loads(c.strip())
                    except json.JSONDecodeError:
                        # extract embedded JSON object (handles trailing special tokens like <|END_OF_TURN_TOKEN|>)
                        jm = re.search(r'\{.*\}', c, re.DOTALL)
                        if jm:
                            try: parsed = json.loads(jm.group(0))
                            except: parsed = None
                if parsed is None:
                    parsed = {"user_text": content, "timestamp": ""}
                record = {"id": sc["record_id"], "user_text": parsed.get("user_text", content),
                          "timestamp": parsed.get("timestamp", ""), "user_pics": [],
                          "channel": "app", "_model": model_id}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                logger.warning(f"[{model_id}/{sc['record_id']}] failed: {e}")
                continue

def main():
    cfg_path = Path(__file__).resolve().parent.parent / "eval" / "config.eval.yaml"
    cfg.get_config(str(cfg_path))  # load taxonomy for sampling
    models = load_ok_models()
    print(f"=== Diversity generation: {len(models)} models × {N} scenarios ===", flush=True)
    scenarios = presample_scenarios(N, SEED)
    print(f"Presampled {len(scenarios)} scenarios (seed={SEED})", flush=True)
    # Save scenarios for reproducibility
    OUTROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTROOT / "scenarios.json", "w") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)
    with open(OUTROOT / "models.json", "w") as f:
        json.dump(models, f, indent=2)
    print(f"Scenarios + model list saved to {OUTROOT}", flush=True)

    logf = open(OUTROOT / "generation.log", "a")
    for i, model in enumerate(models, 1):
        tag = model.replace("local_llama/", "")
        outdir = OUTROOT / f"gen_{tag}"
        # skip if already complete
        if (outdir / "complaints.jsonl").exists():
            n_done = sum(1 for _ in open(outdir / "complaints.jsonl"))
            if n_done >= N:
                line = f"[{i:3d}/{len(models)}] SKIP {tag} ({n_done} done)"
                print(line, flush=True); logf.write(line+"\n"); logf.flush(); continue
        t0 = time.time()
        line = f"[{i:3d}/{len(models)}] GEN  {tag}"
        print(line, flush=True); logf.write(line+"\n"); logf.flush()
        try:
            generate_for_model(model, scenarios, outdir)
        except Exception as e:
            # One model failing must NOT kill the whole run. Log and continue.
            line = f"[{i:3d}/{len(models)}] ERROR {tag}: {e}"
            print(line, flush=True); logf.write(line+"\n"); logf.flush()
            continue
        dt = time.time() - t0
        n_done = sum(1 for _ in open(outdir / "complaints.jsonl")) if (outdir / "complaints.jsonl").exists() else 0
        line = f"[{i:3d}/{len(models)}] DONE {tag} ({n_done}/{N}, {dt:.0f}s)"
        print(line, flush=True); logf.write(line+"\n"); logf.flush()
        gc.collect()
        time.sleep(20)  # cooldown for llama-swap
    logf.close()
    print("=== DIVERSITY GENERATION COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()