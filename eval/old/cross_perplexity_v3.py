#!/usr/bin/env python3
"""Cross-perplexity experiment with temp=0 generation (all local, fixed diagonal).

Flow per scenario:
  1. For each generator model G:
     - /apply-template to get the formatted prompt (with chat template, enable_thinking=false)
     - /completion with temp=0 (greedy) to generate text + capture gen_tokens
     - Record: prompt_fmt (formatted), content (generated text), gen_tokens
  2. For each evaluator model E:
     - Tokenize full_text = prompt_fmt + content with E's tokenizer
     - Compute logprob of EVERY token (index 1+, BOS is null)
     - Perplexity = exp(-mean(logprobs))

With temp=0, G's generated tokens are deterministic. When E==G, the tokens
E tokenizes match what G generated → logprobs ≈ 0 → perplexity ≈ 1.

All models run via local llama.cpp server (one at a time, with GPU offload).

Output: eval/results/diversity/cross_perplexity_v3.json
"""
import argparse, json, math, os, sys, time, logging, subprocess
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]  # archived under eval/old/
sys.path.insert(0, str(ROOT / "miurba-generator"))
from src.prompts import build_generation_prompt

DIVERSITY_DIR = ROOT / "eval" / "results" / "diversity"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_PROBS = 1000
FLOOR_LOGPROB = -20.0


def load_scenarios():
    return json.load(open(DIVERSITY_DIR / "scenarios.json"))


def reconstruct_messages(scenario):
    return build_generation_prompt(
        record_id=scenario["record_id"],
        problems=scenario["problems"],
        community=scenario["community"],
        num_pics=scenario["num_pics"],
        persona=scenario.get("persona"),
        modifier=scenario.get("modifier"),
    )


def start_local_server(server_bin, gguf_path, port, ctx=4096, threads=4):
    log_file = f"/tmp/xppl_server_{port}.log"
    srv = subprocess.Popen(
        [server_bin, "-m", gguf_path, "--port", str(port),
         "--host", "127.0.0.1", "-c", str(ctx), "-t", str(threads), "-ngl", "99"],
        stdout=open(log_file, "w"), stderr=subprocess.STDOUT,
    )
    endpoint = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            if requests.get(f"{endpoint}/health", timeout=3).json().get("status") == "ok":
                return srv, endpoint
        except:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server not ready at {endpoint}")


def stop_local_server(srv):
    srv.kill()
    srv.wait()
    time.sleep(3)


def parse_user_text(content):
    """Parse JSON from model output to extract user_text."""
    import re
    if not content:
        return None
    try:
        parsed = json.loads(content.strip())
        return parsed.get("user_text", "") if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    c = content
    if c.startswith("```"):
        c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
    try:
        parsed = json.loads(c.strip())
        return parsed.get("user_text", "") if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^{}]*"user_text"[^{}]*\}', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()).get("user_text", "")
        except:
            pass
    return None


def generate_with_local_server(endpoint, scenario, timeout=300):
    """Generate complaint text with local server, temp=0.

    Returns (prompt_fmt, content, gen_tokens).
    """
    msgs = reconstruct_messages(scenario)
    # 1. Apply chat template (with enable_thinking=false for thinking models)
    r = requests.post(f"{endpoint}/apply-template", json={
        "messages": msgs,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=30)
    r.raise_for_status()
    prompt_fmt = r.json()["prompt"]
    # 2. Generate with /completion (greedy), capture tokens
    r2 = requests.post(f"{endpoint}/completion", json={
        "prompt": prompt_fmt,
        "n_predict": 10000,
        "temperature": 0,
        "top_k": 1,
        "return_tokens": True,
        "n_probs": 0,
        "cache_prompt": False,
    }, timeout=timeout)
    r2.raise_for_status()
    d = r2.json()
    content = d.get("content", "")
    gen_tokens = d.get("tokens", [])
    user_text = parse_user_text(content)
    return prompt_fmt, content, gen_tokens, user_text


def tokenize(endpoint, text):
    r = requests.post(f"{endpoint}/tokenize", json={"content": text}, timeout=30)
    r.raise_for_status()
    return r.json()["tokens"]


def compute_logprobs_incremental(endpoint, tokens, timeout=120):
    """Compute logprob of every token from index 1 onwards."""
    logprobs = []
    misses = 0
    for k in range(1, len(tokens)):
        target = tokens[k]
        try:
            r = requests.post(f"{endpoint}/completion", json={
                "prompt": tokens[:k],
                "n_predict": 1,
                "n_probs": N_PROBS,
                "cache_prompt": False,
                "temperature": 0,
                "top_k": 1,
            }, timeout=timeout)
            r.raise_for_status()
            d = r.json()
            cp = d.get("completion_probabilities", [])
            if cp and cp[0].get("top_logprobs"):
                found = None
                for t in cp[0]["top_logprobs"]:
                    if t["id"] == target:
                        found = t["logprob"]
                        break
                if found is not None:
                    logprobs.append(found)
                else:
                    logprobs.append(FLOOR_LOGPROB)
                    misses += 1
            else:
                logprobs.append(FLOOR_LOGPROB)
                misses += 1
        except Exception as e:
            logger.warning(f"token {k} failed: {e}")
            logprobs.append(FLOOR_LOGPROB)
            misses += 1
    return logprobs, misses


def perplexity_from_logprobs(logprobs):
    if not logprobs:
        return float("inf"), 0.0
    mean_lp = sum(logprobs) / len(logprobs)
    ppl = math.exp(-mean_lp) if mean_lp > -50 else float("inf")
    return ppl, mean_lp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--n-scenarios", type=int, default=5)
    parser.add_argument("--server-port", type=int, default=18888)
    parser.add_argument("--server-bin", default="/home/dani/llama.cpp-xppl/build/bin/llama-server")
    parser.add_argument("--gguf-dir", default="/home/dani/models/gguf")
    parser.add_argument("--output", default=str(DIVERSITY_DIR / "cross_perplexity_v3.json"))
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    models = args.models
    scenarios = load_scenarios()[:args.n_scenarios]
    logger.info(f"Models: {models}")
    logger.info(f"Scenarios: {len(scenarios)}")

    gguf_map = {
        "qwen3-0.6b": f"{args.gguf_dir}/qwen3-0.6b/Qwen3-0.6B.Q4_K_M.gguf",
        "llama-3.2-1b-instruct": f"{args.gguf_dir}/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct.Q4_K_M.gguf",
        "gemma-2-2b-it": f"{args.gguf_dir}/gemma-2-2b-it/gemma-2-2b-it.Q4_K_M.gguf",
        "falcon3-1b-instruct": f"{args.gguf_dir}/falcon3-1b-instruct/Falcon3-1B-Instruct-Q4_K_M.gguf",
    }

    results = {}
    if os.path.exists(args.output):
        results = json.load(open(args.output))

    gen_cache_file = DIVERSITY_DIR / "xppl_gen_cache_v3.json"
    gen_cache = {}
    if gen_cache_file.exists():
        gen_cache = json.load(open(gen_cache_file))

    # ── Phase 1: Generate texts with each model (temp=0, local) ──
    logger.info("=== PHASE 1: Generate texts with temp=0 (local) ===")
    for gen_tag in models:
        gguf = gguf_map.get(gen_tag)
        if not gguf or not os.path.exists(gguf):
            logger.error(f"GGUF not found for {gen_tag}: {gguf}")
            continue
        if gen_tag not in gen_cache:
            gen_cache[gen_tag] = {}

        all_cached = all(sc["record_id"] in gen_cache[gen_tag] for sc in scenarios)
        if all_cached and not args.regenerate:
            logger.info(f"[GEN] {gen_tag} all cached, skip")
            continue

        logger.info(f"[GEN] Loading {gen_tag} ({gguf})...")
        srv, endpoint = start_local_server(args.server_bin, gguf, args.server_port)
        logger.info(f"  server ready")

        try:
            for sc in scenarios:
                rid = sc["record_id"]
                if rid in gen_cache[gen_tag] and not args.regenerate:
                    logger.info(f"[GEN] {gen_tag}/{rid} cached")
                    continue
                logger.info(f"[GEN] {gen_tag}/{rid} generating (temp=0)...")
                try:
                    prompt_fmt, content, gen_tokens, user_text = generate_with_local_server(endpoint, sc)
                    if content and gen_tokens:
                        gen_cache[gen_tag][rid] = {
                            "prompt_fmt": prompt_fmt,
                            "content": content,
                            "gen_tokens": gen_tokens,
                            "user_text": user_text or "",
                        }
                        logger.info(f"  generated {len(content)} chars, {len(gen_tokens)} tokens")
                        with open(gen_cache_file, "w") as f:
                            json.dump(gen_cache, f, indent=2, ensure_ascii=False)
                    else:
                        logger.warning(f"  empty result for {gen_tag}/{rid}")
                except Exception as e:
                    logger.error(f"  failed: {e}")
        finally:
            stop_local_server(srv)

    # ── Phase 2: For each evaluator, compute perplexity of all generators' sequences ──
    logger.info("=== PHASE 2: Compute cross-perplexity ===")
    for eval_tag in models:
        gguf = gguf_map.get(eval_tag)
        if not gguf or not os.path.exists(gguf):
            logger.error(f"GGUF not found for {eval_tag}: {gguf}")
            continue

        if eval_tag not in results:
            results[eval_tag] = {}

        logger.info(f"[EVAL] Loading {eval_tag} ({gguf})...")
        srv, endpoint = start_local_server(args.server_bin, gguf, args.server_port)
        logger.info(f"  server ready")

        try:
            for gen_tag in models:
                if gen_tag in results[eval_tag] and results[eval_tag][gen_tag].get("n_scenarios", 0) >= len(scenarios):
                    logger.info(f"  [{eval_tag}] {gen_tag} already done")
                    continue

                all_logprobs = []
                total_misses = 0
                total_tokens = 0
                for sc in scenarios:
                    rid = sc["record_id"]
                    cached = gen_cache.get(gen_tag, {}).get(rid)
                    if not cached:
                        logger.warning(f"  no cached text for {gen_tag}/{rid}")
                        continue

                    if eval_tag == gen_tag:
                        # Diagonal: use real gen_tokens (exact tokens the model generated)
                        # Tokenize prompt_fmt with the same model, then append gen_tokens
                        try:
                            prompt_tokens = tokenize(endpoint, cached["prompt_fmt"])
                        except Exception as e:
                            logger.warning(f"  tokenize prompt failed: {e}")
                            continue
                        tokens = prompt_tokens + cached["gen_tokens"]
                    else:
                        # Cross: retokenize full text with evaluator's tokenizer
                        full_text = cached["prompt_fmt"] + cached["content"]
                        try:
                            tokens = tokenize(endpoint, full_text)
                        except Exception as e:
                            logger.warning(f"  tokenize failed: {e}")
                            continue

                    t0 = time.time()
                    logprobs, misses = compute_logprobs_incremental(endpoint, tokens)
                    dt = time.time() - t0
                    all_logprobs.extend(logprobs)
                    total_misses += misses
                    total_tokens += len(logprobs)
                    logger.info(f"  [{eval_tag}] {gen_tag}/{rid} {len(logprobs)} tok in {dt:.1f}s ({dt/max(len(logprobs),1)*1000:.0f}ms/tok, {misses} misses)")

                if all_logprobs:
                    ppl, mean_lp = perplexity_from_logprobs(all_logprobs)
                    results[eval_tag][gen_tag] = {
                        "mean_ppl": ppl,
                        "mean_logprob": mean_lp,
                        "n_scenarios": len(scenarios),
                        "n_tokens": total_tokens,
                        "n_misses": total_misses,
                    }
                    logger.info(f"  [{eval_tag}] {gen_tag}: ppl={ppl:.2f} mean_lp={mean_lp:.3f} ({total_tokens} tok, {total_misses} misses)")

                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
        finally:
            stop_local_server(srv)

    logger.info(f"Saved to {args.output}")
    logger.info("=== CROSS-PERPLEXITY V3 DONE ===")


if __name__ == "__main__":
    main()