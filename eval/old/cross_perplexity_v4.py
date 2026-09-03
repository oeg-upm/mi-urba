#!/usr/bin/env python3
"""Cross-perplexity v4: uses prompt_logprobs (patched llama.cpp), 1 request per text.

Flow:
  1. For each generator G: generate text with temp=0 via /completion (local server)
     - Record prompt_fmt + content + n_prompt_tokens
  2. For each evaluator E:
     - For each generator's text:
       - Tokenize prompt_fmt + content with E -> full_tokens, and prompt_fmt alone -> n_prompt_tokens
       - Send full_tokens as prompt with n_predict:0, prompt_logprobs:true, n_probs:1
       - The response has prompt_probabilities with len(full_tokens)-1 entries:
         lps[i] = logprob of full_tokens[i+1] (no entry for pos 0)
       - Take lps[n_prompt_tokens-1:] = logprobs of the CONTENT tokens only
       - Perplexity = exp(-mean(content logprobs))

For the diagonal (E==G), the content tokens are exactly what G generated with temp=0,
so their logprobs ≈ 0 → perplexity ≈ 1.0.

Uses patched llama.cpp with prompt_logprobs support.
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


def start_local_server(server_bin, gguf_path, port, ctx=8192, threads=4):
    endpoint = f"http://127.0.0.1:{port}"
    # assert the port is truly free: a stale server would answer /health and
    # silently make every evaluation run against the wrong model
    try:
        requests.get(f"{endpoint}/health", timeout=2)
        raise RuntimeError(f"port {port} already in use - kill stale llama-server first")
    except requests.exceptions.ConnectionError:
        pass
    log_file = f"/tmp/xppl_server_{port}.log"
    srv = subprocess.Popen(
        [server_bin, "-m", gguf_path, "--port", str(port),
         "--host", "127.0.0.1", "-c", str(ctx), "-t", str(threads), "-ngl", "99"],
        stdout=open(log_file, "w"), stderr=subprocess.STDOUT,
    )
    for _ in range(40):
        if srv.poll() is not None:
            raise RuntimeError(f"llama-server died on start (log: {log_file})")
        try:
            if requests.get(f"{endpoint}/health", timeout=3).json().get("status") == "ok":
                return srv, endpoint
        except:
            pass
        time.sleep(2)
    srv.kill()
    raise RuntimeError(f"Server not ready at {endpoint}")


def stop_local_server(srv):
    srv.kill()
    srv.wait()
    time.sleep(3)


def parse_user_text(content):
    import re
    if not content:
        return None
    try:
        return json.loads(content.strip()).get("user_text", "")
    except:
        pass
    c = content
    if c.startswith("```"):
        c = "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
    try:
        return json.loads(c.strip()).get("user_text", "")
    except:
        pass
    m = re.search(r'\{[^{}]*"user_text"[^{}]*\}', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()).get("user_text", "")
        except:
            pass
    return None


def generate_with_local_server(endpoint, scenario, timeout=300):
    msgs = reconstruct_messages(scenario)
    r = requests.post(f"{endpoint}/apply-template", json={
        "messages": msgs, "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=30)
    r.raise_for_status()
    prompt_fmt = r.json()["prompt"]
    r2 = requests.post(f"{endpoint}/completion", json={
        "prompt": prompt_fmt, "n_predict": 10000, "temperature": 0, "top_k": 1,
        # neutral penalties: with repeat_penalty=1.1 (llama.cpp default) the sampled
        # token is not the raw argmax, which breaks the diagonal==1.0 property of
        # the cross-perplexity matrix
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "return_tokens": True, "n_probs": 0, "cache_prompt": True,
    }, timeout=timeout)
    r2.raise_for_status()
    d = r2.json()
    return prompt_fmt, d.get("content", ""), d.get("tokens", [])


def tokenize(endpoint, text):
    r = requests.post(f"{endpoint}/tokenize", json={"content": text}, timeout=30)
    r.raise_for_status()
    return r.json()["tokens"]


def get_prompt_logprobs(endpoint, tokens, timeout=120):
    """Send tokens as prompt with n_predict:0, prompt_logprobs:true, n_probs:1.
    Returns list of logprobs (index 1+, first is None for BOS).
    """
    r = requests.post(f"{endpoint}/completion", json={
        "prompt": tokens, "n_predict": 0, "n_probs": 1,
        "prompt_logprobs": True, "cache_prompt": False,
    }, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    pp = d.get("prompt_probabilities", [])
    return [e.get("logprob") for e in pp]


def eval_one_text(endpoint, prompt_fmt, content, max_content_tokens):
    """Evaluate content tokens under the model at `endpoint`.
    Returns (content_logprobs, n_content_tokens) or raises.
    Truncates content to max_content_tokens to fit context.
    """
    full_text = prompt_fmt + content
    full_tokens = tokenize(endpoint, full_text)
    prompt_tokens = tokenize(endpoint, prompt_fmt)
    n_prompt = len(prompt_tokens)
    # truncate: keep prompt + first max_content_tokens of content
    limit = n_prompt + max_content_tokens
    if len(full_tokens) > limit:
        full_tokens = full_tokens[:limit]
    lps = get_prompt_logprobs(endpoint, full_tokens)
    # sanity: tokenization boundary must be stable (prompt is a prefix of full)
    if full_tokens[:n_prompt] != prompt_tokens:
        logger.warning(
            "tokenization boundary mismatch: n_prompt=%d, "
            "prefix match=%d/%d", n_prompt,
            next((k for k in range(min(n_prompt, len(full_tokens)))
                  if full_tokens[k] != prompt_tokens[k]), 0),
            n_prompt)
    # lps[i] = logprob of full_tokens[i+1]; len(lps) == len(full_tokens) - 1
    # content tokens start at index n_prompt -> logprobs start at lps[n_prompt-1]
    content_lps = lps[n_prompt - 1:]
    return content_lps


def perplexity_from_logprobs(logprobs):
    valid = [lp for lp in logprobs if lp is not None]
    if not valid:
        return float("inf"), 0.0
    mean_lp = sum(valid) / len(valid)
    ppl = math.exp(-mean_lp) if mean_lp > -50 else float("inf")
    return ppl, mean_lp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--n-scenarios", type=int, default=5)
    parser.add_argument("--server-port", type=int, default=18888)
    parser.add_argument("--server-bin", default="/home/dani/llama.cpp-xppl/build/bin/llama-server")
    parser.add_argument("--gguf-dir", default="/home/dani/models/gguf")
    parser.add_argument("--output", default=str(DIVERSITY_DIR / "cross_perplexity_v4.json"))
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--max-content-tokens", type=int, default=1000,
                        help="max content tokens to evaluate per text (fit context)")
    args = parser.parse_args()

    models = args.models
    scenarios = load_scenarios()[:args.n_scenarios]
    logger.info(f"Models: {models}, Scenarios: {len(scenarios)}")

    gguf_map = {
        "qwen3-0.6b": f"{args.gguf_dir}/qwen3-0.6b/Qwen3-0.6B.Q4_K_M.gguf",
        "llama-3.2-1b-instruct": f"{args.gguf_dir}/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct.Q4_K_M.gguf",
        "gemma-2-2b-it": f"{args.gguf_dir}/gemma-2-2b-it/gemma-2-2b-it.Q4_K_M.gguf",
        "falcon3-1b-instruct": f"{args.gguf_dir}/falcon3-1b-instruct/Falcon3-1B-Instruct-Q4_K_M.gguf",
    }

    results = {}
    if os.path.exists(args.output):
        results = json.load(open(args.output))

    gen_cache_file = DIVERSITY_DIR / "xppl_gen_cache_v4.json"
    gen_cache = {}
    if gen_cache_file.exists():
        gen_cache = json.load(open(gen_cache_file))

    # Phase 1: generate
    logger.info("=== PHASE 1: Generate texts with temp=0 (local) ===")
    for gen_tag in models:
        gguf = gguf_map.get(gen_tag)
        if not gguf or not os.path.exists(gguf):
            logger.error(f"GGUF not found for {gen_tag}")
            continue
        if gen_tag not in gen_cache:
            gen_cache[gen_tag] = {}
        if all(sc["record_id"] in gen_cache[gen_tag] for sc in scenarios) and not args.regenerate:
            logger.info(f"[GEN] {gen_tag} cached")
            continue
        logger.info(f"[GEN] Loading {gen_tag}...")
        srv, endpoint = start_local_server(args.server_bin, gguf, args.server_port)
        try:
            for sc in scenarios:
                rid = sc["record_id"]
                if rid in gen_cache[gen_tag] and not args.regenerate:
                    continue
                logger.info(f"[GEN] {gen_tag}/{rid} generating...")
                try:
                    prompt_fmt, content, gen_tokens = generate_with_local_server(endpoint, sc)
                    if content:
                        gen_cache[gen_tag][rid] = {"prompt_fmt": prompt_fmt, "content": content}
                        logger.info(f"  generated {len(content)} chars")
                        with open(gen_cache_file, "w") as f:
                            json.dump(gen_cache, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"  failed: {e}")
        finally:
            stop_local_server(srv)

    # Phase 2: cross-perplexity
    logger.info("=== PHASE 2: Cross-perplexity (prompt_logprobs, 1 req per text) ===")
    for eval_tag in models:
        gguf = gguf_map.get(eval_tag)
        if not gguf or not os.path.exists(gguf):
            continue
        if eval_tag not in results:
            results[eval_tag] = {}
        logger.info(f"[EVAL] Loading {eval_tag}...")
        srv, endpoint = start_local_server(args.server_bin, gguf, args.server_port)
        try:
            for gen_tag in models:
                if gen_tag in results[eval_tag] and results[eval_tag][gen_tag].get("n_scenarios", 0) >= len(scenarios):
                    logger.info(f"  [{eval_tag}] {gen_tag} done")
                    continue
                all_logprobs = []
                total_tokens = 0
                for sc in scenarios:
                    rid = sc["record_id"]
                    cached = gen_cache.get(gen_tag, {}).get(rid)
                    if not cached:
                        continue
                    t0 = time.time()
                    try:
                        content_lps = eval_one_text(
                            endpoint, cached["prompt_fmt"], cached["content"],
                            args.max_content_tokens)
                    except Exception as e:
                        logger.warning(f"  [{eval_tag}] {gen_tag}/{rid} FAILED: {e}")
                        continue
                    dt = time.time() - t0
                    all_logprobs.extend(content_lps)
                    total_tokens += len(content_lps)
                    logger.info(f"  [{eval_tag}] {gen_tag}/{rid} {len(content_lps)} content tok in {dt:.1f}s")
                if all_logprobs:
                    ppl, mean_lp = perplexity_from_logprobs(all_logprobs)
                    if math.isfinite(ppl):
                        results[eval_tag][gen_tag] = {
                            "mean_ppl": ppl, "mean_logprob": mean_lp,
                            "n_scenarios": len(scenarios), "n_tokens": total_tokens,
                        }
                        logger.info(f"  [{eval_tag}] {gen_tag}: ppl={ppl:.3f} mean_lp={mean_lp:.3f} ({total_tokens} tok)")
                    else:
                        logger.warning(f"  [{eval_tag}] {gen_tag}: no valid logprobs, skipping entry")
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
        finally:
            stop_local_server(srv)

    logger.info(f"Saved to {args.output}")
    logger.info("=== CROSS-PERPLEXITY V4 DONE ===")


if __name__ == "__main__":
    main()