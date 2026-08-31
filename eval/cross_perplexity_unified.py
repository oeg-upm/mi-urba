#!/usr/bin/env python3
"""Cross-perplexity matrix between LLMs — consolidated script (v1+v2+v3+v4).

Measures how similar the OUTPUT DISTRIBUTIONS of two models are, using the
texts each model produces as probes: for each evaluator E and generator G,
compute the mean NLL (nats/token) of G's generated content under E, conditioned
on G's own generation prompt. Low  -> E finds G's text natural (similar).
High -> E finds G's text surprising (distinct).

Design decisions, and which version each comes from
---------------------------------------------------
1. Score ONLY the generated content tokens, never the prompt (v4).
   v1-v3 pooled prompt+content NLL, which pollutes the matrix with prompt
   length and prompt-familiarity effects unrelated to model similarity.
2. temp=0 greedy generation with NEUTRAL penalties
   (repeat_penalty=1.0, presence=0, frequency=0) (v4).
   llama.cpp's default repeat_penalty=1.1 makes the sampled token differ from
   the raw argmax, breaking the property "diagonal E==G -> PPL ~= 1".
3. Exact diagonal via captured gen_tokens (v3 idea, hardened):
   the generator's own tokens are stored in the gen cache and reused when
   E==G, avoiding any re-tokenization round-trip instability.
4. ONE request per text via prompt_logprobs (v4, patched llama.cpp).
   v1-v3 did one /completion call per token (O(L) requests per text, minutes
   per cell). Requires the patched server:
       eval/llama_cpp_prompt_logprobs.patch
       server: /home/dani/llama.cpp-xppl/build/bin/llama-server
   Fallback --method incremental keeps the v1-v3 mechanism (per-token requests,
   top-N logprob lookup with floor) for unpatched servers. --method auto
   tries prompt_logprobs first and falls back automatically.
5. Per-scenario granularity (new): every (eval, gen, scenario) cell stores its
   own mean_logprob + n_tokens, enabling paired/outlier-aware analysis; the
   pooled per-cell aggregate is derived from these, not from a second pass.
6. Resumable + crash-safe (v4 hardening): results saved atomically after every
   cell; completed cells are skipped on re-run; the port is asserted free
   before launching a server so a stale process can't silently evaluate
   against the wrong model (v4 lesson).
7. GGUF discovery: explicit map first (v4 paths), then glob fallback under
   --gguf-dir, so new models don't require editing the map.
8. End-of-run summary (same definitions as analyze_cross_perplexity.py):
   H[A][B]  cross-entropy nats/token
   KLD[A][B] = H(A,text_B) - H(B,text_B)  ~= KL(P_B || P_A)   model-vs-model
   J[A][B]  = KLD[A][B] + KLD[B][A]       Jeffreys, symmetric pairwise distance
   distinctiveness(B) = mean_A!=B KLD[A][B]  ranking of most distinctive models

Output
------
  eval/results/diversity/cross_perplexity_unified.json
      { evaluator: { generator: {
          mean_ppl, mean_logprob, n_scenarios, n_tokens, n_misses, method,
          scenarios: { record_id: { mean_logprob, n_tokens } } } } }
  eval/results/diversity/xppl_gen_cache_unified.json
      { generator: { record_id: { prompt_fmt, content, gen_tokens? } } }

Usage
-----
  python3 eval/cross_perplexity_unified.py \
      --models qwen3-0.6b llama-3.2-1b-instruct gemma-2-2b-it falcon3-1b-instruct \
      --n-scenarios 5
"""
import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "miurba-generator"))
from src.prompts import build_generation_prompt  # noqa: E402

DIVERSITY_DIR = ROOT / "eval" / "results" / "diversity"
DEFAULT_OUTPUT = DIVERSITY_DIR / "cross_perplexity_unified.json"
DEFAULT_GEN_CACHE = DIVERSITY_DIR / "xppl_gen_cache_unified.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xppu")

N_PROBS = 1000          # top-N lookup for the incremental fallback
FLOOR_LOGPROB = -20.0   # floor for tokens not found in top-N (incremental only)


# ────────────────────────── scenarios & prompts ──────────────────────────

def load_scenarios(n_scenarios):
    scenarios = json.load(open(DIVERSITY_DIR / "scenarios.json"))
    return scenarios[:n_scenarios]


def reconstruct_messages(scenario):
    return build_generation_prompt(
        record_id=scenario["record_id"],
        problems=scenario["problems"],
        community=scenario["community"],
        num_pics=scenario["num_pics"],
        persona=scenario.get("persona"),
        modifier=scenario.get("modifier"),
    )


def parse_user_text(content):
    """Extract user_text from a model's JSON(-ish) output."""
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
        except Exception:
            pass
    return None


# ────────────────────────── local server management ──────────────────────────

def start_local_server(server_bin, gguf_path, port, ctx, threads=4, gpu_layers=99):
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
         "--host", "127.0.0.1", "-c", str(ctx), "-t", str(threads),
         "-ngl", str(gpu_layers)],
        stdout=open(log_file, "w"), stderr=subprocess.STDOUT,
    )
    for _ in range(40):
        if srv.poll() is not None:
            raise RuntimeError(f"llama-server died on start (log: {log_file})")
        try:
            if requests.get(f"{endpoint}/health", timeout=3).json().get("status") == "ok":
                return srv, endpoint
        except Exception:
            pass
        time.sleep(2)
    srv.kill()
    raise RuntimeError(f"Server not ready at {endpoint} (log: {log_file})")


def stop_local_server(srv):
    srv.kill()
    srv.wait()
    time.sleep(3)


def find_gguf(gguf_dir, tag):
    """Locate a GGUF for a model tag: explicit map first, then glob fallback."""
    explicit = {
        "qwen3-0.6b": "qwen3-0.6b/Qwen3-0.6B.Q4_K_M.gguf",
        "llama-3.2-1b-instruct": "llama-3.2-1b-instruct/Llama-3.2-1B-Instruct.Q4_K_M.gguf",
        "gemma-2-2b-it": "gemma-2-2b-it/gemma-2-2b-it.Q4_K_M.gguf",
        "falcon3-1b-instruct": "falcon3-1b-instruct/Falcon3-1B-Instruct-Q4_K_M.gguf",
    }
    if tag in explicit:
        p = Path(gguf_dir) / explicit[tag]
        if p.exists():
            return str(p)
    root = Path(gguf_dir)
    # <gguf_dir>/<tag>/*.gguf, then <gguf_dir>/*<tag>*/*.gguf, then any match
    for pattern in (tag, f"*{tag}*"):
        hits = sorted(root.glob(pattern + "/*.gguf"))
        if hits:
            return str(hits[0])
    hits = sorted(root.rglob("*.gguf"))
    for h in hits:
        if tag.lower() in str(h).lower():
            return str(h)
    return None


# ────────────────────────── phase 1: greedy generation ──────────────────────────

def generate_greedy(endpoint, scenario, timeout=300):
    """Generate complaint text with the loaded model, temp=0, neutral penalties.

    Returns (prompt_fmt, content, gen_tokens).
    """
    msgs = reconstruct_messages(scenario)
    r = requests.post(f"{endpoint}/apply-template", json={
        "messages": msgs,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=30)
    r.raise_for_status()
    prompt_fmt = r.json()["prompt"]
    r2 = requests.post(f"{endpoint}/completion", json={
        "prompt": prompt_fmt,
        "n_predict": 10000,
        "temperature": 0,
        "top_k": 1,
        # neutral penalties: sampled token == raw argmax, which keeps the
        # diagonal (E==G -> PPL ~= 1) property of the matrix
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "return_tokens": True,
        "n_probs": 0,
        "cache_prompt": True,
    }, timeout=timeout)
    r2.raise_for_status()
    d = r2.json()
    return prompt_fmt, d.get("content", ""), d.get("tokens", [])


def _run_generation_phase_gateway(models, scenarios, args, gen_cache):
    """Gateway mode: prompt templates rendered from the GGUF template cache
    (jinja2) when available, else locally via a CPU server (/apply-template);
    greedy generation through llama-swap."""
    templates = load_json(args.templates_file, {}).get("templates", {})
    for gen_tag in models:
        gen_cache.setdefault(gen_tag, {})
        missing = [sc for sc in scenarios
                   if not gen_cache[gen_tag].get(sc["record_id"], {}).get("content")]
        if not missing and not args.regenerate:
            logger.info(f"[GEN] {gen_tag}: all cached (gateway)")
            continue
        # precedence: local GGUF -> exact /apply-template; else GGUF template
        # cache (jinja2 render; minor llama.cpp-specific differences are OK —
        # the diagonal stays exact by construction since content is generated
        # from the same prompt_fmt that is later scored)
        base_tag = gen_tag.removesuffix("-xppl")
        gguf = find_gguf(args.gguf_dir, base_tag)
        if gguf:
            logger.info(f"[GEN] {gen_tag}: rendering {len(missing)} templates locally (exact, CPU)")
            srv, endpoint = start_local_server(args.server_bin, gguf, args.template_port,
                                               ctx=2048, gpu_layers=0)
            try:
                for sc in missing:
                    rid = sc["record_id"]
                    if "prompt_fmt" not in gen_cache[gen_tag].get(rid, {}):
                        gen_cache[gen_tag][rid] = {"prompt_fmt": render_prompt_fmt(endpoint, sc)}
                        save_json(args.gen_cache, gen_cache)
            finally:
                stop_local_server(srv)
        elif gen_tag in templates:
            for sc in missing:
                rid = sc["record_id"]
                if "prompt_fmt" not in gen_cache[gen_tag].get(rid, {}):
                    try:
                        pf = render_prompt_fmt_from_template(templates[gen_tag]["chat_template"], sc)
                    except Exception as e:
                        logger.error(f"[GEN] {gen_tag}/{rid} template render failed: {e}")
                        continue
                    gen_cache[gen_tag][rid] = {"prompt_fmt": pf}
                    save_json(args.gen_cache, gen_cache)
            logger.info(f"[GEN] {gen_tag}: rendered {len(missing)} templates from GGUF template cache")
        else:
            logger.error(f"[GEN] {gen_tag}: no template in cache and no local GGUF for {base_tag}")
            continue
        for sc in missing:
            rid = sc["record_id"]
            if "prompt_fmt" not in gen_cache[gen_tag].get(rid, {}):
                continue
            prompt_fmt = gen_cache[gen_tag][rid]["prompt_fmt"]
            logger.info(f"[GEN] {gen_tag}/{rid} generating via gateway (temp=0)...")
            try:
                content = generate_greedy_gateway(args.gateway, gen_tag, prompt_fmt,
                                                  timeout=2 * args.req_timeout,
                                                  max_gen_tokens=args.max_gen_tokens)
                if content:
                    gen_cache[gen_tag][rid]["content"] = content
                    save_json(args.gen_cache, gen_cache)
                    logger.info(f"  {len(content)} chars")
                else:
                    logger.warning(f"  empty result for {gen_tag}/{rid}")
            except Exception as e:
                logger.error(f"  {gen_tag}/{rid} failed: {e}")


def _run_eval_phase_gateway(models, scenarios, args, gen_cache, results):
    """Cross-perplexity entirely through llama-swap (see eval_text_gateway)."""
    for eval_tag in models:
        results.setdefault(eval_tag, {})
        for gen_tag in models:
            cached_all = gen_cache.get(gen_tag, {})
            have = [sc["record_id"] for sc in scenarios
                    if cached_all.get(sc["record_id"], {}).get("content")]
            prev = results[eval_tag].get(gen_tag)
            if prev and prev.get("n_scenarios", 0) >= len(have) and not args.regenerate:
                logger.info(f"  [{eval_tag}] {gen_tag}: done ({prev['n_scenarios']}), skip")
                continue
            scen_out = {}
            all_lps, total_misses, total_tokens = [], 0, 0
            for sc in scenarios:
                rid = sc["record_id"]
                cached = cached_all.get(rid) or {}
                if not cached.get("content"):
                    continue
                t0 = time.time()
                try:
                    lps, misses = eval_text_gateway(
                        args.gateway, eval_tag, cached["prompt_fmt"],
                        cached["content"], args.max_content_tokens,
                        req_timeout=args.req_timeout)
                except Exception as e:
                    logger.warning(f"  [{eval_tag}] {gen_tag}/{rid} FAILED: {e}")
                    continue
                dt = time.time() - t0
                if not lps:
                    logger.warning(f"  [{eval_tag}] {gen_tag}/{rid}: no valid logprobs")
                    continue
                scen_out[rid] = {"mean_logprob": sum(lps) / len(lps), "n_tokens": len(lps)}
                all_lps.extend(lps)
                total_misses += misses
                total_tokens += len(lps)
                logger.info(f"  [{eval_tag}] {gen_tag}/{rid} {len(lps)} tok in {dt:.1f}s")
            if all_lps:
                mean_lp = sum(all_lps) / len(all_lps)
                ppl = math.exp(-mean_lp) if mean_lp > -50 else float("inf")
                results[eval_tag][gen_tag] = {
                    "mean_ppl": ppl, "mean_logprob": mean_lp,
                    "n_scenarios": len(scen_out), "n_tokens": total_tokens,
                    "n_misses": total_misses, "method": "gateway-prompt_logprobs",
                    "scenarios": scen_out,
                }
                logger.info(f"  [{eval_tag}] {gen_tag}: ppl={ppl:.3f} H={-mean_lp:.3f} nats "
                            f"({total_tokens} tok)")
            save_json(args.output, results)


def run_generation_phase(models, scenarios, args, gen_cache):
    """Ensure gen_cache[gen][rid] = {prompt_fmt, content, gen_tokens?} for all."""
    if args.gateway:
        _run_generation_phase_gateway(models, scenarios, args, gen_cache)
        return
    for gen_tag in models:
        gguf = find_gguf(args.gguf_dir, gen_tag)
        if not gguf:
            logger.error(f"[GEN] GGUF not found for {gen_tag} under {args.gguf_dir}")
            continue
        gen_cache.setdefault(gen_tag, {})
        missing = [sc["record_id"] for sc in scenarios if sc["record_id"] not in gen_cache[gen_tag]]
        if not missing and not args.regenerate:
            logger.info(f"[GEN] {gen_tag}: all {len(scenarios)} scenarios cached")
            continue
        logger.info(f"[GEN] Loading {gen_tag} ({gguf}), {len(missing)} to generate")
        srv, endpoint = start_local_server(args.server_bin, gguf, args.server_port, args.ctx)
        try:
            for sc in scenarios:
                rid = sc["record_id"]
                if rid in gen_cache[gen_tag] and not args.regenerate:
                    continue
                logger.info(f"[GEN] {gen_tag}/{rid} generating (temp=0)...")
                try:
                    prompt_fmt, content, gen_tokens = generate_greedy(endpoint, sc)
                    if content and gen_tokens:
                        gen_cache[gen_tag][rid] = {
                            "prompt_fmt": prompt_fmt,
                            "content": content,
                            "gen_tokens": gen_tokens,
                        }
                        logger.info(f"  {len(content)} chars, {len(gen_tokens)} tokens")
                        save_json(args.gen_cache, gen_cache)
                    else:
                        logger.warning(f"  empty result for {gen_tag}/{rid}")
                except Exception as e:
                    logger.error(f"  {gen_tag}/{rid} failed: {e}")
        finally:
            stop_local_server(srv)


def render_prompt_fmt(endpoint, scenario):
    """Chat-template rendering only (no generation) — used in gateway mode,
    where llama-swap does not expose /apply-template."""
    msgs = reconstruct_messages(scenario)
    r = requests.post(f"{endpoint}/apply-template", json={
        "messages": msgs,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=30)
    r.raise_for_status()
    return r.json()["prompt"]


def render_prompt_fmt_from_template(template, scenario):
    """Render a GGUF chat template locally with jinja2 (HF conventions:
    trim_blocks/lstrip_blocks, strftime_now, add_generation_prompt).
    Equivalent to llama-server's /apply-template for standard templates."""
    import datetime
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def strftime_now(fmt):
        return datetime.datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        keep_trailing_newline=True)
    env.globals["strftime_now"] = strftime_now
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    t = template.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    msgs = reconstruct_messages(scenario)
    return env.from_string(t).render(
        messages=msgs, add_generation_prompt=True, tools=[], documents=[])


def generate_greedy_gateway(endpoint, tag, prompt_fmt, timeout=600, max_gen_tokens=10000):
    """Greedy generation through llama-swap (routes /completion by the `model`
    field). Neutral penalties keep the diagonal-exact property."""
    r = requests.post(f"{endpoint}/completion", json={
        "model": tag, "prompt": prompt_fmt, "n_predict": max_gen_tokens,
        "temperature": 0, "top_k": 1,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "return_tokens": False, "n_probs": 0, "cache_prompt": False,
    }, timeout=timeout)
    r.raise_for_status()
    return r.json().get("content", "")


def eval_text_gateway(endpoint, tag, prompt_fmt, content, max_content_tokens, req_timeout=300):
    """Content-only NLL through llama-swap, without /tokenize:
      req1 scores prompt_fmt alone   -> n_prompt = len(pp)+1 (BOS has no entry)
      req2 scores prompt_fmt+content -> slice content logprobs at n_prompt-1
    Same slicing convention as eval_text_prompt_logprobs. Requires the prompt
    to be a stable tokenization prefix (holds for same-family GGUFs; exact when
    E==G). NOTE: cache_prompt must stay False — with a cached prefix this
    llama.cpp build omits prompt_probabilities entirely.
    """
    content = content[:max_content_tokens * 4]  # rough char cap for ctx fit

    def req(prompt):
        r = requests.post(f"{endpoint}/completion", json={
            "model": tag, "prompt": prompt, "n_predict": 0, "n_probs": 1,
            "prompt_logprobs": True, "cache_prompt": False,
        }, timeout=req_timeout)
        r.raise_for_status()
        pp = r.json().get("prompt_probabilities")
        if not pp:
            raise RuntimeError("server returned no prompt_probabilities")
        return pp

    pp_prompt = req(prompt_fmt)
    n_prompt = len(pp_prompt) + 1
    pp_full = req(prompt_fmt + content)
    content_lps = [e.get("logprob") for e in pp_full[n_prompt - 1:]]
    misses = sum(1 for lp in content_lps if lp is None)
    return [lp for lp in content_lps if lp is not None], misses


# ────────────────────────── phase 2: cross-perplexity ──────────────────────────

def tokenize(endpoint, text):
    r = requests.post(f"{endpoint}/tokenize", json={"content": text}, timeout=30)
    r.raise_for_status()
    return r.json()["tokens"]


def get_prompt_logprobs(endpoint, tokens, timeout=300):
    """Patched llama.cpp: 1 request, logprob of every prompt token (index 1+).

    Returns list where lps[i] = logprob of tokens[i+1] (len == len(tokens)-1),
    or None if the server does not support prompt_logprobs.
    """
    r = requests.post(f"{endpoint}/completion", json={
        "prompt": tokens, "n_predict": 0, "n_probs": 1,
        "prompt_logprobs": True, "cache_prompt": False,
    }, timeout=timeout)
    r.raise_for_status()
    pp = r.json().get("prompt_probabilities")
    if not pp:
        return None
    return [e.get("logprob") for e in pp]


def eval_text_prompt_logprobs(endpoint, cached, is_diagonal, max_content_tokens):
    """Content-only NLL via the patched server. Returns (logprobs, misses)."""
    if is_diagonal and cached.get("gen_tokens"):
        # exact alignment: the very tokens the generator produced
        prompt_tokens = tokenize(endpoint, cached["prompt_fmt"])
        full_tokens = prompt_tokens + cached["gen_tokens"]
        n_prompt = len(prompt_tokens)
    else:
        full_tokens = tokenize(endpoint, cached["prompt_fmt"] + cached["content"])
        prompt_tokens = tokenize(endpoint, cached["prompt_fmt"])
        n_prompt = len(prompt_tokens)
        if full_tokens[:n_prompt] != prompt_tokens:
            logger.warning("tokenization boundary mismatch (prompt not a prefix of full)")
    limit = n_prompt + max_content_tokens
    if len(full_tokens) > limit:
        full_tokens = full_tokens[:limit]
    lps = get_prompt_logprobs(endpoint, full_tokens)
    if lps is None:
        return None, 0
    # lps[i] = logprob of full_tokens[i+1]; content starts at index n_prompt
    content_lps = lps[n_prompt - 1:]
    return [lp for lp in content_lps if lp is not None], sum(1 for lp in content_lps if lp is None)


def eval_text_incremental(endpoint, cached, is_diagonal, max_content_tokens):
    """Content-only NLL, one request per token (v1-v3 mechanism).

    Works on ANY llama.cpp server. Returns (logprobs, misses).
    """
    prompt_tokens = tokenize(endpoint, cached["prompt_fmt"])
    n_prompt = len(prompt_tokens)
    if is_diagonal and cached.get("gen_tokens"):
        full_tokens = prompt_tokens + cached["gen_tokens"]
    else:
        full_tokens = tokenize(endpoint, cached["prompt_fmt"] + cached["content"])
        if full_tokens[:n_prompt] != prompt_tokens:
            logger.warning("tokenization boundary mismatch (prompt not a prefix of full)")
    limit = n_prompt + max_content_tokens
    if len(full_tokens) > limit:
        full_tokens = full_tokens[:limit]
    logprobs, misses = [], 0
    for k in range(n_prompt, len(full_tokens)):
        target = full_tokens[k]
        try:
            r = requests.post(f"{endpoint}/completion", json={
                "prompt": full_tokens[:k],
                "n_predict": 1,
                "n_probs": N_PROBS,
                "cache_prompt": True,
                "temperature": 0,
                "top_k": 1,
            }, timeout=120)
            r.raise_for_status()
            cp = r.json().get("completion_probabilities") or []
            found = None
            if cp and cp[0].get("top_logprobs"):
                for t in cp[0]["top_logprobs"]:
                    if t["id"] == target:
                        found = t["logprob"]
                        break
            if found is not None:
                logprobs.append(found)
            else:
                logprobs.append(FLOOR_LOGPROB)
                misses += 1
        except Exception as e:
            logger.warning(f"  token {k} failed: {e}")
            logprobs.append(FLOOR_LOGPROB)
            misses += 1
    return logprobs, misses


def run_eval_phase(models, scenarios, args, gen_cache, results):
    """For every evaluator E and generator G, fill results[E][G]."""
    if args.gateway:
        _run_eval_phase_gateway(models, scenarios, args, gen_cache, results)
        return
    for eval_tag in models:
        gguf = find_gguf(args.gguf_dir, eval_tag)
        if not gguf:
            logger.error(f"[EVAL] GGUF not found for {eval_tag} under {args.gguf_dir}")
            continue
        results.setdefault(eval_tag, {})
        done = results[eval_tag].get("__done_scenarios__", 0)
        # resume check is per-cell below; __done_scenarios__ kept for back-compat
        logger.info(f"[EVAL] Loading {eval_tag} ({gguf})")
        srv, endpoint = start_local_server(args.server_bin, gguf, args.server_port, args.ctx)
        method_used = None
        try:
            for gen_tag in models:
                cached_all = gen_cache.get(gen_tag, {})
                have = [sc["record_id"] for sc in scenarios if sc["record_id"] in cached_all]
                prev = results[eval_tag].get(gen_tag)
                if prev and prev.get("n_scenarios", 0) >= len(have) and not args.regenerate:
                    logger.info(f"  [{eval_tag}] {gen_tag}: done ({prev['n_scenarios']} scenarios), skip")
                    continue
                is_diagonal = (eval_tag == gen_tag)
                scen_out = {}
                all_lps, total_misses, total_tokens = [], 0, 0
                for sc in scenarios:
                    rid = sc["record_id"]
                    cached = cached_all.get(rid)
                    if not cached:
                        continue
                    t0 = time.time()
                    try:
                        lps, misses = (None, 0)
                        if args.method in ("auto", "prompt_logprobs"):
                            lps, misses = eval_text_prompt_logprobs(
                                endpoint, cached, is_diagonal, args.max_content_tokens)
                            if lps is None:  # server lacks the patch
                                if args.method == "prompt_logprobs":
                                    raise RuntimeError("server has no prompt_logprobs support")
                                logger.warning("  no prompt_logprobs support, falling back to incremental")
                                lps, misses = eval_text_incremental(
                                    endpoint, cached, is_diagonal, args.max_content_tokens)
                                method_used = "incremental"
                            else:
                                method_used = "prompt_logprobs"
                        else:
                            lps, misses = eval_text_incremental(
                                endpoint, cached, is_diagonal, args.max_content_tokens)
                            method_used = "incremental"
                    except Exception as e:
                        logger.warning(f"  [{eval_tag}] {gen_tag}/{rid} FAILED: {e}")
                        continue
                    dt = time.time() - t0
                    if not lps:
                        logger.warning(f"  [{eval_tag}] {gen_tag}/{rid}: no valid logprobs")
                        continue
                    scen_out[rid] = {"mean_logprob": sum(lps) / len(lps), "n_tokens": len(lps)}
                    all_lps.extend(lps)
                    total_misses += misses
                    total_tokens += len(lps)
                    logger.info(f"  [{eval_tag}] {gen_tag}/{rid} {len(lps)} tok in {dt:.1f}s")
                if all_lps:
                    mean_lp = sum(all_lps) / len(all_lps)
                    ppl = math.exp(-mean_lp) if mean_lp > -50 else float("inf")
                    results[eval_tag][gen_tag] = {
                        "mean_ppl": ppl,
                        "mean_logprob": mean_lp,
                        "n_scenarios": len(scen_out),
                        "n_tokens": total_tokens,
                        "n_misses": total_misses,
                        "method": method_used,
                        "scenarios": scen_out,
                    }
                    logger.info(f"  [{eval_tag}] {gen_tag}: ppl={ppl:.3f} H={-mean_lp:.3f} nats "
                                f"({total_tokens} tok, {total_misses} misses, {method_used})")
                save_json(args.output, results)
        finally:
            stop_local_server(srv)


# ────────────────────────── summary (analyze_cross_perplexity.py logic) ──────────────────────────

def summarize(results):
    """H -> KLD ~= KL(P_gen||P_eval) -> Jeffreys -> distinctiveness ranking."""
    cells = {e: {g: c for g, c in r.items() if isinstance(c, dict) and "mean_logprob" in c}
             for e, r in results.items()}
    gens = sorted({g for r in cells.values() for g in r})
    H = {e: {g: -cells[e][g]["mean_logprob"] for g in cells[e] if g in gens} for e in cells}
    KLD = {e: {g: (H[e][g] - H[g][g]) if g in H.get(e, {}) and e in H.get(g, {}) else None
               for g in gens if g in H[e]} for e in H}

    print("\n=== Cross-entropy H[A][B] (nats/token; row=evaluator, col=generator) ===")
    _table(H, gens)
    K = {e: {g: v for g, v in KLD.get(e, {}).items() if v is not None} for e in KLD}
    print("\n=== KLD[A][B] ~= KL(P_B || P_A) (nats/token; high = B distinct from A) ===")
    _table(K, gens)

    J = {}
    for a in gens:
        J[a] = {}
        for b in gens:
            if a == b:
                J[a][b] = 0.0
            elif a in K and b in K.get(a, {}) and b in K and a in K.get(b, {}):
                J[a][b] = K[a][b] + K[b][a]
    print("\n=== Jeffreys J[A][B] (symmetric pairwise distance) ===")
    _table({a: {b: v for b, v in J[a].items() if v is not None} for a in J}, gens)

    print("\n=== Nearest neighbor (min Jeffreys) per model ===")
    for a in gens:
        others = [(J[a][b], b) for b in gens if b != a and J[a].get(b) is not None]
        if others:
            j, b = min(others)
            print(f"  {a:28s} -> {b:28s} (J={j:.3f} nats)")

    scores = {}
    for g in gens:
        vals = [K[e][g] for e in K if e != g and g in K.get(e, {}) and K[e].get(g) is not None]
        if vals:
            scores[g] = sum(vals) / len(vals)
    if scores:
        print("\n=== Distinctiveness: mean KL(P_g || P_others) — high = unique voice ===")
        for g, m in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"  {g:28s} {m:.3f} nats")


def _table(M, gens):
    short = {g: g[:16] for g in gens}
    print("  " + " ".join(f"{short[g]:>18s}" for g in gens))
    for a in gens:
        if a not in M:
            continue
        row = " ".join(
            f"{M[a][b]:18.3f}" if b in M.get(a, {}) and M[a].get(b) is not None else " " * 18
            for b in gens)
        print(f"  {a[:16]:16s} {row}")


# ────────────────────────── IO & CLI ──────────────────────────

def save_json(path, obj):
    """Atomic write: tmp + rename, so an interrupt never corrupts results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.rename(path)


def load_json(path, default):
    p = Path(path)
    if p.exists():
        try:
            return json.load(open(p))
        except json.JSONDecodeError:
            backup = p.with_suffix(p.suffix + ".corrupt")
            p.rename(backup)
            logger.warning(f"{p} was corrupt JSON, moved to {backup}")
    return default


def main():
    parser = argparse.ArgumentParser(description="Unified cross-perplexity matrix (v1-v4 consolidated)")
    parser.add_argument("--models", nargs="+", required=True, help="model tags (generator+evaluator)")
    parser.add_argument("--n-scenarios", type=int, default=5)
    parser.add_argument("--server-port", type=int, default=18888)
    parser.add_argument("--server-bin", default="/home/dani/llama.cpp-xppl/build/bin/llama-server")
    parser.add_argument("--gguf-dir", default="/home/dani/models/gguf")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--method", choices=["auto", "prompt_logprobs", "incremental"], default="auto",
                        help="prompt_logprobs needs the patched server (1 req/text); "
                             "incremental works on any llama.cpp (N req/text)")
    parser.add_argument("--max-content-tokens", type=int, default=1000,
                        help="max content tokens scored per text (context-fit)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gen-cache", default=str(DEFAULT_GEN_CACHE))
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--gateway", default=None,
                        help="llama-swap URL (e.g. http://192.168.10.11:9292): run "
                             "generation+eval through the gateway; model tags get "
                             "routed via the `model` field; templates are rendered "
                             "locally with the -xppl suffix stripped")
    parser.add_argument("--template-port", type=int, default=18989,
                        help="local port for the CPU template-rendering server")
    parser.add_argument("--templates-file", default=str(DIVERSITY_DIR / "chat_templates.json"),
                        help="GGUF chat-template cache (see extract_chat_template.py)")
    parser.add_argument("--req-timeout", type=int, default=300,
                        help="per-request timeout seconds (generation uses 2x)")
    parser.add_argument("--max-gen-tokens", type=int, default=10000,
                        help="greedy generation cap per text")
    args = parser.parse_args()

    models = args.models
    scenarios = load_scenarios(args.n_scenarios)
    logger.info(f"Models: {models}")
    logger.info(f"Scenarios: {len(scenarios)}, method: {args.method}"
                + (f", gateway: {args.gateway}" if args.gateway else ""))

    gen_cache = load_json(args.gen_cache, {})
    results = load_json(args.output, {})

    logger.info("=== PHASE 1: greedy generation (temp=0, neutral penalties) ===")
    run_generation_phase(models, scenarios, args, gen_cache)

    logger.info("=== PHASE 2: cross-perplexity (content-only NLL) ===")
    run_eval_phase(models, scenarios, args, gen_cache, results)

    logger.info(f"Results saved to {args.output}")
    if not args.no_summary:
        summarize(results)
    logger.info("=== CROSS-PERPLEXITY UNIFIED DONE ===")


if __name__ == "__main__":
    main()
