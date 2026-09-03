#!/usr/bin/env python3
"""Compute cross-perplexity matrix between generator models.

For each generator model A and each of its 20 complaint texts:
  1. Reconstruct the full generation prompt (system + user message) that
     produced the text, using the saved scenarios + build_generation_prompt.
  2. The complaint user_text is the "assistant response" to that prompt.

For each evaluator model B, compute the perplexity of each complaint text
under model B, IN THE CONTEXT of the original generation prompt:
  - Tokenize: prompt_A + " " + user_text_A  (as a single sequence)
  - For each token of user_text_A, get its logprob given the preceding context
    using incremental /completion calls with cache_prompt.
  - Perplexity = exp(-mean(logprobs of user_text_A tokens))

The cross-perplexity matrix P[A][B] = mean perplexity of A's texts under B.
Low P[A][B] = B finds A's text natural (similar style).
High P[A][B] = B finds A's text surprising (different style).

Uses a local llama.cpp server (one model loaded at a time). The script:
  - For each evaluator model B (loaded in the server):
    - For each generator A:
      - For each of A's 20 texts:
        - Reconstruct prompt_A + user_text_A
        - Tokenize the full sequence
        - Find the token boundary where user_text_A starts
        - Incremental logprob of each user_text_A token
        - Record logprobs

Output: eval/results/diversity/cross_perplexity.json
        { evaluator: { generator: { mean_ppl, mean_logprob, n_tokens, n_misses } } }

Usage:
  python3 eval/cross_perplexity.py --server-port 18888 --models-dir /path/to/gguf
  (expects a llama-server running with one model; the script loads models
   one at a time via /models/load, OR you pass --model for a single loaded model)

Simpler usage: run with the server already loaded with a model, pass --model.
  python3 eval/cross_perplexity.py --endpoint http://127.0.0.1:18888 --model pantheon-rp-8b-llama3
"""
import argparse, json, math, os, sys, time, logging
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "miurba-generator"))
from src.prompts import build_generation_prompt

DIVERSITY_DIR = ROOT / "eval" / "results" / "diversity"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_PROBS = 1000  # top-N logprobs to request per token (vocab is ~128k)
FLOOR_LOGPROB = -20.0  # floor for tokens not in top-N


def load_scenarios_and_texts():
    """Load scenarios + each model's complaint texts.
    Returns: scenarios {record_id: scenario}, models {tag: [texts]}
    """
    scenarios = {s["record_id"]: s for s in json.load(open(DIVERSITY_DIR / "scenarios.json"))}
    models = {}
    for d in sorted(os.listdir(DIVERSITY_DIR)):
        if not d.startswith("gen_"):
            continue
        p = DIVERSITY_DIR / d / "complaints.jsonl"
        if not p.exists():
            continue
        tag = d.replace("gen_", "")
        texts = []
        for line in open(p):
            if line.strip():
                texts.append(json.loads(line))
        if len(texts) >= 15:
            models[tag] = texts
    return scenarios, models


def reconstruct_prompt(scenario):
    """Reconstruct the generation prompt (system + user message) for a scenario."""
    msgs = build_generation_prompt(
        record_id=scenario["record_id"],
        problems=scenario["problems"],
        community=scenario["community"],
        num_pics=scenario["num_pics"],
        persona=scenario.get("persona"),
        modifier=scenario.get("modifier"),
    )
    return msgs  # [system, user]


def build_full_prompt_text(scenario, user_text):
    """Build the full text: system + user message + user_text (as assistant response).

    For perplexity, we need a single text sequence. The chat template would wrap
    this, but /completion with a raw string is simplest. We concatenate:
      system_msg + "\n" + user_msg + "\n" + user_text
    The evaluator model sees the prompt context + the complaint text.
    """
    msgs = reconstruct_prompt(scenario)
    # Concatenate as raw text (the model evaluates the sequence as-is)
    full = msgs[0]["content"] + "\n\n" + msgs[1]["content"] + "\n\n" + user_text
    return full, msgs


def tokenize(endpoint, text):
    """Tokenize text using the server's /tokenize endpoint."""
    r = requests.post(f"{endpoint}/tokenize", json={"content": text}, timeout=30)
    r.raise_for_status()
    return r.json()["tokens"]


def find_boundary_tokens(endpoint, prompt_text, full_text):
    """Find the token index where user_text starts in the full sequence.

    Tokenize prompt_text and full_text. The boundary is len(prompt_tokens).
    (Tokenizers are deterministic, so prompt_tokens is a prefix of full_tokens.)
    """
    prompt_tokens = tokenize(endpoint, prompt_text)
    full_tokens = tokenize(endpoint, full_text)
    # The prompt should be a prefix (tokenizers are left-to-right)
    # but BPE merging at the boundary can differ. Verify.
    if full_tokens[:len(prompt_tokens)] == prompt_tokens:
        return len(prompt_tokens)
    # If not exact prefix, find the longest common prefix
    boundary = 0
    for i in range(min(len(prompt_tokens), len(full_tokens))):
        if prompt_tokens[i] == full_tokens[i]:
            boundary = i + 1
        else:
            break
    return boundary


def compute_logprobs_incremental(endpoint, tokens, start_idx, timeout=60):
    """For tokens[start_idx:], compute logprob of each token given preceding context.

    Uses incremental /completion calls with cache_prompt=true.
    Returns: list of logprobs, n_misses.
    """
    logprobs = []
    misses = 0
    for k in range(start_idx, len(tokens)):
        target = tokens[k]
        try:
            r = requests.post(f"{endpoint}/completion", json={
                "prompt": tokens[:k],
                "n_predict": 1,
                "n_probs": N_PROBS,
                "cache_prompt": True,
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
    """Compute perplexity from logprobs: exp(-mean(logprobs))."""
    if not logprobs:
        return float("inf"), 0.0
    mean_lp = sum(logprobs) / len(logprobs)
    ppl = math.exp(-mean_lp) if mean_lp > -50 else float("inf")
    return ppl, mean_lp


def main():
    parser = argparse.ArgumentParser(description="Cross-perplexity matrix")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18888",
                        help="llama-server endpoint")
    parser.add_argument("--model", required=True,
                        help="Name of the model loaded in the server (for labeling)")
    parser.add_argument("--generators", nargs="*",
                        help="Generator model tags to evaluate (default: all with >=15 texts)")
    parser.add_argument("--n-texts", type=int, default=5,
                        help="Number of texts per generator to evaluate (default 5)")
    parser.add_argument("--output", default=str(DIVERSITY_DIR / "cross_perplexity.json"),
                        help="Output JSON file (merged with existing)")
    args = parser.parse_args()

    scenarios, models = load_scenarios_and_texts()
    logger.info(f"Loaded {len(scenarios)} scenarios, {len(models)} generators")

    if args.generators:
        gen_tags = [t.replace("local_llama/", "") for t in args.generators]
        models = {t: models[t] for t in gen_tags if t in models}
    else:
        # Use the 10 selected models
        sel = json.load(open(DIVERSITY_DIR / "selected_10.json"))
        gen_tags = [t.replace("local_llama/", "") for t in sel["selected"]]
        models = {t: models[t] for t in gen_tags if t in models}
    logger.info(f"Evaluating {len(models)} generators with {args.n_texts} texts each")

    # Load existing results (to resume / accumulate across evaluator models)
    results = {}
    if os.path.exists(args.output):
        results = json.load(open(args.output))
    if args.model not in results:
        results[args.model] = {}

    # For each generator, compute perplexity of its texts under this evaluator model
    for gen_tag, texts in models.items():
        # Check if already done
        if gen_tag in results[args.model] and results[args.model][gen_tag].get("n_texts", 0) >= args.n_texts:
            logger.info(f"[{args.model}] {gen_tag} already done, skip")
            continue

        all_logprobs = []
        total_misses = 0
        total_tokens = 0
        for i, rec in enumerate(texts[:args.n_texts]):
            rid = rec["id"]
            user_text = rec["user_text"]
            scenario = scenarios.get(rid)
            if not scenario:
                logger.warning(f"scenario {rid} not found, skip")
                continue

            # Build full text: prompt + user_text
            prompt_msgs = reconstruct_prompt(scenario)
            prompt_text = prompt_msgs[0]["content"] + "\n\n" + prompt_msgs[1]["content"]
            full_text = prompt_text + "\n\n" + user_text

            # Tokenize and find boundary
            try:
                prompt_tokens = tokenize(args.endpoint, prompt_text)
                full_tokens = tokenize(args.endpoint, full_text)
            except Exception as e:
                logger.warning(f"[{gen_tag}/{rid}] tokenize failed: {e}")
                continue

            # Find boundary (where user_text starts)
            if full_tokens[:len(prompt_tokens)] == prompt_tokens:
                boundary = len(prompt_tokens)
            else:
                boundary = 0
                for j in range(min(len(prompt_tokens), len(full_tokens))):
                    if prompt_tokens[j] == full_tokens[j]:
                        boundary = j + 1
                    else:
                        break

            t0 = time.time()
            logprobs, misses = compute_logprobs_incremental(args.endpoint, full_tokens, boundary)
            dt = time.time() - t0
            all_logprobs.extend(logprobs)
            total_misses += misses
            total_tokens += len(logprobs)
            logger.info(f"[{args.model}] {gen_tag}/{rid} {len(logprobs)} tok in {dt:.1f}s ({dt/max(len(logprobs),1)*1000:.0f}ms/tok, {misses} misses)")

        if all_logprobs:
            ppl, mean_lp = perplexity_from_logprobs(all_logprobs)
            results[args.model][gen_tag] = {
                "mean_ppl": ppl,
                "mean_logprob": mean_lp,
                "n_texts": len(texts[:args.n_texts]),
                "n_tokens": total_tokens,
                "n_misses": total_misses,
            }
            logger.info(f"[{args.model}] {gen_tag}: ppl={ppl:.2f} mean_lp={mean_lp:.3f} ({total_tokens} tok, {total_misses} misses)")

        # Save after each generator
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved to {args.output}")
    logger.info("=== CROSS-PERPLEXITY DONE ===")


if __name__ == "__main__":
    main()