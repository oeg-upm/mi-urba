#!/usr/bin/env python3
"""Normalized analysis of a cross-perplexity matrix.

Raw cross-PPL mixes three things: (a) true distributional distance between
generator and evaluator, (b) the evaluator's confidence/calibration (its
diagonal level), and (c) the intrinsic entropy of each generator's texts
(degenerate repetition is trivially predictable for EVERYONE).

This script derives metrics that separate those effects:

  H[A][B]        cross-entropy in nats/token (from mean_logprob, no exp)
  dH[A][B]       = H(A, text_B) - H(A, text_A)   excess surprise vs own text
  R[A][B]        = exp(dH)                        PPL ratio
  S[A][B]        = (dH[A][B] + dH[B][A]) / 2      symmetrized (signed)
  J[A][B]        = dH[A][B] + dH[B][A]            symmetric total (can be
                   negative when text entropies differ strongly: NOT a KL)

Diversity score of generator B: mean over evaluators A!=B of -dH[A][B]
(positive = B's text is MORE predictable to others than their own = redundant;
negative = surprising = diverse contribution).

Usage:
  python3 eval/analyze_cross_perplexity.py [matrix.json] [--gen-cache cache.json]
"""
import argparse, json, math, re, sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "results" / "diversity" / "cross_perplexity_v4.json"


def load_matrix(path):
    r = json.load(open(path))
    # mean_logprob = mean(log p) <= 0; cross-entropy H = -mean(log p) >= 0
    H = {e: {g: -r[e][g]["mean_logprob"] for g in r[e]} for e in r}
    return H, sorted(r.keys())


def dH_matrix(H):
    return {e: {g: H[e][g] - H[e][e] for g in H[e]} for e in H}


def symmetrized(dH):
    S = {}
    for a in dH:
        S[a] = {}
        for b in dH:
            if a == b:
                S[a][b] = 0.0
            else:
                # average of both directions; negative values are possible and
                # meaningful (sample texts have different intrinsic entropies)
                S[a][b] = (dH[a][b] + dH[b][a]) / 2
    return S


def lexical_diversity(cache, tag):
    """Mean distinct-2-gram ratio of a generator's texts (degeneracy flag)."""
    ratios = []
    for rid, entry in cache.get(tag, {}).items():
        words = re.findall(r"\w+", entry["content"].lower())
        if len(words) < 4:
            continue
        bigrams = list(zip(words, words[1:]))
        ratios.append(len(set(bigrams)) / len(bigrams))
    return sum(ratios) / len(ratios) if ratios else None


def fmt_table(title, M, gens, fmt="{:+.3f}", diag_zero=True):
    print(f"\n=== {title} ===")
    print("eval\\gen".ljust(22) + "".join(g[:13].rjust(15) for g in gens))
    for e in gens:
        cells = []
        for g in gens:
            v = M[e][g]
            cells.append((("0.000" if diag_zero and e == g else fmt.format(v))).rjust(15))
        print(e[:20].ljust(22) + "".join(cells))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    H = None
    # accept either the raw results json or a pre-built H json
    raw = json.load(open(path))
    first_val = next(iter(next(iter(raw.values())).values()))
    if isinstance(first_val, dict) and "mean_logprob" in first_val:
        gens = sorted(raw.keys())
        H = {e: {g: -raw[e][g]["mean_logprob"] for g in raw[e]} for e in raw}
    else:
        H = raw  # assume already positive cross-entropy
    gens = sorted(H.keys())
    print(f"Matrix from: {path}")
    print(f"Models ({len(gens)}): {', '.join(gens)}")

    dH = dH_matrix(H)
    S = symmetrized(dH)
    R = {e: {g: (1.0 if e == g else sys.float_info.max) for g in gens} for e in gens}
    for e in gens:
        for g in gens:
            if e != g:
                R[e][g] = math.exp(dH[e][g])

    fmt_table("dH[A][B] = excess cross-entropy (nats/token; 0 = own text)", dH, gens)
    fmt_table("R[A][B] = PPL ratio vs own text (diag = 1)", R, gens, fmt="{:.3f}", diag_zero=False)
    fmt_table("S[A][B] = symmetrized (dH_AB + dH_BA)/2", S, gens)

    # diversity ranking: how predictable is each generator's text to the OTHER models
    print("\n=== Generator diversity (mean over other evaluators of dH[A][B]) ===")
    print("positive dH -> text surprises others = contributes diversity")
    print("negative dH -> text MORE predictable than evaluators' own = redundant (degenerate)")
    scores = {}
    for g in gens:
        vals = [dH[e][g] for e in gens if e != g and g in dH[e]]
        if vals:
            m = sum(vals) / len(vals)
            scores[g] = m
            print(f"  {g[:28]:28s} mean dH = {m:+.3f} nats")
    print("  -> ranking (most diverse first): " +
          " > ".join(g for g, _ in sorted(scores.items(), key=lambda kv: -kv[1])))
    # degeneracy flags if cache provided
    cache_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if cache_path and Path(cache_path).exists():
        cache = json.load(open(cache_path))
        print("\n=== Lexical diversity of generations (distinct-2 bigram ratio; <0.3 ~ degenerate) ===")
        for g in gens:
            d = lexical_diversity(cache, g)
            if d is not None:
                print(f"  {g[:28]:28s} {d:.3f}")


if __name__ == "__main__":
    main()