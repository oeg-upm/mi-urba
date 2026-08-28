#!/usr/bin/env python3
"""Normalized analysis of a cross-perplexity matrix.

Raw cross-PPL mixes three things: (a) true distributional distance between
generator and evaluator, (b) the evaluator's confidence/calibration (its
diagonal level), and (c) the intrinsic entropy of each generator's texts
(degenerate repetition is trivially predictable for EVERYONE).

This script derives metrics that separate those effects:

  H[A][B]        cross-entropy in nats/token (from mean_logprob, no exp)
  KLD[A][B]      = H(A,text_B) - H(B,text_B)  sample estimate of
                   KL(P_B || P_A): extra nats/token to encode B's output
                   under A instead of under B itself  <-- model-vs-model
  J[A][B]        = KLD[A][B] + KLD[B][A]      Jeffreys divergence:
                   symmetric, >=0 in expectation, THE pairwise distance
  dH[A][B]       = H(A,text_B) - H(A,text_A)   diagnostic only: excess
                   surprise vs evaluator's own text (calibration-flavored)
  S[A][B]        = (dH[A][B] + dH[B][A]) / 2      symmetrized (signed)

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


def kld_matrix(H):
    """KLD[A][B] = H(A,text_B) - H(B,text_B) ~= KL(P_B || P_A).
    Baseline is the GENERATOR's own-text entropy (the entropy of its output
    distribution), not the evaluator's self-confidence.
    """
    return {e: {g: H[e][g] - H[g][g] for g in H[e]} for e in H}


def jeffreys(KLD):
    """J[A][B] = KL(P_B||P_A) + KL(P_A||P_B); symmetric, >= 0 in expectation."""
    J = {}
    for a in KLD:
        J[a] = {}
        for b in KLD:
            if a == b:
                J[a][b] = 0.0
            else:
                J[a][b] = KLD[a][b] + KLD[b][a]
    return J


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
    KLD = kld_matrix(H)
    J = jeffreys(KLD)
    S = symmetrized(dH)

    fmt_table("KLD[A][B] ~= KL(P_B||P_A): extra nats to encode B's text under A (directed)",
              KLD, gens)
    fmt_table("J[A][B] = Jeffreys divergence (symmetric, primary model-to-model distance)",
              J, gens)
    # nearest neighbors per model
    print("\n=== Nearest neighbor (min Jeffreys) per model ===")
    for a in gens:
        others = [(J[a][b], b) for b in gens if b != a]
        if others:
            j, b = min(others)
            print(f"  {a[:28]:28s} -> {b[:28]:28s} (J={j:.3f} nats)")

    fmt_table("diagnostic: dH[A][B] = H(A,text_B)-H(A,text_A) (calibration-flavored)",
              dH, gens)
    fmt_table("diagnostic: S[A][B] = symmetrized dH (signed)", S, gens)

    # diversity ranking: how predictable is each generator's text to the OTHER models
    # model-vs-model: how far is each model's DISTRIBUTION from the panel's?
    print("\n=== Model distinctiveness (mean over other evaluators of KLD[A][B] = KL(P_B||P_A)) ===")
    print("high  -> B's output distribution is far from every other model")
    print("low   -> B's output could have come from any of the others (redundant)")
    scores = {}
    for g in gens:
        vals = [KLD[e][g] for e in gens if e != g and g in KLD[e]]
        if vals:
            m = sum(vals) / len(vals)
            scores[g] = m
            print(f"  {g[:28]:28s} mean KL(P_g||P_others) = {m:.3f} nats")
    print("  -> ranking (most distinctive first): " +
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