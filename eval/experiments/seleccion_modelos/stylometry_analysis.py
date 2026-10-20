#!/usr/bin/env python3
"""Stylometric fingerprinting of LLM generators (E1 trigram-delta, E2 POS/dep profiles).

E1: trigram-cosine delta clustering (Salnikov & Bonch-Osmolovskaya 2025):
    top-1000 char trigrams, z-scored relative frequencies, cosine distance.
E2: POS + dependency profiles per generator (Rosenfeld & Lazebnik 2024 style):
    spaCy es profile vectors, cosine distance, top discriminative POS.

Outputs eval/results/diversity/stylometry_analysis.json + correlations with
the embedding-fused and KLD distance matrices.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DIVERSITY_DIR = ROOT / "eval" / "results" / "diversity"


def load_texts(gen_cache):
    cache = json.load(open(gen_cache))
    entries, texts = [], []
    for tag, m in cache.items():
        for rid, d in m.items():
            c = (d.get("content") or "").strip()
            if c:
                texts.append(c[:1200])
                entries.append(tag)
    tags = sorted(set(entries))
    return texts, entries, tags


def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def delta_matrix(X, metric="cosine"):
    Z = zscore(X)
    if metric == "cosine":
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        D = 1.0 - Zn @ Zn.T
    else:
        D = np.abs(Z[:, None, :] - Z[None, :, :]).mean(axis=2)
    np.fill_diagonal(D, 0.0)
    return D


def agglomerative(D, k):
    cl = [[i] for i in range(len(D))]
    while len(cl) > k:
        b = (1e18, 0, 1)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                d = np.mean([D[p, q] for p in cl[i] for q in cl[j]])
                if d < b[0]:
                    b = (d, i, j)
        _, i, j = b
        cl[i] += cl[j]
        cl.pop(j)
    return cl


def mds2(D):
    S = -0.5 * D ** 2
    S -= S.mean(0, keepdims=True)
    S -= S.mean(1, keepdims=True)
    w, V = np.linalg.eigh(S)
    idx = np.argsort(w)[::-1]
    Y = V[:, idx[:2]] * np.sqrt(np.clip(w[idx[:2]], 0, None))
    return Y, w[idx[:2]] / w.sum()


def spear(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def pair_list(D, tags):
    return [(D[i, j], tags[i], tags[j])
            for i in range(len(tags)) for j in range(i + 1, len(tags))]


def e1_trigram_delta(texts, tags, out):
    print("\n=== E1: trigram-cosine delta ===", flush=True)
    counters = []
    for t in texts:
        s = re.sub(r"\s+", " ", t.lower())
        tris = [s[i:i + 3] for i in range(len(s) - 2)]
        tris = [g for g in tris if g.isalnum()]
        counters.append(Counter(tris))
    df = Counter()
    for c in counters:
        df.update(c.keys())
    top = [g for g, _ in df.most_common(1000)]
    ti = {g: i for i, g in enumerate(top)}
    X = np.zeros((len(texts), len(top)), dtype=np.float32)
    for i, c in enumerate(counters):
        tot = sum(c.values()) or 1
        for g, v in c.items():
            j = ti.get(g)
            if j is not None:
                X[i, j] = v / tot
    # textos por generador
    rows_by_gen = [[i for i, t in enumerate(tags) if False] for t in tags] if False else None
    gen_rows = {}
    for i, (t, txt) in enumerate(zip(texts, texts)):
        pass
    # per-generator mean profile (entries passed separately)
    return X, top


def build_gen_profiles(texts, entries, tags, top):
    ti = {g: i for i, g in enumerate(top)}
    counters = []
    for t in texts:
        s = re.sub(r"\s+", " ", t.lower())
        tris = [s[i:i + 3] for i in range(len(s) - 2)]
        tris = [g for g in tris if g.isalnum()]
        counters.append(Counter(tris))
    G = np.zeros((len(tags), len(top)), dtype=np.float32)
    counts = [0] * len(tags)
    gi = {t: i for i, t in enumerate(tags)}
    for c, e in zip(counters, entries):
        k = gi[e]
        tot = sum(c.values()) or 1
        G[k] += [c.get(g, 0) / tot for g in top]
        counts[k] += 1
    G /= np.array(counts)[:, None]
    return G


def e2_pos_profiles(texts, entries, tags, out):
    print("\n=== E2: POS/dependency profiles ===", flush=True)
    try:
        import spacy
        nlp = spacy.load("es_core_news_sm")
    except Exception as e:
        print("spaCy es no disponible:", e)
        return None, None
    gi = {t: i for i, t in enumerate(tags)}
    pos_c = [Counter() for _ in tags]
    dep_c = [Counter() for _ in tags]
    for t, txt in zip(entries, texts):
        doc = nlp(txt[:2000])
        k = gi[t]
        pos_c[k].update(tok.pos_ for tok in doc if not tok.is_space)
        dep_c[k].update(tok.dep_ for tok in doc)
    pos_keys = sorted({k for c in pos_c for k in c})
    dep_keys = sorted({k for c in dep_c for k in c})
    def vec(c, keys):
        tot = sum(c.values()) or 1
        return np.array([c.get(k, 0) / tot for k in keys], dtype=np.float32)
    G = np.vstack([np.concatenate([vec(pos_c[i], pos_keys), vec(dep_c[i], dep_keys)])
                   for i in range(len(tags))])
    print(f"E2: perfil {G.shape} ({len(pos_keys)} POS + {len(dep_keys)} dep)")
    # per-model discriminative POS
    profs = np.vstack([vec(pos_c[i], pos_keys) for i in range(len(tags))])
    mean_p = profs.mean(0)
    out_pos = {}
    for i, t in enumerate(tags):
        diff = profs[i] - mean_p
        top = [(pos_keys[j], round(float(diff[j]), 4)) for j in np.argsort(diff)[-4:][::-1]]
        out_pos[t.replace('-xppl','')] = top
    out["e2_pos_overuse"] = out_pos
    return G, (pos_keys, dep_keys)


def corr_with(D, tags, other_path, key):
    try:
        o = json.load(open(other_path))
        otags = o[key]["tags"]
        O = np.array(o[key]["dist"] if "dist" in o[key] else o[key]["cos"])
        oi = {t: i for i, t in enumerate(otags)}
        common = [t for t in tags if t in oi]
        xi = {t: i for i, t in enumerate(tags)}
        d1 = [D[xi[a], xi[b]] for i, a in enumerate(common) for b in common[i+1:]]
        d2 = [O[oi[a], oi[b]] for i, a in enumerate(common) for b in common[i+1:]]
        if len(d1) >= 20:
            rho = spear(d1, d2)
            print(f"  vs {key} (n={len(d1)}): rho={rho:+.3f}")
            return rho
    except Exception as e:
        print(f"  ({key} skip: {e})")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-cache", default=str(DIVERSITY_DIR / "xppl_gen_cache_gateway_full.json"))
    ap.add_argument("--output", default=str(DIVERSITY_DIR / "stylometry_analysis.json"))
    args = ap.parse_args()
    texts, entries, tags = load_texts(args.gen_cache)
    print(f"texts: {len(texts)} | generators: {len(tags)}")
    out = {"tags": tags, "n_texts": len(texts)}

    # E1: per-generator trigram profiles
    gi = {t: i for i, t in enumerate(tags)}
    tri_c = [Counter() for _ in tags]
    n_g = [0] * len(tags)
    for txt, e in zip(texts, entries):
        k = gi[e]
        s = re.sub(r"\s+", " ", txt.lower())
        tris = [s[i:i+3] for i in range(len(s) - 2)]
        tri_c[k].update(g for g in tris if g.isalnum())
        n_g[k] += 1
    df = Counter()
    for c in tri_c:
        df.update(c.keys())
    top = [g for g, _ in df.most_common(1000)]
    ti = {g: i for i, g in enumerate(top)}
    G = np.zeros((len(tags), len(top)), dtype=np.float32)
    for k, c in enumerate(tri_c):
        tot = sum(c.values()) or 1
        for g, v in c.items():
            j = ti.get(g)
            if j is not None:
                G[k, j] = v / tot
    print(f"E1: per-generator trigram profile {G.shape}")
    D1 = delta_matrix(G, "cosine")
    Y, var = mds2(D1)
    print(f"E1: var explicada {var[0]*100:.0f}%+{var[1]*100:.0f}%")
    pairs = pair_list(D1, tags)
    pairs.sort()
    print("E1 parejas más similares:")
    for d, a, b in pairs[:10]:
        print(f"  {d:5.3f}  {a.replace('-xppl','')[:30]:30s} ~ {b.replace('-xppl','')[:30]}")
    out["e1_top_similar"] = [(round(float(d), 4), a.replace('-xppl',''), b.replace('-xppl','')) for d, a, b in pairs[:15]]
    out["e1_top_distant"] = [(round(float(d), 4), a.replace('-xppl',''), b.replace('-xppl','')) for d, a, b in pairs[-10:]]
    out["e1_delta"] = {a: {b: round(float(D1[i, j]), 4) for j, b in enumerate(tags)} for i, a in enumerate(tags)}
    out["e1_mds"] = {t: [round(float(Y[i, 0]), 3), round(float(Y[i, 1]), 3)] for i, t in enumerate(tags)}

    # correlaciones
    print("\n=== correlaciones de E1 con otras métricas ===")
    out["e1_rho_vs_embedding"] = corr_with(D1, tags, DIVERSITY_DIR / "embedding_similarity_7j.json", "fused")

    # E2
    G2, keys = e2_pos_profiles(texts, entries, tags, out)
    if G2 is not None:
        D2 = delta_matrix(G2, "cosine")
        pairs2 = pair_list(D2, tags)
        pairs2.sort()
        print("E2 parejas más similares (POS+dep):")
        for d, a, b in pairs2[:10]:
            print(f"  {d:5.3f}  {a.replace('-xppl','')[:30]:30s} ~ {b.replace('-xppl','')[:30]}")
        out["e2_top_similar"] = [(round(float(d), 4), a.replace('-xppl',''), b.replace('-xppl','')) for d, a, b in pairs2[:15]]
        out["e2_delta"] = {a: {b: round(float(D2[i, j]), 4) for j, b in enumerate(tags)} for i, a in enumerate(tags)}
        r12 = spear([D1[i, j] for i in range(len(tags)) for j in range(i+1, len(tags))],
                    [D2[i, j] for i in range(len(tags)) for j in range(i+1, len(tags))])
        print(f"\n=== rho(E1-trigram, E2-POS) = {r12:+.3f} ===")
        out["e1_vs_e2_rho"] = r12

    # PNG dendrogram-ish: MDS E1
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 11))
        ax.scatter(Y[:, 0], Y[:, 1], s=120, color="#2a72a8", edgecolor="white", zorder=3)
        for i, t in enumerate(tags):
            ax.annotate(t.replace("-xppl", "").replace("-instruct", "").replace("-it", "")[:18],
                        (Y[i, 0], Y[i, 1]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_title(f"Trigram-cosine delta MDS — {len(tags)} generadores (estilometría E1)")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(DIVERSITY_DIR / "stylometry_mds_e1.png", dpi=140)
        print("PNG: stylometry_mds_e1.png")
    except Exception as e:
        print("(png skip)", e)

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\nJSON: {args.output}")


if __name__ == "__main__":
    main()