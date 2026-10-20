#!/usr/bin/env python3
"""Embedding-based similarity between generator models, judged by 3 embedding models.

Reads the generated complaint texts from the big cross-perplexity run's gen cache,
embeds each text with each judge (local llama-server /v1/embeddings), builds
per-judge generator-centroid cosine matrices, and a fused matrix (z-scored
1-cos averaged across judges). Outputs JSON + MDS/clustering prints.

Judges (local llama.cpp servers, embedding mode):
  qwen3-emb-4b   : Qwen3-Embedding-4B-Q6_K       last pooling, raw text (docs)
  nemotron-emb-8b: llama-embed-nemotron-8b-Q8_0  last pooling, "passage: " prefix
  gemma-emb-300m : embeddinggemma-300m-qat-Q8_0  mean pooling, "title: none | text: " prefix
"""
import argparse
import json
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[3]
DIVERSITY_DIR = ROOT / "eval" / "results" / "diversity"

JUDGES = {
    "qwen3-emb-4b":   {"port": 18871, "fmt": lambda t: t},
    "nemotron-emb-8b": {"port": 18872, "fmt": lambda t: f"passage: {t}"},
    "gemma-emb-300m": {"port": 18873, "fmt": lambda t: f"title: none | text: {t}"},
    "MiniLM-paraphrase": {"port": 18874, "fmt": lambda t: t},
    "LaBSE": {"port": 18875, "fmt": lambda t: t},
    "bert-base-m-cased": {"port": 18876, "fmt": lambda t: t},
}


def embed_batch(port, texts, timeout=300):
    r = requests.post(f"http://localhost:{port}/v1/embeddings",
                      json={"input": texts}, timeout=timeout)
    if r.status_code != 200:
        # one text poisons the batch: isolate and truncate adaptively
        out = []
        for t in texts:
            tt = t
            for _ in range(6):
                r2 = requests.post(f"http://localhost:{port}/v1/embeddings",
                                   json={"input": [tt]}, timeout=timeout)
                if r2.status_code == 200:
                    break
                tt = tt[: max(64, len(tt) // 2)]
            r2.raise_for_status()
            out.append(r2.json()["data"][0]["embedding"])
        return np.array(out, dtype=np.float32)
    return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)


def embed_all(port, fmt, texts, batch=8):
    out = []
    import time
    t0 = time.time()
    for i in range(0, len(texts), batch):
        out.append(embed_batch(port, [fmt(t) for t in texts[i:i + batch]]))
        if (i // batch) % 8 == 0:
            print(f"    [port {port}] {i + len(texts[i:i+batch])}/{len(texts)} embeds", flush=True)
    E = np.vstack(out)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E


def agglomerative(D, k):
    """Average-linkage agglomerative clustering to k clusters (numpy only)."""
    clusters = [[i] for i in range(len(D))]
    while len(clusters) > k:
        best = (1e18, 0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = np.mean([D[p, q] for p in clusters[i] for q in clusters[j]])
                if d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        clusters[i] += clusters[j]
        clusters.pop(j)
    return clusters


def mds2(D):
    S = -0.5 * D ** 2
    S -= S.mean(0, keepdims=True)
    S -= S.mean(1, keepdims=True)
    w, V = np.linalg.eigh(S)
    idx = np.argsort(w)[::-1]
    Y = V[:, idx[:2]] * np.sqrt(np.clip(w[idx[:2]], 0, None))
    return Y, w[idx[:2]] / w.sum()


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-cache", default=str(DIVERSITY_DIR / "xppl_gen_cache_gateway_full.json"))
    ap.add_argument("--output", default=str(DIVERSITY_DIR / "embedding_similarity.json"))
    ap.add_argument("--png", default=str(DIVERSITY_DIR / "mds_voces_embeddings_7j.png"))
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    cache = json.load(open(args.gen_cache))
    entries, texts = [], []
    for tag, rids in cache.items():
        for rid, d in rids.items():
            c = (d.get("content") or "").strip()
            if c:
                texts.append(c[:1200])
                entries.append(tag)
    tags = sorted(set(entries))
    gen_idx = {t: i for i, t in enumerate(tags)}
    gen_of = np.array([gen_idx[t] for t in entries])
    print(f"texts: {len(texts)} | generators: {len(tags)}")

    results = {"judges": {}, "meta": {"n_texts": len(texts), "n_generators": len(tags)}}
    mats = {}
    for judge, cfg in JUDGES.items():
        try:
            E = embed_all(cfg["port"], cfg["fmt"], texts)
        except Exception as e:
            print(f"[{judge}] FALLO REAL: {type(e).__name__}: {e}")
            raise
        C = np.zeros((len(tags), E.shape[1]), dtype=np.float32)
        for gi, t in enumerate(tags):
            m = gen_of == gi
            C[gi] = E[m].mean(0)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        S = C @ C.T
        mats[judge] = S
        results[judge] = {"tags": tags, "cos": S.tolist()}
        print(f"[{judge}] centroides {C.shape}, sim media off-diag: "
              f"{S[np.triu_indices(len(tags),1)].mean():.3f}")

    # fused distance: mean of per-judge z-scored (1-cos), off-diagonal only
    n = len(tags)
    tri = np.triu_indices(n, 1)
    Z = np.zeros((n, n), dtype=np.float32)
    for judge, S in mats.items():
        d = 1.0 - S
        off = d[tri]
        z = (off - off.mean()) / (off.std() + 1e-9)
        Z[tri] += z
        Z.T[tri] += z
    Dcomb = Z / len(mats)
    np.fill_diagonal(Dcomb, 0.0)
    Dcomb -= Dcomb.min()
    results["fused"] = {"tags": tags, "dist": Dcomb.tolist()}

    # inter-judge agreement (Spearman of off-diagonal cosines)
    print("\n=== acuerdo entre jueces (Spearman, off-diagonal cos) ===")
    js = list(JUDGES)
    for i in range(len(js)):
        for j in range(i + 1, len(js)):
            a = mats[js[i]][tri]; b = mats[js[j]][tri]
            print(f"  {js[i]} ~ {js[j]}: rho={spearman(a, b):+.3f}")

    # agreement with KLD distance (for generators with diagonal in the XPP run)
    try:
        xpp = json.load(open(DIVERSITY_DIR / "cross_perplexity_gateway_full.json"))
        def hasc(c): return isinstance(c, dict) and "mean_logprob" in c
        xrows = sorted(k for k, v in xpp.items() if any(hasc(c) for c in v.values()))
        Hx = {a: {b: -xpp[a][b]["mean_logprob"] for b in xpp[a] if hasc(xpp[a][b])} for a in xrows}
        def hasc(c): return isinstance(c, dict) and "mean_logprob" in c
        common = [t for t in tags if t in xrows and t in Hx.get(t, {})]
        if len(common) >= 10:
            ki, kj = [], []
            ti = {t: i for i, t in enumerate(tags)}
            xi = {t: i for i, t in enumerate(xrows)}
            for i, a in enumerate(common):
                for bb in common[i + 1:]:
                    d_e = Dcomb[ti[a], ti[bb]]
                    ka = Hx[a].get(bb); kb = Hx[bb].get(a)
                    if ka is None or kb is None:
                        continue
                    j = (Hx[a][bb] - Hx[bb][bb]) + (Hx[bb][a] - Hx[a][a])
                    ki.append(Dcomb[ti[a], ti[bb]]); kj.append(j)
            if len(ki) >= 20:
                print(f"\n=== acuerdo embedding-fused vs KLD-Jeffreys (n={len(ki)} pares): "
                      f"rho={spearman(ki, kj):+.3f} ===")
    except Exception as e:
        print(f"(KLD comparison skipped: {e})")

    # clustering + MDS on fused distance
    clusters = agglomerative(Dcomb, args.k)
    print(f"\n=== Clusters K={args.k} (fused embeddings) ===")
    for ci, c in enumerate(sorted(clusters, key=lambda c: -len(c))):
        print(f"  C{ci}: [{', '.join(tags[p].replace('-xppl','') for p in c[:10])}"
              f"{',...' if len(c) > 10 else ''}]")

    S = -0.5 * Dcomb ** 2
    S -= S.mean(0, keepdims=True); S -= S.mean(1, keepdims=True)
    w, V = np.linalg.eigh(S)
    idx = np.argsort(w)[::-1]
    Y = V[:, idx[:2]] * np.sqrt(np.clip(w[idx[:2]], 0, None))
    results["fused"]["mds"] = {tags[i]: [round(float(Y[i,0]),3), round(float(Y[i,1]),3)] for i in range(n)}

    # PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmap = plt.cm.tab10
        fig, ax = plt.subplots(figsize=(15, 11))
        for ci, c in enumerate(clusters):
            pts = Y[c]
            ax.scatter(pts[:,0], pts[:,1], s=120, color=cmap(ci % 10), edgecolor="white", lw=1.2, zorder=3)
            for p in c:
                ax.annotate(tags[p].replace("-xppl",""), (Y[p,0], Y[p,1]),
                            xytext=(5, 4), textcoords="offset points", fontsize=8,
                            color=cmap(ci % 10))
        ax.set_title(f"Fused-embedding voice map — {n} generators, 3 judges (MDS, {w[idx[0]]/w.sum()*100:.0f}%+{w[idx[1]]/w.sum()*100:.0f}% var)")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(args.png, dpi=140)
        print(f"\nPNG: {args.png}")
    except Exception as e:
        print(f"(png skipped: {e})")

    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"\nJSON: {args.output}")


if __name__ == "__main__":
    main()