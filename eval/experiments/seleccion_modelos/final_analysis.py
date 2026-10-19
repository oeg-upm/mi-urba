#!/usr/bin/env python3
"""Análisis consolidado del experimento de similaridad entre generadores:
correlaciones triangulares de las 4 métricas, rigidity vs distinctiveness,
figuras finales y números para el .tex.
"""
import json
import numpy as np
from pathlib import Path

DIV = Path(__file__).resolve().parents[3] / "eval" / "results" / "diversity"


def spear(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def load():
    xpp = json.load(open(DIV / "cross_perplexity_gateway_full.json"))
    emb = json.load(open(DIV / "embedding_similarity_7j.json"))
    sty = json.load(open(DIV / "stylometry_analysis.json"))
    rig = json.load(open(DIV / "rigidity_probe.json"))
    return xpp, emb, sty, rig


def main():
    xpp, emb, sty, rig = load()

    def hasc(c):
        return isinstance(c, dict) and "mean_logprob" in c

    xrows = sorted(k for k, v in xpp.items() if any(hasc(c) for c in v.values()))
    Hx = {a: {b: -xpp[a][b]["mean_logprob"] for b in xpp[a] if hasc(xpp[a][b])} for a in xrows}

    # KLD-J matrix (18 evaluadores-generadores con diagonal)
    xtags = [t for t in xrows if t in Hx.get(t, {})]
    n = len(xtags)
    D_kld = np.full((n, n), np.nan)
    xi = {t: i for i, t in enumerate(xtags)}
    for a in xtags:
        for b in xtags:
            if a == b:
                continue
            if a in Hx[b] and b in Hx[a] and b in Hx[b] and a in Hx[a]:
                D_kld[xi[a], xi[b]] = (Hx[a][b] - Hx[b][b]) + (Hx[b][a] - Hx[a][a])
            else:
                D_kld[xi[a], xi[b]] = (Hx[a][b] - Hx[b][b]) if (b in Hx.get(a, {}) and b in Hx[b]) else np.nan
    # simetrizar media de lo disponible (para comparar con matrices simétricas)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(D_kld[i, j]) or np.isnan(D_kld[j, i]):
                v = D_kld[i, j] if not np.isnan(D_kld[i, j]) else D_kld[j, i]
                D_kld[i, j] = D_kld[j, i] = v

    etags = emb["fused"]["tags"]
    D_emb = np.array(emb["fused"]["dist"])
    tags_s = sty["tags"]
    D_tri = np.array([[sty["e1_delta"][a][b] for b in tags_s] for a in tags_s])
    D_pos = np.array([[sty["e2_delta"][a][b] for b in tags_s] for a in tags_s])

    mats = [("KLD", D_kld, xtags), ("Emb7", D_emb, etags),
            ("Tri", D_tri, tags_s), ("POS", D_pos, tags_s)]

    def tri_corr(D1, t1, D2, t2):
        i1 = {t: i for i, t in enumerate(t1)}
        i2 = {t: i for i, t in enumerate(t2)}
        c = [t for t in t1 if t in i2]
        x, y = [], []
        for i, a in enumerate(c):
            for b in c[i + 1:]:
                x.append(D1[i1[a], i1[b]])
                y.append(D2[i2[a], i2[b]])
        x2 = [a for a, b in zip(x, y) if a == a and b == b]
        y2 = [b for a, b in zip(x, y) if a == a and b == b]
        return spear(x2, y2), len(x2)

    names = ["KLD", "Emb7", "Tri", "POS"]
    M = np.full((4, 4), np.nan)
    print("=== Correlaciones triangulares ===")
    for i in range(4):
        for j in range(i + 1, 4):
            rho, cnt = tri_corr(mats[i][1], mats[i][2], mats[j][1], mats[j][2])
            M[i, j] = M[j, i] = rho
            print(f"  {names[i]:5s} ~ {names[j]:5s}: rho={rho:+.3f} (n={cnt})")

    # distinctiveness embedding (mean KL-like: mean 1-cos hacia otros) y rigidity
    tri_idx = np.triu_indices(len(etags), 1)
    # distinctiveness from fused distance: mean distance to others
    D_emb0 = D_emb.copy()
    np.fill_diagonal(D_emb0, np.nan)
    dist_emb = np.nanmean(D_emb0, axis=1)
    ei = {t: i for i, t in enumerate(etags)}
    # distinctiveness from tri
    D_tri0 = D_tri.copy(); np.fill_diagonal(D_tri0, np.nan)
    dist_tri = np.nanmean(D_tri0, axis=1)
    si = {t: i for i, t in enumerate(tags_s)}

    chrf = {t: v["chrf"] for t, v in rig["results"].items()}
    common = [t for t in etags if t in chrf and t in si]
    print("\n=== Rigidity vs distinctiveness ===")
    d_e = [dist_emb[ei[t]] for t in common]
    r_ = [chrf[t] for t in common]
    rho_re = spear(r_, d_e)
    print(f"  chrf vs emb-distinctiveness: rho={rho_re:+.3f} (n={len(common)})")
    d_t = [dist_tri[si[t]] for t in common]
    rho_rt = spear(r_, d_t)
    print(f"  chrf vs tri-distinctiveness: rho={rho_rt:+.3f}")

    # figuras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig 1: heatmap de correlaciones
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    Mf = M.copy()
    im = ax.imshow(Mf, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(names, fontsize=11); ax.set_yticklabels(names, fontsize=11)
    for i in range(4):
        for j in range(4):
            if not np.isnan(Mf[i, j]):
                ax.text(j, i, f"{Mf[i,j]:+.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="black" if 0.3 < abs(Mf[i,j]) < 0.9 else "white")
    plt.colorbar(im, label="Spearman rho")
    ax.set_title("Convergencia de las 4 métricas de similaridad\n(Spearman sobre pares comunes)")
    plt.tight_layout()
    plt.savefig(DIV / "fig_correlations_4metrics.png", dpi=140)
    print("\nPNG: fig_correlations_4metrics.png")

    # fig 2: rigidity vs distinctiveness scatter
    fig, ax = plt.subplots(figsize=(11, 9))
    for t in common:
        ax.scatter(chrf[t], dist_emb[ei[t]], s=60, alpha=0.8,
                   color="#2a72a8", edgecolor="white", zorder=3)
        nm = t.replace("-xppl", "").replace("-instruct", "").replace("-it", "")[:15]
        ax.annotate(nm, (chrf[t], dist_emb[ei[t]]), xytext=(5, 4),
                    textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Rigidity probe: CHRF(texto original, reescritura) — alto = estilo invasivo")
    ax.set_ylabel("Distinctiveness embedding (media dist. fusionada)")
    ax.set_title(f"Rigidez estilística vs. distintividad semántica (ρ={rho_re:+.2f})")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(DIV / "fig_rigidity_vs_distinctiveness.png", dpi=140)
    print("PNG: fig_rigidity_vs_distinctiveness.png")

    # guardar números para el tex
    out = {
        "corr_matrix": M.tolist(), "names": names,
        "rho_re": rho_re, "rho_rt": rho_rt,
        "n_generators": len(tags_s), "n_eval_rows": n,
        "rho_e1_emb": sty.get("e1_rho_vs_embedding"),
        "rho_e1_e2": sty.get("e1_vs_e2_rho"),
        "rigidity_top": sorted(chrf.items(), key=lambda kv: -kv[1])[:6],
        "rigidity_low": sorted(chrf.items(), key=lambda kv: kv[1])[:6],
        "chrf_stats": [float(np.mean(list(chrf.values()))), float(np.std(list(chrf.values()))),
                       float(np.min(list(chrf.values()))), float(np.max(list(chrf.values())))],
    }
    json.dump(out, open(DIV / "final_numbers.json", "w"), ensure_ascii=False, indent=1)
    print("JSON: final_numbers.json")


if __name__ == "__main__":
    main()