#!/usr/bin/env python3
"""E3: style-rigidity probe (SSLA 2026 adapted, local rewriter).

Rewrite each generated text with a common LLM (gemma-2-2b-it via local
llama.cpp, prompt: paraphrase keeping content, change nothing else) and measure
the style drift between original and rewrite at 3 levels:
  - surface:   CHRF++ approximation (char trigram cosine)
  - embedding: cosine in the fused-judge space (7 judges, if servers up)
Generators whose texts survive the rewrite with minimal drift = "rígidos"
(fuerte fingerprint); altos drift = estilos menos marcados.

Output: eval/results/diversity/rigidity_probe.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
DIVERSITY_DIR = ROOT / "eval" / "results" / "diversity"
REWRITER = "http://127.0.0.1:18890"
EMBED_JUDGES = {
    "qwen3-emb-4b": (18871, lambda t: t),
    "nemotron-emb-8b": (18872, lambda t: f"passage: {t}"),
    "gemma-emb-300m": (18873, lambda t: f"title: none | text: {t}"),
    "MiniLM-paraphrase": (18874, lambda t: t),
}


def rewrite(text, model_id):
    r = requests.post(f"{REWRITER}/v1/chat/completions", json={
        "model": model_id,
        "messages": [
            {"role": "user", "content":
             "Parafrasea el siguiente texto con tus propias palabras, conservando "
             "exactamente el mismo contenido y significado, pero cambiando la redacción:\n\n" + text}
        ], "temperature": 0.2, "max_tokens": 1200,
    }, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def trigram_profile(text):
    s = re.sub(r"\s+", " ", text.lower())
    tris = [s[i:i + 3] for i in range(len(s) - 2)]
    c = Counter(g for g in tris if g.isalnum())
    tot = sum(c.values()) or 1
    return {k: v / tot for k, v in c.items()}


def chrf_cos(t1, t2):
    p1, p2 = trigram_profile(t1), trigram_profile(t2)
    common = set(p1) | set(p2)
    a = np.array([p1.get(g, 0) for g in common])
    b = np.array([p2.get(g, 0) for g in common])
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def embed_cos(port, fmt, t1, t2):
    r = requests.post(f"http://localhost:{port}/v1/embeddings",
                      json={"input": [fmt(t1), fmt(t2)]}, timeout=300)
    r.raise_for_status()
    e = np.array([d["embedding"] for d in r.json()["data"]])
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return float(e[0] @ e[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-cache", default=str(DIVERSITY_DIR / "xppl_gen_cache_gateway_full.json"))
    ap.add_argument("--n-per-gen", type=int, default=1)
    ap.add_argument("--output", default=str(DIVERSITY_DIR / "rigidity_probe.json"))
    args = ap.parse_args()

    cache = json.load(open(args.gen_cache))
    sample = []
    for tag in sorted(cache):
        taken = 0
        for rid, d in cache[tag].items():
            c = (d.get("content") or "").strip()
            if c and taken < args.n_per_gen:
                sample.append((tag, rid, c[:1000]))
                taken += 1
    print(f"muestras: {len(sample)} textos de {len(set(s[0] for s in sample))} generadores", flush=True)

    out = {"rewriter": "gemma-2-2b-it", "results": {}}
    for i, (tag, rid, txt) in enumerate(sample):
        try:
            rw = rewrite(txt, "gemma-2-2b-it-xppl")
            chrf = chrf_cos(txt, rw)
            embs = {}
            for jname, (port, fmt) in EMBED_JUDGES.items():
                try:
                    embs[jname] = round(embed_cos(port, fmt, txt, rw), 4)
                except Exception:
                    pass
            out["results"][tag] = {
                "chrf": round(chrf, 4),
                "emb": embs,
                "orig_len": len(txt),
                "rewritten_len": len(rw),
            }
            print(f"[{i+1}/{len(sample)}] {tag}: chrf={chrf:.3f} emb={embs}", flush=True)
            with open(args.output, "w") as f:
                json.dump(out, f, ensure_ascii=False)
        except Exception as e:
            print(f"[{i+1}] {tag} FALLO: {e}", flush=True)
    # summary
    chrf_by = {t: v["chrf"] for t, v in out["results"].items()}
    emb_by = {t: np.mean([v for v in r["emb"].values()]) for t, r in out["results"].items() if r.get("emb")}
    print("\n=== TOP rigidez (menos drift = estilo más rígido/invasivo) ===")
    for t, v in sorted(chrf.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  chrf={v:.3f}  {t.replace('-xppl','')}")
    print("\n=== TOP flexibilidad (más drift) ===")
    for t, v in sorted(chrf.items(), key=lambda kv: kv[1])[:8]:
        print(f"  chrf={v:.3f}  {t.replace('-xppl','')}")
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()