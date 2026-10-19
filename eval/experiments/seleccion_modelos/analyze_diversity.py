#!/usr/bin/env python3
"""Build similarity matrix between generator models based on their complaint texts.

Uses pure-Python TF-IDF + cosine similarity (no external deps).
For each model, computes the mean embedding (TF-IDF vector) across its 20 complaints.
Then pairwise cosine similarity between models → 81x81 matrix.
Also computes per-model quality metrics (text length, diversity, etc).

Input: eval/results/diversity/gen_<model>/complaints.jsonl
Output: eval/results/diversity/similarity_matrix.json
        eval/results/diversity/model_quality.json
"""
import json, math, re, os
from pathlib import Path
from collections import Counter

DIVERSITY_DIR = Path(__file__).resolve().parents[3] / "eval" / "results" / "diversity"

def tokenize(text):
    """Simple Spanish tokenizer: lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r'[^\wáéíóúñüç]', ' ', text)
    return [w for w in text.split() if len(w) > 2]

def load_model_texts():
    """Load all complaints grouped by model. Returns {model_tag: [texts]}."""
    models = {}
    for d in sorted(os.listdir(DIVERSITY_DIR)):
        if not d.startswith("gen_"): continue
        p = DIVERSITY_DIR / d / "complaints.jsonl"
        if not p.exists(): continue
        tag = d.replace("gen_","")
        texts = []
        for line in open(p):
            if line.strip():
                rec = json.loads(line)
                texts.append(rec.get("user_text",""))
        if texts:
            models[tag] = texts
    return models

def compute_tfidf_vectors(models):
    """Compute TF-IDF vector for each model (aggregated across its complaints)."""
    # Build vocabulary from all texts
    docs = {}  # model -> Counter of term freqs
    df = Counter()  # document frequency per term
    for tag, texts in models.items():
        tf = Counter()
        for text in texts:
            for w in tokenize(text):
                tf[w] += 1
        docs[tag] = tf
        for term in tf:
            df[term] += 1
    N = len(docs)
    idf = {term: math.log(N / dfi) for term, dfi in df.items()}
    # TF-IDF vector per model
    vectors = {}
    for tag, tf in docs.items():
        vec = {}
        for term, count in tf.items():
            vec[term] = count * idf.get(term, 0)
        # L2 normalize
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        vec = {k: v/norm for k, v in vec.items()}
        vectors[tag] = vec
    return vectors

def cosine_sim(v1, v2):
    """Cosine similarity between two sparse vectors (already L2-normalized)."""
    # dot product over shared keys
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(v * v2.get(k, 0) for k, v in v1.items())

def compute_quality(models):
    """Per-model quality metrics."""
    quality = {}
    for tag, texts in models.items():
        lengths = [len(t) for t in texts if t]
        # type-token ratio (lexical diversity)
        all_tokens = []
        for t in texts:
            all_tokens.extend(tokenize(t))
        ttr = len(set(all_tokens)) / max(len(all_tokens), 1) if all_tokens else 0
        # fraction of non-empty texts
        nonempty = sum(1 for t in texts if len(t.strip()) > 20)
        quality[tag] = {
            "n_texts": len(texts),
            "nonempty": nonempty,
            "avg_length": sum(lengths)/max(len(lengths),1) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "ttr": ttr,
            "unique_tokens": len(set(all_tokens)),
        }
    return quality

def main():
    models = load_model_texts()
    print(f"Loaded {len(models)} models with texts", flush=True)
    # Filter: only models with >= 15 non-empty texts (quality threshold)
    quality = compute_quality(models)
    valid = {tag for tag, q in quality.items() if q["nonempty"] >= 15}
    print(f"Models with >=15 valid texts: {len(valid)}", flush=True)
    # Filter models dict
    models_valid = {tag: texts for tag, texts in models.items() if tag in valid}
    vectors = compute_tfidf_vectors(models_valid)
    tags = sorted(models_valid.keys())
    # Similarity matrix
    sim = {}
    for i, t1 in enumerate(tags):
        sim[t1] = {}
        for t2 in tags:
            sim[t1][t2] = cosine_sim(vectors[t1], vectors[t2])
    # Mean similarity per model (excluding self)
    mean_sim = {}
    for t in tags:
        others = [sim[t][t2] for t2 in tags if t2 != t]
        mean_sim[t] = sum(others)/len(others) if others else 0
    # Sort by mean similarity (ascending = most distinct first)
    ranked = sorted(tags, key=lambda t: mean_sim[t])
    print("\n=== Models ranked by distinctiveness (most distinct first) ===", flush=True)
    for i, t in enumerate(ranked):
        q = quality[t]
        print(f"  {i+1:3d}. {t:30s} mean_sim={mean_sim[t]:.3f} avg_len={q['avg_length']:.0f} ttr={q['ttr']:.3f} n={q['nonempty']}", flush=True)
    # Save
    out = {
        "similarity": sim,
        "mean_similarity": mean_sim,
        "ranked": ranked,
        "quality": quality,
    }
    with open(DIVERSITY_DIR / "similarity_matrix.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(DIVERSITY_DIR / "model_quality.json", "w") as f:
        json.dump(quality, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {DIVERSITY_DIR}/similarity_matrix.json", flush=True)
    print("=== SIMILARITY DONE ===", flush=True)

if __name__ == "__main__":
    main()