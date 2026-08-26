#!/usr/bin/env python3
"""Select the 10 most distinct models with a minimum quality bar.

Quality criteria (applied to the model's 20 complaints):
- avg_length between 100 and 1500 chars (not telegraphic, not rambling)
- ttr (type-token ratio) between 0.15 and 0.60 (not repetitive, not chaotic)
- no garbled text: fraction of non-ASCII non-Spanish chars < 5%
- mentions the problem domain (fuga/agua/tubería/garaje) in >=60% of texts

Then pick the 10 with LOWEST mean_similarity (most distinct) that pass quality.
To ensure diversity, after picking a model, penalize others very similar to it
(greedy farthest-first selection).
"""
import json, re, os
from pathlib import Path

DIVERSITY_DIR = Path(__file__).resolve().parent.parent / "eval" / "results" / "diversity"

def load_quality_and_texts():
    sim_data = json.load(open(DIVERSITY_DIR / "similarity_matrix.json"))
    quality = sim_data["quality"]
    sim = sim_data["similarity"]
    texts = {}
    for tag in quality:
        p = DIVERSITY_DIR / f"gen_{tag}" / "complaints.jsonl"
        if p.exists():
            texts[tag] = [json.loads(l).get("user_text","") for l in open(p) if l.strip()]
    return sim_data, quality, sim, texts

def quality_ok(tag, q, texts):
    # Length
    if q["avg_length"] < 100 or q["avg_length"] > 1500: return False, "length"
    # TTR
    if q["ttr"] < 0.15 or q["ttr"] > 0.60: return False, "ttr"
    # Garbled: count non-ASCII non-Spanish chars (allow áéíóúñü¿¡)
    spanish_extra = set("áéíóúÁÉÍÓÚñÑüÜ¿¡")
    garbled_count = 0; total_chars = 0
    for t in texts.get(tag, []):
        for ch in t:
            total_chars += 1
            o = ord(ch)
            if o > 127 and ch not in spanish_extra:
                # non-latin (chinese, korean, cyrillic, etc.)
                if o > 0x2E80:  # CJK and beyond
                    garbled_count += 1
    if total_chars and garbled_count / total_chars > 0.02: return False, "garbled"
    # Domain mentions
    domain_words = ["fuga","agua","tuber","garaj","fontaner","goter","inund","suelo","pared","tubería","plaza"]
    if texts.get(tag):
        hits = sum(1 for t in texts[tag] if any(w in t.lower() for w in domain_words))
        if hits / len(texts[tag]) < 0.60: return False, "domain"
    return True, "ok"

def main():
    sim_data, quality, sim, texts = load_quality_and_texts()
    print("=== Quality filtering ===", flush=True)
    candidates = []
    for tag in quality:
        ok, reason = quality_ok(tag, quality[tag], texts)
        mark = "OK " if ok else "DROP"
        print(f"  {mark} {tag:30s} len={quality[tag]['avg_length']:.0f} ttr={quality[tag]['ttr']:.2f} ({reason})", flush=True)
        if ok:
            candidates.append(tag)
    print(f"\n{len(candidates)} models pass quality filter", flush=True)

    # Greedy farthest-first selection: pick the most distinct, then iteratively
    # pick the one farthest from all already-picked (maximize min-distance).
    mean_sim = sim_data["mean_similarity"]
    # Start with the most distinct overall (lowest mean_sim)
    selected = [min(candidates, key=lambda t: mean_sim[t])]
    remaining = [t for t in candidates if t not in selected]
    while len(selected) < 10 and remaining:
        # for each remaining, compute min similarity to any selected (we want LOW)
        # pick the one with the LOWEST max-similarity to selected = most distinct from all
        best = None; best_score = -1
        for t in remaining:
            # max similarity to any selected (want this LOW = far from all)
            max_sim = max(sim[t][s] for s in selected)
            # we want the model with the smallest max_sim (farthest from cluster)
            if max_sim > best_score or best is None:
                # actually we want MIN max_sim, so track the minimum
                pass
        # redo: pick min of max-sim
        best = min(remaining, key=lambda t: max(sim[t][s] for s in selected))
        selected.append(best)
        remaining.remove(best)

    print(f"\n=== Selected 10 most distinct (quality-filtered) ===", flush=True)
    for i, tag in enumerate(selected, 1):
        q = quality[tag]
        print(f"  {i:2d}. {tag:30s} mean_sim={mean_sim[tag]:.3f} len={q['avg_length']:.0f} ttr={q['ttr']:.2f}", flush=True)

    # Save selection
    out = {
        "selected": ["local_llama/" + t for t in selected],
        "selected_tags": selected,
    }
    with open(DIVERSITY_DIR / "selected_10.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {DIVERSITY_DIR}/selected_10.json", flush=True)
    print("=== SELECTION DONE ===", flush=True)

if __name__ == "__main__":
    main()