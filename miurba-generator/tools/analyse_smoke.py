#!/usr/bin/env python3
"""Análisis de modos de fallo del generador y del juez, desglosado por modelo.

Lee las salidas de una campaña de smoke:
  <run>/<modelo>/raw/{complaints,rejected,conversations}.jsonl

Uso:
  python tools/analyse_smoke.py <run_dir> [--out informe.md]
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

FIELD_REASONS = ("category", "subcategory", "urgency", "affected_elements",
                 "risks", "location")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def reason_tokens(detail: str) -> list[str]:
    """Split a detail string into normalised reason tokens."""
    tokens = []
    for chunk in (detail or "").split(", "):
        head = chunk.split(":", 1)[0].strip()
        if head.startswith("urgency["):
            tokens.append("urgency")
        elif head in FIELD_REASONS:
            tokens.append(head)
        elif head.startswith("inventado"):
            tokens.append("inventado")
        elif head.startswith("n_problems"):
            tokens.append("n_problems")
        elif head in ("evidencia", "expresion_citada"):
            tokens.append("_evidence")
    return tokens


def analyse_model(raw_dir: Path) -> dict:
    complaints = load_jsonl(raw_dir / "complaints.jsonl")
    rejected = load_jsonl(raw_dir / "rejected.jsonl")
    convs = load_jsonl(raw_dir / "conversations.jsonl")

    n_records = len(convs) or (len(complaints) + len(rejected))
    accepted = len(complaints)

    attempts = collections.Counter(c["attempts"] for c in convs)
    first_try = attempts.get(1, 0)

    judge_kinds = collections.Counter()
    judge_fail_by_attempt = collections.Counter()
    for c in convs:
        for turn in c["turns"]:
            for diag in (turn.get("judge") or {}).get("_diagnostics") or []:
                judge_kinds[diag["kind"]] += 1
                if diag["kind"] != "ok":
                    judge_fail_by_attempt[diag["attempt"]] += 1

    reject_reasons = collections.Counter()
    reject_examples = collections.defaultdict(list)
    hygiene = collections.Counter()
    for r in rejected:
        detail = r.get("detail") or ""
        if r["reason"] == "leak":
            hygiene[detail.split(";")[0]] += 1
            reject_reasons["hygiene"] += 1
            if len(reject_examples["hygiene"]) < 2:
                reject_examples["hygiene"].append((r["id"], detail[:200]))
            continue
        if r["reason"] == "judge_unavailable":
            reject_reasons["judge_unavailable"] += 1
            continue
        for tok in reason_tokens(detail):
            if tok == "_evidence":
                continue
            reject_reasons[tok] += 1
            if len(reject_examples[tok]) < 2:
                reject_examples[tok].append((r["id"], detail[:260]))

    lengths = [len(c["user_text"]) for c in complaints]
    return {
        "records": n_records,
        "accepted": accepted,
        "acceptance": round(100 * accepted / n_records, 1) if n_records else 0.0,
        "attempts": dict(attempts),
        "first_try": first_try,
        "first_try_pct": round(100 * first_try / n_records, 1) if n_records else 0.0,
        "judge_kinds": dict(judge_kinds),
        "judge_fail_by_attempt": dict(judge_fail_by_attempt),
        "reject_reasons": dict(reject_reasons),
        "hygiene": dict(hygiene),
        "reject_examples": {k: v for k, v in reject_examples.items()},
        "length_mean": round(statistics.mean(lengths), 0) if lengths else 0,
        "length_p10": round(sorted(lengths)[len(lengths) // 10], 0) if lengths else 0,
        "length_p90": round(sorted(lengths)[9 * len(lengths) // 10], 0) if lengths else 0,
        "total_judge_calls": sum(judge_kinds.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    models = sorted(p.name for p in run.iterdir() if (p / "raw").is_dir())
    results = {m: analyse_model(run / m / "raw") for m in models}

    lines = ["# Modos de fallo del generador y del juez, por modelo", "",
             f"Campaña: `{run}`", ""]

    lines += ["## Resumen", "",
              "| Generador | Registros | Aceptados | Aceptación | Al primer intento | Media juicios/registro | Longitud media |",
              "|---|---|---|---|---|---|---|"]
    for m, r in results.items():
        per_record = r["total_judge_calls"] / r["records"] if r["records"] else 0
        lines.append(f"| {m} | {r['records']} | {r['accepted']} | {r['acceptance']}% | "
                     f"{r['first_try']} ({r['first_try_pct']}%) | {per_record:.2f} | {r['length_mean']} |")

    lines += ["", "## Reparto de intentos (cuántas regeneraciones necesitó cada registro)", "",
              "| Generador | 1 | 2 | 3 | 4 (agotado) |", "|---|---|---|---|---|"]
    for m, r in results.items():
        a = r["attempts"]
        lines.append(f"| {m} | {a.get(1,0)} | {a.get(2,0)} | {a.get(3,0)} | {a.get(4,0)} |")

    lines += ["", "## Veredictos del juez", "",
              "| Generador | ok | fallos | tipo de fallo | fallo en intento 1 | en intento 2 |",
              "|---|---|---|---|---|---|"]
    for m, r in results.items():
        k = r["judge_kinds"]
        fails = {kk: v for kk, v in k.items() if kk != "ok"}
        fb = r["judge_fail_by_attempt"]
        lines.append(f"| {m} | {k.get('ok',0)} | {sum(fails.values())} | "
                     f"{', '.join(f'{kk}={v}' for kk, v in fails.items()) or '—'} | "
                     f"{fb.get(1,0)} | {fb.get(2,0)} |")

    lines += ["", "## Motivos de rechazo por campo (un registro puede acumular varios)", "",
              "| Generador | urgencia | elementos | riesgos | subcat | ubicación | inventado | nº problemas | higiene | juez no disponible |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in results.items():
        rr = r["reject_reasons"]
        lines.append("| " + m + " | " + " | ".join(
            str(rr.get(k, 0)) for k in
            ("urgency", "affected_elements", "risks", "subcategory", "location",
             "inventado", "n_problems", "hygiene", "judge_unavailable")) + " |")

    lines += ["", "## Fallos de higiene por tipo", ""]
    for m, r in results.items():
        if r["hygiene"]:
            lines.append(f"- **{m}**: " + ", ".join(f"{k} ({v})" for k, v in r["hygiene"].items()))
        else:
            lines.append(f"- **{m}**: ninguno")

    lines += ["", "## Ejemplos reales por modo de fallo", ""]
    for m, r in results.items():
        lines.append(f"### {m}")
        if not r["reject_examples"]:
            lines.append("")
            lines.append("Sin rechazos.")
            lines.append("")
            continue
        for reason, examples in r["reject_examples"].items():
            lines.append(f"- **{reason}**")
            for rid, detail in examples:
                lines.append(f"  - `{rid}`: {detail}")
        lines.append("")

    md = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"informe -> {args.out}")
    print(md)


if __name__ == "__main__":
    main()
