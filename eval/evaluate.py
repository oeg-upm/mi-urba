#!/usr/bin/env python3
"""
miUrba — Evaluación del extractor contra ground truth.

Compara las extracciones del VLM/LLM contra el ground truth del generador.
Calcula métricas por campo, por queja y por modelo.

Uso:
  python evaluate.py --gt <ground_truth.jsonl> --pred <extractions.jsonl> [--model-name kimi-k2.6]
  python evaluate.py --gt gt.jsonl --pred ext1.jsonl --model-name kimi --report report.md
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── Matching GT vs predicciones ──────────────────────────────────────────────

def match_extractions(gt_extractions: list[dict], pred_extractions: list[dict]) -> list[tuple]:
    """
    Emparejar extracciones GT con predicciones.

    Estrategia:
    1. Match exacto por (category.principal, category.subcategory)
    2. Match parcial por category.principal
    3. Resto: no emparejados

    Returns: lista de (gt, pred, match_type) donde gt o pred pueden ser None
    (None = unmatched = falso negativo o falso positivo)
    """
    matched = []
    gt_used = [False] * len(gt_extractions)
    pred_used = [False] * len(pred_extractions)

    # Paso 1: match exacto (principal + subcategory)
    for i, gt in enumerate(gt_extractions):
        if gt_used[i]:
            continue
        gt_cat = gt.get("category", {})
        gt_key = (gt_cat.get("principal"), gt_cat.get("subcategory"))
        for j, pred in enumerate(pred_extractions):
            if pred_used[j]:
                continue
            pred_cat = pred.get("category", {})
            pred_key = (pred_cat.get("principal"), pred_cat.get("subcategory"))
            if gt_key == pred_key:
                matched.append((gt, pred, "exact"))
                gt_used[i] = True
                pred_used[j] = True
                break

    # Paso 2: match por principal
    for i, gt in enumerate(gt_extractions):
        if gt_used[i]:
            continue
        gt_principal = gt.get("category", {}).get("principal")
        for j, pred in enumerate(pred_extractions):
            if pred_used[j]:
                continue
            pred_principal = pred.get("category", {}).get("principal")
            if gt_principal == pred_principal:
                matched.append((gt, pred, "partial"))
                gt_used[i] = True
                pred_used[j] = True
                break

    # Paso 3: no emparejados
    for i, gt in enumerate(gt_extractions):
        if not gt_used[i]:
            matched.append((gt, None, "fn"))  # falso negativo
    for j, pred in enumerate(pred_extractions):
        if not pred_used[j]:
            matched.append((None, pred, "fp"))  # falso positivo

    return matched


# ── Métricas por campo ────────────────────────────────────────────────────────

def field_accuracy(matched: list[tuple], field: str) -> dict:
    """Calcular accuracy para un campo sobre los pares emparejados (gt, pred)."""
    correct = 0
    total = 0
    for gt, pred, mtype in matched:
        if gt is None or pred is None:
            continue  # solo pares emparejados
        total += 1
        if field == "category_principal":
            gt_val = gt.get("category", {}).get("principal", "")
            pred_val = pred.get("category", {}).get("principal", "")
        elif field == "category_subcategory":
            gt_val = gt.get("category", {}).get("subcategory", "")
            pred_val = pred.get("category", {}).get("subcategory", "")
        elif field == "category_full":
            gt_val = (gt.get("category", {}).get("principal", ""),
                      gt.get("category", {}).get("subcategory", ""))
            pred_val = (pred.get("category", {}).get("principal", ""),
                        pred.get("category", {}).get("subcategory", ""))
        elif field == "urgency":
            gt_val = gt.get("urgency", "")
            pred_val = pred.get("urgency", "")
        elif field == "location_zone":
            gt_val = gt.get("location", {}).get("zone", "")
            pred_val = pred.get("location", {}).get("zone", "")
        elif field == "location_detail":
            gt_val = gt.get("location", {}).get("detail", "").lower().strip()
            pred_val = pred.get("location", {}).get("detail", "").lower().strip()
            # Match parcial: uno contiene al otro
            if gt_val and pred_val:
                if gt_val in pred_val or pred_val in gt_val:
                    correct += 1
                continue
            elif not gt_val and not pred_val:
                correct += 1
                continue
            else:
                continue
        elif field == "risks_jaccard":
            gt_set = set(gt.get("risks", []))
            pred_set = set(pred.get("risks", []))
            if not gt_set and not pred_set:
                correct += 1  # ambos vacíos = correcto
            elif gt_set and pred_set:
                jaccard = len(gt_set & pred_set) / len(gt_set | pred_set)
                correct += jaccard  # contribución fraccional
            continue
        elif field == "affected_elements_precision":
            gt_elems = {(e.get("principal"), e.get("subcategory"))
                        for e in gt.get("affected_elements", [])}
            pred_elems = {(e.get("principal"), e.get("subcategory"))
                          for e in pred.get("affected_elements", [])}
            if not gt_elems and not pred_elems:
                correct += 1
            elif pred_elems:
                correct += len(gt_elems & pred_elems) / len(pred_elems)
            continue
        elif field == "affected_elements_recall":
            gt_elems = {(e.get("principal"), e.get("subcategory"))
                        for e in gt.get("affected_elements", [])}
            pred_elems = {(e.get("principal"), e.get("subcategory"))
                          for e in pred.get("affected_elements", [])}
            if not gt_elems and not pred_elems:
                correct += 1
            elif gt_elems:
                correct += len(gt_elems & pred_elems) / len(gt_elems)
            continue

        if gt_val == pred_val:
            correct += 1

    if total == 0:
        return {"accuracy": 0.0, "total": 0}
    return {"accuracy": round(correct / total, 4), "total": total}


# ── Métricas por queja ────────────────────────────────────────────────────────

def per_complaint_metrics(gt_extractions: list[dict], pred_extractions: list[dict]) -> dict:
    """Precision, recall, F1 a nivel de queja (match por categoría principal)."""
    gt_cats = {e.get("category", {}).get("principal") for e in gt_extractions}
    pred_cats = {e.get("category", {}).get("principal") for e in pred_extractions}

    tp = len(gt_cats & pred_cats)
    fp = len(pred_cats - gt_cats)
    fn = len(gt_cats - pred_cats)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    exact_match = (gt_cats == pred_cats)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": exact_match,
        "tp": tp, "fp": fp, "fn": fn,
        "gt_count": len(gt_extractions),
        "pred_count": len(pred_extractions),
    }


# ── Evaluación completa ──────────────────────────────────────────────────────

def evaluate(gt_file: str, pred_file: str, model_name: str = "") -> dict:
    """Evaluar un fichero de predicciones contra el ground truth."""
    with open(gt_file) as f:
        gt_data = {json.loads(l)["complaint_id"]: json.loads(l)
                   for l in f if l.strip()}
    with open(pred_file) as f:
        pred_data = {}
        for l in f:
            if l.strip():
                obj = json.loads(l)
                cid = obj.get("complaint_id", "")
                # Si hay duplicados, quedarse con el último
                pred_data[cid] = obj

    # Métricas agregadas
    all_matched = []
    per_complaint = {}
    missing = []

    for cid, gt in gt_data.items():
        if cid not in pred_data:
            missing.append(cid)
            continue
        pred = pred_data[cid]
        gt_exts = gt.get("extractions", [])
        pred_exts = pred.get("extractions", [])
        matched = match_extractions(gt_exts, pred_exts)
        all_matched.extend(matched)
        per_complaint[cid] = per_complaint_metrics(gt_exts, pred_exts)

    # Field metrics
    fields = [
        "category_principal", "category_subcategory", "category_full",
        "urgency", "location_zone", "location_detail",
        "risks_jaccard",
        "affected_elements_precision", "affected_elements_recall",
    ]
    field_metrics = {}
    for field in fields:
        field_metrics[field] = field_accuracy(all_matched, field)

    # Aggregate complaint metrics
    n = len(per_complaint)
    if n > 0:
        agg = {
            "precision": round(sum(c["precision"] for c in per_complaint.values()) / n, 4),
            "recall": round(sum(c["recall"] for c in per_complaint.values()) / n, 4),
            "f1": round(sum(c["f1"] for c in per_complaint.values()) / n, 4),
            "exact_match_rate": round(sum(1 for c in per_complaint.values() if c["exact_match"]) / n, 4),
            "avg_gt_extractions": round(sum(c["gt_count"] for c in per_complaint.values()) / n, 2),
            "avg_pred_extractions": round(sum(c["pred_count"] for c in per_complaint.values()) / n, 2),
        }
    else:
        agg = {k: 0 for k in ["precision", "recall", "f1", "exact_match_rate", "avg_gt_extractions", "avg_pred_extractions"]}

    # Match type breakdown
    match_types = Counter(m[2] for m in all_matched)

    return {
        "model": model_name,
        "total_gt": len(gt_data),
        "total_pred": len(pred_data),
        "evaluated": n,
        "missing": missing,
        "field_metrics": field_metrics,
        "complaint_metrics": agg,
        "match_types": dict(match_types),
    }


# ── Reporte legible ────────────────────────────────────────────────────────────

def generate_report(results: list[dict], output_path: str):
    """Generar informe markdown con los resultados."""
    lines = [
        "# Evaluación del extractor — Resultados\n",
        f"**Dataset:** {results[0]['total_gt']} quejas con ground truth\n",
        "## Métricas por modelo\n",
        "| Modelo | Precision | Recall | F1 | Exact Match | GT ext/complaint | Pred ext/complaint |",
        "|--------|-----------|--------|----|-------------|--------------------|---------------------|",
    ]

    for r in results:
        cm = r["complaint_metrics"]
        lines.append(
            f"| {r['model']} | {cm['precision']} | {cm['recall']} | {cm['f1']} | "
            f"{cm['exact_match_rate']} | {cm['avg_gt_extractions']} | {cm['avg_pred_extractions']} |"
        )

    lines.append("\n## Accuracy por campo (sobre extracciones emparejadas)\n")
    lines.append("| Modelo | Cat. principal | Cat. subcategory | Cat. completa | Urgency | Loc. zone | Loc. detail | Risks (Jaccard) | Elements P | Elements R |")
    lines.append("|--------|---------------|------------------|---------------|---------|-----------|-------------|-----------------|------------|------------|")

    for r in results:
        fm = r["field_metrics"]
        lines.append(
            f"| {r['model']} | "
            f"{fm['category_principal']['accuracy']} | "
            f"{fm['category_subcategory']['accuracy']} | "
            f"{fm['category_full']['accuracy']} | "
            f"{fm['urgency']['accuracy']} | "
            f"{fm['location_zone']['accuracy']} | "
            f"{fm['location_detail']['accuracy']} | "
            f"{fm['risks_jaccard']['accuracy']} | "
            f"{fm['affected_elements_precision']['accuracy']} | "
            f"{fm['affected_elements_recall']['accuracy']} |"
        )

    lines.append("\n## Tipos de match\n")
    lines.append("| Modelo | Exact | Partial | Falsos negativos | Falsos positivos |")
    lines.append("|--------|-------|---------|-------------------|------------------|")

    for r in results:
        mt = r["match_types"]
        lines.append(
            f"| {r['model']} | {mt.get('exact', 0)} | {mt.get('partial', 0)} | "
            f"{mt.get('fn', 0)} | {mt.get('fp', 0)} |"
        )

    lines.append(f"\n## Notas\n")
    lines.append("- **Exact**: categoría principal + subcategoría coinciden")
    lines.append("- **Partial**: solo categoría principal coincide")
    lines.append("- **Falsos negativos (FN)**: el GT tiene un problema que el extractor no detectó")
    lines.append("- **Falsos positivos (FP)**: el extractor detectó un problema que no está en el GT")
    lines.append("- **Risks (Jaccard)**: |GT ∩ pred| / |GT ∪ pred| (promediado)")
    lines.append("- **Elements P/R**: precision/recall sobre (principal, subcategory) de elementos afectados")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Reporte guardado en {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluar extractor contra ground truth")
    parser.add_argument("--gt", required=True, help="Ground truth JSONL")
    parser.add_argument("--pred", required=True, help="Predicciones JSONL (puede repetirse con --model)")
    parser.add_argument("--model-name", default="", help="Nombre del modelo evaluado")
    parser.add_argument("--report", default=None, help="Guardar reporte markdown en este path")
    args = parser.parse_args()

    result = evaluate(args.gt, args.pred, args.model_name)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.report:
        generate_report([result], args.report)


if __name__ == "__main__":
    main()