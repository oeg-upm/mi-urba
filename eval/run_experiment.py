#!/usr/bin/env python3
"""
miUrba — Orquestador del experimento de evaluación.

Ejecuta el pipeline completo:
  1. Genera dataset (si no existe)
  2. Corre extractores sobre el dataset
  3. Evalúa cada extractor contra el ground truth
  4. Genera reporte comparativo

Uso:
  python run_experiment.py                    # todo
  python run_experiment.py --generate-only     # solo fase 1
  python run_experiment.py --extract-only      # solo fase 2 (necesita fase 1)
  python run_experiment.py --evaluate-only     # solo fase 3 (necesita fases 1+2)

Reproducibilidad:
  - Config fija en eval/config.eval.yaml
  - Resultados en eval/results/
  - Logs de cada extractor en eval/results/extractions/<model>/log.txt
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
CONFIG = str(EVAL_DIR / "config.eval.yaml")
RESULTS = EVAL_DIR / "results"
DATASET = RESULTS / "dataset"
EXTRACTIONS = RESULTS / "extractions"

# Modelos a evaluar (sin glm-5.2 que es el generador)
EXTRACTORS = ["kimi-k2.6", "kimi-k2.7-code", "qwen3.5-397b", "qwen3.6-35b"]

# Generador fijo
GENERATOR_MODEL = "glm-5.2"
N_RECORDS = 100


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Fase 1: Generar dataset ───────────────────────────────────────────────────

def generate_dataset():
    """Generar dataset con el generador (GLM-5.2, 100 registros)."""
    out_dir = str(DATASET)
    raw_dir = DATASET / "raw"

    if raw_dir.exists() and (raw_dir / "complaints.jsonl").exists():
        with open(raw_dir / "complaints.jsonl") as f:
            n = sum(1 for l in f if l.strip())
        if n >= N_RECORDS * 0.9:  # al menos 90%
            log(f"Dataset ya existe ({n} quejas), saltando generación")
            return

    log(f"Generando {N_RECORDS} quejas con {GENERATOR_MODEL}...")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "miurba-generator" / "cli.py"),
        "generate",
        "-n", str(N_RECORDS),
        "--model", GENERATOR_MODEL,
        "--config", CONFIG,
        "--output-dir", out_dir,
        "--no-resume",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT / "miurba-generator"))
    if result.returncode != 0:
        log("ERROR: Generación fallida")
        sys.exit(1)
    log("Generación completada")


# ── Fase 2: Extraer ───────────────────────────────────────────────────────────

def run_extractor(model: str):
    """Correr un extractor sobre el dataset."""
    model_dir = EXTRACTIONS / model
    raw_dir = model_dir / "raw"
    images_dir = model_dir / "images"
    annotated_dir = model_dir / "annotated"

    # Verificar si ya está completo (ANTES de limpiar)
    out_file = annotated_dir / "extractions.jsonl"
    if out_file.exists():
        with open(out_file) as f:
            n = sum(1 for l in f if l.strip())
        ids = set()
        with open(out_file) as f:
            for l in f:
                if l.strip():
                    ids.add(json.loads(l).get("complaint_id", ""))
        if len(ids) >= N_RECORDS * 0.9 and len(ids) == n:
            log(f"  {model}: ya completo ({n} extracciones, sin dupes), saltando")
            return

    # Limpiar directorio (evitar duplicados de runs anteriores)
    if annotated_dir.exists():
        shutil.rmtree(annotated_dir)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copiar complaints
    shutil.copy2(DATASET / "raw" / "complaints.jsonl", raw_dir / "complaints.jsonl")

    log(f"  {model}: extrayendo...")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "miurba-extractor" / "cli.py"),
        "extract",
        "--config", CONFIG,
        "--model", model,
        "--raw-dir", str(raw_dir),
        "--images-dir", str(images_dir),
        "--annotated-dir", str(annotated_dir),
        "--no-resume",
        "--timeout", "600",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT / "miurba-extractor"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file = model_dir / "log.txt"
    log_file.write_text(result.stdout)

    if result.returncode != 0:
        log(f"  {model}: ERROR (ver {log_file})")
        return

    if out_file.exists():
        with open(out_file) as f:
            n = sum(1 for l in f if l.strip())
        log(f"  {model}: {n} extracciones completadas")
    else:
        log(f"  {model}: sin output (ver {log_file})")


def run_all_extractors():
    """Correr todos los extractores."""
    log("Fase 2: Extracción")
    for model in EXTRACTORS:
        run_extractor(model)
    log("Extracción completada")


# ── Fase 3: Evaluar ───────────────────────────────────────────────────────────

def evaluate_all():
    """Evaluar todos los extractores contra el ground truth."""
    log("Fase 3: Evaluación")

    gt_file = str(DATASET / "raw" / "ground_truth.jsonl")
    if not Path(gt_file).exists():
        log("ERROR: Ground truth no encontrado. Ejecuta --generate-only primero.")
        sys.exit(1)

    # Importar el evaluador
    sys.path.insert(0, str(EVAL_DIR))
    from evaluate import evaluate, generate_report

    all_results = []
    for model in EXTRACTORS:
        pred_file = EXTRACTIONS / model / "annotated" / "extractions.jsonl"
        if not pred_file.exists():
            log(f"  {model}: sin extracciones, saltando")
            continue

        log(f"  Evaluando {model}...")
        result = evaluate(gt_file, str(pred_file), model)
        all_results.append(result)

        # Stats rápidas
        cm = result["complaint_metrics"]
        log(f"    P={cm['precision']} R={cm['recall']} F1={cm['f1']} EM={cm['exact_match_rate']}")

    if not all_results:
        log("No hay resultados para evaluar")
        sys.exit(1)

    # Guardar métricas JSON
    metrics_file = RESULTS / "metrics.json"
    metrics_file.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"Métricas guardadas en {metrics_file}")

    # Generar reporte
    report_file = RESULTS / "report.md"
    generate_report(all_results, str(report_file))
    log(f"Reporte guardado en {report_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experimento de evaluación miUrba")
    parser.add_argument("--generate-only", action="store_true", help="Solo generar dataset")
    parser.add_argument("--extract-only", action="store_true", help="Solo correr extractores")
    parser.add_argument("--evaluate-only", action="store_true", help="Solo evaluar")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.generate_only:
        generate_dataset()
    elif args.extract_only:
        run_all_extractors()
    elif args.evaluate_only:
        evaluate_all()
    else:
        generate_dataset()
        run_all_extractors()
        evaluate_all()

    log("Done")


if __name__ == "__main__":
    main()