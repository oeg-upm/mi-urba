#!/usr/bin/env python3
"""
miUrba — Extractor / summarizador de quejas vecinales (VLM).

Extrae información estructurada de quejas/incidencias reportadas por
vecinos y produce extracciones conformes a
schemas/extraction_schema.json (en la raíz del repositorio).

Usa un VLM (modelo multimodal vision) que ve el texto del vecino Y las
fotos adjuntas. El VLM describe las imágenes (image_descriptions) y extrae
la estructura (extractions) en una sola llamada. Describir las imágenes es
tarea del extractor; el generador ya no produce descripciones.

Recibe la SALIDA del proyecto `miurba-generator`:
  - complaints.jsonl  (texto del vecino, canal, timestamp, user_pics)
  - <images_dir>/*.png (las fotos reales referenciadas por user_pics)

y produce:
  - extractions.jsonl  (image_descriptions + extracciones estructuradas)

Uso:
  python cli.py extract                  # extraer con config por defecto
  python cli.py extract --model <vlm>     # otro modelo vision
  python cli.py extract --raw-dir data/raw --images-dir data/images --annotated-dir data/annotated

Toda la configuración se lee de la config COMPARTIDA en la raíz del
repositorio (../config.yaml por defecto, o la indicada con --config).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add this project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config, get_paths, get_vision_model_params
from src.llm import LLMClient
from src.pipeline import extract_dataset


def _get_path_args(args, config):
    """Build path kwargs from CLI overrides or config defaults."""
    paths = get_paths(config)
    return {
        "raw_subdir": args.raw_subdir or paths.get("raw_subdir", "raw"),
        "annotated_subdir": args.annotated_subdir or paths.get("annotated_subdir", "annotated"),
        "images_subdir": args.images_subdir or paths.get("images_subdir", "images"),
        "complaints_file": args.complaints_file or paths.get("complaints_file", "complaints.jsonl"),
        "scenarios_file": args.scenarios_file or paths.get("scenarios_file", "scenarios.jsonl"),
        "extractions_file": args.extractions_file or paths.get("extractions_file", "extractions.jsonl"),
    }


def cmd_extract(args):
    """Extract structured data + image descriptions from raw complaints + images."""
    config = get_config(args.config)
    paths = _get_path_args(args, config)
    gen_cfg = config.get("generation", {})
    ann_cfg = config.get("annotation", {})

    # VLM (vision-capable) model: explicit --model > annotation.vision_model > annotation.model
    model_cfg = get_vision_model_params(args.model, config)
    model_name = model_cfg["model"]

    temperature = args.temperature or ann_cfg.get("temperature", 0.2)

    # Resolve directories: CLI override > explicit paths config > default
    default_output = args.output_dir or gen_cfg.get("default_output_dir", "data")
    raw_dir = args.raw_dir or os.path.join(default_output, paths["raw_subdir"])
    annotated_dir = args.annotated_dir or os.path.join(default_output, paths["annotated_subdir"])
    images_dir = args.images_dir or os.path.join(default_output, paths["images_subdir"])

    client = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        temperature=temperature,
        timeout=args.timeout,
        api_key=model_cfg.get("api_key"),
    )

    print(f"miUrba — Extraction (raw + images → structured, VLM)")
    print(f"{'=' * 55}")
    print(f"  Model (VLM): {model_name}")
    print(f"  Endpoint:   {model_cfg['base_url']}")
    print(f"  Source:     {raw_dir}/{paths['complaints_file']}")
    print(f"  Images:     {images_dir}")
    print(f"  Output:     {annotated_dir}/{paths['extractions_file']}")
    print(f"  Resume:     {not args.no_resume}")
    print(f"{'=' * 55}")

    stats = extract_dataset(
        client=client,
        raw_dir=raw_dir,
        images_dir=images_dir,
        output_dir=annotated_dir,
        raw_complaints_file=paths["complaints_file"],
        raw_scenarios_file=paths["scenarios_file"],
        extractions_file=paths["extractions_file"],
        resume=not args.no_resume,
    )

    print(f"\n{'=' * 55}")
    print(f"Extraction complete!")
    print(f"  Extracted:  {stats['extracted']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Skipped:    {stats['skipped']}")
    print(f"  Total:      {stats['total']}")
    print(f"  Tokens in:  {stats['total_tokens_in']:,}")
    print(f"  Tokens out: {stats['total_tokens_out']:,}")
    print(f"  Images dir: {stats.get('images_dir')}")
    print(f"  Output:     {stats['output_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="miUrba — Extractor / summarizador de quejas vecinales (VLM)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Shared arguments ───────────────────────────────────
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None,
                        help="Path to shared config.yaml (default: ../config.yaml at repo root)")
    common.add_argument("--output-dir", default=None, help="Output directory (default: from config)")
    common.add_argument("--raw-subdir", default=None, help="Raw subdirectory name (default: from config)")
    common.add_argument("--annotated-subdir", default=None, help="Annotated subdirectory name (default: from config)")
    common.add_argument("--images-subdir", default=None, help="Images subdirectory name (default: from config)")
    common.add_argument("--raw-dir", default=None, help="Raw directory path (default: <output_dir>/<raw_subdir>)")
    common.add_argument("--annotated-dir", default=None, help="Annotated directory path (default: <output_dir>/<annotated_subdir>)")
    common.add_argument("--images-dir", default=None, help="Directory with the photo files (default: <output_dir>/<images_subdir>)")
    common.add_argument("--complaints-file", default=None, help="Complaints filename (default: from config)")
    common.add_argument("--scenarios-file", default=None, help="Scenarios filename (default: from config)")
    common.add_argument("--extractions-file", default=None, help="Extractions filename (default: from config)")
    common.add_argument("--no-resume", action="store_true", help="Start fresh")
    common.add_argument("--timeout", type=int, default=300, help="Request timeout (seconds)")
    common.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # ── Extract ────────────────────────────────────────────
    ext_parser = subparsers.add_parser("extract", parents=[common],
                                        help="Extract structured data + image descriptions from raw complaints + images")
    ext_parser.add_argument("--model", default=None,
                             help="VLM model for extraction (default: annotation.vision_model or annotation.model)")
    ext_parser.add_argument("--temperature", type=float, default=None,
                             help="Temperature for extraction")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    commands = {
        "extract": cmd_extract,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()