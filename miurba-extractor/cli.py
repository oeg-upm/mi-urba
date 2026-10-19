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

Uso (config-only — TODOS los parámetros viven en el config):

  python cli.py extract --config ../config.yaml

El comando `extract` acepta un único argumento, `--config` (por defecto
el config compartido en la raíz del repositorio). Toda la configuración
pertinente se lee de tres secciones:

  extraction:
    model              Modelo VLM (clave de models:). Si es null:
                       annotation.vision_model → annotation.model
    temperature        Temperatura de las llamadas de extracción
    default_output_dir Base de rutas por defecto (null = generation.
                       default_output_dir)
    raw_dir            Directorio de entrada (null = <base>/<raw_subdir>)
    annotated_dir      Directorio de salida (null = <base>/<annotated_subdir>)
    images_dir         Directorio de fotos (null = <base>/<images_subdir>)
    resume             Saltar quejas ya presentes en extractions.jsonl
    timeout            Timeout por petición (segundos)
    max_retries        Reintentos por petición
    retry_delay        Espera base entre reintentos (segundos)

  paths:                 Nombres de subdirectorios y ficheros JSONL
  annotation:            Fallback del modelo VLM (vision_model / model)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add this project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (get_logging_config,
                        get_config, get_extraction_config, get_paths,
                        get_vision_model_params, get_model_params,
                        get_extraction_json_schema,
                        get_image_description_json_schema)
from src.llm import LLMClient
from src.pipeline import extract_dataset

logger = logging.getLogger(__name__)


def resolve_dirs(ext_cfg: dict, paths: dict, fallback_base: str) -> tuple[str, str, str]:
    """Resolve (raw_dir, annotated_dir, images_dir) from the extraction config.

    Precedence per directory: explicit extraction.<dir> > base dir +
    the corresponding `paths:` subdir. The base dir is
    extraction.default_output_dir, falling back to `fallback_base`
    (generation.default_output_dir) and lastly "data".
    """
    base = ext_cfg["default_output_dir"] or fallback_base or "data"
    raw_dir = ext_cfg["raw_dir"] or os.path.join(base, paths["raw_subdir"])
    annotated_dir = ext_cfg["annotated_dir"] or os.path.join(base, paths["annotated_subdir"])
    images_dir = ext_cfg["images_dir"] or os.path.join(base, paths["images_subdir"])
    return raw_dir, annotated_dir, images_dir


def cmd_extract(args):
    """Extract structured data + image descriptions from raw complaints + images."""
    config = get_config(args.config)
    paths = get_paths(config)
    ext_cfg = get_extraction_config(config)

    # VLM model: extraction.model > annotation.vision_model > annotation.model
    model_cfg = get_vision_model_params(ext_cfg["model"], config)
    model_name = model_cfg["model"]

    # Constrained decoding: per-model structured_output mode. json_schema
    # loads the extraction schema (enums synced from config) and the server
    # enforces it token-by-token. json_object only guarantees valid JSON.
    structured_mode = config.get("models", {}).get(ext_cfg["model"], {}).get("structured_output")
    json_schema = get_extraction_json_schema(config) if structured_mode == "json_schema" else None

    fallback_base = config.get("generation", {}).get("default_output_dir", "data")
    raw_dir, annotated_dir, images_dir = resolve_dirs(ext_cfg, paths, fallback_base)

    client = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        temperature=ext_cfg["temperature"],
        timeout=ext_cfg["timeout"],
        max_retries=ext_cfg["max_retries"],
        retry_delay=ext_cfg["retry_delay"],
        api_key=model_cfg.get("api_key"),
        structured_output=structured_mode,
        json_schema=json_schema,
        model_load_wait=ext_cfg["model_load_wait"],
        connection_backoff=model_cfg.get("connection_backoff", False),
        extra_params=model_cfg.get("extra_params", {}),
    )

    # Stage-split: optional vision client for stage 1 (photo descriptions)
    stage_split = bool(ext_cfg.get("stage_split"))
    vision_client = None
    if stage_split:
        vision_name = ext_cfg.get("vision_model")
        if vision_name:
            v_cfg = get_model_params(vision_name, config)
            v_mode = config.get("models", {}).get(vision_name, {}).get("structured_output")
            vision_client = LLMClient(
                base_url=v_cfg["base_url"],
                model=v_cfg["model"],
                temperature=v_cfg.get("temperature", 0.2),
                timeout=ext_cfg["timeout"],
                max_retries=ext_cfg["max_retries"],
                retry_delay=ext_cfg["retry_delay"],
                api_key=v_cfg.get("api_key"),
                structured_output=v_mode,
                json_schema=get_image_description_json_schema(config) if v_mode == "json_schema" else None,
                model_load_wait=ext_cfg["model_load_wait"],
                connection_backoff=v_cfg.get("connection_backoff", False),
                extra_params=v_cfg.get("extra_params", {}),
            )
        else:
            logger.warning("extraction.stage_split=true but extraction.vision_model is null — "
                           "falling back to single-call")
            stage_split = False

    print(f"miUrba — Extraction (raw + images → structured, VLM)")
    print(f"{'=' * 55}")
    print(f"  Config:     {args.config or 'config.yaml (repo root)'}")
    print(f"  Model (VLM): {model_name}")
    print(f"  Endpoint:   {model_cfg['base_url']}")
    print(f"  Source:     {raw_dir}/{paths['complaints_file']}")
    print(f"  Images:     {images_dir}")
    print(f"  Output:     {annotated_dir}/{paths['extractions_file']}")
    print(f"  Resume:     {ext_cfg['resume']}")
    print(f"  Constrained: {structured_mode or 'off'}")
    print(f"  Workers:    {ext_cfg['workers']}")
    print(f"  Stage-split: {stage_split}" +
          (f" (vision: {ext_cfg['vision_model']})" if stage_split else ""))
    print(f"{'=' * 55}")

    stats = extract_dataset(
        client=client,
        raw_dir=raw_dir,
        images_dir=images_dir,
        output_dir=annotated_dir,
        raw_complaints_file=paths["complaints_file"],
        extractions_file=paths["extractions_file"],
        resume=ext_cfg["resume"],
        stage_split=stage_split,
        vision_client=vision_client,
        num_workers=ext_cfg["workers"],
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
    if client._stats.get("errors") or client._stats.get("load_waits"):
        print(f"  ⚠ errors absorbed: {client._stats['errors']}  "
              f"(load waits: {client._stats['load_waits']})")


def main():
    parser = argparse.ArgumentParser(
        description="miUrba — Extractor / summarizador de quejas vecinales (VLM) "
                    "(todos los parámetros se leen del config compartido)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Shared arguments (config-only) ─────────────────────
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None,
                        help="Path to shared config.yaml (default: ../config.yaml at repo root)")
    common.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers.add_parser("extract", parents=[common],
                          help="Extract structured data + image descriptions (config-only)")

    args = parser.parse_args()

    log_cfg = get_logging_config(get_config(args.config))
    handlers: list = [logging.StreamHandler()]
    if log_cfg.get("file"):
        log_file = Path(log_cfg["file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    if log_cfg.get("file"):
        print(f"Logging to: {log_cfg['file']}")

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
