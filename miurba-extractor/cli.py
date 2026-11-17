#!/usr/bin/env python3
"""
miUrba — Extractor / summarizador de quejas vecinales (LLM).

Extrae información estructurada de quejas/incidencias reportadas por
vecinos y produce extracciones conformes a
schemas/extraction_schema.json (en la raíz del repositorio).

Recibe la SALIDA del proyecto `miurba-generator`:
  - complaints.jsonl  (texto del vecino, timestamp)

y produce:
  - extractions.jsonl  (extracciones estructuradas)

Uso (config-only — TODOS los parámetros viven en el config):

  python cli.py extract --config config/config.yaml

El comando `extract` acepta un único argumento, `--config` (por defecto
config/config.yaml del subproyecto). Toda la configuración
pertinente se lee de tres secciones:

  extraction:
    model              Modelo extractor (clave de models:). OBLIGATORIO.
    temperature        Temperatura de las llamadas de extracción
    default_output_dir Base de rutas por defecto (null = generation.
                       default_output_dir)
    raw_dir            Directorio de entrada (null = <base>/<raw_subdir>)
    annotated_dir      Directorio de salida (null = <base>/<annotated_subdir>)
    resume             Saltar quejas ya presentes en extractions.jsonl
    timeout            Timeout por petición (segundos)
    max_retries        Reintentos por petición
    retry_delay        Espera base entre reintentos (segundos)

  paths:                 Nombres de subdirectorios y ficheros JSONL
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
                        get_extraction_model_params,
                        get_extraction_temperature,
                        get_extraction_json_schema)
from src.llm import LLMClient
from src.pipeline import extract_dataset

logger = logging.getLogger(__name__)


def resolve_dirs(ext_cfg: dict, paths: dict, fallback_base: str) -> tuple[str, str]:
    """Resolve (raw_dir, annotated_dir) from the extraction config.

    Precedence per directory: explicit extraction.<dir> > base dir +
    the corresponding `paths:` subdir. The base dir is
    extraction.default_output_dir, falling back to `fallback_base`
    (generation.default_output_dir) and lastly "data".
    """
    base = ext_cfg["default_output_dir"] or fallback_base or "data"
    raw_dir = ext_cfg["raw_dir"] or os.path.join(base, paths["raw_subdir"])
    annotated_dir = ext_cfg["annotated_dir"] or os.path.join(base, paths["annotated_subdir"])
    return raw_dir, annotated_dir


def cmd_extract(args):
    """Extract structured data from raw complaints."""
    config = get_config(args.config)
    paths = get_paths(config)
    ext_cfg = get_extraction_config(config)

    # Extraction model: extraction.model (REQUIRED, no fallback)
    model_name, model_cfg = get_extraction_model_params(ext_cfg["model"], config)

    # Constrained decoding: per-model structured_output mode. json_schema
    # loads the extraction schema (enums synced from config) and the server
    # enforces it token-by-token. json_object only guarantees valid JSON.
    structured_mode = config.get("models", {}).get(model_name, {}).get("structured_output")
    # Constrained decoding (server-side sampling constraint) — only with json_schema
    json_schema = get_extraction_json_schema(config) if structured_mode == "json_schema" else None
    # BUT our own validation ALWAYS applies: the output must conform to the
    # schema no matter how the model was sampled.
    validation_schema = get_extraction_json_schema(config)

    fallback_base = config.get("generation", {}).get("default_output_dir", "data")
    raw_dir, annotated_dir = resolve_dirs(ext_cfg, paths, fallback_base)

    client = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        temperature=get_extraction_temperature(ext_cfg, model_cfg, config, model_name),
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
    if validation_schema is not None:
        logger.info("Output validation against extraction_schema.json: ENABLED")

    print(f"miUrba — Extraction (raw → structured, LLM)")
    print(f"{'=' * 55}")
    print(f"  Config:     {args.config or 'config/config.yaml (subproject default)'}")
    print(f"  Model:      {model_name}")
    print(f"  Endpoint:   {model_cfg['base_url']}")
    print(f"  Source:     {raw_dir}/{paths['complaints_file']}")
    print(f"  Output:     {annotated_dir}/{paths['extractions_file']}")
    print(f"  Resume:     {ext_cfg['resume']}")
    print(f"  Constrained: {structured_mode or 'off'}")
    print(f"  Workers:    {ext_cfg['workers']}")
    print(f"{'=' * 55}")

    stats = extract_dataset(
        client=client,
        raw_dir=raw_dir,
        output_dir=annotated_dir,
        raw_complaints_file=paths["complaints_file"],
        extractions_file=paths["extractions_file"],
        resume=ext_cfg["resume"],
        num_workers=ext_cfg["workers"],
        validation_schema=validation_schema,
    )

    print(f"\n{'=' * 55}")
    if stats.get("input_missing"):
        print("Input complaints file not found — nothing extracted.")
    else:
        print("Extraction complete!")
    print(f"  Extracted:  {stats['extracted']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Skipped:    {stats['skipped']}")
    print(f"  Total:      {stats['total']}")
    print(f"  Tokens in:  {stats['total_tokens_in']:,}")
    print(f"  Tokens out: {stats['total_tokens_out']:,}")
    print(f"  Output:     {stats['output_path']}")
    if client._stats.get("errors") or client._stats.get("load_waits"):
        print(f"  ⚠ errors absorbed: {client._stats['errors']}  "
              f"(load waits: {client._stats['load_waits']})")

    # Exit codes: 2 = input missing, 1 = every pending record failed, 0 otherwise
    # (a fully resumed run produces nothing but is a success).
    if stats.get("input_missing"):
        sys.exit(2)
    if stats["failed"] > 0 and stats["extracted"] == 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="miUrba — Extractor / summarizador de quejas vecinales "
                    "(todos los parámetros se leen del config)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Shared arguments (config-only) ─────────────────────
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None,
                        help="Path to config.yaml (default: config/config.yaml in this subproject)")

    subparsers.add_parser("extract", parents=[common],
                          help="Extract structured data (config-only)")

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
