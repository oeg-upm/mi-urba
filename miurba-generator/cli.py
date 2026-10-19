#!/usr/bin/env python3
"""
miUrba — Generador de datasets sintéticos de quejas vecinales.

Genera quejas/incidencias sintéticas reportadas por vecinos de una
comunidad de vecinos. Produce dos ficheros JSONL por cada ejecución:
  - complaints.jsonl  (el texto del vecino, canal, timestamp, pics)
  - ground_truth.jsonl  (extracciones esperadas)
  - metadata.jsonl      (persona, modifier, community, image_prompts)

La extracción estructurada de estas quejas es una tarea independiente
que corresponde al proyecto `miurba-extractor`.

Uso (config-only — TODOS los parámetros viven en el config):

  python cli.py generate --config ../config.yaml    # generar según el plan
  python cli.py merge    --config ../config.yaml    # fusionar salidas por modelo

El comando `generate` acepta un único argumento, `--config` (por defecto
el config compartido en la raíz del repositorio). Toda la configuración
pertinente se lee de dos secciones:

  generation:
    default_output_dir   Directorio base de salida (<dir>/<modelo>/)
    batches              Plan por lotes: [{model, count}, ...]
    models               Lista explícita de modelos (count = num_records)
    num_records          Registros por modelo cuando se usa `models`
    start_id             Primer número de ID (Q<start_id>, secuencial)
    seed                 Semilla base de muestreo (null = no determinista)
    resume               Saltar registros ya presentes en la salida
    timeout              Timeout por petición (segundos)
    max_retries          Reintentos por petición
    retry_delay          Espera base entre reintentos (segundos)

  paths:                 Nombres de subdirectorios y ficheros JSONL
  image_generation:      Generación de imágenes (enabled: true/false)

Cada modelo escribe en <default_output_dir>/<modelo>/; al final del plan
`generate` fusiona todas las salidas en <default_output_dir>/<raw_subdir>/.
El comando `merge` repite solo la fusión.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

# Add this project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (get_config, get_generation_config, get_model_params,
                        get_paths, get_image_generation_config, get_logging_config)
from src.llm import LLMClient
from src.pipeline import generate_dataset, merge_outputs


def _generation_plan(gen_cfg: dict) -> list[tuple[str, int]]:
    """Resolve the (model, count) plan from the generation config.

    Precedence: explicit `models` list (each with `num_records`) over
    `batches`. Raises ValueError when neither is configured.
    """
    if gen_cfg.get("models"):
        n = gen_cfg["num_records"]
        return [(m, n) for m in gen_cfg["models"]]
    if gen_cfg.get("batches"):
        return [(b["model"], b["count"]) for b in gen_cfg["batches"]]
    raise ValueError(
        "config.yaml: define generation.batches ([{model, count}, ...]) "
        "or generation.models (list) — nothing to generate."
    )


def cmd_generate(args):
    """Generate synthetic raw complaints — ALL parameters come from the config."""
    config = get_config(args.config)
    paths = get_paths(config)
    gen_cfg = get_generation_config(config)
    ig_config = get_image_generation_config(config)
    enable_images = bool(ig_config.get("enabled", False))

    plan = _generation_plan(gen_cfg)
    output_dir = gen_cfg["default_output_dir"]
    raw_subdir = paths["raw_subdir"]

    # Optional judge model for 2-stage validation (None = generator itself)
    validator_client = None
    if gen_cfg.get("validate") and gen_cfg.get("validator"):
        v_cfg = get_model_params(gen_cfg["validator"], config)
        validator_client = LLMClient(
            base_url=v_cfg["base_url"],
            model=v_cfg["model"],
            temperature=v_cfg["temperature"],
            timeout=gen_cfg["timeout"],
            max_retries=gen_cfg["max_retries"],
            retry_delay=gen_cfg["retry_delay"],
            api_key=v_cfg.get("api_key"),
            model_load_wait=gen_cfg["model_load_wait"],
            connection_backoff=v_cfg.get("connection_backoff", False),
            extra_params=v_cfg.get("extra_params", {}),
        )

    if gen_cfg["seed"] is not None:
        random.seed(gen_cfg["seed"])

    print("miUrba — Generation (config-only)")
    print("=" * 60)
    print(f"  Config:     {args.config or 'config.yaml (repo root)'}")
    print(f"  Output:     {output_dir}/")
    print(f"  Images:     {'enabled (FLUX.2)' if enable_images else 'disabled'}")
    print(f"  Seed:       {gen_cfg['seed']}")
    print(f"  Resume:     {gen_cfg['resume']}")
    print(f"  Workers:    {gen_cfg['workers']}")
    print(f"  Validate:   {gen_cfg['validate']}" +
          (f" (judge: {gen_cfg['validator']})" if gen_cfg.get("validate") and gen_cfg.get("validator") else ""))
    print(f"  Plan:       {', '.join(f'{m}×{c}' for m, c in plan)}")
    print("=" * 60)

    all_stats = []
    id_offset = gen_cfg["start_id"] - 1

    for model_name, count in plan:
        model_cfg = get_model_params(model_name, config)
        model_dir = f"{output_dir}/{model_name}"

        print(f"\n{'─' * 60}")
        print(f"  Model: {model_name} ({model_cfg['model']})")
        print(f"  Endpoint: {model_cfg['base_url']}")
        print(f"  Records: {count}")
        print(f"  IDs: Q{id_offset + 1:04d} — Q{id_offset + count:04d}")
        print(f"{'─' * 60}")

        client = LLMClient(
            base_url=model_cfg["base_url"],
            model=model_cfg["model"],
            temperature=model_cfg["temperature"],
            timeout=gen_cfg["timeout"],
            max_retries=gen_cfg["max_retries"],
            retry_delay=gen_cfg["retry_delay"],
            api_key=model_cfg.get("api_key"),
            model_load_wait=gen_cfg["model_load_wait"],
            connection_backoff=model_cfg.get("connection_backoff", False),
            extra_params=model_cfg.get("extra_params", {}),
        )

        stats = generate_dataset(
            client=client,
            num_records=count,
            start_id=id_offset + 1,
            output_dir=model_dir,
            raw_subdir=raw_subdir,
            images_subdir=paths.get("images_subdir", "images"),
            complaints_file=paths["complaints_file"],
            ground_truth_file=paths["ground_truth_file"],
            metadata_file=paths["metadata_file"],
            resume=gen_cfg["resume"],
            enable_images=enable_images,
            ig_config=ig_config,
            validate=gen_cfg["validate"],
            max_validation_retries=gen_cfg["max_validation_retries"],
            validator_client=validator_client,
            num_workers=gen_cfg["workers"],
        )

        stats["model"] = model_name
        all_stats.append(stats)
        id_offset += count

        print(f"\n  {model_name} done:")
        print(f"    Generated: {stats['generated']}")
        print(f"    Failed:    {stats['failed']}")
        print(f"    Tokens:    in={stats['total_tokens_in']:,} out={stats['total_tokens_out']:,}")

    _merge(config, output_dir)

    print(f"\n{'=' * 60}")
    print("Summary per model:")
    for s in all_stats:
        print(f"  {s['model']:15s}  gen={s['generated']}  fail={s['failed']}  "
              f"tok_in={s['total_tokens_in']:>8,}  tok_out={s['total_tokens_out']:>8,}")
    if client._stats.get("errors") or client._stats.get("load_waits"):
        print(f"  ⚠ errors absorbed: {client._stats['errors']}  "
              f"(load waits: {client._stats['load_waits']})")
    if validator_client is not None and (validator_client._stats.get("errors") or validator_client._stats.get("load_waits")):
        print(f"  ⚠ judge errors absorbed: {validator_client._stats['errors']}  "
              f"(load waits: {validator_client._stats['load_waits']})")
    print("=" * 60)


def _merge(config, output_dir: str | None = None) -> dict:
    """Merge per-model outputs into <output_dir>/<raw_subdir>/."""
    paths = get_paths(config)
    gen_cfg = get_generation_config(config)
    output_dir = output_dir or gen_cfg["default_output_dir"]
    plan = _generation_plan(gen_cfg)
    batch_dirs = [m for m, _ in plan]

    merge_stats = merge_outputs(
        output_dir,
        batch_dirs,
        raw_subdir=paths["raw_subdir"],
        complaints_file=paths["complaints_file"],
        ground_truth_file=paths["ground_truth_file"],
        metadata_file=paths["metadata_file"],
    )

    print(f"Merged {merge_stats['complaints']} complaints → {merge_stats['complaints_path']}")
    print(f"Merged {merge_stats['ground_truth']} ground_truth → {merge_stats['ground_truth_path']}")
    print(f"Merged {merge_stats['metadata']} metadata → {merge_stats['metadata_path']}")
    return merge_stats


def cmd_merge(args):
    """Merge per-model outputs into the shared raw dir (config-only)."""
    config = get_config(args.config)
    _merge(config)


def main():
    parser = argparse.ArgumentParser(
        description="miUrba — Generador de datasets sintéticos de quejas vecinales "
                    "(todos los parámetros se leen del config compartido)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Shared arguments (config-only) ─────────────────────
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None,
                        help="Path to shared config.yaml (default: ../config.yaml at repo root)")
    common.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers.add_parser("generate", parents=[common],
                          help="Generate synthetic raw complaints (plan from config)")
    subparsers.add_parser("merge", parents=[common],
                          help="Merge per-model outputs (plan from config)")

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
        "generate": cmd_generate,
        "merge": cmd_merge,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
