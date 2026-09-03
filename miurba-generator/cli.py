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

Uso:
  python cli.py generate                   # Generar con config por defecto
  python cli.py generate -n 100            # Generar 100 registros
  python cli.py generate --model glm-5.1   # Usar otro modelo
  python cli.py generate-multi             # Generación por lotes multi-modelo
  python cli.py merge                      # Fusionar salidas por modelo

Toda la configuración se lee de la config COMPARTIDA en la raíz del
repositorio (../config.yaml por defecto, o la indicada con --config).
"""

import argparse
import logging
import sys
from pathlib import Path

# Add this project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config, get_model_params, get_paths, get_image_generation_config
from src.llm import LLMClient
from src.pipeline import generate_dataset, merge_outputs


def _get_path_args(args, config):
    """Build path kwargs from CLI overrides or config defaults."""
    paths = get_paths(config)
    return {
        "raw_subdir": args.raw_subdir or paths.get("raw_subdir", "raw"),
        "annotated_subdir": args.annotated_subdir or paths.get("annotated_subdir", "annotated"),
        "complaints_file": args.complaints_file or paths.get("complaints_file", "complaints.jsonl"),
        "ground_truth_file": args.ground_truth_file or paths.get("ground_truth_file", "ground_truth.jsonl"),
        "metadata_file": args.metadata_file or paths.get("metadata_file", "metadata.jsonl"),
        "extractions_file": args.extractions_file or paths.get("extractions_file", "extractions.jsonl"),
    }


def cmd_generate(args):
    """Generate synthetic raw complaints with a single model."""
    if args.seed is not None:
        import random; random.seed(args.seed)
    config = get_config(args.config)
    paths = _get_path_args(args, config)
    model_name = args.model
    model_cfg = get_model_params(model_name, config)

    client = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        temperature=model_cfg["temperature"],
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        api_key=model_cfg.get("api_key"),
        extra_params=model_cfg.get("extra_params", {}),
    )

    raw_subdir = paths["raw_subdir"]
    ig_config = get_image_generation_config(config)
    enable_images = bool(ig_config.get("enabled", False)) and not args.no_images
    print(f"miUrba — Generation (single model)")
    print(f"{'=' * 55}")
    print(f"  Model:      {model_name} ({model_cfg['model']})")
    print(f"  Endpoint:   {model_cfg['base_url']}")
    print(f"  Records:    {args.num_records}")
    print(f"  IDs:        Q{args.start_id:04d} — Q{args.start_id + args.num_records - 1:04d}")
    print(f"  Output:     {args.output_dir}/{raw_subdir}/")
    print(f"  Images:     {'enabled (FLUX.2)' if enable_images else 'disabled'}")
    print(f"  Resume:     {not args.no_resume}")
    print(f"{'=' * 55}")

    stats = generate_dataset(
        client=client,
        num_records=args.num_records,
        start_id=args.start_id,
        output_dir=args.output_dir,
        raw_subdir=raw_subdir,
        images_subdir=paths.get("images_subdir", "images"),
        complaints_file=paths["complaints_file"],
        ground_truth_file=paths["ground_truth_file"],
        metadata_file=paths["metadata_file"],
        resume=not args.no_resume,
        enable_images=enable_images,
        ig_config=ig_config,
        validate=args.validate,
        max_validation_retries=args.max_validation_retries,
        num_workers=args.workers,
    )

    print(f"\n{'=' * 55}")
    print(f"Generation complete!")
    print(f"  Generated:  {stats['generated']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Tokens in:  {stats['total_tokens_in']:,}")
    print(f"  Tokens out: {stats['total_tokens_out']:,}")
    print(f"  Raw:        {stats['raw_path']}")
    print(f"  Ground truth: {stats['ground_truth_path']}")
    print(f"  Metadata:    {stats['metadata_path']}")
    print(f"  Images dir: {stats.get('images_dir')}")


def cmd_generate_multi(args):
    """Generate with multiple models as configured in config.yaml."""
    if args.seed is not None:
        import random; random.seed(args.seed)
    config = get_config(args.config)
    paths = _get_path_args(args, config)
    gen_cfg = config["generation"]
    batches = gen_cfg["batches"]
    output_dir = args.output_dir or gen_cfg["default_output_dir"]
    raw_subdir = paths["raw_subdir"]
    ig_config = get_image_generation_config(config)
    enable_images = bool(ig_config.get("enabled", False)) and not args.no_images

    print("miUrba — Multi-model generation")
    print("=" * 60)

    all_stats = []
    id_offset = 0

    for batch in batches:
        model_name = batch["model"]
        count = batch["count"]
        model_cfg = get_model_params(model_name, config)

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
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            api_key=model_cfg.get("api_key"),
            extra_params=model_cfg.get("extra_params", {}),
        )

        model_dir = f"{output_dir}/{model_name}"

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
            resume=not args.no_resume,
            enable_images=enable_images,
            ig_config=ig_config,
        )

        stats["model"] = model_name
        all_stats.append(stats)
        id_offset += count

        print(f"\n  {model_name} done:")
        print(f"    Generated: {stats['generated']}")
        print(f"    Failed:    {stats['failed']}")
        print(f"    Tokens:    in={stats['total_tokens_in']:,} out={stats['total_tokens_out']:,}")

    # Merge
    print(f"\n{'=' * 60}")
    print("Merging outputs...")
    batch_dirs = [b["model"] for b in batches]
    merge_stats = merge_outputs(
        output_dir,
        batch_dirs,
        raw_subdir=raw_subdir,
        complaints_file=paths["complaints_file"],
        ground_truth_file=paths["ground_truth_file"],
        metadata_file=paths["metadata_file"],
    )

    print(f"  Merged: {merge_stats['complaints']} complaints → {merge_stats['complaints_path']}")
    print(f"  Merged: {merge_stats['ground_truth']} ground_truth → {merge_stats['ground_truth_path']}")
    print(f"  Merged: {merge_stats['metadata']} metadata → {merge_stats['metadata_path']}")

    print(f"\n{'=' * 60}")
    print("Summary per model:")
    for s in all_stats:
        print(f"  {s['model']:15s}  gen={s['generated']}  fail={s['failed']}  "
              f"tok_in={s['total_tokens_in']:>8,}  tok_out={s['total_tokens_out']:>8,}")
    print(f"{'=' * 60}")
    print(f"Total merged: {merge_stats['complaints']} complaints")


def cmd_merge(args):
    """Merge per-model outputs into data/raw/."""
    config = get_config(args.config)
    paths = _get_path_args(args, config)
    gen_cfg = config["generation"]
    output_dir = args.output_dir or gen_cfg["default_output_dir"]
    batch_dirs = [b["model"] for b in gen_cfg["batches"]]

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


def main():
    parser = argparse.ArgumentParser(
        description="miUrba — Generador de datasets sintéticos de quejas vecinales"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Shared arguments ───────────────────────────────────
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None,
                        help="Path to shared config.yaml (default: ../config.yaml at repo root)")
    common.add_argument("--output-dir", default=None, help="Output directory (default: from config)")
    common.add_argument("--raw-subdir", default=None, help="Raw subdirectory name (default: from config)")
    common.add_argument("--annotated-subdir", default=None, help="Annotated subdirectory name (default: from config)")
    common.add_argument("--raw-dir", default=None, help="Raw directory path (default: <output_dir>/<raw_subdir>)")
    common.add_argument("--annotated-dir", default=None, help="Annotated directory path (default: <output_dir>/<annotated_subdir>)")
    common.add_argument("--complaints-file", default=None, help="Complaints filename (default: from config)")
    common.add_argument("--ground-truth-file", default=None, help="Ground truth filename (default: from config)")
    common.add_argument("--metadata-file", default=None, help="Metadata filename (default: from config)")
    common.add_argument("--extractions-file", default=None, help="Extractions filename (default: from config)")
    common.add_argument("--no-resume", action="store_true", help="Start fresh")
    common.add_argument("--timeout", type=int, default=300, help="Request timeout (seconds)")
    common.add_argument("--max-retries", type=int, default=3, help="Max API retries per request")
    common.add_argument("--retry-delay", type=float, default=5.0, help="Base retry delay (seconds)")
    common.add_argument("--no-images", action="store_true",
                        help="Disable image generation even if enabled in config")
    common.add_argument("--validate", action="store_true",
                        help="Validate generated complaints against GT (2-stage: generate + validate)")
    common.add_argument("--max-validation-retries", type=int, default=3,
                        help="Max validation retries before keeping invalid complaint")
    common.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    common.add_argument("--workers", type=int, default=1, help="Number of concurrent generation workers (default: 1)")
    common.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # ── Generate ────────────────────────────────────────────
    gen_parser = subparsers.add_parser("generate", parents=[common],
                                        help="Generate synthetic raw complaints")
    gen_parser.add_argument("-n", "--num-records", type=int, default=100,
                             help="Number of records")
    gen_parser.add_argument("--start-id", type=int, default=1,
                             help="Starting ID number")
    gen_parser.add_argument("--model", default="kimi-k2.6",
                             help="Model name (from config.yaml)")

    # ── Generate multi ──────────────────────────────────────
    multi_parser = subparsers.add_parser("generate-multi", parents=[common],
                                          help="Multi-model batch generation")

    # ── Merge ───────────────────────────────────────────────
    merge_parser = subparsers.add_parser("merge", parents=[common],
                                          help="Merge per-model outputs")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    commands = {
        "generate": cmd_generate,
        "generate-multi": cmd_generate_multi,
        "merge": cmd_merge,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()