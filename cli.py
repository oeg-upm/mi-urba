#!/usr/bin/env python3
"""
miUrba — Synthetic complaint dataset generator and annotator.

Usage:
  python cli.py generate                   # Generate with config defaults
  python cli.py generate -n 100             # Generate 100 records
  python cli.py generate --model glm-5.1    # Use a different model
  python cli.py generate-multi              # Multi-model batch generation
  python cli.py annotate                    # Annotate raw complaints
  python cli.py merge                       # Merge per-model outputs

All configuration comes from config.yaml — no hardcoded values.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config, get_model_params
from src.llm import LLMClient
from src.pipeline import annotate_dataset, generate_dataset, merge_outputs


def cmd_generate(args):
    """Generate synthetic raw complaints with a single model."""
    config = get_config(args.config)
    model_name = args.model
    model_cfg = get_model_params(model_name, config)

    client = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        temperature=model_cfg["temperature"],
        timeout=args.timeout,
    )

    print(f"miUrba — Generation (single model)")
    print(f"{'=' * 55}")
    print(f"  Model:      {model_name} ({model_cfg['model']})")
    print(f"  Endpoint:   {model_cfg['base_url']}")
    print(f"  Records:    {args.num_records}")
    print(f"  IDs:        Q{args.start_id:04d} — Q{args.start_id + args.num_records - 1:04d}")
    print(f"  Output:     {args.output_dir}/raw/")
    print(f"  Resume:     {not args.no_resume}")
    print(f"{'=' * 55}")

    stats = generate_dataset(
        client=client,
        num_records=args.num_records,
        start_id=args.start_id,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )

    print(f"\n{'=' * 55}")
    print(f"Generation complete!")
    print(f"  Generated:  {stats['generated']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Tokens in:  {stats['total_tokens_in']:,}")
    print(f"  Tokens out: {stats['total_tokens_out']:,}")
    print(f"  Raw:        {stats['raw_path']}")
    print(f"  Scenarios:  {stats['scenario_path']}")


def cmd_generate_multi(args):
    """Generate with multiple models as configured in config.yaml."""
    config = get_config(args.config)
    gen_cfg = config["generation"]
    batches = gen_cfg["batches"]
    output_dir = args.output_dir or gen_cfg["default_output_dir"]

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
        )

        model_dir = f"{output_dir}/{model_name}"

        stats = generate_dataset(
            client=client,
            num_records=count,
            start_id=id_offset + 1,
            output_dir=model_dir,
            resume=not args.no_resume,
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
    merge_stats = merge_outputs(output_dir, batch_dirs)

    print(f"  Merged: {merge_stats['complaints']} complaints → {merge_stats['complaints_path']}")
    print(f"  Merged: {merge_stats['scenarios']} scenarios → {merge_stats['scenarios_path']}")

    print(f"\n{'=' * 60}")
    print("Summary per model:")
    for s in all_stats:
        print(f"  {s['model']:15s}  gen={s['generated']}  fail={s['failed']}  "
              f"tok_in={s['total_tokens_in']:>8,}  tok_out={s['total_tokens_out']:>8,}")
    print(f"{'=' * 60}")
    print(f"Total merged: {merge_stats['complaints']} complaints")


def cmd_annotate(args):
    """Annotate raw complaints with structured extractions."""
    config = get_config(args.config)
    ann_cfg = config.get("annotation", {})
    model_name = args.model or ann_cfg.get("model", "kimi-k2.6")
    model_cfg = get_model_params(model_name, config)

    temperature = args.temperature or ann_cfg.get("temperature", 0.2)

    client = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        temperature=temperature,
        timeout=args.timeout,
    )

    print(f"miUrba — Annotation (raw → structured)")
    print(f"{'=' * 55}")
    print(f"  Model:      {model_name} ({model_cfg['model']})")
    print(f"  Endpoint:   {model_cfg['base_url']}")
    print(f"  Source:     {args.raw_dir}/complaints.jsonl")
    print(f"  Output:     {args.output_dir}/annotated/extractions.jsonl")
    print(f"  Resume:     {not args.no_resume}")
    print(f"{'=' * 55}")

    stats = annotate_dataset(
        client=client,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )

    print(f"\n{'=' * 55}")
    print(f"Annotation complete!")
    print(f"  Annotated:  {stats['annotated']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Skipped:    {stats['skipped']}")
    print(f"  Total:      {stats['total']}")
    print(f"  Tokens in:  {stats['total_tokens_in']:,}")
    print(f"  Tokens out: {stats['total_tokens_out']:,}")
    print(f"  Output:     {stats['output_path']}")


def cmd_merge(args):
    """Merge per-model outputs into data/raw/."""
    config = get_config(args.config)
    gen_cfg = config["generation"]
    output_dir = args.output_dir or gen_cfg["default_output_dir"]
    batch_dirs = [b["model"] for b in gen_cfg["batches"]]

    merge_stats = merge_outputs(output_dir, batch_dirs)

    print(f"Merged {merge_stats['complaints']} complaints → {merge_stats['complaints_path']}")
    print(f"Merged {merge_stats['scenarios']} scenarios → {merge_stats['scenarios_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="miUrba — Synthetic complaint dataset generator and annotator"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Shared arguments ───────────────────────────────────
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    common.add_argument("--output-dir", default=None, help="Output directory (default: from config)")
    common.add_argument("--no-resume", action="store_true", help="Start fresh")
    common.add_argument("--timeout", type=int, default=300, help="Request timeout (seconds)")
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

    # ── Annotate ────────────────────────────────────────────
    ann_parser = subparsers.add_parser("annotate", parents=[common],
                                        help="Annotate raw complaints")
    ann_parser.add_argument("--raw-dir", default="data/raw",
                             help="Directory with raw complaints")
    ann_parser.add_argument("--model", default=None,
                             help="Model for annotation (default: from config)")
    ann_parser.add_argument("--temperature", type=float, default=None,
                             help="Temperature for annotation")

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
        "annotate": cmd_annotate,
        "merge": cmd_merge,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()