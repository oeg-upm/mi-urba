"""
Pipeline functions — synthetic complaint GENERATION.

Orchestrates scenario sampling, LLM calls (text), optional image generation,
and file I/O to produce a dataset of raw complaints:

  - complaints.jsonl  (id, user_text, user_pics, channel, timestamp, ...)
  - scenarios.jsonl    (ground truth: problems, community, ..., image_prompts)
  - <images_subdir>/*.png  (actual photos, only if image generation is enabled)

The LLM produces ONLY user_text + timestamp. It does NOT describe images:
describing images is the extractor's job. When image generation is enabled,
the photos are generated from text-to-image prompts derived from the scenario.
"""

import json
import logging
import random
from pathlib import Path

from . import config as cfg
from .llm import LLMClient
from .prompts import build_generation_prompt, build_image_prompts
from .sampling import (
    sample_channel,
    sample_community,
    sample_photo_count,
    sample_persona,
    sample_modifier,
    sample_scenario,
    should_be_multi_problem,
    should_have_photos,
)

logger = logging.getLogger(__name__)


def _stable_seed(record_id: str, index: int, seed_base: int) -> int:
    """Deterministic seed for reproducible image generation."""
    h = sum((i + 1) * ord(c) for i, c in enumerate(record_id))
    return (h * 1000 + index + seed_base) % (2**32)


# ── Generation ──────────────────────────────────────────────────

def generate_complaint(
    client: LLMClient,
    record_id: str,
    scenario: dict | None = None,
    image_gen=None,
    images_dir: str | Path | None = None,
    seed_base: int = 42,
) -> tuple[dict, dict] | None:
    """
    Generate a single raw complaint record (and optionally its photos).

    Returns:
        Tuple of (complaint_dict, scenario_dict) or None on failure.

    When `image_gen` is provided and the complaint has photos, the actual
        PNG files are generated to `images_dir`.
    """
    if scenario is None:
        scenario = sample_scenario()

    community = sample_community()
    channel = sample_channel()
    persona = sample_persona()
    modifier = sample_modifier()

    # Multi-problem?
    if should_be_multi_problem():
        n_problems = random.choices([2, 3], weights=[0.75, 0.25], k=1)[0]
    else:
        n_problems = 1

    problems = [scenario] + [sample_scenario() for _ in range(n_problems - 1)]

    # Photos?
    has_photos = should_have_photos()
    num_pics = sample_photo_count(has_photos)

    # Build prompt and call LLM (text only: user_text + timestamp)
    messages = build_generation_prompt(
        record_id=record_id,
        problems=problems,
        channel=channel,
        community=community,
        num_pics=num_pics,
        persona=persona,
        modifier=modifier,
    )

    try:
        response = client.chat(messages)
        content = client.extract_content(response)

        if not content:
            logger.warning(f"[{record_id}] Empty content, skipping")
            return None

        try:
            data = client.extract_json(response)
        except json.JSONDecodeError as e:
            logger.warning(f"[{record_id}] JSON parse failed: {e}")
            logger.debug(f"[{record_id}] Raw content: {content[:500]}")
            return None

    except RuntimeError as e:
        logger.error(f"[{record_id}] LLM call failed: {e}")
        return None

    user_text = data.get("user_text", "")
    timestamp = data.get("timestamp", "")

    if not user_text:
        logger.warning(f"[{record_id}] No user_text in LLM output")
        return None

    # Photo filenames (.png); actual files generated below if enabled
    pic_names = [f"{record_id}_{i}.png" for i in range(1, num_pics + 1)]

    # Optional image generation from scenario-derived prompts
    image_prompts: list[str] = []
    if image_gen is not None and num_pics > 0:
        prompts = build_image_prompts(problems, community, num_pics)
        img_dir = Path(images_dir) if images_dir else Path("data/images")
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(prompts, 1):
            out_path = str(img_dir / f"{record_id}_{i}.png")
            if Path(out_path).exists():
                logger.debug(f"[{record_id}] image {i} exists, skipping")
            else:
                logger.info(f"[{record_id}] Generating image {i}/{num_pics}...")
                try:
                    image_gen.generate(p, _stable_seed(record_id, i, seed_base), out_path)
                except Exception as e:
                    logger.error(f"[{record_id}] image {i} generation failed: {e}")
            image_prompts.append(p)

    complaint = {
        "id": record_id,
        "user_text": user_text,
        "user_pics": pic_names,
        "channel": channel,
        "timestamp": timestamp,
        "community_ref": community["id"],
        "persona": persona["id"] if persona else None,
    }

    scenario_ref = {
        "id": record_id,
        "problems": problems,
        "community": community,
        "channel": channel,
        "persona": persona["id"] if persona else None,
        "modifier": modifier["id"] if modifier else None,
        "num_pics": num_pics,
        "image_prompts": image_prompts,
    }

    return complaint, scenario_ref


def generate_dataset(
    client: LLMClient,
    num_records: int,
    start_id: int = 1,
    output_dir: str = "data",
    raw_subdir: str = "raw",
    images_subdir: str = "images",
    complaints_file: str = "complaints.jsonl",
    scenarios_file: str = "scenarios.jsonl",
    resume: bool = True,
    enable_images: bool = False,
    ig_config: dict | None = None,
) -> dict:
    """
    Generate raw complaint dataset.

    Produces:
      - <output_dir>/<raw_subdir>/<complaints_file>
      - <output_dir>/<raw_subdir>/<scenarios_file>
      - <output_dir>/<images_subdir>/*.png  (only if enable_images is True)

    When image generation is enabled, the FLUX backend is loaded ONCE here
    and reused for every record.
    """
    raw_path = Path(output_dir) / raw_subdir / complaints_file
    scenario_path = Path(output_dir) / raw_subdir / scenarios_file
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional image generation backend (loaded once)
    image_gen = None
    images_dir = Path(output_dir) / images_subdir
    seed_base = 42
    if enable_images:
        ig = ig_config or {}
        seed_base = ig.get("seed_base", 42)
        try:
            from .imagegen import FluxImageGenerator
            print("Loading image generation backend (FLUX.2)...")
            image_gen = FluxImageGenerator(
                cache=ig.get("cache", "diffusers/FLUX.2-dev-bnb-4bit"),
                size=ig.get("size", 1024),
                steps=ig.get("steps", 28),
                guidance=ig.get("guidance", 4.0),
                device0=ig.get("device0", "cuda:0"),
                device1=ig.get("device1", "cuda:1"),
            )
            image_gen.load()
            images_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Image generation backend unavailable: {e}")
            print(f"⚠️  Image generation disabled (backend error): {e}")
            image_gen = None

    # Find already generated IDs for resume
    existing_ids = set()
    if resume and raw_path.exists():
        with open(raw_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_ids.add(rec.get("id", ""))
        logger.info(f"Resuming: {len(existing_ids)} records already exist")

    generated = 0
    failed = 0

    for i in range(start_id, start_id + num_records):
        record_id = f"Q{i:04d}"

        if record_id in existing_ids:
            logger.debug(f"[{record_id}] Already exists, skipping")
            generated += 1
            continue

        logger.info(f"[{record_id}] Generating ({generated + 1}/{num_records})...")

        result = generate_complaint(
            client, record_id,
            image_gen=image_gen,
            images_dir=images_dir,
            seed_base=seed_base,
        )

        if result is None:
            failed += 1
            logger.warning(f"[{record_id}] Generation failed")
            continue

        complaint, scenario_ref = result

        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(complaint, ensure_ascii=False) + "\n")

        with open(scenario_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(scenario_ref, ensure_ascii=False) + "\n")

        generated += 1

        if generated % 10 == 0:
            logger.info(
                f"Progress: {generated}/{num_records} generated, "
                f"{failed} failed, "
                f"tokens: in={client.stats['tokens_in']:,}, "
                f"out={client.stats['tokens_out']:,}"
            )

    return {
        "generated": generated,
        "failed": failed,
        "images_enabled": image_gen is not None,
        "total_tokens_in": client.stats["tokens_in"],
        "total_tokens_out": client.stats["tokens_out"],
        "raw_path": str(raw_path),
        "scenario_path": str(scenario_path),
        "images_dir": str(images_dir) if image_gen is not None else None,
    }


# ── Merge (multi-model batches) ─────────────────────────────────

def merge_outputs(
    output_dir: str = "data",
    batch_dirs: list[str] | None = None,
    raw_subdir: str = "raw",
    complaints_file: str = "complaints.jsonl",
    scenarios_file: str = "scenarios.jsonl",
) -> dict:
    """
    Merge per-model output directories into final <output_dir>/<raw_subdir>/.
    """
    raw_out = Path(output_dir) / raw_subdir
    raw_out.mkdir(parents=True, exist_ok=True)

    merged_complaints = []
    merged_scenarios = []

    for model_dir in (batch_dirs or []):
        model_path = Path(output_dir) / model_dir / raw_subdir
        complaints_file_path = model_path / complaints_file
        scenarios_file_path = model_path / scenarios_file

        if complaints_file_path.exists():
            with open(complaints_file_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged_complaints.append(json.loads(line))

        if scenarios_file_path.exists():
            with open(scenarios_file_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged_scenarios.append(json.loads(line))

    merged_complaints.sort(key=lambda x: x["id"])
    merged_scenarios.sort(key=lambda x: x["id"])

    complaints_path = raw_out / complaints_file
    scenarios_path = raw_out / scenarios_file

    with open(complaints_path, "w", encoding="utf-8") as f:
        for c in merged_complaints:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(scenarios_path, "w", encoding="utf-8") as f:
        for s in merged_scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Merged {len(merged_complaints)} complaints → {complaints_path}")
    logger.info(f"Merged {len(merged_scenarios)} scenarios → {scenarios_path}")

    return {
        "complaints": len(merged_complaints),
        "scenarios": len(merged_scenarios),
        "complaints_path": str(complaints_path),
        "scenarios_path": str(scenarios_path),
    }