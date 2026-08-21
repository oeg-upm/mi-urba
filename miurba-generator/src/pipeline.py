"""
Pipeline functions — synthetic complaint GENERATION.

Orchestrates scenario sampling, LLM calls (text), optional image generation,
and file I/O to produce a dataset of raw complaints:

  - complaints.jsonl   (id, user_text, user_pics, timestamp)
  - ground_truth.jsonl  (complaint_id, extractions, image_descriptions)
  - metadata.jsonl      (complaint_id, persona, modifier, community, num_pics, image_prompts)
  - <images_subdir>/*.png  (actual photos, only if image generation is enabled)

The LLM produces ONLY user_text + timestamp. It does NOT describe images:
describing images is the extractor's job. When image generation is enabled,
the photos are generated from text-to-image prompts derived from the scenario.
"""

import json
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as cfg
from .llm import LLMClient
from .prompts import build_generation_prompt, build_image_prompts
from .sampling import (
    sample_community,
    sample_photo_count,
    sample_persona,
    sample_modifier,
    sample_scenario,
    should_be_multi_problem,
    should_have_photos,
)

logger = logging.getLogger(__name__)


# ── Validation ─────────────────────────────────────────────────

def _build_validation_prompt(user_text: str, problems: list[dict]) -> list[dict]:
    """Build a prompt to validate that user_text faithfully describes the scenario."""
    problem_descs = []
    for i, p in enumerate(problems, 1):
        desc = f"  {i}. {p['category']}/{p['subcategory']} en {p['location_zone']}"
        if p.get("location_detail"):
            desc += f", {p['location_detail']}"
        desc += f" (urgencia: {p['urgency']})"
        if p.get("affected_elements"):
            elem_strs = [f"{e['principal']} ({e['subcategory']})" for e in p["affected_elements"]]
            desc += f"\n     Elementos: {', '.join(elem_strs)}"
        if p.get("risks") and p["risks"] != ["ninguno"]:
            desc += f"\n     Riesgos: {', '.join(p['risks'])}"
        problem_descs.append(desc)
    problems_text = "\n".join(problem_descs)

    system_msg = (
        "Eres un validador de calidad de quejas sinteticas. Recibes un texto "
        "generado por un vecino y el escenario que se le pidio que describiera. "
        "Tu tarea es verificar que el texto ALIGINE con el escenario en todos sus campos."
    )

    user_msg = (
        f"ESCENARIO QUE SE LE PIDIO DESCRIBIR:\n{problems_text}\n\n"
        f"TEXTO GENERADO POR EL VECINO:\n{user_text}\n\n"
        f"Verifica si el texto alinea con el escenario en estos campos:\n"
        f"1. category: El texto describe un problema de la categoria indicada?\n"
        f"2. subcategory: El tipo de problema especifico es correcto?\n"
        f"3. urgency: El tono del texto refleja el nivel de urgencia indicado?\n"
        f"4. affected_elements: Menciona los elementos afectados indicados?\n"
        f"5. risks: Los riesgos mencionados son coherentes con los indicados?\n"
        f"6. location: La ubicacion mencionada es correcta?\n\n"
        f"RESPONDE SOLO con JSON:\n"
        f'{{"valid": true/false, "issues": ["lista de campos que no alinean, vacio si todo ok"]}}'
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def validate_complaint(client: LLMClient, user_text: str, problems: list[dict]) -> tuple[bool, list[str]]:
    """Validate that user_text faithfully describes the scenario.

    Returns (is_valid, issues_list).
    """
    messages = _build_validation_prompt(user_text, problems)
    try:
        response = client.chat(messages)
        data = client.extract_json(response)
        is_valid = data.get("valid", True)
        issues = data.get("issues", [])
        return bool(is_valid), issues if isinstance(issues, list) else []
    except Exception as e:
        logger.warning(f"Validation call failed: {e}, assuming valid")
        return True, []


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
        "timestamp": timestamp,
    }

    # Ground truth: aligned with extraction_schema.json
    extractions = [
        {
            "category": {"principal": p["category"], "subcategory": p["subcategory"]},
            "affected_elements": p["affected_elements"],
            "location": {"zone": p["location_zone"], "detail": p.get("location_detail", "")},
            "risks": p["risks"],
            "urgency": p["urgency"],
        }
        for p in problems
    ]

    ground_truth = {
        "complaint_id": record_id,
        "extractions": extractions,
        "image_descriptions": [],  # filled by extractor (VLM), not by generator
    }

    metadata = {
        "complaint_id": record_id,
        "persona": persona["id"] if persona else None,
        "modifier": modifier["id"] if modifier else None,
        "community": community,
        "num_pics": num_pics,
        "image_prompts": image_prompts,
    }

    return complaint, ground_truth, metadata


def generate_dataset(
    client: LLMClient,
    num_records: int,
    start_id: int = 1,
    output_dir: str = "data",
    raw_subdir: str = "raw",
    images_subdir: str = "images",
    complaints_file: str = "complaints.jsonl",
    ground_truth_file: str = "ground_truth.jsonl",
    metadata_file: str = "metadata.jsonl",
    resume: bool = True,
    enable_images: bool = False,
    ig_config: dict | None = None,
    validate: bool = False,
    max_validation_retries: int = 3,
    num_workers: int = 1,
) -> dict:
    """
    Generate raw complaint dataset.

    Produces:
      - <output_dir>/<raw_subdir>/<complaints_file>
      - <output_dir>/<raw_subdir>/<ground_truth_file>
      - <output_dir>/<raw_subdir>/<metadata_file>
      - <output_dir>/<images_subdir>/*.png  (only if enable_images is True)

    When image generation is enabled, the FLUX backend is loaded ONCE here
    and reused for every record.
    """
    raw_path = Path(output_dir) / raw_subdir / complaints_file
    ground_truth_path = Path(output_dir) / raw_subdir / ground_truth_file
    metadata_path = Path(output_dir) / raw_subdir / metadata_file
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
    write_lock = threading.Lock()

    # Build list of IDs to generate
    ids_to_generate = []
    for i in range(start_id, start_id + num_records):
        record_id = f"Q{i:04d}"
        if record_id not in existing_ids:
            ids_to_generate.append(record_id)
        else:
            generated += 1

    logger.info(f"Generating {len(ids_to_generate)} records with {num_workers} workers...")

    def _generate_one(record_id: str) -> tuple[str, dict | None]:
        """Generate one complaint (with validation if enabled). Returns (record_id, result_or_None)."""
        for attempt in range(max_validation_retries + 1) if validate else [0]:
            result = generate_complaint(
                client, record_id,
                image_gen=image_gen,
                images_dir=images_dir,
                seed_base=seed_base,
            )
            if result is None:
                return record_id, None

            complaint, ground_truth, metadata = result

            if not validate:
                return record_id, result

            problems_for_validation = [
                {
                    "category": e["category"]["principal"],
                    "subcategory": e["category"]["subcategory"],
                    "urgency": e["urgency"],
                    "location_zone": e["location"]["zone"],
                    "location_detail": e["location"].get("detail", ""),
                    "affected_elements": e.get("affected_elements", []),
                    "risks": e.get("risks", []),
                }
                for e in ground_truth["extractions"]
            ]
            is_valid, issues = validate_complaint(client, complaint["user_text"], problems_for_validation)

            if is_valid:
                if attempt > 0:
                    logger.info(f"[{record_id}] Validated after {attempt + 1} attempts")
                return record_id, result
            else:
                logger.info(f"[{record_id}] Invalid (attempt {attempt + 1}/{max_validation_retries + 1}): {issues}")
                if attempt < max_validation_retries:
                    logger.info(f"[{record_id}] Regenerating...")
                else:
                    logger.warning(f"[{record_id}] Kept despite validation issues: {issues}")
                    return record_id, result
        return record_id, None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_generate_one, rid): rid for rid in ids_to_generate}
        for future in as_completed(futures):
            record_id = futures[future]
            try:
                rid, result = future.result()
            except Exception as e:
                logger.error(f"[{record_id}] Worker error: {e}")
                failed += 1
                continue

            if result is None:
                failed += 1
                logger.warning(f"[{record_id}] Generation failed")
                continue

            complaint, ground_truth, metadata = result
            with write_lock:
                with open(raw_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(complaint, ensure_ascii=False) + "\n")
                with open(ground_truth_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ground_truth, ensure_ascii=False) + "\n")
                with open(metadata_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                generated += 1

                if generated % 10 == 0:
                    logger.info(
                        f"Progress: {generated}/{len(ids_to_generate)} generated, "
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
        "ground_truth_path": str(ground_truth_path),
        "metadata_path": str(metadata_path),
        "images_dir": str(images_dir) if image_gen is not None else None,
    }


# ── Merge (multi-model batches) ─────────────────────────────────

def merge_outputs(
    output_dir: str = "data",
    batch_dirs: list[str] | None = None,
    raw_subdir: str = "raw",
    complaints_file: str = "complaints.jsonl",
    ground_truth_file: str = "ground_truth.jsonl",
    metadata_file: str = "metadata.jsonl",
) -> dict:
    """
    Merge per-model output directories into final <output_dir>/<raw_subdir>/.
    """
    raw_out = Path(output_dir) / raw_subdir
    raw_out.mkdir(parents=True, exist_ok=True)

    merged_complaints = []
    merged_ground_truth = []
    merged_metadata = []

    for model_dir in (batch_dirs or []):
        model_path = Path(output_dir) / model_dir / raw_subdir

        for fname, merged in [
            (complaints_file, merged_complaints),
            (ground_truth_file, merged_ground_truth),
            (metadata_file, merged_metadata),
        ]:
            fpath = model_path / fname
            if fpath.exists():
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            merged.append(json.loads(line))

    merged_complaints.sort(key=lambda x: x["id"])
    merged_ground_truth.sort(key=lambda x: x["complaint_id"])
    merged_metadata.sort(key=lambda x: x["complaint_id"])

    complaints_path = raw_out / complaints_file
    ground_truth_path = raw_out / ground_truth_file
    metadata_path = raw_out / metadata_file

    with open(complaints_path, "w", encoding="utf-8") as f:
        for c in merged_complaints:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(ground_truth_path, "w", encoding="utf-8") as f:
        for g in merged_ground_truth:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    with open(metadata_path, "w", encoding="utf-8") as f:
        for m in merged_metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    logger.info(f"Merged {len(merged_complaints)} complaints → {complaints_path}")
    logger.info(f"Merged {len(merged_ground_truth)} ground_truth → {ground_truth_path}")
    logger.info(f"Merged {len(merged_metadata)} metadata → {metadata_path}")

    return {
        "complaints": len(merged_complaints),
        "ground_truth": len(merged_ground_truth),
        "metadata": len(merged_metadata),
        "complaints_path": str(complaints_path),
        "ground_truth_path": str(ground_truth_path),
        "metadata_path": str(metadata_path),
    }