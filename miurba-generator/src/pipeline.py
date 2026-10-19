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


# ── Validation (2-stage) ──────────────────────────────────────────────

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
        "Tu tarea es verificar que el texto ALINEA con el escenario en todos sus campos."
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

    Fail-open: any validation error (LLM call, malformed JSON) counts as
    valid so a broken judge never blocks dataset generation.

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
    validate: bool = False,
    max_validation_retries: int = 3,
    validator_client: LLMClient | None = None,
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

    # 2-stage validation: judge LLM checks text/scenario alignment. On
    # mismatch, REGENERATE (same scenario, new text) up to
    # max_validation_retries times; the last text is always kept.
    if validate:
        judge = validator_client or client
        for attempt in range(max_validation_retries):
            is_valid, issues = validate_complaint(judge, user_text, problems)
            if is_valid:
                break
            logger.warning(
                f"[{record_id}] validation failed ({', '.join(map(str, issues)) or 'unknown'}); "
                f"regenerating (attempt {attempt + 1}/{max_validation_retries})"
            )
            try:
                response = client.chat(messages)
                data = client.extract_json(response)
                regen_text = data.get("user_text", "")
                if not regen_text:
                    logger.warning(f"[{record_id}] regeneration produced no user_text, keeping previous")
                    break
                user_text = regen_text
                timestamp = data.get("timestamp", timestamp)
            except (RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"[{record_id}] regeneration LLM call failed: {e}, keeping previous text")
                break

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
    validator_client: LLMClient | None = None,
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

    def _generate_and_write(record_id: str) -> bool:
        """Generate one record and append it to the JSONL files (thread-safe)."""
        result = generate_complaint(
            client, record_id,
            image_gen=image_gen,
            images_dir=images_dir,
            seed_base=seed_base,
            validate=validate,
            max_validation_retries=max_validation_retries,
            validator_client=validator_client,
        )
        if result is None:
            logger.warning(f"[{record_id}] Generation failed")
            return False

        complaint, ground_truth, metadata = result
        with write_lock:
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(complaint, ensure_ascii=False) + "\n")
            with open(ground_truth_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ground_truth, ensure_ascii=False) + "\n")
            with open(metadata_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        return True

    pending = [f"Q{i:04d}" for i in range(start_id, start_id + num_records)
               if f"Q{i:04d}" not in existing_ids]
    generated += num_records - len(pending)  # resumed records count as done

    if num_workers > 1:
        logger.info(f"Generating with {num_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_generate_and_write, rid): rid for rid in pending}
            for n, future in enumerate(as_completed(futures), 1):
                if future.result():
                    generated += 1
                else:
                    failed += 1
                if n % 10 == 0:
                    logger.info(
                        f"Progress: {n}/{len(pending)} processed, {failed} failed, "
                        f"tokens: in={client.stats['tokens_in']:,}, "
                        f"out={client.stats['tokens_out']:,}"
                    )
    else:
        for record_id in pending:
            logger.info(f"[{record_id}] Generating ({generated + 1}/{num_records})...")
            if _generate_and_write(record_id):
                generated += 1
            else:
                failed += 1

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