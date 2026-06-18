"""
Pipeline functions — generation and annotation.

Orchestrates scenario sampling, LLM calls, and file I/O.
"""

import json
import logging
import random
from pathlib import Path

from . import config as cfg
from .llm import LLMClient
from .prompts import build_annotation_prompt, build_generation_prompt
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


# ── Generation ──────────────────────────────────────────────────

def generate_complaint(
    client: LLMClient,
    record_id: str,
    scenario: dict | None = None,
) -> tuple[dict, dict] | None:
    """
    Generate a single raw complaint record.

    Returns:
        Tuple of (complaint_dict, scenario_dict) or None on failure.
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

    # Build prompt and call LLM
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
    image_descriptions = data.get("image_descriptions", [])
    timestamp = data.get("timestamp", "")

    if not user_text:
        logger.warning(f"[{record_id}] No user_text in LLM output")
        return None

    # Pad/truncate image_descriptions to match num_pics
    while len(image_descriptions) < num_pics:
        image_descriptions.append(
            f"Imagen de {problems[0]['category']} en zona de {problems[0]['location_zone']}"
        )
    image_descriptions = image_descriptions[:num_pics]

    pic_names = [f"{record_id}_{i}.jpg" for i in range(1, num_pics + 1)]

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
        "image_descriptions": image_descriptions,
    }

    return complaint, scenario_ref


def generate_dataset(
    client: LLMClient,
    num_records: int,
    start_id: int = 1,
    output_dir: str = "data",
    resume: bool = True,
) -> dict:
    """
    Generate raw complaint dataset.

    Produces:
      - <output_dir>/raw/complaints.jsonl
      - <output_dir>/raw/scenarios.jsonl
    """
    raw_path = Path(output_dir) / "raw" / "complaints.jsonl"
    scenario_path = Path(output_dir) / "raw" / "scenarios.jsonl"

    raw_path.parent.mkdir(parents=True, exist_ok=True)

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

        result = generate_complaint(client, record_id)

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
        "total_tokens_in": client.stats["tokens_in"],
        "total_tokens_out": client.stats["tokens_out"],
        "raw_path": str(raw_path),
        "scenario_path": str(scenario_path),
    }


# ── Annotation ──────────────────────────────────────────────────

def annotate_complaint(
    client: LLMClient,
    complaint: dict,
    image_descriptions: list[str],
) -> dict | None:
    """
    Annotate a single complaint.

    Receives the same input the extractor would see in production
    (user_text + image_descriptions) and produces the structured extraction.
    """
    messages = build_annotation_prompt(complaint, image_descriptions)

    try:
        response = client.chat(messages)
        content = client.extract_content(response)

        if not content:
            logger.warning(f"[{complaint['id']}] Empty annotation content")
            return None

        try:
            extraction = client.extract_json(response)
        except json.JSONDecodeError as e:
            logger.warning(f"[{complaint['id']}] JSON parse failed: {e}")
            logger.debug(f"[{complaint['id']}] Raw content: {content[:500]}")
            return None

    except RuntimeError as e:
        logger.error(f"[{complaint['id']}] LLM call failed: {e}")
        return None

    # Ensure complaint_id is correct
    extraction["complaint_id"] = complaint["id"]

    # Ensure image_descriptions are included
    if "image_descriptions" not in extraction:
        extraction["image_descriptions"] = image_descriptions

    return extraction


def annotate_dataset(
    client: LLMClient,
    raw_dir: str = "data/raw",
    output_dir: str = "data/annotated",
    resume: bool = True,
) -> dict:
    """
    Annotate all raw complaints.

    Reads from <raw_dir>/complaints.jsonl and <raw_dir>/scenarios.jsonl,
    produces <output_dir>/extractions.jsonl.
    """
    raw_path = Path(raw_dir) / "complaints.jsonl"
    scenario_path = Path(raw_dir) / "scenarios.jsonl"
    output_path = Path(output_dir) / "extractions.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.error(f"Raw complaints not found: {raw_path}")
        return {"annotated": 0, "failed": 0, "skipped": 0}

    # Load complaints and scenarios
    complaints = {}
    with open(raw_path) as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                complaints[c["id"]] = c

    scenarios = {}
    if scenario_path.exists():
        with open(scenario_path) as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    scenarios[s["id"]] = s

    # Find already annotated IDs for resume
    existing_ids = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    existing_ids.add(e.get("complaint_id", ""))
        logger.info(f"Resuming: {len(existing_ids)} annotations already exist")

    annotated = 0
    failed = 0
    skipped = 0

    for cid, complaint in complaints.items():
        if cid in existing_ids:
            skipped += 1
            continue

        scenario = scenarios.get(cid, {})
        image_descriptions = scenario.get("image_descriptions", [])

        logger.info(f"[{cid}] Annotating ({annotated + 1}/{len(complaints) - skipped})...")

        extraction = annotate_complaint(client, complaint, image_descriptions)

        if extraction is None:
            failed += 1
            logger.warning(f"[{cid}] Annotation failed")
            continue

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(extraction, ensure_ascii=False) + "\n")

        annotated += 1

        if annotated % 10 == 0:
            logger.info(
                f"Progress: {annotated} annotated, {failed} failed, "
                f"{skipped} skipped, "
                f"tokens: in={client.stats['tokens_in']:,}, "
                f"out={client.stats['tokens_out']:,}"
            )

    return {
        "annotated": annotated,
        "failed": failed,
        "skipped": skipped,
        "total": len(complaints),
        "total_tokens_in": client.stats["tokens_in"],
        "total_tokens_out": client.stats["tokens_out"],
        "output_path": str(output_path),
    }


# ── Merge ───────────────────────────────────────────────────────

def merge_outputs(
    output_dir: str = "data",
    batch_dirs: list[str] | None = None,
) -> dict:
    """
    Merge per-model output directories into final data/raw/.

    Args:
        output_dir: Base data directory.
        batch_dirs: List of subdirectory names (one per model batch).

    Returns:
        Dict with counts.
    """
    raw_out = Path(output_dir) / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)

    merged_complaints = []
    merged_scenarios = []

    for model_dir in (batch_dirs or []):
        model_path = Path(output_dir) / model_dir / "raw"
        complaints_file = model_path / "complaints.jsonl"
        scenarios_file = model_path / "scenarios.jsonl"

        if complaints_file.exists():
            with open(complaints_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged_complaints.append(json.loads(line))

        if scenarios_file.exists():
            with open(scenarios_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged_scenarios.append(json.loads(line))

    # Sort by ID
    merged_complaints.sort(key=lambda x: x["id"])
    merged_scenarios.sort(key=lambda x: x["id"])

    complaints_path = raw_out / "complaints.jsonl"
    scenarios_path = raw_out / "scenarios.jsonl"

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