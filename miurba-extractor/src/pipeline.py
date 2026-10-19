"""
Pipeline functions — structured EXTRACTION from raw complaints + images.

Orchestrates VLM calls and file I/O to turn a raw complaint dataset
(complaints.jsonl + the actual photo files) into structured extractions
(extractions.jsonl) conforming to schemas/extraction_schema.json at the
repository root.

The extractor uses a VLM (vision-language model) that sees the user_text AND
the photos. It produces both the image_descriptions (describing the photos)
and the structured extractions. Describing images is the extractor's job —
the generator no longer provides image descriptions.

It does NOT generate synthetic complaints; that is the generator project's job.
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import get_vision_model_params
from .llm import LLMClient
from .prompts import build_extraction_prompt, build_image_description_prompt, build_extraction_text_prompt

logger = logging.getLogger(__name__)


def _resolve_image_paths(complaint: dict, images_dir: str | Path) -> list[str]:
    """Resolve the photo file paths for a complaint from its user_pics."""
    img_dir = Path(images_dir)
    return [str(img_dir / pic) for pic in complaint.get("user_pics", [])]


def validate_extraction(extraction: dict, schema: dict) -> str | None:
    """Validate an extraction record against the schema.

    Returns None when valid, or the human-readable error reason. This is
    ALWAYS applied by the CLI regardless of the model's structured_output
    mode: that flag controls sampling constraints server-side, not whether
    our code trusts the output.
    """
    try:
        import jsonschema
    except ImportError:
        logger.warning("jsonschema not installed — extraction output NOT validated")
        return None
    try:
        jsonschema.validate(instance=extraction, schema=schema)
        return None
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        return f"{e.message} (at {path})"


def extract_complaint(
    client: LLMClient,
    complaint: dict,
    images_dir: str | Path,
    stage_split: bool = False,
    vision_client: LLMClient | None = None,
    validation_schema: dict | None = None,
) -> dict | None:
    """
    Extract structured data + image descriptions from a single complaint.

    Default (single-call): the VLM sees user_text + photos and produces
    {image_descriptions, extractions} conforming to
    schemas/extraction_schema.json.

    If stage_split=True (P2) and there are photos and a vision_client:
      1. vision_client describes photos (multimodal, image_description schema)
      2. client extracts structure from text + descriptions (text-only)
    Falls back to the single-call path when there are no photos or no
    vision client.
    """
    image_paths = _resolve_image_paths(complaint, images_dir)

    # P2 stage split: only when there are photos and a vision client
    if stage_split and image_paths and vision_client is not None:
        result = _extract_complaint_stage_split(client, vision_client, complaint, image_paths)
        if result is not None and validation_schema is not None:
            reason = validate_extraction(result, validation_schema)
            if reason:
                logger.warning(f"[{complaint['id']}] schema validation failed: {reason}")
                return None
        return result

    messages = build_extraction_prompt(complaint, image_paths)

    try:
        response = client.chat(messages)
        content = client.extract_content(response)

        if not content:
            logger.warning(f"[{complaint['id']}] Empty extraction content")
            return None

        try:
            extraction = client.extract_json(response)
        except json.JSONDecodeError as e:
            logger.warning(f"[{complaint['id']}] JSON parse failed: {e}")
            logger.debug(f"[{complaint['id']}] Raw content: {content[:500]}")
            return None

    except RuntimeError as e:
        logger.error(f"[{complaint['id']}] VLM call failed: {e}")
        return None

    # Ensure complaint_id is correct
    extraction["complaint_id"] = complaint["id"]

    # Normalize keys the model may hispanize
    if "extractions" not in extraction and "extracciones" in extraction:
        extraction["extractions"] = extraction.pop("extracciones")

    if validation_schema is not None:
        reason = validate_extraction(extraction, validation_schema)
        if reason:
            logger.warning(f"[{complaint['id']}] schema validation failed: {reason}")
            return None

    # Ensure image_descriptions is present (default to [] if the model omitted it)
    if "image_descriptions" not in extraction:
        extraction["image_descriptions"] = []
    if not isinstance(extraction["image_descriptions"], list):
        extraction["image_descriptions"] = []

    # extractions MUST be a list: anything else means the model output is
    # untrustworthy — fail the record (caller counts it as failed) instead
    # of writing schema-violating garbage to extractions.jsonl.
    if not isinstance(extraction.get("extractions"), list):
        logger.warning(f"[{complaint['id']}] 'extractions' is not a list — failing record")
        return None

    return extraction


def _extract_complaint_stage_split(
    text_client: LLMClient,
    vision_client: LLMClient,
    complaint: dict,
    image_paths: list[str],
) -> dict | None:
    """P2 two-call pipeline: vision describes, then text extracts."""
    # Stage 1: describe photos (vision)
    desc_messages = build_image_description_prompt(complaint, image_paths)
    try:
        desc_response = vision_client.chat(desc_messages)
        desc_data = vision_client.extract_json(desc_response)
    except (RuntimeError, json.JSONDecodeError) as e:
        logger.warning(f"[{complaint['id']}] Stage 1 (vision) failed: {e}")
        desc_data = None

    # Stage 2: extract structure (text-only), using descriptions if available
    descriptions = []
    if desc_data and isinstance(desc_data.get("descriptions"), list):
        descriptions = desc_data["descriptions"]
        logger.info(f"[{complaint['id']}] Stage 1 produced {len(descriptions)} descriptions")

    ext_messages = build_extraction_text_prompt(complaint, descriptions)
    try:
        ext_response = text_client.chat(ext_messages)
        content = text_client.extract_content(ext_response)
        if not content:
            logger.warning(f"[{complaint['id']}] Stage 2 empty content")
            return None
        extraction = text_client.extract_json(ext_response)
    except (RuntimeError, json.JSONDecodeError) as e:
        logger.error(f"[{complaint['id']}] Stage 2 (text) failed: {e}")
        return None

    # Normalize: image_descriptions as list of strings (captions)
    if not isinstance(extraction.get("image_descriptions"), list):
        extraction["image_descriptions"] = \
            [d.get("caption", "") for d in descriptions] if descriptions else []
    extraction["complaint_id"] = complaint["id"]
    if not isinstance(extraction.get("extractions"), list):
        extraction["extractions"] = []
    return extraction


def extract_dataset(
    client: LLMClient,
    raw_dir: str = "data/raw",
    images_dir: str | None = None,
    output_dir: str = "data/annotated",
    raw_complaints_file: str = "complaints.jsonl",
    extractions_file: str = "extractions.jsonl",
    resume: bool = True,
    stage_split: bool = False,
    vision_client: LLMClient | None = None,
    num_workers: int = 1,
    validation_schema: dict | None = None,
) -> dict:
    """
    Extract structured data from all raw complaints.

    Reads <raw_dir>/<raw_complaints_file> (the complaint text + user_pics),
    resolves the photo files from `images_dir`, and produces
    <output_dir>/<extractions_file>.

    `images_dir` defaults to <parent of raw_dir>/<images_subdir> (i.e. a
    sibling 'images' folder next to 'raw'), matching where the generator
    writes photos. Override with an explicit path.

    The raw complaints are the output of the generator project; copy/link
    them (and their images) into this project's data/ before running.
    """
    raw_path = Path(raw_dir) / raw_complaints_file
    output_path = Path(output_dir) / extractions_file

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.error(f"Raw complaints not found: {raw_path}")
        return {"extracted": 0, "failed": 0, "skipped": 0, "total": 0,
                "total_tokens_in": 0, "total_tokens_out": 0,
                "images_dir": str(images_dir), "output_path": str(output_path)}

    # Resolve images dir: explicit > sibling of raw_dir
    if images_dir is None:
        images_dir = Path(raw_dir).parent / "images"
    images_dir = Path(images_dir)
    if not images_dir.exists():
        logger.warning(f"Images dir not found: {images_dir} (photos will be skipped)")

    # Load complaints
    complaints = {}
    with open(raw_path) as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                complaints[c["id"]] = c

    # Find already extracted IDs for resume
    existing_ids = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    existing_ids.add(e.get("complaint_id", ""))
        logger.info(f"Resuming: {len(existing_ids)} extractions already exist")

    extracted = 0
    failed = 0
    skipped = 0
    write_lock = threading.Lock()

    def _extract_and_write(cid: str, complaint: dict) -> bool:
        extraction = extract_complaint(client, complaint, images_dir,
                                       stage_split=stage_split,
                                       vision_client=vision_client,
                                       validation_schema=validation_schema)
        if extraction is None:
            logger.warning(f"[{cid}] Extraction failed")
            return False
        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(extraction, ensure_ascii=False) + "\n")
        return True

    pending = [(cid, c) for cid, c in complaints.items() if cid not in existing_ids]
    skipped = len(complaints) - len(pending)

    if num_workers > 1:
        logger.info(f"Extracting with {num_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_extract_and_write, cid, c): cid for cid, c in pending}
            for n, future in enumerate(as_completed(futures), 1):
                if future.result():
                    extracted += 1
                else:
                    failed += 1
                if n % 10 == 0:
                    logger.info(
                        f"Progress: {n}/{len(pending)} processed, {failed} failed, "
                        f"tokens: in={client.stats['tokens_in']:,}, "
                        f"out={client.stats['tokens_out']:,}"
                    )
    else:
        for cid, complaint in pending:
            logger.info(f"[{cid}] Extracting ({extracted + 1}/{len(pending)})...")

            if _extract_and_write(cid, complaint):
                extracted += 1
            else:
                failed += 1

            if extracted % 10 == 0:
                logger.info(
                    f"Progress: {extracted} extracted, {failed} failed, "
                    f"{skipped} skipped, "
                    f"tokens: in={client.stats['tokens_in']:,}, "
                    f"out={client.stats['tokens_out']:,}"
                )

    return {
        "extracted": extracted,
        "failed": failed,
        "skipped": skipped,
        "total": len(complaints),
        "total_tokens_in": client.stats["tokens_in"],
        "total_tokens_out": client.stats["tokens_out"],
        "output_path": str(output_path),
        "images_dir": str(images_dir),
    }