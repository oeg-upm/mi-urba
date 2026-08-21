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
from .prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


def _resolve_image_paths(complaint: dict, images_dir: str | Path) -> list[str]:
    """Resolve the photo file paths for a complaint from its user_pics."""
    img_dir = Path(images_dir)
    return [str(img_dir / pic) for pic in complaint.get("user_pics", [])]


def extract_complaint(
    client: LLMClient,
    complaint: dict,
    images_dir: str | Path,
) -> dict | None:
    """
    Extract structured data + image descriptions from a single complaint.

    Receives the user_text and the actual photo files; the VLM sees both and
    produces {image_descriptions, extractions} conforming to
    schemas/extraction_schema.json.
    """
    image_paths = _resolve_image_paths(complaint, images_dir)
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

    # Ensure image_descriptions is present (default to [] if the model omitted it)
    if "image_descriptions" not in extraction:
        extraction["image_descriptions"] = []
    if not isinstance(extraction["image_descriptions"], list):
        extraction["image_descriptions"] = []

    # Ensure extractions is a list
    if "extractions" not in extraction or not isinstance(extraction["extractions"], list):
        extraction.setdefault("extractions", [])

    return extraction


def extract_dataset(
    client: LLMClient,
    raw_dir: str = "data/raw",
    images_dir: str | None = None,
    output_dir: str = "data/annotated",
    raw_complaints_file: str = "complaints.jsonl",
    extractions_file: str = "extractions.jsonl",
    resume: bool = True,
    max_retries: int = 3,
    num_workers: int = 1,
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
        return {"extracted": 0, "failed": 0, "skipped": 0}

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

    total_complaints = len(complaints)
    total_extracted = 0
    total_failed = 0

    for attempt in range(max_retries + 1):
        # Find already extracted IDs for resume
        existing_ids = set()
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        existing_ids.add(e.get("complaint_id", ""))
        skipped = len(existing_ids)
        if skipped > 0:
            logger.info(f"Resuming: {skipped} extractions already exist")

        if attempt > 0:
            remaining = total_complaints - skipped
            logger.info(f"=== Retry {attempt}/{max_retries}: {remaining} pending, {total_failed} failed so far ===")

        extracted = 0
        failed = 0
        write_lock = threading.Lock()

        ids_to_extract = [cid for cid in complaints if cid not in existing_ids]
        logger.info(f"Extracting {len(ids_to_extract)} complaints with {num_workers} workers...")

        def _extract_one(cid: str) -> tuple[str, dict | None]:
            complaint = complaints[cid]
            return cid, extract_complaint(client, complaint, images_dir)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_extract_one, cid): cid for cid in ids_to_extract}
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    rid, extraction = future.result()
                except Exception as e:
                    logger.error(f"[{cid}] Worker error: {e}")
                    failed += 1
                    total_failed += 1
                    continue

                if extraction is None:
                    failed += 1
                    total_failed += 1
                    logger.warning(f"[{cid}] Extraction failed")
                    continue

                with write_lock:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(extraction, ensure_ascii=False) + "\n")
                    extracted += 1
                    total_extracted += 1

                    if (extracted + skipped) % 10 == 0:
                        logger.info(
                            f"Progress: {extracted + skipped}/{total_complaints} done, "
                            f"{failed} failed this pass, "
                            f"tokens: in={client.stats['tokens_in']:,}, "
                            f"out={client.stats['tokens_out']:,}"
                        )

        if failed == 0:
            break
        if attempt < max_retries:
            logger.info(f"Pass {attempt} complete: {extracted} extracted, {failed} failed. Retrying in 30s...")
            import time
            time.sleep(30)

    final_existing = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    final_existing.add(json.loads(line).get("complaint_id", ""))

    return {
        "extracted": len(final_existing),
        "failed": total_complaints - len(final_existing),
        "skipped": 0,
        "total": total_complaints,
        "total_tokens_in": client.stats["tokens_in"],
        "total_tokens_out": client.stats["tokens_out"],
        "output_path": str(output_path),
        "images_dir": str(images_dir),
    }