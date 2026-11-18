"""
Pipeline functions — structured EXTRACTION from raw complaints.

Orchestrates LLM calls and file I/O to turn a raw complaint dataset
(complaints.jsonl) into structured extractions (extractions.jsonl) conforming
to schemas/extraction_schema.json at the repository root.

It does NOT generate synthetic complaints; that is the generator project's job.
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .llm import LLMClient
from .prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Write a JSONL file through a temp file + atomic replace."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
        raise RuntimeError(
            "jsonschema is required for output validation "
            "(pip install jsonschema) — refusing to skip validation"
        )
    try:
        jsonschema.validate(instance=extraction, schema=schema)
        return None
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        return f"{e.message} (at {path})"


def extract_complaint(
    client: LLMClient,
    complaint: dict,
    validation_schema: dict | None = None,
) -> dict | None:
    """
    Extract structured data from a single complaint (text-only).

    The LLM sees user_text and produces {extractions} conforming to
    schemas/extraction_schema.json.
    """
    messages = build_extraction_prompt(complaint)

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
        logger.error(f"[{complaint['id']}] LLM call failed: {e}")
        return None

    # A JSON document that is not an object (list, string, null) is untrustworthy:
    # fail the record instead of crashing on it later.
    if not isinstance(extraction, dict):
        logger.warning(
            f"[{complaint['id']}] model returned a JSON {type(extraction).__name__}, "
            f"not an object — failing record")
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

    # extractions MUST be a list: anything else means the model output is
    # untrustworthy — fail the record (caller counts it as failed) instead
    # of writing schema-violating garbage to extractions.jsonl.
    if not isinstance(extraction.get("extractions"), list):
        logger.warning(f"[{complaint['id']}] 'extractions' is not a list — failing record")
        return None

    return extraction


def extract_dataset(
    client: LLMClient,
    raw_dir: str = "data/raw",
    output_dir: str = "data/annotated",
    raw_complaints_file: str = "complaints.jsonl",
    extractions_file: str = "extractions.jsonl",
    resume: bool = True,
    num_workers: int = 1,
    validation_schema: dict | None = None,
) -> dict:
    """
    Extract structured data from all raw complaints.

    Reads <raw_dir>/<raw_complaints_file> (the complaint text) and produces
    <output_dir>/<extractions_file>. The raw complaints are the output of
    the generator project.
    """
    raw_path = Path(raw_dir) / raw_complaints_file
    output_path = Path(output_dir) / extractions_file

    if not raw_path.exists():
        logger.error(f"Raw complaints not found: {raw_path}")
        return {"extracted": 0, "failed": 0, "skipped": 0, "total": 0,
                "total_tokens_in": 0, "total_tokens_out": 0,
                "output_path": str(output_path), "input_missing": True}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load complaints. Duplicate ids would silently collapse one of the two
    # records, so they are reported before proceeding.
    complaints: dict[str, dict] = {}
    duplicates = 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"{raw_path.name}: unreadable line, skipped")
                continue
            if c["id"] in complaints:
                duplicates += 1
                continue
            complaints[c["id"]] = c
    if duplicates:
        logger.warning(f"{duplicates} duplicate ids in {raw_path.name} (first kept)")

    # Find already extracted IDs for resume; extractions whose complaint no
    # longer exists are dropped so text and extraction cannot drift apart.
    # Known limitation: the match is by id only, so regenerating the corpus with
    # the same ids but different text leaves the old extraction in place. A
    # source-text fingerprint would be needed to detect that.
    existing_ids: set[str] = set()
    if resume and output_path.exists():
        if not complaints:
            # Nothing to compare against: pruning here would erase a valid
            # extraction file when the input corpus is empty or misconfigured.
            logger.warning("Input corpus is empty: keeping existing extractions as they are")
        else:
            kept: list[dict] = []
            stale = 0
            with open(output_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"{output_path.name}: unreadable line, skipped")
                        continue
                    if not isinstance(e, dict):
                        logger.warning(f"{output_path.name}: line is not a JSON object, skipped")
                        continue
                    if str(e.get("complaint_id", "")) not in complaints:
                        stale += 1
                        continue
                    kept.append(e)
                    existing_ids.add(str(e.get("complaint_id", "")))
            if stale:
                _write_jsonl_atomic(output_path, kept)
                logger.warning(f"Dropped {stale} stale extractions (complaint no longer present)")
        logger.info(f"Resuming: {len(existing_ids)} extractions already exist")

    extracted = 0
    failed = 0
    skipped = 0
    write_lock = threading.Lock()
    reject_path = output_path.parent / "rejected.jsonl"

    def _reject(record_id: str, reason: str, detail: str) -> None:
        with write_lock:
            with open(reject_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"complaint_id": record_id, "reason": reason,
                                    "detail": detail[:1000]}, ensure_ascii=False) + "\n")

    def _extract_and_write(cid: str, complaint: dict) -> bool:
        try:
            extraction = extract_complaint(client, complaint,
                                           validation_schema=validation_schema)
        except Exception as e:  # noqa: BLE001 - per-record isolation
            logger.exception(f"[{cid}] unexpected failure: {e}")
            _reject(cid, "unexpected_error", str(e))
            return False
        if extraction is None:
            logger.warning(f"[{cid}] Extraction failed")
            _reject(cid, "extraction_failed", "model output rejected")
            return False
        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(extraction, ensure_ascii=False) + "\n")
        return True

    # resume=False means start fresh: truncate previous output and its audit trail
    if not resume:
        for p_ in (output_path, reject_path):
            if p_.exists():
                p_.write_text("")
        logger.info(f"Start fresh: truncated {output_path}")

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
    }