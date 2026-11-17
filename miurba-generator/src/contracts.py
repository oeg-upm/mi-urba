"""
Output contracts — validation of emitted records against the published schemas.

`schemas/` is the contract of the corpus, so the generator checks its own
output against it before writing: a record that does not conform is rejected
(and audited) instead of silently entering the corpus.

The extraction schema is validated with the taxonomy enums injected, exactly
as the extractor does, so both sides agree on the vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema

from . import config as cfg

_SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"

_cache: dict[str, dict] = {}


def _read_schema(name: str) -> dict:
    if name not in _cache:
        with open(_SCHEMAS_DIR / name, encoding="utf-8") as f:
            _cache[name] = json.load(f)
    return _cache[name]


def complaint_schema() -> dict:
    return _read_schema("complaint_schema.json")


def metadata_schema() -> dict:
    return _read_schema("metadata_schema.json")


def extraction_schema(cfg_dict: dict | None = None) -> dict:
    """Extraction schema with the taxonomy enums injected (same as the extractor)."""
    schema = json.loads(json.dumps(_read_schema("extraction_schema.json")))
    c = cfg_dict if cfg_dict is not None else cfg.get_config()
    zones = cfg.get_location_zones(c)
    schema["$defs"]["risk"]["enum"] = cfg.get_risks(c)
    schema["$defs"]["urgency"]["enum"] = cfg.get_urgency_levels(c)
    schema["$defs"]["location"]["properties"]["zone"]["enum"] = (
        list(zones) if isinstance(zones, dict) else list(zones))
    return schema


def _violation(record: dict, schema: dict) -> str | None:
    try:
        jsonschema.validate(instance=record, schema=schema)
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        return f"{e.message} (at {path})"
    return None


def validate_record(
    complaint: dict,
    ground_truth: dict,
    metadata: dict,
    cfg_dict: dict | None = None,
) -> list[str]:
    """Return the contract violations of one record (empty list means valid)."""
    violations = []
    for label, record, schema in (
        ("complaint", complaint, complaint_schema()),
        ("ground_truth", ground_truth, extraction_schema(cfg_dict)),
        ("metadata", metadata, metadata_schema()),
    ):
        error = _violation(record, schema)
        if error:
            violations.append(f"{label}: {error}")
    return violations


def check_schemas() -> None:
    """Fail fast at startup when a published schema is malformed."""
    for schema in (complaint_schema(), metadata_schema(), extraction_schema()):
        jsonschema.Draft202012Validator.check_schema(schema)


def config_hash(cfg_dict: dict[str, Any]) -> str:
    """Stable SHA-256 of the effective config (provenance fingerprint)."""
    payload = json.dumps(cfg_dict, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
