"""
Configuration loader — reads the SHARED config.yaml at the repo root
and provides typed access.

The extractor shares a single config.yaml with the generator, located at
the repository root (one level above this project folder). All
configuration lives there; this module loads it once and exposes the
parsed data. No hardcoded values elsewhere.

Default path resolution (miurba-extractor/src/config.py):
    __file__          -> .../miurba-extractor/src/config.py
    parent            -> .../miurba-extractor/src
    parent.parent     -> .../miurba-extractor
    parent.parent.parent -> .../mi-urba  (repo root, where config.yaml lives)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG: dict | None = None

# Shared config lives at the repo root, two levels up from this file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration."""
    global _CONFIG
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _CONFIG = cfg
    return cfg


def get_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return cached config, loading from path if needed."""
    global _CONFIG
    if _CONFIG is None or path is not None:
        _CONFIG = load_config(path)
    return _CONFIG


# ── Endpoint & model helpers ──────────────────────────────────

def get_endpoint_url(name: str, cfg: dict | None = None) -> str:
    """Return the base_url for a named endpoint (without /v1)."""
    cfg = cfg or get_config()
    return cfg["endpoints"][name]["base_url"].rstrip("/")


def get_model_params(name: str, cfg: dict | None = None) -> dict[str, Any]:
    """Return (base_url, model_id, temperature, api_key) for a named model."""
    cfg = cfg or get_config()
    m = cfg["models"][name]
    endpoint_name = m["endpoint"]
    endpoint = cfg["endpoints"][endpoint_name]
    return {
        "base_url": get_endpoint_url(endpoint_name, cfg),
        "model": m["model"],
        "temperature": m.get("temperature", 0.85),
        "api_key": endpoint.get("api_key"),
        "extra_params": m.get("extra_params", {}),
    }


# ── Taxonomy access ────────────────────────────────────────────

def get_categories(cfg: dict | None = None) -> dict[str, list[str]]:
    cfg = cfg or get_config()
    return cfg["categories"]


def get_affected_elements(cfg: dict | None = None) -> dict[str, list[str]]:
    cfg = cfg or get_config()
    return cfg["affected_elements"]


def get_location_zones(cfg: dict | None = None) -> dict[str, list[str]]:
    cfg = cfg or get_config()
    return cfg["location_zones"]


def get_risks(cfg: dict | None = None) -> list[str]:
    cfg = cfg or get_config()
    return cfg["risks"]


def get_urgency_levels(cfg: dict | None = None) -> list[str]:
    cfg = cfg or get_config()
    return cfg["urgency_levels"]


def get_channels(cfg: dict | None = None) -> dict[str, dict]:
    cfg = cfg or get_config()
    return cfg["channels"]


def get_communities(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or get_config()
    return cfg["communities"]


def get_weights(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    return cfg["weights"]


def get_paths(cfg: dict | None = None) -> dict[str, str]:
    """Return path configuration with sensible defaults."""
    cfg = cfg or get_config()
    return cfg.get("paths", {
        "raw_subdir": "raw",
        "annotated_subdir": "annotated",
        "images_subdir": "images",
        "complaints_file": "complaints.jsonl",
        "ground_truth_file": "ground_truth.jsonl",
        "metadata_file": "metadata.jsonl",
        "extractions_file": "extractions.jsonl",
    })


def get_extraction_json_schema(cfg: dict | None = None) -> dict[str, Any]:
    """Load schemas/extraction_schema.json with enums synced from config.

    Reads the JSON schema file at the repo root and injects the enum values
    from config.yaml (risks, urgency_levels, location_zones) so that the
    schema and the taxonomy never diverge. Also adds maxItems to arrays
    to prevent generation loops in small models.
    """
    import json as _json
    schema_path = DEFAULT_CONFIG_PATH.parent / "schemas" / "extraction_schema.json"
    with open(schema_path, encoding="utf-8") as f:
        schema = _json.load(f)

    cfg = cfg or get_config()

    # Sync enums from config -> schema
    risks = get_risks(cfg)
    urgency = get_urgency_levels(cfg)
    zones_cfg = get_location_zones(cfg)
    zones = list(zones_cfg.keys()) if isinstance(zones_cfg, dict) else zones_cfg

    schema["$defs"]["risk"]["enum"] = risks
    schema["$defs"]["urgency"]["enum"] = urgency
    schema["$defs"]["location"]["properties"]["zone"]["enum"] = zones

    # Add maxItems to prevent generation loops in small models
    schema["$defs"]["problemExtraction"]["properties"]["affected_elements"]["maxItems"] = 5
    schema["$defs"]["problemExtraction"]["properties"]["risks"]["maxItems"] = 5

    return schema


def get_image_description_json_schema(cfg: dict | None = None) -> dict[str, Any]:
    """Load schemas/image_description_schema.json (for P2 stage split, stage 1)."""
    import json as _json
    schema_path = DEFAULT_CONFIG_PATH.parent / "schemas" / "image_description_schema.json"
    with open(schema_path, encoding="utf-8") as f:
        schema = _json.load(f)
    return schema


def get_vision_model_params(name: str | None = None, cfg: dict | None = None) -> dict[str, Any]:
    """Return model params for the VLM used to read images.

    Resolution order:
      1. explicit `name` argument
      2. annotation.vision_model in config
      3. annotation.model in config

    The chosen model MUST be vision-capable (multimodal) so it can see the
    attached photos.
    """
    cfg = cfg or get_config()
    ann = cfg.get("annotation", {})
    model_name = name or ann.get("vision_model") or ann.get("model", "kimi-k2.6")
    return get_model_params(model_name, cfg)