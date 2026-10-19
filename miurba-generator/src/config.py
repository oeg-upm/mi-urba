"""
Configuration loader — reads the SHARED config.yaml at the repo root
and provides typed access.

The generator and the extractor share a single config.yaml located at
the repository root (one level above this project folder). All
configuration lives there; this module loads it once and exposes the
parsed data. No hardcoded values elsewhere.

Default path resolution (miurba-generator/src/config.py):
    __file__          -> .../miurba-generator/src/config.py
    parent            -> .../miurba-generator/src
    parent.parent     -> .../miurba-generator
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
        "connection_backoff": endpoint.get("connection_backoff", False),
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


def get_risk_expressions(cfg: dict | None = None) -> dict[str, str]:
    """Natural-language expression per risk (drives the generation prompt)."""
    cfg = cfg or get_config()
    return cfg.get("risk_expressions", {})


def get_urgency_expressions(cfg: dict | None = None) -> dict[str, list[str]]:
    """Expressions the generated text must include, per urgency level."""
    cfg = cfg or get_config()
    return cfg.get("urgency_expressions", {})


def get_communities(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or get_config()
    return cfg["communities"]


def get_weights(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    return cfg["weights"]


_DEFAULT_PATHS = {
    "raw_subdir": "raw",
    "annotated_subdir": "annotated",
    "images_subdir": "images",
    "complaints_file": "complaints.jsonl",
    "ground_truth_file": "ground_truth.jsonl",
    "metadata_file": "metadata.jsonl",
    "extractions_file": "extractions.jsonl",
}


def get_paths(cfg: dict | None = None) -> dict[str, str]:
    """Return path configuration: defaults merged with the paths: section."""
    cfg = cfg or get_config()
    return {**_DEFAULT_PATHS, **cfg.get("paths", {})}


# ── Generation section ─────────────────────────────────────────

_DEFAULT_GENERATION = {
    # Where per-model outputs are written: <default_output_dir>/<model>/
    "default_output_dir": "data",
    # Batch plan: [{model: <name in models:>, count: <int>}, ...].
    # Ignored when `models` is set explicitly.
    "batches": [],
    # Explicit model list; each model generates `num_records` records.
    "models": None,
    # Records per model when `models` is used (ignored with `batches`).
    "num_records": 100,
    # First ID number (IDs are Q<start_id>.. sequential across the plan).
    "start_id": 1,
    # Base random seed for sampling (None = non-deterministic).
    "seed": None,
    # Skip records already present in the output files.
    "resume": True,
    # Per-request timeout and retry policy for the LLM client.
    "timeout": 300,
    "max_retries": 3,
    "retry_delay": 5.0,
    # 2-stage validation: after generating, a judge LLM checks that the text
    # aligns with the scenario; the generation is retried up to
    # max_validation_retries times on mismatch (last text is kept).
    "validate": False,
    "max_validation_retries": 3,
    # Judge model (key in models:) for validation. None = validate with the
    # generator's own client. Recommended: a DIFFERENT model to avoid
    # shared biases between generator and judge.
    "validator": None,
    # Patience (seconds) with cold models: 5xx within this window are treated
    # as "model loading" and do not consume retries. null = disabled.
    "model_load_wait": None,
    # Concurrent generation workers (1 = sequential, current behaviour).
    "workers": 1,
}


def get_generation_config(cfg: dict | None = None) -> dict[str, Any]:
    """Return the `generation:` section with defaults for missing keys."""
    cfg = cfg or get_config()
    gen = dict(_DEFAULT_GENERATION)
    gen.update(cfg.get("generation", {}))
    return gen


# ── Image generation config (optional) ──────────────────────────

_DEFAULT_IMAGE_GEN = {
    "enabled": False,
    "backend": "flux2",
    "cache": "diffusers/FLUX.2-dev-bnb-4bit",
    "output_subdir": "images",
    "size": 1024,
    "steps": 28,
    "guidance": 4.0,
    "device0": "cuda:0",
    "device1": "cuda:1",
    "seed_base": 42,
}


def get_image_generation_config(cfg: dict | None = None) -> dict[str, Any]:
    """Return the image-generation config (disabled by default)."""
    cfg = cfg or get_config()
    ig = dict(_DEFAULT_IMAGE_GEN)
    ig.update(cfg.get("image_generation", {}))
    return ig