"""
Configuration loader — reads config.yaml and provides typed access.

All configuration lives in config.yaml. This module loads it once
and exposes the parsed data. No hardcoded values elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG: dict | None = None

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


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
    """Return (base_url, model_id, temperature) for a named model."""
    cfg = cfg or get_config()
    m = cfg["models"][name]
    return {
        "base_url": get_endpoint_url(m["endpoint"], cfg),
        "model": m["model"],
        "temperature": m.get("temperature", 0.85),
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