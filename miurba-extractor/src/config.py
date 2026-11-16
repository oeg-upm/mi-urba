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

import os
import re
from pathlib import Path
from typing import Any

import yaml

_CONFIG: dict | None = None

# Shared config lives at the repo root, two levels up from this file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Domain vocabulary sections: they live in taxonomy.yaml, NOT in config.yaml
# (mirrors miurba-generator's loader).
TAXONOMY_KEYS = (
    "categories", "affected_elements", "location_zones", "risks",
    "risk_expressions", "urgency_expressions", "urgency_levels",
    "communities", "personas", "weights", "modifiers", "modifier_prob",
    "category_elements", "category_zones", "category_risks",
    "urgency_weights", "element_subcategories",
)


def _load_dotenv() -> None:
    """Populate os.environ from the repo-root .env (existing vars win)."""
    env_path = DEFAULT_CONFIG_PATH.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _expand_vars(node: Any) -> Any:
    """Recursively expand ${VAR} references from the environment.

    Secrets (API keys) never live in config.yaml: reference them as
    ${VAR_NAME} and define them in the environment or the repo-root .env.
    Missing variables fail fast with a clear error.
    """
    if isinstance(node, dict):
        return {k: _expand_vars(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_vars(v) for v in node]
    if isinstance(node, str):
        def repl(m: "re.Match[str]") -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ValueError(
                    f"config references ${{{name}}} but it is not defined "
                    f"in the environment or the repo-root .env"
                )
            return os.environ[name]
        return _VAR_RE.sub(repl, node)
    return node


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` on top of `base` (override wins)."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_taxonomy(cfg: dict, child: Path) -> None:
    """Merge taxonomy.yaml into cfg; fail fast on stale inline taxonomy keys."""
    stale = [k for k in TAXONOMY_KEYS if k in cfg]
    if stale:
        raise ValueError(
            f"config {child.name} declares taxonomy keys ({', '.join(stale)}); "
            f"they live in taxonomy.yaml — move them there and reference it with "
            f"`taxonomy: <path>`"
        )
    ref = cfg.pop("taxonomy", None)
    default = DEFAULT_CONFIG_PATH.parent / "taxonomy.yaml"
    tax_path = Path(ref) if ref and Path(ref).is_absolute() else (
        (child.parent / ref).resolve() if ref and (child.parent / ref).resolve().exists() else default
    )
    if not tax_path.exists():
        raise FileNotFoundError(f"taxonomy file not found: {tax_path}")
    with open(tax_path, encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f) or {})


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration, expanding ${VAR} references.

    Domain vocabulary is loaded from taxonomy.yaml (key `taxonomy:` in the
    config, default `<repo root>/taxonomy.yaml`) and merged into the result;
    declaring taxonomy keys directly in the config fails fast.
    """
    global _CONFIG
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    _load_dotenv()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    _load_taxonomy(cfg, path)
    _CONFIG = _expand_vars(cfg)
    return _CONFIG


def get_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return cached config, loading from path if needed."""
    global _CONFIG
    if _CONFIG is None or path is not None:
        _CONFIG = load_config(path)
    return _CONFIG


# ── Logging section ────────────────────────────────────────────

def get_logging_config(cfg: dict | None = None) -> dict[str, Any]:
    """Return the shared `logging:` section (level + optional run file)."""
    cfg = cfg or get_config()
    out = {"level": "INFO", "file": None}
    out.update(cfg.get("logging", {}))
    return out


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


def get_vision_model_params(name: str | None = None, cfg: dict | None = None) -> tuple[str, dict[str, Any]]:
    """Return (resolved_name, model_params) for the VLM used to read images.

    Resolution order:
      1. explicit `name` argument
      2. extraction.model in config (REQUIRED — no fallback)

    The chosen model MUST be vision-capable (multimodal) so it can see the
    attached photos. Fail fast when no model is configured.
    """
    cfg = cfg or get_config()
    ext = cfg.get("extraction", {})
    model_name = name or ext.get("model")
    if not model_name:
        raise ValueError(
            "extraction.model is required (a key of models:) — set it in config.yaml"
        )
    return model_name, get_model_params(model_name, cfg)


# ── Extraction section ─────────────────────────────────────────

_DEFAULT_EXTRACTION = {
    # Model name (key in models:) used for extraction. REQUIRED: the CLI
    # fails fast when missing. No fallback chain.
    "model": None,
    # Sampling temperature for extraction calls.
    "temperature": 0.2,
    # Explicit directories. When None, each resolves to
    # <default_output_dir>/<raw|annotated|images subdir> from `paths:`.
    "default_output_dir": None,
    "raw_dir": None,
    "annotated_dir": None,
    "images_dir": None,
    # Skip complaints already present in the extractions file.
    "resume": True,
    # Per-request timeout and retry policy for the LLM client.
    "timeout": 300,
    "max_retries": 3,
    "retry_delay": 5.0,
    # P2 stage-split: two-call pipeline (vision describes, text extracts).
    # Requires photos in the complaint; falls back to single-call otherwise.
    "stage_split": False,
    # Vision model (key in models:) for stage 1 when stage_split is true.
    # null → stage-split falls back to single-call with a warning.
    "vision_model": None,
    # Patience (seconds) with cold models: 5xx within this window are treated
    # as "model loading" and do not consume retries. null = disabled.
    "model_load_wait": None,
    # Concurrent extraction workers (1 = sequential, current behaviour).
    "workers": 1,
}


def get_extraction_config(cfg: dict | None = None) -> dict[str, Any]:
    """Return the `extraction:` section with defaults for missing keys."""
    cfg = cfg or get_config()
    ext = dict(_DEFAULT_EXTRACTION)
    ext.update(cfg.get("extraction", {}))
    return ext


# ── JSON schemas for constrained decoding ───────────────────────

def _load_schema(name: str) -> dict[str, Any]:
    import json as _json
    schema_path = DEFAULT_CONFIG_PATH.parent / "schemas" / name
    with open(schema_path, encoding="utf-8") as f:
        return _json.load(f)


def get_extraction_json_schema(cfg: dict | None = None) -> dict[str, Any]:
    """Load schemas/extraction_schema.json with enums synced from config.

    Injects the taxonomy values from config.yaml (risks, urgency_levels,
    location_zones) into the schema so the two can never diverge. Adds
    maxItems to free-form arrays to prevent generation loops in small
    models.
    """
    schema = _load_schema("extraction_schema.json")
    cfg = cfg or get_config()

    risks = get_risks(cfg)
    urgency = get_urgency_levels(cfg)
    zones_cfg = get_location_zones(cfg)
    zones = list(zones_cfg.keys()) if isinstance(zones_cfg, dict) else zones_cfg

    schema["$defs"]["risk"]["enum"] = risks
    schema["$defs"]["urgency"]["enum"] = urgency
    schema["$defs"]["location"]["properties"]["zone"]["enum"] = zones

    problem = schema["$defs"]["problemExtraction"]["properties"]
    problem["affected_elements"]["maxItems"] = 5
    problem["risks"]["maxItems"] = 5
    return schema


def get_image_description_json_schema(cfg: dict | None = None) -> dict[str, Any]:
    """Load schemas/image_description_schema.json (stage-1 vision output)."""
    return _load_schema("image_description_schema.json")