"""
Configuration loader — reads the subproject's config/ directory
and provides typed access.

Default path resolution (miurba-generator/src/config.py):
    __file__          -> .../miurba-generator/src/config.py
    parent            -> .../miurba-generator/src
    parent.parent     -> .../miurba-generator
    parent.parent / "config" / "config.yaml"  (default run config)

The repo root holds only shared contracts: schemas/ (JSON Schemas and
taxonomy.yaml). Each subproject owns its config/ directory with the default
config and examples.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_CONFIG: dict | None = None

# Subproject root (miurba-generator/) and repo root (mi-urba/).
_SUBPROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SUBPROJECT_ROOT.parent

# Default run config: <subproject>/config/config.yaml
DEFAULT_CONFIG_PATH = _SUBPROJECT_ROOT / "config" / "config.yaml"

# Shared domain contracts live at the repo root: schemas/taxonomy.yaml
DEFAULT_TAXONOMY_PATH = _REPO_ROOT / "schemas" / "taxonomy.yaml"


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Domain vocabulary sections: they live in taxonomy.yaml, NOT in config.yaml.
# Keeping them out of run configs is what prevents stale duplicated taxonomies
# (the demo config once drifted ~950 lines this way).
TAXONOMY_KEYS = (
    "categories", "affected_elements", "location_zones", "risks",
    "risk_expressions", "urgency_expressions", "urgency_levels",
    "communities", "personas", "weights", "modifiers", "modifier_prob",
    "category_elements", "category_zones", "category_risks",
    "urgency_weights", "element_subcategories",
)


def _load_dotenv(config_path: Path | None = None) -> None:
    """Populate os.environ from .env files (existing vars win).

    Tried in order: the subproject root, next to the loaded config, and the
    config's parent directory.
    """
    candidates = [_SUBPROJECT_ROOT / ".env"]
    if config_path is not None:
        candidates.append(config_path.parent / ".env")
        candidates.append(config_path.parent.parent / ".env")
    seen: set[Path] = set()
    for env_path in candidates:
        if env_path in seen or not env_path.exists():
            continue
        seen.add(env_path)
        _apply_dotenv(env_path)


def _apply_dotenv(env_path: Path) -> None:
    """Populate os.environ from one .env file (existing vars win)."""
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
    ${VAR_NAME} and define them in the environment or the subproject's .env.
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
                    f"in the environment or the subproject's .env"
                )
            return os.environ[name]
        return _VAR_RE.sub(repl, node)
    return node


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration, expanding ${VAR} references.

    Domain vocabulary is loaded from taxonomy.yaml (key `taxonomy:` in the
    config, default `<repo root>/schemas/taxonomy.yaml`) and merged into the
    result, so every taxonomy accessor keeps working unchanged. Declaring any
    taxonomy key directly in the config fails fast.
    """
    global _CONFIG
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    _load_dotenv(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    _load_taxonomy(cfg, path)
    _CONFIG = _expand_vars(cfg)
    return _CONFIG


def _resolve_taxonomy_ref(ref: str | None, child: Path) -> Path:
    """Resolve the `taxonomy:` reference (absolute, relative to the config,
    or the repo-root default)."""
    default = DEFAULT_TAXONOMY_PATH
    if ref is None:
        return default
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate
    resolved = (child.parent / candidate).resolve()
    if resolved.exists():
        return resolved
    import warnings
    warnings.warn(
        f"taxonomy: {ref!r} not found at {resolved}; falling back to the "
        f"default {default}", stacklevel=3)
    return default


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
    tax_path = _resolve_taxonomy_ref(ref, child)
    if not tax_path.exists():
        raise FileNotFoundError(f"taxonomy file not found: {tax_path}")
    with open(tax_path, encoding="utf-8") as f:
        tax = yaml.safe_load(f) or {}
    cfg.update(tax)


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


def get_element_subcategories() -> dict:
    """Coherence table: category -> problem subcategory -> element -> valid subcategories.

    Lives in taxonomy.yaml (`element_subcategories:`) because it is domain
    knowledge, not run configuration. Legacy standalone element_subcategories.yaml
    next to the config is still honored with a deprecation warning.
    """
    cfg = get_config()
    if "element_subcategories" in cfg:
        return cfg["element_subcategories"]
    legacy = _REPO_ROOT / "element_subcategories.yaml"
    if legacy.exists():
        import warnings
        warnings.warn(
            f"{legacy.name} is deprecated: merge it into taxonomy.yaml "
            f"(element_subcategories: key)", DeprecationWarning, stacklevel=2)
        with open(legacy, encoding="utf-8") as f:
            return yaml.safe_load(f)["element_subcategories"]
    raise FileNotFoundError("element_subcategories not found: neither in the "
                            "loaded config (taxonomy.yaml) nor as a legacy file")


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
    # max_validation_retries times on mismatch and the record is discarded
    # (audited in rejected.jsonl) when the budget is exhausted.
    "validate": False,
    "max_validation_retries": 3,
    # Judge model (key in models:) for validation. REQUIRED when validate is
    # true: startup fails fast if it is missing, so the generator is never
    # used as its own judge. Recommended: a model from another family.
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
