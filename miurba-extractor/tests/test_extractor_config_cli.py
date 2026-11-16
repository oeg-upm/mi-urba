"""Tests for the config-only extractor CLI: extraction config, dir resolution
and the VLM model fallback chain.

Run from miurba-extractor/:  python -m pytest tests/ -v
No network access is required.
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cli import resolve_dirs  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    get_config,
    get_extraction_config,
    get_extraction_model_params,
)
from src.llm import LLMClient  # noqa: E402


@pytest.fixture
def config_file(tmp_path):
    """Minimal valid config.yaml with a required extraction.model."""
    cfg = {
        "endpoints": {
            "test": {"base_url": "http://localhost:9999", "api_key": "sk-test"},
        },
        "models": {
            "vlm-a": {"endpoint": "test", "model": "vlm-a-model", "temperature": 0.1},
            "vlm-b": {"endpoint": "test", "model": "vlm-b-model", "temperature": 0.2},
        },
        "extraction": {"model": "vlm-a", "temperature": 0.2},
        "generation": {"default_output_dir": str(tmp_path / "out")},
        "paths": {},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (tmp_path / "taxonomy.yaml").write_text(yaml.safe_dump(
        {"categories": {"cubierta": ["tejas"]}, "risks": ["r"],
         "urgency_levels": ["alta"], "location_zones": {"portal": []}}),
        encoding="utf-8")
    return p


# ── Extraction config defaults ────────────────────────────────

def test_extraction_defaults_fill_missing_keys(config_file):
    ext = get_extraction_config(get_config(config_file))
    for key, default in [
        # ("model", None) excluded: the fixture sets it (it is REQUIRED anyway)
        ("temperature", 0.2), ("default_output_dir", None),
        ("raw_dir", None), ("annotated_dir", None),
        ("resume", True), ("timeout", 300), ("max_retries", 3),
        ("retry_delay", 5.0),
    ]:
        assert ext[key] == default, f"{key} should default to {default}"


def test_repo_config_extraction_section_fully_consumed():
    """Every key in the repo's config.yaml extraction: section must be known."""
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    from src.config import _DEFAULT_EXTRACTION
    unknown = set(cfg.get("extraction", {})) - set(_DEFAULT_EXTRACTION)
    assert not unknown, f"config.yaml has extraction keys the code ignores: {unknown}"


# ── VLM model resolution (required, no fallback) ──────────────

def test_vlm_resolves_extraction_model(config_file):
    """extraction.model resolves to its models: entry."""
    cfg = get_config(config_file)
    name, params = get_extraction_model_params(None, cfg)
    assert name == "vlm-a" and params["model"] == "vlm-a-model"


def test_vlm_explicit_name_wins(config_file):
    """An explicitly passed name overrides extraction.model."""
    cfg = get_config(config_file)
    name, params = get_extraction_model_params("vlm-b", cfg)
    assert name == "vlm-b" and params["model"] == "vlm-b-model"


def test_vlm_missing_model_fails_fast(config_file):
    """extraction.model missing/null → ValueError (required)."""
    cfg = get_config(config_file)
    cfg["extraction"]["model"] = None
    with pytest.raises(ValueError, match="extraction.model is required"):
        get_extraction_model_params(None, cfg)


# ── Directory resolution ──────────────────────────────────────

PATHS = {
    "raw_subdir": "raw",
    "annotated_subdir": "annotated",
}


def test_dirs_from_generation_default_output_dir(config_file):
    ext = get_extraction_config(get_config(config_file))
    raw, ann = resolve_dirs(ext, PATHS, fallback_base="/base")
    assert raw == "/base/raw"
    assert ann == "/base/annotated"


def test_dirs_explicit_override(config_file):
    cfg = get_config(config_file)
    cfg["extraction"].update({
        "raw_dir": "/x/raw", "annotated_dir": "/x/ann",
    })
    raw, ann = resolve_dirs(get_extraction_config(cfg), PATHS, fallback_base="/base")
    assert (raw, ann) == ("/x/raw", "/x/ann")


def test_dirs_extraction_base_wins_over_generation(config_file):
    cfg = get_config(config_file)
    cfg["extraction"]["default_output_dir"] = "/ext-base"
    raw, _ = resolve_dirs(get_extraction_config(cfg), PATHS, fallback_base="/gen-base")
    assert raw == "/ext-base/raw"


# ── CLI wiring ────────────────────────────────────────────────

def test_llm_client_receives_extraction_retry_policy(config_file):
    cfg = get_config(config_file)
    ext = get_extraction_config(cfg)
    ext.update({"timeout": 120, "max_retries": 5, "retry_delay": 2.0})
    client = LLMClient(
        base_url="http://localhost:9999",
        model="vlm-a-model",
        temperature=ext["temperature"],
        timeout=ext["timeout"],
        max_retries=ext["max_retries"],
        retry_delay=ext["retry_delay"],
    )
    assert (client.timeout, client.max_retries, client.retry_delay) == (120, 5, 2.0)
