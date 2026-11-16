"""Tests for the config/taxonomy split.

config.yaml holds run parameters only; domain vocabulary lives in
taxonomy.yaml and is merged at load time. Declaring taxonomy keys in a
config must fail fast.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

import src.config as sc  # noqa: E402


def test_repo_config_loads_with_taxonomy():
    cfg = sc.load_config(REPO_ROOT / "config.yaml")
    for key in sc.TAXONOMY_KEYS:
        assert key in cfg, f"taxonomy key {key!r} missing after load"
    assert "taxonomy" not in cfg  # the reference key is consumed, not kept


def test_stale_taxonomy_keys_fail_fast(tmp_path, monkeypatch):
    cfg = {"endpoints": {"e": {"base_url": "http://x"}},
           "categories": {"fontanería": ["fugas"]}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    (tmp_path / "taxonomy.yaml").write_text(yaml.safe_dump({"categories": {}}))
    monkeypatch.setattr(sc, "DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
    sc._CONFIG = None
    try:
        with pytest.raises(ValueError, match="categories.*taxonomy.yaml"):
            sc.load_config(tmp_path / "config.yaml")
    finally:
        monkeypatch.undo()
        sc._CONFIG = None


def test_custom_taxonomy_path(tmp_path, monkeypatch):
    cfg = {"endpoints": {"e": {"base_url": "http://x"}},
           "taxonomy": "mytax.yaml"}
    tax = {"categories": {"x": ["y"]}, "risks": ["r"], "urgency_levels": ["alta"],
           "location_zones": {"portal": []}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "mytax.yaml").write_text(yaml.safe_dump(tax))
    monkeypatch.setattr(sc, "DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
    sc._CONFIG = None
    try:
        loaded = sc.load_config(tmp_path / "config.yaml")
        assert loaded["categories"] == {"x": ["y"]}
        assert loaded["endpoints"]["e"]["base_url"] == "http://x"
    finally:
        monkeypatch.undo()
        sc._CONFIG = None


def test_missing_taxonomy_file_fails_fast(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"endpoints": {}}))
    monkeypatch.setattr(sc, "DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
    sc._CONFIG = None
    try:
        with pytest.raises(FileNotFoundError, match="taxonomy"):
            sc.load_config(tmp_path / "config.yaml")
    finally:
        monkeypatch.undo()
        sc._CONFIG = None


def test_element_subcategories_come_from_taxonomy():
    """The coherence table now lives inside taxonomy.yaml."""
    cfg = sc.load_config(REPO_ROOT / "config.yaml")
    table = cfg["element_subcategories"]
    assert "fontanería" in table
    # and the accessor reads the merged config
    assert sc.get_element_subcategories() == table
