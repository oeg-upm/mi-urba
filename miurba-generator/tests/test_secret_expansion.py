"""Tests for ${VAR} secret expansion in the config loader.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as sc  # noqa: E402


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    """Temp repo-root with a config referencing ${TEST_SECRET}."""
    (tmp_path / "schemas").mkdir()
    cfg = {"endpoints": {"e": {"base_url": "http://x", "api_key": "${TEST_SECRET}"}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "taxonomy.yaml").write_text(yaml.safe_dump({"categories": {}}))
    monkeypatch.setattr(sc, "DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
    sc._CONFIG = None
    yield tmp_path
    monkeypatch.undo()
    sc._CONFIG = None


def test_expansion_from_env(cfg_dir, monkeypatch):
    monkeypatch.setenv("TEST_SECRET", "sk-123")
    cfg = sc.load_config(cfg_dir / "config.yaml")
    assert cfg["endpoints"]["e"]["api_key"] == "sk-123"


def test_expansion_from_dotenv(cfg_dir, monkeypatch):
    (cfg_dir / ".env").write_text("TEST_SECRET=sk-from-env\n")
    monkeypatch.delenv("TEST_SECRET", raising=False)
    cfg = sc.load_config(cfg_dir / "config.yaml")
    assert cfg["endpoints"]["e"]["api_key"] == "sk-from-env"


def test_missing_var_fails_fast(cfg_dir, monkeypatch):
    monkeypatch.delenv("TEST_SECRET", raising=False)
    with pytest.raises(ValueError, match="TEST_SECRET"):
        sc.load_config(cfg_dir / "config.yaml")


def test_plain_values_untouched(cfg_dir, monkeypatch):
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump({"a": "no vars here", "b": 5}))
    cfg = sc.load_config(cfg_dir / "config.yaml")
    assert {k: v for k, v in cfg.items() if k not in sc.TAXONOMY_KEYS} == {"a": "no vars here", "b": 5}
