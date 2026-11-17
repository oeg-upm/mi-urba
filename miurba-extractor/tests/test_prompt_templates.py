"""Prompt templates: byte-exact rendering and loader behaviour.

The golden fixture (tests/fixtures/prompt_golden.json) was captured before the
templates were moved out of the code; these tests guarantee that externalising
them did not change a single character of what the model receives.
"""
import json
from pathlib import Path

import pytest

from src import prompt_loader
from src.prompts import build_extraction_prompt

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((Path(__file__).parent / "fixtures/prompt_golden.json")
                    .read_text(encoding="utf-8"))

COMPLAINT = {"id": "Q0001",
             "user_text": "En el portal hay una fuga de agua en la tubería vertical."}


@pytest.fixture(autouse=True)
def _pin_repo_config(monkeypatch):
    """Pin the config tables the prompts depend on (other tests mutate the cache)."""
    import yaml
    from src import config as cfg_module
    data = yaml.safe_load((ROOT.parent / "schemas" / "taxonomy.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(cfg_module, "get_config", lambda path=None: data)


def test_extraction_text_only_matches_golden():
    assert build_extraction_prompt(COMPLAINT) == GOLDEN["extraction_text_only"]


def test_loader_leaves_unknown_placeholders_visible():
    assert "{{input_text}}" in prompt_loader.render("extract_complaint", "user")


def test_loader_raises_on_unknown_section():
    with pytest.raises(prompt_loader.PromptError, match="no section"):
        prompt_loader.render("extract_complaint", "nope")


def test_fingerprint_is_stable_sha256():
    fp = prompt_loader.fingerprint("extract_complaint")
    assert len(fp) == 64 and fp == prompt_loader.fingerprint("extract_complaint")
    assert set(prompt_loader.fingerprint_all()) == {"extract_complaint"}
