"""Prompt templates: byte-exact rendering and loader behaviour.

The golden fixture (tests/fixtures/prompt_golden.json) was captured before the
templates were moved out of the code; these tests guarantee that externalising
them did not change a single character of what the model receives.
"""
import json
from pathlib import Path

import pytest

from src import prompt_loader
from src.prompts import (build_extraction_prompt, build_extraction_text_prompt,
                         build_image_description_prompt)

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((Path(__file__).parent / "fixtures/prompt_golden.json")
                    .read_text(encoding="utf-8"))

COMPLAINT = {"id": "Q0001",
             "user_text": "En el portal hay una fuga de agua en la tubería vertical.",
             "user_pics": []}
COMPLAINT_PICS = {"id": "Q0002",
                  "user_text": "Como se ve en la foto, la fachada tiene grietas.",
                  "user_pics": ["Q0002_1.png"]}
DESCRIPTIONS = [{"caption": "Grietas en el revoco de la fachada",
                 "objects": ["fachada", "revoco"], "damage_indicators": ["grietas"],
                 "not_in_text": False},
                {"caption": "Detalle del canalón", "objects": [],
                 "damage_indicators": [], "not_in_text": True}]


@pytest.fixture(autouse=True)
def _pin_repo_config(monkeypatch):
    """Pin the config tables the prompts depend on (other tests mutate the cache)."""
    import yaml
    from src import config as cfg_module
    data = yaml.safe_load((ROOT.parent / "taxonomy.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(cfg_module, "get_config", lambda path=None: data)


def _pixel_png() -> str:
    img = ROOT / ".tmp" / "pixel.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    if not img.exists():
        img.write_bytes(bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082"))
    return str(img)


def test_extraction_text_only_matches_golden():
    assert build_extraction_prompt(COMPLAINT, []) == GOLDEN["extraction_text_only"]


def test_extraction_with_image_matches_golden():
    assert build_extraction_prompt(COMPLAINT_PICS, [_pixel_png()]) == GOLDEN["extraction_with_image"]


def test_describe_image_matches_golden():
    assert build_image_description_prompt(COMPLAINT_PICS, [_pixel_png()]) == GOLDEN["describe_image"]


def test_extract_text_stage2_matches_golden():
    assert build_extraction_text_prompt(COMPLAINT_PICS, DESCRIPTIONS) == GOLDEN["extract_text_stage2"]


def test_extract_text_stage2_without_photos_matches_golden():
    assert build_extraction_text_prompt(COMPLAINT, []) == GOLDEN["extract_text_stage2_no_photos"]


def test_missing_image_is_announced_not_silent():
    msgs = build_image_description_prompt(COMPLAINT_PICS, ["/no/existe.png"])
    text = msgs[1]["content"][1]["text"]
    assert "no disponible" in text


def test_loader_leaves_unknown_placeholders_visible():
    assert "{{user_text}}" in prompt_loader.render("describe_image", "user")


def test_loader_raises_on_unknown_section():
    with pytest.raises(prompt_loader.PromptError, match="no section"):
        prompt_loader.render("extract_complaint", "nope")


def test_fingerprint_is_stable_sha256():
    fp = prompt_loader.fingerprint("extract_complaint")
    assert len(fp) == 64 and fp == prompt_loader.fingerprint("extract_complaint")
    assert set(prompt_loader.fingerprint_all()) == {
        "extract_complaint", "describe_image", "extract_text"}
