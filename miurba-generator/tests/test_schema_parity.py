"""Parity: complaint_schema.json static enums vs config.yaml taxonomy.

The persona/modifier enums in the schema are hand-maintained copies;
this test fails when they drift from the config.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

SCHEMA = json.load(open(REPO_ROOT / "schemas" / "complaint_schema.json"))
CONFIG = yaml.safe_load(open(REPO_ROOT / "taxonomy.yaml"))


def _enum(prop: str) -> list:
    return SCHEMA["properties"][prop]["enum"]


def test_persona_enum_matches_config():
    assert sorted(_enum("persona")) == sorted(CONFIG["personas"].keys())


def test_modifier_enum_matches_config():
    assert sorted(_enum("modifier")) == sorted(CONFIG["modifiers"].keys())
