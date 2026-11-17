"""Parity between the published schemas and schemas/taxonomy.yaml.

The persona/modifier enums in metadata_schema.json and the risk/urgency/zone
enums in extraction_schema.json are hand-maintained copies of the taxonomy;
these tests fail when they drift.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

METADATA = json.load(open(REPO_ROOT / "schemas" / "metadata_schema.json"))
EXTRACTION = json.load(open(REPO_ROOT / "schemas" / "extraction_schema.json"))
TAXONOMY = yaml.safe_load(open(REPO_ROOT / "schemas" / "taxonomy.yaml"))


def _non_null_enum(schema: dict, prop: str) -> set:
    return {v for v in schema["properties"][prop]["enum"] if v is not None}


def test_persona_enum_matches_taxonomy():
    assert _non_null_enum(METADATA, "persona") == set(TAXONOMY["personas"])


def test_modifier_enum_matches_taxonomy():
    assert _non_null_enum(METADATA, "modifier") == set(TAXONOMY["modifiers"])


def test_risk_enum_matches_taxonomy():
    assert set(EXTRACTION["$defs"]["risk"]["enum"]) == set(TAXONOMY["risks"])


def test_urgency_enum_matches_taxonomy():
    assert set(EXTRACTION["$defs"]["urgency"]["enum"]) == set(TAXONOMY["urgency_levels"])


def test_zone_enum_matches_taxonomy():
    zones = EXTRACTION["$defs"]["location"]["properties"]["zone"]["enum"]
    assert set(zones) == set(TAXONOMY["location_zones"])
