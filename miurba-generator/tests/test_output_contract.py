"""Output contract: every emitted record must satisfy the published schemas.

The generator validates complaint / ground_truth / metadata against
schemas/*.json before writing; a violation rejects the record (audited in
rejected.jsonl) instead of letting it into the corpus.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import contracts  # noqa: E402
from src import pipeline as pl  # noqa: E402

CLEAN_TEXT = (
    "Buenos días, escribo para reportar una fuga de agua en el garaje del edificio, "
    "concretamente en la plaza número veintitrés, donde el agua sale a presión de una "
    "tubería y empieza a extenderse por el suelo. Además se ha afectado la barandilla "
    "de la rampa de acceso y hay riesgo de inundación para los vehículos almacenados, "
    "por lo que ruego vengan cuanto antes a arreglarlo. Muchas gracias de antemano, "
    "un saludo del vecino de la plaza veintitrés."
)

SCENARIO = {"category": "fontanería", "subcategory": "fugas",
            "location_zone": "garaje", "location_detail": "plaza 23",
            "urgency": "alta", "affected_elements": [], "risks": ["inundación"]}
COMMUNITY = {"id": "C001", "name": "X", "type": "bloque_pisos", "num_viviendas": 10,
             "year": 2000, "has_elevator": False, "has_pool": False, "has_garage": True}


class StubClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}],
                "usage": {}}

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]

    def extract_json(self, response):
        return json.loads(response["choices"][0]["message"]["content"])


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(pl, "sample_scenario", lambda: dict(SCENARIO))
    monkeypatch.setattr(pl, "should_be_multi_problem", lambda: False)
    monkeypatch.setattr(pl, "sample_community", lambda: dict(COMMUNITY))
    monkeypatch.setattr(pl, "sample_persona", lambda: None)
    monkeypatch.setattr(pl, "sample_modifier", lambda: None)
    monkeypatch.setattr(pl, "build_generation_prompt", lambda **kw: [
        {"role": "user", "content": "stub"}])
    return pl


def test_schemas_are_valid():
    contracts.check_schemas()


def test_valid_record_passes_the_contract(env):
    client = StubClient({"user_text": CLEAN_TEXT, "timestamp": "2024-03-15T11:23:47+0000"})
    result = env.generate_complaint(client, "Q0001")
    assert result is not None
    complaint, ground_truth, metadata = result
    assert contracts.validate_record(complaint, ground_truth, metadata) == []
    assert "config_hash" in metadata["provenance"]


def test_missing_timestamp_is_rejected(env):
    rejected = []
    client = StubClient({"user_text": CLEAN_TEXT})
    result = env.generate_complaint(client, "Q0001", reject_sink=rejected.append)
    assert result is None
    assert rejected and rejected[0]["stage"] == "contract"
    assert "timestamp" in rejected[0]["detail"]


def test_unparseable_timestamp_is_rejected(env):
    rejected = []
    client = StubClient({"user_text": CLEAN_TEXT, "timestamp": "ayer por la tarde"})
    result = env.generate_complaint(client, "Q0001", reject_sink=rejected.append)
    assert result is None
    assert rejected[0]["reason"] == "schema_violation"


def test_timestamp_is_normalised_to_iso(env):
    client = StubClient({"user_text": CLEAN_TEXT, "timestamp": "2024-03-15T11:23:47+0000"})
    complaint, _, _ = env.generate_complaint(client, "Q0001")
    assert complaint["timestamp"] == "2024-03-15T11:23:47+00:00"


def test_provenance_is_recorded(env):
    client = StubClient({"user_text": CLEAN_TEXT, "timestamp": "2024-03-15T11:23:47+0000"})
    provenance = {"model": "generator", "model_id": "some-model", "seed": 42,
                  "config_hash": "abc", "prompt_fingerprints": {"generate_complaint": "f"}}
    _, _, metadata = env.generate_complaint(client, "Q0001", provenance=provenance)
    prov = metadata["provenance"]
    assert prov["model"] == "generator"
    assert prov["model_id"] == "some-model"
    assert prov["seed"] == 42
    assert prov["config_hash"] == "abc"
    assert prov["generated_at"]


def test_config_hash_is_stable_and_order_independent():
    a = contracts.config_hash({"b": 1, "a": [1, 2]})
    b = contracts.config_hash({"a": [1, 2], "b": 1})
    assert a == b and len(a) == 64


def test_record_missing_required_field_is_rejected(env, monkeypatch):
    """A ground truth that violates the contract is rejected, not written."""
    complaint = {"id": "Q0001", "user_text": CLEAN_TEXT,
                 "timestamp": "2024-03-15T11:23:47+00:00"}
    ground_truth = {"complaint_id": "Q0001", "extractions": [
        {"category": {"principal": "fontanería", "subcategory": "fugas"},
         "affected_elements": [],
         "location": {"zone": "garaje", "detail": ""},
         "risks": ["inundación"]}]}  # urgency missing
    metadata = {"complaint_id": "Q0001", "persona": None, "modifier": None,
                "community": dict(COMMUNITY),
                "provenance": {"model": "", "seed": None, "config_hash": ""}}
    violations = contracts.validate_record(complaint, ground_truth, metadata)
    assert any("urgency" in v for v in violations), violations


@pytest.mark.parametrize("raw,expected", [
    ("2024-03-15T11:23:47+0000", "2024-03-15T11:23:47+00:00"),
    ("2024-03-15T11:23:47Z", "2024-03-15T11:23:47+00:00"),
    ("2024-03-15T11:23:47.123456+02:00", "2024-03-15T11:23:47.123456+02:00"),
    ("2024-03-15T11:23:47", "2024-03-15T11:23:47+00:00"),
])
def test_timestamp_formats_are_normalised(raw, expected):
    assert pl.normalize_timestamp(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, 1710500000, True, "ayer por la tarde", "2024-13-45"])
def test_unparseable_timestamps_return_none(raw):
    assert pl.normalize_timestamp(raw) is None


def test_config_hash_ignores_credentials():
    a = contracts.config_hash({"endpoints": {"e": {"api_key": "sk-one"}}})
    b = contracts.config_hash({"endpoints": {"e": {"api_key": "sk-two"}}})
    assert a == b


def test_metadata_without_modifier_key_is_rejected():
    complaint = {"id": "Q0001", "user_text": CLEAN_TEXT, "timestamp": "2024-03-15T11:23:47+00:00"}
    ground_truth = {"complaint_id": "Q0001", "extractions": [
        {"category": {"principal": "fontanería", "subcategory": "fugas"},
         "affected_elements": [], "location": {"zone": "garaje", "detail": ""},
         "risks": ["inundación"], "urgency": "alta"}]}
    metadata = {"complaint_id": "Q0001", "persona": None, "community": dict(COMMUNITY),
                "provenance": {"model": "m", "seed": None, "config_hash": "h"}}
    assert any("modifier" in v for v in contracts.validate_record(complaint, ground_truth, metadata))
