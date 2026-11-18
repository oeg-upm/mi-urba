"""Tests added after the final re-audit (informe/reaudit_y_cobertura.md).

Covers: R2 (string verdict), resume truncate pin, judge end-to-end with
validation_log, real build_generation_prompt, and extract_json tolerance.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import LLMClient  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src.prompts import build_generation_prompt  # noqa: E402

CLEAN_TEXT = 'Buenos días, escribo para reportar una fuga de agua en el garaje del edificio, concretamente en la plaza número veintitrés, donde el agua sale a presión de una tubería y empieza a extenderse por el suelo. Además se ha afectado la barandilla de la rampa de acceso y hay riesgo de inundación para los vehículos almacenados, por lo que ruego vengan cuanto antes a arreglarlo. Muchas gracias de antemano, un saludo del vecino de la plaza veintitrés.'



# ── fixtures ──────────────────────────────────────────────────

@pytest.fixture
def env(monkeypatch):
    scenario = {"category": "cubierta", "subcategory": "tejas",
                "location_zone": "exterior", "location_detail": "azotea",
                "urgency": "alta", "affected_elements": [], "risks": ["ninguno"]}
    monkeypatch.setattr(pl, "sample_scenario", lambda: dict(scenario))
    monkeypatch.setattr(pl, "should_be_multi_problem", lambda: False)
    monkeypatch.setattr(pl, "sample_community", lambda: {
        "id": "C001", "name": "X", "type": "bloque", "num_viviendas": 10, "year": 2000,
        "has_elevator": False, "has_pool": False, "has_garage": False})
    monkeypatch.setattr(pl, "sample_persona", lambda: None)
    monkeypatch.setattr(pl, "sample_modifier", lambda: None)
    monkeypatch.setattr(pl, "build_generation_prompt", lambda **kw: [
        {"role": "user", "content": "s"}])
    return pl


class GenClient:
    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return {"choices": [{"message": {"content": json.dumps(
            {"user_text": self._texts[min(self.calls - 1, len(self._texts) - 1)],
             "timestamp": "2024-03-15T11:23:47+00:00"})}}]}

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]

    def extract_json(self, response):
        return json.loads(response["choices"][0]["message"]["content"])

    @property
    def stats(self):
        return {"tokens_in": 0, "tokens_out": 0}


class Judge:
    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return {"choices": [{"message": {"content": json.dumps(
            self._verdicts[min(self.calls - 1, len(self._verdicts) - 1)])}}]}

    def extract_json(self, response):
        return json.loads(response["choices"][0]["message"]["content"])


# ── R2: string verdict must not pass ──────────────────────────

def test_verdict_string_false_is_rejected():
    client = GenClient([CLEAN_TEXT])
    judge = Judge([{"valid": "false", "issues": []}])
    log = []
    result = pl.generate_complaint(client, "Q0001", validate=True,
                                   max_validation_retries=1,
                                   validator_client=judge,
                                   validation_log=log)
    # string "false" is NOT True → rejected; 1 retry budget → 2 candidates → discarded
    assert judge.calls == 2 and result is None
    assert log == ["rejected", "rejected"]  # never "ok": strict, not truthy


def test_verdict_numeric_is_rejected():
    client = GenClient([CLEAN_TEXT])
    judge = Judge([{"valid": 1, "issues": []}])
    log = []
    result = pl.generate_complaint(client, "Q0001", validate=True,
                                   max_validation_retries=1,
                                   validator_client=judge,
                                   validation_log=log)
    assert judge.calls == 2 and result is None
    assert log == ["rejected", "rejected"]


def test_verdict_boolean_true_passes():
    client = GenClient([CLEAN_TEXT])
    judge = Judge([{"valid": True, "issues": []}])
    result = pl.generate_complaint(client, "Q0001", validate=True,
                                   validator_client=judge)
    assert result is not None


# ── judge end-to-end with validation_log ──────────────────────

def test_validation_log_records_outcomes(env):
    pl = env
    log = []
    client = GenClient(["Q999_1.png esto es un texto contaminado con nombre de archivo interno", CLEAN_TEXT])
    judge = Judge([{"valid": False, "issues": ["urgency"]},
                   {"valid": True, "issues": []}])
    result = pl.generate_complaint(client, "Q0001", validate=True,
                                   max_validation_retries=3,
                                   validator_client=judge,
                                   validation_log=log)
    assert result[0]["user_text"] == CLEAN_TEXT
    assert log == ["rejected", "ok"]


def test_validation_log_records_degraded(env, monkeypatch):
    pl = env
    monkeypatch.setattr(pl, "validate_complaint",
                        lambda c, t, p: (True, [], True, {}))
    log = []
    judge = Judge([{"valid": True, "issues": []}])
    result = pl.generate_complaint(GenClient([CLEAN_TEXT]), "Q0001",
                                   validate=True,
                                   max_validation_retries=1,
                                   validator_client=judge,
                                   validation_log=log)
    assert log == ["degraded", "degraded"]
    assert result is None  # an unreachable judge never counts as approval


# ── resume=False truncates (pin fix 119ae09) ──────────────────

def test_resume_false_truncates_outputs(env, tmp_path):
    pl = env
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    (raw / "complaints.jsonl").write_text(
        json.dumps({"id": "Q0001", "user_text": "viejo"}) + "\n")
    (raw / "ground_truth.jsonl").write_text("{}\n")
    (raw / "metadata.jsonl").write_text("{}\n")

    pl.generate_dataset(GenClient([CLEAN_TEXT]), num_records=1, start_id=1,
                        output_dir=str(tmp_path), resume=False)
    lines = (raw / "complaints.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["user_text"] == CLEAN_TEXT
    assert (raw / "ground_truth.jsonl").read_text().strip() != "{}"


# ── real build_generation_prompt (replaces the "fake" test) ───

def test_generation_prompt_from_config_components():
    """REAL test of build_generation_prompt (not a reimplementation)."""
    problems = [{
        "category": "electricidad", "subcategory": "enchufes",
        "location_zone": "portal", "location_detail": "hall",
        "urgency": "urgente",
        "affected_elements": [{"principal": "enchufe", "subcategory": "instalación_eléctrica"}],
        "risks": ["incendio"],
    }]
    community = {"name": "X", "type": "bloque", "num_viviendas": 10,
                 "year": 2000, "has_elevator": False, "has_pool": False,
                 "has_garage": False}
    messages = build_generation_prompt("Q0001", problems, community)
    user = messages[1]["content"]

    assert "enchufe (instalación eléctrica)" in user       # elements rendered (natural, no snake_case)
    assert "incendio" in user                              # risks present
    assert "urgente" in user                               # urgency label
    assert '"es urgente"' in user                          # expressions wired
    assert "RESPONDE SOLO con JSON" in user                # template present
    assert "user_text" in user and "timestamp" in user


def test_generation_prompt_omits_ninguno_risks():
    problems = [{
        "category": "jardinería", "subcategory": "riego",
        "location_zone": "zonas_verdes", "urgency": "baja",
        "affected_elements": [], "risks": ["ninguno"],
    }]
    community = {"name": "X", "type": "bloque", "num_viviendas": 10,
                 "year": 2000, "has_elevator": False, "has_pool": False,
                 "has_garage": False}
    user = build_generation_prompt("Q0001", problems, community)[1]["content"]
    problems_section = user.split("CONTEXTO:")[0]
    assert "Riesgos:" not in problems_section  # scenario desc omits risks


# ── extract_json tolerance ────────────────────────────────────

def test_extract_json_strips_markdown_fences():
    client = LLMClient(base_url="http://x")
    resp = {"choices": [{"message": {"content":
        "```json\n{\"a\": 1}\n```"}}]}
    assert client.extract_json(resp) == {"a": 1}


def test_extract_json_tolerates_control_chars():
    client = LLMClient(base_url="http://x")
    resp = {"choices": [{"message": {"content":
        '{"user_text": "linea1\nlinea2"}'}}]}  # raw newline inside string
    assert client.extract_json(resp)["user_text"] == "linea1\nlinea2"
