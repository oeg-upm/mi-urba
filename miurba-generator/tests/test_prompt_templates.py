"""Prompt templates: byte-exact rendering and loader behaviour.

The golden fixture (tests/fixtures/prompt_golden.json) was captured before the
templates were moved out of the code; these tests guarantee that externalising
them did not change a single character of what the model receives.
`prompt_golden.v1.json` is the pre-templates baseline kept for reference; the
prompts changed on purpose when the worked examples were added.
"""
import json
from pathlib import Path

import pytest

from src import prompt_loader
from src.prompts import build_generation_prompt, build_validation_prompt

GOLDEN = json.loads((Path(__file__).parent / "fixtures/prompt_golden.json")
                    .read_text(encoding="utf-8"))

# Hermetic: the urgency map is the one in the repo config.yaml at capture time.
# Other tests mutate the global config cache, so the golden must not depend on it.
URGENCY_EXPRESSIONS = {
    "inmediata": ["es una emergencia", "peligro inminente", "necesito que vengan ya mismo"],
    "urgente": ["es urgente", "no puede esperar", "vengan cuanto antes"],
    "alta": ["es importante", "me preocupa", "deberían revisarlo pronto"],
    "media": ["cuando puedan", "no tiene prisa pero"],
    "baja": ["es una sugerencia", "para cuando haya tiempo"],
    "planificable": ["mantenimiento preventivo", "para revisar a largo plazo"],
}


RISK_EXPRESSIONS = {
    "eléctrico": "riesgo eléctrico",
    "estructural": "riesgo de derrumbe",
    "sanitario": "riesgo sanitario",
    "incendio": "riesgo de incendio",
    "inundación": "riesgo de inundación",
    "caída": "peligro de caída",
    "accesibilidad": "riesgo de accidente por accesibilidad",
    "seguridad_persona": "riesgo para las personas",
}


@pytest.fixture(autouse=True)
def _fixed_prompt_tables(monkeypatch):
    """Pin the config tables the prompts depend on (other tests mutate them)."""
    from src import config as cfg_module
    monkeypatch.setattr(cfg_module, "get_urgency_expressions",
                        lambda cfg=None: URGENCY_EXPRESSIONS)
    monkeypatch.setattr(cfg_module, "get_risk_expressions",
                        lambda cfg=None: RISK_EXPRESSIONS)

COMMUNITY = {"name": "Urbanización El Roble", "type": "urbanizacion", "num_viviendas": 24,
             "year": 1998, "has_elevator": True, "has_pool": True, "has_garage": False,
             "id": "C001"}

P1 = {"category": "fontanería", "subcategory": "fugas", "location_zone": "portal",
      "location_detail": "hall de entrada", "urgency": "media",
      "affected_elements": [{"principal": "tubería", "subcategory": "vertical"}],
      "risks": ["ninguno"]}

P2 = {"category": "cubierta", "subcategory": "goteras", "location_zone": "exterior",
      "location_detail": "cubierta", "urgency": "alta",
      "affected_elements": [{"principal": "cubierta", "subcategory": "teja"},
                            {"principal": "muro", "subcategory": "cortafuegos"}],
      "risks": ["estructural", "inundación"]}

PERSONA = {"profile": "Inquilino transeúnte", "style": "mensajes cortos y directos"}
MODIFIER = {"description": "Escribe con prisa", "style_override": "frases telegráficas"}


def test_generation_prompt_simple_matches_golden():
    assert build_generation_prompt("Q0001", [P1], COMMUNITY) == GOLDEN["generation_simple"]


def test_generation_prompt_multi_matches_golden():
    got = build_generation_prompt("Q0002", [P1, P2], COMMUNITY,
                                  persona=PERSONA, modifier=MODIFIER)
    assert got == GOLDEN["generation_multi"]


def test_validation_prompt_matches_golden():
    assert build_validation_prompt("Texto de prueba de la queja.", [P1, P2]) == GOLDEN["validation"]


def test_urgency_prohibitions_are_derived_from_config():
    """The forbidden urgency expressions must come from the config, never hand-written.

    For a 'media' scenario the expressions of every other level must be listed as
    prohibited, and the expressions of 'media' itself must not be.
    """
    user = build_generation_prompt("Q0001", [P1], COMMUNITY)[1]["content"]
    forbidden_block = user.split("PROHIBIDAS en este registro")[1].split("- Elementos")[0]
    for level, exprs in URGENCY_EXPRESSIONS.items():
        for e in exprs:
            if level == "media":
                assert f"'{e}' (nivel {level})" not in forbidden_block
            else:
                assert f"'{e}' (nivel {level})" in forbidden_block


def test_loader_leaves_unknown_placeholders_visible():
    text = prompt_loader.render("generate_complaint", "user")
    assert "{{problems_text}}" in text, "a missing variable must stay visible, not vanish"


def test_loader_raises_on_unknown_section():
    with pytest.raises(prompt_loader.PromptError, match="no section"):
        prompt_loader.render("validate_record", "nope")


def test_fingerprint_is_stable_and_sha256():
    fp = prompt_loader.fingerprint("generate_complaint")
    assert len(fp) == 64 and fp == prompt_loader.fingerprint("generate_complaint")
    assert set(prompt_loader.fingerprint_all()) >= {"generate_complaint", "validate_record",
                                                    "examples"}


def test_retry_is_a_real_multi_turn_conversation():
    """Retries must be system,user + one assistant/user pair per rejected attempt."""
    base = build_generation_prompt("Q0001", [P1], COMMUNITY)
    assert [m["role"] for m in base] == ["system", "user"]

    one = build_generation_prompt("Q0001", [P1], COMMUNITY,
                                  history=[('{"user_text": "intento 1"}', "rechazado por urgency")])
    assert [m["role"] for m in one] == ["system", "user", "assistant", "user"]
    assert one[1]["content"] == base[1]["content"]      # el prompt base no se toca
    assert one[2]["content"] == '{"user_text": "intento 1"}'
    assert "rechazado por urgency" in one[3]["content"]

    two = build_generation_prompt("Q0001", [P1], COMMUNITY,
                                  history=[("g1", "c1"), ("g2", "c2")])
    assert [m["role"] for m in two] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert two[2]["content"] == "g1" and two[3]["content"] == "c1"
    assert two[4]["content"] == "g2" and two[5]["content"] == "c2"


def test_judge_evidence_reaches_the_retry_prompt():
    """The quoted evidence must travel with the issues, not be discarded."""
    from src.pipeline import validate_complaint

    class _Judge:
        def chat(self, messages):
            return {"choices": [{"message": {"content": json.dumps({
                "fields": {"urgency": "grave"},
                "urgencias_detectadas": ["inmediata"],
                "expresion_citada": "es una emergencia",
                "evidencia": "es una emergencia, vengan ya",
                "valid": False})}}]}

        def extract_json(self, response):
            return json.loads(response["choices"][0]["message"]["content"])

    is_valid, issues, _, _raw = validate_complaint(_Judge(), "texto", [P1])
    assert is_valid is False
    joined = " | ".join(issues)
    assert "evidencia: es una emergencia, vengan ya" in joined
    assert "expresion_citada: es una emergencia" in joined
