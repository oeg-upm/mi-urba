"""Unit tests for 2-stage validation logic (pure code paths).

Covers validate_complaint's parsing/fail-open branches and the
regeneration loop in generate_complaint, using stub LLM clients.
No network, no real config, no prompt-content assertions.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import validate_complaint  # noqa: E402

CLEAN_TEXT = 'Buenos días, escribo para reportar una fuga de agua en el garaje del edificio, concretamente en la plaza número veintitrés, donde el agua sale a presión de una tubería y empieza a extenderse por el suelo. Además se ha afectado la barandilla de la rampa de acceso y hay riesgo de inundación para los vehículos almacenados, por lo que ruego vengan cuanto antes a arreglarlo. Muchas gracias de antemano, un saludo del vecino de la plaza veintitrés.'



class StubClient:
    """Minimal LLMClient stand-in driven by scripted responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        idx = min(self.calls - 1, len(self._responses) - 1)
        return self._responses[idx]

    def extract_json(self, response):
        import json
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)


def _resp(payload):
    import json
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


PROBLEMS = [{"category": "cubierta", "subcategory": "tejas",
             "location_zone": "exterior", "urgency": "alta",
             "affected_elements": [], "risks": ["ninguno"]}]


# ── validate_complaint branches ───────────────────────────────

def test_valid_verdict_returns_true():
    client = StubClient([_resp({"valid": True, "issues": []})])
    is_valid, issues, degraded, _raw = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is True and issues == [] and degraded is False


def test_invalid_verdict_returns_false_with_issues():
    client = StubClient([_resp({"valid": False, "issues": ["urgency"]})])
    is_valid, issues, degraded, _raw = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is False and issues == ["urgency"] and degraded is False


def test_malformed_json_fails_closed():
    client = StubClient([{"choices": [{"message": {"content": "not json {"}}]}])
    is_valid, issues, degraded, _raw = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is False and degraded is True
    assert issues and issues[0].startswith("judge_error:")


def test_validation_exception_fails_closed():
    class BrokenClient:
        def chat(self, messages):
            raise RuntimeError("server down")

        def extract_json(self, response):  # pragma: no cover
            raise AssertionError("should not be reached")

    is_valid, issues, degraded, raw = validate_complaint(BrokenClient(), "texto", PROBLEMS)
    assert is_valid is False and degraded is True
    assert issues == ["judge_error:runtimeerror"]
    assert raw["error_kind"] == "runtimeerror"
    # el fallo queda caracterizado: dos intentos registrados, ambos con su tipo
    kinds = [d["kind"] for d in raw["_diagnostics"]]
    assert kinds == ["runtimeerror", "runtimeerror"]


def test_non_list_issues_coerced_to_list():
    client = StubClient([_resp({"valid": False, "issues": "urgency"})])
    is_valid, issues, _, _raw = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is False and issues == ["urgency"]


def test_missing_valid_key_is_not_approval():
    client = StubClient([_resp({"issues": []})])
    is_valid, _, degraded, _raw = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is False and degraded is False


# ── generate_complaint regeneration loop ──────────────────────

@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    """Stub sampling + prompt building so generate_complaint is deterministic."""
    from src import pipeline as pl

    scenario = {"category": "cubierta", "subcategory": "tejas",
                "location_zone": "exterior", "location_detail": "azotea",
                "urgency": "alta", "affected_elements": [], "risks": ["ninguno"]}
    monkeypatch.setattr(pl, "sample_scenario", lambda *a, **k: dict(scenario))
    monkeypatch.setattr(pl, "should_be_multi_problem", lambda *a, **k: False)
    monkeypatch.setattr(pl, "sample_community", lambda *a, **k: {
        "id": "C001", "name": "X", "type": "bloque", "num_viviendas": 10, "year": 2000,
        "has_elevator": False, "has_pool": False, "has_garage": False})
    monkeypatch.setattr(pl, "sample_persona", lambda *a, **k: None)
    monkeypatch.setattr(pl, "sample_modifier", lambda *a, **k: None)
    monkeypatch.setattr(pl, "build_generation_prompt", lambda **kw: [
        {"role": "user", "content": "stub"}])
    return pl


def _gen_client(texts):
    """Generator stub returning one queued generation per call."""
    import json

    class GenClient:
        def __init__(self):
            self._texts = list(texts)
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            content = json.dumps({"user_text": self._texts[min(self.calls - 1, len(self._texts) - 1)],
                                  "timestamp": "2026-10-19T10:00:00"})
            return {"choices": [{"message": {"content": content}}]}

        def extract_content(self, response):
            return response["choices"][0]["message"]["content"]

        def extract_json(self, response):
            return json.loads(response["choices"][0]["message"]["content"])

        @property
        def stats(self):
            return {"tokens_in": 0, "tokens_out": 0}

    return GenClient()


class JudgeClient:
    """Judge stub with a scripted sequence of verdicts."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        idx = min(self.calls - 1, len(self._verdicts) - 1)
        import json
        return {"choices": [{"message": {"content": json.dumps(self._verdicts[idx])}}]}

    def extract_json(self, response):
        import json
        return json.loads(response["choices"][0]["message"]["content"])


def test_generation_rejected_without_explicit_judge(fake_env):
    """validate=True without a judge client raises instead of self-judging."""
    pl = fake_env
    gen = _gen_client([CLEAN_TEXT])
    with pytest.raises(ValueError, match="generation.validator"):
        pl.generate_complaint(gen, "Q0001", validate=True, validator_client=None)
    assert gen.calls == 0


def test_generation_accepted_when_judge_valid(fake_env):
    pl = fake_env
    gen = _gen_client([CLEAN_TEXT])
    judge = JudgeClient([{"valid": True, "issues": []}])
    result = pl.generate_complaint(gen, "Q0001", validate=True,
                                   validator_client=judge)
    assert result is not None
    assert result[0]["user_text"] == CLEAN_TEXT
    assert gen.calls == 1 and judge.calls == 1


def test_regenerates_on_invalid_verdict_keeps_last(fake_env):
    pl = fake_env
    gen = _gen_client(["Q999_1.png esto es un texto contaminado con nombre de archivo interno y más contenido para superar el límite de longitud establecido", CLEAN_TEXT])
    judge = JudgeClient([{"valid": False, "issues": ["urgency"]},
                         {"valid": True, "issues": []}])
    result = pl.generate_complaint(gen, "Q0001", validate=True,
                                   max_validation_retries=3,
                                   validator_client=judge)
    assert result[0]["user_text"] == CLEAN_TEXT
    assert gen.calls == 3  # initial (dirty) + hygiene regen + judge regen


def test_discards_record_after_exhausting_retries(fake_env):
    pl = fake_env
    gen = _gen_client(["Q999_1.png texto contaminado con identificador interno para forzar regeneraciones del sistema de higiene automático", "Q998_1.png segundo texto contaminado también con identificador interno para seguir forzando", CLEAN_TEXT])          # 1 initial + 2 regenerations
    judge = JudgeClient([{"valid": False, "issues": ["x"]}])  # always invalid
    result = pl.generate_complaint(gen, "Q0001", validate=True,
                                   max_validation_retries=2,
                                   validator_client=judge)
    # Hygiene regenerates the two dirty texts; the 3rd is clean and passes
    # hygiene, then the judge (always invalid) exhausts the validation
    # retries — the record is discarded, never written.
    assert result is None
    assert gen.calls == 3  # initial + 2 regenerations, both rejected by hygiene


def test_no_validation_by_default_single_call(fake_env):
    pl = fake_env
    gen = _gen_client([CLEAN_TEXT])
    result = pl.generate_complaint(gen, "Q0001")
    assert result is not None and gen.calls == 1


# ── taxonomy validation: no silent fallbacks ──────────────────

def _minimal_taxonomy():
    return {
        "categories": {"fontanería": ["fugas"]},
        "weights": {"category": {"fontanería": 1}},
        "category_elements": {"fontanería": ["tubería"]},
        "category_zones": {"fontanería": ["portal"]},
        "category_risks": {"fontanería": ["ninguno"]},
        "urgency_weights": {"fontanería": {"media": 1}},
        "affected_elements": {"tubería": ["vertical"]},
        "location_zones": {"portal": ["escalera"]},
        "risks": ["ninguno"],
        "urgency_levels": ["media"],
        "risk_expressions": {"ninguno": ""},
        "urgency_expressions": {"media": ""},
    }


def test_taxonomy_accepts_complete_config():
    from src.validators import validate_taxonomy
    validate_taxonomy(_minimal_taxonomy())


def test_taxonomy_rejects_category_without_subcategories():
    from src.validators import validate_taxonomy
    cfg = _minimal_taxonomy()
    cfg["categories"]["varios"] = []
    with pytest.raises(ValueError, match="no subcategories"):
        validate_taxonomy(cfg)


def test_taxonomy_rejects_missing_category_mapping():
    from src.validators import validate_taxonomy
    cfg = _minimal_taxonomy()
    del cfg["category_risks"]["fontanería"]
    with pytest.raises(ValueError, match="category_risks"):
        validate_taxonomy(cfg)


def test_taxonomy_rejects_unknown_element_reference():
    from src.validators import validate_taxonomy
    cfg = _minimal_taxonomy()
    cfg["category_elements"]["fontanería"] = ["inexistente"]
    with pytest.raises(ValueError, match="unknown element"):
        validate_taxonomy(cfg)


def test_sampling_does_not_invent_subcategory(monkeypatch):
    """An empty subcategory list must fail, never yield the literal 'general'."""
    from src import sampling
    monkeypatch.setattr(sampling.cfg, "get_weights", lambda: {"category": {"varios": 1}})
    monkeypatch.setattr(sampling.cfg, "get_categories", lambda: {"varios": []})
    with pytest.raises(ValueError, match="no subcategories"):
        sampling.sample_scenario()


# ── judge v2: per-field verdicts, invented items, problem count ──

def test_grave_field_invalidates_record():
    client = StubClient([_resp({"valid": True, "issues": [],
                                "fields": {"urgency": "grave", "category": "ok"}})])
    is_valid, issues, _, _raw = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is False and "urgency" in issues


def test_minor_fields_do_not_invalidate():
    client = StubClient([_resp({"valid": True, "issues": [],
                                "fields": {"urgency": "menor", "category": "ok"}})])
    is_valid, _, _, _raw = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is True


def test_invented_items_invalidate_record():
    client = StubClient([_resp({"valid": True, "issues": [],
                                "inventado": {"elementos": ["muro/cortafuegos"]}})])
    is_valid, issues, _, _raw = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is False and any(i.startswith("inventado:") for i in issues)


def test_problem_count_mismatch_invalidates_record():
    client = StubClient([_resp({"valid": True, "issues": [], "n_problems_texto": 3})])
    is_valid, issues, _, _raw = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is False and any(i.startswith("n_problems:") for i in issues)


def test_retry_conversation_includes_previous_attempt_and_verdict(fake_env, monkeypatch):
    """History must grow as (assistant generation, judge comment) pairs."""
    pl = fake_env
    histories = []

    def fake_build(**kw):
        histories.append(list(kw.get("history") or []))
        return [{"role": "user", "content": "stub"}]

    monkeypatch.setattr(pl, "build_generation_prompt", fake_build)

    gen = _gen_client([CLEAN_TEXT])
    judge = JudgeClient([{"valid": False, "issues": ["urgency"]},
                         {"valid": True, "issues": []}])
    result = pl.generate_complaint(gen, "Q0001", validate=True,
                                   max_validation_retries=2,
                                   validator_client=judge)

    assert result is not None
    assert len(histories) == 2
    assert histories[0] == []                      # primer intento: sin historial
    assert len(histories[1]) == 1                  # reintento: un par
    import json as _json
    assistant_json, comment = histories[1][0]
    assert _json.loads(assistant_json)["user_text"] == CLEAN_TEXT
    assert "urgency" in comment


# ── element coherence table ───────────────────────────────────

def test_element_subcategories_reject_unknown_category():
    from src.validators import validate_element_subcategories
    cfg = _minimal_taxonomy()
    cfg["categories"]["nueva"] = ["x"]
    cfg["category_elements"]["nueva"] = ["tubería"]
    cfg["category_zones"]["nueva"] = ["portal"]
    cfg["category_risks"]["nueva"] = ["ninguno"]
    cfg["urgency_weights"]["nueva"] = {"media": 1}
    with pytest.raises(ValueError, match="no entry for category 'nueva'"):
        validate_element_subcategories(cfg, table={"fontanería": {"*": {"tubería": ["vertical"]}}})


def test_element_subcategories_reject_subcategory_not_in_vocabulary():
    from src.validators import validate_element_subcategories
    cfg = _minimal_taxonomy()
    with pytest.raises(ValueError, match="tubería/inventada"):
        validate_element_subcategories(
            cfg, table={"fontanería": {"*": {"tubería": ["inventada"]}}})


def test_element_subcategories_reject_element_outside_category():
    from src.validators import validate_element_subcategories
    cfg = _minimal_taxonomy()
    cfg["affected_elements"]["muro"] = ["carga"]
    with pytest.raises(ValueError, match="not allowed for category 'fontanería'"):
        validate_element_subcategories(
            cfg, table={"fontanería": {"*": {"muro": ["carga"]}}})


def test_sampling_only_returns_coherent_element_subcategories(monkeypatch):
    """agua_caliente must never draw tubería/saneamiento or desagüe."""
    from src import sampling
    monkeypatch.setattr(sampling.cfg, "get_element_subcategories", lambda: {
        "fontanería": {"*": {"tubería": ["vertical", "horizontal"]},
                       "agua_caliente": {"tubería": ["vertical", "horizontal"]}}})
    monkeypatch.setattr(sampling.cfg, "get_affected_elements", lambda: {
        "tubería": ["vertical", "horizontal", "desagüe", "saneamiento"]})
    for _ in range(200):
        for e in sampling.sample_affected_elements("fontanería", "agua_caliente"):
            assert e["subcategory"] in ("vertical", "horizontal")


# ── judge failures: retry + characterisation ──────────────────

class _FlakyJudge:
    """Fails the first call with the given exception, then answers correctly."""

    def __init__(self, first_exc, content=None):
        import json as _j
        self.first_exc = first_exc
        self.content = content if content is not None else _j.dumps({"valid": True, "issues": []})
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            if isinstance(self.first_exc, str):
                return {"choices": [{"message": {"content": self.first_exc}}]}
            raise self.first_exc
        return {"choices": [{"message": {"content": self.content}}]}

    def extract_json(self, response):
        import json as _j
        return _j.loads(response["choices"][0]["message"]["content"])

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]


def test_unparseable_verdict_is_retried_once():
    judge = _FlakyJudge("no soy json")
    is_valid, issues, degraded, raw = validate_complaint(judge, "texto", PROBLEMS)
    assert judge.calls == 2 and degraded is False and is_valid is True
    kinds = [d["kind"] for d in raw["_diagnostics"]]
    assert kinds == ["unparseable", "ok"]
    assert raw["_diagnostics"][0]["raw"] == "no soy json"


def test_two_unparseable_verdicts_stay_invalid_with_diagnostics():
    class _AlwaysBad(_FlakyJudge):
        def __init__(self):
            super().__init__("basura", content="basura")

    judge = _AlwaysBad()
    is_valid, issues, degraded, raw = validate_complaint(judge, "texto", PROBLEMS)
    assert is_valid is False and degraded is True
    assert issues == ["judge_error:unparseable"]
    assert [d["kind"] for d in raw["_diagnostics"]] == ["unparseable", "unparseable"]
