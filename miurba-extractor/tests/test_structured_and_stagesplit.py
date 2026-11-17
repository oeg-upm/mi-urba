"""Unit tests for structured-output modes and record validation.

Pure code paths with stub clients and temp fixtures — no network, no
real config values, no prompt-content assertions.

Run from miurba-extractor/:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import LLMClient, VALID_STRUCTURED_MODES  # noqa: E402
from src.config import get_extraction_json_schema  # noqa: E402
from src.pipeline import extract_complaint  # noqa: E402


# ── Structured-output modes ───────────────────────────────────

def test_mode_none_sends_no_response_format(monkeypatch):
    captured = {}
    client = LLMClient(base_url="http://x")
    monkeypatch.setattr("src.llm.requests.post", lambda url, **kw: captured.update(kw["json"]) or _fake_resp())
    client.chat([{"role": "user", "content": "hi"}])
    assert "response_format" not in captured


def test_json_object_mode_sends_type_only(monkeypatch):
    captured = {}
    client = LLMClient(base_url="http://x", structured_output="json_object")
    monkeypatch.setattr("src.llm.requests.post", lambda url, **kw: captured.update(kw["json"]) or _fake_resp())
    client.chat([{"role": "user", "content": "hi"}])
    assert captured["response_format"] == {"type": "json_object"}


def test_json_schema_mode_sends_schema(monkeypatch):
    captured = {}
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    client = LLMClient(base_url="http://x", structured_output="json_schema",
                       json_schema=schema)
    monkeypatch.setattr("src.llm.requests.post", lambda url, **kw: captured.update(kw["json"]) or _fake_resp())
    client.chat([{"role": "user", "content": "hi"}])
    rf = captured["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="structured_output"):
        LLMClient(base_url="http://x", structured_output="true")


def test_json_schema_mode_requires_schema():
    with pytest.raises(ValueError, match="requires json_schema"):
        LLMClient(base_url="http://x", structured_output="json_schema")


def _fake_resp():
    class R:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "{}"}}], "usage": {}}
    return R()


# ── Schema loaders (enums synced from injected fixture config) ──

@pytest.fixture
def fake_taxonomy(monkeypatch):
    """Inject a fake taxonomy into the config cache for prompt building."""
    import src.config as sc
    monkeypatch.setattr(sc, "_CONFIG", {
        "categories": {"cubierta": ["tejas"]},
        "location_zones": {"norte": []},
        "risks": ["riesgo_x"],
        "urgency_levels": ["alta"],
    })


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Isolate tests from the global config cache."""
    import src.config as sc
    sc._CONFIG = None
    yield
    sc._CONFIG = None


@pytest.fixture
def fixture_cfg(tmp_path):
    """Minimal config with a custom taxonomy."""
    cfg = {
        "taxonomy": "taxonomy.yaml",
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (tmp_path / "taxonomy.yaml").write_text(yaml.safe_dump({
        "risks": ["riesgo_x", "riesgo_y"],
        "urgency_levels": ["alta", "baja"],
        "location_zones": {"norte": [], "sur": []},
        "categories": {"cubierta": ["tejas"]},
        "affected_elements": {"muro": ["carga"], "suelo": ["solera"]},
    }, allow_unicode=True), encoding="utf-8")
    return p


def test_enum_sync_values(fixture_cfg, monkeypatch):
    from src.config import get_config
    monkeypatch.setattr("src.config.DEFAULT_CONFIG_PATH", fixture_cfg)
    schema = get_extraction_json_schema(get_config(fixture_cfg))
    defs = schema["$defs"]
    assert defs["risk"]["enum"] == ["riesgo_x", "riesgo_y"]
    assert defs["urgency"]["enum"] == ["alta", "baja"]
    assert defs["location"]["properties"]["zone"]["enum"] == ["norte", "sur"]
    props = defs["problemExtraction"]["properties"]
    assert props["affected_elements"]["maxItems"] == 5
    assert props["risks"]["maxItems"] == 5


def test_vocabulary_is_free_by_default(fixture_cfg, monkeypatch):
    """Open mode: category/elements accept any string, no taxonomy constraint."""
    from src.config import get_config
    monkeypatch.setattr("src.config.DEFAULT_CONFIG_PATH", fixture_cfg)
    schema = get_extraction_json_schema(get_config(fixture_cfg))
    props = schema["$defs"]["problemExtraction"]["properties"]
    assert "oneOf" not in props["category"]
    assert "oneOf" not in props["affected_elements"]["items"]


def test_constrain_vocabulary_builds_taxonomy_oneof(fixture_cfg, monkeypatch):
    """Closed mode: every label must exist in the taxonomy."""
    import jsonschema
    from src.config import get_config
    monkeypatch.setattr("src.config.DEFAULT_CONFIG_PATH", fixture_cfg)
    cfg = get_config(fixture_cfg)
    cfg["extraction"] = {"model": "m", "constrain_vocabulary": True}
    schema = get_extraction_json_schema(cfg)
    jsonschema.Draft202012Validator.check_schema(schema)
    props = schema["$defs"]["problemExtraction"]["properties"]
    assert len(props["category"]["oneOf"]) == 1  # only 'cubierta'
    assert len(props["affected_elements"]["items"]["oneOf"]) == 2

    def valid(category, elements):
        record = {"complaint_id": "Q0001", "extractions": [{
            "category": category, "affected_elements": elements,
            "location": {"zone": "norte", "detail": ""},
            "risks": ["riesgo_x"], "urgency": "alta"}]}
        return jsonschema.Draft202012Validator(schema).is_valid(record)

    assert valid({"principal": "cubierta", "subcategory": "tejas"}, [])
    assert not valid({"principal": "cubierta", "subcategory": "otra"}, [])
    assert not valid({"principal": "inventada", "subcategory": "tejas"}, [])
    assert valid({"principal": "cubierta", "subcategory": "tejas"},
                 [{"principal": "muro", "subcategory": "carga"}])
    assert not valid({"principal": "cubierta", "subcategory": "tejas"},
                     [{"principal": "muro", "subcategory": "solera"}])


# ── Stub client ───────────────────────────────────────────────

class StubLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]

    def extract_json(self, response):
        return json.loads(response["choices"][0]["message"]["content"])


# ── M2: invalid extractions shape fails the record ────────────

def test_non_list_extractions_fail_the_record(fake_taxonomy):
    complaint = {"id": "Q0001", "user_text": "texto"}
    text = StubLLM([{"choices": [{"message": {"content": json.dumps({
        "complaint_id": "Q0001", "extractions": "no soy lista"})}}]}])
    out = extract_complaint(text, complaint)
    assert out is None  # record failed, no garbage written


def test_missing_extractions_fail_the_record(fake_taxonomy):
    complaint = {"id": "Q0001", "user_text": "texto"}
    text = StubLLM([{"choices": [{"message": {"content": json.dumps({
        "complaint_id": "Q0001"})}}]}])
    out = extract_complaint(text, complaint)
    assert out is None


# ── M8: always-on schema validation ───────────────────────────

def test_invalid_record_fails_when_schema_provided(fake_taxonomy, fixture_cfg, monkeypatch):
    monkeypatch.setattr("src.config.DEFAULT_CONFIG_PATH", fixture_cfg)
    from src.config import get_config, get_extraction_json_schema
    schema = get_extraction_json_schema(get_config(fixture_cfg))
    complaint = {"id": "Q0001", "user_text": "texto"}
    # urgency "mega" no está en el enum del schema (alta/baja)
    text = StubLLM([{"choices": [{"message": {"content": json.dumps({
        "complaint_id": "Q0001",
        "extractions": [{"category": {"principal": "cubierta", "subcategory": "tejas"},
                          "affected_elements": [],
                          "location": {"zone": "norte", "detail": ""},
                          "risks": ["riesgo_x"], "urgency": "mega"}]})}}]}])
    out = extract_complaint(text, complaint, validation_schema=schema)
    assert out is None  # invalid → record failed


def test_valid_record_passes_schema_validation(fake_taxonomy, fixture_cfg, monkeypatch):
    monkeypatch.setattr("src.config.DEFAULT_CONFIG_PATH", fixture_cfg)
    from src.config import get_config, get_extraction_json_schema
    schema = get_extraction_json_schema(get_config(fixture_cfg))
    complaint = {"id": "Q0001", "user_text": "texto"}
    text = StubLLM([{"choices": [{"message": {"content": json.dumps({
        "complaint_id": "Q0001",
        "extractions": [{"category": {"principal": "cubierta", "subcategory": "tejas"},
                          "affected_elements": [],
                          "location": {"zone": "norte", "detail": ""},
                          "risks": ["riesgo_x"], "urgency": "alta"}]})}}]}])
    out = extract_complaint(text, complaint, validation_schema=schema)
    assert out is not None
