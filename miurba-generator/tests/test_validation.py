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
    is_valid, issues = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is True and issues == []


def test_invalid_verdict_returns_false_with_issues():
    client = StubClient([_resp({"valid": False, "issues": ["urgency"]})])
    is_valid, issues = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is False and issues == ["urgency"]


def test_malformed_json_fails_open():
    client = StubClient([{"choices": [{"message": {"content": "not json {"}}]}])
    is_valid, issues = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is True and issues == []


def test_validation_exception_fails_open():
    class BrokenClient:
        def chat(self, messages):
            raise RuntimeError("server down")

        def extract_json(self, response):  # pragma: no cover
            raise AssertionError("should not be reached")

    is_valid, _ = validate_complaint(BrokenClient(), "texto", PROBLEMS)
    assert is_valid is True


def test_non_list_issues_coerced_to_empty():
    client = StubClient([_resp({"valid": False, "issues": "urgency"})])
    is_valid, issues = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is False and issues == []


def test_missing_valid_key_defaults_to_valid():
    client = StubClient([_resp({"issues": []})])
    is_valid, _ = validate_complaint(client, "texto", PROBLEMS)
    assert is_valid is True


# ── generate_complaint regeneration loop ──────────────────────

@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    """Stub sampling + prompt building so generate_complaint is deterministic."""
    from src import pipeline as pl

    scenario = {"category": "cubierta", "subcategory": "tejas",
                "location_zone": "exterior", "location_detail": "azotea",
                "urgency": "alta", "affected_elements": [], "risks": ["ninguno"]}
    monkeypatch.setattr(pl, "sample_scenario", lambda: dict(scenario))
    monkeypatch.setattr(pl, "should_be_multi_problem", lambda: False)
    monkeypatch.setattr(pl, "sample_community", lambda: {
        "name": "X", "type": "bloque", "num_viviendas": 10, "year": 2000,
        "has_elevator": False, "has_pool": False, "has_garage": False})
    monkeypatch.setattr(pl, "sample_persona", lambda: None)
    monkeypatch.setattr(pl, "sample_modifier", lambda: None)
    monkeypatch.setattr(pl, "should_have_photos", lambda: False)
    monkeypatch.setattr(pl, "sample_photo_count", lambda has: 0)
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


def test_generation_accepted_when_judge_valid(fake_env):
    pl = fake_env
    gen = _gen_client(["texto bueno"])
    judge = JudgeClient([{"valid": True, "issues": []}])
    result = pl.generate_complaint(gen, "Q0001", validate=True,
                                   validator_client=judge)
    assert result is not None
    assert result[0]["user_text"] == "texto bueno"
    assert gen.calls == 1 and judge.calls == 1


def test_regenerates_on_invalid_verdict_keeps_last(fake_env):
    pl = fake_env
    gen = _gen_client(["texto malo", "texto bueno"])
    judge = JudgeClient([{"valid": False, "issues": ["urgency"]},
                         {"valid": True, "issues": []}])
    result = pl.generate_complaint(gen, "Q0001", validate=True,
                                   max_validation_retries=3,
                                   validator_client=judge)
    assert result[0]["user_text"] == "texto bueno"
    assert gen.calls == 2


def test_keeps_last_text_after_exhausting_retries(fake_env):
    pl = fake_env
    gen = _gen_client(["t1", "t2", "t3"])          # 1 initial + 2 regenerations
    judge = JudgeClient([{"valid": False, "issues": ["x"]}])  # always invalid
    result = pl.generate_complaint(gen, "Q0001", validate=True,
                                   max_validation_retries=2,
                                   validator_client=judge)
    assert result is not None
    assert result[0]["user_text"] == "t3"  # last text kept
    assert gen.calls == 3


def test_no_validation_by_default_single_call(fake_env):
    pl = fake_env
    gen = _gen_client(["texto"])
    result = pl.generate_complaint(gen, "Q0001")
    assert result is not None and gen.calls == 1
