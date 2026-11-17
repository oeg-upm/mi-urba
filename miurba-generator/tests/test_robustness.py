"""Unit tests for LLM client robustness (5a/5b/5c) and parallel workers (5d).

Pure code paths with stubbed HTTP — no network, no real config.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import LLMClient  # noqa: E402
from src import pipeline as pl  # noqa: E402

CLEAN_TEXT = 'Buenos días, escribo para reportar una fuga de agua en el garaje del edificio, concretamente en la plaza número veintitrés, donde el agua sale a presión de una tubería y empieza a extenderse por el suelo. Además se ha afectado la barandilla de la rampa de acceso y hay riesgo de inundación para los vehículos almacenados, por lo que ruego vengan cuanto antes a arreglarlo. Muchas gracias de antemano, un saludo del vecino de la plaza veintitrés.'



class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {"choices": [{"message": {"content": "{}"}}], "usage": {}}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests as rq
            err = rq.HTTPError(f"{self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._payload


# ── 5a: model_load_wait ───────────────────────────────────────

def test_model_load_wait_5xx_does_not_consume_retries(monkeypatch):
    """Two 503 (loading) then success: no retry consumed, call succeeds."""
    calls = {"n": 0}
    def fake_post(url, **kw):
        calls["n"] += 1
        return FakeResp(503) if calls["n"] <= 2 else FakeResp(200)

    monkeypatch.setattr("src.llm.requests.post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = LLMClient(base_url="http://x", model_load_wait=900, max_retries=3)
    data = client.chat([{"role": "user", "content": "hi"}])  # must not raise
    assert data["choices"][0]["message"]["content"] == "{}"
    assert client._stats["errors"] == 2  # counted as errors but no attempts


def test_no_model_load_wait_5xx_consumes_retries(monkeypatch):
    """Without model_load_wait, a 503 burns retries and eventually fails."""
    def fake_post(url, **kw):
        return FakeResp(503)

    monkeypatch.setattr("src.llm.requests.post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = LLMClient(base_url="http://x", max_retries=2)
    with pytest.raises(RuntimeError, match="2 attempts"):
        client.chat([{"role": "user", "content": "hi"}])


def test_model_load_window_expiry_consumes_retries(monkeypatch):
    """5xx after the load window expired counts as a normal retry."""
    def fake_post(url, **kw):
        return FakeResp(503)

    monkeypatch.setattr("src.llm.requests.post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = LLMClient(base_url="http://x", model_load_wait=0, max_retries=2)
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hi"}])


# ── 5b: connection_backoff ────────────────────────────────────

def test_connection_backoff_waits_longer(monkeypatch):
    """With connection_backoff, connection errors wait >= 20s * attempt."""
    waits = []
    def fake_post(url, **kw):
        raise requests.ConnectionError("server down")

    monkeypatch.setattr("src.llm.requests.post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda s: waits.append(s))
    client = LLMClient(base_url="http://x", retry_delay=5.0,
                       connection_backoff=True, max_retries=2)
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hi"}])
    assert waits[0] >= 20.0  # generous floor


def test_no_connection_backoff_standard_wait(monkeypatch):
    waits = []
    def fake_post(url, **kw):
        raise requests.ConnectionError("server down")

    monkeypatch.setattr("src.llm.requests.post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda s: waits.append(s))
    client = LLMClient(base_url="http://x", retry_delay=5.0, max_retries=2)
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hi"}])
    assert waits[0] == pytest.approx(5.0)  # retry_delay * 1


# ── 5c: extra_params passthrough ──────────────────────────────

def test_extra_params_landed_in_payload(monkeypatch):
    captured = {}
    def fake_post(url, **kw):
        captured.update(kw["json"])
        return FakeResp(200)

    monkeypatch.setattr("src.llm.requests.post", fake_post)
    client = LLMClient(base_url="http://x",
                       extra_params={"enable_thinking": False,
                                     "chat_template_kwargs": {"enable_thinking": False}})
    client.chat([{"role": "user", "content": "hi"}])
    assert captured["enable_thinking"] is False
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["temperature"] == 0.85  # normal params intact


def test_no_extra_params_clean_payload(monkeypatch):
    captured = {}
    def fake_post(url, **kw):
        captured.update(kw["json"])
        return FakeResp(200)

    monkeypatch.setattr("src.llm.requests.post", fake_post)
    LLMClient(base_url="http://x").chat([{"role": "user", "content": "hi"}])
    assert set(captured) == {"model", "messages", "temperature"}


# ── 5d: parallel workers + thread-safe appends ────────────────

class StubClient:
    """Client stub that simulates slow, parallel-safe LLM calls."""

    def __init__(self, texts, delay=0.01):
        self._texts = texts
        self._delay = delay
        self.lock = threading.Lock()
        self.calls = 0
        self._stats = {"tokens_in": 0, "tokens_out": 0}

    stats = property(lambda self: self._stats)

    def chat(self, messages):
        with self.lock:
            self.calls += 1
        time.sleep(self._delay)
        filler = ("el vecino reporta una fuga de agua en el garaje y pide reparacion "
                  "urgente de la tuberia ademas se ha afectado la luz de la escalera "
                  f"numero {self.calls} y ruega disculpen las molestias ocasionadas")
        return {"choices": [{"message": {"content": json.dumps(
            {"user_text": f"texto {self.calls} {filler}",
             "timestamp": "2026-10-22T10:00:00"})}}]}

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]

    def extract_json(self, response):
        return json.loads(response["choices"][0]["message"]["content"])


@pytest.fixture
def pipeline_env(monkeypatch, tmp_path):
    from src import pipeline as p
    scenario = {"category": "cubierta", "subcategory": "tejas",
                "location_zone": "exterior", "urgency": "alta",
                "affected_elements": [], "risks": ["ninguno"]}
    monkeypatch.setattr(p, "sample_scenario", lambda *a, **k: dict(scenario))
    monkeypatch.setattr(p, "should_be_multi_problem", lambda *a, **k: False)
    monkeypatch.setattr(p, "sample_community", lambda *a, **k: {
        "id": "C001", "name": "X", "type": "bloque", "num_viviendas": 10, "year": 2000,
        "has_elevator": False, "has_pool": False, "has_garage": False})
    monkeypatch.setattr(p, "sample_persona", lambda *a, **k: None)
    monkeypatch.setattr(p, "sample_modifier", lambda *a, **k: None)
    monkeypatch.setattr(p, "build_generation_prompt", lambda **kw: [
        {"role": "user", "content": "s"}])
    return p


def test_parallel_workers_write_all_records_without_duplicates(pipeline_env, tmp_path):
    pl = pipeline_env
    client = StubClient(texts=None, delay=0.02)
    stats = pl.generate_dataset(
        client, num_records=8, start_id=1,
        output_dir=str(tmp_path), num_workers=4,
    )
    assert stats["generated"] == 8 and stats["failed"] == 0
    lines = (tmp_path / "raw" / "complaints.jsonl").read_text().splitlines()
    ids = [json.loads(l)["id"] for l in lines]
    assert len(ids) == 8 and len(set(ids)) == 8  # all present, no duplicates


def test_sequential_still_works_after_refactor(pipeline_env, tmp_path):
    pl = pipeline_env
    stats = pl.generate_dataset(
        StubClient(None), num_records=3, start_id=1,
        output_dir=str(tmp_path), num_workers=1,
    )
    assert stats["generated"] == 3


def test_resume_counts_as_generated(pipeline_env, tmp_path):
    pl = pipeline_env
    raw = tmp_path / "raw" / "complaints.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"id": "Q0001", "user_text": "x"}) + "\n")
    stats = pl.generate_dataset(
        StubClient(None), num_records=2, start_id=1,
        output_dir=str(tmp_path), resume=True, num_workers=2,
    )
    assert stats["generated"] == 2  # Q0001 skipped+counted, Q0002 generated
