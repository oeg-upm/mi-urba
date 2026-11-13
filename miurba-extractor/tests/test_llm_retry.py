"""Tests for the extractor LLMClient retry loop (largest coverage gap).

Stubs requests.post — no network. Mirrors generator test_robustness but
targets the extractor client (60% coverage before this file).

Run from miurba-extractor/:  python -m pytest tests/ -v
"""

import sys
import time
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import LLMClient, VALID_STRUCTURED_MODES  # noqa: E402


class Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {"choices": [{"message": {"content": "{}"}}],
                                    "usage": {}}

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(str(self.status_code))
            err.response = self
            raise err

    def json(self):
        return self._payload


def test_retry_loop_succeeds_after_transient_500(monkeypatch):
    calls = {"n": 0}
    def fake_post(url, **kw):
        calls["n"] += 1
        return Resp(500) if calls["n"] < 3 else Resp(200)

    monkeypatch.setattr("src.llm.requests.post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = LLMClient(base_url="http://x", max_retries=3)
    data = client.chat([{"role": "user", "content": "hi"}])
    assert data["choices"][0]["message"]["content"] == "{}"
    assert calls["n"] == 3


def test_retries_exhausted_raises_runtime_error(monkeypatch):
    monkeypatch.setattr("src.llm.requests.post", lambda u, **k: Resp(500))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = LLMClient(base_url="http://x", max_retries=2)
    with pytest.raises(RuntimeError, match="2 attempts"):
        client.chat([{"role": "user", "content": "hi"}])


def test_model_load_wait_polls_without_consuming_retries(monkeypatch):
    """Polls 15s while the load window is open; expires after max_retries.

    time is faked: advances 30s per call so the 600s window closes after
    ~20 polls without the test waiting real seconds.
    """
    import types
    clock = {"t": 1000.0}
    fake_time = types.SimpleNamespace(
        time=lambda: clock["t"],
        sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
    )
    calls = {"n": 0}
    def fake_post(url, **kw):
        calls["n"] += 1
        return Resp(503)  # always "loading"

    monkeypatch.setattr("src.llm.time", fake_time)
    monkeypatch.setattr("src.llm.requests.post", fake_post)
    client = LLMClient(base_url="http://x", model_load_wait=600, max_retries=3)
    with pytest.raises(RuntimeError, match="3 attempts"):
        client.chat([{"role": "user", "content": "hi"}])
    assert client._stats["load_waits"] > 3  # polled for free, retries intact
    assert client._stats["errors"] == 3     # only the real attempts counted


def test_connection_backoff_generous_wait(monkeypatch):
    waits = []
    monkeypatch.setattr("src.llm.requests.post",
                        lambda u, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    monkeypatch.setattr(time, "sleep", lambda s: waits.append(s))
    client = LLMClient(base_url="http://x", retry_delay=5.0,
                       connection_backoff=True, max_retries=2)
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hi"}])
    assert waits[0] >= 20.0


def test_api_key_header_sent(monkeypatch):
    captured = {}
    monkeypatch.setattr("src.llm.requests.post",
                        lambda u, **k: captured.update(k) or Resp(200))
    LLMClient(base_url="http://x", api_key="sk-secret").chat(
        [{"role": "user", "content": "hi"}])
    assert captured["headers"]["Authorization"] == "Bearer sk-secret"


def test_base_url_normalizes_v1_suffix():
    c1 = LLMClient(base_url="http://x/v1")
    c2 = LLMClient(base_url="http://x/v1/")
    c3 = LLMClient(base_url="http://x")
    assert c1.api_url == c2.api_url == c3.api_url == "http://x/v1/chat/completions"


def test_extract_json_fences_and_control_chars():
    client = LLMClient(base_url="http://x")
    fenced = {"choices": [{"message": {"content": "```json\n{\"a\": 1}\n```"}}]}
    assert client.extract_json(fenced) == {"a": 1}
    control = {"choices": [{"message": {"content": '{"t": "a\nb"}'}}]}
    assert client.extract_json(control)["t"] == "a\nb"


def test_structured_modes_still_validated():
    assert LLMClient(base_url="http://x", structured_output="json_object")
    with pytest.raises(ValueError):
        LLMClient(base_url="http://x", structured_output="xml")
