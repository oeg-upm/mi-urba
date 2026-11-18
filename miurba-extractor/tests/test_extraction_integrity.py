"""Extraction robustness: per-record isolation, rejection audit, resume safety.

Run from miurba-extractor/:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import extract_dataset  # noqa: E402


class Stub:
    stats = {"tokens_in": 0, "tokens_out": 0}

    def __init__(self, payload="{}", raises=None):
        self._payload = payload
        self._raises = raises
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self._raises:
            raise self._raises
        return {"choices": [{"message": {"content": self._payload}}], "usage": {}}

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]

    def extract_json(self, response):
        return json.loads(response["choices"][0]["message"]["content"])


def _raw(tmp_path: Path, ids: list[str]) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with open(raw / "complaints.jsonl", "w", encoding="utf-8") as f:
        for rid in ids:
            f.write(json.dumps({"id": rid, "user_text": f"texto {rid}",
                                "timestamp": "2024-03-15T11:23:47+00:00"}) + "\n")
    return raw


def _run(tmp_path, client, **kwargs):
    return extract_dataset(
        client=client, raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "annotated"), **kwargs)


def test_json_list_output_fails_the_record_not_the_run(tmp_path):
    raw = _raw(tmp_path, ["Q0001", "Q0002"])
    stats = _run(tmp_path, Stub(payload="[]"), resume=False)
    assert stats["total"] == 2
    assert stats["failed"] == 2 and stats["extracted"] == 0
    assert (tmp_path / "annotated" / "rejected.jsonl").read_text().count("\n") == 2


def test_unexpected_exception_is_contained(tmp_path):
    _raw(tmp_path, ["Q0001"])
    stats = _run(tmp_path, Stub(raises=AttributeError("boom")), resume=False)
    assert stats["failed"] == 1
    rejected = json.loads((tmp_path / "annotated" / "rejected.jsonl").read_text().splitlines()[0])
    assert rejected["reason"] == "unexpected_error"


def test_missing_input_reports_input_missing(tmp_path):
    stats = extract_dataset(client=Stub(), raw_dir=str(tmp_path / "nope"),
                            output_dir=str(tmp_path / "ann"))
    assert stats["input_missing"] is True
    assert not (tmp_path / "ann").exists()  # no empty dir created


def test_duplicate_ids_are_reported(tmp_path, caplog):
    raw = tmp_path / "raw"
    raw.mkdir()
    with open(raw / "complaints.jsonl", "w", encoding="utf-8") as f:
        for _ in range(2):
            f.write(json.dumps({"id": "Q0001", "user_text": "x"}) + "\n")
    stats = _run(tmp_path, Stub(payload=json.dumps({"extractions": []})), resume=False)
    assert stats["total"] == 1
    assert "duplicate ids" in caplog.text


def test_stale_extractions_are_dropped_on_resume(tmp_path):
    _raw(tmp_path, ["Q0001"])
    ann = tmp_path / "annotated"
    ann.mkdir()
    with open(ann / "extractions.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"complaint_id": "Q0001", "extractions": []}) + "\n")
        f.write(json.dumps({"complaint_id": "Q9999", "extractions": []}) + "\n")
    client = Stub()
    stats = _run(tmp_path, client, resume=True)
    assert stats["extracted"] == 0 and stats["skipped"] == 1
    kept = [json.loads(l)["complaint_id"] for l in
            (ann / "extractions.jsonl").read_text().splitlines()]
    assert kept == ["Q0001"]  # Q9999 no longer has a complaint
    assert client.calls == 0


def test_empty_input_keeps_existing_extractions(tmp_path, caplog):
    """An empty input corpus must not prune a valid extractions file."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "complaints.jsonl").write_text("", encoding="utf-8")
    ann = tmp_path / "annotated"
    ann.mkdir()
    previous = json.dumps({"complaint_id": "Q0001", "extractions": []}) + "\n"
    (ann / "extractions.jsonl").write_text(previous, encoding="utf-8")

    stats = _run(tmp_path, Stub(payload=json.dumps({"extractions": []})), resume=True)
    assert stats["total"] == 0
    assert (ann / "extractions.jsonl").read_text(encoding="utf-8") == previous
    assert "Input corpus is empty" in caplog.text


def test_corrupt_lines_in_input_are_skipped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "complaints.jsonl").write_text(
        json.dumps({"id": "Q0001", "user_text": "ok"}) + "\n{\"id\": \"Q0002\"",
        encoding="utf-8")
    stats = _run(tmp_path, Stub(payload=json.dumps({"extractions": []})), resume=False)
    assert stats["total"] == 1
