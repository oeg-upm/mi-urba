"""Corpus integrity: reconciliation, per-record isolation and merge safety.

The three corpus files are separate appends, so a run killed between them (or a
re-plan that reassigns id ranges) must not leave the corpus inconsistent or
duplicated.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline as pl  # noqa: E402


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ids(path: Path, key: str) -> list[str]:
    if not path.exists():
        return []
    return [json.loads(l)[key] for l in path.read_text().splitlines() if l.strip()]


# ── reconcile_outputs ─────────────────────────────────────────

def test_reconcile_keeps_complete_records(tmp_path):
    raw, gt, md = (tmp_path / "c.jsonl", tmp_path / "g.jsonl", tmp_path / "m.jsonl")
    _write(raw, [{"id": "Q0001"}])
    _write(gt, [{"complaint_id": "Q0001"}])
    _write(md, [{"complaint_id": "Q0001"}])
    assert pl.reconcile_outputs(raw, gt, md) == {"Q0001"}
    assert _ids(raw, "id") == ["Q0001"]


def test_reconcile_drops_record_missing_ground_truth(tmp_path):
    raw, gt, md = (tmp_path / "c.jsonl", tmp_path / "g.jsonl", tmp_path / "m.jsonl")
    _write(raw, [{"id": "Q0001"}, {"id": "Q0002"}])
    _write(gt, [{"complaint_id": "Q0001"}])  # Q0002 lost mid-write
    _write(md, [{"complaint_id": "Q0001"}])
    assert pl.reconcile_outputs(raw, gt, md) == {"Q0001"}
    assert _ids(raw, "id") == ["Q0001"]  # removed from every file


def test_reconcile_drops_record_missing_metadata(tmp_path):
    raw, gt, md = (tmp_path / "c.jsonl", tmp_path / "g.jsonl", tmp_path / "m.jsonl")
    _write(raw, [{"id": "Q0001"}])
    _write(gt, [{"complaint_id": "Q0001"}])
    _write(md, [])
    assert pl.reconcile_outputs(raw, gt, md) == set()
    assert _ids(raw, "id") == []


def test_reconcile_skips_corrupt_lines(tmp_path):
    raw, gt, md = (tmp_path / "c.jsonl", tmp_path / "g.jsonl", tmp_path / "m.jsonl")
    raw.write_text(json.dumps({"id": "Q0001"}) + "\n{\"id\": \"Q0002\"")  # truncated, no newline
    _write(gt, [{"complaint_id": "Q0001"}])
    _write(md, [{"complaint_id": "Q0001"}])
    assert pl.reconcile_outputs(raw, gt, md) == {"Q0001"}
    # the corrupt tail is gone and the file ends with a newline, so the next
    # append cannot be glued onto it
    assert raw.read_text() == json.dumps({"id": "Q0001"}) + "\n"


def test_reconcile_drops_non_object_lines(tmp_path):
    raw, gt, md = (tmp_path / "c.jsonl", tmp_path / "g.jsonl", tmp_path / "m.jsonl")
    raw.write_text(json.dumps(["not", "an", "object"]) + "\n"
                   + json.dumps({"id": "Q0001"}) + "\n")
    _write(gt, [{"complaint_id": "Q0001"}])
    _write(md, [{"complaint_id": "Q0001"}])
    assert pl.reconcile_outputs(raw, gt, md) == {"Q0001"}
    assert _ids(raw, "id") == ["Q0001"]


def test_reconcile_drops_duplicate_ids(tmp_path):
    raw, gt, md = (tmp_path / "c.jsonl", tmp_path / "g.jsonl", tmp_path / "m.jsonl")
    _write(raw, [{"id": "Q0001"}, {"id": "Q0001"}])
    _write(gt, [{"complaint_id": "Q0001"}])
    _write(md, [{"complaint_id": "Q0001"}])
    assert pl.reconcile_outputs(raw, gt, md) == {"Q0001"}
    assert _ids(raw, "id") == ["Q0001"]


def test_reconcile_is_noop_when_files_absent(tmp_path):
    assert pl.reconcile_outputs(tmp_path / "c", tmp_path / "g", tmp_path / "m") == set()


# ── merge safety ──────────────────────────────────────────────

def test_merge_without_model_dirs_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="no model output"):
        pl.merge_outputs(str(tmp_path), ["missing"], raw_subdir="raw")


def test_merge_is_atomic_and_leaves_no_temp_files(tmp_path):
    for model, ids in (("alpha", ["Q0001"]), ("beta", ["Q0002"])):
        _write(tmp_path / model / "raw" / "complaints.jsonl", [{"id": i} for i in ids])
        _write(tmp_path / model / "raw" / "ground_truth.jsonl", [{"complaint_id": i} for i in ids])
        _write(tmp_path / model / "raw" / "metadata.jsonl", [{"complaint_id": i} for i in ids])
    stats = pl.merge_outputs(str(tmp_path), ["alpha", "beta"])
    assert stats["complaints"] == 2
    assert not list((tmp_path / "raw").glob("*.tmp"))


def test_merge_reports_file_length_mismatch(tmp_path, caplog):
    _write(tmp_path / "alpha" / "raw" / "complaints.jsonl",
           [{"id": "Q0001"}, {"id": "Q0002"}])
    _write(tmp_path / "alpha" / "raw" / "ground_truth.jsonl", [{"complaint_id": "Q0001"}])
    _write(tmp_path / "alpha" / "raw" / "metadata.jsonl", [{"complaint_id": "Q0001"}])
    stats = pl.merge_outputs(str(tmp_path), ["alpha"])
    assert stats["complaints"] == 2 and stats["ground_truth"] == 1
    assert "disagree in length" in caplog.text


# ── per-record isolation ──────────────────────────────────────

def test_unexpected_failure_rejects_only_that_record(tmp_path, monkeypatch):
    """A crash inside generate_complaint must not abort the dataset run."""
    calls = {"n": 0}

    def flaky(client, record_id, **kwargs):
        calls["n"] += 1
        if record_id == "Q0002":
            raise AttributeError("'list' object has no attribute 'get'")
        return ({"id": record_id, "user_text": "texto limpio de prueba", "timestamp": "t"},
                {"complaint_id": record_id, "extractions": []},
                {"complaint_id": record_id})

    monkeypatch.setattr(pl, "generate_complaint", flaky)

    class Stub:
        stats = {"tokens_in": 0, "tokens_out": 0}

    stats = pl.generate_dataset(
        client=Stub(), num_records=3, start_id=1,
        output_dir=str(tmp_path / "out"), resume=False)

    assert stats["generated"] == 2 and stats["failed"] == 1
    assert calls["n"] == 3
    rejected = (tmp_path / "out" / "raw" / "rejected.jsonl").read_text().splitlines()
    assert len(rejected) == 1
    assert json.loads(rejected[0])["reason"] == "unexpected_error"
