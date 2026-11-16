"""Tests for extract_dataset: resume, truncate, workers, stats and edge cases.

Previously 0% covered (pipeline.py extract_dataset block). Uses stub LLM
clients — no network, no real config.

Run from miurba-extractor/:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import extract_dataset  # noqa: E402


class StubLLM:
    def __init__(self, fail_ids=()):
        self.calls = 0
        self.fail_ids = set(fail_ids)

    def chat(self, messages):
        self.calls += 1
        return {"choices": [{"message": {"content": json.dumps({
            "extractions": [], "image_descriptions": []})}}]}

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]

    def extract_json(self, response):
        return json.loads(response["choices"][0]["message"]["content"])

    @property
    def stats(self):
        return {"tokens_in": 0, "tokens_out": 0}


@pytest.fixture
def raw_dataset(tmp_path):
    """3 complaints in raw/ without images (single-call path)."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    with open(raw / "complaints.jsonl", "w") as f:
        for i in range(1, 4):
            f.write(json.dumps({
                "id": f"Q{i:04d}", "user_text": f"texto {i}",
                "user_pics": [], "timestamp": "t"}) + "\n")
    return tmp_path


def _run(tmp_path, resume=True, workers=1, client=None):
    return extract_dataset(
        client=client or StubLLM(),
        raw_dir=str(tmp_path / "raw"),
        images_dir=str(tmp_path / "images"),
        output_dir=str(tmp_path / "annotated"),
        resume=resume,
        num_workers=workers,
    )


def test_happy_path_stats_and_output(raw_dataset):
    stats = _run(raw_dataset)
    assert stats["extracted"] == 3 and stats["failed"] == 0 and stats["total"] == 3
    lines = (raw_dataset / "annotated" / "extractions.jsonl").read_text().splitlines()
    assert len(lines) == 3


def test_resume_skips_existing(raw_dataset):
    _run(raw_dataset)
    client = StubLLM()
    stats = _run(raw_dataset, client=client)
    assert stats["skipped"] == 3 and stats["extracted"] == 0
    assert client.calls == 0  # no LLM calls on resumed records


def test_resume_false_truncates_instead_of_duplicating(raw_dataset):
    """R5 pin: resume=False must truncate, not append duplicates."""
    _run(raw_dataset)
    stats = _run(raw_dataset, resume=False)
    assert stats["extracted"] == 3
    lines = (raw_dataset / "annotated" / "extractions.jsonl").read_text().splitlines()
    assert len(lines) == 3  # not 6


def test_missing_raw_dir_returns_complete_stats(tmp_path):
    stats = extract_dataset(
        client=StubLLM(),
        raw_dir=str(tmp_path / "nope"),
        images_dir=None,
        output_dir=str(tmp_path / "ann"),
    )
    # pin fix b245330: complete stats even on early return
    assert stats["extracted"] == 0 and stats["total"] == 0
    assert "total_tokens_in" in stats and "output_path" in stats


def test_workers_parallel_writes_all_records(raw_dataset):
    stats = _run(raw_dataset, workers=3)
    assert stats["extracted"] == 3 and stats["failed"] == 0
    lines = (raw_dataset / "annotated" / "extractions.jsonl").read_text().splitlines()
    assert len(lines) == 3  # thread-safe appends, no duplicates


def test_failed_record_counted(raw_dataset):
    # all complaints have no images; make the client raise on every call
    class Broken:
        stats = {"tokens_in": 0, "tokens_out": 0}

        def chat(self, messages):
            raise RuntimeError("down")
    stats = _run(raw_dataset, client=Broken())
    assert stats["failed"] == 3 and stats["extracted"] == 0
