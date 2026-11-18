"""Reproducibility: per-record sampling streams and campaign manifest.

A seeded campaign must sample the same scenario for a given record id
regardless of the number of workers, the processing order or how many retries
earlier records consumed.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import json
import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline as pl  # noqa: E402
from src import sampling  # noqa: E402


def _scenarios(seed, ids):
    return {rid: pl.sample_scenario(pl._record_rng(seed, rid)) for rid in ids}


def test_same_seed_and_id_gives_same_scenario():
    a = pl.sample_scenario(pl._record_rng(42, "Q0001"))
    b = pl.sample_scenario(pl._record_rng(42, "Q0001"))
    assert a == b


def test_different_ids_give_different_scenarios():
    a = pl.sample_scenario(pl._record_rng(42, "Q0001"))
    b = pl.sample_scenario(pl._record_rng(42, "Q0002"))
    assert a != b


def test_different_seeds_give_different_scenarios():
    a = pl.sample_scenario(pl._record_rng(42, "Q0001"))
    b = pl.sample_scenario(pl._record_rng(43, "Q0001"))
    assert a != b


def test_scenario_is_independent_of_processing_order():
    ids = [f"Q{i:04d}" for i in range(1, 11)]
    forward = _scenarios(7, ids)
    backward = _scenarios(7, list(reversed(ids)))
    assert forward == backward


def test_scenario_is_independent_of_workers(tmp_path, monkeypatch):
    """The same seed and ids give the same scenarios with 1 and with 8 workers.

    Drives two real dataset runs with a stub client and compares the sampled
    scenario (ground truth) and profile of every record.
    """

    def fake(client, record_id, **kwargs):
        scenario = sampling.sample_scenario(kwargs.get("rng"))
        community = sampling.sample_community(kwargs.get("rng"))
        persona = sampling.sample_persona(kwargs.get("rng"))
        return (
            {"id": record_id, "user_text": "texto limpio de prueba",
             "timestamp": "2024-03-15T11:23:47+00:00"},
            {"complaint_id": record_id,
             "extractions": [{"category": {"principal": scenario["category"],
                                            "subcategory": scenario["subcategory"]},
                              "affected_elements": scenario["affected_elements"],
                              "location": {"zone": scenario["location_zone"], "detail": ""},
                              "risks": scenario["risks"],
                              "urgency": scenario["urgency"]}]},
            {"complaint_id": record_id, "persona": persona["id"] if persona else None,
             "modifier": None, "community": community,
             "provenance": {"model": "m", "seed": 11, "config_hash": "h"}},
        )

    monkeypatch.setattr(pl, "generate_complaint", fake)

    class Stub:
        stats = {"tokens_in": 0, "tokens_out": 0}

    def run(workers, name):
        out = tmp_path / name
        pl.generate_dataset(client=Stub(), num_records=8, start_id=1,
                            output_dir=str(out), num_workers=workers, seed=11,
                            resume=False)
        gt = {json.loads(l)["complaint_id"]: json.loads(l)
              for l in (out / "raw" / "ground_truth.jsonl").read_text().splitlines()}
        md = {json.loads(l)["complaint_id"]: json.loads(l)
              for l in (out / "raw" / "metadata.jsonl").read_text().splitlines()}
        return {rid: {"scenario": gt[rid]["extractions"],
                      "persona": md[rid]["persona"],
                      "community": md[rid]["community"]} for rid in gt}

    assert run(1, "one") == run(8, "eight")


def test_no_seed_uses_global_rng():
    assert pl._record_rng(None, "Q0001") is None


def test_manifest_is_written(tmp_path, monkeypatch):
    def fake(client, record_id, **kwargs):
        return ({"id": record_id, "user_text": "texto limpio", "timestamp": "2024-03-15T11:23:47+00:00"},
                {"complaint_id": record_id, "extractions": []},
                {"complaint_id": record_id})

    monkeypatch.setattr(pl, "generate_complaint", fake)

    class Stub:
        stats = {"tokens_in": 0, "tokens_out": 0}

    stats = pl.generate_dataset(
        client=Stub(), num_records=2, start_id=1,
        output_dir=str(tmp_path / "out"), resume=False, seed=99,
        provenance={"model": "m", "model_id": "m-1", "seed": 99, "config_hash": "h"})

    manifest = json.loads(Path(stats["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["sampling"]["seed"] == 99
    assert "per-record" in manifest["sampling"]["policy"]
    assert manifest["provenance"]["model"] == "m"
    assert manifest["runs"][-1]["counts"]["generated"] == 2
    assert manifest["python"] and manifest["code_version"]


def test_samplers_accept_an_explicit_stream():
    rng = random.Random("x")
    assert sampling.sample_scenario(rng)["category"] in sampling.cfg.get_categories()
