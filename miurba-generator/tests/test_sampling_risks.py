"""Regression test: a sampled scenario must be satisfiable.

'ninguno' means "no risk" and cannot coexist with a concrete risk. The sampler
used to pick it as one more member of the pool, producing ground truths like
['ninguno', 'sanitario']: the 10K corpus carried 289 of them (2.2% of 13.159
extractions). The judge then had to decide whether a text can mention 'ninguno'
and 'sanitario' at once, which is undecidable, and every judge lost quality on
those cases.

Run from miurba-generator/:  python -m pytest tests/test_sampling_risks.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg  # noqa: E402
from src import sampling  # noqa: E402


def _load_default_config():
    cfg.load_config(PROJECT_ROOT / "config" / "config.yaml")


def test_risks_never_mix_ninguno_with_concrete_risks():
    _load_default_config()
    for category in cfg.get_categories():
        for _ in range(200):
            risks = sampling.sample_risks(category, "media")
            assert risks, f"{category}: sampled an empty risk list"
            if "ninguno" in risks:
                assert risks == ["ninguno"], (
                    f"{category}: 'ninguno' sampled with concrete risks: {risks}"
                )
            else:
                assert all(r != "ninguno" for r in risks)


def test_risks_come_from_the_category_pool():
    _load_default_config()
    pools = cfg.get_config()["category_risks"]
    for category, pool in pools.items():
        allowed = set(pool)
        for _ in range(100):
            risks = sampling.sample_risks(category, "baja")
            concrete = [r for r in risks if r != "ninguno"]
            for risk in concrete:
                assert risk in allowed, f"{category}: {risk!r} not in {sorted(allowed)}"


def test_sample_scenario_risks_are_coherent():
    _load_default_config()
    for _ in range(300):
        scenario = sampling.sample_scenario()
        risks = scenario["risks"]
        assert risks, "scenario without risks"
        if "ninguno" in risks:
            assert risks == ["ninguno"], f"impossible scenario risks: {risks}"
