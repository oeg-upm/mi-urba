"""Tests for the config-only CLI: generation plan, config defaults and wiring.

Run from miurba-generator/:  python -m pytest tests/ -v
No network access is required.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cli import _generation_plan, cmd_merge  # noqa: E402
from src.config import get_config, get_generation_config, get_paths  # noqa: E402
from src.llm import LLMClient  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def config_file(tmp_path):
    """Minimal valid config.yaml with a 2-model generation plan."""
    cfg = {
        "endpoints": {
            "test": {"base_url": "http://localhost:9999", "api_key": "sk-test"},
        },
        "models": {
            "alpha": {"endpoint": "test", "model": "alpha-model", "temperature": 0.5},
            "beta": {"endpoint": "test", "model": "beta-model", "temperature": 0.6},
        },
        "generation": {
            "default_output_dir": str(tmp_path / "out"),
            "batches": [{"model": "alpha", "count": 2}, {"model": "beta", "count": 3}],
        },
        "paths": {},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


# ── Config defaults ───────────────────────────────────────────

def test_generation_defaults_fill_missing_keys(config_file):
    cfg = get_config(config_file)
    gen = get_generation_config(cfg)
    for key, default in [
        ("num_records", 100), ("start_id", 1), ("seed", None),
        ("resume", True), ("timeout", 300), ("max_retries", 3),
        ("retry_delay", 5.0),
    ]:
        assert gen[key] == default, f"{key} should default to {default}"


def test_generation_config_respects_explicit_values(config_file):
    cfg = get_config(config_file)
    cfg["generation"]["seed"] = 42
    cfg["generation"]["max_retries"] = 8
    gen = get_generation_config(cfg)
    assert gen["seed"] == 42
    assert gen["max_retries"] == 8


def test_repo_config_generation_section_fully_consumed():
    """Every key in the repo's config.yaml generation: section must be known."""
    from src.config import DEFAULT_CONFIG_PATH, _DEFAULT_GENERATION
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    unknown = set(cfg.get("generation", {})) - set(_DEFAULT_GENERATION)
    assert not unknown, f"config.yaml has generation keys the code ignores: {unknown}"


# ── Generation plan ───────────────────────────────────────────

def test_plan_from_batches(config_file):
    plan = _generation_plan(get_generation_config(get_config(config_file)))
    assert plan == [("alpha", 2), ("beta", 3)]


def test_plan_from_models_uses_num_records(config_file):
    cfg = get_config(config_file)
    cfg["generation"]["models"] = ["alpha", "beta"]
    cfg["generation"]["num_records"] = 7
    plan = _generation_plan(get_generation_config(cfg))
    assert plan == [("alpha", 7), ("beta", 7)]


def test_plan_empty_raises(config_file):
    cfg = get_config(config_file)
    cfg["generation"]["batches"] = []
    cfg["generation"]["models"] = None
    with pytest.raises(ValueError, match="generation.batches"):
        _generation_plan(get_generation_config(cfg))


# ── CLI wiring ────────────────────────────────────────────────

def test_cli_exposes_only_config_flag(config_file, monkeypatch):
    """The real parser accepts `generate --config <path>` and nothing else."""
    import cli

    captured = {}

    def fake_cmd_generate(args):
        captured["config"] = args.config

    monkeypatch.setattr(cli, "cmd_generate", fake_cmd_generate)
    monkeypatch.setattr(sys, "argv", ["cli.py", "generate", "--config", str(config_file)])
    cli.main()
    assert captured["config"] == str(config_file)

    monkeypatch.setattr(sys, "argv", ["cli.py", "generate", "-v"])
    with pytest.raises(SystemExit):
        cli.main()  # -v is not a flag of this CLI


def test_merge_command_merges_model_outputs(config_file, tmp_path, capsys):
    """cmd_merge merges each model dir into the shared raw dir, deduplicated by id."""
    cfg = get_config(config_file)
    out = cfg["generation"]["default_output_dir"]
    for model, n in [("alpha", 2), ("beta", 3)]:
        raw = Path(out) / model / "raw"
        raw.mkdir(parents=True)
        for i in range(n):
            rid = f"Q{i + 1:04d}"
            (raw / "complaints.jsonl").open("a").write(
                json.dumps({"id": rid, "user_text": f"text {model} {i}"}) + "\n")
            (raw / "ground_truth.jsonl").open("a").write(json.dumps({"complaint_id": rid}) + "\n")
            (raw / "metadata.jsonl").open("a").write(json.dumps({"complaint_id": rid}) + "\n")

    class Args:
        config = str(config_file)

    cmd_merge(Args())

    lines = (Path(out) / "raw" / "complaints.jsonl").read_text().splitlines()
    records = [json.loads(l) for l in lines]
    ids = [r["id"] for r in records]
    assert ids == ["Q0001", "Q0002", "Q0003"]  # overlapping ids collapsed
    assert records[0]["user_text"] == "text alpha 0"  # first occurrence wins
    for name in ("ground_truth.jsonl", "metadata.jsonl"):
        assert len((Path(out) / "raw" / name).read_text().splitlines()) == 3


def test_llm_client_receives_config_retry_policy(config_file):
    """timeout/max_retries/retry_delay from generation: reach the client."""
    cfg = get_config(config_file)
    gen = get_generation_config(cfg)
    client = LLMClient(
        base_url="http://localhost:9999",
        model="alpha-model",
        temperature=0.5,
        timeout=gen["timeout"],
        max_retries=gen["max_retries"],
        retry_delay=gen["retry_delay"],
    )
    assert client.timeout == 300
    assert client.max_retries == 3
    assert client.retry_delay == 5.0


def test_paths_defaults(config_file):
    paths = get_paths(get_config(config_file))
    assert paths["complaints_file"] == "complaints.jsonl"
    assert paths["raw_subdir"] == "raw"
