"""Unit tests for prompt-building helper code paths (pure logic).

These tests cover CODE branches only — formatting, fallbacks and edge
cases of the helpers in src/prompts.py and accessors in src/config.py.
They do not assert any specific config value or prompt wording: sample
data is injected, never read from config.yaml.

Run from miurba-generator/:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import prompts  # noqa: E402


# ── format_affected_elements ──────────────────────────────────

def test_elements_rendered_as_principal_subcategory_pairs():
    p = {"affected_elements": [
        {"principal": "canalón", "subcategory": "cubierta"},
        {"principal": "muro", "subcategory": "fachada"},
    ]}
    out = prompts.format_affected_elements(p)
    assert "canalón (cubierta)" in out
    assert "muro (fachada)" in out


def test_elements_empty_or_missing_render_nothing():
    assert prompts.format_affected_elements({}) == ""
    assert prompts.format_affected_elements({"affected_elements": []}) == ""


# ── format_risks ──────────────────────────────────────────────

RISK_EXPR = {"incendio": "riesgo de incendio", "caída": "peligro de caída"}


def test_risks_mapped_via_expression_table():
    out = prompts.format_risks({"risks": ["incendio"]}, RISK_EXPR)
    assert "riesgo de incendio" in out
    assert "incendio" not in out.replace("riesgo de incendio", "")


def test_unknown_risk_falls_back_to_verbatim():
    out = prompts.format_risks({"risks": ["desconocido"]}, RISK_EXPR)
    assert "desconocido" in out  # no KeyError, no dropped risk


def test_risks_ninguno_renders_nothing():
    assert prompts.format_risks({"risks": ["ninguno"]}, RISK_EXPR) == ""
    assert prompts.format_risks({"risks": []}, RISK_EXPR) == ""
    assert prompts.format_risks({}, RISK_EXPR) == ""


def test_multiple_risks_all_rendered():
    out = prompts.format_risks({"risks": ["incendio", "caída", "raro"]}, RISK_EXPR)
    assert "riesgo de incendio" in out
    assert "peligro de caída" in out
    assert "raro" in out


def test_risk_expr_defaults_to_config_when_omitted():
    """Calling without the table must not raise (reads config internally)."""
    out = prompts.format_risks({"risks": ["incendio"]})
    assert isinstance(out, str)


# ── Urgency expression block builder (code path) ─────────────

def test_urgency_block_joins_levels_as_lines():
    """The block builder renders one line per level, quoting expressions."""
    lines = [
        f"     * {level}: debe incluir alguna de " + ", ".join(f'"{e}"' for e in exprs)
        for level, exprs in {
            "alta": ["es importante"],
            "baja": ["es una sugerencia", "para cuando haya tiempo"],
        }.items()
    ]
    block = "\n".join(lines)
    assert block.count("\n") == 1  # two levels → two lines
    assert '"es importante"' in block
    assert '"para cuando haya tiempo"' in block


# ── validate_expression_sections (code path) ─────────────────

def test_validate_expression_sections_passes_with_full_coverage(monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "get_risks", lambda: ["incendio", "ninguno"])
    monkeypatch.setattr(cfg, "get_risk_expressions", lambda: {"incendio": "riesgo de incendio"})
    monkeypatch.setattr(cfg, "get_urgency_levels", lambda: ["alta"])
    monkeypatch.setattr(cfg, "get_urgency_expressions", lambda: {"alta": ["expr"]})
    from src.prompts import validate_expression_sections
    validate_expression_sections()  # no raise


def test_validate_expression_sections_fails_on_missing_risk(monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "get_risks", lambda: ["incendio", "raro"])
    monkeypatch.setattr(cfg, "get_risk_expressions", lambda: {"incendio": "riesgo"})
    monkeypatch.setattr(cfg, "get_urgency_levels", lambda: [])
    monkeypatch.setattr(cfg, "get_urgency_expressions", lambda: {})
    from src.prompts import validate_expression_sections
    with pytest.raises(ValueError, match="raro"):
        validate_expression_sections()


def test_validate_expression_sections_fails_on_missing_urgency(monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "get_risks", lambda: [])
    monkeypatch.setattr(cfg, "get_risk_expressions", lambda: {})
    monkeypatch.setattr(cfg, "get_urgency_levels", lambda: ["alta", "baja"])
    monkeypatch.setattr(cfg, "get_urgency_expressions", lambda: {"alta": ["x"]})
    from src.prompts import validate_expression_sections
    with pytest.raises(ValueError, match="baja"):
        validate_expression_sections()
