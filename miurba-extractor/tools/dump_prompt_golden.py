#!/usr/bin/env python3
"""Dump rendered extractor prompts to a golden fixture (byte-exact baseline).

Run from miurba-extractor/:  python tools/dump_prompt_golden.py [out.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import build_extraction_prompt  # noqa: E402

COMPLAINT = {"id": "Q0001",
             "user_text": "En el portal hay una fuga de agua en la tubería vertical."}

out = {
    "extraction_text_only": build_extraction_prompt(COMPLAINT),
}

path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/fixtures/prompt_golden.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"golden escrito en {path} ({len(out)} prompts)")
