#!/usr/bin/env python3
"""Dump rendered extractor prompts to a golden fixture (byte-exact baseline).

Run from miurba-extractor/:  python tools/dump_prompt_golden.py [out.json]
"""
import argparse
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", nargs="?", help="output path (default: tests/fixtures/prompt_golden.json)")
args = parser.parse_args()

path = Path(args.out) if args.out else ROOT / "tests/fixtures/prompt_golden.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"golden escrito en {path} ({len(out)} prompts)")
