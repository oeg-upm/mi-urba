#!/usr/bin/env python3
"""Dump rendered extractor prompts to a golden fixture (byte-exact baseline).

Run from miurba-extractor/:  python tools/dump_prompt_golden.py [out.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import (build_extraction_prompt, build_extraction_text_prompt,  # noqa: E402
                         build_image_description_prompt)

COMPLAINT = {"id": "Q0001",
             "user_text": "En el portal hay una fuga de agua en la tubería vertical.",
             "user_pics": []}
COMPLAINT_PICS = {"id": "Q0002",
                  "user_text": "Como se ve en la foto, la fachada tiene grietas.",
                  "user_pics": ["Q0002_1.png"]}
DESCRIPTIONS = [{"caption": "Grietas en el revoco de la fachada",
                 "objects": ["fachada", "revoco"], "damage_indicators": ["grietas"],
                 "not_in_text": False},
                {"caption": "Detalle del canalón", "objects": [],
                 "damage_indicators": [], "not_in_text": True}]

IMG = ROOT / ".tmp" / "pixel.png"
IMG.parent.mkdir(parents=True, exist_ok=True)
if not IMG.exists():
    IMG.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082"))

out = {
    "extraction_text_only": build_extraction_prompt(COMPLAINT, []),
    "extraction_with_image": build_extraction_prompt(COMPLAINT_PICS, [str(IMG)]),
    "describe_image": build_image_description_prompt(COMPLAINT_PICS, [str(IMG)]),
    "describe_image_missing_file": build_image_description_prompt(COMPLAINT_PICS, ["/no/existe.png"]),
    "extract_text_stage2": build_extraction_text_prompt(COMPLAINT_PICS, DESCRIPTIONS),
    "extract_text_stage2_no_photos": build_extraction_text_prompt(COMPLAINT, []),
}

# Las rutas absolutas del tmp no deben quedar en el golden (dependen del entorno)
out["describe_image_missing_file"] = [
    {**m, "content": [{**p, "text": p.get("text", "").replace("/no/existe.png", "<missing>")}
                      if isinstance(p, dict) else p for p in m["content"]]}
    if isinstance(m["content"], list) else m
    for m in out["describe_image_missing_file"]
]

path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/fixtures/prompt_golden.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"golden escrito en {path} ({len(out)} prompts)")
