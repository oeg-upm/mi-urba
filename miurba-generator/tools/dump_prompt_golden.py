#!/usr/bin/env python3
"""Dump rendered prompts to a golden fixture (byte-exact regression baseline).

Run from miurba-generator/:  python tools/dump_prompt_golden.py [out.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import build_generation_prompt, build_validation_prompt  # noqa: E402

COMMUNITY = {"name": "Urbanización El Roble", "type": "urbanizacion", "num_viviendas": 24,
             "year": 1998, "has_elevator": True, "has_pool": True, "has_garage": False,
             "id": "C001"}

P1 = {"category": "fontanería", "subcategory": "fugas", "location_zone": "portal",
      "location_detail": "hall de entrada", "urgency": "media",
      "affected_elements": [{"principal": "tubería", "subcategory": "vertical"}],
      "risks": ["ninguno"]}

P2 = {"category": "cubierta", "subcategory": "goteras", "location_zone": "exterior",
      "location_detail": "cubierta", "urgency": "alta",
      "affected_elements": [{"principal": "cubierta", "subcategory": "teja"},
                            {"principal": "muro", "subcategory": "cortafuegos"}],
      "risks": ["estructural", "inundación"]}

PERSONA = {"profile": "Inquilino transeúnte", "style": "mensajes cortos y directos"}
MODIFIER = {"description": "Escribe con prisa", "style_override": "frases telegráficas"}

out = {
    "generation_simple": build_generation_prompt("Q0001", [P1], COMMUNITY),
    "generation_multi": build_generation_prompt("Q0002", [P1, P2], COMMUNITY,
                                                persona=PERSONA, modifier=MODIFIER),
    "validation": build_validation_prompt("Texto de prueba de la queja.", [P1, P2]),
}

path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/fixtures/prompt_golden.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"golden escrito en {path} ({len(out)} prompts)")
