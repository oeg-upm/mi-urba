"""
Prompt templates for structured EXTRACTION with an LLM.

This module belongs to the extractor project. The extractor receives a raw
complaint (user_text) and extracts the structured information (extractions)
conforming to schemas/extraction_schema.json at the repository root.
"""

import json

from . import config as cfg
from .prompt_loader import render as _render


def _build_text_part(complaint: dict) -> str:
    """Build the text portion of the extraction prompt."""
    input_text = f"TEXTO DEL VECINO:\n{complaint['user_text']}"

    extraction_schema = {
        "complaint_id": complaint["id"],
        "extractions": [
            {
                "category": {"principal": "...", "subcategory": "..."},
                "affected_elements": [
                    {"principal": "...", "subcategory": "..."}
                ],
                "location": {"zone": "...", "detail": "..."},
                "risks": ["..."],
                "urgency": "inmediata|urgente|alta|media|baja|planificable",
            }
        ],
    }

    # Valid values reference — read from config
    categories = cfg.get_categories()
    zones = list(cfg.get_location_zones().keys())
    risks = cfg.get_risks()
    urgencies = cfg.get_urgency_levels()

    cat_lines = []
    for cat, subs in categories.items():
        if subs:
            cat_lines.append(f"- {cat}: {', '.join(subs)}")
        else:
            cat_lines.append(f"- {cat}: (cualquier otro problema)")

    system_msg = _render("extract_complaint", "system")
    user_text_part = _render(
        "extract_complaint", "user",
        input_text=input_text,
        cat_lines="\n".join(cat_lines),
        zones=", ".join(zones),
        risks=", ".join(risks),
        urgencies=", ".join(urgencies),
        schema_json=json.dumps(extraction_schema, indent=2, ensure_ascii=False),
    )

    return system_msg, user_text_part


def build_extraction_prompt(complaint: dict) -> list[dict]:
    """
    Build messages for extracting structured data from a complaint (text-only).

    Args:
        complaint: dict with at least id, user_text.

    Returns:
        A list of messages: [{system}, {user}].
    """
    system_msg, user_text_part = _build_text_part(complaint)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text_part},
    ]
