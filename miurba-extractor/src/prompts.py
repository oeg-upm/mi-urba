"""
Prompt templates for structured EXTRACTION with a VLM.

This module belongs to the extractor project. The extractor receives a raw
complaint (user_text) plus the ACTUAL PHOTO FILES and uses a vision-language
model (VLM) to:

  1. describe each photo (image_descriptions), and
  2. extract the structured information (extractions).

Both are produced in a single multimodal call, conforming to
schemas/extraction_schema.json at the repository root.

The VLM sees the images directly (OpenAI-compatible multimodal messages).
Describing the images is the extractor's responsibility — the generator no
longer produces image descriptions.
"""

import json

from . import config as cfg


def _build_text_part(complaint: dict) -> str:
    """Build the text portion of the extraction prompt."""
    context_parts = [f"TEXTO DEL VECINO:\n{complaint['user_text']}"]

    if complaint.get("user_pics"):
        n = len(complaint["user_pics"])
        context_parts.append(f"FOTOS ADJUNTAS: {n} foto(s) que se muestran a continuación.")

    input_text = "\n\n".join(context_parts)

    # Extraction schema — now includes image_descriptions
    extraction_schema = {
        "complaint_id": complaint["id"],
        "image_descriptions": [
            "<descripción detallada y objetiva de lo que se ve en la foto 1>"
        ],
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

    system_msg = (
        "Eres un experto en mantenimiento de comunidades de vecinos con capacidad "
        "de visión. Recibes el texto que envió el vecino y, si hay fotos, las propias "
        "imágenes. Tu tarea tiene dos partes:\n"
        "1. Describir CADA foto de forma detallada y objetiva (image_descriptions): "
        "tipo de daño, materiales, colores, iluminación, composición, entorno, escala, "
        "texturas. Indica también elementos relevantes que el vecino NO haya mencionado "
        "en su texto, marcándolos con la frase 'no mencionado por el vecino'.\n"
        "2. Extraer información estructurada (extracciones): identificar TODOS los "
        "problemas mencionados o visibles (puede haber más de uno) y, para cada uno, "
        "extraer categoría, elementos afectados, ubicación, riesgos y urgencia. "
        "Considera tanto el texto como lo que ves en las fotos.\n\n"
        "RESPONDE SOLO con JSON válido, sin markdown ni texto adicional."
    )

    user_text_part = (
        f"{input_text}\n\n"
        f"---\n\n"
        f"Describe las fotos y extrae la información estructurada de la queja.\n\n"
        f"CATEGORÍAS VÁLIDAS (principal → subcategorías):\n"
        + "\n".join(cat_lines) + "\n\n"
        f"ZONAS VÁLIDAS: {', '.join(zones)}\n"
        f"RIESGOS VÁLIDOS: {', '.join(risks)}\n"
        f"URGENCIAS VÁLIDAS: {', '.join(urgencies)}\n\n"
        f"IMPORTANTE: image_descriptions debe tener una entrada por cada foto adjunta "
        f"(en el mismo orden). Si no hay fotos, deja image_descriptions como [].\n\n"
        f"RESPONDE SOLO con JSON en este formato:\n"
        f"{json.dumps(extraction_schema, indent=2, ensure_ascii=False)}"
    )

    return system_msg, user_text_part


def build_extraction_prompt(
    complaint: dict,
    image_paths: list[str],
) -> list[dict]:
    """
    Build MULTIMODAL messages for extracting structured data from a complaint.

    The VLM receives the user_text and the actual photo files (as base64 data
    URLs in OpenAI-compatible image_url parts).

    Args:
        complaint: dict with at least id, user_text, user_pics.
        image_paths: list of paths to the photo files (one per user_pic, in
            order). Missing files are skipped (the model is told a photo was
            expected but not provided).

    Returns:
        A list of messages: [{system}, {user}] where the user message content
        is a list of text and image_url parts.
    """
    import base64
    import mimetypes
    from pathlib import Path

    system_msg, user_text_part = _build_text_part(complaint)

    content: list[dict] = [{"type": "text", "text": user_text_part}]

    for i, p in enumerate(image_paths, 1):
        path = Path(p)
        if not path.exists():
            content.append({
                "type": "text",
                "text": f"[Foto {i} no disponible: {path.name}]",
            })
            continue
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content},
    ]

# ── Stage-split prompts (P2: two-call pipeline) ─────────────────


def _image_b64_part(path: str, index: int) -> dict:
    """Build an image_url content part, or a note when the file is missing."""
    import base64
    import mimetypes
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {"type": "text", "text": f"[Foto {index} no disponible: {p.name}]"}
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def build_image_description_prompt(
    complaint: dict,
    image_paths: list[str],
) -> list[dict]:
    """
    Stage 1 (MULTIMODAL): describe each photo into structured entries.

    Output conforms to schemas/image_description_schema.json:
    per-photo {caption, objects, damage_indicators, not_in_text}.
    No taxonomy, no extraction — pure visual description.
    """
    system_msg = (
        "Eres un descriptor objetivo de imágenes. Recibes las fotos que adjuntó "
        "un vecino junto a su queja. Para CADA foto produce una entrada con:\n"
        "- caption: descripción factual en una frase.\n"
        "- objects: objetos relevantes visibles.\n"
        "- damage_indicators: señales visuales de daño o deterioro.\n"
        "- not_in_text: true si la foto muestra algo que el texto NO menciona.\n\n"
        "RESPONDE SOLO con JSON válido, sin markdown ni texto adicional."
    )

    n = len(image_paths)
    content: list[dict] = [{
        "type": "text",
        "text": (
            f"TEXTO DEL VECINO:\n{complaint['user_text']}\n\n"
            f"FOTOS ADJUNTAS: {n} foto(s), a continuación en orden "
            f"(foto 1 = user_pics[0], etc.).\n\n"
            f"Describe cada foto. RESPONDE SOLO con JSON en este formato:\n"
            f"{json.dumps({'complaint_id': complaint['id'], 'descriptions': [{'caption': '...', 'objects': ['...'], 'damage_indicators': ['...'], 'not_in_text': False}]}, indent=2, ensure_ascii=False)}"
        ),
    }]
    for i, p in enumerate(image_paths, 1):
        content.append(_image_b64_part(p, i))

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content},
    ]


def build_extraction_text_prompt(
    complaint: dict,
    descriptions: list[dict],
) -> list[dict]:
    """
    Stage 2 (TEXT-ONLY): extract structure from user_text + stage-1
    descriptions. Output conforms to schemas/extraction_schema.json.
    """
    system_msg = (
        "Eres un experto en mantenimiento de comunidades de vecinos. Recibes el "
        "texto de una queja y, si las hay, descripciones objetivas de las fotos "
        "adjuntas (producidas por un modelo de visión). Extrae la información "
        "estructurada: TODOS los problemas mencionados o visibles, con categoría, "
        "elementos afectados, ubicación, riesgos y urgencia.\n\n"
        "RESPONDE SOLO con JSON válido, sin markdown ni texto adicional."
    )

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

    if descriptions:
        desc_lines = "\n".join(
            f"  Foto {i}: {d.get('caption', '')}"
            + (f" [objetos: {', '.join(d['objects'])}]" if d.get("objects") else "")
            + (f" [daño: {', '.join(d['damage_indicators'])}]" if d.get("damage_indicators") else "")
            + (" [NO mencionado en el texto]" if d.get("not_in_text") else "")
            for i, d in enumerate(descriptions, 1)
        )
        desc_block = f"DESCRIPCIONES DE LAS FOTOS (etapa de visión):\n{desc_lines}\n\n"
    else:
        desc_block = "DESCRIPCIONES DE LAS FOTOS: (sin fotos o no disponibles)\n\n"

    extraction_template = {
        "complaint_id": complaint["id"],
        "image_descriptions": ["<caption breve de cada foto, en orden>"],
        "extractions": [
            {
                "category": {"principal": "...", "subcategory": "..."},
                "affected_elements": [{"principal": "...", "subcategory": "..."}],
                "location": {"zone": "...", "detail": "..."},
                "risks": ["..."],
                "urgency": "inmediata|urgente|alta|media|baja|planificable",
            }
        ],
    }

    user_text = (
        f"TEXTO DEL VECINO:\n{complaint['user_text']}\n\n"
        f"{desc_block}"
        f"---\n\n"
        f"Extrae la información estructurada de la queja.\n\n"
        f"CATEGORÍAS VÁLIDAS (principal → subcategorías):\n"
        + "\n".join(cat_lines) + "\n\n"
        f"ZONAS VÁLIDAS: {', '.join(zones)}\n"
        f"RIESGOS VÁLIDOS: {', '.join(risks)}\n"
        f"URGENCIAS VÁLIDAS: {', '.join(urgencies)}\n\n"
        f"IMPORTANTE: image_descriptions = captions breves de cada foto (mismo "
        f"orden); [] si no hay fotos.\n\n"
        f"RESPONDE SOLO con JSON en este formato:\n"
        f"{json.dumps(extraction_template, indent=2, ensure_ascii=False)}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text},
    ]
