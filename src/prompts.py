"""
Prompt templates for generation and annotation.

All prompts are defined here. No prompt text elsewhere.
"""

import json

from . import config as cfg


# ── Generation prompts ──────────────────────────────────────────

def build_generation_prompt(
    record_id: str,
    problems: list[dict],
    channel: str,
    community: dict,
    num_pics: int,
    persona: dict | None = None,
    modifier: dict | None = None,
) -> list[dict[str, str]]:
    """
    Build messages for generating raw complaint text.

    The LLM produces only: user_text, image_descriptions, timestamp.
    It does NOT produce structured extractions.

    Args:
        persona: Optional persona dict with 'profile' and 'style' keys.
        modifier: Optional style modifier dict with 'description' and 'style_override'.
    """
    # Describe problems in natural language
    problem_descs = []
    for i, p in enumerate(problems, 1):
        desc = f"  {i}. {p['category']}/{p['subcategory']} en {p['location_zone']}"
        if p.get("location_detail"):
            desc += f", {p['location_detail']}"
        desc += f" (urgencia: {p['urgency']})"
        problem_descs.append(desc)
    problems_text = "\n".join(problem_descs)

    # Channel style
    channels = cfg.get_channels()
    channel_style = channels[channel]["style"]

    # Community description
    comm = community
    comm_desc = (
        f"{comm['name']} — {comm['type']} "
        f"de {comm['num_viviendas']} viviendas, "
        f"construida en {comm['year']}"
    )
    if comm["has_elevator"]:
        comm_desc += ", con ascensor"
    if comm["has_pool"]:
        comm_desc += ", con piscina"
    if comm["has_garage"]:
        comm_desc += ", con garaje"

    # Photo info
    if num_pics > 0:
        pic_names = [f"{record_id}_{i}.jpg" for i in range(1, num_pics + 1)]
        photo_text = f"El vecino adjunta {num_pics} foto(s): {', '.join(pic_names)}"
    else:
        pic_names = []
        photo_text = "El vecino no adjunta fotos."

    # JSON template
    json_template = {
        "user_text": "<lo que escribe/dice el vecino>",
        "image_descriptions": [
            "<descripción detallada de lo que se ve en la foto 1>"
        ] if num_pics > 0 else [],
        "timestamp": "<ISO8601, realista para el canal>",
    }

    # Persona conditioning
    persona_block = ""
    if persona:
        persona_block += (
            f"- Perfil del vecino: {persona['profile']}\n"
            f"- Estilo de escritura: {persona['style']}\n"
        )
    if modifier:
        persona_block += (
            f"- MODIFICADOR DE ESTILO: {modifier['description']}\n"
            f"- Aplica también este estilo: {modifier['style_override']}\n"
        )

    system_msg = (
        "Eres un vecino de una comunidad de vecinos. Escribe lo que "
        "un vecino REAL escribiría para reportar un problema de mantenimiento. "
        "RESPONDE SOLO con JSON válido, sin markdown ni texto adicional."
    )
    if persona or modifier:
        parts = ["Adopta la siguiente personalidad y escribe como ese vecino:"]
        if persona:
            parts.append(f"Perfil: {persona['profile']}")
            parts.append(f"Estilo base: {persona['style']}")
        if modifier:
            parts.append(f"Modificador de estilo: {modifier['description']}")
            parts.append(f"Estilo adicional: {modifier['style_override']}")
        parts.append("")
        parts.append("Escribe lo que ESE vecino escribiría para reportar un problema "
                     "de mantenimiento en su comunidad. "
                     "RESPONDE SOLO con JSON válido, sin markdown ni texto adicional.")
        system_msg = "\n".join(parts)

    user_msg = (
        f"PROBLEMA(S) A REPORTAR:\n"
        f"{problems_text}\n\n"
        f"CONTEXTO:\n"
        f"- Comunidad: {comm_desc}\n"
        f"- Canal: {channel}\n"
        f"- Estilo del texto: {channel_style}\n"
        f"{persona_block}"
        f"- {photo_text}\n"
        f"- ID: {record_id}\n\n"
        f"INSTRUCCIONES:\n"
        f"1. Escribe el user_text como lo expresaría este vecino REAL "
        f"por ese canal. Sé creativo: varía tono, extensión, "
        f"nivel de detalle. A veces enfadado, a veces casual, "
        f"a veces muy específico, a veces vago.\n"
        f"2. Para cada foto, describe lo que se vería en la imagen "
        f"(tipo de daño, materiales, colores, gravedad aparente, "
        f"contexto del entorno). Como si un modelo VLM la describiera.\n"
        f"3. El timestamp debe ser realista para el canal "
        f"(WhatsApp a cualquier hora, email en horario laboral...).\n\n"
        f"RESPONDE SOLO con JSON en este formato:\n"
        f"{json.dumps(json_template, indent=2, ensure_ascii=False)}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ── Annotation prompts ──────────────────────────────────────────

def build_annotation_prompt(
    complaint: dict,
    image_descriptions: list[str],
) -> list[dict[str, str]]:
    """
    Build messages for annotating a raw complaint.

    The annotator receives ONLY the same input that the extractor
    would see in production: user_text and image_descriptions.
    """
    # Build context
    context_parts = []
    context_parts.append(f"TEXTO DEL VECINO:\n{complaint['user_text']}")

    if image_descriptions:
        descs = []
        for i, (pic, desc) in enumerate(
            zip(complaint.get("user_pics", []), image_descriptions)
        ):
            descs.append(f"Foto {i+1} ({pic}): {desc}")
        context_parts.append("DESCRIPCIÓN DE FOTOS:\n" + "\n\n".join(descs))

    context_parts.append(f"CANAL: {complaint['channel']}")

    if complaint.get("community_ref"):
        context_parts.append(f"COMUNIDAD: {complaint['community_ref']}")

    input_text = "\n\n".join(context_parts)

    # Extraction schema
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

    # Build category listing
    cat_lines = []
    for cat, subs in categories.items():
        if subs:
            cat_lines.append(f"- {cat}: {', '.join(subs)}")
        else:
            cat_lines.append(f"- {cat}: (cualquier otro problema)")

    system_msg = (
        "Eres un experto en mantenimiento de comunidades de vecinos. "
        "Tu tarea es EXTRAER información estructurada de quejas y sugerencias "
        "de los vecinos. Recibes el texto que envió el vecino y, si hay fotos, "
        "su descripción. Debes identificar TODOS los problemas mencionados "
        "(puede haber más de uno) y extraer para cada uno: "
        "categoría, elementos afectados, ubicación, riesgos y urgencia.\n\n"
        "RESPONDE SOLO con JSON válido, sin markdown ni texto adicional."
    )

    user_msg = (
        f"{input_text}\n\n"
        f"---\n\n"
        f"Extrae la información estructurada de la queja anterior.\n\n"
        f"CATEGORÍAS VÁLIDAS (principal → subcategorías):\n"
        + "\n".join(cat_lines) + "\n\n"
        f"ZONAS VÁLIDAS: {', '.join(zones)}\n"
        f"RIESGOS VÁLIDOS: {', '.join(risks)}\n"
        f"URGENCIAS VÁLIDAS: {', '.join(urgencies)}\n\n"
        f"RESPONDE SOLO con JSON en este formato:\n"
        f"{json.dumps(extraction_schema, indent=2, ensure_ascii=False)}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]