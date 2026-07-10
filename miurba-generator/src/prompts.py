"""
Prompt templates for synthetic complaint GENERATION.

This module belongs to the generator project. It contains:

* `build_generation_prompt`  — messages for the LLM that produces the raw
  complaint text (user_text + timestamp). It does NOT produce image
  descriptions: describing images is the extractor's job.

* `build_image_prompts`      — English text-to-image prompts derived from the
  sampled scenario, used by the (optional) image-generation step to produce
  the actual photo files. The prompts are built from the scenario fields
  (category, location, affected elements, urgency, community), never from a
  textual description of the image.
"""

import json

from . import config as cfg


# ── English maps for image prompts ──────────────────────────────

_CATEGORY_EN: dict[str, str] = {
    "fontanería": "plumbing", "electricidad": "electrical", "cubierta": "roof",
    "fachada": "facade", "carpintería": "window and door carpentry",
    "jardinería": "gardening and landscaping", "limpieza": "cleaning and waste",
    "ascensor": "elevator", "climatización": "HVAC and heating",
    "piscina": "swimming pool", "garaje": "garage", "plagas": "pest infestation",
    "mobiliario": "communal furniture", "seguridad": "security",
    "ruido": "noise nuisance", "varios": "maintenance",
}

_ZONE_EN: dict[str, str] = {
    "garaje": "underground garage", "portal": "building entrance lobby",
    "exterior": "exterior facade", "zonas_verdes": "communal garden areas",
    "piscina": "pool area", "trasteros": "storage room",
    "azotea": "rooftop", "locales": "commercial premises",
}

_URGENCY_EN: dict[str, str] = {
    "inmediata": "severe and urgent", "urgente": "clearly urgent",
    "alta": "notable", "media": "moderate", "baja": "minor",
    "planificable": "early-stage preventive",
}

_ELEMENT_EN: dict[str, str] = {
    "tubería": "pipes", "instalación_sanitaria": "plumbing fixtures",
    "revestimiento": "plaster and wall coating", "suelo": "flooring",
    "instalación_eléctrica": "electrical wiring", "equipamiento": "equipment",
    "cubierta": "roof surface", "muro": "wall", "ventana": "window",
    "puerta": "door", "jardín": "garden", "cerramiento": "fence and enclosure",
    "mobiliario_común": "common furniture", "estructura": "structural element",
}


def _en(map_: dict, key: str) -> str:
    return map_.get(key, key.replace("_", " "))


# ── Generation prompts (text LLM) ───────────────────────────────

def build_generation_prompt(
    record_id: str,
    problems: list[dict],
    community: dict,
    num_pics: int,
    persona: dict | None = None,
    modifier: dict | None = None,
) -> list[dict[str, str]]:
    """
    Build messages for generating raw complaint text.

    The LLM produces ONLY: user_text and timestamp. It does NOT produce
    image descriptions (those are the extractor's responsibility) nor
    structured extractions.

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

    # Photo info (the vecino may mention photos, but the LLM does NOT describe them)
    if num_pics > 0:
        pic_names = [f"{record_id}_{i}.png" for i in range(1, num_pics + 1)]
        photo_text = (
            f"El vecino adjunta {num_pics} foto(s): {', '.join(pic_names)}. "
            f"Puede referirse a ellas en el texto (\"adjunto foto\", \"como se ve en la imagen\", etc.), "
            f"pero NO tienes que describir lo que se ve en las fotos."
        )
    else:
        pic_names = []
        photo_text = "El vecino no adjunta fotos."

    # JSON template — only user_text and timestamp
    json_template = {
        "user_text": "<lo que escribe/dice el vecino>",
        "timestamp": "<ISO8601 realista>",
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
        f"{persona_block}"
        f"- {photo_text}\n"
        f"- ID: {record_id}\n\n"
        f"INSTRUCCIONES:\n"
        f"1. Escribe el user_text como lo expresaría este vecino REAL "
        f"a través de una app de gestión de comunidades. Sé creativo: varía tono, "
        f"extensión, nivel de detalle. A veces enfadado, a veces casual, "
        f"a veces muy específico, a veces vago."
        f" Si el vecino se identifica, inventa un nombre y apellidos "
        f"realistas en español. NUNCA uses placeholders como "
        f"[Nombre del vecino], [sin especificar] o similares."
        f" Escribe EXCLUSIVAMENTE en español."
        f" No mezcles palabras sueltas de otros idiomas"
        f" (como 'many', 'this', etc.) dentro del texto en español."
        f" Revisa la ortografía antes de finalizar.\n"
        f"2. El timestamp debe ser realista.\n\n"
        f"RESPONDE SOLO con JSON en este formato:\n"
        f"{json.dumps(json_template, indent=2, ensure_ascii=False)}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ── Image prompts (text-to-image, derived from the scenario) ────

def build_image_prompts(
    problems: list[dict],
    community: dict,
    num_pics: int,
) -> list[str]:
    """
    Build English text-to-image prompts, one per photo.

    Prompts are derived from the scenario fields (category, subcategory,
    location, affected elements, urgency, community) — NOT from any textual
    description of the image. Each photo focuses on one of the sampled
    problems (cycling through them when there are more photos than problems).
    """
    if num_pics <= 0:
        return []

    comm_type_en = {
        "bloque_pisos": "apartment block", "casa_pareada": "paired house",
        "urbanizacion": "residential complex",
    }.get(community.get("type", ""), "residential building")

    base = (
        f"a {comm_type_en} with {community.get('num_viviendas', 20)} apartments, "
        f"built around {community.get('year', 2000)}"
    )

    prompts = []
    for i in range(num_pics):
        p = problems[i % len(problems)]
        cat_en = _en(_CATEGORY_EN, p["category"])
        zone_en = _en(_ZONE_EN, p.get("location_zone", ""))
        urg_en = _en(_URGENCY_EN, p.get("urgency", "media"))

        elements = p.get("affected_elements", [])
        if elements:
            elems_en = ", ".join(_en(_ELEMENT_EN, e["principal"]) for e in elements)
        else:
            elems_en = "building fixtures"

        detail = p.get("location_detail", "")
        detail_clause = f", specific area: {detail}" if detail else ""

        sub = p.get("subcategory", "general").replace("_", " ")

        prompt = (
            f"Documentary inspection photograph of {cat_en} damage "
            f"({sub}) in the {zone_en}{detail_clause} of {base}. "
            f"Visible affected elements: {elems_en}. "
            f"The damage looks {urg_en}. "
            f"Realistic, professional maintenance inspection photo, "
            f"natural lighting, sharp focus, no people, no text overlays."
        )
        prompts.append(prompt)

    return prompts