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


# ── Generation prompts (text LLM) ───────────────────────────────────


def format_affected_elements(p: dict) -> str:
    """Render a problem's affected elements as 'principal (subcategory)' list."""
    elems = p.get("affected_elements") or []
    if not elems:
        return ""
    strs = [f"{e['principal']} ({e['subcategory']})" for e in elems]
    return f"\n     Elementos afectados: {', '.join(strs)}"


def validate_expression_sections() -> None:
    """Fail fast when risk_expressions/urgency_expressions are empty.

    Without them the strict generation prompt loses its mandatory
    risk/urgency phrases and the ground truth degrades silently.
    """
    risk_expr = cfg.get_risk_expressions()
    urg_expr = cfg.get_urgency_expressions()
    risks = [r for r in cfg.get_risks() if r != "ninguno"]
    missing_risk = [r for r in risks if r not in risk_expr]
    missing_urg = [u for u in cfg.get_urgency_levels() if u not in urg_expr]
    if missing_risk:
        raise ValueError(
            "config.yaml: risk_expressions is missing entries for: "
            + ", ".join(missing_risk)
        )
    if missing_urg:
        raise ValueError(
            "config.yaml: urgency_expressions is missing entries for: "
            + ", ".join(missing_urg)
        )


def format_risks(p: dict, risk_expr: dict[str, str] | None = None) -> str:
    """Render a problem's risks with their natural-language expressions.

    Risks mapped in `risk_expr` use their expression; unknown risks pass
    through verbatim. 'ninguno' (or an empty list) renders nothing.
    """
    risk_expr = risk_expr if risk_expr is not None else cfg.get_risk_expressions()
    risks = p.get("risks") or []
    if not risks or risks == ["ninguno"]:
        return ""
    strs = [risk_expr.get(r, r) for r in risks]
    return f"\n     Riesgos: {', '.join(strs)} (el texto debe mencionarlos)"


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
    # Describe problems in natural language (incl. affected elements + risks
    # so the generated text reflects the full ground-truth scenario)
    problem_descs = []
    for i, p in enumerate(problems, 1):
        desc = f"  {i}. {p['category']}/{p['subcategory']} en {p['location_zone']}"
        if p.get("location_detail"):
            desc += f", {p['location_detail']}"
        desc += f" (urgencia: {p['urgency']})"
        desc += format_affected_elements(p)
        desc += format_risks(p)
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

    # Urgency expression rules (from config) rendered as prompt lines
    urgency_expr = cfg.get_urgency_expressions()
    urgency_lines = [
        f"     * {level}: debe incluir alguna de " +
        ", ".join(f'"{e}"' for e in exprs)
        for level, exprs in urgency_expr.items()
    ]
    urgency_block = "\n".join(urgency_lines)

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
        f"a través de una app de gestión de comunidades.\n\n"
        f"CONTENIDO ESTRICTO (debe aparecer en el texto, sin excepción):\n"
        f"   - Categoría y subcategoría: el problema descrito debe corresponder "
        f"EXACTAMENTE con la categoría y subcategoría indicadas. No describas un "
        f"problema distinto.\n"
        f"   - Subcategoría reconocible: usa la palabra de la subcategoría o un "
        f"sinónimo claro que un lector pueda identificar (p. ej. 'cristales' → "
        f"'cristales' o 'vidrios'; 'tejas' → 'tejas' o 'tejado').\n"
        f"   - Elementos afectados: menciona EXPLÍCITAMENTE cada elemento indicado, "
        f"en lenguaje natural (un vecino no diría 'instalación_eléctrica' pero sí "
        f"'la instalación eléctrica' o 'los cables'). Los elementos afectados son "
        f"CONSECUENCIAS o PARTES del problema principal, NUNCA averías "
        f"independientes: usa frases como 'además se ha afectado el canalón' o "
        f"'también está dañado el muro cortafuegos'.\n"
        f"   - Ubicación: la zona y el detalle deben coincidir con lo indicado.\n"
        f"   - Urgencia: el tono debe reflejar el nivel asignado e incluir "
        f"OBLIGATORIAMENTE alguna de las expresiones del nivel correspondiente:\n"
        f"{urgency_block}\n"
        f"   - Riesgos: si hay riesgos indicados (no 'ninguno'), el texto debe "
        f"mencionarlos con la expresión indicada (p. ej. 'riesgo eléctrico', "
        f"'peligro de caída').\n\n"
        f"CONTENIDO PROHIBIDO:\n"
        f"   - NO describas problemas, averías o incidencias que NO estén en la "
        f"lista de problemas indicados arriba.\n"
        f"   - Las menciones a elementos afectados deben enmarcarse como síntomas "
        f"del problema, no como averías separadas.\n"
        f"   - Si el vecino menciona algo colateral ('por cierto', 'también'), debe "
        f"ser contexto, no una nueva incidencia.\n\n"
        f"ESTRUCTURA GUIADA (flexible, no una plantilla rígida):\n"
        f"   - Saludo inicial + ubicación: empieza mencionando DÓNDE está el "
        f"problema (zona + detalle).\n"
        f"   - Descripción del problema: nombra la subcategoría con palabras "
        f"reconocibles al introducirlo.\n"
        f"   - Elementos afectados: después del problema principal, como "
        f"consecuencias ('además...', 'también se ha...').\n"
        f"   - Urgencia al cierre: cierra con la expresión del nivel asignado.\n\n"
        f"ESTILO FLEXIBLE (libertad creativa):\n"
        f"   - Tono y extensión: varía. A veces enfadado, a veces casual, a veces "
        f"muy específico, a veces vago. Depende de la personalidad y del "
        f"modificador de estilo.\n"
        f"   - Contexto adicional: el vecino puede añadir detalles sobre su vida, "
        f"la comunidad o el historial del problema.\n"
        f"   - Si el vecino se identifica, inventa un nombre y apellidos realistas "
        f"en español. NUNCA uses placeholders como '[Nombre del vecino]', "
        f"'[sin especificar]' o similares.\n"
        f"   - Escribe EXCLUSIVAMENTE en español. No mezcles palabras sueltas de "
        f"otros idiomas. Revisa la ortografía antes de finalizar.\n"
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