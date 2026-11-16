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
from .prompt_loader import render as _render_prompt


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


def _natural(s: str) -> str:
    """Render a config token as natural Spanish (snake_case → words)."""
    return s.replace("_", " ").strip()


def format_affected_elements(p: dict) -> str:
    """Render a problem's affected elements as a natural-language list."""
    elems = p.get("affected_elements") or []
    if not elems:
        return ""
    strs = [f"la {_natural(e['principal'])} ({_natural(e['subcategory'])})" for e in elems]
    return f"\n     Elementos dañados además: {', '.join(strs)}"


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
    history: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Build messages for generating raw complaint text.

    The LLM produces ONLY: user_text and timestamp. It does NOT produce
    image descriptions (those are the extractor's responsibility) nor
    structured extractions.

    Args:
        persona: Optional persona dict with 'profile' and 'style' keys.
        modifier: Optional style modifier dict with 'description' and 'style_override'.
        history: Optional list of (assistant_json, judge_comment) turns from previous
            rejected attempts. When given, the returned messages are a real
            multi-turn conversation: system, user(prompt), then one
            assistant(generation) + user(judge comment) pair per attempt.
    """
    # Describe problems in natural language (incl. affected elements + risks
    # so the generated text reflects the full ground-truth scenario)
    problem_descs = []
    for i, p in enumerate(problems, 1):
        desc = (f"  {i}. Problema: {_natural(p['subcategory'])} "
                f"(zona: {_natural(p['location_zone'])}")
        if p.get("location_detail"):
            desc += f", {_natural(p['location_detail'])}"
        desc += f"). Gravedad: {_natural(p['urgency'])}."
        desc += format_affected_elements(p)
        desc += format_risks(p)
        problem_descs.append(desc)
    problems_text = "\n".join(problem_descs)

    # Community description
    comm = community
    comm_desc = (
        f"{_natural(comm['name'])} — {_natural(comm['type'])} "
        f"de {comm['num_viviendas']} viviendas, "
        f"construida en {comm['year']}"
    )
    if comm["has_elevator"]:
        comm_desc += ", con ascensor"
    if comm["has_pool"]:
        comm_desc += ", con piscina"
    if comm["has_garage"]:
        comm_desc += ", con garaje"

    # Photo info (the vecino may mention photos, but the LLM does NOT describe
    # them and must NEVER cite internal filenames — only natural references)
    if num_pics > 0:
        plural = "" if num_pics == 1 else "s"
        photo_text = (
            f"El vecino adjunta {num_pics} foto{plural} del problema. "
            f"Puede referirse a ellas de forma natural ('adjunto foto', 'como se ve en la imagen', etc.), "
            f"pero NUNCA menciones nombres de archivo, identificadores internos ni describas su contenido."
        )
    else:
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

    if persona or modifier:
        persona_lines = []
        if persona:
            persona_lines.append(f"Perfil: {persona['profile']}")
            persona_lines.append(f"Estilo base: {persona['style']}")
        if modifier:
            persona_lines.append(f"Modificador de estilo: {modifier['description']}")
            persona_lines.append(f"Estilo adicional: {modifier['style_override']}")
        system_msg = _render_prompt("generate_complaint", "system_persona",
                                    persona_lines="\n".join(persona_lines))
    else:
        system_msg = _render_prompt("generate_complaint", "system")

    # Urgency expression rules (from config) rendered as prompt lines
    urgency_expr = cfg.get_urgency_expressions()
    urgency_lines = [
        f"     * {level}: debe incluir alguna de " +
        ", ".join(f'"{e}"' for e in exprs)
        for level, exprs in urgency_expr.items()
    ]
    urgency_block = "\n".join(urgency_lines)

    # Worked examples for every field (single source: prompts/examples.md).
    # The forbidden urgency expressions are derived from the config, not written
    # in the template: the levels assigned to this record are the only allowed ones.
    urgency_examples = "\n".join(
        f"  * {level}: " + ", ".join(f"'{e}'" for e in exprs)
        for level, exprs in urgency_expr.items()
    )
    assigned_levels = {str(p.get("urgency")) for p in problems if p.get("urgency")}
    prohibited = [
        f"'{e}' (nivel {level})"
        for level, exprs in urgency_expr.items()
        for e in exprs
        if level not in assigned_levels
    ]
    prohibited_block = "; ".join(prohibited) if prohibited else "(ninguna: este registro usa todas las expresiones pedidas)"
    examples_block = _render_prompt(
        "examples", "generation",
        urgency_examples=urgency_examples,
        prohibited_block=prohibited_block,
    )

    user_msg = _render_prompt(
        "generate_complaint", "user",
        problems_text=problems_text,
        comm_desc=comm_desc,
        persona_block=persona_block,
        photo_text=photo_text,
        urgency_block=urgency_block,
        examples_block=examples_block,
        json_template=json.dumps(json_template, indent=2, ensure_ascii=False),
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    for assistant_json, judge_comment in history or []:
        messages.append({"role": "assistant", "content": assistant_json})
        messages.append({"role": "user", "content": judge_comment})
    return messages


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

        prompt = _render_prompt("image_prompt",
                                cat_en=cat_en, sub=sub, zone_en=zone_en,
                                detail_clause=detail_clause, base=base,
                                elems_en=elems_en, urg_en=urg_en)
        prompts.append(prompt)

    return prompts


def build_validation_prompt(user_text: str, problems: list[dict]) -> list[dict]:
    """Judge prompt (prompts/validate_record.md): per-field verdicts with evidence.

    The judge is the only semantic gate of the corpus, so it must extract rather
    than opine: which urgency expression appears, which scenario elements are
    named, which elements/risks the text adds. The rules are applied in code
    (see validate_complaint), not left to the judge's discretion.
    """
    problem_descs = []
    for i, p in enumerate(problems, 1):
        desc = (f"  {i}. Problema: {p['subcategory']} "
                f"(categoría: {p['category']}; zona: {p['location_zone']}")
        if p.get("location_detail"):
            desc += f", {p['location_detail']}"
        desc += f"; gravedad: {p['urgency']})"
        if p.get("affected_elements"):
            elem_strs = [f"{e['principal']}/{e['subcategory']}" for e in p["affected_elements"]]
            desc += f"\n     Elementos afectados: {', '.join(elem_strs)}"
        desc += f"\n     Riesgos: {', '.join(p['risks']) if p.get('risks') else 'ninguno'}"
        problem_descs.append(desc)
    problems_text = "\n".join(problem_descs)

    urg_expr = cfg.get_urgency_expressions()
    urg_lines = "\n".join(
        f"  - {level}: {', '.join(f'{chr(34)}{e}{chr(34)}' for e in exprs)}"
        for level, exprs in urg_expr.items()
    )

    return [
        {"role": "system", "content": _render_prompt("validate_record", "system")},
        {"role": "user", "content": _render_prompt(
            "validate_record", "user",
            n_problems=len(problems), problems_text=problems_text,
            user_text=user_text, urg_lines=urg_lines,
            examples_block=_render_prompt("examples", "judge"))},
    ]
