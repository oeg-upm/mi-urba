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
    affected_elements_cfg = cfg.get_affected_elements()

    cat_lines = []
    for cat, subs in categories.items():
        if subs:
            cat_lines.append(f"- {cat}: {', '.join(subs)}")
        else:
            cat_lines.append(f"- {cat}: (cualquier otro problema)")

    elem_lines = []
    for elem, subs in affected_elements_cfg.items():
        if subs:
            elem_lines.append(f"- {elem}: {', '.join(subs)}")
        else:
            elem_lines.append(f"- {elem}")

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
        "DISTINCIÓN CRÍTICA: PROBLEMA vs ELEMENTO AFECTADO\n"
        "Un PROBLEMA es la avería o incidencia principal (ej: 'fontanería/fugas', 'carpintería/ventanas').\n"
        "Un ELEMENTO AFECTADO es lo que sufre el daño como consecuencia del problema "
        "(ej: la puerta que no cierra por una fuga, el revestimiento manchado por humedad, "
        "el suelo mojado por un atasco). Los elementos afectados van DENTRO del problema "
        "como affected_elements, NO como extracciones separadas.\n\n"
        "ERRORES COMUNES A EVITAR:\n"
        "- Si el vecino dice 'las tejas están mal y el canalón no funciona', es UN problema "
        "de cubierta con dos elementos (teja + canalón), no dos problemas separados.\n"
        "- Si dice 'el cristal está roto y la puerta también', es UN problema de carpintería "
        "con elementos (ventana/cristal + puerta), no dos problemas.\n"
        "- Solo extrae como problema SEPARADO si el vecino describe una avería con causa "
        "distinta e independiente (ej: 'tengo una fuga de agua Y además el ascensor no funciona').\n"
        "- Una mención pasajera o contextual ('por cierto, la acera está rota') no es un "
        "problema a menos que el vecino le dedique descripción y gravedad.\n\n"
        "GUÍA DE MAPEO TEXTO → RIESGOS:\n"
        "- 'peligro de caída', 'puede caerse', 'se va a caer' → riesgo 'caída'\n"
        "- 'riesgo eléctrico', 'puede dar un calambrazo', 'puede cortocircuitar' → riesgo 'eléctrico'\n"
        "- 'riesgo de inundación', 'se va a inundar', 'el agua sube' → riesgo 'inundación'\n"
        "- 'riesgo de derrumbe', 'se va a derrumbar', 'puede colapsar' → riesgo 'derrumbe'\n"
        "- 'riesgo de incendio', 'puede incendiarse', 'peligro de fuego' → riesgo 'incendio'\n"
        "- 'riesgo sanitario', 'insalubre', 'puede provocar enfermedades' → riesgo 'sanitario'\n"
        "- 'riesgo de seguridad', 'inseguro', 'puede entrar alguien' → riesgo 'seguridad'\n"
        "- 'riesgo de contaminación', 'contaminado', 'tóxico' → riesgo 'contaminación'\n"
        "- Si el vecino no menciona ningún riesgo explícito ni implícito, devuelve ['ninguno'].\n"
        "- NO inventes riesgos que el texto no sugiera. Solo extrae riesgos claramente aludidos.\n\n"
        "GUÍA DE MAPEO TEXTO → SUBCATEGORÍA (busca coincidencia de token raíz, no solo sinónimos):\n"
        "- 'cristales rotos', 'el cristal', 'el vidrio' → subcategoría 'cristales' (no 'ventanas')\n"
        "- 'las tejas', 'el tejado', 'las tejas sueltas' → subcategoría 'tejas' (no 'cubierta')\n"
        "- 'el bajante', 'el tubo del desagüe', 'la bajante' → subcategoría 'bajante'\n"
        "- 'el ascensor no funciona' → subcategoría del ascensor (no 'instalación eléctrica')\n"
        "- Si el vecino usa sinónimos o variaciones, mapea SIEMPRE a la subcategoría más cercana de la taxonomía.\n\n"
        "PRINCIPIO DE UNA SOLA CAUSA PRINCIPAL:\n"
        "- Antes de extraer un segundo problema, verifica que tenga una causa INDEPENDIENTE.\n"
        "- Lo normal en una queja es UN problema. Solo extrae dos+ si el texto describe averías con causas claramente distintas.\n"
        "- La mención de elementos dañados ('además el parqué está abombado', 'también la pared tiene humedad') NO es un problema extra.\n\n"
        "GUÍA DE URGENCIA POR TONO DEL VECINO:\n"
        "- 'es una emergencia', 'peligro inminente', 'necesito que vengan ya mismo', 'ya', 'ahora' → inmediata\n"
        "- 'es urgente', 'no puede esperar', 'por favor vengan cuanto antes' → urgente\n"
        "- 'es importante', 'me preocupa', 'deberían revisarlo pronto' → alta\n"
        "- 'cuando puedan', 'no tiene prisa pero' → media\n"
        "- 'es una sugerencia', 'para cuando haya tiempo' → baja\n"
        "- 'mantenimiento preventivo', 'para revisar a largo plazo' → planificable\n"
        "RESPONDE SOLO con JSON válido, sin markdown ni texto adicional."
    )

    user_text_part = (
        f"{input_text}\n\n"
        f"---\n\n"
        f"Describe las fotos y extrae la información estructurada de la queja.\n\n"
        f"CATEGORÍAS VÁLIDAS (principal → subcategorías):\n"
        + "\n".join(cat_lines) + "\n\n"
        f"ELEMENTOS AFECTADOS VÁLIDOS (principal → subcategorías):\n"
        + "\n".join(elem_lines) + "\n\n"
        f"ZONAS VÁLIDAS: {', '.join(zones)}\n"
        f"RIESGOS VÁLIDOS: {', '.join(risks)}\n"
        f"URGENCIAS VÁLIDAS: {', '.join(urgencies)}\n\n"
        f"IMPORTANTE: Usa SOLO los valores de las listas anteriores para category, "
        f"affected_elements, location.zone, risks y urgency. No inventes valores "
        f"fuera de la taxonomía.\n"
        f"IMPORTANTE (FORMATO EXACTO): affected_elements SIEMPRE como lista de diccionarios "
        f"con las claves 'principal' y 'subcategory'. Ejemplo válido: "
        f"[{{\"principal\": \"carpintería\", \"subcategory\": \"ventanas\"}}]. "
        f"NUNCA como lista de strings (ej: [\"carpintería/ventanas\"]), ni como string suelto, "
        f"ni omitas las claves. Cada elemento va en un diccionario propio.\n"
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

# ── Stage split (P2): two-call pipeline ────────────────────────

def _build_image_description_system() -> str:
    """System message for stage 1: VLM describes photos (no taxonomy)."""
    return (
        "Eres un experto en inspección de mantenimiento de edificios con capacidad "
        "de visión. Recibes fotos adjuntadas a una queja vecinal y el texto del vecino "
        "(como contexto). Tu tarea es describir CADA foto de forma detallada y objetiva, "
        "sin clasificar ni extraer estructura.\n\n"
        "Para cada foto describe:\n"
        "- caption: tipo de daño visible, materiales, colores, iluminación, composición, "
        "entorno, escala, texturas. Sé específico y objetivo.\n"
        "- objects: elementos relevantes visibles (tubería, cristal, muro, suelo, etc.)\n"
        "- damage_indicators: signos de daño, desgaste o problemas visibles\n"
        "- not_in_text: elementos visibles que el vecino NO mencionó en su texto "
        "(marca con frases claras lo que él omitió)\n\n"
        "NO clasifiques en categorías, NO asignes urgencia, NO mapees a taxonomía. "
        "Solo describe lo que ves.\n"
        "RESPONDE SOLO con JSON válido, sin markdown ni texto adicional."
    )


def build_image_description_prompt(
    complaint: dict,
    image_paths: list[str],
) -> list[dict]:
    """Stage 1 (P2): build MULTIMODAL messages for photo description.

    The VLM sees the photos + the user_text (as context only) and produces
    structured descriptions per photo, conforming to
    schemas/image_description_schema.json.
    """
    import base64
    import mimetypes
    from pathlib import Path

    system_msg = _build_image_description_system()
    schema_template = {
        "complaint_id": complaint["id"],
        "descriptions": [
            {"photo_index": 1, "caption": "...", "objects": ["..."],
             "damage_indicators": ["..."], "not_in_text": ["..."]}
        ],
    }
    user_text = (
        f"TEXTO DEL VECINO (contexto, no lo clasifiques):\n{complaint['user_text']}\n\n"
        f"---\n\n"
        f"Describe cada una de las {len(image_paths)} foto(s) adjuntada(s) a continuación. "
        f"Una entrada por foto, en orden.\n\n"
        f"RESPONDE SOLO con JSON en este formato:\n"
        f"{json.dumps(schema_template, indent=2, ensure_ascii=False)}"
    )

    content: list[dict] = [{"type": "text", "text": user_text}]
    for i, p in enumerate(image_paths, 1):
        path = Path(p)
        if not path.exists():
            content.append({"type": "text", "text": f"[Foto {i} no disponible: {path.name}]"})
            continue
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content},
    ]


def _build_extraction_text_system() -> str:
    """System message for stage 2: LLM extracts structure from text only."""
    return (
        "Eres un experto en mantenimiento de comunidades de vecinos. Recibes el texto "
        "que envió el vecino y, si hubo fotos, las descripciones que un inspector visual "
        "hizo de ellas. Tu tarea es extraer información estructurada: identificar TODOS los "
        "problemas mencionados (puede haber más de uno) y, para cada uno, extraer categoría, "
        "elementos afectados, ubicación, riesgos y urgencia.\n\n"
        "DISTINCIÓN CRÍTICA: PROBLEMA vs ELEMENTO AFECTADO\n"
        "Un PROBLEMA es la avería o incidencia principal (ej: 'fontanería/fugas', 'carpintería/ventanas').\n"
        "Un ELEMENTO AFECTADO es lo que sufre el daño como consecuencia del problema "
        "(ej: la puerta que no cierra por una fuga, el revestimiento manchado por humedad, "
        "el suelo mojado por un atasco). Los elementos afectados van DENTRO del problema "
        "como affected_elements, NO como extracciones separadas.\n\n"
        "ERRORES COMUNES A EVITAR:\n"
        "- Si el vecino dice 'las tejas están mal y el canalón no funciona', es UN problema "
        "de cubierta con dos elementos (teja + canalón), no dos problemas separados.\n"
        "- Si dice 'el cristal está roto y la puerta también', es UN problema de carpintería "
        "con elementos (ventana/cristal + puerta), no dos problemas.\n"
        "- Solo extrae como problema SEPARADO si el vecino describe una avería con causa "
        "distinta e independiente (ej: 'tengo una fuga de agua Y además el ascensor no funciona').\n"
        "- Una mención pasajera o contextual ('por cierto, la acera está rota') no es un "
        "problema a menos que el vecino le dedique descripción y gravedad.\n\n"
        "GUÍA DE MAPEO TEXTO → RIESGOS:\n"
        "- 'peligro de caída', 'puede caerse', 'se va a caer' → riesgo 'caída'\n"
        "- 'riesgo eléctrico', 'puede dar un calambrazo', 'puede cortocircuitar' → riesgo 'eléctrico'\n"
        "- 'riesgo de inundación', 'se va a inundar', 'el agua sube' → riesgo 'inundación'\n"
        "- 'riesgo de derrumbe', 'se va a derrumbar', 'puede colapsar' → riesgo 'derrumbe'\n"
        "- 'riesgo de incendio', 'puede incendiarse', 'peligro de fuego' → riesgo 'incendio'\n"
        "- 'riesgo sanitario', 'insalubre', 'puede provocar enfermedades' → riesgo 'sanitario'\n"
        "- 'riesgo de seguridad', 'inseguro', 'puede entrar alguien' → riesgo 'seguridad'\n"
        "- 'riesgo de contaminación', 'contaminado', 'tóxico' → riesgo 'contaminación'\n"
        "- Si el vecino no menciona ningún riesgo explícito ni implícito, devuelve ['ninguno'].\n"
        "- NO inventes riesgos que el texto no sugiera. Solo extrae riesgos claramente aludidos.\n\n"
        "GUÍA DE MAPEO TEXTO → SUBCATEGORÍA (busca coincidencia de token raíz, no solo sinónimos):\n"
        "- 'cristales rotos', 'el cristal', 'el vidrio' → subcategoría 'cristales' (no 'ventanas')\n"
        "- 'las tejas', 'el tejado', 'las tejas sueltas' → subcategoría 'tejas' (no 'cubierta')\n"
        "- 'el bajante', 'el tubo del desagüe', 'la bajante' → subcategoría 'bajante'\n"
        "- 'el ascensor no funciona' → subcategoría del ascensor (no 'instalación eléctrica')\n"
        "- Si el vecino usa sinónimos o variaciones, mapea SIEMPRE a la subcategoría más cercana de la taxonomía.\n\n"
        "PRINCIPIO DE UNA SOLA CAUSA PRINCIPAL:\n"
        "- Antes de extraer un segundo problema, verifica que tenga una causa INDEPENDIENTE.\n"
        "- Lo normal en una queja es UN problema. Solo extrae dos+ si el texto describe averías con causas claramente distintas.\n"
        "- La mención de elementos dañados ('además el parqué está abombado', 'también la pared tiene humedad') NO es un problema extra.\n\n"
        "GUÍA DE URGENCIA POR TONO DEL VECINO:\n"
        "- 'es una emergencia', 'peligro inminente', 'necesito que vengan ya mismo', 'ya', 'ahora' → inmediata\n"
        "- 'es urgente', 'no puede esperar', 'por favor vengan cuanto antes' → urgente\n"
        "- 'es importante', 'me preocupa', 'deberían revisarlo pronto' → alta\n"
        "- 'cuando puedan', 'no tiene prisa pero' → media\n"
        "- 'es una sugerencia', 'para cuando haya tiempo' → baja\n"
        "- 'mantenimiento preventivo', 'para revisar a largo plazo' → planificable\n"
        "RESPONDE SOLO con JSON válido, sin markdown ni texto adicional."
    )


def build_extraction_text_prompt(
    complaint: dict,
    image_descriptions: list[dict],
) -> list[dict]:
    """Stage 2 (P2): build TEXT-ONLY messages for structured extraction.

    The LLM receives the user_text + the photo descriptions from stage 1
    (or empty list if no photos) and produces {complaint_id, image_descriptions,
    extractions} conforming to schemas/extraction_schema.json.

    No image_url parts — pure text, allowing constrained decoding and a
    simpler prompt.
    """
    categories = cfg.get_categories()
    zones = list(cfg.get_location_zones().keys())
    risks = cfg.get_risks()
    urgencies = cfg.get_urgency_levels()
    affected_elements_cfg = cfg.get_affected_elements()

    cat_lines = []
    for cat, subs in categories.items():
        if subs:
            cat_lines.append(f"- {cat}: {', '.join(subs)}")
        else:
            cat_lines.append(f"- {cat}: (cualquier otro problema)")

    elem_lines = []
    for elem, subs in affected_elements_cfg.items():
        if subs:
            elem_lines.append(f"- {elem}: {', '.join(subs)}")
        else:
            elem_lines.append(f"- {elem}")

    # Include image descriptions as text context
    if image_descriptions:
        desc_text = "\n\n".join(
            f"[Foto {d.get('photo_index', i+1)}]\n"
            f"  Descripción: {d.get('caption', '')}\n"
            f"  Objetos: {', '.join(d.get('objects', []))}\n"
            f"  Indicadores de daño: {', '.join(d.get('damage_indicators', []))}\n"
            f"  No mencionado por el vecino: {', '.join(d.get('not_in_text', []))}"
            for i, d in enumerate(image_descriptions)
        )
        photos_section = f"DESCRIPCIONES DE LAS FOTOS (hechas por un inspector visual):\n{desc_text}\n\n"
    else:
        photos_section = ""

    system_msg = _build_extraction_text_system()

    extraction_schema = {
        "complaint_id": complaint["id"],
        "image_descriptions": ["string"],
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
        f"{photos_section}"
        f"---\n\n"
        f"Extrae la información estructurada de la queja.\n\n"
        f"CATEGORÍAS VÁLIDAS (principal → subcategorías):\n"
        + "\n".join(cat_lines) + "\n\n"
        f"ELEMENTOS AFECTADOS VÁLIDOS (principal → subcategorías):\n"
        + "\n".join(elem_lines) + "\n\n"
        f"ZONAS VÁLIDAS: {', '.join(zones)}\n"
        f"RIESGOS VÁLIDOS: {', '.join(risks)}\n"
        f"URGENCIAS VÁLIDAS: {', '.join(urgencies)}\n\n"
        f"IMPORTANTE: Usa SOLO los valores de las listas anteriores para category, "
        f"affected_elements, location.zone, risks y urgency. No inventes valores "
        f"fuera de la taxonomía.\n"
        f"IMPORTANTE: image_descriptions debe contener las descripciones de las fotos "
        f"en texto plano (una string por foto, en orden). Si no hay fotos, deja [].\n\n"
        f"RESPONDE SOLO con JSON en este formato:\n"
        f"{json.dumps(extraction_schema, indent=2, ensure_ascii=False)}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text},
    ]
