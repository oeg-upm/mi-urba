"""
Scenario sampling — picks problem parameters from taxonomies and weights.

All data comes from config.yaml. This module only contains sampling logic.
"""

import random

from . import config as cfg


# ── Persona weights (derived from config at import time) ─────────

def _persona_weights() -> dict[str, float]:
    """Build persona weight dict from config."""
    personas = cfg.get_config().get("personas", {})
    return {pid: p["weight"] for pid, p in personas.items()}


def weighted_sample(choices: dict[str, float]) -> str:
    """Sample a key from a dict weighted by its values."""
    keys = list(choices.keys())
    weights = [choices[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


# ── Category → plausible affected elements ──────────────────────

_CATEGORY_ELEMENTS: dict[str, list[str]] = {
    "fontanería": ["tubería", "instalación_sanitaria", "revestimiento", "suelo"],
    "electricidad": ["instalación_eléctrica", "equipamiento"],
    "cubierta": ["cubierta", "revestimiento", "muro"],
    "fachada": ["muro", "revestimiento"],
    "carpintería": ["ventana", "puerta"],
    "jardinería": ["jardín", "cerramiento"],
    "limpieza": ["suelo", "mobiliario_común"],
    "ascensor": ["equipamiento", "puerta"],
    "climatización": ["equipamiento", "instalación_sanitaria", "tubería"],
    "piscina": ["equipamiento", "instalación_sanitaria"],
    "garaje": ["suelo", "puerta", "instalación_eléctrica", "instalación_sanitaria"],
    "plagas": ["muro", "revestimiento", "jardín"],
    "mobiliario": ["mobiliario_común"],
    "seguridad": ["cerramiento", "equipamiento", "instalación_eléctrica"],
    "ruido": ["muro", "estructura"],
}

# ── Category → plausible location zones ────────────────────────

_CATEGORY_ZONES: dict[str, list[str]] = {
    "fontanería": ["garaje", "portal", "exterior", "azotea", "locales"],
    "electricidad": ["portal", "garaje", "exterior", "azotea"],
    "cubierta": ["exterior", "azotea"],
    "fachada": ["exterior"],
    "carpintería": ["portal", "exterior"],
    "jardinería": ["zonas_verdes"],
    "limpieza": ["portal", "garaje", "zonas_verdes", "piscina"],
    "ascensor": ["portal"],
    "climatización": ["azotea", "locales", "portal"],
    "piscina": ["piscina"],
    "garaje": ["garaje"],
    "plagas": ["portal", "garaje", "zonas_verdes", "exterior"],
    "mobiliario": ["zonas_verdes", "portal", "piscina"],
    "seguridad": ["exterior", "portal", "garaje"],
    "ruido": ["portal", "garaje", "locales"],
}

# ── Category → plausible risks ────────────────────────────────

_CATEGORY_RISKS: dict[str, list[str]] = {
    "fontanería": ["inundación", "eléctrico", "sanitario"],
    "electricidad": ["eléctrico", "incendio"],
    "cubierta": ["caída", "inundación"],
    "fachada": ["caída", "estructural"],
    "carpintería": ["seguridad_persona", "accesibilidad"],
    "jardinería": ["ninguno"],
    "limpieza": ["sanitario", "ninguno"],
    "ascensor": ["accesibilidad", "seguridad_persona"],
    "climatización": ["ninguno", "sanitario"],
    "piscina": ["sanitario", "seguridad_persona"],
    "garaje": ["ninguno", "seguridad_persona"],
    "plagas": ["sanitario"],
    "mobiliario": ["caída", "ninguno"],
    "seguridad": ["seguridad_persona"],
    "ruido": ["ninguno"],
    "varios": ["ninguno"],
}

# ── Category → urgency weights ─────────────────────────────────

_URGENCY_WEIGHTS: dict[str, dict[str, float]] = {
    "fontanería":    {"inmediata": 5,  "urgente": 20, "alta": 30, "media": 30, "baja": 10, "planificable": 5},
    "electricidad":  {"inmediata": 10, "urgente": 25, "alta": 30, "media": 25, "baja": 5,  "planificable": 5},
    "cubierta":      {"inmediata": 5,  "urgente": 20, "alta": 35, "media": 25, "baja": 10, "planificable": 5},
    "fachada":       {"inmediata": 5,  "urgente": 10, "alta": 20, "media": 35, "baja": 20, "planificable": 10},
    "carpintería":  {"inmediata": 2,  "urgente": 8,  "alta": 20, "media": 40, "baja": 20, "planificable": 10},
    "jardinería":   {"inmediata": 2,  "urgente": 5,  "alta": 10, "media": 30, "baja": 35, "planificable": 18},
    "limpieza":     {"inmediata": 2,  "urgente": 5,  "alta": 15, "media": 35, "baja": 30, "planificable": 13},
    "ascensor":     {"inmediata": 15, "urgente": 30, "alta": 30, "media": 15, "baja": 5,  "planificable": 5},
    "climatización": {"inmediata": 5,  "urgente": 10, "alta": 20, "media": 35, "baja": 20, "planificable": 10},
    "piscina":      {"inmediata": 5,  "urgente": 10, "alta": 20, "media": 35, "baja": 20, "planificable": 10},
    "garaje":       {"inmediata": 5,  "urgente": 10, "alta": 25, "media": 35, "baja": 15, "planificable": 10},
    "plagas":       {"inmediata": 5,  "urgente": 15, "alta": 25, "media": 30, "baja": 15, "planificable": 10},
    "mobiliario":   {"inmediata": 1,  "urgente": 4,  "alta": 10, "media": 30, "baja": 35, "planificable": 20},
    "seguridad":    {"inmediata": 10, "urgente": 25, "alta": 30, "media": 20, "baja": 10, "planificable": 5},
    "ruido":        {"inmediata": 5,  "urgente": 15, "alta": 25, "media": 35, "baja": 15, "planificable": 5},
    "varios":       {"inmediata": 2,  "urgente": 8,  "alta": 15, "media": 35, "baja": 25, "planificable": 15},
}


def sample_affected_elements(category: str) -> list[dict]:
    """Sample 1-3 affected elements relevant to the problem category."""
    elements_cfg = cfg.get_affected_elements()
    available = _CATEGORY_ELEMENTS.get(category, list(elements_cfg.keys()))
    n = random.choices([1, 2, 3], weights=[0.50, 0.35, 0.15], k=1)[0]
    n = min(n, len(available))
    chosen = random.sample(available, n)

    result = []
    for elem in chosen:
        subs = elements_cfg.get(elem, [])
        sub = random.choice(subs) if subs else "general"
        result.append({"principal": elem, "subcategory": sub})
    return result


def sample_location(category: str) -> dict:
    """Sample a location zone and detail relevant to the problem category."""
    zones_cfg = cfg.get_location_zones()
    zones = _CATEGORY_ZONES.get(category, list(zones_cfg.keys()))
    zone = random.choice(zones)
    details = zones_cfg.get(zone, [])
    detail = random.choice(details) if details else ""
    return {"zone": zone, "detail": detail}


def sample_risks(category: str, urgency: str) -> list[str]:
    """Sample relevant risks based on category and urgency."""
    possible = _CATEGORY_RISKS.get(category, ["ninguno"])
    if urgency in ("inmediata", "urgente") and "ninguno" in possible and len(possible) > 1:
        possible = [r for r in possible if r != "ninguno"]
    n = random.choices([0, 1, 2], weights=[0.20, 0.60, 0.20], k=1)[0]
    n = min(n, len(possible))
    if n == 0:
        return ["ninguno"]
    return random.sample(possible, n)


def sample_urgency(category: str) -> str:
    """Sample urgency level weighted by category."""
    weights = _URGENCY_WEIGHTS.get(category, _URGENCY_WEIGHTS["varios"])
    return weighted_sample(weights)


def sample_scenario() -> dict:
    """Sample a complete problem scenario from config."""
    weights = cfg.get_weights()
    categories = cfg.get_categories()

    category = weighted_sample(weights["category"])
    subcategories = categories[category]
    subcategory = random.choice(subcategories) if subcategories else "general"

    urgency = sample_urgency(category)
    location = sample_location(category)
    risks = sample_risks(category, urgency)
    affected_elements = sample_affected_elements(category)

    return {
        "category": category,
        "subcategory": subcategory,
        "urgency": urgency,
        "location_zone": location["zone"],
        "location_detail": location["detail"],
        "risks": risks,
        "affected_elements": affected_elements,
    }


def sample_community() -> dict:
    """Sample a community uniformly."""
    communities = cfg.get_communities()
    return random.choice(communities)


def should_have_photos() -> bool:
    """Decide if this complaint includes photos."""
    prob = cfg.get_weights()["photo_prob"]
    return random.random() < prob


def sample_photo_count(has_photos: bool) -> int:
    """Sample number of photos."""
    if not has_photos:
        return 0
    dist = cfg.get_weights()["photo_count"]
    return random.choices(list(dist.keys()), weights=list(dist.values()), k=1)[0]


def should_be_multi_problem() -> bool:
    """Decide if this complaint has multiple problems."""
    prob = cfg.get_weights()["multi_problem"]
    return random.random() < prob


def sample_persona() -> dict:
    """Sample a neighbor persona from config."""
    personas = cfg.get_config().get("personas", {})
    pid = weighted_sample(_persona_weights())
    persona = personas[pid]
    return {
        "id": pid,
        "profile": persona["profile"],
        "style": persona["style"],
        "weight": persona["weight"],
    }


def sample_modifier() -> dict | None:
    """Optionally sample a style modifier to compose with a persona.

    Returns None if no modifier is applied (based on modifier_prob).
    When applied, the modifier is sampled from its own probability distribution.
    """
    config = cfg.get_config()
    modifier_prob = config.get("modifier_prob", 0.30)

    if random.random() >= modifier_prob:
        return None

    modifiers = config.get("modifiers", {})
    if not modifiers:
        return None

    # Sample modifier by its own probability
    mod_weights = {mid: m["prob"] for mid, m in modifiers.items()}
    mid = weighted_sample(mod_weights)
    mod = modifiers[mid]
    return {
        "id": mid,
        "description": mod["description"],
        "style_override": mod["style_override"],
        "prob": mod["prob"],
    }