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


# ── Sampling functions ──────────────────────────────────────────

def sample_affected_elements(category: str) -> list[dict]:
    """Sample 1-3 affected elements relevant to the problem category."""
    elements_cfg = cfg.get_affected_elements()
    category_elements = cfg.get_config().get("category_elements", {})
    available = category_elements.get(category, list(elements_cfg.keys()))
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
    category_zones = cfg.get_config().get("category_zones", {})
    zones = category_zones.get(category, list(zones_cfg.keys()))
    zone = random.choice(zones)
    details = zones_cfg.get(zone, [])
    detail = random.choice(details) if details else ""
    return {"zone": zone, "detail": detail}


def sample_risks(category: str, urgency: str) -> list[str]:
    """Sample relevant risks based on category and urgency."""
    category_risks = cfg.get_config().get("category_risks", {})
    possible = category_risks.get(category, ["ninguno"])
    if urgency in ("inmediata", "urgente") and "ninguno" in possible and len(possible) > 1:
        possible = [r for r in possible if r != "ninguno"]
    n = random.choices([0, 1, 2], weights=[0.20, 0.60, 0.20], k=1)[0]
    n = min(n, len(possible))
    if n == 0:
        return ["ninguno"]
    return random.sample(possible, n)


def sample_urgency(category: str) -> str:
    """Sample urgency level weighted by category."""
    urgency_weights = cfg.get_config().get("urgency_weights", {})
    weights = urgency_weights.get(category, urgency_weights.get("varios", {}))
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
    return int(random.choices(list(dist.keys()), weights=list(dist.values()), k=1)[0])


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