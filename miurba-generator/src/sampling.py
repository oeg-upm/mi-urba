"""
Scenario sampling — picks problem parameters from taxonomies and weights.

All data comes from config.yaml. Every sampler accepts an optional
`random.Random` so a record can be sampled from its own derived stream: that
is what makes a seeded campaign reproducible independently of the order of
records, the number of workers or how many retries happened before.
"""

import random

from . import config as cfg


def _rng(rng: random.Random | None) -> random.Random:
    """Return the given stream or the global one (non-deterministic mode)."""
    return rng if rng is not None else random


# ── Persona weights (derived from config at import time) ─────────

def _persona_weights() -> dict[str, float]:
    """Build persona weight dict from config."""
    personas = cfg.get_config().get("personas", {})
    return {pid: p["weight"] for pid, p in personas.items()}


def weighted_sample(choices: dict[str, float], rng: random.Random | None = None) -> str:
    """Sample a key from a dict weighted by its values."""
    keys = list(choices.keys())
    weights = [choices[k] for k in keys]
    return _rng(rng).choices(keys, weights=weights, k=1)[0]


# ── Sampling functions ──────────────────────────────────────────

def _require_mapping(mapping: dict, category: str, section: str) -> list:
    """Return mapping[category] or fail loudly: no silent fallbacks."""
    if category not in mapping:
        raise ValueError(
            f"category {category!r} has no entry in {section}: add one to the config "
            f"(known keys: {', '.join(sorted(mapping)) or 'none'})"
        )
    values = mapping[category]
    if not values:
        raise ValueError(f"category {category!r} has an empty list in {section}")
    return values


def _allowed_elements(category: str, subcategory: str) -> dict[str, list[str]]:
    """Elements (and their subcategories) compatible with this problem.

    Uses the coherence table element_subcategories.yaml: category-level baseline
    under "*" plus optional per-subcategory overrides. No silent fallback: a
    missing category is a configuration error.
    """
    table = cfg.get_element_subcategories()
    elements_cfg = cfg.get_affected_elements()
    if category not in table:
        raise ValueError(
            f"category {category!r} has no entry in element_subcategories.yaml"
        )
    branch = table[category]
    allowed = branch.get(subcategory) or branch.get("*")
    if allowed == "*":  # catch-all category: every element with all its subcategories
        return {el: list(subs) for el, subs in elements_cfg.items()}
    if not allowed:
        raise ValueError(
            f"no element mapping for {category}/{subcategory} in element_subcategories.yaml"
        )
    out: dict[str, list[str]] = {}
    for el, subs in allowed.items():
        if el not in elements_cfg:
            raise ValueError(f"element_subcategories: unknown element {el!r} in {category}")
        if subs == "*":
            out[el] = list(elements_cfg[el])
        else:
            missing = [s for s in subs if s not in elements_cfg[el]]
            if missing:
                raise ValueError(
                    f"element_subcategories: {el}/{missing[0]} does not exist in affected_elements"
                )
            out[el] = list(subs)
    return out


def sample_affected_elements(category: str, subcategory: str,
                             rng: random.Random | None = None) -> list[dict]:
    """Sample 1-3 affected elements coherent with the given problem.

    Both the element and its subcategory come from the coherence table, so the
    scenario never asks the generator to describe an impossible pairing (e.g. a
    hot-water fault on the sewage installation).
    """
    allowed = _allowed_elements(category, subcategory)
    principals = list(allowed)
    r = _rng(rng)
    n = r.choices([1, 2, 3], weights=[0.50, 0.35, 0.15], k=1)[0]
    n = min(n, len(principals))
    chosen = r.sample(principals, n)
    return [{"principal": el, "subcategory": r.choice(allowed[el])} for el in chosen]


def sample_location(category: str, rng: random.Random | None = None) -> dict:
    """Sample a location zone and detail relevant to the problem category."""
    zones_cfg = cfg.get_location_zones()
    category_zones = cfg.get_config().get("category_zones", {})
    zones = _require_mapping(category_zones, category, "category_zones")
    zone = _rng(rng).choice(zones)
    if zone not in zones_cfg:
        raise ValueError(
            f"category_zones[{category!r}] references unknown zone {zone!r}; "
            f"declare it in location_zones"
        )
    details = zones_cfg[zone]
    detail = _rng(rng).choice(details) if details else ""
    return {"zone": zone, "detail": detail}


def sample_risks(category: str, urgency: str,
                 rng: random.Random | None = None) -> list[str]:
    """Sample relevant risks based on category and urgency.

    'ninguno' is never sampled alongside a concrete risk: it is the exclusive
    outcome when the scenario carries no risk at all. Sampling it as just
    another member of the pool produced ground truths like
    ['ninguno', 'sanitario'], which no text can satisfy and which made the
    judge's risk check undecidable.
    """
    category_risks = cfg.get_config().get("category_risks", {})
    possible = [r for r in _require_mapping(category_risks, category, "category_risks")
                if r != "ninguno"]
    r = _rng(rng)
    n = r.choices([0, 1, 2], weights=[0.20, 0.60, 0.20], k=1)[0]
    n = min(n, len(possible))
    if n == 0:
        return ["ninguno"]
    return r.sample(possible, n)


def sample_urgency(category: str, rng: random.Random | None = None) -> str:
    """Sample urgency level weighted by category."""
    urgency_weights = cfg.get_config().get("urgency_weights", {})
    weights = _require_mapping(urgency_weights, category, "urgency_weights")
    return weighted_sample(weights, rng)


def sample_scenario(rng: random.Random | None = None) -> dict:
    """Sample a complete problem scenario from config."""
    weights = cfg.get_weights()
    categories = cfg.get_categories()

    category = weighted_sample(weights["category"], rng)
    if category not in categories:
        raise ValueError(
            f"weights.category references unknown category {category!r}; "
            f"declare it in categories"
        )
    subcategories = categories[category]
    if not subcategories:
        raise ValueError(
            f"category {category!r} has no subcategories; declare at least one in categories"
        )
    subcategory = _rng(rng).choice(subcategories)

    urgency = sample_urgency(category, rng)
    location = sample_location(category, rng)
    risks = sample_risks(category, urgency, rng)
    affected_elements = sample_affected_elements(category, subcategory, rng)

    return {
        "category": category,
        "subcategory": subcategory,
        "urgency": urgency,
        "location_zone": location["zone"],
        "location_detail": location["detail"],
        "risks": risks,
        "affected_elements": affected_elements,
    }


def sample_community(rng: random.Random | None = None) -> dict:
    """Sample a community uniformly."""
    communities = cfg.get_communities()
    return _rng(rng).choice(communities)


def should_be_multi_problem(rng: random.Random | None = None) -> bool:
    """Decide if this complaint has multiple problems."""
    prob = cfg.get_weights()["multi_problem"]
    return _rng(rng).random() < prob


def sample_persona(rng: random.Random | None = None) -> dict:
    """Sample a neighbor persona from config."""
    personas = cfg.get_config().get("personas", {})
    pid = weighted_sample(_persona_weights(), rng)
    persona = personas[pid]
    return {
        "id": pid,
        "profile": persona["profile"],
        "style": persona["style"],
        "weight": persona["weight"],
    }


def sample_modifier(rng: random.Random | None = None) -> dict | None:
    """Optionally sample a style modifier to compose with a persona.

    Returns None if no modifier is applied (based on modifier_prob).
    When applied, the modifier is sampled from its own probability distribution.
    """
    config = cfg.get_config()
    modifier_prob = config.get("modifier_prob", 0.30)

    if _rng(rng).random() >= modifier_prob:
        return None

    modifiers = config.get("modifiers", {})
    if not modifiers:
        return None

    # Sample modifier by its own probability
    mod_weights = {mid: m["prob"] for mid, m in modifiers.items()}
    mid = weighted_sample(mod_weights, rng)
    mod = modifiers[mid]
    return {
        "id": mid,
        "description": mod["description"],
        "style_override": mod["style_override"],
        "prob": mod["prob"],
    }