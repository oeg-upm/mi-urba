"""Validation of the taxonomy and the expression tables in config.yaml.

Separate from prompt building: these run at startup and fail fast, so a config
that would produce inconsistent ground truth never reaches generation.
"""
from . import config as cfg


def validate_taxonomy(cfg: dict) -> None:
    """Fail fast when the taxonomies are incomplete or cross-reference badly.

    Sampling never invents a value: if a category lacks subcategories, zones,
    risks, urgency weights or element mappings, generation stops here instead
    of silently falling back to another domain (which is how the
    undeclared label 'varios/general' ended up in the 50k corpus).
    """
    problems: list[str] = []
    categories = cfg.get("categories") or {}
    weights = (cfg.get("weights") or {}).get("category") or {}

    for cat in weights:
        if cat not in categories:
            problems.append(f"weights.category lists unknown category {cat!r}")

    for cat, subs in categories.items():
        if not subs:
            problems.append(
                f"categories[{cat!r}] has no subcategories; declare at least one"
            )

    for section in (
        "category_elements",
        "category_zones",
        "category_risks",
        "urgency_weights",
    ):
        mapping = cfg.get(section) or {}
        for cat in categories:
            if cat not in mapping or not mapping[cat]:
                problems.append(f"{section} has no entry for category {cat!r}")

    elements = cfg.get("affected_elements") or {}
    for cat, elems in (cfg.get("category_elements") or {}).items():
        for elem in elems:
            if elem not in elements:
                problems.append(
                    f"category_elements[{cat!r}] references unknown element {elem!r}"
                )
    for elem, subs in elements.items():
        if not subs:
            problems.append(
                f"affected_elements[{elem!r}] has no subcategories; declare at least one"
            )

    zones = cfg.get("location_zones") or {}
    for cat, zs in (cfg.get("category_zones") or {}).items():
        for zone in zs:
            if zone not in zones:
                problems.append(
                    f"category_zones[{cat!r}] references unknown zone {zone!r}"
                )

    declared_risks = set(cfg.get("risks") or [])
    for cat, rs in (cfg.get("category_risks") or {}).items():
        for risk in rs:
            if risk not in declared_risks:
                problems.append(f"category_risks[{cat!r}] uses undeclared risk {risk!r}")

    urgency_levels = set(cfg.get("urgency_levels") or [])
    for cat, levels in (cfg.get("urgency_weights") or {}).items():
        for level in levels:
            if level not in urgency_levels:
                problems.append(
                    f"urgency_weights[{cat!r}] uses undeclared urgency {level!r}"
                )

    if problems:
        raise ValueError(
            "incomplete taxonomy in config (silent fallbacks are not allowed):\n  - "
            + "\n  - ".join(problems)
        )


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


def validate_element_subcategories(config: dict, table: dict | None = None) -> None:
    """Fail fast when the element coherence table is inconsistent.

    Every category needs an entry; every element must exist in affected_elements
    and be allowed for that category in category_elements; every element
    subcategory must be declared for that element; no empty lists.
    """
    from . import config as cfg_module

    table = table if table is not None else cfg_module.get_element_subcategories()
    categories = config.get("categories") or {}
    elements = config.get("affected_elements") or {}
    category_elements = config.get("category_elements") or {}

    problems: list[str] = []
    for cat in categories:
        if cat not in table:
            problems.append(f"element_subcategories has no entry for category {cat!r}")

    for cat, branches in table.items():
        if not branches:
            problems.append(f"element_subcategories[{cat!r}] is empty")
            continue
        for branch, mapping in branches.items():
            where = f"element_subcategories[{cat!r}][{branch!r}]"
            if branch != "*" and branch not in (categories.get(cat) or []):
                problems.append(f"{where}: unknown problem subcategory")
            if mapping == "*":
                continue
            if not mapping:
                problems.append(f"{where}: empty mapping")
                continue
            for el, subs in mapping.items():
                if el not in elements:
                    problems.append(f"{where}: unknown element {el!r}")
                    continue
                if el not in (category_elements.get(cat) or []):
                    problems.append(
                        f"{where}: element {el!r} is not allowed for category {cat!r} "
                        f"in category_elements"
                    )
                if subs == "*":
                    continue
                if not subs:
                    problems.append(f"{where}[{el!r}]: empty subcategory list")
                    continue
                for sub in subs:
                    if sub not in elements[el]:
                        problems.append(f"{where}: {el}/{sub} is not in affected_elements")

    if problems:
        raise ValueError(
            "inconsistent element_subcategories table in taxonomy.yaml:\n  - " + "\n  - ".join(problems)
        )
