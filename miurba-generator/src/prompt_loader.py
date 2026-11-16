"""Loader for prompt templates kept outside the code.

Prompts live in `<repo>/prompts/*.md` as plain text with `{{placeholder}}`
substitutions, so they can be reviewed in a diff, versioned and fingerprinted.
A template may declare sections with `## name` headings; a file with no
headings is treated as a single `default` section.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SECTION_RE = re.compile(r"^##[ \t]*([A-Za-z0-9_]+)[ \t]*$", re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


class PromptError(RuntimeError):
    """Raised when a template or a placeholder is missing."""


@lru_cache(maxsize=None)
def load_template(name: str) -> dict[str, str]:
    """Return {section: text} for prompts/<name>.md (cached)."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise PromptError(f"prompt template not found: {path}")
    raw = path.read_text(encoding="utf-8")
    matches = list(_SECTION_RE.finditer(raw))
    if not matches:
        return {"default": raw.strip("\n")}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections[m.group(1)] = raw[m.end():end].strip("\n")
    return sections


def render(name: str, section: str = "default", **variables: object) -> str:
    """Render one section, substituting {{placeholders}}.

    Unknown placeholders are left untouched on purpose so a missing variable is
    visible in the output instead of silently becoming an empty string.
    """
    try:
        template = load_template(name)[section]
    except KeyError:
        available = ", ".join(sorted(load_template(name)))
        raise PromptError(
            f"prompt {name!r} has no section {section!r} (available: {available})"
        ) from None
    return _PLACEHOLDER_RE.sub(
        lambda m: str(variables[m.group(1)]) if m.group(1) in variables else m.group(0),
        template,
    )


def fingerprint(name: str) -> str:
    """SHA-256 of the template file: identifies which prompt produced a corpus."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise PromptError(f"prompt template not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint_all() -> dict[str, str]:
    """Fingerprints of every template in the prompts directory."""
    return {p.stem: fingerprint(p.stem) for p in sorted(PROMPTS_DIR.glob("*.md"))}
