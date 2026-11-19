"""Guard de portabilidad de las herramientas de `tools/` de los dos subproyectos.

Ya se rompió dos veces por lo mismo: rutas absolutas de la máquina del autor
(`/backup/...`) y endpoints de la red local (`192.168.*`) dentro de un script.
Este test falla en cuanto vuelvan a aparecer.
"""

import ast
import importlib
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIRS = [PROJECT_ROOT / "tools",
              PROJECT_ROOT.parent / "miurba-extractor" / "tools"]
FORBIDDEN = (r"/backup/", r"192\.168\.", r"\.pi/agent")


def tool_files():
    for directory in TOOLS_DIRS:
        yield from sorted(directory.glob("*.py"))


def test_tools_directories_have_scripts():
    for directory in TOOLS_DIRS:
        assert directory.is_dir(), f"falta {directory}"
        assert sorted(directory.glob("*.py")), f"{directory} no tiene scripts"


def test_no_hardcoded_machine_paths_or_lan_endpoints():
    offenders = []
    for path in tool_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern in FORBIDDEN:
                if re.search(pattern, line):
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "rutas o endpoints hardcodeados:\n" + "\n".join(offenders)


def test_tools_parse():
    for path in tool_files():
        ast.parse(path.read_text(encoding="utf-8"))


def test_generator_tools_import():
    """Solo los del generador: necesitan su propio `src` en sys.path."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    failures = []
    for path in sorted((PROJECT_ROOT / "tools").glob("*.py")):
        try:
            importlib.import_module(f"tools.{path.stem}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
    assert not failures, "no importan:\n" + "\n".join(failures)
