from __future__ import annotations

import ast
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1] / "app"

# These rules protect the current package layout while the new physical packages
# are introduced incrementally. New inner packages are added here as they land.
FORBIDDEN_IMPORT_PREFIXES = {
    "domains": (
        "app.api",
        "app.services",
        "app.providers",
        "app.brains",
        "app.infrastructure",
    ),
    "providers": ("app.api", "app.services"),
    "brains": ("app.api", "app.services", "app.providers"),
    "services": ("app.api",),
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    ("package_name", "forbidden"),
    FORBIDDEN_IMPORT_PREFIXES.items(),
)
def test_backend_packages_do_not_import_outward_layers(
    package_name: str,
    forbidden: tuple[str, ...],
) -> None:
    violations: list[str] = []
    for path in (APP_ROOT / package_name).rglob("*.py"):
        for imported in _imports(path):
            if imported.startswith(forbidden):
                violations.append(
                    f"{path.relative_to(APP_ROOT)} imports {imported}"
                )

    assert not violations, "\n".join(violations)


def test_frontend_uses_single_application_entrypoint() -> None:
    frontend_root = APP_ROOT.parents[1] / "frontend"
    html = (frontend_root / "index.html").read_text(encoding="utf-8")

    assert 'src="./js/app.js' in html
    assert 'src="./js/main.js' not in html

