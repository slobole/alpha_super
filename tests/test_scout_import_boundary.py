"""Import boundaries: live code never reaches Scout, and alpha/stats stays light.

The scanner resolves every import form statically: `import alpha.scout`,
`from alpha import scout`, relative imports (`from ..scout import x`) and
transitive chains (alpha.live -> some alpha module -> alpha.scout). Dynamic
imports (`importlib.import_module("...")`) are not visible to it; none exist in
alpha/live today and code review must keep it that way.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]


def _module_name_str(source_path: Path, root_path: Path) -> str:
    relative_part_tuple = source_path.relative_to(root_path).with_suffix("").parts
    if relative_part_tuple[-1] == "__init__":
        relative_part_tuple = relative_part_tuple[:-1]
    return ".".join(relative_part_tuple)


def _imported_names(source_path: Path, module_name_str: str) -> set[str]:
    """Fully qualified names imported by one file, with relative and `from pkg import sub` forms expanded."""
    is_package_bool = source_path.name == "__init__.py"
    package_part_list = module_name_str.split(".") if is_package_bool else module_name_str.split(".")[:-1]
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    name_set: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            name_set.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_part_list = package_part_list[: len(package_part_list) - (node.level - 1)]
                base_str = ".".join(base_part_list + ([node.module] if node.module else []))
            else:
                base_str = node.module or ""
            name_set.add(base_str)
            # `from alpha import scout` imports the submodule alpha.scout.
            name_set.update(f"{base_str}.{alias.name}" for alias in node.names)
    return name_set


def _import_graph(root_path: Path, package_str: str = "alpha") -> dict[str, set[str]]:
    graph_dict: dict[str, set[str]] = {}
    for source_path in (root_path / package_str).rglob("*.py"):
        module_name_str = _module_name_str(source_path, root_path)
        graph_dict[module_name_str] = _imported_names(source_path, module_name_str)
    return graph_dict


def _reachable_modules(graph_dict: dict[str, set[str]], start_prefix_str: str) -> set[str]:
    """Every module name reachable from modules under start_prefix_str (including imported names themselves)."""
    frontier_list = [name for name in graph_dict if name == start_prefix_str or name.startswith(start_prefix_str + ".")]
    seen_set: set[str] = set(frontier_list)
    while frontier_list:
        module_name_str = frontier_list.pop()
        for imported_str in graph_dict.get(module_name_str, set()):
            # An import of a.b.c also executes a and a.b.
            part_list = imported_str.split(".")
            for depth_int in range(1, len(part_list) + 1):
                candidate_str = ".".join(part_list[:depth_int])
                if candidate_str not in seen_set:
                    seen_set.add(candidate_str)
                    if candidate_str in graph_dict:
                        frontier_list.append(candidate_str)
    return seen_set


def _scout_names(name_set: set[str]) -> set[str]:
    return {name for name in name_set if name == "alpha.scout" or name.startswith("alpha.scout.")}


def test_scanner_catches_every_static_form(tmp_path):
    """Negative control: each import form in a fake live package must be flagged."""
    form_list = [
        "import alpha.scout",
        "import alpha.scout.ledger as ledger_module",
        "from alpha.scout import ledger",
        "from alpha.scout.ledger import Ledger",
        "from alpha import scout",
        "from .. import scout",
        "from ..scout import ledger",
        "from ..scout.ledger import Ledger",
    ]
    (tmp_path / "alpha" / "scout").mkdir(parents=True)
    (tmp_path / "alpha" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "alpha" / "scout" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "alpha" / "scout" / "ledger.py").write_text("", encoding="utf-8")
    for form_idx_int, form_str in enumerate(form_list):
        live_path = tmp_path / "alpha" / "live" / f"pkg{form_idx_int}"
        live_path.mkdir(parents=True)
        (tmp_path / "alpha" / "live" / "__init__.py").write_text("", encoding="utf-8")
        (live_path / "__init__.py").write_text("", encoding="utf-8")
        # Relative forms are written one level deeper so ".." lands on alpha.
        target_path = live_path / "module.py" if not form_str.startswith("from ..") else tmp_path / "alpha" / "live" / f"mod{form_idx_int}.py"
        target_path.write_text(f"def f():\n    if True:\n        {form_str}\n", encoding="utf-8")
        graph_dict = _import_graph(tmp_path)
        module_name_str = _module_name_str(target_path, tmp_path)
        assert _scout_names(graph_dict[module_name_str]), form_str
        target_path.unlink()

    # Transitive: alpha.live.runner -> alpha.helper -> alpha.scout
    (tmp_path / "alpha" / "helper.py").write_text("from alpha.scout import ledger\n", encoding="utf-8")
    (tmp_path / "alpha" / "live" / "runner.py").write_text("import alpha.helper\n", encoding="utf-8")
    assert _scout_names(_reachable_modules(_import_graph(tmp_path), "alpha.live"))


def test_live_never_reaches_scout():
    graph_dict = _import_graph(REPO_ROOT_PATH)
    live_module_list = [name for name in graph_dict if name.startswith("alpha.live")]
    assert len(live_module_list) > 20  # positive control: the scan really covered alpha/live
    assert not _scout_names(_reachable_modules(graph_dict, "alpha.live"))


def test_stats_package_stays_light():
    allowed_root_set = {"__future__", "dataclasses", "typing", "collections", "itertools", "math", "numpy", "pandas", "scipy", "alpha"}
    graph_dict = _import_graph(REPO_ROOT_PATH)
    stats_module_list = [name for name in graph_dict if name.startswith("alpha.stats")]
    assert len(stats_module_list) >= 8
    for module_name_str in stats_module_list:
        for imported_str in graph_dict[module_name_str]:
            root_str = imported_str.split(".")[0]
            assert root_str in allowed_root_set, (module_name_str, imported_str)
            if root_str == "alpha":
                assert imported_str.startswith("alpha.stats"), (module_name_str, imported_str)
