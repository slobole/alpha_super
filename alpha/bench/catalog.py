"""Strategy and portfolio discovery for Bench.

This module answers two product questions with cheap filesystem reads:

  * "What strategies do I have, and which are WIRED?"  -> :func:`list_strategies`
  * "What portfolios are defined?"                     -> :func:`list_portfolios`

Discovery is convention-based, exactly like the existing runners:

  * a strategy is any ``strategies/**/strategy_*.py`` file,
  * a strategy is *wired* when its dotted module path appears in
    ``alpha.live.release_manifest.SUPPORTED_STRATEGY_IMPORT_TUPLE``,
  * a strategy is *runnable* when it exposes a top-level ``run_variant`` def
    (that is the hook the generic runner calls).

Nothing here imports a strategy module. We parse the source with ``ast`` so a
strategy that fails to import (missing data subscription, etc.) still shows up
in the catalog instead of breaking the page.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from alpha.live.release_manifest import SUPPORTED_STRATEGY_IMPORT_TUPLE


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT_PATH = REPO_ROOT_PATH / "strategies"
PORTFOLIOS_ROOT_PATH = REPO_ROOT_PATH / "portfolios"

# Friendly labels for the strategy sub-folders. Unknown folders fall back to a
# title-cased version of the folder name, so a brand-new family still renders.
CATEGORY_LABEL_DICT: dict[str, str] = {
    "dv2": "DV2 mean-reversion",
    "qpi": "QPI mean-reversion",
    "taa_df": "TAA dual-momentum",
    "momentum": "Momentum",
    "vix_stuff": "VIX / volatility",
    "eom_tlt_vs_spy": "End-of-month",
    "bom_tlt": "Beginning-of-month",
    "alpha19": "Alpha19",
    "seasonality": "Seasonality",
}


@dataclass(frozen=True)
class StrategyEntry:
    """One runnable strategy file, plus the metadata Bench renders."""

    stem_str: str  # e.g. "strategy_mr_dv2" — also the results-tree run name
    display_name_str: str  # prettified, e.g. "Mr Dv2"
    category_str: str  # the containing folder, e.g. "dv2"
    category_label_str: str
    module_import_str: str  # e.g. "strategies.dv2.strategy_mr_dv2"
    rel_path_str: str  # posix path relative to the repo root
    is_wired_bool: bool
    has_run_variant_bool: bool
    summary_str: str  # first line of the module docstring (may be empty)


@dataclass(frozen=True)
class PortfolioPod:
    strategy_str: str
    weight_float: float


# The repo has two portfolio YAML schemas, run by two different scripts:
#   * "simple"  — keys ``name`` / ``capital`` / pods[].strategy / pods[].weight,
#                 built by ``strategies/run_portfolio.py`` (combines saved pkls).
#   * "manager" — keys ``name_str`` / ``capital_base_float`` /
#                 pods[].strategy_import_str / pods[].weight_float, run fresh by
#                 ``strategies/run_portfolio_manager.py`` (PortfolioManager).
SCHEMA_SIMPLE_STR = "simple"
SCHEMA_MANAGER_STR = "manager"


@dataclass(frozen=True)
class PortfolioEntry:
    name_str: str  # file stem — also the results-tree run name
    config_name_str: str  # the name field inside the YAML
    rel_path_str: str  # posix path relative to the repo root
    schema_str: str  # SCHEMA_SIMPLE_STR | SCHEMA_MANAGER_STR
    capital_float: float | None
    rebalance_str: str | None
    pod_tuple: tuple[PortfolioPod, ...]
    error_str: str | None  # set when the YAML could not be parsed


def _module_import_str(module_path: Path) -> str:
    relative_module_path = module_path.resolve().relative_to(REPO_ROOT_PATH)
    return ".".join(relative_module_path.with_suffix("").parts)


def _rel_posix_str(some_path: Path) -> str:
    return some_path.resolve().relative_to(REPO_ROOT_PATH).as_posix()


def prettify_stem(stem_str: str) -> str:
    """``strategy_mr_dv2`` -> ``Mr Dv2`` for a friendlier display label."""
    trimmed_str = stem_str
    for prefix_str in ("strategy_", "run_"):
        if trimmed_str.startswith(prefix_str):
            trimmed_str = trimmed_str[len(prefix_str) :]
    return trimmed_str.replace("_", " ").strip().title() or stem_str


def _category_label(category_str: str) -> str:
    return CATEGORY_LABEL_DICT.get(category_str, category_str.replace("_", " ").title())


def _wired_module_set() -> set[str]:
    # Entries are either "module" or "module:Class"; we key on the module path.
    return {entry_str.split(":", maxsplit=1)[0] for entry_str in SUPPORTED_STRATEGY_IMPORT_TUPLE}


@lru_cache(maxsize=1024)
def _parse_strategy_source(path_str: str, mtime_ns_int: int) -> tuple[str, bool]:
    """Return ``(first_docstring_line, has_run_variant)`` for a strategy file.

    Cached on ``(path, mtime_ns)`` so edits invalidate the entry automatically.
    Parsing is best-effort: a syntactically broken file degrades to no summary
    and "not runnable" rather than raising.
    """
    try:
        # Some strategy files start with a UTF-8 BOM and/or carry cp1252 bytes
        # (em-dashes in comments). "utf-8-sig" strips a leading BOM the way the
        # import machinery does, and errors="replace" tolerates the odd byte —
        # which only ever lives in a comment or string, never in the tokens we
        # care about (the module docstring and the run_variant def).
        source_str = Path(path_str).read_bytes().decode("utf-8-sig", errors="replace")
        module_ast = ast.parse(source_str)
    except (OSError, SyntaxError, ValueError):
        return ("", False)

    docstring_str = ast.get_docstring(module_ast) or ""
    first_line_str = ""
    for raw_line_str in docstring_str.strip().splitlines():
        if raw_line_str.strip():
            first_line_str = raw_line_str.strip()
            break

    has_run_variant_bool = any(
        isinstance(node_obj, (ast.FunctionDef, ast.AsyncFunctionDef)) and node_obj.name == "run_variant"
        for node_obj in module_ast.body
    )
    return (first_line_str, has_run_variant_bool)


def list_strategies() -> list[StrategyEntry]:
    """All ``strategy_*.py`` files under ``strategies/``, wired ones first.

    Sort order: wired before experimental, then by category, then by name —
    so the strategies you actually trade sit at the top of the catalog.
    """
    wired_module_set = _wired_module_set()
    entry_list: list[StrategyEntry] = []

    for module_path in sorted(STRATEGIES_ROOT_PATH.rglob("strategy_*.py")):
        module_import_str = _module_import_str(module_path)
        summary_str, has_run_variant_bool = _parse_strategy_source(
            str(module_path), module_path.stat().st_mtime_ns
        )
        category_str = (
            module_path.parent.name if module_path.parent != STRATEGIES_ROOT_PATH else "uncategorized"
        )
        entry_list.append(
            StrategyEntry(
                stem_str=module_path.stem,
                display_name_str=prettify_stem(module_path.stem),
                category_str=category_str,
                category_label_str=_category_label(category_str),
                module_import_str=module_import_str,
                rel_path_str=_rel_posix_str(module_path),
                is_wired_bool=module_import_str in wired_module_set,
                has_run_variant_bool=has_run_variant_bool,
                summary_str=summary_str,
            )
        )

    entry_list.sort(
        key=lambda entry_obj: (
            not entry_obj.is_wired_bool,
            entry_obj.category_label_str.lower(),
            entry_obj.stem_str.lower(),
        )
    )
    return entry_list


def get_strategy_by_module(module_import_str: str) -> StrategyEntry | None:
    for entry_obj in list_strategies():
        if entry_obj.module_import_str == module_import_str:
            return entry_obj
    return None


def list_categories() -> list[tuple[str, str]]:
    """Distinct ``(category, label)`` pairs present in the catalog, sorted."""
    seen_dict: dict[str, str] = {}
    for entry_obj in list_strategies():
        seen_dict[entry_obj.category_str] = entry_obj.category_label_str
    return sorted(seen_dict.items(), key=lambda pair: pair[1].lower())


def _short_strategy_label(strategy_ref_str: str) -> str:
    """``strategies.taa_df.strategy_taa_x:Cls`` -> ``strategy_taa_x``."""
    module_ref_str = strategy_ref_str.split(":", maxsplit=1)[0]
    return module_ref_str.rsplit(".", maxsplit=1)[-1]


def _coerce_pod_tuple(raw_pods_obj: object) -> tuple[PortfolioPod, ...]:
    """Read pods from either schema (``strategy``/``weight`` or
    ``strategy_import_str``/``weight_float``)."""
    if not isinstance(raw_pods_obj, list):
        return ()
    pod_list: list[PortfolioPod] = []
    for raw_pod_obj in raw_pods_obj:
        if not isinstance(raw_pod_obj, dict):
            continue
        strategy_obj = raw_pod_obj.get("strategy", raw_pod_obj.get("strategy_import_str"))
        weight_obj = raw_pod_obj.get("weight", raw_pod_obj.get("weight_float"))
        if not isinstance(strategy_obj, str):
            continue
        try:
            weight_float = float(weight_obj)
        except (TypeError, ValueError):
            weight_float = 0.0
        pod_list.append(
            PortfolioPod(strategy_str=_short_strategy_label(strategy_obj), weight_float=weight_float)
        )
    return tuple(pod_list)


def list_portfolios() -> list[PortfolioEntry]:
    """All ``portfolios/*.yaml`` configs, parsed defensively for display."""
    if not PORTFOLIOS_ROOT_PATH.exists():
        return []

    entry_list: list[PortfolioEntry] = []
    for config_path in sorted(PORTFOLIOS_ROOT_PATH.glob("*.yaml")):
        rel_path_str = _rel_posix_str(config_path)
        try:
            config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(config_dict, dict):
                raise ValueError("top-level YAML is not a mapping")
            is_manager_bool = (
                "name_str" in config_dict
                or "capital_base_float" in config_dict
                or "allocation_policy_str" in config_dict
            )
            capital_obj = config_dict.get("capital", config_dict.get("capital_base_float"))
            entry_list.append(
                PortfolioEntry(
                    name_str=config_path.stem,
                    config_name_str=str(
                        config_dict.get("name", config_dict.get("name_str", config_path.stem))
                    ),
                    rel_path_str=rel_path_str,
                    schema_str=SCHEMA_MANAGER_STR if is_manager_bool else SCHEMA_SIMPLE_STR,
                    capital_float=float(capital_obj) if capital_obj is not None else None,
                    rebalance_str=config_dict.get("rebalance"),
                    pod_tuple=_coerce_pod_tuple(config_dict.get("pods")),
                    error_str=None,
                )
            )
        except (OSError, ValueError, yaml.YAMLError) as exception_obj:
            entry_list.append(
                PortfolioEntry(
                    name_str=config_path.stem,
                    config_name_str=config_path.stem,
                    rel_path_str=rel_path_str,
                    schema_str=SCHEMA_SIMPLE_STR,
                    capital_float=None,
                    rebalance_str=None,
                    pod_tuple=(),
                    error_str=str(exception_obj),
                )
            )
    return entry_list


def get_portfolio_by_rel_path(rel_path_str: str) -> PortfolioEntry | None:
    for entry_obj in list_portfolios():
        if entry_obj.rel_path_str == rel_path_str:
            return entry_obj
    return None
