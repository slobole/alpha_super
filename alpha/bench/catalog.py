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

from alpha import strategy_registry
from alpha.strategy_registry import MaturityTier


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT_PATH = REPO_ROOT_PATH / "strategies"
PORTFOLIOS_ROOT_PATH = REPO_ROOT_PATH / "portfolios"

# Friendly labels for the strategy sub-folders. Unknown folders fall back to a
# title-cased version of the folder name, so a brand-new family still renders.
CATEGORY_LABEL_DICT: dict[str, str] = {
    "dv2": "DV2 mean-reversion",
    "hpi": "HPI mean-reversion",
    "mean_reversion": "Sector Dispersion",
    "qpi": "QPI mean-reversion",
    "taa_df": "TAA dual-momentum",
    "momentum": "Momentum",
    "vix_stuff": "VIX / volatility",
    "eom_tlt_vs_spy": "End-of-month",
    "bom_tlt": "Beginning-of-month",
    "alpha19": "Alpha19",
    "seasonality": "Seasonality",
}

MOMENTUM_CATEGORY_STR = "momentum"
MOMENTUM_SUBCATEGORY_LABEL_DICT: dict[str, str] = {
    "atr_normalized_rotation": "ATR-Normalized Rotation",
    "smooth_trend": "Smooth Trend",
    "canary_rotation": "Canary Rotation",
    "etf_timing_trend": "ETF Timing & Trend",
    "gap_overnight": "Gap / Overnight",
    "cross_sectional": "Cross-Sectional Momentum",
    "other_allocation": "Other Allocation",
}

CROSS_SECTIONAL_MOMENTUM_STEM_SET = frozenset(
    {
        "strategy_mo_alpha23_breakout",
        "strategy_mo_clenow_top10_vol63",
        "strategy_mo_ev_lrb_252_ndx",
        "strategy_mo_jt_12_1_top20",
        "strategy_mo_paper_b_russell1000_vol15",
        "strategy_mo_paper_b_russell3000_vol10",
        "strategy_mo_pretom_loser_short_sp500",
        "strategy_mo_pta_winner_continuation",
        "strategy_mo_sp500_ret3m_natr21_long_short",
        "strategy_mo_weekly_sector_momentum",
    }
)


# run_variant kwargs Bench will never offer as a form field. These are the
# runner's own plumbing — Bench sets them itself when it builds the command, and
# letting the operator override them would change where artifacts land or
# whether they are written at all.
RUNNER_CONTROLLED_KWARG_SET = frozenset(
    {"show_display_bool", "save_results_bool", "output_dir_str"}
)

# A --strategy-kwarg value is one JSON scalar or string on a command line, so
# only scalar parameters can be offered as form fields. The repo's Domain_Type
# naming convention makes that decidable from the name alone: run_variant kwargs
# like ``pricing_data_df`` or a bare ``config`` take objects no text box can
# express, and offering them would produce a job that dies on launch.
SCALAR_KWARG_SUFFIX_TUPLE = ("_str", "_float", "_int", "_bool")


@dataclass(frozen=True)
class RunVariantParam:
    """One overridable keyword argument of a strategy's ``run_variant``."""

    name_str: str
    default_repr_str: str  # source text of the default, or "" when there is none


@dataclass(frozen=True)
class StrategyEntry:
    """One runnable strategy file, plus the metadata Bench renders."""

    stem_str: str  # e.g. "strategy_mr_dv2" — also the results-tree run name
    display_name_str: str  # prettified, e.g. "Mr Dv2"
    category_str: str  # the containing folder, e.g. "dv2"
    category_label_str: str
    subcategory_str: str | None  # finer UI grouping; currently used for Momentum
    subcategory_label_str: str | None
    module_import_str: str  # e.g. "strategies.dv2.strategy_mr_dv2"
    rel_path_str: str  # posix path relative to the repo root
    tier_int: int  # alpha.strategy_registry.MaturityTier
    tier_label_str: str  # "research" | "pm-ready" | "wired"
    has_run_variant_bool: bool
    has_capacity_analysis_bool: bool
    has_timing_analysis_bool: bool
    summary_str: str  # first line of the module docstring (may be empty)
    run_variant_param_tuple: tuple[RunVariantParam, ...]  # overridable kwargs

    @property
    def is_wired_bool(self) -> bool:
        """Connected to a live account route. Derived so callers predating the
        tiers keep working unchanged."""
        return self.tier_int >= int(MaturityTier.WIRED)

    @property
    def is_pm_ready_bool(self) -> bool:
        """May be allocated to inside a portfolio book. Wired implies pm-ready."""
        return self.tier_int >= int(MaturityTier.PM_READY)


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
    name_str: str  # YAML filename stem used by the Bench card
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


def _momentum_subcategory_str(stem_str: str) -> str:
    """Return a stable UI family for one strategy in ``strategies/momentum``."""
    if stem_str.startswith("strategy_mo_atr_normalized_") or stem_str == "strategy_mo_radge_ndx":
        return "atr_normalized_rotation"
    if stem_str.startswith("strategy_mo_smooth_trend_"):
        return "smooth_trend"
    if "_sphb_splv_canary" in stem_str:
        return "canary_rotation"
    if (
        stem_str.startswith("strategy_mo_mtum_timed_by_")
        or stem_str.startswith("strategy_mo_pdp_timed_by_")
        or stem_str == "strategy_mo_spy_vol_adj_ema"
    ):
        return "etf_timing_trend"
    if stem_str.startswith("strategy_mo_gappers_"):
        return "gap_overnight"
    if stem_str in CROSS_SECTIONAL_MOMENTUM_STEM_SET:
        return "cross_sectional"
    return "other_allocation"


def _subcategory_pair(category_str: str, stem_str: str) -> tuple[str | None, str | None]:
    if category_str != MOMENTUM_CATEGORY_STR:
        return (None, None)
    subcategory_str = _momentum_subcategory_str(stem_str)
    return (subcategory_str, MOMENTUM_SUBCATEGORY_LABEL_DICT[subcategory_str])


def _tier_by_module_dict() -> dict[str, MaturityTier]:
    """Maturity tier keyed on the module path.

    Registry entries are either ``module`` or ``module:Class``; the catalog
    discovers files, so it only ever knows the module.
    """
    return {
        strategy_registry.module_import_str(entry_str): tier_obj
        for entry_str, tier_obj in strategy_registry.STRATEGY_TIER_DICT.items()
    }


def _run_variant_param_tuple(run_variant_node_obj) -> tuple[RunVariantParam, ...]:
    """Overridable keyword arguments of one ``run_variant`` definition.

    Read from the signature rather than assumed, because the kwargs are not
    uniform across the catalog: most strategies take
    ``backtest_start_date_str``/``end_date_str``, but plenty do not, and
    ``run_strategy.py`` raises on a kwarg the target does not declare. Offering
    a field the strategy cannot accept would produce a job that fails on launch.

    Positional-only parameters and ``*args``/``**kwargs`` are skipped: only
    named parameters can be passed as ``--strategy-kwarg KEY=VALUE``.
    """
    arguments_obj = run_variant_node_obj.args
    named_arg_list = list(arguments_obj.args) + list(arguments_obj.kwonlyargs)
    # Defaults right-align against args; kwonly defaults pair up positionally.
    positional_default_list = list(arguments_obj.defaults)
    padded_default_list: list[object] = (
        [None] * (len(arguments_obj.args) - len(positional_default_list))
        + positional_default_list
        + list(arguments_obj.kw_defaults)
    )

    param_list: list[RunVariantParam] = []
    for arg_obj, default_obj in zip(named_arg_list, padded_default_list):
        if arg_obj.arg in RUNNER_CONTROLLED_KWARG_SET:
            continue
        if not arg_obj.arg.endswith(SCALAR_KWARG_SUFFIX_TUPLE):
            continue
        default_repr_str = ""
        if default_obj is not None:
            try:
                default_repr_str = ast.unparse(default_obj)
            except (AttributeError, ValueError):
                default_repr_str = ""
        param_list.append(
            RunVariantParam(name_str=arg_obj.arg, default_repr_str=default_repr_str)
        )
    return tuple(param_list)


@lru_cache(maxsize=1024)
def _parse_strategy_source(
    path_str: str, mtime_ns_int: int
) -> tuple[str, bool, bool, bool, tuple[RunVariantParam, ...]]:
    """Return docstring and analysis-hook availability for a strategy file.

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
        return ("", False, False, False, ())

    docstring_str = ast.get_docstring(module_ast) or ""
    first_line_str = ""
    for raw_line_str in docstring_str.strip().splitlines():
        if raw_line_str.strip():
            first_line_str = raw_line_str.strip()
            break

    run_variant_node_obj = next(
        (
            node_obj
            for node_obj in module_ast.body
            if isinstance(node_obj, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node_obj.name == "run_variant"
        ),
        None,
    )
    has_capacity_analysis_bool = any(
        isinstance(node_obj, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node_obj.name == "build_capacity_analysis_inputs"
        for node_obj in module_ast.body
    )
    has_timing_analysis_bool = any(
        isinstance(node_obj, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node_obj.name == "build_execution_timing_analysis_inputs"
        for node_obj in module_ast.body
    )
    return (
        first_line_str,
        run_variant_node_obj is not None,
        has_capacity_analysis_bool,
        has_timing_analysis_bool,
        () if run_variant_node_obj is None else _run_variant_param_tuple(run_variant_node_obj),
    )


def list_strategies() -> list[StrategyEntry]:
    """All ``strategy_*.py`` files under ``strategies/``, wired ones first.

    Sort order: most mature first, then by category, then by name — so the
    strategies you actually trade sit at the top of the catalog, with the
    promoted-but-not-live ones directly beneath them.
    """
    tier_by_module_dict = _tier_by_module_dict()
    entry_list: list[StrategyEntry] = []

    for module_path in sorted(STRATEGIES_ROOT_PATH.rglob("strategy_*.py")):
        module_import_str = _module_import_str(module_path)
        (
            summary_str,
            has_run_variant_bool,
            has_capacity_analysis_bool,
            has_timing_analysis_bool,
            run_variant_param_tuple,
        ) = _parse_strategy_source(str(module_path), module_path.stat().st_mtime_ns)
        category_str = (
            module_path.parent.name if module_path.parent != STRATEGIES_ROOT_PATH else "uncategorized"
        )
        subcategory_str, subcategory_label_str = _subcategory_pair(category_str, module_path.stem)
        entry_list.append(
            StrategyEntry(
                stem_str=module_path.stem,
                display_name_str=prettify_stem(module_path.stem),
                category_str=category_str,
                category_label_str=_category_label(category_str),
                subcategory_str=subcategory_str,
                subcategory_label_str=subcategory_label_str,
                module_import_str=module_import_str,
                rel_path_str=_rel_posix_str(module_path),
                tier_int=int(
                    tier_by_module_dict.get(module_import_str, MaturityTier.RESEARCH)
                ),
                tier_label_str=strategy_registry.TIER_LABEL_DICT[
                    tier_by_module_dict.get(module_import_str, MaturityTier.RESEARCH)
                ],
                has_run_variant_bool=has_run_variant_bool,
                has_capacity_analysis_bool=has_capacity_analysis_bool,
                has_timing_analysis_bool=has_timing_analysis_bool,
                summary_str=summary_str,
                run_variant_param_tuple=run_variant_param_tuple,
            )
        )

    # Most mature first, so the strategies carrying real money lead the catalog
    # and the promoted-but-not-live ones sit directly under them.
    entry_list.sort(
        key=lambda entry_obj: (
            -entry_obj.tier_int,
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


def list_momentum_subcategories() -> list[tuple[str, str, int]]:
    """Momentum UI groups as ``(key, label, strategy_count)`` in display order."""
    count_by_subcategory_dict = {
        subcategory_str: 0 for subcategory_str in MOMENTUM_SUBCATEGORY_LABEL_DICT
    }
    for entry_obj in list_strategies():
        if entry_obj.category_str != MOMENTUM_CATEGORY_STR or entry_obj.subcategory_str is None:
            continue
        count_by_subcategory_dict[entry_obj.subcategory_str] += 1
    return [
        (subcategory_str, label_str, count_by_subcategory_dict[subcategory_str])
        for subcategory_str, label_str in MOMENTUM_SUBCATEGORY_LABEL_DICT.items()
        if count_by_subcategory_dict[subcategory_str] > 0
    ]


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
