"""The promotion gate: cheap checks that keep a maturity tier honest.

A tier that is earned by editing a list rots within a month. These are the
checks fast enough to run on every commit, so a strategy cannot claim a tier it
no longer satisfies. The expensive half — that capital is honoured, the
benchmark is declared truthfully, and runs are deterministic — lives in
``scripts/research/check_pm_readiness.py`` because it needs real backtests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from alpha import strategy_registry
from alpha.engine import portfolio_manager
from alpha.live import release_manifest
from alpha.strategy_registry import MaturityTier


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]

# What PortfolioManager calls on every fresh-run pod. A strategy missing any of
# these cannot be allocated to, so claiming PM_READY without them is a lie the
# operator only discovers when a book dies mid-run.
PM_RUN_VARIANT_PARAM_TUPLE = (
    "show_display_bool",
    "save_results_bool",
    "output_dir_str",
    "backtest_start_date_str",
    "capital_base_float",
    "end_date_str",
)


def _module_path(strategy_import_str: str) -> Path:
    module_str = strategy_registry.module_import_str(strategy_import_str)
    return REPO_ROOT_PATH / Path(*module_str.split(".")).with_suffix(".py")


def _module_ast(strategy_import_str: str) -> ast.Module:
    source_str = _module_path(strategy_import_str).read_bytes().decode(
        "utf-8-sig", errors="replace"
    )
    return ast.parse(source_str)


def _run_variant_node(module_ast: ast.Module):
    return next(
        (
            node_obj
            for node_obj in module_ast.body
            if isinstance(node_obj, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node_obj.name == "run_variant"
        ),
        None,
    )


PM_READY_IMPORT_TUPLE = strategy_registry.pm_ready_import_tuple()
WIRED_IMPORT_TUPLE = strategy_registry.wired_import_tuple()


def test_registry_is_not_empty():
    assert PM_READY_IMPORT_TUPLE
    assert WIRED_IMPORT_TUPLE


def test_wired_is_a_subset_of_pm_ready():
    """The invariant a hand-kept list broke: wired but not portfolio-eligible.

    Two HPI strategies were live-wired while the portfolio engine refused to
    load them. Deriving both lists from one ordered tier makes the state
    unwritable; this asserts it stays that way.
    """
    assert set(WIRED_IMPORT_TUPLE) <= set(PM_READY_IMPORT_TUPLE)


def test_portfolio_manager_allowlist_is_the_registry():
    assert set(portfolio_manager.SUPPORTED_STRATEGY_IMPORT_TUPLE) == set(
        PM_READY_IMPORT_TUPLE
    )


def test_release_manifest_allowlist_matches_the_registry():
    """The live list is still declared in its own module, so assert they agree.

    Deriving it there is a follow-up; until then this test is what catches a
    strategy being wired for live without being registered, which is exactly
    how the two lists drifted apart in the first place.
    """
    assert set(release_manifest.SUPPORTED_STRATEGY_IMPORT_TUPLE) == set(
        WIRED_IMPORT_TUPLE
    )


@pytest.mark.parametrize("strategy_import_str", PM_READY_IMPORT_TUPLE)
def test_registered_strategy_file_exists(strategy_import_str):
    assert _module_path(strategy_import_str).is_file()


@pytest.mark.parametrize("strategy_import_str", PM_READY_IMPORT_TUPLE)
def test_registered_class_reference_resolves(strategy_import_str):
    """A ``module:Class`` entry must name a class the module actually defines."""
    if ":" not in strategy_import_str:
        return
    class_name_str = strategy_import_str.split(":", maxsplit=1)[1]
    defined_name_set = {
        node_obj.name
        for node_obj in _module_ast(strategy_import_str).body
        if isinstance(node_obj, ast.ClassDef)
    }
    assert class_name_str in defined_name_set


@pytest.mark.parametrize("strategy_import_str", PM_READY_IMPORT_TUPLE)
def test_pm_ready_strategy_exposes_the_common_run_variant(strategy_import_str):
    """PortfolioManager calls run_variant with these six on every fresh run."""
    run_variant_node = _run_variant_node(_module_ast(strategy_import_str))
    assert run_variant_node is not None, f"{strategy_import_str} has no run_variant"

    parameter_set = {
        arg_obj.arg
        for arg_obj in list(run_variant_node.args.args)
        + list(run_variant_node.args.kwonlyargs)
    }
    missing_list = [
        parameter_str
        for parameter_str in PM_RUN_VARIANT_PARAM_TUPLE
        if parameter_str not in parameter_set
    ]
    assert not missing_list, f"{strategy_import_str} is missing {missing_list}"


def test_unregistered_strategy_defaults_to_research():
    assert (
        strategy_registry.tier_for("strategies.nowhere.strategy_made_up")
        is MaturityTier.RESEARCH
    )


def test_tier_resolves_from_the_module_path_alone():
    """Bench discovers files, so it knows the module but not the class."""
    assert (
        strategy_registry.tier_for("strategies.dv2.strategy_mr_dv2")
        is MaturityTier.WIRED
    )
    assert (
        strategy_registry.tier_for("strategies.dv2.strategy_mr_dv2:DVO2Strategy")
        is MaturityTier.WIRED
    )


def test_tiers_are_ordered_so_a_floor_can_be_expressed():
    assert MaturityTier.RESEARCH < MaturityTier.PM_READY < MaturityTier.WIRED
    assert set(strategy_registry.strategy_import_tuple_at_least(MaturityTier.RESEARCH)) == set(
        strategy_registry.STRATEGY_TIER_DICT
    )


def test_every_tier_has_a_display_label():
    for tier_obj in MaturityTier:
        assert strategy_registry.TIER_LABEL_DICT[tier_obj]


def test_pod_minimum_capital_keys_are_registered():
    """A funding floor for a strategy no book can hold would never apply."""
    for strategy_import_str in portfolio_manager.POD_MINIMUM_ALLOCATED_CAPITAL_FLOAT_DICT:
        assert strategy_registry.tier_for(strategy_import_str) >= MaturityTier.PM_READY
