"""Tests for the Bench research control panel (``alpha.bench``).

These cover the parts that could silently break trading-adjacent workflows or
expose the side-effecting controls: catalog/wired detection, the results
reader, the run-API command wiring + CSRF gating, sandboxed artifact serving,
and the job runner (dedupe + honest restart semantics). No real backtests are
launched — the run API is exercised with a recording stub, and the job runner
with trivial subprocesses.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha.bench import catalog, runs
from alpha.bench.app import (
    REPORT_TOOLTIP_SCRIPT_SHA256_BASE64_STR,
    _recent_strategy_stem_set,
    create_app,
)
from alpha.engine.report import _METRIC_TOOLTIP_SCRIPT_STR


DV2_MODULE_STR = "strategies.dv2.strategy_mr_dv2"
EOM_ZROZ_SPY_SSO_MODULE_STR = "strategies.eom_tlt_vs_spy.strategy_eom_zroz_spy_sso_variant"
SEASONALITY_MODULE_STR = "strategies.seasonality.strategy_seasonality"
ATR_NDX_MODULE_STR = "strategies.momentum.strategy_mo_atr_normalized_ndx"
US_SECTOR_ETF_IBS_DOWNSHOCK_MODULE_STR = (
    "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock"
)
US_SECTOR_ETF_IBS_DOWNSHOCK_NO_XLC_MODULE_STR = (
    "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_no_xlc"
)
US_SECTOR_ETF_IBS_DOWNSHOCK_VOX_IYR_MODULE_STR = (
    "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr"
)


class RecordingJobManager:
    """Stub matching the JobManager surface the app/templates touch."""

    def __init__(self) -> None:
        self.call_list: list[tuple[str, str, list[str]]] = []
        self.cancelled_id_list: list[str] = []

    def submit(self, label_str, target_str, kind_str, command_list):
        self.call_list.append((kind_str, target_str, list(command_list)))

        class _Job:
            job_id_str = "stub"

        return _Job()

    def active_count(self) -> int:
        return 0

    def list_jobs(self):
        return []

    def get_job(self, job_id_str):
        return None

    def cancel(self, job_id_str):
        self.cancelled_id_list.append(job_id_str)
        return True


@pytest.fixture()
def recording_client():
    recording_job_manager = RecordingJobManager()
    app = create_app(job_manager_obj=recording_job_manager)
    token_str = app.config["bench_token_str"]
    return app.test_client(), recording_job_manager, token_str


# ── catalog ────────────────────────────────────────────────────────────────


def test_catalog_lists_strategies_and_flags_wired():
    strategy_entry_list = catalog.list_strategies()
    assert len(strategy_entry_list) > 0

    dv2_entry = catalog.get_strategy_by_module(DV2_MODULE_STR)
    assert dv2_entry is not None
    assert dv2_entry.is_wired_bool is True
    assert dv2_entry.has_run_variant_bool is True  # guards the BOM/cp1252 decode fix
    assert dv2_entry.has_capacity_analysis_bool is True

    eom_zroz_entry = catalog.get_strategy_by_module(EOM_ZROZ_SPY_SSO_MODULE_STR)
    assert eom_zroz_entry is not None
    assert eom_zroz_entry.has_run_variant_bool is True
    assert eom_zroz_entry.has_capacity_analysis_bool is False

    seasonality_entry = catalog.get_strategy_by_module(SEASONALITY_MODULE_STR)
    assert seasonality_entry is not None
    assert seasonality_entry.has_run_variant_bool is True

    sector_etf_entry = catalog.get_strategy_by_module(
        US_SECTOR_ETF_IBS_DOWNSHOCK_MODULE_STR
    )
    assert sector_etf_entry is not None
    assert sector_etf_entry.is_wired_bool is False
    assert sector_etf_entry.has_run_variant_bool is True
    assert sector_etf_entry.has_capacity_analysis_bool is True

    no_xlc_entry = catalog.get_strategy_by_module(
        US_SECTOR_ETF_IBS_DOWNSHOCK_NO_XLC_MODULE_STR
    )
    assert no_xlc_entry is not None
    assert no_xlc_entry.is_wired_bool is False
    assert no_xlc_entry.has_run_variant_bool is True
    assert no_xlc_entry.has_capacity_analysis_bool is True

    vox_iyr_entry = catalog.get_strategy_by_module(
        US_SECTOR_ETF_IBS_DOWNSHOCK_VOX_IYR_MODULE_STR
    )
    assert vox_iyr_entry is not None
    assert vox_iyr_entry.is_wired_bool is False
    assert vox_iyr_entry.has_run_variant_bool is True
    assert vox_iyr_entry.has_capacity_analysis_bool is True

    capacity_entry_list = [
        entry_obj
        for entry_obj in strategy_entry_list
        if entry_obj.has_capacity_analysis_bool
    ]
    assert len(capacity_entry_list) == 31
    assert (
        "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi"
        in {entry_obj.module_import_str for entry_obj in capacity_entry_list}
    )


def test_catalog_handles_non_utf8_sources_without_crashing():
    strategy_entry_list = catalog.list_strategies()
    runnable_count = sum(1 for entry in strategy_entry_list if entry.has_run_variant_bool)
    assert runnable_count >= 7  # at least every wired strategy is runnable


def test_mean_reversion_folder_uses_sector_dispersion_display_label():
    assert catalog._category_label("mean_reversion") == "Sector Dispersion"


@pytest.mark.parametrize(
    ("stem_str", "expected_subcategory_str"),
    [
        ("strategy_mo_atr_normalized_ndx", "atr_normalized_rotation"),
        ("strategy_mo_radge_ndx", "atr_normalized_rotation"),
        ("strategy_mo_smooth_trend_long_sp500", "smooth_trend"),
        ("strategy_mo_spy_sphb_splv_canary", "canary_rotation"),
        ("strategy_mo_mtum_timed_by_mom", "etf_timing_trend"),
        ("strategy_mo_gappers_iwc_close_to_open", "gap_overnight"),
        ("strategy_mo_jt_12_1_top20", "cross_sectional"),
        ("strategy_mo_future_unclassified", "other_allocation"),
    ],
)
def test_momentum_subcategory_classifier(stem_str, expected_subcategory_str):
    assert catalog._momentum_subcategory_str(stem_str) == expected_subcategory_str


def test_momentum_subcategories_cover_the_live_catalog():
    momentum_entry_list = [
        entry_obj
        for entry_obj in catalog.list_strategies()
        if entry_obj.category_str == catalog.MOMENTUM_CATEGORY_STR
    ]
    assert momentum_entry_list
    assert all(entry_obj.subcategory_str for entry_obj in momentum_entry_list)
    assert sum(
        strategy_count_int
        for _subcategory_str, _label_str, strategy_count_int in catalog.list_momentum_subcategories()
    ) == len(momentum_entry_list)

    dv2_entry_obj = catalog.get_strategy_by_module(DV2_MODULE_STR)
    assert dv2_entry_obj is not None
    assert dv2_entry_obj.subcategory_str is None
    assert dv2_entry_obj.subcategory_label_str is None


def test_catalog_parses_both_portfolio_schemas():
    portfolio_by_name = {entry.name_str: entry for entry in catalog.list_portfolios()}

    simple_entry = portfolio_by_name["multipod"]
    assert simple_entry.schema_str == catalog.SCHEMA_SIMPLE_STR
    assert len(simple_entry.pod_tuple) > 0

    manager_entry = portfolio_by_name["ladder_3_growth"]
    assert manager_entry.schema_str == catalog.SCHEMA_MANAGER_STR
    assert manager_entry.capital_float == pytest.approx(150000.0)
    assert len(manager_entry.pod_tuple) == 4


# ── results reader + artifact serving ────────────────────────────────────────


def test_artifact_path_guard_blocks_traversal():
    assert runs.resolve_artifact_path("../../alpha/bench/app.py") is None
    assert runs.resolve_artifact_path("does/not/exist.html") is None


def test_run_index_builds_without_error():
    index_obj = runs.build_strategy_run_index()
    assert isinstance(index_obj.runs_by_run_name_dict, dict)


def test_portfolio_page_finds_runs_written_under_yaml_config_name(monkeypatch, tmp_path):
    results_root_path = tmp_path / "results"
    timestamp_dir_path = (
        results_root_path
        / "research"
        / "portfolio"
        / "multipod_low_risk_no_xlc"
        / "vanilla_backtest"
        / "2026-07-11_220000"
    )
    timestamp_dir_path.mkdir(parents=True)
    (timestamp_dir_path / "report.html").write_text("<h1>report</h1>", encoding="utf-8")

    portfolio_entry_obj = catalog.PortfolioEntry(
        name_str="multipod_low_risk_noXLC",
        config_name_str="multipod_low_risk_no_xlc",
        rel_path_str="portfolios/multipod_low_risk_noXLC.yaml",
        schema_str=catalog.SCHEMA_MANAGER_STR,
        capital_float=100000.0,
        rebalance_str=None,
        pod_tuple=(),
        error_str=None,
    )
    monkeypatch.setattr(runs, "RESULTS_ROOT_PATH", results_root_path)
    monkeypatch.setattr(
        runs,
        "RESEARCH_PORTFOLIO_ROOT_PATH",
        results_root_path / "research" / "portfolio",
    )
    monkeypatch.setattr(catalog, "list_portfolios", lambda: [portfolio_entry_obj])

    client = create_app(job_manager_obj=RecordingJobManager()).test_client()
    html_str = client.get("/portfolios").get_data(as_text=True)

    assert "1 run" in html_str
    assert ">Report</a>" in html_str
    assert "multipod_low_risk_no_xlc/vanilla_backtest/2026-07-11_220000/report.html" in html_str


def _run_entry(
    run_name_str: str,
    timestamp_str: str,
    metadata_dict: dict | None = None,
    activity_timestamp_float: float = 0.0,
) -> runs.RunEntry:
    return runs.RunEntry(
        run_name_str=run_name_str,
        analysis_dir_str="vanilla_backtest",
        analysis_label_str="Vanilla",
        timestamp_str=timestamp_str,
        rel_dir_from_results_str=f"research/strategy/{run_name_str}/vanilla_backtest/{timestamp_str}",
        has_report_bool=True,
        activity_timestamp_float=activity_timestamp_float,
        metadata_dict={} if metadata_dict is None else metadata_dict,
    )


def _capacity_run_entry(
    timestamp_str: str,
    metadata_dict: dict | None = None,
) -> runs.RunEntry:
    return runs.RunEntry(
        run_name_str="strategy_mr_dv2",
        analysis_dir_str="capacity_analysis",
        analysis_label_str="Capacity",
        timestamp_str=timestamp_str,
        rel_dir_from_results_str=(
            "research/strategy/strategy_mr_dv2/capacity_analysis/" + timestamp_str
        ),
        has_report_bool=True,
        metadata_dict={} if metadata_dict is None else metadata_dict,
    )


def test_capacity_run_labels_legacy_and_v2_1_window_dates():
    legacy_run_obj = _capacity_run_entry("2026-07-12_120000")
    current_run_obj = _capacity_run_entry(
        "2026-07-12_110000",
        {
            "model_version_str": "capacity_v2_1",
            "window_date_dict": {
                "recent_5y": {
                    "actual_start_date_str": "2021-07-11",
                    "actual_end_date_str": "2026-07-11",
                },
                "full_history": {
                    "actual_start_date_str": "2004-01-02",
                    "actual_end_date_str": "2026-07-11",
                },
            },
        },
    )

    assert legacy_run_obj.is_legacy_capacity_bool is True
    assert legacy_run_obj.display_analysis_label_str == "Capacity · Legacy v1"
    assert current_run_obj.is_legacy_capacity_bool is False
    assert current_run_obj.display_analysis_label_str == "Capacity · v2.1"
    assert current_run_obj.capacity_window_date_summary_str == (
        "Recent: 2021-07-11 to 2026-07-11 · "
        "Full: 2004-01-02 to 2026-07-11"
    )


def test_strategy_page_prefers_non_legacy_capacity_report(monkeypatch):
    legacy_run_obj = _capacity_run_entry("2026-07-12_120000")
    vanilla_run_obj = _run_entry("strategy_mr_dv2", "2026-07-12_113000")
    current_run_obj = _capacity_run_entry(
        "2026-07-12_110000",
        {
            "model_version_str": "capacity_v2_1",
            "window_date_dict": {
                "recent_5y": {
                    "actual_start_date_str": "2021-07-11",
                    "actual_end_date_str": "2026-07-11",
                },
                "full_history": {
                    "actual_start_date_str": "2004-01-02",
                    "actual_end_date_str": "2026-07-11",
                },
            },
        },
    )
    strategy_entry_obj = catalog.get_strategy_by_module(DV2_MODULE_STR)
    assert strategy_entry_obj is not None
    run_index_obj = SimpleNamespace(
        runs_for=lambda _module_import_str, _stem_str: [
            legacy_run_obj,
            vanilla_run_obj,
            current_run_obj,
        ]
    )
    monkeypatch.setattr(catalog, "get_strategy_by_module", lambda _module_str: strategy_entry_obj)
    monkeypatch.setattr(runs, "build_strategy_run_index", lambda: run_index_obj)

    client = create_app(job_manager_obj=RecordingJobManager()).test_client()
    html_str = client.get(f"/strategy/{DV2_MODULE_STR}").get_data(as_text=True)

    assert current_run_obj.report_artifact_str in html_str
    assert f'src="/artifact/{legacy_run_obj.report_artifact_str}"' not in html_str
    assert "Capacity · v2.1" in html_str
    assert "Recent: 2021-07-11 to 2026-07-11" in html_str
    assert "Full: 2004-01-02 to 2026-07-11" in html_str


def test_run_index_prefers_exact_stem_when_wrapper_metadata_points_to_base_module():
    base_module_str = "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs"
    wrapper_module_str = "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie"
    base_run_obj = _run_entry(
        "strategy_mr_sector_dispersion_ibs",
        "2026-07-05_210000",
        {"class_module": base_module_str},
    )
    wrapper_run_obj = _run_entry(
        "strategy_mr_sector_dispersion_ibs_kie",
        "2026-07-05_220000",
        {"class_module": base_module_str},
    )
    index_obj = runs.StrategyRunIndex(
        runs_by_module_dict={base_module_str: [wrapper_run_obj, base_run_obj]},
        runs_by_run_name_dict={
            base_run_obj.run_name_str: [base_run_obj],
            wrapper_run_obj.run_name_str: [wrapper_run_obj],
        },
        strategy_stem_set={base_run_obj.run_name_str, wrapper_run_obj.run_name_str},
    )

    assert index_obj.runs_for(base_module_str, base_run_obj.run_name_str) == [base_run_obj]
    assert index_obj.runs_for(wrapper_module_str, wrapper_run_obj.run_name_str) == [wrapper_run_obj]


def test_run_index_keeps_non_catalog_parameter_variants_with_the_base_strategy():
    base_module_str = "strategies.momentum.strategy_mo_ev_lrb_252_ndx"
    base_run_obj = _run_entry(
        "strategy_mo_ev_lrb_252_ndx",
        "2026-06-17_230000",
        {"class_module": base_module_str},
    )
    parameter_run_obj = _run_entry(
        "strategy_mo_ev_lrb_252_ndx_positive_ev",
        "2026-06-17_231532",
        {"class_module": base_module_str},
    )
    wrapper_run_obj = _run_entry(
        "strategy_mo_ev_lrb_252_ndx_wrapper",
        "2026-06-17_232000",
        {"class_module": base_module_str},
    )
    index_obj = runs.StrategyRunIndex(
        runs_by_module_dict={
            base_module_str: [wrapper_run_obj, parameter_run_obj, base_run_obj]
        },
        runs_by_run_name_dict={
            base_run_obj.run_name_str: [base_run_obj],
            parameter_run_obj.run_name_str: [parameter_run_obj],
            wrapper_run_obj.run_name_str: [wrapper_run_obj],
        },
        strategy_stem_set={base_run_obj.run_name_str, wrapper_run_obj.run_name_str},
    )

    assert index_obj.runs_for(base_module_str, base_run_obj.run_name_str) == [
        parameter_run_obj,
        base_run_obj,
    ]


def test_run_index_merges_legacy_exact_run_with_metadata_linked_variant():
    base_module_str = "strategies.momentum.strategy_mo_ev_lrb_252_ndx"
    legacy_base_run_obj = _run_entry(
        "strategy_mo_ev_lrb_252_ndx",
        "2026-06-17_230000",
    )
    parameter_run_obj = _run_entry(
        "strategy_mo_ev_lrb_252_ndx_positive_ev",
        "2026-06-17_231532",
        {"class_module": base_module_str},
    )
    index_obj = runs.StrategyRunIndex(
        runs_by_module_dict={base_module_str: [parameter_run_obj]},
        runs_by_run_name_dict={
            legacy_base_run_obj.run_name_str: [legacy_base_run_obj],
            parameter_run_obj.run_name_str: [parameter_run_obj],
        },
        strategy_stem_set={legacy_base_run_obj.run_name_str},
    )

    assert index_obj.runs_for(base_module_str, legacy_base_run_obj.run_name_str) == [
        parameter_run_obj,
        legacy_base_run_obj,
    ]


def test_run_scanner_keeps_artifact_leaves_and_ignores_empty_nested_containers(
    monkeypatch,
    tmp_path,
):
    results_root_path = tmp_path / "results"
    run_name_dir_path = results_root_path / "research" / "strategy" / "sample_strategy"
    analysis_dir_path = run_name_dir_path / "variant_comparison"

    metadata_leaf_path = analysis_dir_path / "2026-07-01_120000"
    metadata_leaf_path.mkdir(parents=True)
    (metadata_leaf_path / "metadata.json").write_text(
        json.dumps({"saved_at": "2026-07-03T12:00:00"}),
        encoding="utf-8",
    )

    legacy_leaf_path = analysis_dir_path / "sp500_2012_core"
    legacy_leaf_path.mkdir()
    legacy_artifact_path = legacy_leaf_path / "comparison_table.csv"
    legacy_artifact_path.write_text("variant,sharpe\ncore,1.1\n", encoding="utf-8")
    legacy_timestamp_float = datetime(2026, 7, 2, 12, 0, 0).timestamp()
    os.utime(legacy_artifact_path, (legacy_timestamp_float, legacy_timestamp_float))

    empty_container_path = analysis_dir_path / "strategy"
    (empty_container_path / "nested").mkdir(parents=True)
    (empty_container_path / "nested" / "artifact.csv").write_text("x\n", encoding="utf-8")

    monkeypatch.setattr(runs, "RESULTS_ROOT_PATH", results_root_path)
    run_entry_list = runs._scan_run_entries(run_name_dir_path, "sample_strategy")

    assert [run_obj.timestamp_str for run_obj in run_entry_list] == [
        "2026-07-01_120000",
        "sp500_2012_core",
    ]
    assert run_entry_list[0].activity_timestamp_float == pytest.approx(
        datetime(2026, 7, 3, 12, 0, 0).timestamp()
    )
    assert run_entry_list[0].display_timestamp_str == "2026-07-03 12:00:00"
    assert run_entry_list[1].activity_timestamp_float == pytest.approx(legacy_timestamp_float)


def test_run_scanner_reads_the_backtest_window_the_runner_recorded(monkeypatch, tmp_path):
    """The tested window must come from run_info.json, never be inferred.

    Two runs of one strategy over different windows are otherwise identical in
    the history table, which makes it impossible to tell a full-history run from
    a truncated one when comparing metrics.
    """
    results_root_path = tmp_path / "results"
    run_name_dir_path = results_root_path / "research" / "strategy" / "sample_strategy"

    windowed_leaf_path = run_name_dir_path / "vanilla_backtest" / "2026-07-01_120000"
    windowed_leaf_path.mkdir(parents=True)
    (windowed_leaf_path / "run_info.json").write_text(
        json.dumps(
            {
                "analysis_type": "vanilla_backtest",
                "parameters": {
                    "capital": 100000.0,
                    "start_date": "2004-01-02",
                    "end_date": "2026-06-30",
                },
            }
        ),
        encoding="utf-8",
    )

    windowless_leaf_path = run_name_dir_path / "stress_test" / "2026-07-02_120000"
    windowless_leaf_path.mkdir(parents=True)
    (windowless_leaf_path / "summary.json").write_text(json.dumps({"sharpe": 1.0}), encoding="utf-8")

    monkeypatch.setattr(runs, "RESULTS_ROOT_PATH", results_root_path)
    run_by_analysis_dict = {
        run_obj.analysis_dir_str: run_obj
        for run_obj in runs._scan_run_entries(run_name_dir_path, "sample_strategy")
    }

    vanilla_run_obj = run_by_analysis_dict["vanilla_backtest"]
    assert vanilla_run_obj.backtest_window_str == "2004-01-02 → 2026-06-30"
    assert vanilla_run_obj.capital_display_str == "100,000"

    # An analysis that recorded no window must report nothing rather than
    # implying it covered full history.
    stress_run_obj = run_by_analysis_dict["stress_test"]
    assert stress_run_obj.backtest_window_str is None
    assert stress_run_obj.capital_display_str is None


def test_run_entry_ignores_a_partial_or_malformed_window():
    """A half-written window is not a window."""
    run_obj = _run_entry("sample", "vanilla_backtest")
    run_obj.run_info_dict = {"parameters": {"start_date": "2004-01-02", "capital": True}}

    assert run_obj.backtest_window_str is None
    # ``True`` is an int subclass; treating it as capital would print "1".
    assert run_obj.capital_display_str is None


def test_recent_feed_uses_activity_order_and_limit():
    older_run_obj = _run_entry("older", "legacy_older", activity_timestamp_float=100.0)
    newest_run_obj = _run_entry("newest", "legacy_newest", activity_timestamp_float=300.0)
    middle_run_obj = _run_entry("middle", "legacy_middle", activity_timestamp_float=200.0)
    index_obj = runs.StrategyRunIndex(
        runs_by_module_dict={},
        runs_by_run_name_dict={
            "older": [older_run_obj],
            "newest": [newest_run_obj],
            "middle": [middle_run_obj],
        },
    )

    assert index_obj.recent_runs(limit_int=2) == [newest_run_obj, middle_run_obj]


def test_recent_feed_excludes_rows_without_metrics_or_report():
    actionable_run_obj = _run_entry(
        "actionable",
        "actionable",
        activity_timestamp_float=100.0,
    )
    dead_end_run_obj = runs.RunEntry(
        run_name_str="dead_end",
        analysis_dir_str="custom_study",
        analysis_label_str="Custom Study",
        timestamp_str="dead_end",
        rel_dir_from_results_str="research/strategy/dead_end/custom_study/dead_end",
        has_report_bool=False,
        activity_timestamp_float=200.0,
    )
    index_obj = runs.StrategyRunIndex(
        runs_by_module_dict={},
        runs_by_run_name_dict={
            "actionable": [actionable_run_obj],
            "dead_end": [dead_end_run_obj],
        },
    )

    assert index_obj.recent_runs(limit_int=8) == [actionable_run_obj]


def test_recent_strategy_set_maps_parameter_variant_to_its_base_strategy():
    base_module_str = "strategies.momentum.strategy_mo_ev_lrb_252_ndx"
    base_run_obj = _run_entry("base", "base", activity_timestamp_float=99.0)
    named_output_run_obj = _run_entry(
        "base_positive_ev",
        "base_positive_ev",
        {"class_module": base_module_str},
        activity_timestamp_float=100.0,
    )
    index_obj = runs.StrategyRunIndex(
        runs_by_module_dict={base_module_str: [named_output_run_obj]},
        runs_by_run_name_dict={
            "base": [base_run_obj],
            "base_positive_ev": [named_output_run_obj],
        },
        strategy_stem_set={"base"},
    )
    base_entry_obj = SimpleNamespace(
        module_import_str=base_module_str,
        stem_str="base",
    )

    assert _recent_strategy_stem_set(
        [base_entry_obj],
        index_obj,
        cutoff_timestamp_float=100.0,
    ) == {"base"}


def test_recent_strategy_set_does_not_leak_wrapper_activity_to_base():
    base_module_str = "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs"
    wrapper_module_str = f"{base_module_str}_kie"
    wrapper_run_obj = _run_entry(
        "wrapper",
        "wrapper",
        {"class_module": base_module_str},
        activity_timestamp_float=200.0,
    )
    index_obj = runs.StrategyRunIndex(
        runs_by_module_dict={base_module_str: [wrapper_run_obj]},
        runs_by_run_name_dict={"wrapper": [wrapper_run_obj]},
        strategy_stem_set={"base", "wrapper"},
    )
    base_entry_obj = SimpleNamespace(module_import_str=base_module_str, stem_str="base")
    wrapper_entry_obj = SimpleNamespace(
        module_import_str=wrapper_module_str,
        stem_str="wrapper",
    )

    assert _recent_strategy_stem_set(
        [base_entry_obj, wrapper_entry_obj],
        index_obj,
        cutoff_timestamp_float=100.0,
    ) == {"wrapper"}


def test_artifact_response_is_sandboxed(monkeypatch, tmp_path):
    report_path = tmp_path / "report.html"
    report_path.write_text("<h1>ok</h1>", encoding="utf-8")
    monkeypatch.setattr(runs, "resolve_artifact_path", lambda rel_path_str: report_path)

    client = create_app(job_manager_obj=RecordingJobManager()).test_client()
    response = client.get("/artifact/anything/report.html")
    assert response.status_code == 200
    csp_str = response.headers.get("Content-Security-Policy", "")
    assert "sandbox" in csp_str
    assert "sandbox allow-scripts" in csp_str
    assert (
        f"script-src 'sha256-{REPORT_TOOLTIP_SCRIPT_SHA256_BASE64_STR}'"
        in csp_str
    )
    assert "style-src" in csp_str and "'unsafe-inline'" in csp_str  # report styling still works
    assert response.headers.get("X-Content-Type-Options") == "nosniff"


def test_bench_allows_only_the_exact_generated_tooltip_script():
    tooltip_script_body_str = _METRIC_TOOLTIP_SCRIPT_STR.split('<script>', 1)[1].split(
        '</script>',
        1,
    )[0]
    tooltip_script_hash_str = base64.b64encode(
        hashlib.sha256(tooltip_script_body_str.encode('utf-8')).digest()
    ).decode('ascii')
    strategy_template_str = Path('alpha/bench/templates/strategy.html').read_text(
        encoding='utf-8'
    )

    assert tooltip_script_hash_str == REPORT_TOOLTIP_SCRIPT_SHA256_BASE64_STR
    assert 'sandbox="allow-scripts"' in strategy_template_str


# ── run API command wiring + CSRF ────────────────────────────────────────────


def test_run_api_builds_single_analysis_command(recording_client):
    client, job_manager, token_str = recording_client
    response = client.post(
        "/api/run", data={"csrf_token": token_str, "module_import": DV2_MODULE_STR, "analysis": "vanilla"}
    )
    assert response.status_code == 302
    _kind, _target, command_list = job_manager.call_list[-1]
    assert command_list[-2:] == ["--analysis", "vanilla"]
    assert command_list[1].endswith("run_strategy_analysis.py")


def test_run_api_full_preset_passes_all_five_with_keep_going(recording_client):
    client, job_manager, token_str = recording_client
    response = client.post(
        "/api/run",
        data={
            "csrf_token": token_str,
            "module_import": DV2_MODULE_STR,
            "analysis": ["vanilla", "capacity", "timing", "risk", "stress"],
        },
    )
    assert response.status_code == 302
    command_list = job_manager.call_list[-1][2]
    assert command_list.count("--analysis") == 5
    assert "--keep-going" in command_list


def test_run_api_rejects_single_capacity_for_strategy_without_hook(recording_client):
    client, job_manager, token_str = recording_client
    response = client.post(
        "/api/run",
        data={
            "csrf_token": token_str,
            "module_import": EOM_ZROZ_SPY_SSO_MODULE_STR,
            "analysis": "capacity",
        },
    )
    assert response.status_code == 400
    assert job_manager.call_list == []


def test_full_preset_still_launches_for_strategy_without_capacity_hook(recording_client):
    client, job_manager, token_str = recording_client
    response = client.post(
        "/api/run",
        data={
            "csrf_token": token_str,
            "module_import": EOM_ZROZ_SPY_SSO_MODULE_STR,
            "analysis": ["vanilla", "capacity", "timing", "risk", "stress"],
        },
    )
    assert response.status_code == 302
    assert "--keep-going" in job_manager.call_list[-1][2]


def test_run_variant_params_are_read_from_the_signature(recording_client):
    """Fields come from each strategy's own run_variant, not an assumption.

    run_strategy.py raises on a kwarg the target does not declare, so offering
    a field the strategy cannot accept would build a job that dies on launch.
    """
    dv2_entry = catalog.get_strategy_by_module(DV2_MODULE_STR)
    param_name_tuple = tuple(p.name_str for p in dv2_entry.run_variant_param_tuple)

    assert "backtest_start_date_str" in param_name_tuple
    assert "end_date_str" in param_name_tuple
    assert "capital_base_float" in param_name_tuple
    # Bench builds the command, so the runner's own plumbing is never offered.
    assert "output_dir_str" not in param_name_tuple
    assert "save_results_bool" not in param_name_tuple

    default_by_name_dict = {
        p.name_str: p.default_repr_str for p in dv2_entry.run_variant_param_tuple
    }
    assert default_by_name_dict["backtest_start_date_str"] == "'2004-01-01'"


def test_only_scalar_kwargs_are_offered_as_fields():
    """A DataFrame cannot be expressed as --strategy-kwarg KEY=VALUE."""
    offered_name_set = {
        param_obj.name_str
        for entry_obj in catalog.list_strategies()
        for param_obj in entry_obj.run_variant_param_tuple
    }
    assert "pricing_data_df" not in offered_name_set
    assert "config" not in offered_name_set
    assert offered_name_set  # the filter did not empty the catalog
    assert all(
        name_str.endswith(catalog.SCALAR_KWARG_SUFFIX_TUPLE) for name_str in offered_name_set
    )


def test_run_api_forwards_declared_kwargs_and_stamps_them_on_the_label(recording_client):
    client, job_manager, token_str = recording_client
    response = client.post(
        "/api/run",
        data={
            "csrf_token": token_str,
            "module_import": DV2_MODULE_STR,
            "analysis": "vanilla",
            "kwarg__backtest_start_date_str": "2015-01-01",
            "kwarg__end_date_str": "2020-12-31",
            "kwarg__capital_base_float": "",  # blank means "use the default"
        },
    )
    assert response.status_code == 302

    _kind_str, _target_str, command_list = job_manager.call_list[-1]
    assert "--strategy-kwarg" in command_list
    assert "backtest_start_date_str=2015-01-01" in command_list
    assert "end_date_str=2020-12-31" in command_list
    # A blank field must not be forwarded as an empty override.
    assert not any(part_str.startswith("capital_base_float=") for part_str in command_list)


def test_run_api_rejects_a_kwarg_the_strategy_does_not_declare(recording_client):
    """Fail here rather than launching a job that dies inside the runner."""
    client, job_manager, token_str = recording_client
    response = client.post(
        "/api/run",
        data={
            "csrf_token": token_str,
            "module_import": DV2_MODULE_STR,
            "analysis": "vanilla",
            "kwarg__not_a_real_param_str": "x",
        },
    )
    assert response.status_code == 400
    assert job_manager.call_list == []


def test_kwarg_blind_analyses_match_the_runner(recording_client):
    """The warning must name the analyses that genuinely ignore the kwargs.

    *** CRITICAL*** This is asserted against the runner's own command builder,
    not against a hand-maintained list. If a future change starts forwarding
    --strategy-kwarg to capacity or stress, the UI warning becomes false and
    this test fails instead of the operator silently comparing a windowed
    vanilla against a full-history capacity run.
    """
    from alpha.bench.app import KWARG_AWARE_ANALYSIS_TUPLE, SUPPORTED_ANALYSIS_TUPLE
    from scripts.research.run_strategy_analysis import _analysis_command_tuple

    forwarding_analysis_set = set()
    for analysis_str in SUPPORTED_ANALYSIS_TUPLE:
        command_tuple = _analysis_command_tuple(
            analysis_str=analysis_str,
            module_import_str=DV2_MODULE_STR,
            output_dir_str="results",
            save_results_bool=True,
            show_display_bool=False,
            show_signal_progress_bool=False,
            performance_warnings_as_errors_bool=False,
            strategy_kwarg_tuple=("probe_marker_str=1",),
        )
        if "probe_marker_str=1" in command_tuple:
            forwarding_analysis_set.add(analysis_str)

    assert forwarding_analysis_set == set(KWARG_AWARE_ANALYSIS_TUPLE)


def test_run_api_rejects_unknown_module(recording_client):
    client, _job_manager, token_str = recording_client
    response = client.post("/api/run", data={"csrf_token": token_str, "module_import": "does.not.exist", "analysis": "vanilla"})
    assert response.status_code == 400


def test_run_api_rejects_empty_analysis(recording_client):
    client, _job_manager, token_str = recording_client
    response = client.post("/api/run", data={"csrf_token": token_str, "module_import": DV2_MODULE_STR})
    assert response.status_code == 400


def test_run_api_rejects_mixed_invalid_analysis(recording_client):
    client, job_manager, token_str = recording_client
    response = client.post(
        "/api/run",
        data={"csrf_token": token_str, "module_import": DV2_MODULE_STR, "analysis": ["vanilla", "bogus"]},
    )
    assert response.status_code == 400
    assert job_manager.call_list == []  # nothing launched


def test_run_api_requires_csrf_token(recording_client):
    client, job_manager, _token_str = recording_client
    response = client.post("/api/run", data={"module_import": DV2_MODULE_STR, "analysis": "vanilla"})
    assert response.status_code == 403
    assert job_manager.call_list == []


def test_run_api_rejects_foreign_origin(recording_client):
    client, job_manager, token_str = recording_client
    response = client.post(
        "/api/run",
        data={"csrf_token": token_str, "module_import": DV2_MODULE_STR, "analysis": "vanilla"},
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 403
    assert job_manager.call_list == []


def test_portfolio_api_routes_by_schema(recording_client):
    client, job_manager, token_str = recording_client

    client.post("/api/run-portfolio", data={"csrf_token": token_str, "config_rel_path": "portfolios/multipod.yaml"})
    assert job_manager.call_list[-1][2][1].endswith("run_portfolio.py")

    client.post(
        "/api/run-portfolio",
        data={"csrf_token": token_str, "config_rel_path": "portfolios/ladder_3_growth.yaml"},
    )
    assert job_manager.call_list[-1][2][1].endswith("run_portfolio_manager.py")


def test_portfolio_api_requires_csrf_token(recording_client):
    client, job_manager, _token_str = recording_client
    response = client.post("/api/run-portfolio", data={"config_rel_path": "portfolios/multipod.yaml"})
    assert response.status_code == 403
    assert job_manager.call_list == []


# ── pages render ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path_str",
    ["/", "/jobs", "/portfolios", "/research", "/healthz", f"/strategy/{DV2_MODULE_STR}"],
)
def test_pages_render(recording_client, path_str):
    client, _job_manager, _token_str = recording_client
    assert client.get(path_str).status_code == 200


def test_variant_switch_sets_the_cookie_and_restyles_the_console(recording_client):
    client, _job_manager, _token_str = recording_client

    default_text_str = client.get("/").get_data(as_text=True)
    assert "#ffffff" in default_text_str  # swiss page

    switch_response = client.get("/variant/blueprint")
    assert switch_response.status_code == 302
    assert "bench_variant=blueprint" in switch_response.headers["Set-Cookie"]

    blueprint_text_str = client.get("/").get_data(as_text=True)
    assert "#16283e" in blueprint_text_str  # blueprint sheet
    assert "--color-ink: #dce8f5" in blueprint_text_str


def test_variant_switch_rejects_an_unknown_variant(recording_client):
    client, _job_manager, _token_str = recording_client
    assert client.get("/variant/not-a-variant").status_code == 404


def test_console_falls_back_when_the_variant_cookie_is_tampered_with(recording_client):
    """An edited cookie must not 500 every page.

    The value reaches build_bench_theme_css, which raises on an unknown
    variant, so it has to be validated against the allowlist first.
    """
    client, _job_manager, _token_str = recording_client
    client.set_cookie("bench_variant", "'; DROP TABLE", domain="localhost")

    response = client.get("/")
    assert response.status_code == 200
    assert "--color-page: #ffffff" in response.get_data(as_text=True)


def test_variant_switch_does_not_follow_a_foreign_referrer(recording_client):
    """The referrer bounce must not become an open redirect."""
    client, _job_manager, _token_str = recording_client

    response = client.get("/variant/journal", headers={"Referer": "https://evil.example/x"})
    assert response.status_code == 302
    assert "evil.example" not in response.headers["Location"]


def test_strategy_page_marks_capacity_unavailable_without_hook(recording_client):
    client, _job_manager, _token_str = recording_client
    response = client.get(f"/strategy/{EOM_ZROZ_SPY_SSO_MODULE_STR}")
    response_text_str = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Capacity unavailable — missing capacity hook" in response_text_str
    assert 'disabled title="Capacity unavailable — missing capacity hook"' in response_text_str


def test_index_renders_momentum_and_recent_run_filters(recording_client):
    client, _job_manager, _token_str = recording_client
    html_str = client.get("/").get_data(as_text=True)

    assert 'data-filter="recent"' in html_str
    assert 'data-filter="subcat:atr_normalized_rotation"' in html_str
    assert '<span class="filter-group-label">Momentum</span>' not in html_str
    assert 'data-filter="cat:mean_reversion">Sector Dispersion</span>' in html_str
    sector_module_str = "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs"
    sector_card_start_int = html_str.index(f'data-module="{sector_module_str}"')
    sector_card_excerpt_str = html_str[sector_card_start_int : sector_card_start_int + 1_500]
    assert '<span class="badge badge-cat">Sector Dispersion</span>' in sector_card_excerpt_str
    sector_detail_html_str = client.get(f"/strategy/{sector_module_str}").get_data(as_text=True)
    assert '<span class="badge badge-cat">Sector Dispersion</span>' in sector_detail_html_str
    atr_card_start_int = html_str.index(f'data-module="{ATR_NDX_MODULE_STR}"')
    atr_card_excerpt_str = html_str[atr_card_start_int : atr_card_start_int + 1_800]
    assert 'data-subcategory="atr_normalized_rotation"' in atr_card_excerpt_str
    assert "ATR-Normalized Rotation" in atr_card_excerpt_str


def test_sortable_metric_rejects_non_numbers_and_keeps_none_distinct_from_zero():
    """A strategy with no Sharpe has not scored zero.

    Sorting must sink it below every measured strategy, so the helper returns
    None — never 0.0 — for anything that is not a real number. bool is an int
    subclass, so a stray True in a summary would otherwise sort as 1.0.
    """
    from alpha.bench.app import _sortable_metric_float

    assert _sortable_metric_float({"sharpe": 1.35}, "sharpe") == pytest.approx(1.35)
    assert _sortable_metric_float({"sharpe": 2}, "sharpe") == pytest.approx(2.0)
    assert _sortable_metric_float({}, "sharpe") is None
    assert _sortable_metric_float({"sharpe": None}, "sharpe") is None
    assert _sortable_metric_float({"sharpe": "1.2"}, "sharpe") is None
    assert _sortable_metric_float({"sharpe": True}, "sharpe") is None


def test_index_splits_tested_rows_from_the_untested_fold(recording_client):
    """Evidence and absence of evidence render as different objects.

    A tested strategy is a dense sortable row carrying its metrics as data
    attributes; the never-run majority is folded into a collapsed section so it
    cannot drown the strategies that carry a track record.
    """
    client, _job_manager, _token_str = recording_client
    html_str = client.get("/").get_data(as_text=True)

    assert 'id="tested-table"' in html_str
    assert 'id="untested-list"' in html_str
    assert 'data-sort-key="sharpe"' in html_str

    # DV2 has recorded vanilla runs, so its row must sit in the tested table
    # with a numeric sharpe to sort on.
    dv2_row_start_int = html_str.index(f'data-module="{DV2_MODULE_STR}"')
    dv2_row_excerpt_str = html_str[dv2_row_start_int : dv2_row_start_int + 1_500]
    assert 'data-sharpe="' in dv2_row_excerpt_str
    untested_start_int = html_str.index('id="untested-list"')
    assert dv2_row_start_int < untested_start_int  # tested section renders first

    # Every strategy appears exactly once across the two sections.
    assert html_str.count(f'data-module="{DV2_MODULE_STR}"') == 1


def test_research_page_lists_only_unreachable_result_folders(recording_client):
    """Orphan studies get a page; attributed runs must not be duplicated onto it.

    A result folder that some strategy page already lists is not an orphan —
    showing it twice would let the same run read as two pieces of evidence.
    """
    orphan_view_list = runs.orphan_research_view_list(catalog.list_strategies())
    orphan_name_set = {view_dict["name_str"] for view_dict in orphan_view_list}

    # Reachable runs stay off the orphan page.
    assert "strategy_mr_dv2" not in orphan_name_set
    # Every orphan really is a results folder, and every one carries runs.
    run_index_obj = runs.build_strategy_run_index()
    assert orphan_name_set <= set(run_index_obj.runs_by_run_name_dict)
    assert all(view_dict["run_entry_list"] for view_dict in orphan_view_list)

    client, _job_manager, _token_str = recording_client
    html_str = client.get("/research").get_data(as_text=True)
    for orphan_name_str in orphan_name_set:
        assert orphan_name_str in html_str


SECTOR_IBS_MODULE_STR = "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs"


def test_compare_page_shows_latest_vanilla_side_by_side(recording_client):
    client, _job_manager, _token_str = recording_client
    html_str = client.get(
        f"/compare?m={DV2_MODULE_STR}&m={SECTOR_IBS_MODULE_STR}"
    ).get_data(as_text=True)

    assert "strategy_mr_dv2" in html_str
    assert "strategy_mr_sector_dispersion_ibs" in html_str
    assert "Sharpe" in html_str
    assert "Backtest window" in html_str


def test_compare_page_rejects_wrong_counts_and_unknown_modules(recording_client):
    client, _job_manager, _token_str = recording_client

    # One strategy is not a comparison.
    assert client.get(f"/compare?m={DV2_MODULE_STR}").status_code == 400
    # Duplicates are deduped before the count check, so twice-the-same is one.
    assert client.get(f"/compare?m={DV2_MODULE_STR}&m={DV2_MODULE_STR}").status_code == 400
    # An unknown module 404s rather than rendering an empty column.
    assert (
        client.get(f"/compare?m={DV2_MODULE_STR}&m=strategies.nope.strategy_missing").status_code
        == 404
    )


def test_index_reuses_one_run_index(recording_client, monkeypatch):
    client, _job_manager, _token_str = recording_client
    original_build_fn = runs.build_strategy_run_index
    build_call_count_int = 0

    def recording_build_fn(*args, **kwargs):
        nonlocal build_call_count_int
        build_call_count_int += 1
        return original_build_fn(*args, **kwargs)

    monkeypatch.setattr(runs, "build_strategy_run_index", recording_build_fn)
    monkeypatch.setattr(
        runs,
        "recent_runs",
        lambda *args, **kwargs: pytest.fail("home page rebuilt the run index"),
    )

    assert client.get("/").status_code == 200
    assert build_call_count_int == 1


# ── job runner ───────────────────────────────────────────────────────────────


def _wait_for_terminal(job_manager, job_id_str, timeout_seconds_float=10.0):
    deadline_float = time.monotonic() + timeout_seconds_float
    while time.monotonic() < deadline_float:
        job_obj = job_manager.get_job(job_id_str)
        if job_obj is not None and not job_obj.is_active_bool:
            return job_obj
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_job_runner_executes_and_records_outcome(monkeypatch, tmp_path):
    from alpha.bench import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR_PATH", tmp_path)
    job_manager = jobs_module.JobManager(max_concurrency_int=2)

    ok_job = job_manager.submit(
        "ok", "ok", "analysis", [sys.executable, "-c", "print('bench-ok'); raise SystemExit(0)"]
    )
    ok_done = _wait_for_terminal(job_manager, ok_job.job_id_str)
    assert ok_done.status_str == jobs_module.STATUS_PASSED_STR
    assert ok_done.return_code_int == 0
    assert ok_done.pid_int is not None  # pid is captured for restart forensics
    assert "bench-ok" in job_manager.read_log_text(ok_job.job_id_str)

    fail_job = job_manager.submit("fail", "fail", "analysis", [sys.executable, "-c", "raise SystemExit(3)"])
    fail_done = _wait_for_terminal(job_manager, fail_job.job_id_str)
    assert fail_done.status_str == jobs_module.STATUS_FAILED_STR
    assert fail_done.return_code_int == 3


def test_job_runner_dedupes_active_duplicates(monkeypatch, tmp_path):
    from alpha.bench import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR_PATH", tmp_path)
    job_manager = jobs_module.JobManager(max_concurrency_int=2)
    sleep_command_list = [sys.executable, "-c", "import time; time.sleep(0.7)"]

    first_job = job_manager.submit("dup", "dup", "analysis", list(sleep_command_list))
    second_job = job_manager.submit("dup", "dup", "analysis", list(sleep_command_list))
    assert second_job.job_id_str == first_job.job_id_str  # deduped while active

    _wait_for_terminal(job_manager, first_job.job_id_str)
    third_job = job_manager.submit("dup", "dup", "analysis", list(sleep_command_list))
    assert third_job.job_id_str != first_job.job_id_str  # a finished job is not a duplicate
    _wait_for_terminal(job_manager, third_job.job_id_str)


def test_cancel_stops_a_running_job_without_calling_it_a_failure(monkeypatch, tmp_path):
    """A cancelled run is stopped, not judged.

    The kill produces a non-zero exit code; recording that as "failed" would put
    a red row against a backtest nobody ever evaluated.
    """
    from alpha.bench import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR_PATH", tmp_path)
    job_manager = jobs_module.JobManager(max_concurrency_int=2)

    long_job = job_manager.submit(
        "long", "long", "analysis", [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    # Wait for the child to actually exist before cancelling it.
    for _attempt_int in range(100):
        if job_manager.get_job(long_job.job_id_str).pid_int is not None:
            break
        time.sleep(0.05)

    assert job_manager.cancel(long_job.job_id_str) is True
    cancelled_job = _wait_for_terminal(job_manager, long_job.job_id_str)
    assert cancelled_job.status_str == jobs_module.STATUS_CANCELLED_STR
    assert cancelled_job.status_str != jobs_module.STATUS_FAILED_STR


def test_cancel_of_a_queued_job_never_launches_it(monkeypatch, tmp_path):
    """Cancelling in the queue must stop the command running at all."""
    from alpha.bench import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR_PATH", tmp_path)
    job_manager = jobs_module.JobManager(max_concurrency_int=1)

    marker_path = tmp_path / "queued-job-ran.txt"
    blocking_job = job_manager.submit(
        "block", "block", "analysis", [sys.executable, "-c", "import time; time.sleep(2)"]
    )
    queued_job = job_manager.submit(
        "queued",
        "queued",
        "analysis",
        [sys.executable, "-c", f"open(r'{marker_path}', 'w').write('ran')"],
    )

    assert job_manager.cancel(queued_job.job_id_str) is True
    assert job_manager.get_job(queued_job.job_id_str).status_str == jobs_module.STATUS_CANCELLED_STR

    _wait_for_terminal(job_manager, blocking_job.job_id_str)
    time.sleep(0.4)  # give the released semaphore slot a chance to misbehave
    assert not marker_path.exists()
    # See the note in test_queue_position_...: no worker may outlive the test.
    assert job_manager.wait_for_workers() is True


def test_cancel_refuses_a_job_that_already_finished(monkeypatch, tmp_path):
    from alpha.bench import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR_PATH", tmp_path)
    job_manager = jobs_module.JobManager(max_concurrency_int=2)

    quick_job = job_manager.submit("quick", "quick", "analysis", [sys.executable, "-c", "pass"])
    _wait_for_terminal(job_manager, quick_job.job_id_str)
    assert job_manager.cancel(quick_job.job_id_str) is False
    assert job_manager.cancel("no-such-job") is False


def test_queue_position_is_reported_and_not_persisted(monkeypatch, tmp_path):
    from alpha.bench import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR_PATH", tmp_path)
    job_manager = jobs_module.JobManager(max_concurrency_int=1)

    job_manager.submit("block", "block", "analysis", [sys.executable, "-c", "import time; time.sleep(1.5)"])
    first_queued_job = job_manager.submit("q1", "q1", "analysis", [sys.executable, "-c", "pass"])
    second_queued_job = job_manager.submit("q2", "q2", "analysis", [sys.executable, "-c", "import sys; sys.exit(0)"])

    position_by_id_dict = {
        job_obj.job_id_str: job_obj.queue_position_int for job_obj in job_manager.list_jobs()
    }
    assert position_by_id_dict[first_queued_job.job_id_str] == 1
    assert position_by_id_dict[second_queued_job.job_id_str] == 2

    # Position is a live fact about this process, so it must not reach the disk.
    sidecar_dict = json.loads(
        (tmp_path / f"{first_queued_job.job_id_str}.json").read_text(encoding="utf-8")
    )
    assert "queue_position_int" not in sidecar_dict

    # *** CRITICAL*** No worker thread may outlive this test. monkeypatch
    # restores JOBS_DIR_PATH on teardown, and a worker still running past that
    # point persists its sidecar into the real results/_bench/jobs/ tree — test
    # jobs showing up in the operator's console. Waiting on job *status* is not
    # enough: a cancelled job reports terminal immediately while its thread is
    # still finishing.
    for job_obj in job_manager.list_jobs():
        job_manager.cancel(job_obj.job_id_str)
    assert job_manager.wait_for_workers() is True


def test_cancel_api_requires_csrf(recording_client):
    client, job_manager, _token_str = recording_client
    response = client.post("/api/jobs/anything/cancel", data={})
    assert response.status_code == 403
    assert job_manager.cancelled_id_list == []


def test_job_view_only_offers_a_report_for_a_job_that_passed(monkeypatch):
    """A failed or cancelled run may have written partial artifacts.

    Offering those behind a Report button would present an abandoned run as a
    finished result.
    """
    from alpha.bench.app import _job_view_dict_list

    called_target_list: list[str] = []

    def fake_find_fn(self, target_str, started_at_float, kind_str):
        called_target_list.append(target_str)
        return SimpleNamespace(report_artifact_str="x/report.html")

    monkeypatch.setattr(runs.ProducedRunFinder, "find_run_produced_after", fake_find_fn)

    job_view_dict_list = _job_view_dict_list(
        [
            SimpleNamespace(
                job_id_str="passed-job", is_active_bool=False,
                status_str="passed", started_at_str="2026-07-01T10:00:00",
                target_str="passed_target", kind_str="analysis",
            ),
            SimpleNamespace(
                job_id_str="failed-job", is_active_bool=False,
                status_str="failed", started_at_str="2026-07-01T10:00:00",
                target_str="failed_target", kind_str="analysis",
            ),
            SimpleNamespace(
                job_id_str="cancelled-job", is_active_bool=False,
                status_str="cancelled", started_at_str="2026-07-01T10:00:00",
                target_str="cancelled_target", kind_str="analysis",
            ),
        ]
    )

    assert called_target_list == ["passed_target"]
    assert job_view_dict_list[0]["produced_run"] is not None
    assert job_view_dict_list[1]["produced_run"] is None
    assert job_view_dict_list[2]["produced_run"] is None


def test_jobs_view_scans_the_results_tree_at_most_once_per_render(monkeypatch):
    """*** CRITICAL*** One scan per render, not one per job.

    Building a fresh run index per job turned the Jobs page into a ~49 s render
    (156 finished jobs x a 0.3 s walk of results/) on a view that polls every
    two seconds — the page simply never loaded. This pins the shape of the fix.
    """
    from alpha.bench.app import _job_view_dict_list

    scan_count_int = 0
    real_build_fn = runs.build_strategy_run_index

    def counting_build_fn(*args, **kwargs):
        nonlocal scan_count_int
        scan_count_int += 1
        return real_build_fn(*args, **kwargs)

    monkeypatch.setattr(runs, "build_strategy_run_index", counting_build_fn)

    job_list = [
        SimpleNamespace(
            job_id_str=f"job-{index_int}",
            status_str="passed",
            started_at_str="2026-07-01T10:00:00",
            target_str=f"target_{index_int}",
            kind_str="analysis",
            is_active_bool=False,
        )
        for index_int in range(25)
    ]

    _job_view_dict_list(job_list)
    assert scan_count_int <= 1


def test_finished_job_report_lookup_is_memoised_across_renders():
    """The polling view must stop touching results/ once answers are known."""
    from alpha.bench.app import _job_view_dict_list

    finished_job_obj = SimpleNamespace(
        job_id_str="job-done",
        status_str="passed",
        started_at_str="2026-07-01T10:00:00",
        target_str="some_target",
        kind_str="analysis",
        is_active_bool=False,
    )
    active_job_obj = SimpleNamespace(
        job_id_str="job-running",
        status_str="running",
        started_at_str="2026-07-01T10:00:00",
        target_str="some_target",
        kind_str="analysis",
        is_active_bool=True,
    )

    produced_run_cache_dict: dict = {}
    _job_view_dict_list([finished_job_obj, active_job_obj], produced_run_cache_dict)

    # The finished job's answer is final and gets remembered; the running one
    # may not have written its artifacts yet, so it must resolve again later.
    assert "job-done" in produced_run_cache_dict
    assert "job-running" not in produced_run_cache_dict


def test_job_runner_marks_stale_jobs_unknown(monkeypatch, tmp_path):
    from alpha.bench import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR_PATH", tmp_path)
    first_manager = jobs_module.JobManager()
    running_like_job = jobs_module.Job(
        job_id_str="20260101-000000-abcd",
        label_str="x",
        target_str="x",
        kind_str="analysis",
        command_list=["python", "-c", "pass"],
        status_str=jobs_module.STATUS_RUNNING_STR,
        created_at_str="2026-01-01T00:00:00",
    )
    first_manager._persist(running_like_job)

    # A fresh manager (simulating a restart) must not claim the job is still
    # running or that it was cleanly interrupted — it cannot know.
    second_manager = jobs_module.JobManager()
    reloaded_job = second_manager.get_job("20260101-000000-abcd")
    assert reloaded_job is not None
    assert reloaded_job.status_str == jobs_module.STATUS_UNKNOWN_STR
