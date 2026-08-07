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
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha.bench import artifact_view, catalog, runs
from alpha.bench.app import (
    REPORT_TOOLTIP_SCRIPT_SHA256_BASE64_STR,
    _analyzer_view_dict_list,
    _analysis_workspace_dict,
    _analysis_tuple_from_command_list,
    _latest_job_record_by_analysis_dict,
    _summary_status_by_analysis_dict,
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
DV2_SEMIVOL_MODULE_STR = "strategies.dv2.strategy_mr_dv2_semivol_sized"


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
    assert dv2_entry.has_timing_analysis_bool is True

    eom_zroz_entry = catalog.get_strategy_by_module(EOM_ZROZ_SPY_SSO_MODULE_STR)
    assert eom_zroz_entry is not None
    assert eom_zroz_entry.has_run_variant_bool is True
    assert eom_zroz_entry.has_capacity_analysis_bool is False
    assert eom_zroz_entry.has_timing_analysis_bool is True

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
    assert len(capacity_entry_list) == 32
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
    html_str = client.get(
        f"/strategy/{DV2_MODULE_STR}?analysis=capacity"
    ).get_data(as_text=True)

    primary_workspace_html_str = html_str[: html_str.index('<details class="research-tools">')]
    assert f'href="/artifact/{current_run_obj.report_artifact_str}"' in primary_workspace_html_str
    assert f'href="/artifact/{legacy_run_obj.report_artifact_str}"' not in primary_workspace_html_str
    assert "artifact-report-frame" not in html_str
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
    assert "artifact-report-frame" not in strategy_template_str
    assert "html_markup" in strategy_template_str


def test_native_artifact_view_strips_active_content_and_keeps_static_evidence(
    tmp_path,
    monkeypatch,
):
    report_dir_path = tmp_path / "research" / "strategy" / "demo" / "risk_analysis" / "run"
    report_dir_path.mkdir(parents=True)
    (report_dir_path / "report.html").write_text(
        """<html><body><main><h1>Risk report</h1>
        <section class="panel"><h2>Distribution <button type="button" class="metric-help" data-help="P05: loss tail; not a forecast.">i</button></h2>
        <script>window.pwned = true</script>
        <iframe src="https://example.com"></iframe>
        <a href="javascript:alert(1)" onclick="alert(2)">unsafe</a>
        <table><tbody><tr><td style="background-color:#eef2ff">P05</td><td>-12%</td></tr></tbody></table>
        <svg viewBox="0 0 10 10"><line x1="0" y1="0" x2="10" y2="10" stroke="#111111"></line></svg>
        </section></main></body></html>""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runs, "RESULTS_ROOT_PATH", tmp_path)
    run_obj = runs.RunEntry(
        run_name_str="demo",
        analysis_dir_str="risk_analysis",
        analysis_label_str="Risk",
        timestamp_str="run",
        rel_dir_from_results_str="research/strategy/demo/risk_analysis/run",
        has_report_bool=True,
    )

    view_obj = artifact_view.build_artifact_view("risk", run_obj, None)

    assert view_obj is not None
    report_html_str = str(view_obj.selected_tab.html_markup)
    assert "Distribution" in report_html_str
    assert "<table>" in report_html_str
    assert "<svg" in report_html_str
    assert "<script" not in report_html_str
    assert "<iframe" not in report_html_str
    assert "javascript:" not in report_html_str
    assert "onclick" not in report_html_str
    assert 'data-help="P05: loss tail; not a forecast."' in report_html_str


def test_vanilla_native_view_has_all_mockup_sections(tmp_path, monkeypatch):
    report_dir_path = tmp_path / "research" / "strategy" / "demo" / "vanilla_backtest" / "run"
    report_dir_path.mkdir(parents=True)
    heading_str_list = [
        "Equity Curve",
        "Year by Year",
        "Monthly Returns",
        "Relative Performance",
        "Composition",
        "Portfolio Weights",
        "Statistics",
        "Conditional Beta",
        "Open Trades",
        "Closed Trades",
        "Audit & Provenance",
    ]
    plate_html_str = "".join(
        f'<div class="plate" id="plate-{index_int:02d}"><h2>{heading_str}</h2><table><tr><td>{index_int}</td></tr></table></div>'
        for index_int, heading_str in enumerate(heading_str_list, start=1)
    )
    (report_dir_path / "report.html").write_text(
        "<html><body><div class=\"report-shell\">"
        "<header class=\"report-header\"><h1>Demo</h1></header>"
        "<div class=\"spec-masthead\">Period 2020 to 2026</div>"
        "<div class=\"plate-index\">Report contents</div>"
        "<div class=\"headline-comparison\"><table><tr><th>Metric</th><th>Strategy</th><th>SPX</th><th>Delta</th></tr>"
        "<tr><td>CAGR</td><td>12%</td><td>9%</td><td>+3pp</td></tr></table></div>"
        f"{plate_html_str}</div></body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(runs, "RESULTS_ROOT_PATH", tmp_path)
    run_obj = runs.RunEntry(
        run_name_str="demo",
        analysis_dir_str="vanilla_backtest",
        analysis_label_str="Vanilla",
        timestamp_str="run",
        rel_dir_from_results_str="research/strategy/demo/vanilla_backtest/run",
        has_report_bool=True,
        metadata_dict={"class_module": "strategies.demo"},
    )

    view_obj = artifact_view.build_artifact_view("vanilla", run_obj, "statistics")

    assert view_obj is not None
    assert [tab_obj.key_str for tab_obj in view_obj.tab_tuple] == [
        "overview",
        "statistics",
        "composition",
        "trades",
        "audit",
    ]
    assert view_obj.selected_tab.key_str == "statistics"
    assert "Statistics" in str(view_obj.selected_tab.html_markup)
    assert "Conditional Beta" in str(view_obj.selected_tab.html_markup)
    assert "plate-full-equity" in str(view_obj.selected_tab.html_markup)
    assert "plate-full-year-chart" in str(view_obj.selected_tab.html_markup)
    assert "Open Trades" not in str(view_obj.selected_tab.html_markup)
    overview_tab_obj = next(tab_obj for tab_obj in view_obj.tab_tuple if tab_obj.key_str == "overview")
    overview_html_str = str(overview_tab_obj.html_markup)
    assert "Strategy" in overview_html_str
    assert "SPX" in overview_html_str
    assert "Delta" in overview_html_str
    assert "+3pp" in overview_html_str
    assert "Demo" not in overview_html_str
    assert "Period 2020 to 2026" not in overview_html_str
    assert "Report contents" not in overview_html_str
    assert "plate-index" not in overview_html_str
    trades_tab_obj = next(tab_obj for tab_obj in view_obj.tab_tuple if tab_obj.key_str == "trades")
    assert "Statistics" in str(trades_tab_obj.html_markup)
    assert "Open Trades" in str(trades_tab_obj.html_markup)
    assert "Closed Trades" in str(trades_tab_obj.html_markup)
    assert "Audit &amp; Provenance" not in str(trades_tab_obj.html_markup)
    audit_tab_obj = next(tab_obj for tab_obj in view_obj.tab_tuple if tab_obj.key_str == "audit")
    assert "Audit &amp; provenance" in str(audit_tab_obj.html_markup)
    assert "strategies.demo" in str(audit_tab_obj.html_markup)


def test_vanilla_saved_statistics_build_mockup_comparison_without_recomputing_report():
    statistics_html_str = """
    <div class="plate" id="plate-08"><h2>Performance Summary</h2>
      <table><thead><tr><th>Metric</th><th>Strategy</th><th>$SPX</th></tr></thead>
      <tbody>
        <tr><td>Return (Ann.) [%]</td><td>23.77%</td><td>14.98%</td></tr>
        <tr><td>Volatility (Ann.) [%]</td><td>17.32%</td><td>16.86%</td></tr>
        <tr><td>Sharpe Ratio</td><td>1.32</td><td>0.91</td></tr>
        <tr><td>Max. Drawdown [%]</td><td>-17.68%</td><td>-33.79%</td></tr>
        <tr><td>Sortino Ratio</td><td>2.02</td><td>1.26</td></tr>
        <tr><td>CVaR 95% (Daily) [%]</td><td>-2.63%</td><td>-2.54%</td></tr>
      </tbody></table>
    </div>
    """

    comparison_html_str = artifact_view._vanilla_comparison_html_str(statistics_html_str)

    assert "Performance vs benchmark" in comparison_html_str
    assert "CAGR (net)" in comparison_html_str
    assert "$SPX" in comparison_html_str
    assert "+8.79pp" in comparison_html_str
    assert "+0.41" in comparison_html_str
    assert "+16.11pp" in comparison_html_str
    assert "-0.09pp" in comparison_html_str


def test_vanilla_monthly_tables_build_mockup_year_table_from_saved_values():
    header_html_str = "".join(
        f"<th>{label_str}</th>"
        for label_str in (
            "Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
            "Aug", "Sep", "Oct", "Nov", "Dec", "Year", "Vol", "Max DD", "Sharpe",
        )
    )
    strategy_value_str_list = [
        "2025", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        "25.48%", "11.6%", "-6.39%", "2.62",
    ]
    benchmark_value_str_list = [
        "2025", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        "15.89%", "18.8%", "-18.7%", "0.97",
    ]
    strategy_row_html_str = "".join(f"<td>{value_str}</td>" for value_str in strategy_value_str_list)
    benchmark_row_html_str = "".join(f"<td>{value_str}</td>" for value_str in benchmark_value_str_list)
    monthly_html_str = (
        f"<table><tr>{header_html_str}</tr><tr>{strategy_row_html_str}</tr></table>"
        f"<table><tr>{header_html_str}</tr><tr>{benchmark_row_html_str}</tr></table>"
    )

    year_table_html_str = artifact_view._vanilla_year_table_html_str(monthly_html_str)

    assert "Year by year" in year_table_html_str
    assert "2025" in year_table_html_str
    assert "25.48%" in year_table_html_str
    assert "15.89%" in year_table_html_str
    assert "+9.59pp" in year_table_html_str
    assert "-6.39%" in year_table_html_str


def test_vanilla_non_spec_report_falls_back_to_full_native_report(tmp_path, monkeypatch):
    report_dir_path = tmp_path / "research" / "strategy" / "demo" / "vanilla_backtest" / "run"
    report_dir_path.mkdir(parents=True)
    (report_dir_path / "report.html").write_text(
        "<html><body><div class=\"report-shell\"><h1>Demo</h1>"
        "<div class=\"card\"><h2>Statistics</h2><table><tr><td>Sharpe</td><td>1.23</td></tr></table></div>"
        "</div></body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(runs, "RESULTS_ROOT_PATH", tmp_path)
    run_obj = runs.RunEntry(
        run_name_str="demo",
        analysis_dir_str="vanilla_backtest",
        analysis_label_str="Vanilla",
        timestamp_str="run",
        rel_dir_from_results_str="research/strategy/demo/vanilla_backtest/run",
        has_report_bool=True,
    )

    view_obj = artifact_view.build_artifact_view("vanilla", run_obj, None)

    assert view_obj is not None
    assert [tab_obj.key_str for tab_obj in view_obj.tab_tuple] == ["report"]
    report_html_str = str(view_obj.selected_tab.html_markup)
    assert "Statistics" in report_html_str
    assert "Sharpe" in report_html_str
    assert "1.23" in report_html_str


@pytest.mark.parametrize(
    "report_bytes",
    [
        b"\xff\xfe\x00",
        b"<html><body></body></html>",
        b"<html><body><h1>Title only</h1></body></html>",
        b"<html><body><script>window.bad = true</script></body></html>",
    ],
)
def test_strategy_page_distinguishes_unrenderable_saved_report(
    tmp_path, monkeypatch, report_bytes
):
    strategy_entry_obj = catalog.get_strategy_by_module(DV2_MODULE_STR)
    assert strategy_entry_obj is not None
    run_obj = _run_entry("strategy_mr_dv2", "2026-08-07_120000")
    report_path = tmp_path / run_obj.report_artifact_str
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(report_bytes)
    run_index_obj = SimpleNamespace(
        runs_for=lambda _module_import_str, _stem_str: [run_obj]
    )
    monkeypatch.setattr(catalog, "get_strategy_by_module", lambda _module_str: strategy_entry_obj)
    monkeypatch.setattr(runs, "build_strategy_run_index", lambda: run_index_obj)
    monkeypatch.setattr(runs, "RESULTS_ROOT_PATH", tmp_path)

    client = create_app(job_manager_obj=RecordingJobManager()).test_client()
    html_str = client.get(
        f"/strategy/{DV2_MODULE_STR}?analysis=vanilla"
    ).get_data(as_text=True)

    assert "SAVED ARTIFACT" in html_str
    assert "Saved Vanilla report could not be displayed." in html_str
    assert "No saved Vanilla report." not in html_str
    assert f'href="/artifact/{run_obj.report_artifact_str}"' in html_str


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


@pytest.mark.parametrize("analysis_str", ["timing", "stress"])
def test_run_api_rejects_single_timing_family_analysis_without_hook(
    recording_client,
    analysis_str,
):
    client, job_manager, token_str = recording_client
    response = client.post(
        "/api/run",
        data={
            "csrf_token": token_str,
            "module_import": DV2_SEMIVOL_MODULE_STR,
            "analysis": analysis_str,
        },
    )

    assert response.status_code == 400
    assert job_manager.call_list == []


def test_run_api_uses_the_stress_registry_not_the_timing_hook(recording_client):
    client, job_manager, token_str = recording_client
    stress_only_module_str = (
        "strategies.taa_df.strategy_taa_df"
    )

    supported_response = client.post(
        "/api/run",
        data={
            "csrf_token": token_str,
            "module_import": stress_only_module_str,
            "analysis": "stress",
        },
    )
    unsupported_response = client.post(
        "/api/run",
        data={
            "csrf_token": token_str,
            "module_import": EOM_ZROZ_SPY_SSO_MODULE_STR,
            "analysis": "stress",
        },
    )

    assert supported_response.status_code == 302
    assert job_manager.call_list[-1][2][-2:] == ["--analysis", "stress"]
    assert unsupported_response.status_code == 400
    assert len(job_manager.call_list) == 1


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


def test_mockup_shell_uses_fixed_research_sidebar_and_excludes_live(recording_client):
    client, _job_manager, _token_str = recording_client
    html_str = client.get("/").get_data(as_text=True)

    assert 'class="system-topbar"' in html_str
    assert 'class="bench-sidebar"' in html_str
    assert "ALPHA / BENCH" in html_str
    # Pinned to the sidebar's own markup. These previously matched ">Studies</a>"
    # and ">Compare</a>", which the sidebar never emits -- it wraps each label in
    # a <span>. They were passing on unrelated buttons elsewhere on the page.
    assert "<span>Studies</span>" in html_str
    assert "<span>Compare</span>" in html_str
    # The surface switch shows LIVE so the operator can see which of the two
    # surfaces they are on, but Bench must never offer a route into the live
    # book: the segment is an inert, aria-disabled <span>, never a link.
    assert 'aria-disabled="true"' in html_str
    live_index_int = html_str.index(">LIVE<")
    live_element_str = html_str[html_str.rindex("<", 0, live_index_int) : live_index_int]
    assert live_element_str.startswith("<span")
    assert "href" not in live_element_str
    assert "/live" not in html_str
    assert re.search(r"\d{2}:\d{2}:\d{2} (EST|EDT)", html_str)


@pytest.mark.parametrize("line_end_str", ["\n", "\r\n"])
def test_analyzer_job_summary_parser_uses_the_final_summary_table(line_end_str):
    log_text_str = """[1/3] vanilla
PASS
[2/3] capacity
SKIP: missing hook
[3/3] risk
FAIL: analyzer failed

Summary
Analysis  Status  Seconds  Detail
vanilla   PASS       12.4
capacity  SKIP        0.0  missing hook
risk      FAIL        1.2  analyzer failed
""".replace("\n", line_end_str)

    assert _analysis_tuple_from_command_list(
        [
            "python", "runner.py", "--analysis", "vanilla",
            "--analysis", "capacity", "--analysis", "risk",
        ]
    ) == ("vanilla", "capacity", "risk")
    assert _summary_status_by_analysis_dict(log_text_str) == {
        "vanilla": "PASS",
        "capacity": "SKIP",
        "risk": "FAIL",
    }


def test_passed_job_without_summary_is_not_promoted_to_analyzer_pass():
    job_obj = SimpleNamespace(
        kind_str="analysis",
        target_str="strategy_mr_dv2",
        command_list=["python", "runner.py", "--analysis", "vanilla"],
        job_id_str="job-no-summary",
        is_active_bool=False,
        status_str="passed",
        created_at_str="2026-08-06T12:00:00",
    )
    job_manager_obj = SimpleNamespace(
        list_jobs=lambda: [job_obj],
        read_log_text=lambda _job_id_str: "runner exited 0 without a summary table",
    )

    record_dict = _latest_job_record_by_analysis_dict(
        job_manager_obj, "strategy_mr_dv2"
    )

    assert record_dict["vanilla"]["status_str"] == "NOT RUN"


def test_completed_job_end_time_keeps_its_new_artifact_as_explicit_pass():
    strategy_entry_obj = catalog.get_strategy_by_module(DV2_MODULE_STR)
    completed_job_obj = SimpleNamespace(
        kind_str="analysis",
        target_str="strategy_mr_dv2",
        command_list=["python", "runner.py", "--analysis", "vanilla"],
        job_id_str="job-with-report",
        is_active_bool=False,
        status_str="passed",
        created_at_str="2026-08-06T12:00:00",
        started_at_str="2026-08-06T12:00:01",
        ended_at_str="2026-08-06T12:05:00",
    )
    job_manager_obj = SimpleNamespace(
        list_jobs=lambda: [completed_job_obj],
        read_log_text=lambda _job_id_str: (
            "Summary\n"
            "Analysis  Status  Seconds  Detail\n"
            "vanilla   PASS        1.0\n"
        ),
    )
    run_obj = _run_entry(
        "strategy_mr_dv2",
        "2026-08-06_120500",
        activity_timestamp_float=datetime.fromisoformat(
            "2026-08-06T12:05:00.633000"
        ).timestamp(),
    )

    analyzer_view_list = _analyzer_view_dict_list(
        strategy_entry_obj,
        [run_obj],
        job_manager_obj,
    )
    vanilla_view_dict = next(
        view_dict
        for view_dict in analyzer_view_list
        if view_dict["analysis_str"] == "vanilla"
    )

    assert vanilla_view_dict["status_str"] == "PASS"
    assert vanilla_view_dict["detail_str"] == "Latest BENCH job"


def test_overlapping_jobs_choose_latest_completion_not_latest_creation():
    later_created_short_job_obj = SimpleNamespace(
        kind_str="analysis",
        target_str="strategy_mr_dv2",
        command_list=["python", "runner.py", "--analysis", "vanilla"],
        job_id_str="short-fail",
        is_active_bool=False,
        status_str="failed",
        created_at_str="2026-08-06T12:01:00.000000",
        started_at_str="2026-08-06T12:01:01.000000",
        ended_at_str="2026-08-06T12:03:00.000000",
    )
    earlier_created_long_job_obj = SimpleNamespace(
        kind_str="analysis",
        target_str="strategy_mr_dv2",
        command_list=["python", "runner.py", "--analysis", "vanilla"],
        job_id_str="long-pass",
        is_active_bool=False,
        status_str="passed",
        created_at_str="2026-08-06T12:00:00.000000",
        started_at_str="2026-08-06T12:00:01.000000",
        ended_at_str="2026-08-06T12:05:00.000000",
    )
    log_by_id_dict = {
        "short-fail": (
            "Summary\nAnalysis  Status  Seconds  Detail\n"
            "vanilla   FAIL        1.0  failed\n"
        ),
        "long-pass": (
            "Summary\nAnalysis  Status  Seconds  Detail\n"
            "vanilla   PASS        1.0\n"
        ),
    }
    job_manager_obj = SimpleNamespace(
        # Mirrors JobManager.list_jobs(): reverse creation order.
        list_jobs=lambda: [
            later_created_short_job_obj,
            earlier_created_long_job_obj,
        ],
        read_log_text=lambda job_id_str: log_by_id_dict[job_id_str],
    )

    record_dict = _latest_job_record_by_analysis_dict(
        job_manager_obj,
        "strategy_mr_dv2",
    )

    assert record_dict["vanilla"]["status_str"] == "PASS"
    assert record_dict["vanilla"]["job"].job_id_str == "long-pass"


def test_partial_artifact_without_report_is_not_promoted_to_analyzer_pass():
    strategy_entry_obj = catalog.get_strategy_by_module(DV2_MODULE_STR)
    partial_run_obj = _run_entry("strategy_mr_dv2", "2026-08-06_120000")
    partial_run_obj.has_report_bool = False

    analyzer_view_list = _analyzer_view_dict_list(
        strategy_entry_obj,
        [partial_run_obj],
    )
    vanilla_view_dict = next(
        view_dict
        for view_dict in analyzer_view_list
        if view_dict["analysis_str"] == "vanilla"
    )

    assert vanilla_view_dict["status_str"] == "NOT RUN"
    assert vanilla_view_dict["latest_run"] is None


def test_compare_has_a_zero_selection_landing_page(recording_client):
    client, _job_manager, _token_str = recording_client
    html_str = client.get("/compare").get_data(as_text=True)

    assert "Choose 2–5 strategies" in html_str
    assert 'class="compare-picker-check"' in html_str


def test_compare_landing_excludes_partial_vanilla_artifact(monkeypatch):
    strategy_entry_obj = catalog.get_strategy_by_module(DV2_MODULE_STR)
    partial_run_obj = _run_entry("strategy_mr_dv2", "2026-08-06_120000")
    partial_run_obj.has_report_bool = False
    partial_run_obj.summary_dict = {"sharpe": 1.0}

    class _PartialRunIndex:
        def latest_vanilla_for(self, _module_import_str, _stem_str):
            return partial_run_obj

    monkeypatch.setattr(catalog, "list_strategies", lambda: [strategy_entry_obj])
    monkeypatch.setattr(runs, "build_strategy_run_index", lambda **_kwargs: _PartialRunIndex())
    client = create_app(job_manager_obj=RecordingJobManager()).test_client()

    html_str = client.get("/compare").get_data(as_text=True)

    assert "No comparable Vanilla evidence yet" in html_str
    assert 'class="compare-picker-check"' not in html_str


def test_strategy_workspace_exposes_all_analyzers_and_vanilla_depth(recording_client):
    client, _job_manager, _token_str = recording_client
    html_str = client.get(f"/strategy/{DV2_MODULE_STR}").get_data(as_text=True)

    assert "Analyzer contract" in html_str
    for label_str in ("Vanilla", "Capacity", "Timing", "Risk", "Stress"):
        assert label_str in html_str
    for section_str in (
        "Equity",
        "Monthly returns",
        "Composition",
        "Statistics",
        "Trades",
    ):
        assert section_str in html_str
    assert 'class="artifact-meta-grid"' in html_str
    assert 'class="artifact-stat-strip artifact-stat-strip-vanilla"' in html_str
    assert 'class="artifact-report-stage"' in html_str
    assert html_str.index('class="artifact-report-stage"') < html_str.index(
        'class="research-tools"'
    )
    assert '<details class="research-tools">' in html_str


def test_strategy_workspace_selects_analyzer_and_rejects_unknown_key(recording_client):
    client, _job_manager, _token_str = recording_client
    risk_html_str = client.get(
        f"/strategy/{DV2_MODULE_STR}?analysis=risk"
    ).get_data(as_text=True)

    assert re.search(r'class="active">\s*<span>Risk</span>', risk_html_str)
    assert "RISK ANALYSIS" in risk_html_str
    assert client.get(f"/strategy/{DV2_MODULE_STR}?analysis=unknown").status_code == 400


@pytest.mark.parametrize(
    ("analysis_str", "summary_dict", "metadata_dict", "expected_eyebrow_str", "expected_stat_str"),
    [
        ("vanilla", {"ann_return_pct": 12.5, "sharpe": 1.1}, {}, "VANILLA BACKTEST", "CAGR"),
        (
            "capacity",
            {"assessed_order_count_int": 30, "execution_policy_str": "MOO"},
            {},
            "CAPACITY ANALYSIS",
            "RECOMMENDED",
        ),
        (
            "timing",
            {"sharpe": 1.2, "risk_label": "Clean"},
            {"default_entry_timing": "T+1 Open", "default_exit_timing": "T+1 Open"},
            "EXECUTION TIMING ANALYSIS",
            "CVaR 5%",
        ),
        (
            "risk",
            {"simulation_count_int": 10_000},
            {},
            "RISK ANALYSIS",
            "DD P05",
        ),
        (
            "stress",
            {"scenario_count_int": 20},
            {},
            "HISTORICAL STRESS TEST",
            "WORST RETURN",
        ),
    ],
)
def test_saved_analyzer_workspace_has_analysis_specific_summary_strip(
    analysis_str,
    summary_dict,
    metadata_dict,
    expected_eyebrow_str,
    expected_stat_str,
):
    run_obj = runs.RunEntry(
        run_name_str="strategy_example",
        analysis_dir_str=f"{analysis_str}_analysis",
        analysis_label_str=analysis_str.title(),
        timestamp_str="2026-08-05_120000",
        rel_dir_from_results_str=f"research/strategy/strategy_example/{analysis_str}_analysis/2026-08-05_120000",
        has_report_bool=True,
        summary_dict=summary_dict,
        metadata_dict=metadata_dict,
        run_info_dict={"parameters": {"capital": 100_000}},
    )

    workspace_dict = _analysis_workspace_dict(analysis_str, run_obj)

    assert workspace_dict["eyebrow_str"] == expected_eyebrow_str
    assert expected_stat_str in {
        stat_dict["label_str"] for stat_dict in workspace_dict["stat_list"]
    }


def test_risk_workspace_reads_the_current_saved_summary_schema():
    run_obj = runs.RunEntry(
        run_name_str="strategy_example",
        analysis_dir_str="risk_analysis",
        analysis_label_str="Risk",
        timestamp_str="2026-08-05_120000",
        rel_dir_from_results_str="research/strategy/strategy_example/risk_analysis/2026-08-05_120000",
        has_report_bool=True,
        summary_dict={
            "simulation_count_int": 10_000,
            "return_count_int": 3_471,
            "primary_intervals": {
                "max_drawdown_float": {"p05_float": -0.2802444},
                "sharpe_float": {"observed_value_float": 1.2769729},
            },
            "primary_time_underwater_breach_probabilities": {
                "underwater_ge_12m": 0.7296
            },
            "investor_summary": {
                "headline_metric_dict": {
                    "modeled_1y_terminal_p05_block_specific_float": -0.0160475
                }
            },
        },
    )

    workspace_dict = _analysis_workspace_dict("risk", run_obj)
    value_by_label_dict = {
        stat_dict["label_str"]: stat_dict["value_str"]
        for stat_dict in workspace_dict["stat_list"]
    }

    assert value_by_label_dict == {
        "OBS. SHARPE": "1.28",
        "DD P05": "-28.02%",
        "12M+ UNDERWATER": "72.96%",
        "1Y TERMINAL P05": "-1.60%",
        "OBSERVATIONS": "3,471",
    }


@pytest.mark.parametrize(
    ("analysis_str", "metric_label_str"),
    [
        ("vanilla", "SHARPE"),
        ("timing", "SHARPE"),
        ("risk", "OBS. SHARPE"),
        ("stress", "MAX GROSS"),
    ],
)
def test_sparse_saved_summary_preserves_missing_metrics(
    analysis_str,
    metric_label_str,
):
    run_obj = runs.RunEntry(
        run_name_str="strategy_example",
        analysis_dir_str=f"{analysis_str}_analysis",
        analysis_label_str=analysis_str.title(),
        timestamp_str="2026-08-05_120000",
        rel_dir_from_results_str=f"research/strategy/strategy_example/{analysis_str}_analysis/2026-08-05_120000",
        has_report_bool=True,
    )

    workspace_dict = _analysis_workspace_dict(analysis_str, run_obj)
    value_by_label_dict = {
        stat_dict["label_str"]: stat_dict["value_str"]
        for stat_dict in workspace_dict["stat_list"]
    }

    assert value_by_label_dict[metric_label_str] == "—"


def test_workspace_reads_saved_analyzer_dimensions_instead_of_defaults():
    capacity_run_obj = runs.RunEntry(
        run_name_str="strategy_example",
        analysis_dir_str="capacity_analysis",
        analysis_label_str="Capacity",
        timestamp_str="2026-08-05_120000",
        rel_dir_from_results_str="capacity",
        has_report_bool=True,
        summary_dict={"aum_grid_list": [75_000, 2_500_000]},
    )
    timing_run_obj = runs.RunEntry(
        run_name_str="strategy_example",
        analysis_dir_str="execution_timing_analyzer",
        analysis_label_str="Timing",
        timestamp_str="2026-08-05_120000",
        rel_dir_from_results_str="timing",
        has_report_bool=True,
        run_info_dict={
            "parameters": {
                "entry_timing_labels": ["T close", "T+1 open"],
                "exit_timing_labels": ["T+1 open", "T+1 close"],
            }
        },
    )
    stress_run_obj = runs.RunEntry(
        run_name_str="strategy_example",
        analysis_dir_str="stress_test",
        analysis_label_str="Stress",
        timestamp_str="2026-08-05_120000",
        rel_dir_from_results_str="stress",
        has_report_bool=True,
        metadata_dict={"configured_crisis_count": 3},
        run_info_dict={"parameters": {"launch_offsets": [5, 21]}},
    )

    capacity_workspace_dict = _analysis_workspace_dict("capacity", capacity_run_obj)
    timing_workspace_dict = _analysis_workspace_dict("timing", timing_run_obj)
    stress_workspace_dict = _analysis_workspace_dict("stress", stress_run_obj)

    assert capacity_workspace_dict["meta_list"][1]["value_str"] == "$75K → $2.5M"
    assert timing_workspace_dict["meta_list"][0]["value_str"] == (
        "2 entry timings × 2 exit timings"
    )
    assert stress_workspace_dict["meta_list"][0]["value_str"] == (
        "3 crises · 2 launch offsets"
    )


def test_capacity_workspace_distinguishes_missing_from_explicit_not_cleared():
    missing_run_obj = runs.RunEntry(
        run_name_str="strategy_example",
        analysis_dir_str="capacity_analysis",
        analysis_label_str="Capacity",
        timestamp_str="2026-08-05_120000",
        rel_dir_from_results_str="missing",
        has_report_bool=True,
    )
    not_cleared_run_obj = runs.RunEntry(
        run_name_str="strategy_example",
        analysis_dir_str="capacity_analysis",
        analysis_label_str="Capacity",
        timestamp_str="2026-08-05_120000",
        rel_dir_from_results_str="not-cleared",
        has_report_bool=True,
        summary_dict={"recommended_capacity_float": None},
    )

    missing_workspace_dict = _analysis_workspace_dict("capacity", missing_run_obj)
    not_cleared_workspace_dict = _analysis_workspace_dict(
        "capacity", not_cleared_run_obj
    )

    assert missing_workspace_dict["stat_list"][0]["value_str"] == "—"
    assert not_cleared_workspace_dict["stat_list"][0]["value_str"] == "NOT CLEARED"


def test_no_report_workspace_uses_selected_analyzer_status_and_detail():
    workspace_dict = _analysis_workspace_dict(
        "capacity",
        None,
        status_str="SKIP",
        detail_str="Capacity unavailable — missing capacity hook",
    )

    assert workspace_dict["summary_str"] == (
        "Capacity unavailable — missing capacity hook"
    )
    assert workspace_dict["meta_list"][1]["value_str"] == "SKIP"


def test_saved_report_is_historical_evidence_not_current_pass():
    strategy_entry_obj = catalog.get_strategy_by_module(DV2_MODULE_STR)
    analyzer_view_list = _analyzer_view_dict_list(
        strategy_entry_obj,
        [_run_entry("strategy_mr_dv2", "2026-08-05_120000")],
    )
    vanilla_view_dict = next(
        view_dict
        for view_dict in analyzer_view_list
        if view_dict["analysis_str"] == "vanilla"
    )

    assert vanilla_view_dict["status_str"] == "SAVED"


def test_studies_page_marks_skip_as_not_ready(recording_client):
    client, _job_manager, _token_str = recording_client
    html_str = client.get("/research").get_data(as_text=True)

    assert "Analyzer readiness · promoted strategies" in html_str
    assert "SKIP is not ready" in html_str
    assert 'class="readiness-table"' in html_str or "readiness-table" in html_str


def test_console_renders_the_single_house_variant(recording_client):
    """One style, no switcher.

    The variant menu is gone, so the console must still emit a complete desk
    palette from the theme rather than falling through to the stylesheet's
    offline copy.
    """
    client, _job_manager, _token_str = recording_client

    html_str = client.get("/").get_data(as_text=True)
    assert "--color-ink: #16181d" in html_str
    assert "--color-page: #ffffff" in html_str
    assert "--color-accent: #1a73e8" in html_str
    # The retired variants must not be reachable or referenced.
    assert client.get("/variant/blueprint").status_code == 404
    for retired_name_str in ("Blueprint", "Journal", "Swiss"):
        assert retired_name_str not in html_str


def test_density_switch_sets_the_cookie_and_rescales_the_console(recording_client):
    client, _job_manager, _token_str = recording_client

    assert 'data-density="work"' in client.get("/").get_data(as_text=True)

    switch_response = client.get("/density/present")
    assert switch_response.status_code == 302
    assert "bench_density=present" in switch_response.headers["Set-Cookie"]

    assert 'data-density="present"' in client.get("/").get_data(as_text=True)


def test_density_switch_rejects_an_unknown_density(recording_client):
    client, _job_manager, _token_str = recording_client
    assert client.get("/density/not-a-density").status_code == 404


def test_console_falls_back_when_the_density_cookie_is_tampered_with(recording_client):
    """An edited cookie must not reach the rendered attribute.

    The value is echoed into data-density on <html>, so it has to be validated
    against the allowlist rather than trusted.
    """
    client, _job_manager, _token_str = recording_client
    client.set_cookie("bench_density", '"><script>', domain="localhost")

    response = client.get("/")
    html_str = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'data-density="work"' in html_str
    assert "<script>" not in html_str.split("</head>")[0]


def test_density_switch_does_not_follow_a_foreign_referrer(recording_client):
    """The referrer bounce must not become an open redirect."""
    client, _job_manager, _token_str = recording_client

    response = client.get("/density/present", headers={"Referer": "https://evil.example/x"})
    assert response.status_code == 302
    assert "evil.example" not in response.headers["Location"]


def test_strategy_page_marks_capacity_unavailable_without_hook(recording_client):
    client, _job_manager, _token_str = recording_client
    response = client.get(f"/strategy/{EOM_ZROZ_SPY_SSO_MODULE_STR}")
    response_text_str = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Capacity unavailable — missing capacity hook" in response_text_str


def test_strategy_page_names_the_missing_run_variant_hook(recording_client):
    client, _job_manager, _token_str = recording_client
    response_text_str = client.get(
        "/strategy/strategies.alpha19.strategy_mr_alpha19"
    ).get_data(as_text=True)

    assert "Vanilla unavailable — missing run_variant hook" in response_text_str
    assert "Risk unavailable — missing run_variant hook" in response_text_str


def test_strategy_page_explains_registered_but_unrunnable_stress_wrapper(
    recording_client,
):
    client, _job_manager, _token_str = recording_client
    response_text_str = client.get(
        "/strategy/strategies.dv2.strategy_mr_dv2_price_adv_ibs_rsi_exit"
    ).get_data(as_text=True)

    assert "Stress unavailable — missing run_variant hook" in response_text_str


def test_index_renders_momentum_and_recent_run_filters(recording_client):
    client, _job_manager, _token_str = recording_client
    html_str = client.get("/").get_data(as_text=True)

    assert 'data-filter="recent"' in html_str
    # Families live in the catalog's single Family dropdown, not a chip row.
    assert 'value="subcat:atr_normalized_rotation"' in html_str
    assert '<span class="filter-group-label">Momentum</span>' not in html_str
    assert 'value="cat:mean_reversion">Sector Dispersion</option>' in html_str
    sector_module_str = "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs"
    sector_card_start_int = html_str.index(f'data-module="{sector_module_str}"')
    sector_card_excerpt_str = html_str[sector_card_start_int : sector_card_start_int + 1_500]
    # The FAMILY column carries the coarse code; the full label stays on hover.
    assert 'title="Sector Dispersion">MR<' in sector_card_excerpt_str
    sector_detail_html_str = client.get(f"/strategy/{sector_module_str}").get_data(as_text=True)
    assert 'class="artifact-breadcrumb"' in sector_detail_html_str
    atr_card_start_int = html_str.index(f'data-module="{ATR_NDX_MODULE_STR}"')
    atr_card_excerpt_str = html_str[atr_card_start_int : atr_card_start_int + 1_800]
    assert 'data-subcategory="atr_normalized_rotation"' in atr_card_excerpt_str
    assert 'title="ATR-Normalized Rotation">MOM<' in atr_card_excerpt_str


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
