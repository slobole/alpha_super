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


class RecordingJobManager:
    """Stub matching the JobManager surface the app/templates touch."""

    def __init__(self) -> None:
        self.call_list: list[tuple[str, str, list[str]]] = []

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

    eom_zroz_entry = catalog.get_strategy_by_module(EOM_ZROZ_SPY_SSO_MODULE_STR)
    assert eom_zroz_entry is not None
    assert eom_zroz_entry.has_run_variant_bool is True

    seasonality_entry = catalog.get_strategy_by_module(SEASONALITY_MODULE_STR)
    assert seasonality_entry is not None
    assert seasonality_entry.has_run_variant_bool is True


def test_catalog_handles_non_utf8_sources_without_crashing():
    strategy_entry_list = catalog.list_strategies()
    runnable_count = sum(1 for entry in strategy_entry_list if entry.has_run_variant_bool)
    assert runnable_count >= 7  # at least every wired strategy is runnable


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

    manager_entry = portfolio_by_name["current_multipod_all"]
    assert manager_entry.schema_str == catalog.SCHEMA_MANAGER_STR
    assert manager_entry.capital_float == pytest.approx(200000.0)
    assert len(manager_entry.pod_tuple) == 4


# ── results reader + artifact serving ────────────────────────────────────────


def test_artifact_path_guard_blocks_traversal():
    assert runs.resolve_artifact_path("../../alpha/bench/app.py") is None
    assert runs.resolve_artifact_path("does/not/exist.html") is None


def test_run_index_builds_without_error():
    index_obj = runs.build_strategy_run_index()
    assert isinstance(index_obj.runs_by_run_name_dict, dict)


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
            "analysis": ["vanilla", "friction", "timing", "risk", "stress"],
        },
    )
    assert response.status_code == 302
    command_list = job_manager.call_list[-1][2]
    assert command_list.count("--analysis") == 5
    assert "--keep-going" in command_list


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
        data={"csrf_token": token_str, "config_rel_path": "portfolios/current_multipod_all.yaml"},
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
    ["/", "/jobs", "/portfolios", "/healthz", f"/strategy/{DV2_MODULE_STR}"],
)
def test_pages_render(recording_client, path_str):
    client, _job_manager, _token_str = recording_client
    assert client.get(path_str).status_code == 200


def test_index_renders_momentum_and_recent_run_filters(recording_client):
    client, _job_manager, _token_str = recording_client
    html_str = client.get("/").get_data(as_text=True)

    assert 'data-filter="recent"' in html_str
    assert 'data-filter="subcat:atr_normalized_rotation"' in html_str
    atr_card_start_int = html_str.index(f'data-module="{ATR_NDX_MODULE_STR}"')
    atr_card_excerpt_str = html_str[atr_card_start_int : atr_card_start_int + 1_200]
    assert 'data-subcategory="atr_normalized_rotation"' in atr_card_excerpt_str
    assert "ATR-Normalized Rotation" in atr_card_excerpt_str


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
