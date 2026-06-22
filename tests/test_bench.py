"""Tests for the Bench research control panel (``alpha.bench``).

These cover the parts that could silently break trading-adjacent workflows or
expose the side-effecting controls: catalog/wired detection, the results
reader, the run-API command wiring + CSRF gating, sandboxed artifact serving,
and the job runner (dedupe + honest restart semantics). No real backtests are
launched — the run API is exercised with a recording stub, and the job runner
with trivial subprocesses.
"""

from __future__ import annotations

import sys
import time

import pytest

from alpha.bench import catalog, runs
from alpha.bench.app import create_app


DV2_MODULE_STR = "strategies.dv2.strategy_mr_dv2"
EOM_ZROZ_SPY_SSO_MODULE_STR = "strategies.eom_tlt_vs_spy.strategy_eom_zroz_spy_sso_variant"
SEASONALITY_MODULE_STR = "strategies.seasonality.strategy_seasonality"


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


def test_artifact_response_is_sandboxed(monkeypatch, tmp_path):
    report_path = tmp_path / "report.html"
    report_path.write_text("<h1>ok</h1>", encoding="utf-8")
    monkeypatch.setattr(runs, "resolve_artifact_path", lambda rel_path_str: report_path)

    client = create_app(job_manager_obj=RecordingJobManager()).test_client()
    response = client.get("/artifact/anything/report.html")
    assert response.status_code == 200
    csp_str = response.headers.get("Content-Security-Policy", "")
    assert "sandbox" in csp_str
    assert "allow-scripts" not in csp_str  # JS stays blocked
    assert "style-src" in csp_str and "'unsafe-inline'" in csp_str  # report styling still works
    assert response.headers.get("X-Content-Type-Options") == "nosniff"


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
