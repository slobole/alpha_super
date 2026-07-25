from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from alpha.live.norgate_snapshot_sync import (
    SYNC_LOCK_FILE_NAME_STR,
    SYNC_STATUS_FILE_NAME_STR,
    build_norgate_snapshot_status_dict,
    ensure_norgate_snapshots_for_live_tick,
)
from alpha.live.release_manifest import load_release_list
from data.norgate_snapshot_store import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    MANIFEST_FILE_NAME_STR,
    SNAPSHOT_SCHEMA_VERSION_INT,
    load_valid_snapshot_manifest,
    write_snapshot_files,
)


PROFILE_STR = "norgate_eod_etf_plus_vix_helper"
SNAPSHOT_DATE_STR = "2024-01-02"
MARKET_TZ = ZoneInfo("America/New_York")


def _price_snapshot_df(snapshot_date_str: str = SNAPSHOT_DATE_STR) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": snapshot_date_str,
                "symbol_str": "SPY",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
            },
            {
                "date": snapshot_date_str,
                "symbol_str": "$VIX",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 13.0,
                "High": 14.0,
                "Low": 12.5,
                "Close": 13.5,
            },
        ]
    )


def _write_snapshot(
    snapshot_root_path_obj: Path,
    snapshot_date_str: str = SNAPSHOT_DATE_STR,
    schema_version_int: int = 1,
) -> Path:
    price_df = _price_snapshot_df(snapshot_date_str)
    if schema_version_int >= SNAPSHOT_SCHEMA_VERSION_INT:
        price_df = price_df.assign(Dividend=0.0)
    return write_snapshot_files(
        snapshot_root_str=str(snapshot_root_path_obj),
        profile_str=PROFILE_STR,
        snapshot_date_str=snapshot_date_str,
        price_df=price_df,
        required_symbol_list=["SPY"],
        required_helper_symbol_list=["$VIX"],
        adjustment_mode_map_dict={
            "SPY": CAPITALSPECIAL_ADJUSTMENT_STR,
            "$VIX": CAPITALSPECIAL_ADJUSTMENT_STR,
        },
        schema_version_int=schema_version_int,
    )


def _write_release_manifest(
    releases_root_path_obj: Path,
    pod_id_str: str = "pod_test",
    signal_clock_str: str = "eod_snapshot_ready",
    execution_policy_str: str = "next_open_moo",
) -> None:
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / f"{pod_id_str}.yaml").write_text(
        "\n".join(
            [
                "identity:",
                f"  release_id: user_001.{pod_id_str}.paper",
                "  user_id: user_001",
                f"  pod_id: {pod_id_str}",
                "deployment:",
                "  mode: paper",
                "  enabled_bool: true",
                "broker:",
                "  account_route: DU1",
                "strategy:",
                "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
                f"  data_profile_str: {PROFILE_STR}",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                f"  signal_clock_str: {signal_clock_str}",
                f"  execution_policy_str: {execution_policy_str}",
                "execution:",
                "  pod_budget_fraction_float: 0.5",
                "  auto_submit_enabled_bool: true",
                "bootstrap:",
                "  initial_cash_float: 10000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )


def _clear_api_env(monkeypatch) -> None:
    for env_name_str in (
        "NORGATE_API_URL",
        "NORGATE_API_HOST",
        "NORGATE_API_PORT",
        "NORGATE_API_TOKEN",
        "NORGATE_CLIENT_ID",
    ):
        monkeypatch.delenv(env_name_str, raising=False)


def _set_snapshot_mode(monkeypatch, snapshot_root_path_obj: Path) -> None:
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(snapshot_root_path_obj))


def test_direct_mode_never_calls_snapshot_sync(tmp_path: Path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(releases_root_path_obj)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")

    def _fail_if_called(**_kwargs):
        raise AssertionError("sync_required_snapshots should not run in direct mode")

    monkeypatch.setattr("alpha.live.norgate_snapshot_sync.sync_required_snapshots", _fail_if_called)

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 1, 2, 16, 10, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert status_dict["status_str"] == "direct"


def test_snapshot_local_ready_skips_api(tmp_path: Path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(releases_root_path_obj)
    _write_snapshot(snapshot_root_path_obj)
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    _clear_api_env(monkeypatch)

    def _fail_if_called(**_kwargs):
        raise AssertionError("sync_required_snapshots should not run when local snapshots are ready")

    monkeypatch.setattr("alpha.live.norgate_snapshot_sync.sync_required_snapshots", _fail_if_called)

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 1, 2, 16, 10, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert status_dict["status_str"] == "ready"
    assert status_dict["snapshot_date_by_profile_dict"] == {PROFILE_STR: SNAPSHOT_DATE_STR}
    assert (snapshot_root_path_obj / SYNC_STATUS_FILE_NAME_STR).exists()


def test_stale_valid_monthly_snapshot_triggers_api_sync(tmp_path: Path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(
        releases_root_path_obj,
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
    )
    _write_snapshot(snapshot_root_path_obj, "2026-06-18")
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    monkeypatch.setenv("NORGATE_API_URL", "http://127.0.0.1:8787")
    monkeypatch.setenv("NORGATE_API_TOKEN", "secret")
    monkeypatch.setenv("NORGATE_CLIENT_ID", "client_test")
    sync_called_bool = False

    def _fake_sync_required_snapshots(**kwargs):
        nonlocal sync_called_bool
        sync_called_bool = True
        assert kwargs["mode_str"] == "paper"
        return [_write_snapshot(snapshot_root_path_obj, "2026-06-30")]

    monkeypatch.setattr(
        "alpha.live.norgate_snapshot_sync.sync_required_snapshots",
        _fake_sync_required_snapshots,
    )

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2026, 7, 1, 8, 0, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert sync_called_bool is True
    assert status_dict["status_str"] == "ready"
    assert status_dict["reason_code_str"] == "sync_ready"
    assert status_dict["snapshot_date_by_profile_dict"][PROFILE_STR] == "2026-06-30"
    assert status_dict["minimum_required_snapshot_date_by_profile_dict"] == {
        PROFILE_STR: "2026-06-30"
    }
    assert status_dict["stale_profile_list"] == []
    assert status_dict["snapshot_fresh_for_cycle_bool"] is True


def test_stale_valid_monthly_snapshot_sync_without_newer_data_waits_and_logs_warn(
    tmp_path: Path,
    monkeypatch,
):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    log_path_obj = tmp_path / "events.jsonl"
    _write_release_manifest(
        releases_root_path_obj,
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
    )
    _write_snapshot(snapshot_root_path_obj, "2026-06-18")
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    monkeypatch.setenv("NORGATE_API_URL", "http://127.0.0.1:8787")
    monkeypatch.setenv("NORGATE_API_TOKEN", "secret")
    monkeypatch.setenv("NORGATE_CLIENT_ID", "client_test")

    def _fake_sync_required_snapshots(**_kwargs):
        return [snapshot_root_path_obj / PROFILE_STR / "2026-06-18"]

    monkeypatch.setattr(
        "alpha.live.norgate_snapshot_sync.sync_required_snapshots",
        _fake_sync_required_snapshots,
    )

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2026, 7, 1, 8, 0, tzinfo=MARKET_TZ),
        log_path_str=str(log_path_obj),
    )

    operator_log_path_obj = log_path_obj.with_name("events_operator.log")
    operator_log_str = operator_log_path_obj.read_text(encoding="utf-8")

    assert status_dict["status_str"] == "waiting"
    assert status_dict["reason_code_str"] == "snapshot_stale_for_cycle"
    assert status_dict["snapshot_fresh_for_cycle_bool"] is False
    assert status_dict["snapshot_stale_past_alert_deadline_bool"] is True
    assert "Required data date: 2026-06-30" in status_dict["operator_message_str"]
    assert "norgate.sync.waiting" in operator_log_str
    assert "WARN" in operator_log_str

    def _fail_if_called_again(**_kwargs):
        raise AssertionError("cooldown should wait before asking the server again")

    monkeypatch.setattr(
        "alpha.live.norgate_snapshot_sync.sync_required_snapshots",
        _fail_if_called_again,
    )
    cooldown_status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2026, 7, 1, 8, 1, tzinfo=MARKET_TZ),
        log_path_str=str(log_path_obj),
    )

    assert cooldown_status_dict["status_str"] == "waiting"
    assert cooldown_status_dict["reason_code_str"] == "sync_waiting_for_newer_snapshot"


def test_stale_valid_monthly_snapshot_without_api_blocks_new_decision_plan(
    tmp_path: Path,
    monkeypatch,
):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(
        releases_root_path_obj,
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
    )
    _write_snapshot(snapshot_root_path_obj, "2026-06-18")
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    _clear_api_env(monkeypatch)

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2026, 7, 1, 8, 0, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert status_dict["status_str"] == "local_snapshot_only"
    assert status_dict["reason_code_str"] == "api_config_missing"
    assert status_dict["snapshot_fresh_for_cycle_bool"] is False
    assert status_dict["stale_profile_list"] == [PROFILE_STR]
    assert "Required data date: 2026-06-30" in status_dict["operator_message_str"]
    assert "Local data date: 2026-06-18" in status_dict["operator_message_str"]


def test_manual_snapshot_recovery_clears_stale_block_on_next_tick(
    tmp_path: Path,
    monkeypatch,
):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(
        releases_root_path_obj,
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
    )
    _write_snapshot(snapshot_root_path_obj, "2026-06-18")
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    _clear_api_env(monkeypatch)
    as_of_ts = datetime(2026, 7, 1, 8, 0, tzinfo=MARKET_TZ)

    blocked_status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=as_of_ts,
        log_path_str=str(tmp_path / "events.jsonl"),
    )
    _write_snapshot(snapshot_root_path_obj, "2026-06-30")
    recovered_status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=as_of_ts,
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert blocked_status_dict["snapshot_fresh_for_cycle_bool"] is False
    assert recovered_status_dict["status_str"] == "ready"
    assert recovered_status_dict["reason_code_str"] == "local_snapshot_ready"
    assert recovered_status_dict["snapshot_fresh_for_cycle_bool"] is True
    assert recovered_status_dict["stale_profile_list"] == []


def test_stale_daily_snapshot_after_close_buffer_triggers_api_sync(
    tmp_path: Path,
    monkeypatch,
):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(releases_root_path_obj)
    _write_snapshot(snapshot_root_path_obj, "2024-01-02")
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    monkeypatch.setenv("NORGATE_API_URL", "http://127.0.0.1:8787")
    monkeypatch.setenv("NORGATE_API_TOKEN", "secret")
    monkeypatch.setenv("NORGATE_CLIENT_ID", "client_test")
    sync_called_bool = False

    def _fake_sync_required_snapshots(**_kwargs):
        nonlocal sync_called_bool
        sync_called_bool = True
        return [_write_snapshot(snapshot_root_path_obj, "2024-01-03")]

    monkeypatch.setattr(
        "alpha.live.norgate_snapshot_sync.sync_required_snapshots",
        _fake_sync_required_snapshots,
    )

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 1, 3, 16, 20, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert sync_called_bool is True
    assert status_dict["status_str"] == "ready"
    assert status_dict["minimum_required_snapshot_date_by_profile_dict"] == {
        PROFILE_STR: "2024-01-03"
    }
    assert status_dict["snapshot_date_by_profile_dict"][PROFILE_STR] == "2024-01-03"


def test_snapshot_missing_without_api_waits_local_only(tmp_path: Path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(releases_root_path_obj)
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    _clear_api_env(monkeypatch)

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 1, 2, 16, 10, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert status_dict["status_str"] == "waiting"
    assert status_dict["reason_code_str"] == "api_config_missing"
    assert "NORGATE_API_TOKEN" in str(status_dict["error_str"])


def test_snapshot_missing_with_api_syncs_and_promotes(tmp_path: Path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(releases_root_path_obj)
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    monkeypatch.setenv("NORGATE_API_URL", "http://127.0.0.1:8787")
    monkeypatch.setenv("NORGATE_API_TOKEN", "secret")
    monkeypatch.setenv("NORGATE_CLIENT_ID", "client_test")

    def _fake_sync_required_snapshots(**kwargs):
        assert kwargs["mode_str"] == "paper"
        assert kwargs["pod_id_str"] is None
        snapshot_dir_path_obj = _write_snapshot(snapshot_root_path_obj)
        return [snapshot_dir_path_obj]

    monkeypatch.setattr(
        "alpha.live.norgate_snapshot_sync.sync_required_snapshots",
        _fake_sync_required_snapshots,
    )

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 1, 2, 16, 10, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert status_dict["status_str"] == "ready"
    assert status_dict["reason_code_str"] == "sync_ready"
    assert (snapshot_root_path_obj / PROFILE_STR / SNAPSHOT_DATE_STR / MANIFEST_FILE_NAME_STR).exists()


def test_snapshot_missing_with_api_syncs_and_validates_schema_v2(
    tmp_path: Path,
    monkeypatch,
):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(releases_root_path_obj)
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    monkeypatch.setenv("NORGATE_API_URL", "http://127.0.0.1:8787")
    monkeypatch.setenv("NORGATE_API_TOKEN", "secret")
    monkeypatch.setenv("NORGATE_CLIENT_ID", "client_test")

    def _fake_sync_required_snapshots(**_kwargs):
        snapshot_dir_path_obj = _write_snapshot(
            snapshot_root_path_obj,
            schema_version_int=SNAPSHOT_SCHEMA_VERSION_INT,
        )
        return [snapshot_dir_path_obj]

    monkeypatch.setattr(
        "alpha.live.norgate_snapshot_sync.sync_required_snapshots",
        _fake_sync_required_snapshots,
    )

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 1, 2, 16, 10, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )
    snapshot_manifest_obj = load_valid_snapshot_manifest(PROFILE_STR)

    assert status_dict["status_str"] == "ready"
    assert (
        snapshot_manifest_obj.manifest_dict["schema_version"]
        == SNAPSHOT_SCHEMA_VERSION_INT
    )


def test_snapshot_sync_lock_busy_skips_api(tmp_path: Path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(releases_root_path_obj)
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    monkeypatch.setenv("NORGATE_API_URL", "http://127.0.0.1:8787")
    monkeypatch.setenv("NORGATE_API_TOKEN", "secret")
    monkeypatch.setenv("NORGATE_CLIENT_ID", "client_test")
    snapshot_root_path_obj.mkdir(parents=True)
    (snapshot_root_path_obj / SYNC_LOCK_FILE_NAME_STR).write_text("busy", encoding="utf-8")

    def _fail_if_called(**_kwargs):
        raise AssertionError("sync_required_snapshots should not run while the lock is busy")

    monkeypatch.setattr("alpha.live.norgate_snapshot_sync.sync_required_snapshots", _fail_if_called)

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 1, 2, 16, 10, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert status_dict["status_str"] == "waiting"
    assert status_dict["reason_code_str"] == "sync_lock_busy"
    assert status_dict["snapshot_fresh_for_cycle_bool"] is False
    assert status_dict["stale_profile_list"] == [PROFILE_STR]
    assert "Local data date: missing" in status_dict["operator_message_str"]


def test_snapshot_sync_cooldown_skips_api(tmp_path: Path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(releases_root_path_obj)
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    monkeypatch.setenv("NORGATE_API_URL", "http://127.0.0.1:8787")
    monkeypatch.setenv("NORGATE_API_TOKEN", "secret")
    monkeypatch.setenv("NORGATE_CLIENT_ID", "client_test")
    snapshot_root_path_obj.mkdir(parents=True)
    (snapshot_root_path_obj / SYNC_STATUS_FILE_NAME_STR).write_text(
        json.dumps(
            {
                "status_str": "failed",
                "last_attempt_utc_str": datetime.now(tz=UTC).isoformat(),
                "error_str": "server offline",
            }
        ),
        encoding="utf-8",
    )

    def _fail_if_called(**_kwargs):
        raise AssertionError("sync_required_snapshots should not run during cooldown")

    monkeypatch.setattr("alpha.live.norgate_snapshot_sync.sync_required_snapshots", _fail_if_called)

    status_dict = ensure_norgate_snapshots_for_live_tick(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 1, 2, 16, 10, tzinfo=MARKET_TZ),
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    assert status_dict["status_str"] == "waiting"
    assert status_dict["reason_code_str"] == "sync_failure_cooldown"
    assert status_dict["snapshot_fresh_for_cycle_bool"] is False
    assert status_dict["stale_profile_list"] == [PROFILE_STR]
    assert "Local data date: missing" in status_dict["operator_message_str"]


def test_dashboard_snapshot_status_ignores_unrelated_global_sync_status(tmp_path: Path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    snapshot_root_path_obj = tmp_path / "snapshots"
    _write_release_manifest(releases_root_path_obj)
    _write_snapshot(snapshot_root_path_obj)
    _set_snapshot_mode(monkeypatch, snapshot_root_path_obj)
    (snapshot_root_path_obj / SYNC_STATUS_FILE_NAME_STR).write_text(
        json.dumps(
            {
                "status_str": "failed",
                "required_profile_list": ["norgate_eod_sp500_pit"],
                "error_str": "other profile failed",
            }
        ),
        encoding="utf-8",
    )
    release_obj = load_release_list(str(releases_root_path_obj))[0]

    status_dict = build_norgate_snapshot_status_dict(
        release_obj,
        datetime(2024, 1, 2, 16, 10, tzinfo=MARKET_TZ),
    )

    assert status_dict["status_str"] == "ready"
    assert status_dict["snapshot_date_str"] == SNAPSHOT_DATE_STR
    assert status_dict["last_error_str"] is None
