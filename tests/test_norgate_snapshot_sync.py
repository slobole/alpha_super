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
from data.norgate_snapshot_store import CAPITALSPECIAL_ADJUSTMENT_STR, MANIFEST_FILE_NAME_STR, write_snapshot_files


PROFILE_STR = "norgate_eod_etf_plus_vix_helper"
SNAPSHOT_DATE_STR = "2024-01-02"
MARKET_TZ = ZoneInfo("America/New_York")


def _price_snapshot_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": SNAPSHOT_DATE_STR,
                "symbol_str": "SPY",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
            },
            {
                "date": SNAPSHOT_DATE_STR,
                "symbol_str": "$VIX",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 13.0,
                "High": 14.0,
                "Low": 12.5,
                "Close": 13.5,
            },
        ]
    )


def _write_snapshot(snapshot_root_path_obj: Path) -> Path:
    return write_snapshot_files(
        snapshot_root_str=str(snapshot_root_path_obj),
        profile_str=PROFILE_STR,
        snapshot_date_str=SNAPSHOT_DATE_STR,
        price_df=_price_snapshot_df(),
        required_symbol_list=["SPY"],
        required_helper_symbol_list=["$VIX"],
        adjustment_mode_map_dict={
            "SPY": CAPITALSPECIAL_ADJUSTMENT_STR,
            "$VIX": CAPITALSPECIAL_ADJUSTMENT_STR,
        },
    )


def _write_release_manifest(releases_root_path_obj: Path, pod_id_str: str = "pod_test") -> None:
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
                "  signal_clock_str: eod_snapshot_ready",
                "  execution_policy_str: next_open_moo",
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
