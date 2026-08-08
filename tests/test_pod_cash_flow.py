"""pod_cash_flow loader + the flow-adjusted daily math it feeds.

The failure this guards: a deposit read as a monster up-day because
returns were raw equity ratios. Every case pins the end-of-day flow
convention: day D's return is (E_D - F_D) / E_{D-1} - 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha.live.pod_cash_flow import (
    load_flow_by_date_dict,
    net_contribution_through_date_float,
)


def _write_flows(tmp_path: Path, body_str: str) -> Path:
    flows_path = tmp_path / "pod_cash_flows.yaml"
    flows_path.write_text(body_str, encoding="utf-8")
    return flows_path


def test_missing_file_means_no_flows(tmp_path):
    assert load_flow_by_date_dict("pod_x", tmp_path / "absent.yaml") == {}


def test_flows_filter_by_pod_and_sum_same_date(tmp_path):
    flows_path = _write_flows(
        tmp_path,
        """
flows:
  - {pod_id: pod_x, date: 2026-07-01, amount: 10000.0}
  - {pod_id: pod_x, date: 2026-07-01, amount: 5000.0}
  - {pod_id: pod_x, date: 2026-08-01, amount: -2000.0, note: withdrawal}
  - {pod_id: pod_other, date: 2026-07-01, amount: 999.0}
""",
    )
    flow_by_date_dict = load_flow_by_date_dict("pod_x", flows_path)
    assert flow_by_date_dict == {"2026-07-01": 15000.0, "2026-08-01": -2000.0}
    assert net_contribution_through_date_float(flow_by_date_dict, "2026-07-31") == 15000.0
    assert net_contribution_through_date_float(flow_by_date_dict, "2026-08-01") == 13000.0


def test_malformed_entry_fails_loud(tmp_path):
    # Silently dropping a deposit would re-inflate the returns this module
    # exists to fix, so a bad entry must raise, not warn.
    flows_path = _write_flows(
        tmp_path,
        """
flows:
  - {pod_id: pod_x, date: 2026-07-01}
""",
    )
    with pytest.raises(ValueError, match="missing"):
        load_flow_by_date_dict("pod_x", flows_path)


def test_deposit_day_return_is_flow_adjusted(tmp_path, monkeypatch):
    """End-to-end through _build_pod_pnl_dict: a 10k deposit into a 10k pod
    is 0% return, not +100%."""
    import sqlite3

    from alpha.live import dashboard as dashboard_module

    flows_path = _write_flows(
        tmp_path,
        """
flows:
  - {pod_id: pod_x, date: 2026-07-02, amount: 10000.0}
""",
    )
    monkeypatch.setenv("ALPHA_POD_CASH_FLOWS_PATH_STR", str(flows_path))

    connection_obj = sqlite3.connect(":memory:")
    connection_obj.row_factory = sqlite3.Row
    connection_obj.execute(
        """
        CREATE TABLE pod_state_history (
            pod_state_history_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
            pod_id_str TEXT, user_id_str TEXT, account_route_str TEXT,
            position_json_str TEXT, cash_float REAL, total_value_float REAL,
            strategy_state_json_str TEXT, snapshot_stage_str TEXT,
            snapshot_source_str TEXT, updated_timestamp_str TEXT,
            recorded_timestamp_str TEXT
        )
        """
    )
    for date_str, equity_float in [
        ("2026-07-01", 10000.0),
        ("2026-07-02", 20100.0),  # 10k deposit + 100 of trading P&L
        ("2026-07-03", 20301.0),  # plain +1% day
    ]:
        connection_obj.execute(
            """
            INSERT INTO pod_state_history
                (pod_id_str, user_id_str, account_route_str, position_json_str,
                 cash_float, total_value_float, strategy_state_json_str,
                 snapshot_stage_str, snapshot_source_str,
                 updated_timestamp_str, recorded_timestamp_str)
            VALUES (?, 'u', 'r', '{}', 0.0, ?, '{}', 'eod', 'test',
                    ?, ?)
            """,
            ("pod_x", equity_float, f"{date_str}T21:00:00+00:00", f"{date_str}T21:00:00+00:00"),
        )

    class _Release:
        pod_id_str = "pod_x"
        calendar_name_str = "NYSE"
        session_calendar_id_str = "XNYS"

    pnl_dict = dashboard_module._build_pod_pnl_dict(connection_obj, _Release())
    point_by_date = {p["market_date_str"]: p for p in pnl_dict["equity_point_dict_list"]}
    deposit_day = point_by_date["2026-07-02"]
    # *** CRITICAL*** (20100 - 10000) / 10000 - 1 = +1%, not +101%.
    assert deposit_day["daily_pnl_pct_float"] == pytest.approx(0.01)
    assert deposit_day["daily_pnl_float"] == pytest.approx(100.0)
    assert deposit_day["net_contribution_float"] == pytest.approx(10000.0)
    final_day = point_by_date["2026-07-03"]
    # TWR compounds the flow-adjusted days: 1.01 * 1.01 - 1.
    assert final_day["since_start_pnl_pct_float"] == pytest.approx(1.01 * 1.01 - 1.0)
    # Dollars made = equity - baseline - contributions.
    assert final_day["since_start_pnl_float"] == pytest.approx(20301.0 - 10000.0 - 10000.0)
