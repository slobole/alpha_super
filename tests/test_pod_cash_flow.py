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
    validate_pod_cash_flow_pod_ids,
)


def _write_flows(tmp_path: Path, body_str: str) -> Path:
    flows_path = tmp_path / "pod_cash_flows.yaml"
    flows_path.write_text(body_str, encoding="utf-8")
    return flows_path


def test_missing_default_file_means_no_flows(tmp_path, monkeypatch):
    from alpha.live import pod_cash_flow as pod_cash_flow_module

    monkeypatch.delenv("ALPHA_POD_CASH_FLOWS_PATH_STR", raising=False)
    monkeypatch.setattr(
        pod_cash_flow_module,
        "DEFAULT_POD_CASH_FLOWS_PATH",
        tmp_path / "absent.yaml",
    )
    assert load_flow_by_date_dict("pod_x") == {}


def test_missing_explicit_file_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        load_flow_by_date_dict("pod_x", tmp_path / "absent.yaml")


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


@pytest.mark.parametrize(
    ("body_str", "error_pattern_str"),
    [
        ("- not-a-mapping\n", "top level must be a mapping"),
        ("flow: []\n", "missing required 'flows' list"),
        ("flows: {}\n", "'flows' must be a list"),
        ("flows: false\n", "'flows' must be a list"),
        (
            "flows:\n  - {pod_id: pod_x, date: not-a-date, amount: 100.0}\n",
            "ISO market date",
        ),
        (
            "flows:\n  - {pod_id: pod_x, date: 2026-07-01, amount: .nan}\n",
            "finite",
        ),
        (
            "flows:\n  - {pod_id: '', date: 2026-07-01, amount: 100.0}\n",
            "non-empty string",
        ),
        (
            "flows:\n  - {pod_id: pod_x, date: 2026-07-01, amount: true}\n",
            "not boolean",
        ),
    ],
)
def test_invalid_flow_file_values_fail_loud(
    tmp_path,
    body_str: str,
    error_pattern_str: str,
):
    flows_path = _write_flows(tmp_path, body_str)
    with pytest.raises(ValueError, match=error_pattern_str):
        load_flow_by_date_dict("pod_x", flows_path)


def test_unknown_pod_id_fails_validation(tmp_path):
    flows_path = _write_flows(
        tmp_path,
        "flows:\n  - {pod_id: pod_typo, date: 2026-07-01, amount: 100.0}\n",
    )
    with pytest.raises(ValueError, match="unknown pod_id"):
        validate_pod_cash_flow_pod_ids({"pod_x"}, flows_path)


def _build_pnl_dict_from_eod_point_list(
    point_list: list[tuple[str, float]],
):
    import sqlite3

    from alpha.live import dashboard as dashboard_module

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
    for date_str, equity_float in point_list:
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

    return dashboard_module._build_pod_pnl_dict(connection_obj, _Release())


def test_deposit_day_return_is_flow_adjusted(tmp_path, monkeypatch):
    """End-to-end through _build_pod_pnl_dict: a 10k deposit into a 10k pod
    is 0% return, not +100%."""
    flows_path = _write_flows(
        tmp_path,
        """
flows:
  - {pod_id: pod_x, date: 2026-07-02, amount: 10000.0}
""",
    )
    monkeypatch.setenv("ALPHA_POD_CASH_FLOWS_PATH_STR", str(flows_path))
    pnl_dict = _build_pnl_dict_from_eod_point_list(
        [
            ("2026-07-01", 10000.0),
            ("2026-07-02", 20100.0),  # 10k deposit + 100 of trading P&L
            ("2026-07-06", 20301.0),  # next market session, plain +1% day
        ]
    )
    point_by_date = {p["market_date_str"]: p for p in pnl_dict["equity_point_dict_list"]}
    deposit_day = point_by_date["2026-07-02"]
    # *** CRITICAL*** (20100 - 10000) / 10000 - 1 = +1%, not +101%.
    assert deposit_day["daily_pnl_pct_float"] == pytest.approx(0.01)
    assert deposit_day["daily_pnl_float"] == pytest.approx(100.0)
    assert deposit_day["net_contribution_float"] == pytest.approx(10000.0)
    final_day = point_by_date["2026-07-06"]
    # TWR compounds the flow-adjusted days: 1.01 * 1.01 - 1.
    assert final_day["since_start_pnl_pct_float"] == pytest.approx(1.01 * 1.01 - 1.0)
    # Dollars made = equity - baseline - contributions.
    assert final_day["since_start_pnl_float"] == pytest.approx(20301.0 - 10000.0 - 10000.0)


def test_flow_between_eod_snapshots_is_applied_to_the_next_interval(tmp_path, monkeypatch):
    flows_path = _write_flows(
        tmp_path,
        "flows:\n  - {pod_id: pod_x, date: 2026-07-02, amount: 10000.0}\n",
    )
    monkeypatch.setenv("ALPHA_POD_CASH_FLOWS_PATH_STR", str(flows_path))

    pnl_dict = _build_pnl_dict_from_eod_point_list(
        [
            ("2026-07-01", 10000.0),
            ("2026-07-06", 20100.0),
        ]
    )
    final_point_dict = pnl_dict["equity_point_dict_list"][-1]
    # *** CRITICAL*** The 2 July flow belongs to the observed 1-6 July interval.
    # Dollar P&L can be netted exactly, but without the 2 July EOD NetLiq the
    # chain-linked return is unknowable and must not be presented as exact TWR.
    assert final_point_dict["flow_float"] == pytest.approx(10000.0)
    assert final_point_dict["daily_pnl_float"] == pytest.approx(100.0)
    assert final_point_dict["daily_pnl_pct_float"] is None
    assert final_point_dict["since_start_pnl_pct_float"] is None
    assert final_point_dict["flow_timing_approximation_bool"] is True
    assert final_point_dict["interval_session_count_int"] == 2


def test_first_date_flow_is_baseline_and_future_flow_is_ignored(tmp_path, monkeypatch):
    flows_path = _write_flows(
        tmp_path,
        """
flows:
  - {pod_id: pod_x, date: 2026-07-01, amount: 10000.0}
  - {pod_id: pod_x, date: 2026-07-10, amount: 50000.0}
""",
    )
    monkeypatch.setenv("ALPHA_POD_CASH_FLOWS_PATH_STR", str(flows_path))

    pnl_dict = _build_pnl_dict_from_eod_point_list(
        [("2026-07-01", 10000.0), ("2026-07-02", 10100.0)]
    )
    final_point_dict = pnl_dict["equity_point_dict_list"][-1]
    assert final_point_dict["flow_float"] == pytest.approx(0.0)
    assert final_point_dict["net_contribution_float"] == pytest.approx(0.0)
    assert final_point_dict["daily_pnl_float"] == pytest.approx(100.0)
    assert final_point_dict["daily_pnl_pct_float"] == pytest.approx(0.01)


def test_multiple_interval_flows_are_summed_once(tmp_path, monkeypatch):
    flows_path = _write_flows(
        tmp_path,
        """
flows:
  - {pod_id: pod_x, date: 2026-07-02, amount: 10000.0}
  - {pod_id: pod_x, date: 2026-07-02, amount: -2500.0}
""",
    )
    monkeypatch.setenv("ALPHA_POD_CASH_FLOWS_PATH_STR", str(flows_path))

    pnl_dict = _build_pnl_dict_from_eod_point_list(
        [("2026-07-01", 10000.0), ("2026-07-02", 17600.0)]
    )
    final_point_dict = pnl_dict["equity_point_dict_list"][-1]
    assert final_point_dict["flow_float"] == pytest.approx(7500.0)
    assert final_point_dict["net_contribution_float"] == pytest.approx(7500.0)
    assert final_point_dict["daily_pnl_float"] == pytest.approx(100.0)


def test_combined_curves_keep_flows_from_non_common_eod_dates():
    from alpha.live.dashboard import (
        _build_combined_carry_forward_point_dict_list,
        _build_combined_strict_point_dict_list,
    )

    pod_book_dict_list = [
        {
            "pod_id_str": "pod_alpha",
            "session_calendar_id_str": "XNYS",
            "point_by_market_date_dict": {
                "2026-07-01": {
                    "equity_float": 10000.0,
                    "net_contribution_float": 0.0,
                },
                "2026-07-02": {
                    "equity_float": 20100.0,
                    "net_contribution_float": 10000.0,
                },
                "2026-07-03": {
                    "equity_float": 20301.0,
                    "net_contribution_float": 10000.0,
                },
            },
        },
        {
            "pod_id_str": "pod_beta",
            "session_calendar_id_str": "XNYS",
            "point_by_market_date_dict": {
                "2026-07-01": {
                    "equity_float": 10000.0,
                    "net_contribution_float": 0.0,
                },
                "2026-07-03": {
                    "equity_float": 10200.0,
                    "net_contribution_float": 0.0,
                },
            },
        },
    ]

    strict_point_dict_list = _build_combined_strict_point_dict_list(
        pod_book_dict_list
    )
    carry_point_dict_list = _build_combined_carry_forward_point_dict_list(
        pod_book_dict_list
    )

    # *** CRITICAL*** The strict curve skips 2 July, but its next common
    # interval still deducts the contribution declared on that skipped date.
    assert strict_point_dict_list[-1]["flow_float"] == pytest.approx(10000.0)
    assert strict_point_dict_list[-1]["daily_pnl_float"] == pytest.approx(501.0)
    carry_point_by_date_dict = {
        point_dict["market_date_str"]: point_dict
        for point_dict in carry_point_dict_list
    }
    assert carry_point_by_date_dict["2026-07-02"]["flow_float"] == pytest.approx(
        10000.0
    )
    assert carry_point_by_date_dict["2026-07-03"]["flow_float"] == pytest.approx(0.0)


def test_carry_forward_treats_new_pod_equity_as_external_baseline():
    from alpha.live.dashboard import _build_combined_carry_forward_point_dict_list

    pod_book_dict_list = [
        {
            "pod_id_str": "pod_alpha",
            "session_calendar_id_str": "XNYS",
            "point_by_market_date_dict": {
                "2026-07-01": {"equity_float": 10000.0, "net_contribution_float": 0.0},
                "2026-07-02": {"equity_float": 10100.0, "net_contribution_float": 0.0},
            },
        },
        {
            "pod_id_str": "pod_beta",
            "session_calendar_id_str": "XNYS",
            "point_by_market_date_dict": {
                "2026-07-02": {"equity_float": 20000.0, "net_contribution_float": 0.0},
            },
        },
    ]

    final_point_dict = _build_combined_carry_forward_point_dict_list(
        pod_book_dict_list
    )[-1]
    assert final_point_dict["new_pod_baseline_flow_float"] == pytest.approx(20000.0)
    assert final_point_dict["flow_float"] == pytest.approx(20000.0)
    assert final_point_dict["daily_pnl_float"] == pytest.approx(100.0)
    assert final_point_dict["daily_pnl_pct_float"] == pytest.approx(0.01)


def test_strict_pod_contribution_subtracts_external_flow():
    from alpha.live.dashboard import _strict_pod_daily_pnl_float

    point_by_market_date_dict = {
        "2026-07-01": {"equity_float": 10000.0, "net_contribution_float": 0.0},
        "2026-07-02": {"equity_float": 20100.0, "net_contribution_float": 10000.0},
    }
    assert _strict_pod_daily_pnl_float(
        point_by_market_date_dict,
        "2026-07-02",
        "2026-07-01",
    ) == pytest.approx(100.0)
