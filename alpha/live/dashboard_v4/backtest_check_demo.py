"""Synthetic backtest check report for `--demo --demo-backtest` only; the default demo stays byte-identical."""

from datetime import timedelta
import random


def _curve_point_list(row_list, drift_float, seed_int):
    """Backtest on the same days: the live daily moves plus a small cost drift and noise."""
    rng = random.Random(seed_int)
    live_float = test_float = 1.0
    point_list = []
    for row_obj in row_list:
        twr_float = float(row_obj.twr_decimal)
        live_float *= 1 + twr_float
        test_float *= 1 + twr_float + drift_float + rng.gauss(0, 0.0002)
        point_list.append({"date_str": row_obj.market_date_str, "live_float": live_float - 1, "backtest_float": test_float - 1})
    return point_list


def build_demo_report_dict(workspace_dict, snapshot_obj, *, as_of_ts):
    pod_list = []
    row_list_by_account = {}
    for row_obj in sorted(snapshot_obj.row_tuple, key=lambda item_obj: item_obj.market_date_str):
        row_list_by_account.setdefault(row_obj.account_route_str, []).append(row_obj)
    for index_int, row_dict in enumerate(workspace_dict["summary_dict"]["pod_row_dict_list"]):
        account_str = row_dict.get("account_route_str") or "DEMO_" + row_dict["pod_id_str"]
        point_list = _curve_point_list(row_list_by_account.get(account_str, [])[-120:], 0.00008 + 0.00004 * index_int, 11 + index_int)
        session_str = point_list[-1]["date_str"] if point_list else as_of_ts.date().isoformat()
        record_list = [{"session_str": point_dict["date_str"], "status_str": "same"} for point_dict in point_list[-29:]]
        if index_int == 0:
            decision_dict = {"status_str": "differs", "session_str": session_str, "differ_count_int": 2, "row_count_int": 4,
                             "basis_str": "Entries and exits vs the backtest's buys and sells that session.",
                             "data_snapshot_str": session_str, "data_hash_str": "7c1e09a4b2d1",
                             "row_list": [{"asset_str": "XLE", "live_str": "BUY · 10.0%", "backtest_str": "—", "same_bool": False},
                                          {"asset_str": "XLU", "live_str": "—", "backtest_str": "BUY", "same_bool": False},
                                          {"asset_str": "KRE", "live_str": "BUY · 10.0%", "backtest_str": "BUY", "same_bool": True},
                                          {"asset_str": "XLK", "live_str": "SELL", "backtest_str": "SELL", "same_bool": True}]}
            if record_list:
                record_list[-1]["status_str"] = "differs"
        else:
            decision_dict = {"status_str": "same", "session_str": session_str, "differ_count_int": 0, "row_count_int": 3,
                             "basis_str": "Target weights vs the backtest's holdings after the session (±2.5 points).",
                             "data_snapshot_str": session_str, "data_hash_str": "3f9a0c11d7e2",
                             "row_list": [{"asset_str": asset_str, "live_str": weight_str, "backtest_str": weight_str, "same_bool": True}
                                          for asset_str, weight_str in (("QQQ", "40.0%"), ("TLT", "35.0%"), ("GLD", "25.0%"))]}
        gap_float = ((1 + point_list[-1]["live_float"]) / (1 + point_list[-1]["backtest_float"]) - 1) if point_list else None
        pod_list.append({
            "pod_id_str": row_dict["pod_id_str"], "account_route_str": account_str, "release_id_str": row_dict.get("release_id_str") or "demo",
            "session_str": session_str, "decision_dict": decision_dict, "record_list": record_list,
            "reference_status_str": "ok", "reference_reason_str": "",
            "curve_dict": {"point_list": point_list, "start_str": point_list[0]["date_str"] if point_list else None,
                           "gap_float": gap_float, "gap_month_float": gap_float / 3 if gap_float is not None else None},
            "rare_loss_dict": ({"window_int": 3, "live_float": -0.053, "backtest_float": -0.052, "line_float": -0.048,
                                "worst_float": -0.112, "cause_str": "strategy", "beyond_worst_bool": False}
                               if index_int == 1 else None)})
    rng = random.Random(23)
    cost_list = [max(-5.5, min(11.8, rng.gauss(1.6, 2.4) + (rng.uniform(2.5, 8) if rng.random() < .12 else 0))) for _ in range(92)]
    shift_float = 2.1 - sum(cost_list) / len(cost_list)
    cost_list = [value_float + shift_float for value_float in cost_list]
    gate_pod_list = [pod_dict["pod_id_str"] for pod_dict in pod_list[:2]]
    return {"schema_version_str": "backtest_check.v1", "generated_at_utc_str": (as_of_ts - timedelta(hours=1)).isoformat(),
            "as_of_date_str": as_of_ts.date().isoformat(), "pod_list": pod_list, "flex_reason_str": "",
            "cost_gate_dict": {"status_str": "on_track", "fill_count_int": 92, "fill_target_int": 200, "gate_bps_float": 4.0,
                               "average_bps_float": 2.1, "range_low_bps_float": 1.4, "range_high_bps_float": 2.8,
                               "buy_bps_float": 2.2, "sell_bps_float": 2.0, "parking_fill_count_int": 14, "missing_open_count_int": 1,
                               "first_fill_session_str": "2026-11-02", "time_cap_date_str": "2027-05-04",
                               "halt_tracking_str": "Unexplained halts are not counted here yet; check Activity.",
                               "pod_row_list": [{"pod_id_str": pod_id_str, "fill_count_int": count_int, "buy_bps_float": 2.4,
                                                 "sell_bps_float": 1.6, "all_bps_float": 2.0 + 0.2 * index_int, "worst_bps_float": 10.6}
                                                for index_int, (pod_id_str, count_int) in enumerate(zip(gate_pod_list, (51, 41)))],
                               "cost_bps_list": [round(value_float, 4) for value_float in cost_list]}}


def attach_demo_backtest_check(provider_obj, workspace_dict, snapshot_obj, *, now_fn):
    def get_backtest_check_report_dict(*, as_of_ts):
        return build_demo_report_dict(workspace_dict, snapshot_obj, as_of_ts=now_fn())

    provider_obj.get_backtest_check_report_dict = get_backtest_check_report_dict
    if callable(getattr(provider_obj, "get_system_source_dict", None)):
        from alpha.live.dashboard_v4.backtest_check_data import load_report_dict, system_row_dict

        original_fn = provider_obj.get_system_source_dict

        def get_system_source_dict(*args, **kwargs):
            source_dict = dict(original_fn(*args, **kwargs))
            as_of_ts = now_fn()
            source_dict["backtest_check_dict"] = system_row_dict(load_report_dict(provider_obj, as_of_ts=as_of_ts), as_of_ts=as_of_ts)
            return source_dict

        provider_obj.get_system_source_dict = get_system_source_dict
