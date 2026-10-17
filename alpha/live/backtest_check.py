"""Nightly backtest check: is each LIVE pod still the strategy we tested? (report only)

    python -m alpha.live.backtest_check --mode live [--pod-id POD] [--no-reference] [--refresh-ruler]

For every enabled LIVE pod it writes one JSON report next to the watchdog report
(`<event log folder>/backtest_check_latest.json`). Dashboard V4 reads that file.

What each pod gets:
- **Decision check.** The research backtest is run on the same days as the pod (from
  its first target session, with its first budget) and its trades / holdings for the
  latest live session are compared with the live decision: symbols and target
  weights, never shares (live sizes from pre-open broker NAV, the backtest from its
  own ledger, so shares always differ).
- **Same-days curve.** Daily live account TWR (IBKR Flex) next to the backtest's
  daily return on the same days, and the gap between them.
- **Rare loss.** The live 1, 3 and 5 session loss against the backtest's own
  history: rare = worse than 99 of 100 windows of that length. Needs a cached ruler
  (`--refresh-ruler`, one long backtest per pod).
- **MR cost gate.** Stock fills of the MR capsule pods against the official open,
  per side, toward the 200-fill / 4.0 bps rule. BIL parking fills are not counted.

Boundaries (keep them):
- Read-only toward live state: the pod databases are opened with SQLite mode=ro,
  the Flex store through `client_reporting` (mode=ro). No LiveStateStore is built,
  no release is synced, no broker connection is opened, nothing is submitted.
- It writes only its own report and its own cache folder.
- Backtests run sequentially, one pod at a time. TAA-family references are skipped
  in the month-end window, because the TAA data loader may rewrite the shared DTB3
  cache that the live month-end decision also reads.
- Nothing here blocks orders or sends alerts.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import time
import traceback
from zoneinfo import ZoneInfo

SCHEMA_VERSION_STR = "backtest_check.v1"
REPORT_FILE_NAME_STR = "backtest_check_latest.json"
CACHE_DIR_NAME_STR = "backtest_check_cache"
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")

GATE_FILL_TARGET_INT = 200
GATE_BPS_FLOAT = 4.0
GATE_MIN_FILLS_FOR_AVERAGE_INT = 30
GATE_TIME_CAP_DAYS_INT = 183
MR_CAPSULE_IMPORT_PREFIX_STR = "strategies.mr_capsule."
PARKING_ASSET_SET = frozenset({"BIL", "SPMO"})
TAA_IMPORT_PREFIX_STR = "strategies.taa_df."

# The backtest's holdings table carries a cash line; it is not a decision symbol.
NON_ASSET_KEY_SET = frozenset({"Cash", "cash", "CASH"})
WEIGHT_MIN_FLOAT = 0.005
# Full books compare targets with the backtest's weights at that session's close, which carry the day's move
# (a -9% day on a 50% sleeve moves it ~3 points). Only a clear difference counts.
WEIGHT_TOLERANCE_FLOAT = 0.05
RARE_WINDOW_TUPLE = (1, 3, 5)
RARE_QUANTILE_FLOAT = 0.01
STRATEGY_SHARE_FLOAT = 0.7
RECORD_LIMIT_INT = 400
CURVE_LIMIT_INT = 400
RULER_MAX_AGE_DAYS_INT = 35
FLEX_QUERY_NAME_STR = "ALPHA_DAILY_TWR"


# ---------------------------------------------------------------------------------------------- small helpers
def _finite_float(value_obj) -> float | None:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def session_date_str(timestamp_str: str) -> str | None:
    """NY market date of a saved ISO timestamp (target execution time)."""
    try:
        timestamp_ts = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp_ts.tzinfo is None:
        return None
    return timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()


def _json_dict(text_str) -> dict:
    try:
        value_obj = json.loads(text_str or "{}")
    except (TypeError, ValueError):
        return {}
    return value_obj if isinstance(value_obj, dict) else {}


def _json_list(text_str) -> list:
    try:
        value_obj = json.loads(text_str or "[]")
    except (TypeError, ValueError):
        return []
    return value_obj if isinstance(value_obj, list) else []


def _connect_ro(db_path_str: str) -> sqlite3.Connection:
    # *** CRITICAL*** read-only URI: this job must never create, migrate or write a pod database.
    return sqlite3.connect(Path(db_path_str).resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)


# ---------------------------------------------------------------------------------------------- pod database (read-only)
def read_pod_source_dict(db_path_str: str, pod_id_str: str, as_of_date_str: str) -> dict:
    """First VPlan, latest executed VPlan on or before as_of, its DecisionPlan, and every fill of the pod."""
    with closing(_connect_ro(db_path_str)) as connection_obj:
        connection_obj.execute("BEGIN")
        vplan_list = [
            {"vplan_id_int": row[0], "decision_plan_id_int": row[1], "session_date_str": session_date_str(row[2]),
             "pod_budget_float": _finite_float(row[3]), "status_str": row[4]}
            for row in connection_obj.execute(
                "SELECT vplan_id_int, decision_plan_id_int, target_execution_timestamp_str, pod_budget_float, status_str "
                "FROM vplan WHERE pod_id_str = ? ORDER BY vplan_id_int", (pod_id_str,))
        ]
        vplan_list = [row for row in vplan_list if row["session_date_str"]]
        executed_list = [row for row in vplan_list if row["session_date_str"] <= as_of_date_str
                         and row["status_str"] not in {None, "ready", "blocked", "expired", "cancelled"}]
        latest_dict = executed_list[-1] if executed_list else None
        decision_dict = None
        if latest_dict is not None:
            row = connection_obj.execute(
                "SELECT decision_book_type_str, entry_target_weight_json_str, full_target_weight_json_str, "
                "exit_asset_json_str, snapshot_metadata_json_str FROM decision_plan WHERE decision_plan_id_int = ?",
                (latest_dict["decision_plan_id_int"],)).fetchone()
            if row is not None:
                decision_dict = {"decision_book_type_str": row[0], "entry_target_weight_map_dict": _json_dict(row[1]),
                                 "full_target_weight_map_dict": _json_dict(row[2]), "exit_asset_list": _json_list(row[3]),
                                 "snapshot_metadata_dict": _json_dict(row[4])}
        fill_list = [
            {"vplan_id_int": row[0], "asset_str": row[1], "fill_amount_float": _finite_float(row[2]),
             "fill_price_float": _finite_float(row[3]), "official_open_price_float": _finite_float(row[4]),
             "fill_timestamp_str": row[5], "session_date_str": row[6]}
            for row in connection_obj.execute(
                "SELECT f.vplan_id_int, f.asset_str, f.fill_amount_float, f.fill_price_float, f.official_open_price_float, "
                "f.fill_timestamp_str, v.target_execution_timestamp_str FROM vplan_fill f "
                "JOIN vplan v ON v.vplan_id_int = f.vplan_id_int WHERE v.pod_id_str = ? ORDER BY f.fill_record_id_int",
                (pod_id_str,))
        ]
        for fill_dict in fill_list:
            fill_dict["session_date_str"] = session_date_str(fill_dict["session_date_str"])
        connection_obj.rollback()
    return {"first_vplan_dict": vplan_list[0] if vplan_list else None, "latest_vplan_dict": latest_dict,
            "decision_dict": decision_dict, "fill_list": fill_list}


# ---------------------------------------------------------------------------------------------- MR cost gate
def fill_cost_bps(fill_dict) -> float | None:
    """Positive = cost: bought above the official open or sold below it (same sign as runner slippage)."""
    amount_float, price_float = fill_dict.get("fill_amount_float"), fill_dict.get("fill_price_float")
    open_float = fill_dict.get("official_open_price_float")
    if amount_float is None or price_float is None or open_float is None or open_float <= 0 or amount_float == 0:
        return None
    direction_float = 1.0 if amount_float > 0 else -1.0
    return 10000.0 * direction_float * (price_float / open_float - 1.0)


def cost_gate_dict(fill_list_by_pod_dict: dict[str, list[dict]], *, as_of_date_str: str) -> dict:
    stock_list, parking_int, missing_open_int = [], 0, 0
    pod_row_list = []
    for pod_id_str, fill_list in sorted(fill_list_by_pod_dict.items()):
        pod_cost_list, pod_buy_list, pod_sell_list = [], [], []
        for fill_dict in fill_list:
            if fill_dict["asset_str"] in PARKING_ASSET_SET:
                parking_int += 1
                continue
            cost_float = fill_cost_bps(fill_dict)
            if cost_float is None:
                missing_open_int += 1
                continue
            stock_list.append({"pod_id_str": pod_id_str, "cost_bps_float": cost_float,
                               "side_str": "BUY" if fill_dict["fill_amount_float"] > 0 else "SELL",
                               "session_date_str": fill_dict["session_date_str"]})
            pod_cost_list.append(cost_float)
            (pod_buy_list if fill_dict["fill_amount_float"] > 0 else pod_sell_list).append(cost_float)
        pod_row_list.append({"pod_id_str": pod_id_str, "fill_count_int": len(pod_cost_list),
                             "buy_bps_float": statistics.mean(pod_buy_list) if pod_buy_list else None,
                             "sell_bps_float": statistics.mean(pod_sell_list) if pod_sell_list else None,
                             "all_bps_float": statistics.mean(pod_cost_list) if pod_cost_list else None,
                             "worst_bps_float": max(pod_cost_list) if pod_cost_list else None})
    count_int = len(stock_list)
    cost_list = [item["cost_bps_float"] for item in stock_list]
    avg_float = statistics.mean(cost_list) if cost_list else None
    half_float = 1.96 * statistics.stdev(cost_list) / math.sqrt(count_int) if count_int >= 2 else None
    session_list = sorted({item["session_date_str"] for item in stock_list if item["session_date_str"]})
    first_str = session_list[0] if session_list else None
    cap_str = (date.fromisoformat(first_str) + timedelta(days=GATE_TIME_CAP_DAYS_INT)).isoformat() if first_str else None
    if count_int >= GATE_FILL_TARGET_INT:
        status_str = "passed" if avg_float is not None and avg_float <= GATE_BPS_FLOAT else "failed"
    elif cap_str is not None and as_of_date_str > cap_str:
        status_str = "time_cap"
    elif count_int < GATE_MIN_FILLS_FOR_AVERAGE_INT:
        status_str = "collecting"
    else:
        status_str = "above" if avg_float > GATE_BPS_FLOAT else "on_track"
    return {
        "status_str": status_str, "fill_count_int": count_int, "fill_target_int": GATE_FILL_TARGET_INT,
        "gate_bps_float": GATE_BPS_FLOAT, "average_bps_float": avg_float,
        "range_low_bps_float": None if half_float is None else avg_float - half_float,
        "range_high_bps_float": None if half_float is None else avg_float + half_float,
        "buy_bps_float": statistics.mean([i["cost_bps_float"] for i in stock_list if i["side_str"] == "BUY"]) if any(i["side_str"] == "BUY" for i in stock_list) else None,
        "sell_bps_float": statistics.mean([i["cost_bps_float"] for i in stock_list if i["side_str"] == "SELL"]) if any(i["side_str"] == "SELL" for i in stock_list) else None,
        "parking_fill_count_int": parking_int, "missing_open_count_int": missing_open_int,
        "first_fill_session_str": first_str, "time_cap_date_str": cap_str,
        "pod_row_list": pod_row_list, "cost_bps_list": [round(value_float, 4) for value_float in cost_list],
        "halt_tracking_str": "Unexplained halts are not counted here yet; check Activity.",
    }


# ---------------------------------------------------------------------------------------------- decision check
def _clean_weight_map(weight_map_dict, *, skip_set=frozenset()) -> dict[str, float]:
    result_dict = {}
    for asset_str, weight_obj in (weight_map_dict or {}).items():
        weight_float = _finite_float(weight_obj)
        if (weight_float is not None and abs(weight_float) >= WEIGHT_MIN_FLOAT and str(asset_str) not in skip_set
                and str(asset_str) not in NON_ASSET_KEY_SET):
            result_dict[str(asset_str)] = weight_float
    return result_dict


def compare_decision_dict(decision_dict: dict, *, reference_weight_map_dict: dict, reference_trade_map_dict: dict,
                          skip_set=frozenset()) -> dict:
    """Compare the live decision with the backtest for one session. Symbols and weights only, never shares.

    Full books: live full target weights vs the backtest's holdings weights after the session
    (tolerance WEIGHT_TOLERANCE_FLOAT: realised weights carry whole-share rounding and the day's move).
    Entry / exit books: the live entries and exits vs the backtest's buys and sells that session.
    """
    row_list = []
    if decision_dict.get("decision_book_type_str") == "full_target_weight_book":
        live_dict = _clean_weight_map(decision_dict.get("full_target_weight_map_dict"), skip_set=skip_set)
        test_dict = _clean_weight_map(reference_weight_map_dict, skip_set=skip_set)
        for asset_str in sorted(set(live_dict) | set(test_dict)):
            live_float, test_float = live_dict.get(asset_str, 0.0), test_dict.get(asset_str, 0.0)
            row_list.append({"asset_str": asset_str, "live_str": f"{live_float * 100:.1f}%" if live_float else "—",
                             "backtest_str": f"{test_float * 100:.1f}%" if test_float else "—",
                             "same_bool": abs(live_float - test_float) <= WEIGHT_TOLERANCE_FLOAT})
        basis_str = "Target weights vs the backtest's holdings after the session (±5 points)."
    else:
        entry_dict = _clean_weight_map(decision_dict.get("entry_target_weight_map_dict"), skip_set=skip_set)
        exit_set = {str(asset_str) for asset_str in decision_dict.get("exit_asset_list") or [] if str(asset_str) not in skip_set}
        skip_set = frozenset(skip_set) | NON_ASSET_KEY_SET
        buy_set = {asset_str for asset_str, quantity_float in reference_trade_map_dict.items() if quantity_float > 0 and asset_str not in skip_set}
        sell_set = {asset_str for asset_str, quantity_float in reference_trade_map_dict.items() if quantity_float < 0 and asset_str not in skip_set}
        for asset_str in sorted(set(entry_dict) | exit_set | buy_set | sell_set):
            live_str = f"BUY · {entry_dict[asset_str] * 100:.1f}%" if asset_str in entry_dict else "SELL" if asset_str in exit_set else "—"
            test_str = "BUY" if asset_str in buy_set else "SELL" if asset_str in sell_set else "—"
            row_list.append({"asset_str": asset_str, "live_str": live_str, "backtest_str": test_str,
                             "same_bool": live_str.split(" ")[0] == test_str})
        basis_str = "Entries and exits vs the backtest's buys and sells that session."
    differ_int = sum(1 for row_dict in row_list if not row_dict["same_bool"])
    row_list.sort(key=lambda row_dict: (row_dict["same_bool"], row_dict["asset_str"]))
    return {"status_str": "differs" if differ_int else "same", "differ_count_int": differ_int,
            "row_count_int": len(row_list), "row_list": row_list, "basis_str": basis_str}


# ---------------------------------------------------------------------------------------------- returns, gap, rare loss
def daily_returns_from_equity(equity_by_date_dict: dict[str, float]) -> dict[str, float]:
    date_list = sorted(equity_by_date_dict)
    result_dict = {}
    for previous_str, date_str in zip(date_list, date_list[1:]):
        previous_float, value_float = _finite_float(equity_by_date_dict[previous_str]), _finite_float(equity_by_date_dict[date_str])
        if previous_float and value_float is not None and previous_float > 0:
            result_dict[date_str] = value_float / previous_float - 1.0
    return result_dict


def curve_dict(live_return_dict: dict[str, float], reference_return_dict: dict[str, float]) -> dict:
    """Cumulative live and backtest returns on the dates both have, and the gap (live relative to backtest)."""
    date_list = sorted(set(live_return_dict) & set(reference_return_dict))[-CURVE_LIMIT_INT:]
    live_float = test_float = 1.0
    point_list = []
    for date_str in date_list:
        live_float *= 1.0 + live_return_dict[date_str]
        test_float *= 1.0 + reference_return_dict[date_str]
        point_list.append({"date_str": date_str, "live_float": live_float - 1.0, "backtest_float": test_float - 1.0})
    if not point_list:
        return {"point_list": [], "gap_float": None, "gap_month_float": None, "start_str": None}
    month_str = date_list[-1][:7]
    before_list = [point for point in point_list if point["date_str"][:7] < month_str]
    base_live, base_test = (before_list[-1]["live_float"], before_list[-1]["backtest_float"]) if before_list else (0.0, 0.0)
    last_dict = point_list[-1]
    return {"point_list": point_list, "start_str": date_list[0],
            "gap_float": (1 + last_dict["live_float"]) / (1 + last_dict["backtest_float"]) - 1.0,
            "gap_month_float": ((1 + last_dict["live_float"]) / (1 + base_live)) / ((1 + last_dict["backtest_float"]) / (1 + base_test)) - 1.0}


def window_return_float(return_dict: dict[str, float], date_list: list[str]) -> float | None:
    if any(date_str not in return_dict for date_str in date_list):
        return None
    growth_float = 1.0
    for date_str in date_list:
        growth_float *= 1.0 + return_dict[date_str]
    return growth_float - 1.0


def build_ruler_dict(return_list: list[float]) -> dict:
    """1-in-100 and worst loss of the backtest over 1 / 3 / 5 sessions (overlapping windows)."""
    result_dict = {}
    for window_int in RARE_WINDOW_TUPLE:
        window_list = []
        for end_int in range(window_int, len(return_list) + 1):
            growth_float = 1.0
            for value_float in return_list[end_int - window_int:end_int]:
                growth_float *= 1.0 + value_float
            window_list.append(growth_float - 1.0)
        if len(window_list) < 250:
            continue
        window_list.sort()
        result_dict[str(window_int)] = {"one_in_100_float": window_list[int(RARE_QUANTILE_FLOAT * len(window_list))],
                                        "worst_float": window_list[0], "window_count_int": len(window_list)}
    return result_dict


def rare_loss_dict(live_return_dict: dict[str, float], reference_return_dict: dict[str, float], ruler_dict: dict) -> dict | None:
    """The most severe current window below the ruler's 1-in-100 line, or None on a normal day."""
    date_list = sorted(live_return_dict)
    found_dict = None
    for window_int in RARE_WINDOW_TUPLE:
        line_dict = ruler_dict.get(str(window_int))
        # *** CRITICAL*** only a loss can be a rare loss: a ruler whose 1-in-100 line is not negative is unusable.
        if not line_dict or len(date_list) < window_int or not line_dict["one_in_100_float"] < 0:
            continue
        window_date_list = date_list[-window_int:]
        live_float = window_return_float(live_return_dict, window_date_list)
        if live_float is None or live_float >= 0 or live_float > line_dict["one_in_100_float"]:
            continue
        test_float = window_return_float(reference_return_dict, window_date_list)
        cause_str = ("unknown" if test_float is None else
                     "strategy" if test_float <= STRATEGY_SHARE_FLOAT * live_float else "ours")
        candidate_dict = {"window_int": window_int, "live_float": live_float, "backtest_float": test_float,
                          "line_float": line_dict["one_in_100_float"], "worst_float": line_dict["worst_float"],
                          "start_str": window_date_list[0], "end_str": window_date_list[-1], "cause_str": cause_str,
                          "beyond_worst_bool": live_float < line_dict["worst_float"]}
        if found_dict is None or live_float / line_dict["one_in_100_float"] > found_dict["live_float"] / found_dict["line_float"]:
            found_dict = candidate_dict
    return found_dict


def merge_record_list(previous_list, session_str: str | None, status_str: str) -> list[dict]:
    """One mark per checked session: same / differs / not_checked. Keeps the latest RECORD_LIMIT_INT."""
    record_dict = {}
    for item in previous_list or []:
        if isinstance(item, dict) and isinstance(item.get("session_str"), str) and item.get("status_str") in {"same", "differs", "not_checked"}:
            record_dict[item["session_str"]] = item["status_str"]
    if session_str:
        record_dict[session_str] = status_str if status_str in {"same", "differs"} else "not_checked"
    return [{"session_str": key_str, "status_str": record_dict[key_str]} for key_str in sorted(record_dict)][-RECORD_LIMIT_INT:]


# ---------------------------------------------------------------------------------------------- live returns (IBKR Flex, read-only)
def live_return_dict_by_account(performance_db_path_str: str | None, account_set: set[str]) -> dict[str, dict[str, float]]:
    if not performance_db_path_str or not Path(performance_db_path_str).is_file():
        return {}
    from alpha.live.client_reporting import load_broker_reporting_snapshot

    snapshot_obj = load_broker_reporting_snapshot(performance_db_path_str, allowed_account_set=account_set,
                                                  query_name_str=FLEX_QUERY_NAME_STR)
    if snapshot_obj.unavailable_reason_str:
        return {}
    result_dict: dict[str, dict[str, float]] = {}
    for row_obj in snapshot_obj.row_tuple:
        twr_float = _finite_float(row_obj.twr_decimal)
        if twr_float is not None:
            result_dict.setdefault(row_obj.account_route_str, {})[row_obj.market_date_str] = twr_float
    return result_dict


# ---------------------------------------------------------------------------------------------- reference runs
def last_finished_session_str(as_of_date_str: str) -> str:
    """The last NYSE session before the as-of date (the job runs before the open)."""
    import exchange_calendars as xcals

    as_of_obj = date.fromisoformat(as_of_date_str)
    session_list = [ts.date() for ts in xcals.get_calendar("XNYS").sessions_in_range(
        (as_of_obj - timedelta(days=12)).isoformat(), (as_of_obj - timedelta(days=1)).isoformat())]
    return session_list[-1].isoformat() if session_list else as_of_date_str


def month_end_window_bool(as_of_date_obj: date) -> bool:
    """True on the last two sessions of a month and on its first session (or on a non-session day).

    The TAA month-end decision may refresh the shared DTB3 cache file then; a TAA backtest
    in the same window could write that file at the same time, so it is skipped.
    """
    import exchange_calendars as xcals

    calendar_obj = xcals.get_calendar("XNYS")
    first_obj = as_of_date_obj.replace(day=1)
    last_obj = (first_obj + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    month_list = [ts.date() for ts in calendar_obj.sessions_in_range(first_obj.isoformat(), last_obj.isoformat())]
    if as_of_date_obj not in month_list:
        return True
    return as_of_date_obj in set(month_list[-2:]) | set(month_list[:1])


def run_reference_maps_dict(release_obj, *, start_str: str, end_str: str, capital_float: float, output_dir_path: Path) -> dict:
    from alpha.live.reference_compare import build_reference_maps_dict, run_auto_reference_strategy

    output_dir_path.mkdir(parents=True, exist_ok=True)
    strategy_obj = run_auto_reference_strategy(release_obj=release_obj, deployment_start_date_str=start_str,
                                               reference_end_date_str=end_str, deployment_initial_cash_float=capital_float,
                                               output_dir_path_obj=output_dir_path)
    maps_dict = build_reference_maps_dict(strategy_obj)
    maps_dict.pop("reference_strategy_obj", None)
    del strategy_obj
    return maps_dict


def _ruler_key_str(release_obj) -> str:
    return hashlib.sha256(f"{release_obj.strategy_import_str}|{release_obj.release_id_str}|{getattr(release_obj, 'data_profile_str', '')}"
                          .encode("utf-8")).hexdigest()[:16]


def load_or_build_ruler_dict(release_obj, *, cache_dir_path: Path, deployment_start_str: str | None, refresh_bool: bool,
                             today_obj: date) -> dict:
    """The rare-loss ruler comes from one long backtest per pod, ending before the pod went live. Cached."""
    cache_path = cache_dir_path / f"ruler_{release_obj.pod_id_str}_{_ruler_key_str(release_obj)}.json"
    if cache_path.is_file():
        try:
            cached_dict = json.loads(cache_path.read_text(encoding="utf-8"))
            built_obj = date.fromisoformat(cached_dict["built_date_str"])
            if not refresh_bool or (today_obj - built_obj).days < RULER_MAX_AGE_DAYS_INT:
                return cached_dict
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if not refresh_bool:
        return {}
    end_str = (date.fromisoformat(deployment_start_str) - timedelta(days=1)).isoformat() if deployment_start_str else today_obj.isoformat()
    maps_dict = run_reference_maps_dict(release_obj, start_str=None, end_str=end_str, capital_float=1_000_000.0,
                                        output_dir_path=cache_dir_path / "ruler_runs" / release_obj.pod_id_str)
    return_list = [value_float for _, value_float in sorted(daily_returns_from_equity(maps_dict["equity_by_date_dict"]).items())]
    ruler_dict = {"built_date_str": today_obj.isoformat(), "history_end_str": end_str, "session_count_int": len(return_list),
                  "window_dict": build_ruler_dict(return_list)}
    _atomic_write_json(cache_path, ruler_dict)
    return ruler_dict


# ---------------------------------------------------------------------------------------------- one pod
def check_pod_dict(target_obj, *, as_of_date_str: str, live_return_dict: dict, previous_pod_dict: dict | None,
                   cache_dir_path: Path, reference_bool: bool, refresh_ruler_bool: bool, month_end_bool: bool) -> dict:
    release_obj = target_obj.release_obj
    started_float = time.monotonic()
    pod_dict = {"pod_id_str": release_obj.pod_id_str, "account_route_str": release_obj.account_route_str,
                "release_id_str": release_obj.release_id_str, "strategy_import_str": release_obj.strategy_import_str,
                "decision_dict": {"status_str": "not_checked", "reason_str": ""}, "curve_dict": {"point_list": []},
                "rare_loss_dict": None, "reference_status_str": "not_run", "reference_reason_str": ""}
    source_dict = read_pod_source_dict(target_obj.db_path_str, release_obj.pod_id_str, as_of_date_str)
    pod_dict["fill_list"] = source_dict["fill_list"]
    first_dict, latest_dict = source_dict["first_vplan_dict"], source_dict["latest_vplan_dict"]
    session_str = latest_dict["session_date_str"] if latest_dict else None
    pod_dict["session_str"] = session_str
    previous_record_list = (previous_pod_dict or {}).get("record_list") or []
    module_str = str(release_obj.strategy_import_str).split(":")[0]
    reason_str = ""
    if first_dict is None or latest_dict is None or source_dict["decision_dict"] is None:
        reason_str = "No executed plan yet."
    elif not reference_bool:
        reason_str = "Backtest not run (--no-reference)."
    elif module_str.startswith(TAA_IMPORT_PREFIX_STR) and month_end_bool:
        reason_str = "Skipped in the month-end window (shared DTB3 cache)."
    else:
        from alpha.live.reference_compare import inspect_auto_reference_support_dict

        support_dict = inspect_auto_reference_support_dict(release_obj)
        if not support_dict["supported_bool"]:
            reason_str = "No automatic backtest for this strategy yet (" + str(support_dict["reason_str"]) + ")."
    if reason_str:
        pod_dict.update(reference_status_str="skipped", reference_reason_str=reason_str)
        pod_dict["decision_dict"] = {"status_str": "not_checked", "reason_str": reason_str}
        pod_dict["record_list"] = merge_record_list(previous_record_list, None, "not_checked")
        # A skipped night keeps the last curve: the dashboard shows its date.
        if previous_pod_dict and previous_pod_dict.get("curve_dict"):
            pod_dict["curve_dict"] = previous_pod_dict["curve_dict"]
        return pod_dict
    # *** CRITICAL*** the curve and the rare-loss check need the backtest up to the last finished session,
    # not only to the latest plan (a monthly pod plans once a month).
    end_str = max(session_str, last_finished_session_str(as_of_date_str))
    try:
        maps_dict = run_reference_maps_dict(release_obj, start_str=first_dict["session_date_str"], end_str=end_str,
                                            capital_float=first_dict["pod_budget_float"] or 100_000.0,
                                            output_dir_path=cache_dir_path / "reference_runs" / release_obj.pod_id_str)
    except Exception as error_obj:  # one pod's failure is reported and never hides the others
        pod_dict.update(reference_status_str="failed", reference_reason_str=f"{type(error_obj).__name__}: {error_obj}"[:400])
        pod_dict["decision_dict"] = {"status_str": "not_checked", "reason_str": "The backtest could not run."}
        pod_dict["record_list"] = merge_record_list(previous_record_list, session_str, "not_checked")
        return pod_dict
    from alpha.live.reference_compare import get_reference_realized_weight_map_dict

    skip_set = PARKING_ASSET_SET if module_str.startswith(MR_CAPSULE_IMPORT_PREFIX_STR) else frozenset()
    trade_map_dict = {asset_str: float(value_dict["quantity_float"]) for (date_str, asset_str), value_dict
                      in maps_dict["transaction_map_dict"].items() if date_str == session_str}
    decision_dict = compare_decision_dict(source_dict["decision_dict"],
                                          reference_weight_map_dict=get_reference_realized_weight_map_dict(maps_dict, session_str),
                                          reference_trade_map_dict=trade_map_dict, skip_set=skip_set)
    metadata_dict = source_dict["decision_dict"]["snapshot_metadata_dict"]
    decision_dict.update(session_str=session_str, data_snapshot_str=str(metadata_dict.get("norgate_snapshot_date_str") or ""),
                         data_hash_str=str(metadata_dict.get("norgate_manifest_hash_str") or "")[:12])
    pod_dict["decision_dict"] = decision_dict
    pod_dict["record_list"] = merge_record_list(previous_record_list, session_str, decision_dict["status_str"])
    reference_return_dict = daily_returns_from_equity(maps_dict["equity_by_date_dict"])
    start_str = first_dict["session_date_str"]
    live_window_dict = {date_str: value_float for date_str, value_float in live_return_dict.items() if date_str >= start_str}
    pod_dict["curve_dict"] = curve_dict(live_window_dict, reference_return_dict)
    try:
        ruler_dict = load_or_build_ruler_dict(release_obj, cache_dir_path=cache_dir_path, deployment_start_str=start_str,
                                              refresh_bool=refresh_ruler_bool, today_obj=date.fromisoformat(as_of_date_str))
    except Exception as error_obj:
        ruler_dict = {}
        pod_dict["ruler_reason_str"] = f"Ruler unavailable: {type(error_obj).__name__}"[:200]
    pod_dict["ruler_dict"] = ruler_dict.get("window_dict") or {}
    pod_dict["rare_loss_dict"] = rare_loss_dict(live_window_dict, reference_return_dict, pod_dict["ruler_dict"])
    pod_dict.update(reference_status_str="ok", reference_seconds_float=round(time.monotonic() - started_float, 1))
    del maps_dict
    gc.collect()
    return pod_dict


# ---------------------------------------------------------------------------------------------- report
def _atomic_write_json(path_obj: Path, payload_dict: dict) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path_obj.with_name(path_obj.name + f".{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(payload_dict, indent=1, default=str), encoding="utf-8")
    os.replace(temporary_path, path_obj)


def _read_previous_dict(report_path: Path) -> dict:
    try:
        payload_dict = json.loads(report_path.read_text(encoding="utf-8"))
        return payload_dict if payload_dict.get("schema_version_str") == SCHEMA_VERSION_STR else {}
    except (OSError, ValueError, AttributeError):
        return {}


def build_report_dict(target_list, *, as_of_ts: datetime, performance_db_path_str: str | None, previous_dict: dict,
                      cache_dir_path: Path, reference_bool: bool = True, refresh_ruler_bool: bool = False) -> dict:
    as_of_date_obj = as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date()
    live_target_list = [target_obj for target_obj in target_list
                        if target_obj.release_obj.mode_str == "live" and target_obj.release_obj.enabled_bool]
    account_set = {target_obj.release_obj.account_route_str for target_obj in live_target_list}
    try:
        live_by_account_dict = live_return_dict_by_account(performance_db_path_str, account_set)
        flex_reason_str = "" if live_by_account_dict else "No IBKR Flex returns found."
    except Exception as error_obj:
        live_by_account_dict, flex_reason_str = {}, f"IBKR Flex returns unreadable: {type(error_obj).__name__}"
    try:
        month_end_bool = month_end_window_bool(as_of_date_obj)
    except Exception:
        month_end_bool = True  # unknown calendar: be safe and skip TAA references
    previous_by_pod_dict = {pod_dict.get("pod_id_str"): pod_dict for pod_dict in previous_dict.get("pod_list") or [] if isinstance(pod_dict, dict)}
    pod_list, gate_fill_dict = [], {}
    for target_obj in live_target_list:
        release_obj = target_obj.release_obj
        try:
            pod_dict = check_pod_dict(target_obj, as_of_date_str=as_of_date_obj.isoformat(),
                                      live_return_dict=live_by_account_dict.get(release_obj.account_route_str, {}),
                                      previous_pod_dict=previous_by_pod_dict.get(release_obj.pod_id_str),
                                      cache_dir_path=cache_dir_path, reference_bool=reference_bool,
                                      refresh_ruler_bool=refresh_ruler_bool, month_end_bool=month_end_bool)
        except Exception as error_obj:
            pod_dict = {"pod_id_str": release_obj.pod_id_str, "account_route_str": release_obj.account_route_str,
                        "release_id_str": release_obj.release_id_str, "strategy_import_str": release_obj.strategy_import_str,
                        "reference_status_str": "failed", "reference_reason_str": f"{type(error_obj).__name__}: {error_obj}"[:400],
                        "decision_dict": {"status_str": "not_checked", "reason_str": "The pod could not be read."},
                        "curve_dict": {"point_list": []}, "rare_loss_dict": None, "fill_list": [],
                        "record_list": (previous_by_pod_dict.get(release_obj.pod_id_str) or {}).get("record_list") or [],
                        "traceback_str": traceback.format_exc(limit=3)}
        if str(release_obj.strategy_import_str).startswith(MR_CAPSULE_IMPORT_PREFIX_STR):
            gate_fill_dict[release_obj.pod_id_str] = pod_dict.get("fill_list") or []
        pod_dict.pop("fill_list", None)
        pod_list.append(pod_dict)
    return {"schema_version_str": SCHEMA_VERSION_STR, "generated_at_utc_str": datetime.now(UTC).isoformat(),
            "as_of_date_str": as_of_date_obj.isoformat(), "flex_reason_str": flex_reason_str,
            "month_end_window_bool": month_end_bool, "pod_list": pod_list,
            "cost_gate_dict": cost_gate_dict(gate_fill_dict, as_of_date_str=as_of_date_obj.isoformat()) if gate_fill_dict else None}


def main(argv=None) -> int:
    from alpha.live.dashboard import (DEFAULT_CONFIG_PATH_STR, DEFAULT_EVENT_LOG_PATH_STR, DEFAULT_RELEASES_ROOT_PATH_STR,
                                      DEFAULT_RESULTS_ROOT_PATH_STR, DashboardApp)

    parser = argparse.ArgumentParser(description="Report-only backtest check for LIVE pods (writes one JSON file).")
    parser.add_argument("--releases-root", default=DEFAULT_RELEASES_ROOT_PATH_STR)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH_STR)
    parser.add_argument("--event-log-path", default=DEFAULT_EVENT_LOG_PATH_STR,
                        help="The report goes next to this file, beside the watchdog report.")
    parser.add_argument("--performance-db", default=None, help="IBKR Flex store (read-only); default as the dashboard.")
    parser.add_argument("--pod-id", default=None, help="Check one pod only; the others keep their last result.")
    parser.add_argument("--no-reference", action="store_true", help="Skip backtest runs (cost gate only).")
    parser.add_argument("--refresh-ruler", action="store_true", help="Build missing or old rare-loss rulers (long backtests).")
    parser.add_argument("--skip-env-file", action="store_true", help="Do not load config.env (same switch as the dashboard).")
    args = parser.parse_args(argv)
    if not args.skip_env_file:
        # Same environment as the dashboard and the serves: Norgate profile / snapshot mode, Flex store path.
        from scripts.norgate_config_env import load_config_env_file
        load_config_env_file()
    if args.performance_db is None:
        from alpha.live.ibkr_performance import resolve_performance_db_path_str
        args.performance_db = resolve_performance_db_path_str()

    app_obj = DashboardApp(releases_root_path_str=args.releases_root, config_path_str=args.config,
                           results_root_path_str=DEFAULT_RESULTS_ROOT_PATH_STR, event_log_path_str=args.event_log_path)
    target_list = app_obj.get_target_list()
    if args.pod_id:
        target_list = [target_obj for target_obj in target_list if target_obj.release_obj.pod_id_str == args.pod_id]
        if not target_list:
            parser.error(f"No enabled release for pod {args.pod_id}.")
    report_path = Path(args.event_log_path).resolve().parent / REPORT_FILE_NAME_STR
    previous_dict = _read_previous_dict(report_path)
    report_dict = build_report_dict(target_list, as_of_ts=datetime.now(UTC), performance_db_path_str=args.performance_db,
                                    previous_dict=previous_dict, cache_dir_path=report_path.parent / CACHE_DIR_NAME_STR,
                                    reference_bool=not args.no_reference, refresh_ruler_bool=args.refresh_ruler)
    if args.pod_id and previous_dict.get("pod_list"):
        kept_list = [pod_dict for pod_dict in previous_dict["pod_list"] if pod_dict.get("pod_id_str") != args.pod_id]
        report_dict["pod_list"] = sorted(kept_list + report_dict["pod_list"], key=lambda pod_dict: str(pod_dict.get("pod_id_str")))
        # The MR gate pools both capsule pods: a one-pod run keeps the last full-run gate.
        report_dict["cost_gate_dict"] = previous_dict.get("cost_gate_dict")
    _atomic_write_json(report_path, report_dict)
    for pod_dict in report_dict["pod_list"]:
        print(f"{pod_dict['pod_id_str']}: {pod_dict['decision_dict']['status_str']} · reference {pod_dict['reference_status_str']}"
              + (f" ({pod_dict['reference_reason_str']})" if pod_dict.get("reference_reason_str") else ""))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
