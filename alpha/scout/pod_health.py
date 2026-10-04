"""P1: research-side health report for the LIVE pods (report only).

For each live pod it compares the pod's realised daily returns (IBKR Flex daily
TWR, flow-neutral, dividends included) with its expected return process (the
reference backtest) and reports:

- Cold Blood Index of the current drawdown, with RED / AMBER cuts calibrated on
  the daily monitoring procedure (alpha/stats/pod_monitor.py);
- a lower CUSUM on completed calendar-month returns (alpha/stats/health.py);
- the combined false-alarm rate of both detectors for a healthy pod;
- how much such a monitor can see at all: detection rates within one year for
  three "the edge is gone" scenarios, and the years of data needed to confirm
  that the edge has died.

Status, in order of precedence:
    RED       current CBI below the RED cut, or the monthly CUSUM has alarmed
    STALE     the realised data end more than STALE_SESSION_LIMIT_INT sessions before the report date
    TOO_EARLY fewer than MIN_OBSERVATION_INT sessions since the monitoring start (no CBI yet)
    AMBER     current CBI below the AMBER cut
    GREEN     otherwise
A RED earlier in the period (CBI below the cut on a past day) is reported
separately, so it is not lost between runs.

Boundaries:
- Read-only. It reads Flex XML files or the Flex SQLite store through
  `alpha.live.client_reporting` (pure parser / mode=ro reader), backtest pickles
  and research parquet files. It never opens a broker connection, never builds a
  LiveStateStore and never writes into alpha/live.
- `alpha.live` must never import this module (tests/test_scout_import_boundary.py).
- Alerts are not wired anywhere. Wiring into the watchdog is a live change that
  needs the owner's explicit authorisation.

Expected-process choices (stated, conservative where there is a choice):
- The expected process ends before the monitoring start, so the backtest's own
  version of the live days is never part of the yardstick.
- NDX VXN: the WORST (by pre-live Sharpe) of the 21 rebalance-day offsets of the
  parity-checked replica (research/ndx_param_robustness_20260926). A worse
  expected process makes drawdowns less surprising, so alarms are less likely to
  be luck.
- TAA 3x: the latest full vanilla backtest. No offset study exists yet, so its
  timing luck is not reflected (the rebalance-day luck band arrives with S4).
Both references include dividends, like the Flex TWR. Each report records the
source file and its sha256.

Known limits of the model (stated in the report): the 5% / 15% rates hold for a
pod that behaves like the bootstrap of its backtest. A volatility regime above
the backtest's, or crash clustering longer than the 20-session blocks, raises
the real false-alarm rate (P1 review: 1-year historical windows that contain the
2020 crash went RED often). After the first year the CBI becomes blunter, because
history grows and the index does not latch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.stats.health import calibrate_cusum_threshold, lower_cusum_path
from alpha.stats.pod_monitor import (
    build_cbi_table,
    calibrate_cbi_thresholds,
    cbi_mat,
    compound_period_returns,
    drawdown_state_path,
    simulated_live_path_mat,
)

FLEX_QUERY_NAME_STR = "ALPHA_DAILY_TWR"
TRADING_DAYS_PER_YEAR_INT = 252
MIN_OBSERVATION_INT = 21
HORIZON_INT = 252
STALE_SESSION_LIMIT_INT = 5
REFERENCE_PATH_COUNT_INT = 20000
HEALTHY_PATH_COUNT_INT = 4000
DETECTION_PATH_COUNT_INT = 500
MEAN_BLOCK_LENGTH_FLOAT = 20.0
VARIANCE_BLOCK_LENGTH_INT = 63


@dataclass(frozen=True)
class ExpectedProcess:
    return_ser: pd.Series
    source_str: str
    source_sha256_str: str


@dataclass(frozen=True)
class LivePodSpec:
    pod_id_str: str
    label_str: str
    account_route_str: str
    # First session the monitor covers. Realised data must be complete from here on;
    # the expected process ends before it.
    monitoring_start_date_str: str
    expected_loader: Callable[[Path, pd.Timestamp], ExpectedProcess]


def _latest_path(root_path: Path, pattern_str: str) -> Path:
    path_list = sorted(root_path.glob(pattern_str))
    if not path_list:
        raise FileNotFoundError(f"No file matches {root_path / pattern_str}.")
    return path_list[-1]


def _file_sha256_str(path: Path) -> str:
    digest_obj = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk_bytes in iter(lambda: file_obj.read(1 << 20), b""):
            digest_obj.update(chunk_bytes)
    return digest_obj.hexdigest()


def load_taa_expected(root_path: Path, before_date: pd.Timestamp) -> ExpectedProcess:
    from alpha.engine.strategy import Strategy

    pickle_path = _latest_path(
        root_path,
        "results/research/strategy/strategy_taa_df_btal_fallback_tqqq_vix_cash/vanilla_backtest/*/"
        "strategy_taa_df_btal_fallback_tqqq_vix_cash.pkl",
    )
    total_value_ser = Strategy.read_pickle(str(pickle_path)).results["total_value"].astype(float)
    return_ser = total_value_ser.pct_change().dropna()
    return ExpectedProcess(
        return_ser=return_ser.loc[return_ser.index < before_date],
        source_str=f"vanilla backtest {pickle_path.relative_to(root_path).as_posix()}",
        source_sha256_str=_file_sha256_str(pickle_path),
    )


def load_ndx_vxn_expected(root_path: Path, before_date: pd.Timestamp) -> ExpectedProcess:
    parquet_path = root_path / "results/research/ndx_param_robustness_20260926/returns_NDX_engine.parquet"
    offset_df = pd.read_parquet(parquet_path)
    offset_df = offset_df.loc[offset_df.index < before_date]
    live_column_list = [name for name in offset_df.columns if name.startswith("ROC12/ATR20_LIVE|") and name.endswith("|V22-0.25")]
    if len(live_column_list) != 21:
        raise ValueError(f"Expected 21 live-cell offset columns, found {len(live_column_list)}.")
    # Worst offset by pre-live Sharpe only.
    sharpe_ser = offset_df[live_column_list].apply(lambda ser: ser.mean() / ser.std(ddof=1))
    worst_column_str = sharpe_ser.idxmin()
    return_ser = offset_df[worst_column_str].astype(float)
    # The replica starts flat during the indicator warm-up; drop those leading zero days.
    first_active_idx_int = int(np.flatnonzero(return_ser.to_numpy() != 0.0)[0])
    return ExpectedProcess(
        return_ser=return_ser.iloc[first_active_idx_int:],
        source_str=(
            f"worst of 21 rebalance offsets ({worst_column_str.split('|')[5]}) in "
            f"{parquet_path.relative_to(root_path).as_posix()}"
        ),
        source_sha256_str=_file_sha256_str(parquet_path),
    )


LIVE_POD_SPEC_TUPLE = (
    LivePodSpec(
        pod_id_str="pod_taa_btal_fallback_tqqq_vix_cash_live_01",
        label_str="TAA 3x",
        account_route_str="U21192795",
        monitoring_start_date_str="2026-07-01",
        expected_loader=load_taa_expected,
    ),
    LivePodSpec(
        pod_id_str="pod_ndx_atr_normalized_vxn_scaled_live_01",
        label_str="NDX VXN",
        account_route_str="U25384771",
        monitoring_start_date_str="2026-07-01",
        expected_loader=load_ndx_vxn_expected,
    ),
)


# ---------------------------------------------------------------------------------------------- realised returns
def _xnys_sessions(start_date, end_date) -> pd.DatetimeIndex:
    session_index = xcals.get_calendar("XNYS").sessions_in_range(pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize())
    return pd.DatetimeIndex(session_index.tz_localize(None) if session_index.tz is not None else session_index)


def _flex_rows_from_xml(xml_path_list: list[Path], account_set: set[str]) -> list:
    from alpha.live.client_reporting import parse_broker_nav_import

    row_list = []
    for import_idx_int, xml_path in enumerate(xml_path_list):
        raw_xml_str = xml_path.read_text(encoding="utf-8")
        row_list.extend(
            parse_broker_nav_import(
                raw_xml_str,
                allowed_account_set=account_set,
                query_name_str=FLEX_QUERY_NAME_STR,
                source_import_id_int=import_idx_int,
                source_checksum_str=hashlib.sha256(raw_xml_str.encode("utf-8")).hexdigest(),
            )
        )
    return row_list


def _flex_rows_from_db(db_path: Path, account_set: set[str]) -> list:
    from alpha.live.client_reporting import load_broker_reporting_snapshot

    snapshot = load_broker_reporting_snapshot(str(db_path), allowed_account_set=account_set, query_name_str=FLEX_QUERY_NAME_STR)
    if snapshot.unavailable_reason_str:
        raise FileNotFoundError(snapshot.unavailable_reason_str)
    return list(snapshot.row_tuple)


def realized_session_returns(row_list: list, account_route_str: str, monitoring_start_date_str: str) -> pd.Series:
    """Daily TWR on NYSE sessions for one account, from the monitoring start on.

    - Rows for the same date from several files must agree. (The SQLite store instead lets a newer
      import supersede an older one; with XML files, drop the superseded file.)
    - Rows on non-session dates (cash or interest postings) are compounded into the
      next session, so one observation = one trading session, as in the expected process.
    - Every session from the monitoring start to the last row must be present;
      a gap would hide returns, so it raises instead.
    """
    monitoring_start = pd.Timestamp(monitoring_start_date_str)
    twr_by_date_dict: dict[str, float] = {}
    for row in row_list:
        if row.account_route_str != account_route_str or pd.Timestamp(row.market_date_str) < monitoring_start:
            continue
        twr_float = float(row.twr_decimal)
        previous_float = twr_by_date_dict.get(row.market_date_str)
        if previous_float is not None and abs(previous_float - twr_float) > 1e-9:
            raise ValueError(f"Conflicting TWR for {account_route_str} on {row.market_date_str}: {previous_float} vs {twr_float}.")
        twr_by_date_dict[row.market_date_str] = twr_float
    if not twr_by_date_dict:
        return pd.Series(dtype=float)

    raw_ser = pd.Series(twr_by_date_dict).sort_index()
    raw_ser.index = pd.to_datetime(raw_ser.index)
    session_index = _xnys_sessions(monitoring_start, raw_ser.index[-1] + pd.Timedelta(days=10))
    # *** CRITICAL*** map each date to the first session ON OR AFTER it, so a non-session posting
    # is booked on the next session, never on an earlier one.
    target_session_index = session_index[session_index.searchsorted(raw_ser.index, side="left")]
    compounded_ser = (1.0 + raw_ser).groupby(target_session_index.to_numpy()).prod() - 1.0
    compounded_ser.index = pd.DatetimeIndex(compounded_ser.index)
    compounded_ser = compounded_ser.loc[compounded_ser.index <= raw_ser.index[-1]]

    expected_session_index = session_index[session_index <= compounded_ser.index[-1]]
    missing_index = expected_session_index.difference(compounded_ser.index)
    if len(missing_index):
        missing_str = ", ".join(day.strftime("%Y-%m-%d") for day in missing_index[:10])
        raise ValueError(
            f"{account_route_str}: {len(missing_index)} session(s) missing between the monitoring start and the last row "
            f"({missing_str}{' ...' if len(missing_index) > 10 else ''}). Supply the missing Flex files."
        )
    return compounded_ser


def completed_month_returns(daily_return_ser: pd.Series) -> pd.Series:
    """Compounded calendar-month returns of complete months only.

    A month counts only if the data cover its first and its last session, so a
    partial first month (monitoring started mid-month) or the running month are
    never scored as full months.
    """
    if daily_return_ser.empty:
        return pd.Series(dtype=float)
    month_ser = (1.0 + daily_return_ser).groupby(daily_return_ser.index.to_period("M")).prod() - 1.0
    complete_period_list = []
    for period in month_ser.index:
        month_session_index = _xnys_sessions(period.start_time, period.end_time)
        if daily_return_ser.index[0] <= month_session_index[0] and daily_return_ser.index[-1] >= month_session_index[-1]:
            complete_period_list.append(period)
    return month_ser.loc[complete_period_list]


# ---------------------------------------------------------------------------------------------- evaluation
def _annualized_sharpe_float(return_vec) -> float:
    return_arr = np.asarray(return_vec, dtype=float)
    return float(return_arr.mean() / return_arr.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR_INT))


def years_to_confirm_edge_loss(expected_return_vec, power_float: float = 0.8, alpha_float: float = 0.05, block_length_int: int = 1) -> float:
    """Years of daily data for a one-sided test to detect that the mean return fell from μ to 0.

        n = ((z_{1−α} + z_{power}) · σ_eff / μ)²  sessions,   σ_eff² = Var(sum of b sessions) / b

    With b = 1 this assumes independent days. A larger b uses the variance of
    non-overlapping b-session sums, which absorbs short-horizon autocorrelation
    (mean reversion shrinks σ_eff, momentum grows it). The backtest mean itself is
    uncertain, so treat the answer as an order of magnitude.
    """
    from scipy import stats

    return_arr = np.asarray(expected_return_vec, dtype=float)
    mean_float = float(return_arr.mean())
    if mean_float <= 0.0:
        return float("inf")
    block_count_int = return_arr.size // block_length_int
    block_sum_vec = return_arr[: block_count_int * block_length_int].reshape(block_count_int, block_length_int).sum(axis=1)
    effective_std_float = float(np.sqrt(block_sum_vec.var(ddof=1) / block_length_int))
    session_count_float = ((stats.norm.ppf(1 - alpha_float) + stats.norm.ppf(power_float)) * effective_std_float / mean_float) ** 2
    return session_count_float / TRADING_DAYS_PER_YEAR_INT


def _cusum_alarm_vec(daily_path_mat: np.ndarray, calibration) -> np.ndarray:
    alarm_list = []
    for path_vec in daily_path_mat:
        standardized_vec = (compound_period_returns(path_vec, 21) - calibration.expected_mean_float) / calibration.expected_std_float
        alarm_list.append(np.min(lower_cusum_path(standardized_vec, calibration.reference_k_float)) <= -calibration.threshold_h_float)
    return np.array(alarm_list, dtype=bool)


def _detector_rates(daily_path_mat: np.ndarray, table, thresholds, cusum_calibration) -> dict:
    """Share of paths with a CBI RED, a CUSUM alarm, and either, within the horizon; median first-RED session."""
    red_mask_mat = cbi_mat(daily_path_mat, table, thresholds.min_observation_int) < thresholds.red_float
    cbi_red_vec = red_mask_mat.any(axis=1)
    first_red_vec = np.where(cbi_red_vec, red_mask_mat.argmax(axis=1) + 1, np.nan).astype(float)
    cusum_alarm_vec = _cusum_alarm_vec(daily_path_mat, cusum_calibration)
    return {
        "cbi_red_share_float": float(cbi_red_vec.mean()),
        "cusum_alarm_share_float": float(cusum_alarm_vec.mean()),
        "either_share_float": float((cbi_red_vec | cusum_alarm_vec).mean()),
        "median_sessions_to_cbi_red_among_detected_float": float(np.nanmedian(first_red_vec)) if cbi_red_vec.any() else float("nan"),
    }


def evaluate_pod(spec: LivePodSpec, expected: ExpectedProcess, realized_return_ser: pd.Series, as_of_date: date) -> dict:
    monitoring_start = pd.Timestamp(spec.monitoring_start_date_str)
    # *** CRITICAL*** the expected process must end before the monitoring start; otherwise the
    # backtest's own version of the live days sits inside the yardstick it is measured against.
    expected_ser = expected.return_ser.loc[expected.return_ser.index < monitoring_start]
    expected_vec = expected_ser.to_numpy(dtype=float)
    daily_mean_float = float(expected_vec.mean())
    realized_count_int = int(realized_return_ser.size)
    max_observation_int = max(HORIZON_INT, realized_count_int)

    table = build_cbi_table(
        expected_vec, max_observation_int, path_count_int=REFERENCE_PATH_COUNT_INT, mean_block_length_float=MEAN_BLOCK_LENGTH_FLOAT
    )
    thresholds = calibrate_cbi_thresholds(
        expected_vec, table, horizon_int=HORIZON_INT, min_observation_int=MIN_OBSERVATION_INT, healthy_path_count_int=HEALTHY_PATH_COUNT_INT
    )
    cusum_calibration = calibrate_cusum_threshold(compound_period_returns(expected_vec, 21), mean_block_length_float=1.0)

    healthy_path_mat = simulated_live_path_mat(expected_vec, 2000, MEAN_BLOCK_LENGTH_FLOAT, HORIZON_INT, 31)
    healthy_rate_dict = _detector_rates(healthy_path_mat, table, thresholds, cusum_calibration)

    scenario_list = [
        ("edge died (same volatility, zero drift)", -daily_mean_float),
        ("mirror (loses what it used to make)", -2.0 * daily_mean_float),
        ("broken (loses 3x what it used to make)", -4.0 * daily_mean_float),
    ]
    detection_row_list = []
    for scenario_idx_int, (scenario_str, shift_float) in enumerate(scenario_list):
        path_mat = simulated_live_path_mat(
            expected_vec, DETECTION_PATH_COUNT_INT, MEAN_BLOCK_LENGTH_FLOAT, HORIZON_INT, 10 + scenario_idx_int, shift_float
        )
        detection_row_list.append({"scenario_str": scenario_str, **_detector_rates(path_mat, table, thresholds, cusum_calibration)})

    report_dict = {
        "pod_id_str": spec.pod_id_str,
        "label_str": spec.label_str,
        "account_route_str": spec.account_route_str,
        "monitoring_start_str": spec.monitoring_start_date_str,
        "as_of_str": as_of_date.isoformat(),
        "expected": {
            "source_str": expected.source_str,
            "source_sha256_str": expected.source_sha256_str,
            "start_str": str(expected_ser.index[0].date()),
            "end_str": str(expected_ser.index[-1].date()),
            "session_count_int": int(expected_vec.size),
            "annualized_sharpe_float": _annualized_sharpe_float(expected_vec),
            "annualized_volatility_float": float(expected_vec.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR_INT)),
            "years_to_confirm_edge_loss_iid_float": years_to_confirm_edge_loss(expected_vec),
            "years_to_confirm_edge_loss_float": years_to_confirm_edge_loss(expected_vec, block_length_int=VARIANCE_BLOCK_LENGTH_INT),
        },
        "thresholds": {
            "cbi_red_float": thresholds.red_float,
            "cbi_amber_float": thresholds.amber_float,
            "horizon_sessions_int": thresholds.horizon_int,
            "min_observation_int": thresholds.min_observation_int,
            "cusum_h_float": cusum_calibration.threshold_h_float,
            "cusum_k_float": cusum_calibration.reference_k_float,
            "reference_path_count_int": REFERENCE_PATH_COUNT_INT,
            "mean_block_length_float": MEAN_BLOCK_LENGTH_FLOAT,
        },
        "healthy_false_alarm": healthy_rate_dict,
        "detection_power": detection_row_list,
        "live": None,
    }

    if realized_count_int == 0:
        report_dict["status_str"] = "NO_DATA"
        return report_dict

    realized_vec = realized_return_ser.to_numpy(dtype=float)
    depth_arr, age_arr = drawdown_state_path(realized_vec)
    live_cbi_arr = cbi_mat(realized_vec[None, :], table, MIN_OBSERVATION_INT)[0]
    month_ser = completed_month_returns(realized_return_ser)
    if month_ser.size:
        standardized_vec = (month_ser.to_numpy() - cusum_calibration.expected_mean_float) / cusum_calibration.expected_std_float
        cusum_vec = lower_cusum_path(standardized_vec, cusum_calibration.reference_k_float)
        cusum_alarm_bool = bool(np.min(cusum_vec) <= -cusum_calibration.threshold_h_float)
        cusum_now_float = float(cusum_vec[-1])
    else:
        cusum_alarm_bool, cusum_now_float = False, 0.0

    last_data_date = realized_return_ser.index[-1]
    sessions_behind_int = int(len(_xnys_sessions(last_data_date + pd.Timedelta(days=1), pd.Timestamp(as_of_date) - pd.Timedelta(days=1))))
    stale_bool = sessions_behind_int > STALE_SESSION_LIMIT_INT
    current_cbi_float = float(live_cbi_arr[-1])
    red_idx_arr = np.flatnonzero(live_cbi_arr < thresholds.red_float)

    # Early on the CBI is NaN (no evaluation before session MIN_OBSERVATION_INT), so its comparisons are False
    # and a CUSUM alarm or stale data still show.
    if current_cbi_float < thresholds.red_float or cusum_alarm_bool:
        status_str = "RED"
    elif stale_bool:
        status_str = "STALE"
    elif realized_count_int < MIN_OBSERVATION_INT:
        status_str = "TOO_EARLY"
    elif current_cbi_float < thresholds.amber_float:
        status_str = "AMBER"
    else:
        status_str = "GREEN"

    finite_cbi_arr = np.where(np.isfinite(live_cbi_arr), live_cbi_arr, np.inf)
    worst_idx_int = int(np.argmin(finite_cbi_arr))
    worst_is_finite_bool = bool(np.isfinite(finite_cbi_arr[worst_idx_int]))
    report_dict["status_str"] = status_str
    report_dict["live"] = {
        "start_str": str(realized_return_ser.index[0].date()),
        "end_str": str(last_data_date.date()),
        "sessions_behind_int": sessions_behind_int,
        "stale_bool": stale_bool,
        "session_count_int": realized_count_int,
        "cumulative_return_float": float(np.prod(1.0 + realized_vec) - 1.0),
        "annualized_volatility_float": float(realized_vec.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR_INT)) if realized_count_int > 1 else None,
        "current_drawdown_float": float(depth_arr[-1]),
        "current_drawdown_age_int": int(age_arr[-1]),
        "current_cbi_float": current_cbi_float if np.isfinite(current_cbi_float) else None,
        "worst_cbi_float": float(finite_cbi_arr[worst_idx_int]) if worst_is_finite_bool else None,
        "worst_cbi_date_str": str(realized_return_ser.index[worst_idx_int].date()) if worst_is_finite_bool else None,
        "first_red_date_str": str(realized_return_ser.index[red_idx_arr[0]].date()) if red_idx_arr.size else None,
        "completed_month_return_dict": {str(period): float(value) for period, value in month_ser.items()},
        "cusum_now_float": cusum_now_float,
        "cusum_alarm_bool": cusum_alarm_bool,
    }
    return report_dict


# ---------------------------------------------------------------------------------------------- rendering
def _pct_str(value_obj, digits_int: int = 1) -> str:
    return "n/a" if value_obj is None or not np.isfinite(value_obj) else f"{value_obj * 100:.{digits_int}f}%"


def render_markdown(report_list: list[dict], as_of_str: str, data_note_str: str) -> str:
    line_list = [
        f"# Live pod health — {as_of_str}",
        "",
        "Research-side report (Scout P1). Report only: nothing here is wired to alerts or trading.",
        "",
        f"Realised data: {data_note_str}",
        "",
        "| Pod | Status | Data through | Live sessions | Return since start | Drawdown now | CBI now | RED / AMBER cut | CUSUM | Earlier RED |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for report_dict in report_list:
        live_dict = report_dict["live"] or {}
        threshold_dict = report_dict["thresholds"]
        behind_str = f" ({live_dict['sessions_behind_int']} sessions behind)" if live_dict.get("stale_bool") else ""
        line_list.append(
            f"| {report_dict['label_str']} | **{report_dict['status_str']}** | {live_dict.get('end_str', 'n/a')}{behind_str} | "
            f"{live_dict.get('session_count_int', 0)} | {_pct_str(live_dict.get('cumulative_return_float'))} | "
            f"{_pct_str(live_dict.get('current_drawdown_float'))} ({live_dict.get('current_drawdown_age_int', 'n/a')} sessions) | "
            f"{_pct_str(live_dict.get('current_cbi_float'))} | "
            f"{_pct_str(threshold_dict['cbi_red_float'], 2)} / {_pct_str(threshold_dict['cbi_amber_float'], 2)} | "
            f"{'ALARM' if live_dict.get('cusum_alarm_bool') else 'ok'} | {live_dict.get('first_red_date_str') or 'none'} |"
        )
    line_list += [
        "",
        "Status: RED when the current Cold Blood Index (CBI) is below the RED cut or the monthly CUSUM has alarmed;",
        f"STALE when the data end more than {STALE_SESSION_LIMIT_INT} sessions before the report date; AMBER when the CBI is",
        "below the AMBER cut. The CBI cuts are calibrated so that a pod behaving like its expected process goes RED at",
        "least once in a year with 5% probability (AMBER 15%), evaluating every session from session 21. RED from either",
        "detector happens more often than 5%; each pod's combined rate is shown below. The CUSUM alarm latches.",
        "",
    ]
    for report_dict in report_list:
        expected_dict = report_dict["expected"]
        healthy_dict = report_dict["healthy_false_alarm"]
        line_list += [
            f"## {report_dict['label_str']} (`{report_dict['pod_id_str']}`)",
            "",
            f"- **Expected process:** {expected_dict['source_str']} (sha256 {expected_dict['source_sha256_str'][:12]}), "
            f"{expected_dict['start_str']} to {expected_dict['end_str']} ({expected_dict['session_count_int']} sessions): "
            f"Sharpe {expected_dict['annualized_sharpe_float']:.2f}, volatility {_pct_str(expected_dict['annualized_volatility_float'])}.",
            f"- **Healthy pod, one year:** CBI RED {_pct_str(healthy_dict['cbi_red_share_float'], 0)}, CUSUM alarm "
            f"{_pct_str(healthy_dict['cusum_alarm_share_float'], 0)}, either {_pct_str(healthy_dict['either_share_float'], 0)}.",
            f"- **Years of data needed to confirm the edge is gone** (80% power, 5% one-sided): "
            f"{expected_dict['years_to_confirm_edge_loss_float']:.1f} with {VARIANCE_BLOCK_LENGTH_INT}-session autocorrelation "
            f"({expected_dict['years_to_confirm_edge_loss_iid_float']:.1f} assuming independent days). The backtest mean is itself "
            "uncertain, so read this as an order of magnitude.",
        ]
        if report_dict["live"]:
            live_dict = report_dict["live"]
            month_str = ", ".join(f"{period} {_pct_str(value)}" for period, value in live_dict["completed_month_return_dict"].items()) or "none complete"
            line_list += [
                f"- **Live:** monitoring from {report_dict['monitoring_start_str']}, data {live_dict['start_str']} to "
                f"{live_dict['end_str']}; completed months: {month_str}.",
                f"- **Worst CBI so far:** {_pct_str(live_dict['worst_cbi_float'])} on {live_dict['worst_cbi_date_str']}.",
            ]
        line_list += [
            "",
            "What this monitor can see within one year (simulated from the expected process):",
            "",
            "| If the pod's edge… | CBI RED | median sessions to RED (among detected) | CUSUM alarm | either |",
            "|---|---|---|---|---|",
        ]
        for row_dict in report_dict["detection_power"]:
            median_float = row_dict["median_sessions_to_cbi_red_among_detected_float"]
            median_str = "n/a" if not np.isfinite(median_float) else f"{median_float:.0f}"
            line_list.append(
                f"| {row_dict['scenario_str']} | {_pct_str(row_dict['cbi_red_share_float'], 0)} | {median_str} | "
                f"{_pct_str(row_dict['cusum_alarm_share_float'], 0)} | {_pct_str(row_dict['either_share_float'], 0)} |"
            )
        died_row_dict = report_dict["detection_power"][0]
        line_list += [
            "",
            f"If this pod's edge died today, the monitor would notice within a year about "
            f"{_pct_str(died_row_dict['either_share_float'], 0)} of the time, against "
            f"{_pct_str(healthy_dict['either_share_float'], 0)} false alarms for a healthy pod.",
            "",
        ]
    line_list += [
        "## How to read this",
        "",
        "- Returns alone separate a dead edge from bad luck only slowly: compare each pod's 'edge died' row with its",
        "  healthy false-alarm rate. Large breaks are caught within months.",
        "- Implementation breaks (wrong orders, stale data, a missed rebalance) are caught much faster by comparing live",
        "  decisions with the backtest's decisions on the same days (`runner compare_reference`), not by this report.",
        "- The 5% / 15% rates hold for a pod that behaves like the bootstrap of its backtest. A volatility regime above",
        "  the backtest's, or a crash that clusters longer than the 20-session bootstrap blocks, raises the real",
        "  false-alarm rate: a RED during a market crash may be the market, not the pod.",
        "- After the first year the CBI becomes blunter (history grows and the index does not latch); the numbers",
        "  above are first-year numbers.",
        "- Expected processes include dividends, like the IBKR TWR. Live returns also contain costs, cash drag and",
        "  execution noise that the backtest does not, so a small gap is expected.",
    ]
    return "\n".join(line_list) + "\n"


def run_pod_health_report(
    flex_xml_path_list: list[Path] | None = None,
    flex_db_path: Path | None = None,
    output_dir_path: Path | None = None,
    root_path: Path = MAIN_CHECKOUT_ROOT_PATH,
    pod_spec_tuple: tuple[LivePodSpec, ...] = LIVE_POD_SPEC_TUPLE,
    as_of_date: date | None = None,
) -> tuple[Path, list[dict]]:
    as_of_date = as_of_date or date.today()
    account_set = {spec.account_route_str for spec in pod_spec_tuple}
    if flex_db_path is not None:
        row_list = _flex_rows_from_db(flex_db_path, account_set)
        data_note_str = f"IBKR Flex store {flex_db_path}"
    elif flex_xml_path_list:
        row_list = _flex_rows_from_xml(flex_xml_path_list, account_set)
        data_note_str = "IBKR Flex XML " + ", ".join(path.name for path in flex_xml_path_list)
    else:
        raise ValueError("Give either flex_db_path or flex_xml_path_list.")

    report_list = []
    for spec in pod_spec_tuple:
        expected = spec.expected_loader(root_path, pd.Timestamp(spec.monitoring_start_date_str))
        realized_ser = realized_session_returns(row_list, spec.account_route_str, spec.monitoring_start_date_str)
        report_list.append(evaluate_pod(spec, expected, realized_ser, as_of_date))

    as_of_str = as_of_date.isoformat()
    output_dir_path = output_dir_path or (root_path / "results" / "scout" / "pod_health" / as_of_str)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    (output_dir_path / "pod_health.json").write_text(json.dumps(report_list, indent=2, default=str), encoding="utf-8")
    (output_dir_path / "pod_health.md").write_text(render_markdown(report_list, as_of_str, data_note_str), encoding="utf-8")
    return output_dir_path, report_list
