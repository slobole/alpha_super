"""Saved-ledger signoff statistics. No provider, broker, strategy or scheduler IO.

Standalone performance retains first-session P&L. Benchmark comparisons start
at the same first recorded close. Cost stresses preserve the original trades;
they are accounting sensitivities, not new strategies or rerun backtests.
"""
from __future__ import annotations

import argparse
import ast
import gc
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet_module

REPO_PATH = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = REPO_PATH / "results/review/wired_signoff_20261015/audit_statistics"
MODEL_CAPITAL_FLOAT = 1_000_000.0


def clean_obj(value_obj):
    if isinstance(value_obj, dict):
        return {str(key_str): clean_obj(item_obj) for key_str, item_obj in value_obj.items()}
    if isinstance(value_obj, (list, tuple)):
        return [clean_obj(item_obj) for item_obj in value_obj]
    if isinstance(value_obj, np.generic):
        return clean_obj(value_obj.item())
    if isinstance(value_obj, float) and not np.isfinite(value_obj):
        return None
    if isinstance(value_obj, (pd.Timestamp, datetime)):
        return value_obj.isoformat()
    return value_obj


def write_json(path_obj: Path, value_obj: object) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(clean_obj(value_obj), indent=2, allow_nan=False), encoding="utf-8")


def update_index(path_obj: Path, row_list: list[dict]) -> None:
    existing_list = json.loads(path_obj.read_text(encoding="utf-8")) if path_obj.exists() else []
    by_key_dict = {row_dict["key"]: row_dict for row_dict in existing_list}
    by_key_dict.update({row_dict["key"]: row_dict for row_dict in row_list})
    write_json(path_obj, [by_key_dict[key_str] for key_str in sorted(by_key_dict)])


def hash_dict(path_obj: Path) -> dict:
    digest_obj = hashlib.sha256()
    with path_obj.open("rb") as handle_obj:
        for chunk_bytes in iter(lambda: handle_obj.read(1024 * 1024), b""):
            digest_obj.update(chunk_bytes)
    return {"path": str(path_obj.resolve()), "bytes": path_obj.stat().st_size, "sha256": digest_obj.hexdigest()}


def verify_input_manifest(input_manifest_list: list[dict]) -> None:
    for expected_dict in input_manifest_list:
        actual_dict = hash_dict(Path(expected_dict["path"]))
        if actual_dict != expected_dict:
            raise ValueError(f"Saved audit input changed during statistics: {expected_dict['path']}")


def ordered_series(value_ser: pd.Series, name_str: str) -> pd.Series:
    result_ser = value_ser.astype(float).copy()
    result_ser.index = pd.DatetimeIndex(pd.to_datetime(result_ser.index))
    if result_ser.index.tz is not None:
        result_ser.index = result_ser.index.tz_localize(None)
    if result_ser.index.has_duplicates or not result_ser.index.is_monotonic_increasing:
        raise ValueError(f"{name_str} must have unique chronological dates.")
    if not np.isfinite(result_ser.to_numpy()).all():
        raise ValueError(f"{name_str} contains missing or nonfinite values.")
    return result_ser


def equity_metrics(equity_ser: pd.Series, initial_float: float, anchor_ts=None) -> dict:
    equity_ser = ordered_series(equity_ser, "equity")
    if len(equity_ser) == 0 or initial_float <= 0 or (equity_ser <= 0).any():
        raise ValueError("Equity and initial capital must be positive.")
    # *** CRITICAL *** Reporting boundary: first recorded equity is compared
    # with actual preceding capital, not itself. This retains first-day losses.
    previous_nav_vec = np.r_[initial_float, equity_ser.to_numpy()[:-1]]
    return_ser = pd.Series(equity_ser.to_numpy() / previous_nav_vec - 1, index=equity_ser.index)
    peak_vec = np.maximum.accumulate(np.r_[initial_float, equity_ser.to_numpy()])[1:]
    drawdown_ser = equity_ser / peak_vec - 1
    year_float = ((equity_ser.index[-1] - pd.Timestamp(anchor_ts)).total_seconds() / (365.25 * 86400)
        if anchor_ts is not None else (equity_ser.index[-1] - equity_ser.index[0]).days / 365.25)
    volatility_float = return_ser.std(ddof=1) * np.sqrt(252) if len(return_ser) >= 2 else np.nan
    return {"sessions": len(equity_ser), "start": str(equity_ser.index[0].date()),
        "end": str(equity_ser.index[-1].date()), "anchor_date": None if anchor_ts is None else str(pd.Timestamp(anchor_ts).date()),
        "initial_equity": initial_float, "terminal_equity": equity_ser.iloc[-1],
        "total_return": equity_ser.iloc[-1] / initial_float - 1,
        "cagr": (equity_ser.iloc[-1] / initial_float) ** (1 / year_float) - 1 if year_float > 0 else np.nan,
        "volatility": volatility_float,
        "sharpe_zero_rf": return_ser.mean() * 252 / volatility_float if np.isfinite(volatility_float) and volatility_float > 0 else np.nan,
        "max_drawdown": drawdown_ser.min(), "first_return": return_ser.iloc[0]}


def full_return_series(equity_ser: pd.Series, initial_float: float) -> pd.Series:
    equity_ser = ordered_series(equity_ser, "equity")
    return pd.Series(equity_ser.to_numpy() / np.r_[initial_float, equity_ser.to_numpy()[:-1]] - 1,
        index=equity_ser.index, name="strategy_return")


def subperiod_dict(equity_ser: pd.Series, initial_float: float, start_str: str, end_str: str | None = None) -> dict:
    equity_ser = ordered_series(equity_ser, "equity")
    selected_ser = equity_ser.loc[start_str:end_str]
    if selected_ser.empty:
        return {"status": "unavailable", "reason": "no_observations"}
    first_position_int = int(equity_ser.index.get_loc(selected_ser.index[0]))
    prior_float = initial_float if first_position_int == 0 else float(equity_ser.iloc[first_position_int - 1])
    anchor_ts = None if first_position_int == 0 else equity_ser.index[first_position_int - 1]
    return {"status": "available", "boundary": "first selected daily return retains its preceding recorded close",
        **equity_metrics(selected_ser, prior_float, anchor_ts)}


def common_close_comparison(equity_ser: pd.Series, benchmark_ser: pd.Series, window_int: int = 126) -> tuple[dict, pd.DataFrame]:
    equity_ser = ordered_series(equity_ser, "equity")
    benchmark_ser = benchmark_ser.astype(float).reindex(equity_ser.index)
    valid_bool_ser = benchmark_ser.notna() & np.isfinite(benchmark_ser) & benchmark_ser.gt(0)
    coverage_dict = {"expected_sessions": len(equity_ser), "available_sessions": int(valid_bool_ser.sum()),
        "missing_dates": [str(date_ts.date()) for date_ts in equity_ser.index[~valid_bool_ser]]}
    if not valid_bool_ser.all() or len(equity_ser) < 3:
        return {"status": "unavailable", "reason": "missing_benchmark_sessions_or_too_short", "coverage": coverage_dict}, pd.DataFrame()
    # *** CRITICAL *** Same-close comparison: start BOTH paths at their first
    # recorded close. Standalone first-day trading P&L is intentionally outside
    # this comparison. No missing-date drop may bridge a decision boundary.
    pair_df = pd.DataFrame({"strategy_return": equity_ser.pct_change(fill_method=None),
        "benchmark_return": benchmark_ser.pct_change(fill_method=None)}).iloc[1:]
    benchmark_variance_float = pair_df["benchmark_return"].var(ddof=1)
    beta_float = (pair_df["strategy_return"].cov(pair_df["benchmark_return"]) / benchmark_variance_float
        if benchmark_variance_float > 0 else np.nan)
    monthly_df = pair_df.groupby(pair_df.index.to_period("M")).agg(lambda return_ser: (1 + return_ser).prod() - 1)
    monthly_benchmark_variance_float = monthly_df["benchmark_return"].var(ddof=1) if len(monthly_df) >= 2 else np.nan
    # *** CRITICAL *** Trailing descriptive estimates at Close_T use only the
    # last126 completed return pairs throughT. No centered/future window.
    rolling_cov_ser = pair_df["strategy_return"].rolling(window_int, min_periods=window_int).cov(pair_df["benchmark_return"])
    rolling_var_ser = pair_df["benchmark_return"].rolling(window_int, min_periods=window_int).var(ddof=1)
    chart_df = pd.DataFrame(index=equity_ser.index)
    chart_df["strategy_common_close_equity"] = equity_ser / equity_ser.iloc[0] * MODEL_CAPITAL_FLOAT
    chart_df["benchmark_common_close_equity"] = benchmark_ser / benchmark_ser.iloc[0] * MODEL_CAPITAL_FLOAT
    chart_df["rolling126_correlation"] = pair_df["strategy_return"].rolling(window_int, min_periods=window_int).corr(pair_df["benchmark_return"])
    chart_df["rolling126_beta"] = rolling_cov_ser / rolling_var_ser.where(rolling_var_ser > 0)
    comparison_dict = {"status": "available", "coverage": coverage_dict,
        "anchor_close_date": str(equity_ser.index[0].date()), "paired_return_count": len(pair_df),
        "convention": "both normalized at identical first recorded close; excludes initial strategy open-to-close P&L; RF=0",
        "strategy": equity_metrics(equity_ser.iloc[1:], float(equity_ser.iloc[0]), equity_ser.index[0]),
        "benchmark": equity_metrics(benchmark_ser.iloc[1:], float(benchmark_ser.iloc[0]), benchmark_ser.index[0]),
        "daily_correlation": (pair_df["strategy_return"].corr(pair_df["benchmark_return"])
            if pair_df["strategy_return"].std(ddof=1) > 0 and benchmark_variance_float > 0 else np.nan), "daily_beta": beta_float,
        "monthly_correlation": (monthly_df["strategy_return"].corr(monthly_df["benchmark_return"])
            if len(monthly_df) >= 2 and monthly_df["strategy_return"].std(ddof=1) > 0
                and monthly_df["benchmark_return"].std(ddof=1) > 0 else np.nan),
        "monthly_beta": (monthly_df["strategy_return"].cov(monthly_df["benchmark_return"]) / monthly_benchmark_variance_float
            if monthly_benchmark_variance_float > 0 else np.nan),
        "monthly_observation_count": len(monthly_df), "monthly_convention": "calendar-month compound, including partial boundary months",
        "rolling_window_sessions": window_int}
    return comparison_dict, chart_df


def select_benchmarks(pricing_df: pd.DataFrame) -> dict:
    benchmark_dict = {}
    if (("$SPX", "Close") in pricing_df.columns
            and pricing_df.attrs.get("benchmark_data_symbol_dict", {}).get("$SPX") == "$SPXTR"):
        benchmark_dict["SPX_TR"] = {"series": pricing_df[("$SPX", "Close")],
            "source": "saved canonical $SPX Close; Norgate loader maps $SPX label to $SPXTR",
            "adjustment_provenance": pricing_df.attrs.get("benchmark_data_symbol_dict", {})}
    elif ("$SPXTR", "Close") in pricing_df.columns:
        benchmark_dict["SPX_TR"] = {"series": pricing_df[("$SPXTR", "Close")], "source": "saved $SPXTR Close"}
    else:
        benchmark_dict["SPX_TR"] = {"status": "unavailable", "reason": "no_saved_SPX_total_return_column_with_explicit_SPXTR_provenance"}
    adjustment_dict = {**pricing_df.attrs.get("adjustment_by_symbol_dict", {}),
        **pricing_df.attrs.get("norgate_adjustment_by_symbol_dict", {})}
    signal_adjustment_dict = pricing_df.attrs.get("signal_adjustment_by_symbol_dict", {})
    if ("SPY", "Close") in pricing_df.columns and adjustment_dict.get("SPY") == "TOTALRETURN":
        benchmark_dict["SPY_TR"] = {"series": pricing_df[("SPY", "Close")], "source": "saved SPY Close with TOTALRETURN provenance"}
    elif ("ADAPTIVE_TR_SPY", "Close") in pricing_df.columns and signal_adjustment_dict.get("ADAPTIVE_TR_SPY") == "TOTALRETURN":
        benchmark_dict["SPY_TR"] = {"series": pricing_df[("ADAPTIVE_TR_SPY", "Close")], "source": "saved independent ADAPTIVE_TR_SPY TOTALRETURN signal close"}
    else:
        benchmark_dict["SPY_TR"] = {"status": "unavailable", "reason": "independent_SPY_TOTALRETURN_not_saved_with_provenance; CAPITALSPECIAL_SPY_is_not_TR"}
    return benchmark_dict


def order_diagnostics(transactions_df: pd.DataFrame, pricing_df: pd.DataFrame,
        equity_ser: pd.Series, initial_float: float, trace_list: list[dict]) -> tuple[dict, pd.DataFrame]:
    required_set = {"bar", "asset", "amount", "price"}
    if not required_set.issubset(transactions_df.columns):
        raise ValueError("Transactions missing required bar/asset/amount/price fields.")
    transaction_df = transactions_df.copy()
    transaction_df["bar"] = pd.to_datetime(transaction_df["bar"]).dt.tz_localize(None).dt.normalize()
    for field_str in ("amount", "price"):
        transaction_df[field_str] = pd.to_numeric(transaction_df[field_str], errors="raise")
    if not np.isfinite(transaction_df[["amount", "price"]].to_numpy()).all() or (transaction_df["price"] <= 0).any():
        raise ValueError("Invalid transaction quantities or prices.")
    trace_nav_dict = {}
    trace_decision_dict = {}
    ambiguous_trace_set = set()
    for trace_dict in trace_list:
        execution_ts = pd.Timestamp(trace_dict["execution_date"]).tz_localize(None).normalize()
        if execution_ts in trace_nav_dict:
            ambiguous_trace_set.add(execution_ts)
        trace_nav_dict[execution_ts] = float(trace_dict["nav"])
        trace_decision_dict[execution_ts] = pd.Timestamp(trace_dict["decision_date"]).tz_localize(None).normalize()
    for execution_ts in ambiguous_trace_set:
        trace_nav_dict.pop(execution_ts)
        trace_decision_dict.pop(execution_ts)
    liquidity_cache_dict = {}
    for asset_str in transaction_df["asset"].astype(str).unique():
        turnover_key = (asset_str, "Turnover")
        # *** CRITICAL *** At executionT use median(nativeTurnover[T-20:T-1]).
        # shift(1) excludes execution-day volume, which cannot size its open.
        turnover_ser = (pricing_df[turnover_key].astype(float) if turnover_key in pricing_df.columns
            else pd.Series(np.nan, index=pricing_df.index))
        turnover_ser = turnover_ser.where(np.isfinite(turnover_ser) & turnover_ser.gt(0))
        liquidity_cache_dict[asset_str] = turnover_ser.rolling(20, min_periods=20).median().shift(1)
    row_list = []
    for transaction_obj in transaction_df.itertuples(index=False):
        execution_ts = transaction_obj.bar
        asset_str = str(transaction_obj.asset)
        notional_float = abs(float(transaction_obj.amount) * float(transaction_obj.price))
        position_int = int(equity_ser.index.searchsorted(execution_ts))
        fallback_nav_float = initial_float if position_int == 0 else float(equity_ser.iloc[position_int - 1])
        nav_float = trace_nav_dict.get(execution_ts, fallback_nav_float)
        if not np.isfinite(nav_float) or nav_float <= 0:
            raise ValueError("Order decision NAV is invalid.")
        decision_ts = trace_decision_dict.get(execution_ts)
        if decision_ts is None:
            prior_price_idx = pricing_df.index[pricing_df.index < execution_ts]
            decision_ts = None if len(prior_price_idx) == 0 else prior_price_idx[-1]
        adv_float = liquidity_cache_dict[asset_str].get(execution_ts, np.nan)
        adv_valid_bool = np.isfinite(adv_float) and adv_float > 0
        nominal_key = (asset_str, "Unadjusted Close")
        nominal_float = (float(pricing_df.loc[decision_ts, nominal_key])
            if nominal_key in pricing_df.columns and decision_ts in pricing_df.index else np.nan)
        nominal_valid_bool = np.isfinite(nominal_float) and nominal_float > 0
        model_notional_float = notional_float / nav_float * MODEL_CAPITAL_FLOAT
        row_list.append({"execution_date": execution_ts, "decision_date": decision_ts, "asset": asset_str,
            "signed_adjusted_shares": transaction_obj.amount, "recorded_fill_price": transaction_obj.price,
            "actual_ledger_notional": notional_float, "decision_nav": nav_float,
            "decision_nav_source": "captured_native_pretrade_NAV" if execution_ts in trace_nav_dict else "preceding_result_or_initial_cash_fallback",
            "constant_1m_notional": model_notional_float, "native_trailing20_median_turnover": adv_float,
            "adv_status": "available" if adv_valid_bool else "missing_or_insufficient_native_turnover",
            "actual_participation": notional_float / adv_float if adv_valid_bool else np.nan,
            "constant_1m_participation": model_notional_float / adv_float if adv_valid_bool else np.nan,
            "decision_nominal_price": nominal_float,
            "rounding_status": "available" if nominal_valid_bool else "missing_decision_time_nominal_price",
            "one_raw_share_nav_weight_bound": nominal_float / nav_float if nominal_valid_bool else np.nan,
            "one_raw_share_1m_weight_bound": nominal_float / MODEL_CAPITAL_FLOAT if nominal_valid_bool else np.nan})
    order_df = pd.DataFrame(row_list)
    if order_df.empty:
        return {"order_count": 0, "missing_adv_order_count": 0, "interpretation": "no_executed_transactions"}, order_df
    missing_bool_ser = order_df["adv_status"] != "available"
    rounding_missing_bool_ser = order_df["rounding_status"] != "available"
    covered_ser = order_df.loc[~missing_bool_ser, "constant_1m_participation"]
    return {"order_count": len(order_df), "missing_adv_order_count": int(missing_bool_ser.sum()),
        "missing_adv_notional": order_df.loc[missing_bool_ser, "actual_ledger_notional"].sum(),
        "missing_adv_order_asset_list": sorted(order_df.loc[missing_bool_ser, "asset"].unique().tolist()),
        "missing_nominal_price_order_count": int(rounding_missing_bool_ser.sum()),
        "constant_1m_participation_available_subset": {"observations": len(covered_ser), "median": covered_ser.median(),
            "p95": covered_ser.quantile(.95), "p99": covered_ser.quantile(.99), "max": covered_ser.max()},
        "max_actual_ledger_participation": order_df["actual_participation"].max(),
        "max_one_raw_share_1m_weight_bound_available_subset": order_df["one_raw_share_1m_weight_bound"].max(),
        "decision_nav_fallback_order_count": int((order_df["decision_nav_source"] != "captured_native_pretrade_NAV").sum()),
        "interpretation": "Executed legs only, not canceled/unfilled strategy intents. Constant$1M order=abs(recordedshares*recordedfillprice)/decisionNAV*$1M. Participation is not opening-auction impact. MissingADV orders remain in denominator and explicit CSV rows.",
        "rounding_interpretation": "Theoretical live whole-raw-share target rounding is less than one nominal share: bound=UnadjustedClose_atDecision/$1M. This is not a bound on the legacy adjusted-share ledger's historical rounding, a measured execution error, or proof of raw-unit equivalence."}, order_df


def participation_summary(order_df: pd.DataFrame) -> dict:
    if order_df.empty:
        return {"status": "no_executed_transactions", "orders": 0, "missing_adv_orders": 0,
            "missing_adv_notional": 0.0}
    covered_bool_ser = order_df["adv_status"].eq("available")
    def distribution_dict(value_ser):
        return {"observations": len(value_ser), "median": value_ser.median(),
            "p95": value_ser.quantile(.95), "p99": value_ser.quantile(.99), "max": value_ser.max()}
    return {"status": "available", "orders": len(order_df), "missing_adv_orders": int((~covered_bool_ser).sum()),
        "missing_adv_notional": order_df.loc[~covered_bool_ser, "actual_ledger_notional"].sum(),
        "constant_1m_available_subset": distribution_dict(order_df.loc[covered_bool_ser, "constant_1m_participation"]),
        "actual_compounded_ledger_available_subset": distribution_dict(order_df.loc[covered_bool_ser, "actual_participation"])}


def liquidity_period_details(order_df: pd.DataFrame, pricing_df: pd.DataFrame, endpoint_ts: pd.Timestamp) -> dict:
    endpoint_ts = pd.Timestamp(endpoint_ts).normalize()
    start_ts = endpoint_ts - pd.DateOffset(years=3)
    if order_df.empty:
        return {"full": participation_summary(order_df), "last3y": participation_summary(order_df),
            "current_endpoint_date": str(endpoint_ts.date()), "asset_list": []}
    recent_df = order_df.loc[order_df["execution_date"] >= start_ts]
    asset_row_list = []
    for asset_str, asset_order_df in order_df.groupby("asset", sort=True):
        recent_asset_df = asset_order_df.loc[asset_order_df["execution_date"] >= start_ts]
        turnover_key = (asset_str, "Turnover")
        # *** CRITICAL *** This is an explicitly ENDPOINT diagnostic for a
        # prospective order after endpoint Close_T, not the liquidity known for
        # an old trade. The completed endpoint bar may enter this20day median.
        endpoint_turnover_ser = (pricing_df.loc[:endpoint_ts, turnover_key].astype(float).tail(20)
            if turnover_key in pricing_df.columns else pd.Series(dtype=float))
        valid_bool = (len(endpoint_turnover_ser) == 20 and endpoint_turnover_ser.index[-1] == endpoint_ts
            and np.isfinite(endpoint_turnover_ser.to_numpy()).all() and endpoint_turnover_ser.gt(0).all())
        endpoint_adv_float = float(endpoint_turnover_ser.median()) if valid_bool else np.nan
        current_dict = {"status": "available" if valid_bool else "missing_or_insufficient_endpoint_native_turnover",
            "as_of_completed_close": str(endpoint_ts.date()), "native_20session_median_turnover": endpoint_adv_float,
            "native_observation_count": len(endpoint_turnover_ser),
            "last3y_executed_order_count": len(recent_asset_df)}
        if valid_bool and not recent_asset_df.empty:
            model_fraction_ser = recent_asset_df["constant_1m_notional"] / endpoint_adv_float
            actual_fraction_ser = recent_asset_df["actual_ledger_notional"] / endpoint_adv_float
            current_dict.update(
                last3y_order_weights_at_fixed_1m_p99_of_endpoint_ADV=model_fraction_ser.quantile(.99),
                last3y_order_weights_at_fixed_1m_max_of_endpoint_ADV=model_fraction_ser.max(),
                last3y_actual_compounded_notionals_p99_of_endpoint_ADV=actual_fraction_ser.quantile(.99),
                last3y_actual_compounded_notionals_max_of_endpoint_ADV=actual_fraction_ser.max(),
                latest_recorded_order_date=str(recent_asset_df["execution_date"].iloc[-1].date()),
                latest_recorded_actual_notional=recent_asset_df["actual_ledger_notional"].iloc[-1],
                latest_recorded_constant_1m_notional=recent_asset_df["constant_1m_notional"].iloc[-1])
        asset_row_list.append({"asset": asset_str, "full": participation_summary(asset_order_df),
            "last3y": participation_summary(recent_asset_df), "current_endpoint": current_dict})
    return {"full": participation_summary(order_df), "last3y": participation_summary(recent_df),
        "last3y_start": str(start_ts.date()), "current_endpoint_date": str(endpoint_ts.date()),
        "historical_basis": "each historical order uses only preceding20 complete nativeTurnover sessions; missing orders stay explicit",
        "endpoint_basis": "last3y historical order weights normalized to$1M, and separately original compounded notionals, divided by median20 nativeTurnover through endpoint completed close. Retrospective sizing/capacity diagnostic, not historical-order liquidity, a current signal, a broker readiness gate or opening-auction impact.",
        "asset_list": asset_row_list}


def ledger_stress(equity_ser: pd.Series, cash_ser: pd.Series, transactions_df: pd.DataFrame,
        initial_float: float, extra_bps_float: float, financing_rate_float: float,
        anchor_ts=None) -> tuple[dict, pd.Series]:
    equity_ser = ordered_series(equity_ser, "equity")
    cash_ser = ordered_series(cash_ser, "cash").reindex(equity_ser.index)
    if cash_ser.isna().any():
        raise ValueError("Financing sensitivity requires every saved daily cash value.")
    transaction_df = transactions_df.copy()
    transaction_df["bar"] = pd.to_datetime(transaction_df["bar"]).dt.tz_localize(None).dt.normalize()
    transaction_df["extra_slippage"] = (transaction_df["amount"].astype(float) * transaction_df["price"].astype(float)).abs() * extra_bps_float / 10_000
    fee_ser = transaction_df.groupby("bar")["extra_slippage"].sum()
    off_calendar_idx = fee_ser.index.difference(equity_ser.index)
    if len(off_calendar_idx):
        raise ValueError(f"Transaction cash-cost dates absent from saved equity: {list(off_calendar_idx[:5])}")
    fee_ser = fee_ser.reindex(equity_ser.index, fill_value=0)
    # *** CRITICAL *** Financing uses PRIOR recorded close cash only. Initial
    # account is all cash, so first-day prior debt iszero. ACT/360 charges a
    # Friday negative cash balance for3days throughMonday. No future cash leaks.
    previous_cash_vec = np.r_[initial_float, cash_ser.to_numpy()[:-1]]
    calendar_day_vec = np.r_[0.0, np.diff(equity_ser.index.to_numpy()).astype("timedelta64[D]").astype(float)]
    financing_ser = pd.Series(np.maximum(-previous_cash_vec, 0) * financing_rate_float * calendar_day_vec / 360,
        index=equity_ser.index)
    # Fixed-ledger dollar debit: E_stress_T=E_original_T-sum(extra fees throughT).
    # Original shares, orders, turnover, dividends and debt are never recomputed.
    stressed_ser = equity_ser - (fee_ser + financing_ser).cumsum()
    if (stressed_ser <= 0).any():
        return {"status": "nonpositive_equity", "extra_slippage_bps": extra_bps_float,
            "financing_annual_rate": financing_rate_float, "first_nonpositive_date": str(stressed_ser.index[stressed_ser.le(0)][0].date())}, stressed_ser
    return {"status": "available", "extra_slippage_bps": extra_bps_float, "financing_annual_rate": financing_rate_float,
        "extra_slippage_total": fee_ser.sum(), "financing_total": financing_ser.sum(),
        "metrics": equity_metrics(stressed_ser, initial_float, anchor_ts),
        "convention": "ex-post fixed original trade ledger; extra fees subtract cumulatively from cash/NAV; ACT/360 financing on original prior negative cash; no sizing/turnover/borrow/dividend feedback; not a backtest variant"}, stressed_ser


def load_selected_prices(path_obj: Path, asset_set: set[str]) -> pd.DataFrame:
    selected_name_list = []
    for name_str in parquet_module.ParquetFile(path_obj).schema_arrow.names:
        try:
            column_obj = ast.literal_eval(name_str)
        except (ValueError, SyntaxError):
            column_obj = None
        if isinstance(column_obj, tuple) and len(column_obj) == 2:
            asset_str, field_str = column_obj
            if ((asset_str in asset_set and field_str in {"Turnover", "Unadjusted Close"})
                    or (asset_str in {"$SPX", "$SPXTR", "SPY", "ADAPTIVE_TR_SPY"} and field_str == "Close")):
                selected_name_list.append(name_str)
    return pd.read_parquet(path_obj, columns=selected_name_list)


def resolve_input_sources(baseline_path: Path) -> tuple[dict, list[Path], dict | None]:
    source_dict = {name_str: baseline_path / name_str for name_str in
        ("summary.json", "results.parquet", "pricing.parquet", "transactions.csv.gz", "trace.pkl.gz", "input_manifest.json")}
    if all(path_obj.exists() for path_obj in source_dict.values()):
        return source_dict, [], None
    if baseline_path.name != "strategy_mr_hpi_sp500_2_3_5_vote":
        raise FileNotFoundError("Baseline lacks required saved statistics inputs.")
    audit_root_path = baseline_path.parent.parent
    repaired_path = audit_root_path / "mr_hpi235_trace_only"
    qualification_path = repaired_path / "qualification.json"
    qualification_dict = json.loads(qualification_path.read_text(encoding="utf-8"))
    if not (qualification_dict.get("status_str") == "pass"
            and qualification_dict.get("exact_original_baseline_equity_bool") is True
            and qualification_dict.get("source_unchanged_bool") is True
            and qualification_dict.get("new_strategy_variants_int") == 0):
        raise ValueError("HPI saved-trace qualification is incomplete or failed.")
    checker_path = REPO_PATH / "scripts/research/wired_signoff_20261015/mr_hpi_cached_trace.py"
    if hash_dict(checker_path)["sha256"] != qualification_dict["checker_sha256_str"]:
        raise ValueError("HPI trace checker source changed since qualification.")
    canonical_path = audit_root_path / "input_preparation" / baseline_path.name / "pricing.parquet"
    if (hash_dict(canonical_path)["sha256"] != qualification_dict["canonical_input_sha256_str"]
            or hash_dict(repaired_path / "pricing.parquet")["sha256"] != qualification_dict["canonical_input_sha256_str"]):
        raise ValueError("HPI repaired pricing no longer equals the qualified canonical saved input.")
    extra_path_list = [qualification_path, checker_path, repaired_path / "results.parquet", repaired_path / "auxiliary.pkl.gz"]
    evidence_dict = json.loads((baseline_path / "input_manifest.json").read_text(encoding="utf-8"))
    for row_dict in evidence_dict["supplemental_saved_inputs"]:
        source_path = Path(row_dict["path"])
        if hash_dict(source_path)["sha256"] != row_dict["sha256"]:
            raise ValueError("HPI canonical supplemental input changed after original baseline.")
        extra_path_list.append(source_path)
    canonical_auxiliary_path = canonical_path.parent / "auxiliary.pkl.gz"
    pd.testing.assert_frame_equal(pd.read_pickle(canonical_auxiliary_path)["universe_df"],
        pd.read_pickle(repaired_path / "auxiliary.pkl.gz")["universe_df"], check_exact=True)
    recorder_mismatch_list = []
    for source_str, expected_str in qualification_dict["source_hash_dict"].items():
        source_path = REPO_PATH / source_str
        actual_str = hash_dict(source_path)["sha256"]
        if actual_str != expected_str:
            if source_str != "scripts/research/wired_signoff_20261015/mr_native_replay.py":
                raise ValueError(f"HPI qualified trace production source changed: {source_str}")
            archive_path_list = sorted(audit_root_path.glob("mr*/executed_replay_checker.py"))
            matching_archive_path = next((path_obj for path_obj in archive_path_list if hash_dict(path_obj)["sha256"] == expected_str), None)
            recorder_mismatch_list.append({"path": source_str, "qualified_historical_sha256": expected_str,
                "observed_current_sha256": actual_str,
                "matching_archived_source": None if matching_archive_path is None else str(matching_archive_path.resolve()),
                "searched_replay_archive_count": len(archive_path_list),
                "interpretation": "Audit recorder changed after qualification; historical fingerprint is retained, not certified as current. Statistics independently rechecks saved trace NAV/cash against original preceding closes."})
            if matching_archive_path is not None:
                source_path = matching_archive_path
        extra_path_list.append(source_path)
    qualification_dict["statistics_observed_historical_recorder_mismatch_list"] = recorder_mismatch_list
    source_dict.update({name_str: repaired_path / name_str for name_str in ("pricing.parquet", "trace.pkl.gz")})
    if not all(path_obj.exists() for path_obj in source_dict.values()):
        raise FileNotFoundError("Qualified HPI input source map remains incomplete.")
    return source_dict, extra_path_list, qualification_dict


def normalize_trace(trace_obj) -> list[dict]:
    if isinstance(trace_obj, dict):
        return [{**record_dict, "decision_date": decision_ts, "nav": record_dict["nav_float"],
                 "cash": record_dict["cash_float"]} for decision_ts, record_dict in sorted(trace_obj.items())]
    return trace_obj


def verify_repaired_hpi_trace(trace_list: list[dict], qualification_dict: dict,
        original_df: pd.DataFrame, repaired_df: pd.DataFrame, initial_float: float) -> dict:
    pd.testing.assert_frame_equal(original_df[["total_value", "cash"]], repaired_df[["total_value", "cash"]], check_exact=True)
    if len(trace_list) != qualification_dict["trace_session_count_int"]:
        raise ValueError("HPI repaired trace session count differs from qualification.")
    execution_date_set = set()
    for row_dict in trace_list:
        decision_ts, execution_ts = pd.Timestamp(row_dict["decision_date"]), pd.Timestamp(row_dict["execution_date"])
        if execution_ts in execution_date_set or not decision_ts < execution_ts or execution_ts not in original_df.index:
            raise ValueError("HPI repaired trace dates are duplicate or outside original execution calendar.")
        execution_date_set.add(execution_ts)
        execution_position_int = original_df.index.get_loc(execution_ts)
        if execution_position_int == 0:
            expected_nav_float = expected_cash_float = initial_float
        else:
            if decision_ts != original_df.index[execution_position_int - 1]:
                raise ValueError("HPI trace decision is not the immediately preceding saved close.")
            expected_nav_float = float(original_df.loc[decision_ts, "total_value"])
            expected_cash_float = float(original_df.loc[decision_ts, "cash"])
        # *** CRITICAL *** A repaired trace may normalize historical orders
        # only when its pre-fill Close_T NAV/cash matches ORIGINAL baseline T.
        # Exact equity alone is insufficient to establish this cash boundary.
        if not np.isclose(row_dict["nav"], expected_nav_float, rtol=0, atol=1e-8) or not np.isclose(row_dict["cash"], expected_cash_float, rtol=0, atol=1e-8):
            raise ValueError("HPI repaired decision NAV/cash differs from the original baseline close.")
    return {"status": "verified", "exact_original_equity_and_cash": True,
        "original_close_nav_cash_boundary_checked_sessions": len(trace_list),
        "historical_recorder_matches_current_source": not qualification_dict.get("statistics_observed_historical_recorder_mismatch_list", []),
        "historical_recorder_source_disclosure": "Historical recorder hashes are retained in qualification. Any mismatch and exact-archive availability are listed explicitly; independent saved-output comparators do not assert unchanged recorder source.",
        "qualification": qualification_dict,
        "limitation": "Only pricing and decision trace are supplied by the qualified repair; original baseline results and transaction ledger remain authoritative. No additional engine run or independent transaction repeat was performed by statistics."}


def process_baseline(baseline_path: Path, output_root_path: Path) -> dict:
    output_path = output_root_path / baseline_path.name
    summary_path = baseline_path / "summary.json"
    summary_dict = json.loads(summary_path.read_text(encoding="utf-8"))
    complete_bool = (summary_dict.get("status") == "pass"
        and summary_dict.get("deterministic_equity") is True
        and summary_dict.get("deterministic_economic_transactions") is True
        and summary_dict.get("capital_honored") is True
        and summary_dict.get("actual_capital_base") == MODEL_CAPITAL_FLOAT
        and summary_dict.get("actual_double_capital_base") == 2 * MODEL_CAPITAL_FLOAT
        and summary_dict.get("seconds") is not None)
    if not complete_bool:
        return {"key": baseline_path.name, "status": "not_processed", "reason": "baseline_not_final_repeat_and_capital_pass"}
    input_source_dict, extra_source_list, qualification_dict = resolve_input_sources(baseline_path)
    input_path_list = list(dict.fromkeys([*input_source_dict.values(), *extra_source_list]))
    input_manifest_list = [hash_dict(path_obj) for path_obj in input_path_list]
    source_manifest_dict = hash_dict(Path(__file__))
    results_df = pd.read_parquet(input_source_dict["results.parquet"])
    equity_ser = ordered_series(results_df["total_value"], "equity")
    cash_ser = ordered_series(results_df["cash"], "cash")
    transaction_df = pd.read_csv(input_source_dict["transactions.csv.gz"])
    with gzip.open(input_source_dict["trace.pkl.gz"], "rb") as handle_obj:
        trace_list = normalize_trace(pickle.load(handle_obj))
    pricing_df = load_selected_prices(input_source_dict["pricing.parquet"], set(transaction_df["asset"].astype(str)))
    pricing_df.index = pd.DatetimeIndex(pd.to_datetime(pricing_df.index)).tz_localize(None)
    initial_float = float(summary_dict.get("capital", MODEL_CAPITAL_FLOAT))
    trace_qualification_dict = (None if qualification_dict is None else verify_repaired_hpi_trace(
        trace_list, qualification_dict, results_df, pd.read_parquet(input_source_dict["trace.pkl.gz"].parent / "results.parquet"), initial_float))
    return_ser = full_return_series(equity_ser, initial_float)
    annual_ser = return_ser.groupby(return_ser.index.year).agg(lambda year_return_ser: (1 + year_return_ser).prod() - 1)
    monthly_ser = return_ser.groupby(return_ser.index.to_period("M")).agg(lambda month_return_ser: (1 + month_return_ser).prod() - 1)
    prior_price_idx = pricing_df.index[pricing_df.index < equity_ser.index[0]]
    initial_anchor_ts = prior_price_idx[-1] if len(prior_price_idx) else None
    report_dict = {"key": baseline_path.name, "status": "pass", "generated_at_utc": datetime.now(UTC).isoformat(),
        "initial_capital": initial_float, "model_order_capital": MODEL_CAPITAL_FLOAT,
        "standalone_metrics": equity_metrics(equity_ser, initial_float, initial_anchor_ts),
        "standalone_convention": "cash beforefirst recorded session included; CAGR from prior saved price-session close where available; sampleSD,252 sessions,RF=0",
        "subperiods": {}, "benchmarks": {}, "annual_returns": {str(year_int): value_float for year_int, value_float in annual_ser.items()},
        "calendar_monthly_returns": {str(month_obj): value_float for month_obj, value_float in monthly_ser.items()},
        "boundary_period_note": "First/last calendar years and months may be partial; no pre-inception or unobserved returns invented.",
        "input_manifest": input_manifest_list,
        "artifact_source_map": {name_str: str(path_obj.resolve()) for name_str, path_obj in input_source_dict.items()},
        "trace_qualification": trace_qualification_dict,
        "baseline_input_evidence": json.loads((baseline_path / "input_manifest.json").read_text(encoding="utf-8")),
        "baseline_completion_evidence": {name_str: summary_dict[name_str] for name_str in
            ("deterministic_equity", "deterministic_economic_transactions", "capital_honored",
             "actual_capital_base", "actual_double_capital_base", "seconds")},
        "statistics_source": source_manifest_dict, "saved_pricing_attrs": pricing_df.attrs,
        "upstream_source_manifest": hash_dict(baseline_path.parent.parent / "frozen_manifest.json") if (baseline_path.parent.parent / "frozen_manifest.json").exists() else None,
        "limitations": ["Current-vintage saved input; not historical saved decision-time data and no untouched holdout.",
            "No native/provider/broker access. Completeness depends on saved baseline artifacts and captured trace.",
            "NativeTurnover median is aggregate daily capacity, not opening-auction liquidity or marketimpact.",
            "Only executed transactions enter dollar-cost/ADV diagnostics; failed/canceled/unfilled intent is not measured here.",
            "One nominalshare rounding bound does not repair adjusted-share historical commissions or validate broker sizing.",
            "Extra cost scenarios preserve original positions/trades and ignore cash/turnover feedback; they cannot qualify live execution or replace native strategy robustness tests.",
            "Daily Sharpe usesRF=0. Comparison correlations/beta are descriptive and not evidence of causal alpha."]}
    for label_str, start_str, end_str in (("since2016", "2016-01-01", None), ("last3y", "2023-10-15", None),
            ("covid2020", "2020-01-01", "2020-12-31"), ("rates2022", "2022-01-01", "2022-12-31")):
        report_dict["subperiods"][label_str] = subperiod_dict(equity_ser, initial_float, start_str, end_str)
    peak_vec = np.maximum.accumulate(np.r_[initial_float, equity_ser.to_numpy()])[1:]
    chart_df = pd.DataFrame({"equity": equity_ser, "return": return_ser, "drawdown": equity_ser / peak_vec - 1})
    for label_str, benchmark_dict in select_benchmarks(pricing_df).items():
        if "series" not in benchmark_dict:
            report_dict["benchmarks"][label_str] = benchmark_dict
            continue
        comparison_dict, comparison_chart_df = common_close_comparison(equity_ser, benchmark_dict["series"])
        report_dict["benchmarks"][label_str] = {**comparison_dict, **{key_str: value_obj for key_str, value_obj in benchmark_dict.items() if key_str != "series"}}
        if not comparison_chart_df.empty:
            chart_df = chart_df.join(comparison_chart_df.add_prefix(label_str + "_"))
    report_dict["liquidity"], order_df = order_diagnostics(transaction_df, pricing_df, equity_ser, initial_float, trace_list)
    report_dict["liquidity"]["period_details"] = liquidity_period_details(order_df, pricing_df, equity_ser.index[-1])
    report_dict["cost_sensitivities"] = {}
    for label_str, bps_float, funding_float in (("extra2_5bps", 2.5, 0), ("extra7_5bps", 7.5, 0),
            ("finance4pct", 0, .04), ("finance8pct", 0, .08), ("extra2_5bps_finance4pct", 2.5, .04),
            ("extra7_5bps_finance8pct", 7.5, .08)):
        stress_dict, stressed_ser = ledger_stress(equity_ser, cash_ser, transaction_df, initial_float, bps_float, funding_float, initial_anchor_ts)
        report_dict["cost_sensitivities"][label_str] = stress_dict
        chart_df[label_str + "_equity"] = stressed_ser
    verify_input_manifest([*input_manifest_list, source_manifest_dict])
    report_dict["input_source_hashes_unchanged_at_completion"] = True
    output_path.mkdir(parents=True, exist_ok=True)
    chart_df.to_csv(output_path / "daily_chart.csv.gz", index_label="date")
    order_df.to_csv(output_path / "order_diagnostics.csv.gz", index=False)
    report_dict["chart_data_manifest"] = [hash_dict(output_path / name_str) for name_str in ("daily_chart.csv.gz", "order_diagnostics.csv.gz")]
    write_json(output_path / "statistics.json", report_dict)
    return {"key": baseline_path.name, "status": "pass", "statistics_path": str((output_path / "statistics.json").resolve()),
        "missing_adv_orders": report_dict["liquidity"]["missing_adv_order_count"]}


def main() -> None:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--baseline-root", type=Path, default=DEFAULT_OUTPUT_PATH.parent / "baselines")
    parser_obj.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser_obj.add_argument("--only", nargs="*")
    argument_obj = parser_obj.parse_args()
    row_list = []
    for baseline_path in sorted(argument_obj.baseline_root.iterdir()):
        if not baseline_path.is_dir() or not (baseline_path / "summary.json").exists():
            continue
        if argument_obj.only and not any(key_str in baseline_path.name for key_str in argument_obj.only):
            continue
        try:
            row_dict = process_baseline(baseline_path, argument_obj.output_root)
        except Exception as exception_obj:
            row_dict = {"key": baseline_path.name, "status": "error", "error": f"{type(exception_obj).__name__}: {exception_obj}"}
        row_list.append(row_dict)
        update_index(argument_obj.output_root / "index.json", row_list)
        print(json.dumps(row_dict), flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
