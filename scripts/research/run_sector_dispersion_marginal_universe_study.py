from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    DEFAULT_CONFIG,
    ORIGINAL_SYMBOL_TUPLE,
    SectorDispersionIbsConfig,
    SectorDispersionIbsStrategy,
    UNIVERSE_A_SYMBOL_TUPLE,
    UNIVERSE_B_SYMBOL_TUPLE,
    UNIVERSE_C_SYMBOL_TUPLE,
    get_sector_dispersion_ibs_data,
)


IN_SAMPLE_END_TS = pd.Timestamp("2021-12-31")
OUT_OF_SAMPLE_START_TS = pd.Timestamp("2022-01-01")

ACCEPTANCE_RULE_DICT = {
    "min_delta_oos_sharpe_float": 0.0,
    "min_delta_oos_max_drawdown_pct_float": -3.0,
    "max_corr_to_baseline_float": 0.80,
    "min_delta_full_sharpe_float": -0.10,
    "max_delta_cost_drag_ann_pct_float": 0.20,
}

STRESS_RULE_DICT = {
    "base_tail_quantile_float": 0.05,
    "market_tail_quantile_float": 0.10,
    "min_base_tail_delta_mean_return_pct_float": 0.0,
    "min_market_tail_delta_mean_return_pct_float": 0.0,
    "max_base_tail_corr_to_baseline_float": 0.80,
    "max_market_tail_corr_to_baseline_float": 0.80,
    "min_base_tail_candidate_active_pct_float": 5.0,
    "min_market_tail_candidate_active_pct_float": 5.0,
}

CANDIDATE_DESCRIPTION_DICT = {
    "XLF": ("Universe A", "sector", "Financials"),
    "XLE": ("Universe A", "sector", "Energy"),
    "XLI": ("Universe A", "sector", "Industrials"),
    "XLY": ("Universe A", "sector", "Consumer discretionary"),
    "XLP": ("Universe A", "sector", "Consumer staples"),
    "XLU": ("Universe A", "sector", "Utilities"),
    "XLRE": ("Universe A", "sector", "Real estate"),
    "XLB": ("Universe A", "sector", "Materials"),
    "XLC": ("Universe A", "sector", "Communication services"),
    "KRE": ("Universe B", "subsector", "Regional banks"),
    "XOP": ("Universe B", "subsector", "Oil and gas exploration"),
    "ITA": ("Universe B", "subsector", "Aerospace and defense"),
    "XRT": ("Universe B", "subsector", "Retail"),
    "ITB": ("Universe B", "subsector", "Home construction"),
    "XME": ("Universe B", "subsector", "Metals and mining"),
    "IHI": ("Universe B", "subsector", "Medical devices"),
    "XBI": ("Universe C", "research_only", "Biotechnology"),
    "KIE": ("Universe C", "research_only", "Insurance"),
    "IAI": ("Universe C", "research_only", "Broker dealers"),
    "IYT": ("Universe C", "research_only", "Transportation"),
    "IHF": ("Universe C", "research_only", "Healthcare providers"),
    "IHE": ("Universe C", "research_only", "Pharmaceuticals"),
    "XHB": ("Universe C", "research_only", "Homebuilders"),
    "XAR": ("Universe C", "research_only", "Aerospace and defense"),
    "XES": ("Universe C", "research_only", "Energy equipment and services"),
}


def build_candidate_manifest_df() -> pd.DataFrame:
    original_symbol_set = set(ORIGINAL_SYMBOL_TUPLE)
    universe_a_addition_list = [symbol_str for symbol_str in UNIVERSE_A_SYMBOL_TUPLE if symbol_str not in original_symbol_set]
    universe_b_addition_list = [symbol_str for symbol_str in UNIVERSE_B_SYMBOL_TUPLE if symbol_str not in UNIVERSE_A_SYMBOL_TUPLE]
    universe_c_addition_list = [symbol_str for symbol_str in UNIVERSE_C_SYMBOL_TUPLE if symbol_str not in UNIVERSE_B_SYMBOL_TUPLE]
    candidate_symbol_list = universe_a_addition_list + universe_b_addition_list + universe_c_addition_list

    if len(candidate_symbol_list) != len(set(candidate_symbol_list)):
        raise RuntimeError("Candidate manifest contains duplicate symbols.")

    row_dict_list: list[dict[str, object]] = []
    for manifest_rank_int, symbol_str in enumerate(candidate_symbol_list, start=1):
        source_universe_str, bucket_str, description_str = CANDIDATE_DESCRIPTION_DICT[symbol_str]
        row_dict_list.append(
            {
                "manifest_rank_int": manifest_rank_int,
                "symbol_str": symbol_str,
                "source_universe_str": source_universe_str,
                "bucket_str": bucket_str,
                "description_str": description_str,
                "baseline_symbol_tuple_str": ",".join(ORIGINAL_SYMBOL_TUPLE),
                "marginal_symbol_tuple_str": ",".join(ORIGINAL_SYMBOL_TUPLE + (symbol_str,)),
            }
        )

    return pd.DataFrame(row_dict_list)


def _slug_str(raw_value_str: str) -> str:
    keep_char_list: list[str] = []
    for char_str in raw_value_str.lower():
        if char_str.isalnum():
            keep_char_list.append(char_str)
        else:
            keep_char_list.append("_")
    return "_".join(filter(None, "".join(keep_char_list).split("_")))


def _summary_value_obj(summary_df: pd.DataFrame | None, metric_name_str: str) -> object | None:
    if summary_df is None or metric_name_str not in summary_df.index:
        return None
    value_obj = summary_df.loc[metric_name_str]
    if isinstance(value_obj, pd.Series):
        value_obj = value_obj.iloc[0]
    if pd.isna(value_obj):
        return None
    return value_obj


def _summary_value_float(summary_df: pd.DataFrame | None, metric_name_str: str) -> float | None:
    value_obj = _summary_value_obj(summary_df=summary_df, metric_name_str=metric_name_str)
    if value_obj is None:
        return None
    return float(value_obj)


def _summary_value_str(summary_df: pd.DataFrame | None, metric_name_str: str) -> str | None:
    value_obj = _summary_value_obj(summary_df=summary_df, metric_name_str=metric_name_str)
    if value_obj is None:
        return None
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.date().isoformat()
    return str(value_obj)


def _safe_float(value_obj: object) -> float:
    if value_obj is None:
        return float("nan")
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value_float):
        return float("nan")
    return value_float


def _safe_delta_float(left_obj: object, right_obj: object) -> float:
    left_float = _safe_float(left_obj)
    right_float = _safe_float(right_obj)
    if not np.isfinite(left_float) or not np.isfinite(right_float):
        return float("nan")
    return left_float - right_float


def compute_period_metric_dict(
    total_value_ser: pd.Series,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None,
    prefix_str: str,
) -> dict[str, object]:
    normalized_total_value_ser = pd.to_numeric(total_value_ser, errors="coerce").copy()
    normalized_total_value_ser.index = pd.to_datetime(normalized_total_value_ser.index).normalize()

    period_total_value_ser = normalized_total_value_ser.loc[normalized_total_value_ser.index >= start_ts]
    if end_ts is not None:
        period_total_value_ser = period_total_value_ser.loc[period_total_value_ser.index <= end_ts]
    period_total_value_ser = period_total_value_ser.dropna()

    metric_dict: dict[str, object] = {
        f"{prefix_str}_start_date_str": None,
        f"{prefix_str}_end_date_str": None,
        f"{prefix_str}_day_count_int": int(len(period_total_value_ser)),
        f"{prefix_str}_ann_return_pct_float": np.nan,
        f"{prefix_str}_volatility_ann_pct_float": np.nan,
        f"{prefix_str}_sharpe_float": np.nan,
        f"{prefix_str}_max_drawdown_pct_float": np.nan,
    }
    if len(period_total_value_ser) < 2:
        return metric_dict

    # *** CRITICAL*** These split metrics are post-run diagnostics only. They
    # use realized equity after the backtest has completed and must never feed
    # signal construction, sizing, or candidate order inside a backtest.
    period_daily_return_ser = period_total_value_ser.pct_change(fill_method=None).dropna()
    running_peak_ser = period_total_value_ser.cummax()
    drawdown_ser = period_total_value_ser / running_peak_ser - 1.0
    volatility_float = float(period_daily_return_ser.std() * np.sqrt(252.0) * 100.0)
    mean_return_float = float(period_daily_return_ser.mean())
    std_return_float = float(period_daily_return_ser.std())
    sharpe_float = np.nan if std_return_float == 0.0 else mean_return_float / std_return_float * np.sqrt(252.0)

    metric_dict.update(
        {
            f"{prefix_str}_start_date_str": period_total_value_ser.index[0].date().isoformat(),
            f"{prefix_str}_end_date_str": period_total_value_ser.index[-1].date().isoformat(),
            f"{prefix_str}_ann_return_pct_float": (
                (period_total_value_ser.iloc[-1] / period_total_value_ser.iloc[0])
                ** (252.0 / float(len(period_total_value_ser)))
                - 1.0
            )
            * 100.0,
            f"{prefix_str}_volatility_ann_pct_float": volatility_float,
            f"{prefix_str}_sharpe_float": sharpe_float,
            f"{prefix_str}_max_drawdown_pct_float": float(drawdown_ser.min() * 100.0),
        }
    )
    return metric_dict


def _run_strategy_variant(
    strategy_name_str: str,
    symbol_tuple: tuple[str, ...],
    base_config_obj: SectorDispersionIbsConfig,
    pricing_data_df: pd.DataFrame,
    show_progress_bool: bool,
) -> SectorDispersionIbsStrategy:
    config_obj = replace(
        base_config_obj,
        symbol_tuple=tuple(symbol_tuple),
        universe_name_str="original",
    )
    strategy_obj = SectorDispersionIbsStrategy(
        name=strategy_name_str,
        benchmarks=[config_obj.benchmark_symbol_str],
        config_obj=config_obj,
    )

    # *** CRITICAL*** Keep pre-start history for the lagged range scale, but
    # only execute orders on and after backtest_start_date_str. The strategy
    # still maps signal T to house-standard fill Open T+1.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=False,
    )
    return strategy_obj


def _strategy_summary_row_dict(
    strategy_obj: SectorDispersionIbsStrategy,
    variant_kind_str: str,
    candidate_symbol_str: str | None,
    bucket_str: str | None,
) -> dict[str, object]:
    total_value_ser = strategy_obj.results["total_value"]
    summary_df = strategy_obj.summary
    summary_trades_df = strategy_obj.summary_trades
    trade_count_float = _summary_value_float(summary_trades_df, "# Trades")

    row_dict: dict[str, object] = {
        "variant_kind_str": variant_kind_str,
        "strategy_name_str": strategy_obj.name,
        "candidate_symbol_str": candidate_symbol_str,
        "bucket_str": bucket_str,
        "symbol_count_int": len(strategy_obj.symbol_tuple),
        "symbol_tuple_str": ",".join(strategy_obj.symbol_tuple),
        "start_date_str": _summary_value_str(summary_df, "Start"),
        "end_date_str": _summary_value_str(summary_df, "End"),
        "ann_return_pct_float": _summary_value_float(summary_df, "Return (Ann.) [%]"),
        "volatility_ann_pct_float": _summary_value_float(summary_df, "Volatility (Ann.) [%]"),
        "sharpe_float": _summary_value_float(summary_df, "Sharpe Ratio"),
        "max_drawdown_pct_float": _summary_value_float(summary_df, "Max. Drawdown [%]"),
        "mar_float": _summary_value_float(summary_df, "MAR Ratio"),
        "exposure_time_pct_float": _summary_value_float(summary_df, "Exposure Time [%]"),
        "turnover_ann_pct_float": _summary_value_float(summary_df, "Turnover (Ann.) [%]"),
        "cost_drag_ann_pct_float": _summary_value_float(summary_df, "Cost Drag (Ann.) [%]"),
        "trade_count_int": None if trade_count_float is None else int(trade_count_float),
    }
    row_dict.update(
        compute_period_metric_dict(
            total_value_ser=total_value_ser,
            start_ts=pd.Timestamp(strategy_obj.config_obj.backtest_start_date_str),
            end_ts=IN_SAMPLE_END_TS,
            prefix_str="in_sample",
        )
    )
    row_dict.update(
        compute_period_metric_dict(
            total_value_ser=total_value_ser,
            start_ts=OUT_OF_SAMPLE_START_TS,
            end_ts=None,
            prefix_str="oos",
        )
    )
    return row_dict


def _daily_return_ser(strategy_obj: SectorDispersionIbsStrategy) -> pd.Series:
    daily_return_ser = pd.to_numeric(strategy_obj.results["daily_returns"], errors="coerce").copy()
    daily_return_ser.index = pd.to_datetime(daily_return_ser.index).normalize()
    return daily_return_ser


def _active_bool_ser(strategy_obj: SectorDispersionIbsStrategy) -> pd.Series:
    result_index = pd.to_datetime(strategy_obj.results.index).normalize()
    realized_weight_df = getattr(strategy_obj, "realized_weight_df", pd.DataFrame())
    if realized_weight_df is None or len(realized_weight_df) == 0:
        return pd.Series(False, index=result_index)

    weight_df = realized_weight_df.copy()
    weight_df.index = pd.to_datetime(weight_df.index).normalize()
    weight_df.columns = [str(column_obj) for column_obj in weight_df.columns]
    symbol_column_list = [symbol_str for symbol_str in strategy_obj.symbol_tuple if symbol_str in weight_df.columns]
    if len(symbol_column_list) == 0:
        return pd.Series(False, index=result_index)

    active_bool_ser = weight_df[symbol_column_list].fillna(0.0).abs().sum(axis=1).gt(1e-9)
    return active_bool_ser.reindex(result_index, fill_value=False).astype(bool)


def _correlation_float(left_return_ser: pd.Series, right_return_ser: pd.Series) -> float:
    aligned_return_df = pd.concat(
        [left_return_ser.rename("left"), right_return_ser.rename("right")],
        axis=1,
    ).dropna()
    if len(aligned_return_df) < 2:
        return float("nan")
    if float(aligned_return_df["left"].std()) == 0.0 or float(aligned_return_df["right"].std()) == 0.0:
        return float("nan")
    return float(aligned_return_df["left"].corr(aligned_return_df["right"]))


def _bottom_tail_bool_ser(return_ser: pd.Series, quantile_float: float) -> pd.Series:
    normalized_return_ser = pd.to_numeric(return_ser, errors="coerce").replace([np.inf, -np.inf], np.nan)
    normalized_return_ser.index = pd.to_datetime(normalized_return_ser.index).normalize()
    realized_return_ser = normalized_return_ser.dropna()
    if len(realized_return_ser) > 1:
        # *** CRITICAL*** Drop the first return row because it is the backtest
        # anchor, not a realized trading-day return generated by the strategy.
        realized_return_ser = realized_return_ser.iloc[1:]
    if len(realized_return_ser) == 0:
        return pd.Series(False, index=normalized_return_ser.index)

    threshold_float = float(realized_return_ser.quantile(float(quantile_float)))
    tail_bool_ser = normalized_return_ser.le(threshold_float) & normalized_return_ser.notna()
    return tail_bool_ser.reindex(normalized_return_ser.index, fill_value=False).astype(bool)


def _tail_metric_dict(
    prefix_str: str,
    tail_bool_ser: pd.Series,
    baseline_return_ser: pd.Series,
    standalone_return_ser: pd.Series,
    marginal_return_ser: pd.Series,
    benchmark_return_ser: pd.Series,
    standalone_active_bool_ser: pd.Series,
) -> dict[str, object]:
    aligned_tail_df = pd.concat(
        [
            baseline_return_ser.rename("baseline_return"),
            standalone_return_ser.rename("standalone_return"),
            marginal_return_ser.rename("marginal_return"),
            benchmark_return_ser.rename("benchmark_return"),
            standalone_active_bool_ser.rename("standalone_active_bool"),
            tail_bool_ser.rename("tail_bool"),
        ],
        axis=1,
    )
    aligned_tail_df["tail_bool"] = aligned_tail_df["tail_bool"].astype("boolean").fillna(False).astype(bool)
    tail_return_df = aligned_tail_df.loc[aligned_tail_df["tail_bool"]].dropna(
        subset=["baseline_return", "standalone_return", "marginal_return"]
    )

    metric_dict: dict[str, object] = {
        f"{prefix_str}_day_count_int": int(len(tail_return_df)),
        f"{prefix_str}_baseline_mean_return_pct_float": np.nan,
        f"{prefix_str}_standalone_mean_return_pct_float": np.nan,
        f"{prefix_str}_marginal_mean_return_pct_float": np.nan,
        f"{prefix_str}_delta_mean_return_pct_float": np.nan,
        f"{prefix_str}_standalone_corr_to_baseline_float": np.nan,
        f"{prefix_str}_candidate_active_pct_float": np.nan,
        f"{prefix_str}_marginal_beats_base_pct_float": np.nan,
    }
    if len(tail_return_df) == 0:
        return metric_dict

    baseline_tail_return_ser = tail_return_df["baseline_return"].astype(float)
    standalone_tail_return_ser = tail_return_df["standalone_return"].astype(float)
    marginal_tail_return_ser = tail_return_df["marginal_return"].astype(float)
    active_tail_bool_ser = (
        tail_return_df["standalone_active_bool"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    metric_dict.update(
        {
            f"{prefix_str}_baseline_mean_return_pct_float": float(baseline_tail_return_ser.mean() * 100.0),
            f"{prefix_str}_standalone_mean_return_pct_float": float(standalone_tail_return_ser.mean() * 100.0),
            f"{prefix_str}_marginal_mean_return_pct_float": float(marginal_tail_return_ser.mean() * 100.0),
            f"{prefix_str}_delta_mean_return_pct_float": float(
                (marginal_tail_return_ser - baseline_tail_return_ser).mean() * 100.0
            ),
            f"{prefix_str}_standalone_corr_to_baseline_float": _correlation_float(
                standalone_tail_return_ser,
                baseline_tail_return_ser,
            ),
            f"{prefix_str}_candidate_active_pct_float": float(active_tail_bool_ser.mean() * 100.0),
            f"{prefix_str}_marginal_beats_base_pct_float": float(
                marginal_tail_return_ser.gt(baseline_tail_return_ser).mean() * 100.0
            ),
        }
    )
    benchmark_tail_return_ser = pd.to_numeric(tail_return_df["benchmark_return"], errors="coerce").dropna()
    if len(benchmark_tail_return_ser) > 0:
        metric_dict[f"{prefix_str}_benchmark_mean_return_pct_float"] = float(
            benchmark_tail_return_ser.mean() * 100.0
        )
    else:
        metric_dict[f"{prefix_str}_benchmark_mean_return_pct_float"] = np.nan
    return metric_dict


def compute_tail_stress_metric_dict(
    baseline_return_ser: pd.Series,
    standalone_return_ser: pd.Series,
    marginal_return_ser: pd.Series,
    benchmark_return_ser: pd.Series,
    standalone_active_bool_ser: pd.Series,
) -> dict[str, object]:
    base_tail_bool_ser = _bottom_tail_bool_ser(
        baseline_return_ser,
        STRESS_RULE_DICT["base_tail_quantile_float"],
    )
    market_tail_bool_ser = _bottom_tail_bool_ser(
        benchmark_return_ser,
        STRESS_RULE_DICT["market_tail_quantile_float"],
    )

    # *** CRITICAL*** Stress diagnostics are computed only after the strategy
    # runs have completed. They are ranking/report outputs and must not feed
    # signal construction, sizing, execution timing, or candidate ordering.
    metric_dict = _tail_metric_dict(
        prefix_str="base_tail",
        tail_bool_ser=base_tail_bool_ser,
        baseline_return_ser=baseline_return_ser,
        standalone_return_ser=standalone_return_ser,
        marginal_return_ser=marginal_return_ser,
        benchmark_return_ser=benchmark_return_ser,
        standalone_active_bool_ser=standalone_active_bool_ser,
    )
    metric_dict.update(
        _tail_metric_dict(
            prefix_str="market_tail",
            tail_bool_ser=market_tail_bool_ser,
            baseline_return_ser=baseline_return_ser,
            standalone_return_ser=standalone_return_ser,
            marginal_return_ser=marginal_return_ser,
            benchmark_return_ser=benchmark_return_ser,
            standalone_active_bool_ser=standalone_active_bool_ser,
        )
    )
    return metric_dict


def evaluate_acceptance_rule(diagnostic_row_dict: dict[str, object]) -> tuple[bool, str]:
    reject_reason_list: list[str] = []
    delta_oos_sharpe_float = _safe_float(diagnostic_row_dict.get("delta_oos_sharpe_float"))
    delta_oos_drawdown_float = _safe_float(diagnostic_row_dict.get("delta_oos_max_drawdown_pct_float"))
    corr_float = _safe_float(diagnostic_row_dict.get("standalone_corr_to_baseline_float"))
    delta_full_sharpe_float = _safe_float(diagnostic_row_dict.get("delta_full_sharpe_float"))
    delta_cost_drag_float = _safe_float(diagnostic_row_dict.get("delta_cost_drag_ann_pct_float"))

    if (
        not np.isfinite(delta_oos_sharpe_float)
        or delta_oos_sharpe_float < ACCEPTANCE_RULE_DICT["min_delta_oos_sharpe_float"]
    ):
        reject_reason_list.append("oos_sharpe_not_better")
    if (
        not np.isfinite(delta_oos_drawdown_float)
        or delta_oos_drawdown_float < ACCEPTANCE_RULE_DICT["min_delta_oos_max_drawdown_pct_float"]
    ):
        reject_reason_list.append("oos_drawdown_worse_than_limit")
    if not np.isfinite(corr_float) or corr_float > ACCEPTANCE_RULE_DICT["max_corr_to_baseline_float"]:
        reject_reason_list.append("standalone_corr_too_high_or_missing")
    if (
        not np.isfinite(delta_full_sharpe_float)
        or delta_full_sharpe_float < ACCEPTANCE_RULE_DICT["min_delta_full_sharpe_float"]
    ):
        reject_reason_list.append("full_sharpe_degraded_too_much")
    if (
        not np.isfinite(delta_cost_drag_float)
        or delta_cost_drag_float > ACCEPTANCE_RULE_DICT["max_delta_cost_drag_ann_pct_float"]
    ):
        reject_reason_list.append("cost_drag_too_high")

    return len(reject_reason_list) == 0, ";".join(reject_reason_list)


def evaluate_stress_rule(diagnostic_row_dict: dict[str, object]) -> tuple[bool, str]:
    reject_reason_list: list[str] = []
    if not bool(diagnostic_row_dict.get("accept_bool", False)):
        reject_reason_list.append("average_rule_failed")

    base_tail_delta_float = _safe_float(diagnostic_row_dict.get("base_tail_delta_mean_return_pct_float"))
    market_tail_delta_float = _safe_float(diagnostic_row_dict.get("market_tail_delta_mean_return_pct_float"))
    base_tail_corr_float = _safe_float(diagnostic_row_dict.get("base_tail_standalone_corr_to_baseline_float"))
    market_tail_corr_float = _safe_float(diagnostic_row_dict.get("market_tail_standalone_corr_to_baseline_float"))
    base_tail_active_float = _safe_float(diagnostic_row_dict.get("base_tail_candidate_active_pct_float"))
    market_tail_active_float = _safe_float(diagnostic_row_dict.get("market_tail_candidate_active_pct_float"))

    if (
        not np.isfinite(base_tail_delta_float)
        or base_tail_delta_float < STRESS_RULE_DICT["min_base_tail_delta_mean_return_pct_float"]
    ):
        reject_reason_list.append("base_tail_not_helpful")
    if (
        not np.isfinite(market_tail_delta_float)
        or market_tail_delta_float < STRESS_RULE_DICT["min_market_tail_delta_mean_return_pct_float"]
    ):
        reject_reason_list.append("market_tail_not_helpful")
    if (
        not np.isfinite(base_tail_corr_float)
        or base_tail_corr_float > STRESS_RULE_DICT["max_base_tail_corr_to_baseline_float"]
    ):
        reject_reason_list.append("base_tail_corr_too_high_or_missing")
    if (
        not np.isfinite(market_tail_corr_float)
        or market_tail_corr_float > STRESS_RULE_DICT["max_market_tail_corr_to_baseline_float"]
    ):
        reject_reason_list.append("market_tail_corr_too_high_or_missing")
    if (
        not np.isfinite(base_tail_active_float)
        or base_tail_active_float < STRESS_RULE_DICT["min_base_tail_candidate_active_pct_float"]
    ):
        reject_reason_list.append("base_tail_candidate_inactive")
    if (
        not np.isfinite(market_tail_active_float)
        or market_tail_active_float < STRESS_RULE_DICT["min_market_tail_candidate_active_pct_float"]
    ):
        reject_reason_list.append("market_tail_candidate_inactive")

    return len(reject_reason_list) == 0, ";".join(reject_reason_list)


def _diagnostic_row_dict(
    candidate_row_ser: pd.Series,
    baseline_strategy_obj: SectorDispersionIbsStrategy,
    standalone_strategy_obj: SectorDispersionIbsStrategy,
    marginal_strategy_obj: SectorDispersionIbsStrategy,
    benchmark_return_ser: pd.Series,
    baseline_summary_dict: dict[str, object],
    standalone_summary_dict: dict[str, object],
    marginal_summary_dict: dict[str, object],
) -> dict[str, object]:
    baseline_return_ser = _daily_return_ser(baseline_strategy_obj)
    standalone_return_ser = _daily_return_ser(standalone_strategy_obj)
    marginal_return_ser = _daily_return_ser(marginal_strategy_obj)
    baseline_active_bool_ser = _active_bool_ser(baseline_strategy_obj)
    standalone_active_bool_ser = _active_bool_ser(standalone_strategy_obj)
    aligned_active_df = pd.concat(
        [
            baseline_active_bool_ser.rename("baseline_active_bool"),
            standalone_active_bool_ser.rename("standalone_active_bool"),
        ],
        axis=1,
    ).fillna(False)

    standalone_active_day_count_int = int(aligned_active_df["standalone_active_bool"].sum())
    both_active_day_count_int = int(
        (
            aligned_active_df["baseline_active_bool"]
            & aligned_active_df["standalone_active_bool"]
        ).sum()
    )
    if standalone_active_day_count_int == 0:
        active_overlap_pct_float = np.nan
    else:
        active_overlap_pct_float = both_active_day_count_int / standalone_active_day_count_int * 100.0

    diagnostic_row_dict = {
        "manifest_rank_int": int(candidate_row_ser["manifest_rank_int"]),
        "candidate_symbol_str": str(candidate_row_ser["symbol_str"]),
        "source_universe_str": str(candidate_row_ser["source_universe_str"]),
        "bucket_str": str(candidate_row_ser["bucket_str"]),
        "description_str": str(candidate_row_ser["description_str"]),
        "standalone_corr_to_baseline_float": _correlation_float(standalone_return_ser, baseline_return_ser),
        "standalone_active_day_count_int": standalone_active_day_count_int,
        "baseline_active_day_count_int": int(aligned_active_df["baseline_active_bool"].sum()),
        "both_active_day_count_int": both_active_day_count_int,
        "active_overlap_vs_candidate_pct_float": active_overlap_pct_float,
        "standalone_sharpe_float": standalone_summary_dict.get("sharpe_float"),
        "standalone_oos_sharpe_float": standalone_summary_dict.get("oos_sharpe_float"),
        "marginal_ann_return_pct_float": marginal_summary_dict.get("ann_return_pct_float"),
        "marginal_sharpe_float": marginal_summary_dict.get("sharpe_float"),
        "marginal_max_drawdown_pct_float": marginal_summary_dict.get("max_drawdown_pct_float"),
        "marginal_oos_sharpe_float": marginal_summary_dict.get("oos_sharpe_float"),
        "marginal_oos_max_drawdown_pct_float": marginal_summary_dict.get("oos_max_drawdown_pct_float"),
        "marginal_trade_count_int": marginal_summary_dict.get("trade_count_int"),
        "marginal_cost_drag_ann_pct_float": marginal_summary_dict.get("cost_drag_ann_pct_float"),
        "delta_full_ann_return_pct_float": _safe_delta_float(
            marginal_summary_dict.get("ann_return_pct_float"),
            baseline_summary_dict.get("ann_return_pct_float"),
        ),
        "delta_full_sharpe_float": _safe_delta_float(
            marginal_summary_dict.get("sharpe_float"),
            baseline_summary_dict.get("sharpe_float"),
        ),
        "delta_full_max_drawdown_pct_float": _safe_delta_float(
            marginal_summary_dict.get("max_drawdown_pct_float"),
            baseline_summary_dict.get("max_drawdown_pct_float"),
        ),
        "delta_oos_sharpe_float": _safe_delta_float(
            marginal_summary_dict.get("oos_sharpe_float"),
            baseline_summary_dict.get("oos_sharpe_float"),
        ),
        "delta_oos_max_drawdown_pct_float": _safe_delta_float(
            marginal_summary_dict.get("oos_max_drawdown_pct_float"),
            baseline_summary_dict.get("oos_max_drawdown_pct_float"),
        ),
        "delta_trade_count_int": _safe_delta_float(
            marginal_summary_dict.get("trade_count_int"),
            baseline_summary_dict.get("trade_count_int"),
        ),
        "delta_cost_drag_ann_pct_float": _safe_delta_float(
            marginal_summary_dict.get("cost_drag_ann_pct_float"),
            baseline_summary_dict.get("cost_drag_ann_pct_float"),
        ),
    }
    accept_bool, reject_reason_str = evaluate_acceptance_rule(diagnostic_row_dict)
    diagnostic_row_dict["accept_bool"] = bool(accept_bool)
    diagnostic_row_dict["reject_reason_str"] = reject_reason_str
    diagnostic_row_dict.update(
        compute_tail_stress_metric_dict(
            baseline_return_ser=baseline_return_ser,
            standalone_return_ser=standalone_return_ser,
            marginal_return_ser=marginal_return_ser,
            benchmark_return_ser=benchmark_return_ser,
            standalone_active_bool_ser=standalone_active_bool_ser,
        )
    )
    stress_pass_bool, stress_reject_reason_str = evaluate_stress_rule(diagnostic_row_dict)
    diagnostic_row_dict["stress_pass_bool"] = bool(stress_pass_bool)
    diagnostic_row_dict["stress_reject_reason_str"] = stress_reject_reason_str
    return diagnostic_row_dict


def _markdown_value_str(value_obj: object) -> str:
    if value_obj is None:
        return ""
    if isinstance(value_obj, (float, np.floating)):
        if not np.isfinite(float(value_obj)):
            return ""
        return f"{float(value_obj):.3f}"
    if isinstance(value_obj, (int, np.integer)):
        return str(int(value_obj))
    if isinstance(value_obj, bool):
        return "yes" if value_obj else "no"
    return str(value_obj)


def _markdown_table_str(source_df: pd.DataFrame, column_list: list[str], max_rows_int: int) -> str:
    if len(source_df) == 0:
        return "_No rows._"
    table_df = source_df.loc[:, column_list].head(max_rows_int)
    lines_list = [
        "| " + " | ".join(column_list) + " |",
        "| " + " | ".join(["---"] * len(column_list)) + " |",
    ]
    for _, row_ser in table_df.iterrows():
        lines_list.append(
            "| " + " | ".join(_markdown_value_str(row_ser[column_str]) for column_str in column_list) + " |"
        )
    return "\n".join(lines_list)


def _write_accept_reject_md(
    output_path: Path,
    diagnostic_df: pd.DataFrame,
    baseline_summary_dict: dict[str, object],
) -> None:
    accepted_df = diagnostic_df.loc[diagnostic_df["accept_bool"]].copy()
    sorted_df = diagnostic_df.sort_values(
        by=[
            "accept_bool",
            "delta_oos_sharpe_float",
            "delta_full_sharpe_float",
            "standalone_corr_to_baseline_float",
        ],
        ascending=[False, False, False, True],
    )
    accepted_text_str = "No candidate passed the fixed acceptance rule."
    if len(accepted_df) > 0:
        accepted_text_str = ", ".join(accepted_df["candidate_symbol_str"].astype(str).tolist())

    table_column_list = [
        "candidate_symbol_str",
        "source_universe_str",
        "accept_bool",
        "delta_oos_sharpe_float",
        "delta_oos_max_drawdown_pct_float",
        "delta_full_sharpe_float",
        "standalone_corr_to_baseline_float",
        "delta_cost_drag_ann_pct_float",
        "reject_reason_str",
    ]
    accept_reject_md_str = f"""# Sector Dispersion Marginal Universe Study

## Fixed Design

- Research-only; no live/release wiring.
- Baseline basket: `{", ".join(ORIGINAL_SYMBOL_TUPLE)}`.
- Candidate manifest: Universe A, then Universe B additions, then Universe C additions, exactly as pre-listed.
- For each candidate, the script runs:
  - standalone candidate strategy;
  - baseline plus exactly one candidate;
  - diagnostics against the unchanged baseline.
- Execution convention stays `signal T -> Open T+1`; no MOC path is used.
- In-sample window ends `{IN_SAMPLE_END_TS.date().isoformat()}`.
- Out-of-sample diagnostic window starts `{OUT_OF_SAMPLE_START_TS.date().isoformat()}`.

## Acceptance Rule

An asset passes only if all fixed conditions hold:

- `delta_oos_sharpe_float >= {ACCEPTANCE_RULE_DICT["min_delta_oos_sharpe_float"]}`.
- `delta_oos_max_drawdown_pct_float >= {ACCEPTANCE_RULE_DICT["min_delta_oos_max_drawdown_pct_float"]}`.
- `standalone_corr_to_baseline_float <= {ACCEPTANCE_RULE_DICT["max_corr_to_baseline_float"]}`.
- `delta_full_sharpe_float >= {ACCEPTANCE_RULE_DICT["min_delta_full_sharpe_float"]}`.
- `delta_cost_drag_ann_pct_float <= {ACCEPTANCE_RULE_DICT["max_delta_cost_drag_ann_pct_float"]}`.

## Baseline

- Annual return [%]: `{_markdown_value_str(baseline_summary_dict.get("ann_return_pct_float"))}`
- Sharpe: `{_markdown_value_str(baseline_summary_dict.get("sharpe_float"))}`
- Max drawdown [%]: `{_markdown_value_str(baseline_summary_dict.get("max_drawdown_pct_float"))}`
- OOS Sharpe: `{_markdown_value_str(baseline_summary_dict.get("oos_sharpe_float"))}`
- OOS max drawdown [%]: `{_markdown_value_str(baseline_summary_dict.get("oos_max_drawdown_pct_float"))}`

## Accepted

{accepted_text_str}

## Ranked Diagnostics

{_markdown_table_str(sorted_df, table_column_list, max_rows_int=30)}
"""
    (output_path / "accept_reject.md").write_text(accept_reject_md_str, encoding="utf-8")


def _stress_column_list(diagnostic_df: pd.DataFrame) -> list[str]:
    preferred_column_list = [
        "candidate_symbol_str",
        "source_universe_str",
        "bucket_str",
        "stress_pass_bool",
        "accept_bool",
        "base_tail_delta_mean_return_pct_float",
        "base_tail_standalone_corr_to_baseline_float",
        "base_tail_candidate_active_pct_float",
        "base_tail_marginal_beats_base_pct_float",
        "market_tail_delta_mean_return_pct_float",
        "market_tail_standalone_corr_to_baseline_float",
        "market_tail_candidate_active_pct_float",
        "market_tail_marginal_beats_base_pct_float",
        "delta_oos_sharpe_float",
        "delta_oos_max_drawdown_pct_float",
        "delta_full_sharpe_float",
        "delta_cost_drag_ann_pct_float",
        "stress_reject_reason_str",
        "reject_reason_str",
    ]
    return [column_str for column_str in preferred_column_list if column_str in diagnostic_df.columns]


def _write_stress_recommendation_md(
    output_path: Path,
    diagnostic_df: pd.DataFrame,
) -> None:
    stress_sorted_df = diagnostic_df.sort_values(
        by=[
            "stress_pass_bool",
            "base_tail_delta_mean_return_pct_float",
            "market_tail_delta_mean_return_pct_float",
            "delta_oos_sharpe_float",
        ],
        ascending=[False, False, False, False],
    )
    stress_pass_df = diagnostic_df.loc[diagnostic_df["stress_pass_bool"]].copy()
    if len(stress_pass_df) == 0:
        recommendation_str = "No candidate passed the fixed stress rule."
    else:
        top_row_ser = stress_sorted_df.iloc[0]
        recommendation_str = (
            f"Use `{top_row_ser['candidate_symbol_str']}` first under the stress rule. "
            "It passed both the average-case acceptance rule and the fixed tail diagnostics."
        )

    table_column_list = _stress_column_list(diagnostic_df)
    stress_recommendation_md_str = f"""# Sector Dispersion Stress Recommendation

## Stress Design

- Research-only diagnostic layered on top of the marginal universe study.
- No signal, sizing, cost, or execution-timing semantics are changed.
- Candidate list stays frozen from the original Universe A/B/C manifest.
- Base stress days are the worst `{STRESS_RULE_DICT["base_tail_quantile_float"]:.0%}` realized daily returns of the baseline strategy.
- Market stress days are the worst `{STRESS_RULE_DICT["market_tail_quantile_float"]:.0%}` daily returns of `{DEFAULT_CONFIG.benchmark_symbol_str}`.
- A candidate must already pass the average-case rule before it can pass the stress rule.

## Stress Rule

An asset passes only if all fixed conditions hold:

- `base_tail_delta_mean_return_pct_float >= {STRESS_RULE_DICT["min_base_tail_delta_mean_return_pct_float"]}`.
- `market_tail_delta_mean_return_pct_float >= {STRESS_RULE_DICT["min_market_tail_delta_mean_return_pct_float"]}`.
- `base_tail_standalone_corr_to_baseline_float <= {STRESS_RULE_DICT["max_base_tail_corr_to_baseline_float"]}`.
- `market_tail_standalone_corr_to_baseline_float <= {STRESS_RULE_DICT["max_market_tail_corr_to_baseline_float"]}`.
- `base_tail_candidate_active_pct_float >= {STRESS_RULE_DICT["min_base_tail_candidate_active_pct_float"]}`.
- `market_tail_candidate_active_pct_float >= {STRESS_RULE_DICT["min_market_tail_candidate_active_pct_float"]}`.

## Recommendation

{recommendation_str}

## Ranked Stress Diagnostics

{_markdown_table_str(stress_sorted_df, table_column_list, max_rows_int=30)}
"""
    (output_path / "stress_recommendation.md").write_text(stress_recommendation_md_str, encoding="utf-8")


def _json_default_obj(value_obj: object) -> object:
    if isinstance(value_obj, Path):
        return str(value_obj)
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.isoformat()
    if isinstance(value_obj, np.integer):
        return int(value_obj)
    if isinstance(value_obj, np.floating):
        if not np.isfinite(float(value_obj)):
            return None
        return float(value_obj)
    return value_obj


def run_marginal_universe_study(
    output_dir_str: str = "results",
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
) -> Path:
    candidate_manifest_df = build_candidate_manifest_df()
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_sector_dispersion_ibs",
        analysis_type_str="marginal_universe_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)

    candidate_manifest_df.to_csv(output_path / "candidate_manifest.csv", index=False)

    candidate_symbol_tuple = tuple(candidate_manifest_df["symbol_str"].astype(str).tolist())
    all_symbol_tuple = tuple(dict.fromkeys(ORIGINAL_SYMBOL_TUPLE + candidate_symbol_tuple))
    base_config_obj = replace(
        DEFAULT_CONFIG,
        symbol_tuple=all_symbol_tuple,
        universe_name_str="original",
        end_date_str=end_date_str,
    )
    pricing_data_df = get_sector_dispersion_ibs_data(config_obj=base_config_obj)
    benchmark_close_ser = pd.to_numeric(
        pricing_data_df[(base_config_obj.benchmark_symbol_str, "Close")],
        errors="coerce",
    )
    benchmark_close_ser.index = pd.to_datetime(benchmark_close_ser.index).normalize()
    # *** CRITICAL*** Benchmark stress returns are post-run diagnostics only:
    # r_t = Close_t / Close_{t-1} - 1. They must not feed the strategy signal
    # or order generation path.
    benchmark_return_ser = benchmark_close_ser.pct_change(fill_method=None)

    baseline_strategy_obj = _run_strategy_variant(
        strategy_name_str="strategy_mr_sector_dispersion_ibs_marginal_base",
        symbol_tuple=ORIGINAL_SYMBOL_TUPLE,
        base_config_obj=base_config_obj,
        pricing_data_df=pricing_data_df,
        show_progress_bool=show_progress_bool,
    )
    baseline_summary_dict = _strategy_summary_row_dict(
        strategy_obj=baseline_strategy_obj,
        variant_kind_str="baseline",
        candidate_symbol_str=None,
        bucket_str=None,
    )

    standalone_summary_dict_list: list[dict[str, object]] = []
    marginal_summary_dict_list: list[dict[str, object]] = []
    diagnostic_dict_list: list[dict[str, object]] = []

    for _, candidate_row_ser in candidate_manifest_df.iterrows():
        candidate_symbol_str = str(candidate_row_ser["symbol_str"])
        candidate_slug_str = _slug_str(candidate_symbol_str)
        bucket_str = str(candidate_row_ser["bucket_str"])
        print(f"Running candidate {candidate_symbol_str}...", flush=True)

        standalone_strategy_obj = _run_strategy_variant(
            strategy_name_str=f"strategy_mr_sector_dispersion_ibs_standalone_{candidate_slug_str}",
            symbol_tuple=(candidate_symbol_str,),
            base_config_obj=base_config_obj,
            pricing_data_df=pricing_data_df,
            show_progress_bool=show_progress_bool,
        )
        marginal_strategy_obj = _run_strategy_variant(
            strategy_name_str=f"strategy_mr_sector_dispersion_ibs_add_{candidate_slug_str}",
            symbol_tuple=ORIGINAL_SYMBOL_TUPLE + (candidate_symbol_str,),
            base_config_obj=base_config_obj,
            pricing_data_df=pricing_data_df,
            show_progress_bool=show_progress_bool,
        )
        standalone_summary_dict = _strategy_summary_row_dict(
            strategy_obj=standalone_strategy_obj,
            variant_kind_str="standalone",
            candidate_symbol_str=candidate_symbol_str,
            bucket_str=bucket_str,
        )
        marginal_summary_dict = _strategy_summary_row_dict(
            strategy_obj=marginal_strategy_obj,
            variant_kind_str="marginal_add",
            candidate_symbol_str=candidate_symbol_str,
            bucket_str=bucket_str,
        )

        standalone_summary_dict_list.append(standalone_summary_dict)
        marginal_summary_dict_list.append(marginal_summary_dict)
        diagnostic_dict_list.append(
            _diagnostic_row_dict(
                candidate_row_ser=candidate_row_ser,
                baseline_strategy_obj=baseline_strategy_obj,
                standalone_strategy_obj=standalone_strategy_obj,
                marginal_strategy_obj=marginal_strategy_obj,
                benchmark_return_ser=benchmark_return_ser,
                baseline_summary_dict=baseline_summary_dict,
                standalone_summary_dict=standalone_summary_dict,
                marginal_summary_dict=marginal_summary_dict,
            )
        )

    baseline_summary_df = pd.DataFrame([baseline_summary_dict])
    standalone_summary_df = pd.DataFrame(standalone_summary_dict_list)
    marginal_summary_df = pd.DataFrame(marginal_summary_dict_list)
    diagnostic_df = pd.DataFrame(diagnostic_dict_list)

    baseline_summary_df.to_csv(output_path / "baseline_summary.csv", index=False)
    standalone_summary_df.to_csv(output_path / "standalone_summary.csv", index=False)
    marginal_summary_df.to_csv(output_path / "marginal_add_summary.csv", index=False)
    diagnostic_df.to_csv(output_path / "candidate_diagnostics.csv", index=False)
    diagnostic_df.loc[:, _stress_column_list(diagnostic_df)].to_csv(
        output_path / "stress_diagnostics.csv",
        index=False,
    )
    _write_accept_reject_md(
        output_path=output_path,
        diagnostic_df=diagnostic_df,
        baseline_summary_dict=baseline_summary_dict,
    )
    _write_stress_recommendation_md(
        output_path=output_path,
        diagnostic_df=diagnostic_df,
    )

    metadata_dict = {
        "strategy_id_str": "strategy_mr_sector_dispersion_ibs",
        "analysis_type_str": "marginal_universe_study",
        "research_only_bool": True,
        "output_path_str": str(output_path.resolve()),
        "candidate_count_int": int(len(candidate_manifest_df)),
        "all_symbol_tuple": all_symbol_tuple,
        "benchmark_symbol_str": base_config_obj.benchmark_symbol_str,
        "history_start_date_str": base_config_obj.history_start_date_str,
        "backtest_start_date_str": base_config_obj.backtest_start_date_str,
        "end_date_str": end_date_str,
        "in_sample_end_date_str": IN_SAMPLE_END_TS.date().isoformat(),
        "out_of_sample_start_date_str": OUT_OF_SAMPLE_START_TS.date().isoformat(),
        "acceptance_rule_dict": ACCEPTANCE_RULE_DICT,
        "stress_rule_dict": STRESS_RULE_DICT,
        "execution_mapping_str": "signal daily bar T -> Open T+1",
        "notes_str": (
            "Candidate order, acceptance rule, and stress rule are fixed before result sorting "
            "to reduce selection bias."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True, default=_json_default_obj),
        encoding="utf-8",
    )

    print(f"Saved marginal universe study to {output_path.resolve()}", flush=True)
    return output_path


def parse_args(argv_list: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sector-dispersion IBS marginal universe study.")
    parser.add_argument("--output-dir", default="results", help="Root output directory.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive Norgate end date.")
    parser.add_argument("--show-progress", action="store_true", help="Show Vanilla progress bars.")
    return parser.parse_args(argv_list)


def main(argv_list: list[str] | None = None) -> None:
    args = parse_args(argv_list)
    run_marginal_universe_study(
        output_dir_str=args.output_dir,
        end_date_str=args.end_date,
        show_progress_bool=bool(args.show_progress),
    )


if __name__ == "__main__":
    main()
