"""Strategy capacity analysis for completed MOO or MOC backtests.

The normal backtest remains the source of truth for signals, sizing, fills,
commissions, and the baseline 2.5 bps slippage assumption.  This module only
adds non-negative, size-dependent capacity drag and liquidity diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import html
import json

import numpy as np
import pandas as pd

from alpha.engine.report import build_research_output_path
from alpha.engine.strategy import Strategy


CAPACITY_ANALYSIS_TYPE_STR = "capacity_analysis"
CAPACITY_CURVE_CSV_FILENAME_STR = "capacity_curve.csv"
CAPACITY_ORDER_CSV_FILENAME_STR = "capacity_order_diagnostics.csv"
SUMMARY_FILENAME_STR = "summary.json"
METADATA_FILENAME_STR = "metadata.json"
REPORT_FILENAME_STR = "report.html"
CAPACITY_MODEL_VERSION_STR = "capacity_v2_1"

FULL_HISTORY_WINDOW_STR = "full_history"
RECENT_FIVE_YEAR_WINDOW_STR = "recent_5y"
HEADLINE_WINDOW_STR = RECENT_FIVE_YEAR_WINDOW_STR
RECENT_WINDOW_YEAR_INT = 5

MOO_EXECUTION_POLICY_STR = "MOO"
MOC_EXECUTION_POLICY_STR = "MOC"
SUPPORTED_EXECUTION_POLICY_SET = {MOO_EXECUTION_POLICY_STR, MOC_EXECUTION_POLICY_STR}

MOO_LARGE_MIXED_PROFILE_STR = "MOO_LARGE_MIXED"
MOO_NASDAQ_LARGE_PROFILE_STR = "MOO_NASDAQ_LARGE"
MOO_ETF_PROXY_PROFILE_STR = "MOO_ETF_PROXY"
MOO_IMPACT_PROFILE_DICT = {
    MOO_LARGE_MIXED_PROFILE_STR: {
        "central_lambda_1pct_adv_bps_float": 40.0,
        "stress_lambda_1pct_adv_bps_float": 66.4,
        "model_confidence_str": "Medium - common-stock estimate, pre-TCA",
        "proxy_bool": False,
    },
    MOO_NASDAQ_LARGE_PROFILE_STR: {
        "central_lambda_1pct_adv_bps_float": 66.4,
        "stress_lambda_1pct_adv_bps_float": 114.0,
        "model_confidence_str": "Medium - Nasdaq large-stock estimate, pre-TCA",
        "proxy_bool": False,
    },
    MOO_ETF_PROXY_PROFILE_STR: {
        "central_lambda_1pct_adv_bps_float": 40.0,
        "stress_lambda_1pct_adv_bps_float": 66.4,
        "model_confidence_str": "Low - common-stock-derived ETF proxy, pre-TCA",
        "proxy_bool": True,
    },
}

DEFAULT_AUM_GRID_TUPLE = (
    50_000.0,
    100_000.0,
    250_000.0,
    500_000.0,
    1_000_000.0,
    2_000_000.0,
    5_000_000.0,
    10_000_000.0,
    25_000_000.0,
    50_000_000.0,
    100_000_000.0,
)

BASELINE_SLIPPAGE_BPS_FLOAT = 2.5
ADV_MEAN_LOOKBACK_INT = 10
ADV_MEDIAN_LOOKBACK_INT = 20
MOC_CENTRAL_LAMBDA_1PCT_ADV_BPS_FLOAT = 8.2
MOC_STRESS_LAMBDA_1PCT_ADV_BPS_FLOAT = 17.8
IMPACT_EXPONENT_FLOAT = 0.5

MOO_SOFT_ORDER_ADV_LIMIT_FLOAT = 0.0005
MOO_HARD_ORDER_ADV_LIMIT_FLOAT = 0.0010
MOC_SOFT_ORDER_ADV_LIMIT_FLOAT = 0.0025
MOC_HARD_ORDER_ADV_LIMIT_FLOAT = 0.0050

RECOMMENDED_SHARPE_EROSION_LIMIT_FLOAT = 0.20
RECOMMENDED_COST_CONSUMPTION_LIMIT_FLOAT = 0.25
OUTER_HARD_BREACH_SHARE_LIMIT_FLOAT = 0.05
OUTER_COST_CONSUMPTION_LIMIT_FLOAT = 0.50
ROLLING_THREE_YEAR_TRADING_DAYS_INT = 756
ROLLING_BASELINE_SHARPE_FLOOR_FLOAT = 0.30


@dataclass
class CapacityRunResult:
    strategy_name_str: str
    capital_base_float: float
    execution_policy_str: str
    impact_profile_str: str | None
    order_diagnostics_df: pd.DataFrame
    equity_curve_df: pd.DataFrame
    summary_dict: dict[str, object]
    strategy_obj: Strategy
    pricing_data_df: pd.DataFrame


@dataclass
class CapacityStudyResult:
    strategy_name_str: str
    execution_policy_str: str
    impact_profile_str: str | None
    capacity_curve_df: pd.DataFrame
    order_diagnostics_df: pd.DataFrame
    summary_dict: dict[str, object]
    equity_curve_by_window_aum_dict: dict[tuple[str, float], pd.DataFrame]
    output_dir_path: Path | None = None


def normalize_execution_policy_str(execution_policy_str: str) -> str:
    normalized_policy_str = str(execution_policy_str).strip().upper()
    if normalized_policy_str not in SUPPORTED_EXECUTION_POLICY_SET:
        raise ValueError(
            "execution_policy_str must be explicitly declared as 'MOO' or 'MOC'; "
            f"received {execution_policy_str!r}."
        )
    return normalized_policy_str


def normalize_impact_profile_str(
    execution_policy_str: str,
    impact_profile_str: str | None,
) -> str | None:
    normalized_policy_str = normalize_execution_policy_str(execution_policy_str)
    if normalized_policy_str == MOC_EXECUTION_POLICY_STR:
        return None
    normalized_profile_str = str(impact_profile_str or "").strip().upper()
    if normalized_profile_str not in MOO_IMPACT_PROFILE_DICT:
        supported_profile_str = ", ".join(sorted(MOO_IMPACT_PROFILE_DICT))
        raise ValueError(
            "MOO CapacityAnalysis requires impact_profile_str. Supported profiles: "
            f"{supported_profile_str}; received {impact_profile_str!r}."
        )
    return normalized_profile_str


def impact_profile_assumption_dict(
    execution_policy_str: str,
    impact_profile_str: str | None,
) -> dict[str, object]:
    normalized_policy_str = normalize_execution_policy_str(execution_policy_str)
    normalized_profile_str = normalize_impact_profile_str(
        normalized_policy_str,
        impact_profile_str,
    )
    if normalized_policy_str == MOC_EXECUTION_POLICY_STR:
        return {
            "central_lambda_1pct_adv_bps_float": MOC_CENTRAL_LAMBDA_1PCT_ADV_BPS_FLOAT,
            "stress_lambda_1pct_adv_bps_float": MOC_STRESS_LAMBDA_1PCT_ADV_BPS_FLOAT,
            "model_confidence_str": "Medium - closing-auction estimate, pre-TCA",
            "proxy_bool": False,
        }
    return dict(MOO_IMPACT_PROFILE_DICT[normalized_profile_str])


def policy_limit_tuple(execution_policy_str: str) -> tuple[float, float]:
    normalized_policy_str = normalize_execution_policy_str(execution_policy_str)
    if normalized_policy_str == MOO_EXECUTION_POLICY_STR:
        return MOO_SOFT_ORDER_ADV_LIMIT_FLOAT, MOO_HARD_ORDER_ADV_LIMIT_FLOAT
    return MOC_SOFT_ORDER_ADV_LIMIT_FLOAT, MOC_HARD_ORDER_ADV_LIMIT_FLOAT


def square_root_impact_bps_float(
    order_adv_ratio_float: float,
    lambda_1pct_adv_bps_float: float,
) -> float:
    if not _is_finite_float(order_adv_ratio_float) or order_adv_ratio_float < 0.0:
        return np.nan
    if not _is_finite_float(lambda_1pct_adv_bps_float) or lambda_1pct_adv_bps_float < 0.0:
        return np.nan
    return float(
        lambda_1pct_adv_bps_float
        * (order_adv_ratio_float / 0.01) ** IMPACT_EXPONENT_FLOAT
    )


def capacity_implicit_cost_bps_float(
    order_adv_ratio_float: float,
    execution_policy_str: str,
    stress_bool: bool = False,
    impact_profile_str: str | None = None,
) -> float:
    normalized_policy_str = normalize_execution_policy_str(execution_policy_str)
    assumption_dict = impact_profile_assumption_dict(
        normalized_policy_str,
        impact_profile_str,
    )
    lambda_bps_float = float(
        assumption_dict[
            "stress_lambda_1pct_adv_bps_float"
            if stress_bool
            else "central_lambda_1pct_adv_bps_float"
        ]
    )
    impact_bps_float = square_root_impact_bps_float(
        order_adv_ratio_float,
        lambda_bps_float,
    )
    if not _is_finite_float(impact_bps_float):
        return np.nan
    return float(max(BASELINE_SLIPPAGE_BPS_FLOAT, impact_bps_float))


class CapacityAnalysis:
    """Analyze one fully rerun strategy at one capital level."""

    def __init__(
        self,
        strategy_obj: Strategy,
        pricing_data_df: pd.DataFrame,
        execution_policy_str: str,
        impact_profile_str: str | None = None,
    ):
        self.strategy_obj = strategy_obj
        self.pricing_data_df = pricing_data_df.sort_index().copy()
        self.execution_policy_str = normalize_execution_policy_str(execution_policy_str)
        self.impact_profile_str = normalize_impact_profile_str(
            self.execution_policy_str,
            impact_profile_str,
        )

    def run(self) -> CapacityRunResult:
        transaction_df = _completed_transaction_df(self.strategy_obj)
        order_diagnostics_df = _build_order_diagnostics_df(
            transaction_df=transaction_df,
            pricing_data_df=self.pricing_data_df,
            execution_policy_str=self.execution_policy_str,
            impact_profile_str=self.impact_profile_str,
        )
        summary_dict, equity_curve_df = _build_run_summary_tuple(
            strategy_obj=self.strategy_obj,
            pricing_data_df=self.pricing_data_df,
            order_diagnostics_df=order_diagnostics_df,
            execution_policy_str=self.execution_policy_str,
            impact_profile_str=self.impact_profile_str,
        )
        return CapacityRunResult(
            strategy_name_str=str(self.strategy_obj.name),
            capital_base_float=float(self.strategy_obj._capital_base),
            execution_policy_str=self.execution_policy_str,
            impact_profile_str=self.impact_profile_str,
            order_diagnostics_df=order_diagnostics_df,
            equity_curve_df=equity_curve_df,
            summary_dict=summary_dict,
            strategy_obj=self.strategy_obj,
            pricing_data_df=self.pricing_data_df,
        )


def build_capacity_study_result(
    capacity_run_result_by_window_dict: dict[str, list[CapacityRunResult]],
    output_dir_str: str = "results",
    save_output_bool: bool = True,
) -> CapacityStudyResult:
    if not capacity_run_result_by_window_dict:
        raise ValueError("capacity_run_result_by_window_dict must not be empty.")
    unsupported_window_list = sorted(
        set(capacity_run_result_by_window_dict).difference(
            {FULL_HISTORY_WINDOW_STR, RECENT_FIVE_YEAR_WINDOW_STR}
        )
    )
    if unsupported_window_list:
        raise ValueError(f"Unsupported Capacity window labels: {unsupported_window_list}.")
    if any(
        not capacity_run_result_list
        for capacity_run_result_list in capacity_run_result_by_window_dict.values()
    ):
        raise ValueError("Every Capacity window must contain at least one AUM run.")

    all_run_result_list = [
        result_obj
        for capacity_run_result_list in capacity_run_result_by_window_dict.values()
        for result_obj in capacity_run_result_list
    ]
    strategy_name_set = {result_obj.strategy_name_str for result_obj in all_run_result_list}
    policy_set = {result_obj.execution_policy_str for result_obj in all_run_result_list}
    profile_set = {result_obj.impact_profile_str for result_obj in all_run_result_list}
    if len(strategy_name_set) != 1:
        raise ValueError("All AUM runs must belong to the same strategy.")
    if len(policy_set) != 1:
        raise ValueError("All AUM runs must declare the same MOO or MOC policy.")
    if len(profile_set) != 1:
        raise ValueError("All AUM runs must declare the same impact profile.")

    first_result_obj = all_run_result_list[0]
    strategy_name_str = first_result_obj.strategy_name_str
    execution_policy_str = first_result_obj.execution_policy_str
    impact_profile_str = first_result_obj.impact_profile_str

    window_curve_df_list: list[pd.DataFrame] = []
    window_order_df_list: list[pd.DataFrame] = []
    window_summary_dict: dict[str, dict[str, object]] = {}
    equity_curve_by_window_aum_dict: dict[tuple[str, float], pd.DataFrame] = {}
    for window_str, capacity_run_result_list in capacity_run_result_by_window_dict.items():
        sorted_result_list = sorted(
            capacity_run_result_list,
            key=lambda result_obj: result_obj.capital_base_float,
        )
        capacity_window_curve_df = pd.DataFrame(
            [dict(result_obj.summary_dict) for result_obj in sorted_result_list]
        ).sort_values("capital_base_float")
        capacity_window_curve_df.insert(0, "window_str", window_str)
        capacity_window_order_df = pd.concat(
            [
                result_obj.order_diagnostics_df.assign(
                    window_str=window_str,
                    capital_base_float=result_obj.capital_base_float,
                )
                for result_obj in sorted_result_list
            ],
            ignore_index=True,
        )
        window_summary_dict[window_str] = _build_study_summary_dict(
            strategy_name_str=strategy_name_str,
            execution_policy_str=execution_policy_str,
            impact_profile_str=impact_profile_str,
            capacity_curve_df=capacity_window_curve_df,
            order_diagnostics_df=capacity_window_order_df,
        )
        window_curve_df_list.append(capacity_window_curve_df)
        window_order_df_list.append(capacity_window_order_df)
        equity_curve_by_window_aum_dict.update(
            {
                (window_str, result_obj.capital_base_float): result_obj.equity_curve_df
                for result_obj in sorted_result_list
            }
        )

    headline_window_str = (
        HEADLINE_WINDOW_STR
        if HEADLINE_WINDOW_STR in window_summary_dict
        else next(iter(window_summary_dict))
    )
    actual_end_date_set = {
        window_summary_obj["actual_end_date_str"]
        for window_summary_obj in window_summary_dict.values()
    }
    if len(actual_end_date_set) != 1:
        raise ValueError(
            "Full-history and recent Capacity windows must share one actual end date: "
            f"{sorted(actual_end_date_set)}."
        )
    summary_dict = dict(window_summary_dict[headline_window_str])
    summary_dict["headline_window_str"] = headline_window_str
    summary_dict["window_summary_dict"] = window_summary_dict
    full_history_recommended_obj = window_summary_dict.get(
        FULL_HISTORY_WINDOW_STR,
        {},
    ).get("recommended_capacity_float")
    recent_recommended_obj = window_summary_dict.get(
        RECENT_FIVE_YEAR_WINDOW_STR,
        {},
    ).get("recommended_capacity_float")
    summary_dict["historical_feasibility_warning_bool"] = bool(
        recent_recommended_obj is not None
        and (
            full_history_recommended_obj is None
            or float(full_history_recommended_obj) < float(recent_recommended_obj)
        )
    )
    summary_dict["window_date_dict"] = {
        window_str: {
            "actual_start_date_str": window_summary_obj["actual_start_date_str"],
            "actual_end_date_str": window_summary_obj["actual_end_date_str"],
        }
        for window_str, window_summary_obj in window_summary_dict.items()
    }
    capacity_curve_df = pd.concat(window_curve_df_list, ignore_index=True)
    order_diagnostics_df = pd.concat(window_order_df_list, ignore_index=True)
    study_result_obj = CapacityStudyResult(
        strategy_name_str=strategy_name_str,
        execution_policy_str=execution_policy_str,
        impact_profile_str=impact_profile_str,
        capacity_curve_df=capacity_curve_df,
        order_diagnostics_df=order_diagnostics_df,
        summary_dict=summary_dict,
        equity_curve_by_window_aum_dict=equity_curve_by_window_aum_dict,
    )
    if save_output_bool:
        save_capacity_study_results(study_result_obj, output_dir_str)
    return study_result_obj


def save_capacity_study_results(
    study_result_obj: CapacityStudyResult,
    output_dir_str: str = "results",
) -> Path:
    output_dir_path = build_research_output_path(
        output_dir_str,
        "strategy",
        study_result_obj.strategy_name_str,
        CAPACITY_ANALYSIS_TYPE_STR,
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    study_result_obj.capacity_curve_df.to_csv(
        output_dir_path / CAPACITY_CURVE_CSV_FILENAME_STR,
        index=False,
    )
    study_result_obj.order_diagnostics_df.to_csv(
        output_dir_path / CAPACITY_ORDER_CSV_FILENAME_STR,
        index=False,
        date_format="%Y-%m-%d",
    )
    _write_json_file(output_dir_path / SUMMARY_FILENAME_STR, study_result_obj.summary_dict)
    _write_json_file(
        output_dir_path / METADATA_FILENAME_STR,
        {
            "analysis_type_str": CAPACITY_ANALYSIS_TYPE_STR,
            "strategy_name_str": study_result_obj.strategy_name_str,
            "execution_policy_str": study_result_obj.execution_policy_str,
            "impact_profile_str": study_result_obj.impact_profile_str,
            "saved_at": datetime.now().astimezone().isoformat(),
            "model_version_str": CAPACITY_MODEL_VERSION_STR,
            "headline_window_str": study_result_obj.summary_dict["headline_window_str"],
            "recent_window_year_count_int": RECENT_WINDOW_YEAR_INT,
            "window_date_dict": study_result_obj.summary_dict["window_date_dict"],
            "full_history_actual_start_date_str": study_result_obj.summary_dict[
                "window_date_dict"
            ].get(FULL_HISTORY_WINDOW_STR, {}).get("actual_start_date_str"),
            "recent_5y_actual_start_date_str": study_result_obj.summary_dict[
                "window_date_dict"
            ].get(RECENT_FIVE_YEAR_WINDOW_STR, {}).get("actual_start_date_str"),
            "common_actual_end_date_str": study_result_obj.summary_dict[
                "actual_end_date_str"
            ],
        },
    )
    (output_dir_path / REPORT_FILENAME_STR).write_text(
        _build_report_html_str(study_result_obj),
        encoding="utf-8",
    )
    study_result_obj.output_dir_path = output_dir_path
    return output_dir_path


def _completed_transaction_df(strategy_obj: Strategy) -> pd.DataFrame:
    if not hasattr(strategy_obj, "get_transactions"):
        raise TypeError("strategy_obj must expose get_transactions().")
    transaction_df = pd.DataFrame(strategy_obj.get_transactions()).copy()
    required_column_set = {"bar", "asset", "amount", "price", "total_value", "commission"}
    missing_column_list = sorted(required_column_set.difference(transaction_df.columns))
    if len(transaction_df) > 0 and missing_column_list:
        raise ValueError(f"transactions are missing required columns: {missing_column_list}.")
    return transaction_df


def _build_order_diagnostics_df(
    transaction_df: pd.DataFrame,
    pricing_data_df: pd.DataFrame,
    execution_policy_str: str,
    impact_profile_str: str | None,
) -> pd.DataFrame:
    if len(transaction_df) == 0:
        return _empty_order_diagnostics_df()
    if not isinstance(pricing_data_df.columns, pd.MultiIndex):
        raise ValueError("pricing_data_df must have MultiIndex columns.")

    transaction_df = transaction_df.copy()
    transaction_df["bar"] = pd.to_datetime(transaction_df["bar"]).dt.normalize()
    transaction_df["asset"] = transaction_df["asset"].astype(str)
    transaction_df["amount_float"] = pd.to_numeric(transaction_df["amount"], errors="coerce")
    transaction_df["side_str"] = np.where(
        transaction_df["amount_float"] > 0.0,
        "Buy",
        np.where(transaction_df["amount_float"] < 0.0, "Sell", "Flat"),
    )
    transaction_df["order_notional_float"] = transaction_df.apply(
        lambda order_ser: _order_notional_float(order_ser),
        axis=1,
    )
    transaction_df["commission_float"] = pd.to_numeric(
        transaction_df["commission"],
        errors="coerce",
    ).fillna(0.0)

    # The market sees the combined same-side order, not internal strategy rows.
    aggregated_order_df = (
        transaction_df.groupby(["bar", "asset", "side_str"], as_index=False)
        .agg(
            amount_float=("amount_float", "sum"),
            order_notional_float=("order_notional_float", "sum"),
            commission_float=("commission_float", "sum"),
            source_transaction_count_int=("asset", "size"),
        )
        .sort_values(["bar", "asset", "side_str"])
    )
    traded_asset_list = sorted(aggregated_order_df["asset"].unique().tolist())
    adv_map_dict, volume_source_map = _build_lagged_adv_map_dict(
        pricing_data_df,
        traded_asset_list,
    )
    soft_limit_float, hard_limit_float = policy_limit_tuple(execution_policy_str)
    profile_assumption_dict = impact_profile_assumption_dict(
        execution_policy_str,
        impact_profile_str,
    )
    central_lambda_bps_float = float(
        profile_assumption_dict["central_lambda_1pct_adv_bps_float"]
    )
    stress_lambda_bps_float = float(
        profile_assumption_dict["stress_lambda_1pct_adv_bps_float"]
    )
    model_confidence_str = str(profile_assumption_dict["model_confidence_str"])
    proxy_bool = bool(profile_assumption_dict["proxy_bool"])

    row_dict_list: list[dict[str, object]] = []
    for _, order_ser in aggregated_order_df.iterrows():
        bar_ts = pd.Timestamp(order_ser["bar"])
        asset_str = str(order_ser["asset"])
        order_notional_float = _coerce_float(order_ser["order_notional_float"])
        adv10_float = _lookup_bar_value_float(adv_map_dict[asset_str]["adv10"], bar_ts)
        adv20_float = _lookup_bar_value_float(adv_map_dict[asset_str]["adv20"], bar_ts)
        unavailable_reason_str = ""
        if not _is_finite_float(order_notional_float):
            unavailable_reason_str = "missing_order_notional"
        elif not _is_finite_float(adv10_float) or not _is_finite_float(adv20_float):
            unavailable_reason_str = "insufficient_lagged_adv_history"
        elif adv10_float <= 0.0 or adv20_float <= 0.0:
            unavailable_reason_str = "non_positive_lagged_adv"

        if unavailable_reason_str:
            robust_adv_float = np.nan
            order_adv_ratio_float = np.nan
            central_impact_bps_float = np.nan
            central_cost_bps_float = np.nan
            stress_impact_bps_float = np.nan
            stress_cost_bps_float = np.nan
            central_extra_cost_float = np.nan
            stress_extra_cost_float = np.nan
            soft_breach_bool = False
            hard_breach_bool = False
            assessed_bool = False
        else:
            robust_adv_float = min(adv10_float, adv20_float)
            order_adv_ratio_float = order_notional_float / robust_adv_float
            central_impact_bps_float = square_root_impact_bps_float(
                order_adv_ratio_float,
                central_lambda_bps_float,
            )
            stress_impact_bps_float = square_root_impact_bps_float(
                order_adv_ratio_float,
                stress_lambda_bps_float,
            )
            central_cost_bps_float = capacity_implicit_cost_bps_float(
                order_adv_ratio_float,
                execution_policy_str,
                stress_bool=False,
                impact_profile_str=impact_profile_str,
            )
            stress_cost_bps_float = capacity_implicit_cost_bps_float(
                order_adv_ratio_float,
                execution_policy_str,
                stress_bool=True,
                impact_profile_str=impact_profile_str,
            )
            central_extra_bps_float = max(
                0.0,
                central_cost_bps_float - BASELINE_SLIPPAGE_BPS_FLOAT,
            )
            central_extra_cost_float = (
                order_notional_float * central_extra_bps_float / 10_000.0
            )
            stress_extra_cost_float = (
                order_notional_float
                * max(0.0, stress_cost_bps_float - BASELINE_SLIPPAGE_BPS_FLOAT)
                / 10_000.0
                if _is_finite_float(stress_cost_bps_float)
                else np.nan
            )
            soft_breach_bool = bool(order_adv_ratio_float > soft_limit_float)
            hard_breach_bool = bool(order_adv_ratio_float > hard_limit_float)
            assessed_bool = True

        model_extrapolation_bool = bool(
            assessed_bool
            and execution_policy_str == MOO_EXECUTION_POLICY_STR
            and order_adv_ratio_float > 0.01
        )
        academic_extrapolation_bool = bool(model_extrapolation_bool and not proxy_bool)
        proxy_extrapolation_bool = bool(model_extrapolation_bool and proxy_bool)

        row_dict_list.append(
            {
                "bar": bar_ts,
                "asset_str": asset_str,
                "side_str": str(order_ser["side_str"]),
                "execution_policy_str": execution_policy_str,
                "impact_profile_str": impact_profile_str,
                "central_lambda_1pct_adv_bps_float": central_lambda_bps_float,
                "stress_lambda_1pct_adv_bps_float": stress_lambda_bps_float,
                "model_confidence_str": model_confidence_str,
                "proxy_bool": proxy_bool,
                "model_extrapolation_bool": model_extrapolation_bool,
                "academic_extrapolation_bool": academic_extrapolation_bool,
                "proxy_extrapolation_bool": proxy_extrapolation_bool,
                "amount_float": _coerce_float(order_ser["amount_float"]),
                "order_notional_float": order_notional_float,
                "commission_float": _coerce_float(order_ser["commission_float"]),
                "source_transaction_count_int": int(order_ser["source_transaction_count_int"]),
                "dollar_volume_source_str": volume_source_map[asset_str],
                "adv10_mean_dollar_lagged_float": adv10_float,
                "adv20_median_dollar_lagged_float": adv20_float,
                "robust_adv_dollar_lagged_float": robust_adv_float,
                "order_adv_ratio_float": order_adv_ratio_float,
                "order_adv_pct_float": order_adv_ratio_float * 100.0
                if _is_finite_float(order_adv_ratio_float)
                else np.nan,
                "baseline_slippage_bps_float": BASELINE_SLIPPAGE_BPS_FLOAT,
                "central_impact_bps_float": central_impact_bps_float,
                "central_implicit_cost_bps_float": central_cost_bps_float,
                "central_incremental_cost_float": central_extra_cost_float,
                "stress_impact_bps_float": stress_impact_bps_float,
                "stress_implicit_cost_bps_float": stress_cost_bps_float,
                "stress_incremental_cost_float": stress_extra_cost_float,
                "soft_limit_float": soft_limit_float,
                "hard_limit_float": hard_limit_float,
                "soft_breach_bool": soft_breach_bool,
                "hard_breach_bool": hard_breach_bool,
                "assessed_bool": assessed_bool,
                "unavailable_reason_str": unavailable_reason_str,
            }
        )
    return pd.DataFrame(row_dict_list)


def _build_lagged_adv_map_dict(
    pricing_data_df: pd.DataFrame,
    traded_asset_list: list[str],
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, str]]:
    adv_map_dict: dict[str, dict[str, pd.Series]] = {}
    volume_source_map: dict[str, str] = {}
    for asset_str in traded_asset_list:
        if (asset_str, "Turnover") in pricing_data_df.columns:
            dollar_volume_ser = pricing_data_df[(asset_str, "Turnover")].astype(float)
            volume_source_map[asset_str] = "Norgate Turnover"
        elif (
            (asset_str, "Close") in pricing_data_df.columns
            and (asset_str, "Volume") in pricing_data_df.columns
        ):
            close_ser = pricing_data_df[(asset_str, "Close")].astype(float)
            volume_ser = pricing_data_df[(asset_str, "Volume")].astype(float)
            dollar_volume_ser = close_ser * volume_ser
            volume_source_map[asset_str] = "Close x Volume"
        else:
            raise ValueError(
                f"Missing Turnover or Close/Volume liquidity data for {asset_str}."
            )

        # *** CRITICAL *** lookahead-sensitive: an auction order at T may use
        # only completed dollar volume through T-1. Never include same-day volume.
        lagged_dollar_volume_ser = dollar_volume_ser.shift(1)
        adv10_ser = lagged_dollar_volume_ser.rolling(
            ADV_MEAN_LOOKBACK_INT,
            min_periods=ADV_MEAN_LOOKBACK_INT,
        ).mean()
        adv20_ser = lagged_dollar_volume_ser.rolling(
            ADV_MEDIAN_LOOKBACK_INT,
            min_periods=ADV_MEDIAN_LOOKBACK_INT,
        ).median()
        adv_map_dict[asset_str] = {"adv10": adv10_ser, "adv20": adv20_ser}
    return adv_map_dict, volume_source_map


def _build_run_summary_tuple(
    strategy_obj: Strategy,
    pricing_data_df: pd.DataFrame,
    order_diagnostics_df: pd.DataFrame,
    execution_policy_str: str,
    impact_profile_str: str | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    assessed_order_df = order_diagnostics_df[
        order_diagnostics_df.get("assessed_bool", pd.Series(dtype=bool)) == True
    ]
    baseline_equity_ser = _strategy_equity_ser(strategy_obj)
    central_equity_ser = _adjusted_equity_ser(
        baseline_equity_ser,
        assessed_order_df,
        "central_incremental_cost_float",
    )
    stress_equity_ser = _adjusted_equity_ser(
        baseline_equity_ser,
        assessed_order_df,
        "stress_incremental_cost_float",
    )
    # Floating-point reconstruction of the baseline growth path can differ by
    # a few ulps. Enforce the economic ordering promised by this overlay.
    central_equity_ser = central_equity_ser.combine(baseline_equity_ser, min)
    stress_equity_ser = stress_equity_ser.combine(central_equity_ser, min)
    benchmark_annual_return_float, benchmark_symbol_str = _benchmark_annual_return_tuple(
        strategy_obj,
        pricing_data_df,
        baseline_equity_ser.index,
    )
    baseline_annual_return_float = _annualized_return_float(baseline_equity_ser)
    central_annual_return_float = _annualized_return_float(central_equity_ser)
    stress_annual_return_float = _annualized_return_float(stress_equity_ser)
    baseline_benchmark_excess_return_float = _benchmark_excess_return_float(
        baseline_annual_return_float,
        benchmark_annual_return_float,
    )
    central_benchmark_excess_return_float = _benchmark_excess_return_float(
        central_annual_return_float,
        benchmark_annual_return_float,
    )
    stress_benchmark_excess_return_float = _benchmark_excess_return_float(
        stress_annual_return_float,
        benchmark_annual_return_float,
    )
    baseline_sharpe_float = _sharpe_float(baseline_equity_ser)
    central_sharpe_float = _sharpe_float(central_equity_ser)
    stress_sharpe_float = _sharpe_float(stress_equity_ser)
    soft_limit_float, hard_limit_float = policy_limit_tuple(execution_policy_str)
    assessed_count_int = int(len(assessed_order_df))
    total_count_int = int(len(order_diagnostics_df))
    liquidity_complete_bool = total_count_int > 0 and assessed_count_int == total_count_int
    hard_breach_share_float = _safe_divide_float(
        int(assessed_order_df.get("hard_breach_bool", pd.Series(dtype=bool)).sum()),
        assessed_count_int,
    )
    soft_breach_share_float = _safe_divide_float(
        int(assessed_order_df.get("soft_breach_bool", pd.Series(dtype=bool)).sum()),
        assessed_count_int,
    )
    rolling_erosion_float, rolling_eligible_window_count_int = (
        _worst_eligible_rolling_sharpe_erosion_tuple(
        baseline_equity_ser,
        central_equity_ser,
        )
    )
    profile_assumption_dict = impact_profile_assumption_dict(
        execution_policy_str,
        impact_profile_str,
    )
    summary_dict = {
        "strategy_name_str": str(strategy_obj.name),
        "analysis_type_str": CAPACITY_ANALYSIS_TYPE_STR,
        "capital_base_float": float(strategy_obj._capital_base),
        "execution_policy_str": execution_policy_str,
        "impact_profile_str": impact_profile_str,
        "central_lambda_1pct_adv_bps_float": profile_assumption_dict[
            "central_lambda_1pct_adv_bps_float"
        ],
        "stress_lambda_1pct_adv_bps_float": profile_assumption_dict[
            "stress_lambda_1pct_adv_bps_float"
        ],
        "model_confidence_str": profile_assumption_dict["model_confidence_str"],
        "proxy_bool": profile_assumption_dict["proxy_bool"],
        "benchmark_symbol_str": benchmark_symbol_str,
        "benchmark_annual_return_float": benchmark_annual_return_float,
        "actual_start_date_str": (
            pd.Timestamp(baseline_equity_ser.index.min()).date().isoformat()
            if len(baseline_equity_ser)
            else None
        ),
        "actual_end_date_str": (
            pd.Timestamp(baseline_equity_ser.index.max()).date().isoformat()
            if len(baseline_equity_ser)
            else None
        ),
        "total_order_count_int": total_count_int,
        "assessed_order_count_int": assessed_count_int,
        "unavailable_order_count_int": total_count_int - assessed_count_int,
        "unavailable_order_share_float": _safe_divide_float(
            total_count_int - assessed_count_int,
            total_count_int,
        ),
        "liquidity_complete_bool": liquidity_complete_bool,
        "order_adv_p50_float": _quantile_float(assessed_order_df, "order_adv_ratio_float", 0.50),
        "order_adv_p95_float": _quantile_float(assessed_order_df, "order_adv_ratio_float", 0.95),
        "order_adv_p99_float": _quantile_float(assessed_order_df, "order_adv_ratio_float", 0.99),
        "order_adv_max_float": _max_float(assessed_order_df, "order_adv_ratio_float"),
        "soft_limit_float": soft_limit_float,
        "hard_limit_float": hard_limit_float,
        "soft_breach_share_float": soft_breach_share_float,
        "hard_breach_share_float": hard_breach_share_float,
        "baseline_annual_return_float": baseline_annual_return_float,
        "central_annual_return_float": central_annual_return_float,
        "stress_annual_return_float": stress_annual_return_float,
        "baseline_benchmark_excess_return_float": baseline_benchmark_excess_return_float,
        "central_benchmark_excess_return_float": central_benchmark_excess_return_float,
        "stress_benchmark_excess_return_float": stress_benchmark_excess_return_float,
        "baseline_sharpe_float": baseline_sharpe_float,
        "central_sharpe_float": central_sharpe_float,
        "stress_sharpe_float": stress_sharpe_float,
        "sharpe_erosion_float": _erosion_float(
            baseline_sharpe_float,
            central_sharpe_float,
        ),
        "stress_sharpe_erosion_float": _erosion_float(
            baseline_sharpe_float,
            stress_sharpe_float,
        ),
        "central_cost_consumption_of_benchmark_excess_float": _cost_consumption_float(
            baseline_benchmark_excess_return_float,
            central_benchmark_excess_return_float,
        ),
        "stress_cost_consumption_of_benchmark_excess_float": _cost_consumption_float(
            baseline_benchmark_excess_return_float,
            stress_benchmark_excess_return_float,
        ),
        "worst_eligible_rolling_3y_sharpe_erosion_float": rolling_erosion_float,
        "rolling_3y_eligible_window_count_int": rolling_eligible_window_count_int,
        "rolling_3y_available_bool": (
            rolling_eligible_window_count_int > 0 and _is_finite_float(rolling_erosion_float)
        ),
        "central_incremental_cost_float": float(
            assessed_order_df.get("central_incremental_cost_float", pd.Series(dtype=float))
            .fillna(0.0)
            .sum()
        ),
        "stress_incremental_cost_float": float(
            assessed_order_df.get("stress_incremental_cost_float", pd.Series(dtype=float))
            .fillna(0.0)
            .sum()
        ),
        "academic_extrapolation_share_float": _safe_divide_float(
            int(
                assessed_order_df.get(
                    "academic_extrapolation_bool",
                    pd.Series(dtype=bool),
                ).sum()
            ),
            assessed_count_int,
        ),
        "proxy_extrapolation_share_float": _safe_divide_float(
            int(
                assessed_order_df.get(
                    "proxy_extrapolation_bool",
                    pd.Series(dtype=bool),
                ).sum()
            ),
            assessed_count_int,
        ),
        "model_extrapolation_share_float": _safe_divide_float(
            int(
                assessed_order_df.get(
                    "model_extrapolation_bool",
                    pd.Series(dtype=bool),
                ).sum()
            ),
            assessed_count_int,
        ),
    }
    equity_curve_df = pd.concat(
        [
            baseline_equity_ser.rename("baseline_equity_float"),
            central_equity_ser.rename("central_equity_float"),
            stress_equity_ser.rename("stress_equity_float"),
        ],
        axis=1,
    )
    return summary_dict, equity_curve_df


def _build_study_summary_dict(
    strategy_name_str: str,
    execution_policy_str: str,
    impact_profile_str: str | None,
    capacity_curve_df: pd.DataFrame,
    order_diagnostics_df: pd.DataFrame,
) -> dict[str, object]:
    curve_df = capacity_curve_df.copy()
    benchmark_available_ser = curve_df["benchmark_annual_return_float"].map(_is_finite_float)
    curve_df["recommended_raw_pass_bool"] = (
        benchmark_available_ser
        & (curve_df["liquidity_complete_bool"] == True)
        & (curve_df["order_adv_p95_float"] <= curve_df["soft_limit_float"])
        & (curve_df["order_adv_p99_float"] <= curve_df["hard_limit_float"])
        & (curve_df["sharpe_erosion_float"] <= RECOMMENDED_SHARPE_EROSION_LIMIT_FLOAT)
        & (
            curve_df["central_cost_consumption_of_benchmark_excess_float"]
            <= RECOMMENDED_COST_CONSUMPTION_LIMIT_FLOAT
        )
        & (curve_df["rolling_3y_available_bool"] == True)
        & (
            curve_df["worst_eligible_rolling_3y_sharpe_erosion_float"]
            <= RECOMMENDED_SHARPE_EROSION_LIMIT_FLOAT
        )
    )
    curve_df["outer_raw_pass_bool"] = (
        (curve_df["liquidity_complete_bool"] == True)
        & (curve_df["stress_benchmark_excess_return_float"] > 0.0)
        & (curve_df["hard_breach_share_float"] < OUTER_HARD_BREACH_SHARE_LIMIT_FLOAT)
        & (
            curve_df["stress_cost_consumption_of_benchmark_excess_float"]
            < OUTER_COST_CONSUMPTION_LIMIT_FLOAT
        )
    )
    outer_method_str = (
        "MOC academic stress plus hard-liquidity limits"
        if execution_policy_str == MOC_EXECUTION_POLICY_STR
        else "MOO profile stress plus hard-liquidity limits"
    )
    break_even_bracket_str = _break_even_bracket_str(curve_df)

    (
        recommended_pass_ser,
        recommended_capacity_float,
        recommended_non_contiguous_bool,
        recommended_capacity_censored_bool,
    ) = _contiguous_capacity_tuple(curve_df, "recommended_raw_pass_bool")
    (
        outer_pass_ser,
        outer_capacity_float,
        outer_non_contiguous_bool,
        outer_capacity_censored_bool,
    ) = _contiguous_capacity_tuple(curve_df, "outer_raw_pass_bool")
    curve_df["recommended_pass_bool"] = recommended_pass_ser
    curve_df["outer_pass_bool"] = outer_pass_ser

    # Reflect classifications in the saved curve as well as the summary.
    for column_str in [
        "recommended_raw_pass_bool",
        "recommended_pass_bool",
        "outer_raw_pass_bool",
        "outer_pass_bool",
    ]:
        capacity_curve_df[column_str] = curve_df[column_str].astype(bool).values
    optimal_capacity_float = _optimal_capacity_float(curve_df)
    rolling_reference_aum_float = (
        float(recommended_capacity_float)
        if recommended_capacity_float is not None
        else float(curve_df["capital_base_float"].min())
    )
    rolling_reference_ser = curve_df[
        curve_df["capital_base_float"].astype(float) == rolling_reference_aum_float
    ].iloc[0]
    rolling_reference_basis_str = (
        "recommended_capacity"
        if recommended_capacity_float is not None
        else "lowest_diagnostic_aum"
    )
    assessed_order_df = order_diagnostics_df[
        order_diagnostics_df.get("assessed_bool", pd.Series(dtype=bool)) == True
    ]
    return {
        "analysis_type_str": CAPACITY_ANALYSIS_TYPE_STR,
        "strategy_name_str": strategy_name_str,
        "execution_policy_str": execution_policy_str,
        "impact_profile_str": impact_profile_str,
        "optimal_capacity_float": optimal_capacity_float,
        "recommended_capacity_float": recommended_capacity_float,
        "recommended_capacity_censored_bool": recommended_capacity_censored_bool,
        "recommended_non_contiguous_pass_bool": recommended_non_contiguous_bool,
        "outer_capacity_float": outer_capacity_float,
        "outer_capacity_censored_bool": outer_capacity_censored_bool,
        "outer_non_contiguous_pass_bool": outer_non_contiguous_bool,
        "break_even_capacity_bracket_str": break_even_bracket_str,
        "outer_capacity_method_str": outer_method_str,
        "rolling_3y_eligible_window_count_int": int(
            rolling_reference_ser["rolling_3y_eligible_window_count_int"]
        ),
        "rolling_3y_eligible_window_count_basis_str": rolling_reference_basis_str,
        "aum_grid_list": capacity_curve_df["capital_base_float"].astype(float).tolist(),
        "recommended_rule_str": (
            "Complete liquidity coverage, P95 below Soft, P99 below Hard, overall and "
            "rolling 3y Sharpe erosion <= 20% where baseline rolling Sharpe >= 0.30, "
            "and incremental cost <= 25% of benchmark excess annual return."
        ),
        "outer_rule_str": (
            "Hard breaches below 5% plus positive stress benchmark excess annual return "
            "and stress cost below 50% of benchmark excess annual return."
        ),
        "actual_start_date_str": _single_curve_date_str(
            capacity_curve_df,
            "actual_start_date_str",
        ),
        "actual_end_date_str": _single_curve_date_str(
            capacity_curve_df,
            "actual_end_date_str",
        ),
        "assessed_order_count_int": int(len(assessed_order_df)),
        "total_order_count_int": int(len(order_diagnostics_df)),
        "unavailable_order_share_float": _safe_divide_float(
            len(order_diagnostics_df) - len(assessed_order_df),
            len(order_diagnostics_df),
        ),
        "academic_extrapolation_share_float": _safe_divide_float(
            int(
                assessed_order_df.get(
                    "academic_extrapolation_bool",
                    pd.Series(dtype=bool),
                ).sum()
            ),
            len(assessed_order_df),
        ),
        "proxy_extrapolation_share_float": _safe_divide_float(
            int(
                assessed_order_df.get(
                    "proxy_extrapolation_bool",
                    pd.Series(dtype=bool),
                ).sum()
            ),
            len(assessed_order_df),
        ),
        "model_extrapolation_share_float": _safe_divide_float(
            int(
                assessed_order_df.get(
                    "model_extrapolation_bool",
                    pd.Series(dtype=bool),
                ).sum()
            ),
            len(assessed_order_df),
        ),
        "model_assumption_dict": {
            "baseline_slippage_bps_float": BASELINE_SLIPPAGE_BPS_FLOAT,
            "adv_mean_lookback_int": ADV_MEAN_LOOKBACK_INT,
            "adv_median_lookback_int": ADV_MEDIAN_LOOKBACK_INT,
            "moc_central_lambda_1pct_adv_bps_float": MOC_CENTRAL_LAMBDA_1PCT_ADV_BPS_FLOAT,
            "moc_stress_lambda_1pct_adv_bps_float": MOC_STRESS_LAMBDA_1PCT_ADV_BPS_FLOAT,
            "impact_profile_str": impact_profile_str,
            "central_lambda_1pct_adv_bps_float": capacity_curve_df.iloc[0][
                "central_lambda_1pct_adv_bps_float"
            ],
            "stress_lambda_1pct_adv_bps_float": capacity_curve_df.iloc[0][
                "stress_lambda_1pct_adv_bps_float"
            ],
            "impact_exponent_float": IMPACT_EXPONENT_FLOAT,
            "rolling_baseline_sharpe_floor_float": ROLLING_BASELINE_SHARPE_FLOOR_FLOAT,
            "soft_order_adv_limit_float": policy_limit_tuple(execution_policy_str)[0],
            "hard_order_adv_limit_float": policy_limit_tuple(execution_policy_str)[1],
        },
    }


def _strategy_equity_ser(strategy_obj: Strategy) -> pd.Series:
    result_df = getattr(strategy_obj, "results", None)
    if not isinstance(result_df, pd.DataFrame) or "total_value" not in result_df.columns:
        return pd.Series(dtype=float)
    equity_ser = result_df["total_value"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    equity_ser.index = pd.to_datetime(equity_ser.index).normalize()
    return equity_ser.groupby(equity_ser.index).last().sort_index()


def _adjusted_equity_ser(
    baseline_equity_ser: pd.Series,
    assessed_order_df: pd.DataFrame,
    incremental_cost_column_str: str,
) -> pd.Series:
    if len(baseline_equity_ser) == 0:
        return pd.Series(dtype=float)
    if incremental_cost_column_str not in assessed_order_df.columns:
        return pd.Series(dtype=float)
    cost_df = assessed_order_df[["bar", incremental_cost_column_str]].copy()
    cost_df["bar"] = pd.to_datetime(cost_df["bar"]).dt.normalize()
    cost_df[incremental_cost_column_str] = (
        pd.to_numeric(cost_df[incremental_cost_column_str], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    cost_by_day_ser = cost_df.groupby("bar")[incremental_cost_column_str].sum()
    daily_cost_ser = cost_by_day_ser.reindex(baseline_equity_ser.index, fill_value=0.0)
    # Report-only overlay: preserve the baseline daily return path while each
    # incremental cost reduces the capital that compounds from that day onward.
    adjusted_equity_ser = pd.Series(index=baseline_equity_ser.index, dtype=float)
    adjusted_equity_ser.iloc[0] = max(
        float(baseline_equity_ser.iloc[0] - daily_cost_ser.iloc[0]),
        0.0,
    )
    for index_int in range(1, len(baseline_equity_ser)):
        prior_baseline_float = float(baseline_equity_ser.iloc[index_int - 1])
        baseline_growth_float = (
            float(baseline_equity_ser.iloc[index_int]) / prior_baseline_float
            if prior_baseline_float > 0.0
            else 0.0
        )
        adjusted_equity_ser.iloc[index_int] = max(
            float(adjusted_equity_ser.iloc[index_int - 1]) * baseline_growth_float
            - float(daily_cost_ser.iloc[index_int]),
            0.0,
        )
    return adjusted_equity_ser


def _benchmark_annual_return_tuple(
    strategy_obj: Strategy,
    pricing_data_df: pd.DataFrame,
    result_idx: pd.Index,
) -> tuple[float, str | None]:
    benchmark_symbol_obj = getattr(strategy_obj, "_performance_benchmark_symbol_str", None)
    benchmark_adjustment_obj = getattr(
        strategy_obj,
        "_performance_benchmark_adjustment_str",
        None,
    )
    if (
        benchmark_symbol_obj is None
        or benchmark_adjustment_obj in {None, "not_declared"}
        or len(result_idx) < 2
    ):
        return np.nan, None
    benchmark_symbol_str = str(benchmark_symbol_obj)
    benchmark_data_symbol_map_dict = getattr(
        strategy_obj,
        "_benchmark_data_symbol_map_dict",
        {},
    )
    benchmark_data_symbol_str = benchmark_data_symbol_map_dict.get(
        benchmark_symbol_str,
        benchmark_symbol_str,
    )
    if (benchmark_data_symbol_str, "Close") not in pricing_data_df.columns:
        return np.nan, benchmark_symbol_str
    close_ser = pricing_data_df[
        (benchmark_data_symbol_str, "Close")
    ].astype(float).dropna()
    close_ser.index = pd.to_datetime(close_ser.index).normalize()
    aligned_ser = close_ser.reindex(pd.DatetimeIndex(result_idx), method="ffill").dropna()
    return _annualized_return_float(aligned_ser), benchmark_symbol_str


def _annualized_return_float(equity_ser: pd.Series) -> float:
    clean_ser = pd.Series(equity_ser, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_ser) < 2 or clean_ser.iloc[0] <= 0.0 or clean_ser.iloc[-1] <= 0.0:
        return np.nan
    year_float = (len(clean_ser) - 1) / 252.0
    if year_float <= 0.0:
        return np.nan
    return float((clean_ser.iloc[-1] / clean_ser.iloc[0]) ** (1.0 / year_float) - 1.0)


def _sharpe_float(equity_ser: pd.Series) -> float:
    clean_ser = pd.Series(equity_ser, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_ser) < 3 or bool((clean_ser <= 0.0).any()):
        return np.nan
    return_ser = clean_ser.pct_change(fill_method=None).dropna()
    std_float = float(return_ser.std(ddof=1))
    if std_float <= 0.0 or not np.isfinite(std_float):
        return np.nan
    return float(return_ser.mean() / std_float * np.sqrt(252.0))


def _worst_eligible_rolling_sharpe_erosion_tuple(
    baseline_equity_ser: pd.Series,
    central_equity_ser: pd.Series,
) -> tuple[float, int]:
    if len(baseline_equity_ser) <= ROLLING_THREE_YEAR_TRADING_DAYS_INT:
        return np.nan, 0
    baseline_return_ser = baseline_equity_ser.pct_change(fill_method=None)
    central_return_ser = central_equity_ser.pct_change(fill_method=None)
    # *** CRITICAL *** rolling windows end at T and contain only realized returns
    # through T. They are diagnostics, never inputs to historical order sizing.
    baseline_sharpe_ser = (
        baseline_return_ser.rolling(ROLLING_THREE_YEAR_TRADING_DAYS_INT).mean()
        / baseline_return_ser.rolling(ROLLING_THREE_YEAR_TRADING_DAYS_INT).std(ddof=1)
        * np.sqrt(252.0)
    )
    central_sharpe_ser = (
        central_return_ser.rolling(ROLLING_THREE_YEAR_TRADING_DAYS_INT).mean()
        / central_return_ser.rolling(ROLLING_THREE_YEAR_TRADING_DAYS_INT).std(ddof=1)
        * np.sqrt(252.0)
    )
    return _eligible_rolling_sharpe_erosion_tuple(
        baseline_sharpe_ser,
        central_sharpe_ser,
    )


def _eligible_rolling_sharpe_erosion_tuple(
    baseline_sharpe_ser: pd.Series,
    central_sharpe_ser: pd.Series,
) -> tuple[float, int]:
    eligible_baseline_ser = baseline_sharpe_ser[
        baseline_sharpe_ser >= ROLLING_BASELINE_SHARPE_FLOOR_FLOAT
    ]
    eligible_window_count_int = int(len(eligible_baseline_ser))
    if eligible_window_count_int == 0:
        return np.nan, 0
    aligned_central_ser = central_sharpe_ser.reindex(eligible_baseline_ser.index)
    erosion_ser = (
        (eligible_baseline_ser - aligned_central_ser) / eligible_baseline_ser.abs()
    ).clip(lower=0.0)
    erosion_ser = erosion_ser.replace([np.inf, -np.inf], np.nan).dropna()
    return (
        float(erosion_ser.max()) if len(erosion_ser) else np.nan,
        eligible_window_count_int,
    )


def _benchmark_excess_return_float(
    strategy_return_float: float,
    benchmark_return_float: float,
) -> float:
    if not _is_finite_float(strategy_return_float) or not _is_finite_float(benchmark_return_float):
        return np.nan
    return float(strategy_return_float - benchmark_return_float)


def _erosion_float(baseline_value_float: float, adjusted_value_float: float) -> float:
    if not _is_finite_float(baseline_value_float) or baseline_value_float <= 0.0:
        return np.nan
    if not _is_finite_float(adjusted_value_float):
        return np.nan
    return float(max(0.0, (baseline_value_float - adjusted_value_float) / abs(baseline_value_float)))


def _cost_consumption_float(baseline_active_float: float, adjusted_active_float: float) -> float:
    if not _is_finite_float(baseline_active_float) or baseline_active_float <= 0.0:
        return np.nan
    if not _is_finite_float(adjusted_active_float):
        return np.nan
    return float(max(0.0, (baseline_active_float - adjusted_active_float) / baseline_active_float))


def _contiguous_capacity_tuple(
    curve_df: pd.DataFrame,
    raw_pass_column_str: str,
) -> tuple[pd.Series, float | None, bool, bool]:
    sorted_capacity_df = curve_df.sort_values("capital_base_float")
    contiguous_pass_bool_list: list[bool] = []
    failure_seen_bool = False
    later_pass_after_failure_bool = False
    for raw_pass_bool in sorted_capacity_df[raw_pass_column_str].fillna(False).astype(bool):
        if failure_seen_bool:
            contiguous_pass_bool_list.append(False)
            later_pass_after_failure_bool = later_pass_after_failure_bool or bool(raw_pass_bool)
            continue
        if raw_pass_bool:
            contiguous_pass_bool_list.append(True)
        else:
            failure_seen_bool = True
            contiguous_pass_bool_list.append(False)

    contiguous_pass_ser = pd.Series(
        contiguous_pass_bool_list,
        index=sorted_capacity_df.index,
        dtype=bool,
    ).reindex(curve_df.index, fill_value=False)
    passing_capacity_df = curve_df[contiguous_pass_ser]
    capacity_float = (
        float(passing_capacity_df["capital_base_float"].max())
        if len(passing_capacity_df)
        else None
    )
    capacity_censored_bool = bool(
        capacity_float is not None
        and capacity_float == float(curve_df["capital_base_float"].max())
    )
    return (
        contiguous_pass_ser,
        capacity_float,
        later_pass_after_failure_bool,
        capacity_censored_bool,
    )


def _single_curve_date_str(curve_df: pd.DataFrame, column_str: str) -> str | None:
    date_value_list = [
        str(value_obj)
        for value_obj in curve_df[column_str].dropna().unique().tolist()
    ]
    if len(date_value_list) > 1:
        raise ValueError(f"All AUM runs must share one {column_str}: {date_value_list}.")
    return date_value_list[0] if date_value_list else None


def _optimal_capacity_float(curve_df: pd.DataFrame) -> float | None:
    supported_bool_ser = (
        (curve_df["liquidity_complete_bool"] == True)
        & (curve_df["order_adv_p95_float"] <= curve_df["soft_limit_float"])
        & (curve_df["order_adv_p99_float"] <= curve_df["hard_limit_float"])
    )
    supported_df = curve_df[supported_bool_ser].copy()
    supported_df["central_sharpe_float"] = pd.to_numeric(
        supported_df["central_sharpe_float"],
        errors="coerce",
    )
    supported_df = supported_df.dropna(subset=["central_sharpe_float"])
    if len(supported_df) == 0:
        return None
    optimal_row_ser = supported_df.sort_values(
        ["central_sharpe_float", "capital_base_float"],
        ascending=[False, False],
    ).iloc[0]
    return float(optimal_row_ser["capital_base_float"])


def _break_even_bracket_str(curve_df: pd.DataFrame) -> str:
    sorted_df = curve_df.sort_values("capital_base_float")
    finite_pair_list: list[tuple[float, float]] = []
    previous_pair: tuple[float, float] | None = None
    for _, row_ser in sorted_df.iterrows():
        current_pair = (
            float(row_ser["capital_base_float"]),
            _coerce_float(row_ser["central_benchmark_excess_return_float"]),
        )
        if not _is_finite_float(current_pair[1]):
            previous_pair = None
            continue
        finite_pair_list.append(current_pair)
        if current_pair[1] == 0.0:
            return _fmt_dollar_str(current_pair[0])
        if previous_pair is not None and previous_pair[1] * current_pair[1] < 0.0:
            return (
                f"{_fmt_dollar_str(previous_pair[0])} to "
                f"{_fmt_dollar_str(current_pair[0])}"
            )
        previous_pair = current_pair

    if not finite_pair_list:
        return "Not estimable"
    sign_set = {
        1 if benchmark_excess_return_float > 0.0 else -1
        for _, benchmark_excess_return_float in finite_pair_list
    }
    if sign_set == {1}:
        return f"Above {_fmt_dollar_str(finite_pair_list[-1][0])}"
    if sign_set == {-1}:
        return f"Below {_fmt_dollar_str(finite_pair_list[0][0])}"
    return "Not estimable from adjacent finite grid points"


def _build_report_html_str(study_result_obj: CapacityStudyResult) -> str:
    summary_dict = study_result_obj.summary_dict
    headline_window_str = str(summary_dict["headline_window_str"])
    curve_df = study_result_obj.capacity_curve_df[
        study_result_obj.capacity_curve_df["window_str"] == headline_window_str
    ].copy()
    headline_order_df = study_result_obj.order_diagnostics_df[
        study_result_obj.order_diagnostics_df["window_str"] == headline_window_str
    ].copy()
    window_summary_dict = summary_dict["window_summary_dict"]
    full_history_summary_dict = window_summary_dict.get(FULL_HISTORY_WINDOW_STR, {})
    full_history_curve_df = study_result_obj.capacity_curve_df[
        study_result_obj.capacity_curve_df["window_str"] == FULL_HISTORY_WINDOW_STR
    ].copy()
    execution_policy_str = study_result_obj.execution_policy_str
    impact_profile_str = study_result_obj.impact_profile_str
    optimal_capacity_str = _fmt_optional_dollar_str(
        summary_dict.get("optimal_capacity_float")
    )
    recommended_capacity_str = _fmt_capacity_str(
        summary_dict.get("recommended_capacity_float"),
        bool(summary_dict.get("recommended_capacity_censored_bool")),
    )
    outer_capacity_str = _fmt_capacity_str(
        summary_dict.get("outer_capacity_float"),
        bool(summary_dict.get("outer_capacity_censored_bool")),
    )
    break_even_str = html.escape(str(summary_dict.get("break_even_capacity_bracket_str")))
    policy_explanation_str = (
        "This strategy trades at the open. Its explicit impact profile applies square-root "
        "Central and Stress costs as orders grow relative to ADV. Only the incremental amount "
        "above the 2.5 bps already in the baseline is subtracted."
        if execution_policy_str == MOO_EXECUTION_POLICY_STR
        else
        "This strategy trades at the close. The central and stress models use square-root "
        "impact, but 2.5 bps remains a floor so the capacity overlay can never improve the baseline."
    )
    read_first_str = (
        f"Current deployable capacity uses {summary_dict.get('actual_start_date_str')} through "
        f"{summary_dict.get('actual_end_date_str')}. The highest fully supported contiguous "
        f"grid point is {recommended_capacity_str}. "
        f"The outer estimate is {outer_capacity_str}. The strategy declares {execution_policy_str}; "
        "the report uses only that auction model."
    )
    historical_warning_html_str = (
        "<p><b>Historical feasibility warning:</b> current Recommended Max exceeds the "
        "full-history result. The headline describes current modeled deployability, not proof "
        "that the entire historical track record could have carried the same AUM.</p>"
        if bool(summary_dict.get("historical_feasibility_warning_bool"))
        else ""
    )
    non_contiguous_classification_label_list = []
    if bool(summary_dict.get("recommended_non_contiguous_pass_bool")):
        non_contiguous_classification_label_list.append("Recommended Max")
    if bool(summary_dict.get("outer_non_contiguous_pass_bool")):
        non_contiguous_classification_label_list.append("Outer Capacity")
    non_contiguous_warning_html_str = ""
    if non_contiguous_classification_label_list:
        non_contiguous_label_str = " and ".join(
            non_contiguous_classification_label_list
        )
        non_contiguous_warning_html_str = (
            "<p><b>Non-contiguous diagnostic:</b> one or more larger AUM points passed "
            "after an earlier failure for "
            f"{html.escape(non_contiguous_label_str)}. They are diagnostic only and do "
            "not raise the contiguous capacity classification.</p>"
        )
    chart_one_series_list = [
        ("Baseline Sharpe", "baseline_sharpe_float", "#2563eb"),
        ("Capacity Sharpe", "central_sharpe_float", "#059669"),
        ("Stress Sharpe", "stress_sharpe_float", "#dc2626"),
    ]
    performance_chart_str = _line_chart_svg_str(
        curve_df,
        chart_one_series_list,
        "Performance versus AUM",
        percent_axis_bool=False,
    )
    liquidity_chart_df = curve_df.copy()
    liquidity_chart_df["soft_limit_chart_float"] = liquidity_chart_df["soft_limit_float"]
    liquidity_chart_df["hard_limit_chart_float"] = liquidity_chart_df["hard_limit_float"]
    liquidity_chart_str = _line_chart_svg_str(
        liquidity_chart_df,
        [
            ("P95 Order/ADV", "order_adv_p95_float", "#2563eb"),
            ("P99 Order/ADV", "order_adv_p99_float", "#7c3aed"),
            ("Soft limit", "soft_limit_chart_float", "#d97706"),
            ("Hard limit", "hard_limit_chart_float", "#dc2626"),
        ],
        "Liquidity usage versus AUM",
        percent_axis_bool=True,
    )
    breach_chart_str = _line_chart_svg_str(
        curve_df,
        [
            ("Soft breaches", "soft_breach_share_float", "#d97706"),
            ("Hard breaches", "hard_breach_share_float", "#dc2626"),
        ],
        "Share of orders beyond limits",
        percent_axis_bool=True,
    )
    worked_example_str = _worked_example_html_str(
        execution_policy_str,
        impact_profile_str,
    )
    table_html_str = _capacity_curve_table_html_str(curve_df, execution_policy_str)
    bottleneck_html_str = _bottleneck_table_html_str(headline_order_df)
    equity_chart_str, equity_chart_note_str = _selected_equity_chart_tuple(
        study_result_obj,
    )
    profile_assumption_dict = impact_profile_assumption_dict(
        execution_policy_str,
        impact_profile_str,
    )
    profile_display_str = impact_profile_str or "MOC default"
    proxy_warning_str = (
        "This ETF profile uses common-stock auction estimates as a sensitivity proxy. "
        "It is not ETF-specific empirical calibration and must be replaced by live TCA."
        if bool(profile_assumption_dict["proxy_bool"])
        else "This is a pre-TCA model estimate, not a live execution guarantee."
    )
    academic_extrapolation_share_float = _coerce_float(
        summary_dict.get("academic_extrapolation_share_float")
    )
    extrapolation_warning_html_str = (
        "<li><b>Academic extrapolation warning:</b> some common-stock MOO orders "
        "exceed 1% ADV, outside the supported profile range. Treat those AUM results "
        "as diagnostic only.</li>"
        if execution_policy_str == MOO_EXECUTION_POLICY_STR
        and not bool(profile_assumption_dict["proxy_bool"])
        and _is_finite_float(academic_extrapolation_share_float)
        and academic_extrapolation_share_float > 0.0
        else ""
    )
    proxy_extrapolation_share_float = _coerce_float(
        summary_dict.get("proxy_extrapolation_share_float")
    )
    proxy_extrapolation_warning_html_str = (
        "<li><b>ETF proxy extrapolation warning:</b> some ETF MOO orders exceed 1% ADV. "
        "The common-stock-derived coefficient has no empirical support at those sizes; treat "
        "those AUM results as diagnostic only.</li>"
        if bool(profile_assumption_dict["proxy_bool"])
        and _is_finite_float(proxy_extrapolation_share_float)
        and proxy_extrapolation_share_float > 0.0
        else ""
    )
    full_history_section_html_str = _historical_feasibility_section_html_str(
        full_history_summary_dict,
        full_history_curve_df,
        execution_policy_str,
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CapacityAnalysis - {html.escape(study_result_obj.strategy_name_str)}</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ef;--paper:#fff;--bg:#f4f7fb;--blue:#2563eb;--green:#059669;--amber:#d97706;--red:#dc2626}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 18px 60px}} h1{{margin:0 0 4px;font-size:32px}} h2{{margin:34px 0 12px;font-size:22px}} h3{{margin:22px 0 8px}} p{{max-width:900px}} .muted{{color:var(--muted)}}
.panel{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px;margin:14px 0;box-shadow:0 4px 18px rgba(15,23,42,.04)}}
.read-first{{border-left:5px solid var(--blue)}} .cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px}} .card b{{display:block;font-size:23px;margin-top:5px}}
.charts{{display:grid;grid-template-columns:1fr;gap:14px}} svg{{width:100%;height:auto;display:block}} .table-wrap{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{background:#f8fafc}}
code{{background:#eef2ff;padding:2px 5px;border-radius:5px}} .formula{{font-family:ui-monospace,Consolas,monospace;background:#f8fafc;border:1px solid var(--line);padding:12px;border-radius:9px;overflow:auto}}
ul{{padding-left:22px}} a{{color:var(--blue)}} @media(max-width:760px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>CapacityAnalysis</h1><div class="muted">{html.escape(study_result_obj.strategy_name_str)} · Declared execution: {execution_policy_str}</div>
<section class="panel read-first"><h2>Read this first</h2><p>{html.escape(read_first_str)}</p><p>{html.escape(policy_explanation_str)}</p>{historical_warning_html_str}{non_contiguous_warning_html_str}</section>
<section class="cards"><div class="card">Optimal capacity<b>{optimal_capacity_str}</b><span class="muted">Highest Central Sharpe among supported current grid points</span></div><div class="card">Recommended max<b>{recommended_capacity_str}</b><span class="muted">Current deployable estimate; contiguous from the lowest grid point</span></div><div class="card">Outer capacity<b>{outer_capacity_str}</b><span class="muted">Stretch estimate, not an operating target</span></div><div class="card">Break-even bracket<b>{break_even_str}</b><span class="muted">Central benchmark-excess-return crossing on the tested grid</span></div></section>
<h2>Current deployable capacity — recent five years</h2><section class="charts"><div class="panel">{performance_chart_str}</div><div class="panel">{equity_chart_str}<p class="muted">{html.escape(equity_chart_note_str)}</p></div><div class="panel">{liquidity_chart_str}</div><div class="panel">{breach_chart_str}</div></section>
<section class="panel"><h2>Current-window AUM results</h2>{table_html_str}</section>
{full_history_section_html_str}
<section class="panel"><h2>How this works</h2><p><b>ADV</b> means average daily dollar volume. <b>Order/ADV</b> asks how large one completed order is compared with the stock's normal liquidity. Bigger ratios are harder to execute without affecting price.</p><p>For every order we calculate both the lagged 10-day mean and lagged 20-day median dollar ADV. We use the lower value. Both windows stop at the previous session, so the calculation does not know today's final volume.</p><div class="formula">q = absolute net order notional / robust lagged dollar ADV<br>impact bps = lambda at 1% ADV x sqrt(q / 0.01)<br>incremental bps = max(0, impact bps - 2.5)</div><p>The ordinary backtest already includes IBKR Fixed commissions and 2.5 bps slippage. CapacityAnalysis never adds those costs again. It subtracts only the non-negative amount above the existing 2.5 bps floor.</p><p><b>Impact profile:</b> {html.escape(profile_display_str)}. <b>Central lambda:</b> {_fmt_bps_str(profile_assumption_dict['central_lambda_1pct_adv_bps_float'])}. <b>Stress lambda:</b> {_fmt_bps_str(profile_assumption_dict['stress_lambda_1pct_adv_bps_float'])}. <b>Confidence:</b> {html.escape(str(profile_assumption_dict['model_confidence_str']))}.</p>{worked_example_str}</section>
<section class="panel"><h2>Assumptions and limitations</h2><ul><li>One strategy declares one auction policy and every MOO strategy explicitly declares one impact profile. The analyzer does not guess.</li><li>{html.escape(proxy_warning_str)}</li>{extrapolation_warning_html_str}{proxy_extrapolation_warning_html_str}<li>The profile lambdas are unconditional auction estimates; the model does not scale impact by each day's volatility. Stress lambda is the sensitivity buffer.</li><li>Every modeled order receives a complete fill. Partial fills, queue position, routing, auction imbalance, borrow, and live broker behavior are not simulated.</li><li>ETF proxy bias direction is unknown: creation/redemption can add liquidity, while a thin opening auction can be worse than the common-stock proxy.</li><li>Orders are aggregated only within this strategy by date, asset, and side. Client and pod aggregation is not modeled.</li><li>Norgate Close is suitable for MOC research; Norgate Open can differ from an official listing-exchange opening auction print.</li><li>The rolling gate uses only three-year windows with baseline Sharpe at least {ROLLING_BASELINE_SHARPE_FLOOR_FLOAT:.2f}; eligible current-window count at the {html.escape(str(summary_dict.get('rolling_3y_eligible_window_count_basis_str', 'diagnostic'))).replace('_', ' ')} point: {int(summary_dict.get('rolling_3y_eligible_window_count_int', 0))}.</li><li>Recommended and Outer capacity are not awarded when any saved order lacks both required lagged ADV measures.</li><li>Capacity is a research estimate. Live TCA should replace or recalibrate these assumptions.</li><li>Unavailable order share across current-window rows: {_fmt_pct_str(summary_dict.get('unavailable_order_share_float'))}.</li><li>Common-stock academic extrapolation share: {_fmt_pct_str(summary_dict.get('academic_extrapolation_share_float'))}.</li><li>ETF proxy extrapolation share: {_fmt_pct_str(summary_dict.get('proxy_extrapolation_share_float'))}.</li></ul><p class="muted">Research anchors: <a href="https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/price-impact-in-closing-auctions-opening-auctions-and-continuous-markets-a-benchmark-for-cost-of-trading-on-anomalies/0F72910A79C5B42CF6E85F55164CE846">Goyal, Jegadeesh and Wu (2026), Tables 3 and 5</a>; <a href="https://norgatedata.com/data-content-tables.php">Norgate price definitions</a>; <a href="https://www.interactivebrokers.com/en/pricing/commissions-stocks.php">IBKR commissions</a>.</p></section>
<section class="panel"><h2>Largest liquidity bottlenecks</h2><p class="muted">Highest robust Order/ADV observations across the AUM study.</p>{bottleneck_html_str}</section>
</main></body></html>"""


def _historical_feasibility_section_html_str(
    full_history_summary_dict: dict[str, object],
    full_history_curve_df: pd.DataFrame,
    execution_policy_str: str,
) -> str:
    if not full_history_summary_dict or len(full_history_curve_df) == 0:
        return ""
    recommended_str = _fmt_capacity_str(
        full_history_summary_dict.get("recommended_capacity_float"),
        bool(full_history_summary_dict.get("recommended_capacity_censored_bool")),
    )
    outer_str = _fmt_capacity_str(
        full_history_summary_dict.get("outer_capacity_float"),
        bool(full_history_summary_dict.get("outer_capacity_censored_bool")),
    )
    non_contiguous_warning_str = (
        " Larger passing points after the first failure are diagnostic only."
        if bool(full_history_summary_dict.get("recommended_non_contiguous_pass_bool"))
        else ""
    )
    performance_chart_str = _line_chart_svg_str(
        full_history_curve_df,
        [
            ("Baseline Sharpe", "baseline_sharpe_float", "#2563eb"),
            ("Capacity Sharpe", "central_sharpe_float", "#059669"),
            ("Stress Sharpe", "stress_sharpe_float", "#dc2626"),
        ],
        "Full-history performance versus AUM",
        percent_axis_bool=False,
    )
    table_html_str = _capacity_curve_table_html_str(
        full_history_curve_df,
        execution_policy_str,
    )
    return (
        '<section class="panel"><h2>Full-history feasibility</h2>'
        f'<p>This window runs from {html.escape(str(full_history_summary_dict.get("actual_start_date_str")))} '
        f'through {html.escape(str(full_history_summary_dict.get("actual_end_date_str")))}. '
        f'Full-history Recommended Max is <b>{recommended_str}</b>; Outer is <b>{outer_str}</b>.'
        f'{html.escape(non_contiguous_warning_str)}</p>{performance_chart_str}'
        f'<h3>Full-history AUM results</h3>{table_html_str}</section>'
    )


def _worked_example_html_str(
    execution_policy_str: str,
    impact_profile_str: str | None,
) -> str:
    example_ratio_float = 0.001
    if execution_policy_str == MOC_EXECUTION_POLICY_STR:
        impact_float = square_root_impact_bps_float(
            example_ratio_float,
            MOC_CENTRAL_LAMBDA_1PCT_ADV_BPS_FLOAT,
        )
        return (
            "<h3>Worked MOC example</h3><p>An order of $100,000 against $100 million "
            "ADV is 0.10% ADV. Central impact is "
            f"<code>8.2 × sqrt(0.10% / 1.00%) = {impact_float:.2f} bps</code>. "
            f"The applied implicit cost is <code>max(2.5, {impact_float:.2f}) = "
            f"{max(2.5, impact_float):.2f} bps</code>.</p>"
        )
    assumption_dict = impact_profile_assumption_dict(
        execution_policy_str,
        impact_profile_str,
    )
    lambda_bps_float = float(assumption_dict["central_lambda_1pct_adv_bps_float"])
    impact_float = square_root_impact_bps_float(example_ratio_float / 2.0, lambda_bps_float)
    incremental_float = max(0.0, impact_float - BASELINE_SLIPPAGE_BPS_FLOAT)
    return (
        "<h3>Worked MOO example</h3><p>An order of $50,000 against $100 million ADV "
        "is 0.05% ADV. Central impact is "
        f"<code>{lambda_bps_float:g} x sqrt(0.05% / 1.00%) = {impact_float:.2f} bps</code>. "
        f"The additional cost above the existing 2.5 bps is {incremental_float:.2f} bps.</p>"
    )


def _line_chart_svg_str(
    curve_df: pd.DataFrame,
    series_spec_list: list[tuple[str, str, str]],
    title_str: str,
    percent_axis_bool: bool,
) -> str:
    width_float, height_float = 920.0, 330.0
    left_float, right_float, top_float, bottom_float = 74.0, 24.0, 45.0, 58.0
    plot_width_float = width_float - left_float - right_float
    plot_height_float = height_float - top_float - bottom_float
    value_list: list[float] = []
    for _, column_str, _ in series_spec_list:
        value_list.extend(
            [
                float(value_float)
                for value_float in pd.to_numeric(curve_df[column_str], errors="coerce").dropna()
                if np.isfinite(value_float)
            ]
        )
    if not value_list:
        return f"<h3>{html.escape(title_str)}</h3><p>No chartable data.</p>"
    y_min_float = min(0.0, min(value_list))
    y_max_float = max(value_list)
    if y_max_float <= y_min_float:
        y_max_float = y_min_float + 1.0
    y_padding_float = (y_max_float - y_min_float) * 0.08
    y_min_float -= y_padding_float
    y_max_float += y_padding_float

    def x_float(index_int: int) -> float:
        denominator_int = max(1, len(curve_df) - 1)
        return left_float + plot_width_float * index_int / denominator_int

    def y_float(value_float: float) -> float:
        return top_float + plot_height_float * (y_max_float - value_float) / (y_max_float - y_min_float)

    svg_part_list = [
        f'<h3>{html.escape(title_str)}</h3><svg viewBox="0 0 {width_float:.0f} {height_float:.0f}" role="img" aria-label="{html.escape(title_str)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for tick_int in range(5):
        tick_value_float = y_min_float + (y_max_float - y_min_float) * tick_int / 4.0
        tick_y_float = y_float(tick_value_float)
        tick_label_str = (
            f"{tick_value_float * 100.0:.2f}%" if percent_axis_bool else f"{tick_value_float:.2f}"
        )
        svg_part_list.append(
            f'<line x1="{left_float}" y1="{tick_y_float:.1f}" x2="{width_float-right_float}" y2="{tick_y_float:.1f}" stroke="#e5e7eb"/>'
            f'<text x="{left_float-8}" y="{tick_y_float+4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{tick_label_str}</text>'
        )
    for index_int, (_, row_ser) in enumerate(curve_df.iterrows()):
        x_value_float = x_float(index_int)
        svg_part_list.append(
            f'<text x="{x_value_float:.1f}" y="{height_float-25:.1f}" text-anchor="middle" font-size="10" fill="#64748b">{_fmt_compact_dollar_str(row_ser["capital_base_float"])}</text>'
        )
    legend_x_float = left_float
    for label_str, column_str, color_str in series_spec_list:
        point_str_list: list[str] = []
        for index_int, value_obj in enumerate(pd.to_numeric(curve_df[column_str], errors="coerce")):
            if _is_finite_float(value_obj):
                point_str_list.append(f"{x_float(index_int):.1f},{y_float(float(value_obj)):.1f}")
        if point_str_list:
            svg_part_list.append(
                f'<polyline points="{" ".join(point_str_list)}" fill="none" stroke="{color_str}" stroke-width="2.5"/>'
            )
            for point_str in point_str_list:
                point_x_str, point_y_str = point_str.split(",")
                svg_part_list.append(
                    f'<circle cx="{point_x_str}" cy="{point_y_str}" r="3" fill="{color_str}"/>'
                )
        svg_part_list.append(
            f'<line x1="{legend_x_float:.1f}" y1="22" x2="{legend_x_float+18:.1f}" y2="22" stroke="{color_str}" stroke-width="3"/>'
            f'<text x="{legend_x_float+23:.1f}" y="26" font-size="11" fill="#334155">{html.escape(label_str)}</text>'
        )
        legend_x_float += 150.0
    svg_part_list.append("</svg>")
    return "".join(svg_part_list)


def _selected_equity_chart_tuple(
    study_result_obj: CapacityStudyResult,
) -> tuple[str, str]:
    headline_window_str = str(study_result_obj.summary_dict["headline_window_str"])
    recommended_capacity_obj = study_result_obj.summary_dict.get(
        "recommended_capacity_float"
    )
    diagnostic_only_bool = recommended_capacity_obj is None
    available_aum_float_list = [
        aum_float
        for window_str, aum_float in study_result_obj.equity_curve_by_window_aum_dict
        if window_str == headline_window_str
    ]
    selected_aum_float = (
        min(available_aum_float_list)
        if diagnostic_only_bool
        else float(recommended_capacity_obj)
    )
    equity_curve_df = study_result_obj.equity_curve_by_window_aum_dict[
        (headline_window_str, selected_aum_float)
    ]
    normalized_equity_df = pd.DataFrame(index=equity_curve_df.index)
    for column_str in [
        "baseline_equity_float",
        "central_equity_float",
        "stress_equity_float",
    ]:
        equity_ser = pd.to_numeric(equity_curve_df[column_str], errors="coerce")
        first_valid_ser = equity_ser.dropna()
        if len(first_valid_ser) == 0 or first_valid_ser.iloc[0] <= 0.0:
            normalized_equity_df[column_str] = np.nan
        else:
            normalized_equity_df[column_str] = equity_ser / float(first_valid_ser.iloc[0])
    chart_str = _equity_chart_svg_str(
        normalized_equity_df,
        f"Normalized equity at {_fmt_dollar_str(selected_aum_float)}",
    )
    note_str = (
        "Diagnostic only - no AUM point met every Recommended Max rule."
        if diagnostic_only_bool
        else "Shown at Recommended Max Capacity; all curves start at 1.0."
    )
    return chart_str, note_str


def _equity_chart_svg_str(equity_curve_df: pd.DataFrame, title_str: str) -> str:
    clean_df = equity_curve_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if len(clean_df) == 0:
        return f"<h3>{html.escape(title_str)}</h3><p>No chartable equity data.</p>"
    max_point_count_int = 320
    if len(clean_df) > max_point_count_int:
        position_arr = np.linspace(0, len(clean_df) - 1, max_point_count_int).astype(int)
        clean_df = clean_df.iloc[np.unique(position_arr)]

    width_float, height_float = 920.0, 330.0
    left_float, right_float, top_float, bottom_float = 74.0, 24.0, 45.0, 58.0
    plot_width_float = width_float - left_float - right_float
    plot_height_float = height_float - top_float - bottom_float
    numeric_df = clean_df.apply(pd.to_numeric, errors="coerce")
    value_arr = numeric_df.to_numpy(dtype=float)
    finite_value_arr = value_arr[np.isfinite(value_arr)]
    if len(finite_value_arr) == 0:
        return f"<h3>{html.escape(title_str)}</h3><p>No chartable equity data.</p>"
    y_min_float = min(0.0, float(finite_value_arr.min()))
    y_max_float = float(finite_value_arr.max())
    if y_max_float <= y_min_float:
        y_max_float = y_min_float + 1.0
    y_padding_float = (y_max_float - y_min_float) * 0.08
    y_min_float -= y_padding_float
    y_max_float += y_padding_float

    def x_float(index_int: int) -> float:
        return left_float + plot_width_float * index_int / max(1, len(numeric_df) - 1)

    def y_float(value_float: float) -> float:
        return top_float + plot_height_float * (y_max_float - value_float) / (
            y_max_float - y_min_float
        )

    series_spec_list = [
        ("Baseline", "baseline_equity_float", "#2563eb"),
        ("Central", "central_equity_float", "#059669"),
        ("Stress", "stress_equity_float", "#dc2626"),
    ]
    svg_part_list = [
        f'<h3>{html.escape(title_str)}</h3><svg viewBox="0 0 {width_float:.0f} {height_float:.0f}" role="img" aria-label="{html.escape(title_str)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for tick_int in range(5):
        tick_value_float = y_min_float + (y_max_float - y_min_float) * tick_int / 4.0
        tick_y_float = y_float(tick_value_float)
        svg_part_list.append(
            f'<line x1="{left_float}" y1="{tick_y_float:.1f}" x2="{width_float-right_float}" y2="{tick_y_float:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{left_float-10}" y="{tick_y_float+4:.1f}" text-anchor="end" font-size="10" fill="#64748b">{tick_value_float:.2f}</text>'
        )
    for tick_int in range(5):
        row_int = round((len(numeric_df) - 1) * tick_int / 4.0)
        tick_x_float = x_float(row_int)
        date_str = str(pd.Timestamp(numeric_df.index[row_int]).date())
        svg_part_list.append(
            f'<text x="{tick_x_float:.1f}" y="{height_float-25:.1f}" text-anchor="middle" font-size="10" fill="#64748b">{date_str}</text>'
        )
    legend_x_float = left_float
    for label_str, column_str, color_str in series_spec_list:
        point_str_list = []
        for index_int, value_obj in enumerate(numeric_df[column_str]):
            if _is_finite_float(value_obj):
                point_str_list.append(
                    f"{x_float(index_int):.1f},{y_float(float(value_obj)):.1f}"
                )
        if point_str_list:
            svg_part_list.append(
                f'<polyline points="{" ".join(point_str_list)}" fill="none" stroke="{color_str}" stroke-width="2.2"/>'
            )
        svg_part_list.append(
            f'<line x1="{legend_x_float:.1f}" y1="22" x2="{legend_x_float+18:.1f}" y2="22" stroke="{color_str}" stroke-width="3"/>'
            f'<text x="{legend_x_float+23:.1f}" y="26" font-size="11" fill="#334155">{label_str}</text>'
        )
        legend_x_float += 120.0
    svg_part_list.append("</svg>")
    return "".join(svg_part_list)


def _capacity_curve_table_html_str(curve_df: pd.DataFrame, execution_policy_str: str) -> str:
    row_html_list: list[str] = []
    for _, row_ser in curve_df.iterrows():
        stress_sharpe_str = _fmt_float_str(row_ser.get("stress_sharpe_float"))
        row_html_list.append(
            "<tr>"
            f"<td>{_fmt_dollar_str(row_ser['capital_base_float'])}</td>"
            f"<td>{_fmt_float_str(row_ser['baseline_sharpe_float'])}</td>"
            f"<td>{_fmt_float_str(row_ser['central_sharpe_float'])}</td>"
            f"<td>{stress_sharpe_str}</td>"
            f"<td>{_fmt_pct_str(row_ser['order_adv_p95_float'])}</td>"
            f"<td>{_fmt_pct_str(row_ser['order_adv_p99_float'])}</td>"
            f"<td>{_fmt_pct_str(row_ser['central_cost_consumption_of_benchmark_excess_float'])}</td>"
            f"<td>{_fmt_pct_str(row_ser['hard_breach_share_float'])}</td>"
            f"<td>{'Yes' if bool(row_ser.get('recommended_pass_bool')) else 'No'}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>AUM</th><th>Baseline Sharpe</th>'
        '<th>Capacity Sharpe</th><th>Stress Sharpe</th><th>P95 Order/ADV</th>'
        '<th>P99 Order/ADV</th><th>Cost / benchmark excess return</th><th>Hard breaches</th>'
        f'<th>Recommended pass</th></tr></thead><tbody>{"".join(row_html_list)}</tbody></table></div>'
    )


def _bottleneck_table_html_str(order_diagnostics_df: pd.DataFrame) -> str:
    assessed_df = order_diagnostics_df[
        order_diagnostics_df.get("assessed_bool", pd.Series(dtype=bool)) == True
    ].copy()
    if len(assessed_df) == 0:
        return "<p>No assessed orders.</p>"
    top_df = assessed_df.sort_values("order_adv_ratio_float", ascending=False).head(20)
    row_html_list = []
    for _, row_ser in top_df.iterrows():
        row_html_list.append(
            "<tr>"
            f"<td>{_fmt_dollar_str(row_ser['capital_base_float'])}</td>"
            f"<td>{html.escape(str(pd.Timestamp(row_ser['bar']).date()))}</td>"
            f"<td>{html.escape(str(row_ser['asset_str']))}</td>"
            f"<td>{html.escape(str(row_ser['side_str']))}</td>"
            f"<td>{_fmt_dollar_str(row_ser['order_notional_float'])}</td>"
            f"<td>{_fmt_dollar_str(row_ser['robust_adv_dollar_lagged_float'])}</td>"
            f"<td>{_fmt_pct_str(row_ser['order_adv_ratio_float'])}</td>"
            f"<td>{_fmt_bps_str(row_ser['central_implicit_cost_bps_float'])}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>AUM</th><th>Date</th><th>Asset</th>'
        '<th>Side</th><th>Order</th><th>Robust ADV</th><th>Order/ADV</th>'
        f'<th>Central cost</th></tr></thead><tbody>{"".join(row_html_list)}</tbody></table></div>'
    )


def _empty_order_diagnostics_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bar",
            "asset_str",
            "side_str",
            "execution_policy_str",
            "impact_profile_str",
            "central_lambda_1pct_adv_bps_float",
            "stress_lambda_1pct_adv_bps_float",
            "model_confidence_str",
            "proxy_bool",
            "model_extrapolation_bool",
            "academic_extrapolation_bool",
            "proxy_extrapolation_bool",
            "order_notional_float",
            "commission_float",
            "adv10_mean_dollar_lagged_float",
            "adv20_median_dollar_lagged_float",
            "robust_adv_dollar_lagged_float",
            "order_adv_ratio_float",
            "central_implicit_cost_bps_float",
            "central_incremental_cost_float",
            "stress_implicit_cost_bps_float",
            "stress_incremental_cost_float",
            "soft_breach_bool",
            "hard_breach_bool",
            "assessed_bool",
            "unavailable_reason_str",
        ]
    )


def _order_notional_float(order_ser: pd.Series) -> float:
    total_value_float = _coerce_float(order_ser.get("total_value"))
    if _is_finite_float(total_value_float):
        return abs(total_value_float)
    amount_float = _coerce_float(order_ser.get("amount"))
    price_float = _coerce_float(order_ser.get("price"))
    if _is_finite_float(amount_float) and _is_finite_float(price_float):
        return abs(amount_float * price_float)
    return np.nan


def _lookup_bar_value_float(value_ser: pd.Series, bar_ts: pd.Timestamp) -> float:
    if bar_ts in value_ser.index:
        return _coerce_float(value_ser.loc[bar_ts])
    return np.nan


def _quantile_float(data_df: pd.DataFrame, column_str: str, quantile_float: float) -> float:
    value_ser = pd.to_numeric(data_df.get(column_str, pd.Series(dtype=float)), errors="coerce").dropna()
    return float(value_ser.quantile(quantile_float)) if len(value_ser) else np.nan


def _max_float(data_df: pd.DataFrame, column_str: str) -> float:
    value_ser = pd.to_numeric(data_df.get(column_str, pd.Series(dtype=float)), errors="coerce").dropna()
    return float(value_ser.max()) if len(value_ser) else np.nan


def _coerce_float(value_obj) -> float:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return np.nan
    return value_float if np.isfinite(value_float) else np.nan


def _is_finite_float(value_obj) -> bool:
    return bool(np.isfinite(_coerce_float(value_obj)))


def _safe_divide_float(numerator_obj, denominator_obj) -> float:
    numerator_float = _coerce_float(numerator_obj)
    denominator_float = _coerce_float(denominator_obj)
    if not _is_finite_float(numerator_float) or not _is_finite_float(denominator_float):
        return np.nan
    if denominator_float == 0.0:
        return np.nan
    return float(numerator_float / denominator_float)


def _fmt_dollar_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    return f"${value_float:,.0f}" if _is_finite_float(value_float) else "N/A"


def _fmt_optional_dollar_str(value_obj) -> str:
    return "N/A" if value_obj is None else _fmt_dollar_str(value_obj)


def _fmt_capacity_str(value_obj, censored_bool: bool) -> str:
    capacity_str = _fmt_optional_dollar_str(value_obj)
    return f"≥ {capacity_str}" if value_obj is not None and censored_bool else capacity_str


def _fmt_compact_dollar_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    if value_float >= 1_000_000.0:
        return f"${value_float / 1_000_000.0:g}M"
    if value_float >= 1_000.0:
        return f"${value_float / 1_000.0:g}K"
    return f"${value_float:g}"


def _fmt_pct_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    return f"{value_float * 100.0:,.2f}%" if _is_finite_float(value_float) else "N/A"


def _fmt_float_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    return f"{value_float:,.2f}" if _is_finite_float(value_float) else "N/A"


def _fmt_bps_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    return f"{value_float:,.2f} bps" if _is_finite_float(value_float) else "N/A"


def _write_json_file(json_path: Path, data_dict: dict[str, object]) -> None:
    json_path.write_text(
        json.dumps(_sanitize_json_obj(data_dict), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sanitize_json_obj(value_obj):
    if isinstance(value_obj, dict):
        return {str(key_obj): _sanitize_json_obj(nested_obj) for key_obj, nested_obj in value_obj.items()}
    if isinstance(value_obj, (list, tuple)):
        return [_sanitize_json_obj(nested_obj) for nested_obj in value_obj]
    if isinstance(value_obj, Path):
        return str(value_obj)
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.isoformat()
    if isinstance(value_obj, (np.integer,)):
        return int(value_obj)
    if isinstance(value_obj, (np.floating, float)):
        value_float = float(value_obj)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value_obj, np.bool_):
        return bool(value_obj)
    return value_obj
