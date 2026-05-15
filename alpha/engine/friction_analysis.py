"""
Auction-liquidity friction diagnostics for completed strategy runs.

V1 is report-only. It reads a completed strategy's transaction ledger and the
original pricing panel, then estimates each order's participation in a nominal
opening/closing auction proxy. It does not change fills, PnL, commissions,
slippage, vanilla reports, or live execution.

Formula:

    dollar_volume_t = close_t * volume_t
    adv20_dollar_lagged_t = median(dollar_volume_{t-20}, ..., dollar_volume_{t-1})
    auction_proxy_dollar_t = adv20_dollar_lagged_t * auction_fraction
    participation_t = abs(order_dollar_t) / auction_proxy_dollar_t
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import html
import json

import numpy as np
import pandas as pd

from alpha.engine.metrics import sharpe_ratio
from alpha.engine.report import build_research_output_path
from alpha.engine.strategy import Strategy


FRICTION_ANALYSIS_TYPE_STR = "friction_analysis"
FRICTION_ORDER_CSV_FILENAME_STR = "friction_orders.csv"
FRICTION_SUMMARY_CSV_FILENAME_STR = "friction_summary.csv"
METADATA_FILENAME_STR = "metadata.json"
RUN_INFO_FILENAME_STR = "run_info.json"
SUMMARY_FILENAME_STR = "summary.json"
REPORT_FILENAME_STR = "report.html"

MOO_AUCTION_FRACTION_FLOAT = 0.02
MOC_AUCTION_FRACTION_FLOAT = 0.10
ADV_LOOKBACK_INT = 20

FINE_BUCKET_STR = "Fine"
WATCH_BUCKET_STR = "Watch"
CAPACITY_SENSITIVE_BUCKET_STR = "Capacity-sensitive"
RED_BUCKET_STR = "Red"
UNAVAILABLE_BUCKET_STR = "Unavailable"

FINE_MAX_PARTICIPATION_FLOAT = 0.01
WATCH_MAX_PARTICIPATION_FLOAT = 0.025
CAPACITY_SENSITIVE_MAX_PARTICIPATION_FLOAT = 0.05

ADV_FINE_BUCKET_STR = "ADV Fine"
ADV_WATCH_BUCKET_STR = "ADV Watch"
ADV_RED_BUCKET_STR = "ADV Red"

ADV_FINE_MAX_PARTICIPATION_FLOAT = 0.05
ADV_WATCH_MAX_PARTICIPATION_FLOAT = 0.10
ADV_FINE_SLIPPAGE_BPS_FLOAT = 1.0
ADV_WATCH_SLIPPAGE_BPS_FLOAT = 5.0
ADV_RED_SLIPPAGE_BPS_FLOAT = 10.0
DEFAULT_SLIPPAGE_BPS_FLOAT = 2.5
MOO_AUCTION_IMPACT_LAMBDA_BPS_FLOAT = 35.0
MOC_AUCTION_IMPACT_LAMBDA_BPS_FLOAT = 15.0
TRADING_DAYS_PER_YEAR_FLOAT = 252.0
AUCTION_IMPACT_MODEL_LABEL_STR = (
    "Estimated adverse auction impact, proxy from public auction-impact research"
)

AUCTION_IMPACT_LOWER_THAN_DEFAULT_LABEL_STR = "Lower than default"
AUCTION_IMPACT_REASONABLE_LABEL_STR = "Within +/-0.5 bps of default"
AUCTION_IMPACT_MILDLY_OPTIMISTIC_LABEL_STR = "Mildly above default"
AUCTION_IMPACT_MATERIALLY_OPTIMISTIC_LABEL_STR = "Materially above default"

COST_CONSERVATIVE_VERDICT_STR = "Conservative"
COST_REASONABLE_VERDICT_STR = "Reasonable"
COST_OPTIMISTIC_VERDICT_STR = "Optimistic"
COST_PROBLEM_VERDICT_STR = "Cost Problem"

AUCTION_PASS_VERDICT_STR = "Pass"
AUCTION_WATCH_VERDICT_STR = "Watch"
AUCTION_STRESSED_VERDICT_STR = "Stressed"
AUCTION_NOT_BELIEVABLE_VERDICT_STR = "Not Believable"

FRICTION_CONSERVATIVE_VERDICT_STR = "Conservative"
FRICTION_REASONABLE_VERDICT_STR = "Reasonable"
FRICTION_WATCH_VERDICT_STR = "Watch"
FRICTION_OPTIMISTIC_VERDICT_STR = "Optimistic"
FRICTION_NOT_BELIEVABLE_VERDICT_STR = "Not Believable"
FRICTION_NO_DATA_VERDICT_STR = "No Data"

MOO_EXECUTION_POLICY_STR = "MOO"
MOC_EXECUTION_POLICY_STR = "MOC"

_MOO_POLICY_ALIAS_SET = {
    "moo",
    "next_open",
    "next_open_moo",
    "opg",
    "open",
}
_MOC_POLICY_ALIAS_SET = {
    "moc",
    "same_close_moc",
    "next_close",
    "next_close_moc",
    "close",
}
_BUCKET_ORDER_LIST = [
    FINE_BUCKET_STR,
    WATCH_BUCKET_STR,
    CAPACITY_SENSITIVE_BUCKET_STR,
    RED_BUCKET_STR,
    UNAVAILABLE_BUCKET_STR,
]


@dataclass
class FrictionAnalysisResult:
    strategy_name_str: str
    friction_order_df: pd.DataFrame
    friction_summary_df: pd.DataFrame
    summary_dict: dict[str, object]
    execution_policy_str: str = MOO_EXECUTION_POLICY_STR
    auction_fraction_float: float = MOO_AUCTION_FRACTION_FLOAT
    adv_lookback_int: int = ADV_LOOKBACK_INT
    default_slippage_bps_float: float = DEFAULT_SLIPPAGE_BPS_FLOAT
    output_dir_path: Path | None = None


def classify_auction_participation_bucket(participation_float: float) -> str:
    if not _is_finite_float(participation_float):
        return UNAVAILABLE_BUCKET_STR
    if participation_float <= FINE_MAX_PARTICIPATION_FLOAT:
        return FINE_BUCKET_STR
    if participation_float <= WATCH_MAX_PARTICIPATION_FLOAT:
        return WATCH_BUCKET_STR
    if participation_float <= CAPACITY_SENSITIVE_MAX_PARTICIPATION_FLOAT:
        return CAPACITY_SENSITIVE_BUCKET_STR
    return RED_BUCKET_STR


def classify_adv_participation_bucket(adv_participation_float: float) -> str:
    if not _is_finite_float(adv_participation_float):
        return UNAVAILABLE_BUCKET_STR
    if adv_participation_float < ADV_FINE_MAX_PARTICIPATION_FLOAT:
        return ADV_FINE_BUCKET_STR
    if adv_participation_float < ADV_WATCH_MAX_PARTICIPATION_FLOAT:
        return ADV_WATCH_BUCKET_STR
    return ADV_RED_BUCKET_STR


def adv_rule_slippage_bps_float(adv_participation_float: float) -> float:
    if not _is_finite_float(adv_participation_float):
        return np.nan
    if adv_participation_float < ADV_FINE_MAX_PARTICIPATION_FLOAT:
        return ADV_FINE_SLIPPAGE_BPS_FLOAT
    if adv_participation_float < ADV_WATCH_MAX_PARTICIPATION_FLOAT:
        return ADV_WATCH_SLIPPAGE_BPS_FLOAT
    return ADV_RED_SLIPPAGE_BPS_FLOAT


def auction_impact_bps_float(
    adv_participation_float: float,
    auction_impact_lambda_bps_float: float,
) -> float:
    if (
        not _is_finite_float(adv_participation_float)
        or not _is_finite_float(auction_impact_lambda_bps_float)
        or adv_participation_float < 0.0
        or auction_impact_lambda_bps_float < 0.0
    ):
        return np.nan
    return float(auction_impact_lambda_bps_float * np.sqrt(adv_participation_float / 0.01))


def auction_impact_lambda_for_policy_float(execution_policy_str: str) -> float:
    normalized_policy_str = normalize_execution_policy_str(execution_policy_str)
    if normalized_policy_str == MOO_EXECUTION_POLICY_STR:
        return MOO_AUCTION_IMPACT_LAMBDA_BPS_FLOAT
    if normalized_policy_str == MOC_EXECUTION_POLICY_STR:
        return MOC_AUCTION_IMPACT_LAMBDA_BPS_FLOAT
    raise ValueError(f"Unsupported execution_policy_str '{execution_policy_str}'.")


def classify_auction_impact_delta_label_str(
    auction_impact_minus_default_bps_float: float,
) -> str:
    delta_bps_float = _coerce_float(auction_impact_minus_default_bps_float)
    if not _is_finite_float(delta_bps_float):
        return FRICTION_NO_DATA_VERDICT_STR
    if delta_bps_float < -0.50:
        return AUCTION_IMPACT_LOWER_THAN_DEFAULT_LABEL_STR
    if delta_bps_float <= 0.50:
        return AUCTION_IMPACT_REASONABLE_LABEL_STR
    if delta_bps_float <= 2.50:
        return AUCTION_IMPACT_MILDLY_OPTIMISTIC_LABEL_STR
    return AUCTION_IMPACT_MATERIALLY_OPTIMISTIC_LABEL_STR


def classify_cost_verdict_str(adv_rule_minus_default_bps_float: float) -> str:
    if not _is_finite_float(adv_rule_minus_default_bps_float):
        return FRICTION_NO_DATA_VERDICT_STR
    if adv_rule_minus_default_bps_float <= -0.50:
        return COST_CONSERVATIVE_VERDICT_STR
    if adv_rule_minus_default_bps_float <= 0.50:
        return COST_REASONABLE_VERDICT_STR
    if adv_rule_minus_default_bps_float <= 2.50:
        return COST_OPTIMISTIC_VERDICT_STR
    return COST_PROBLEM_VERDICT_STR


def classify_auction_verdict_str(
    red_notional_share_float: float,
    capacity_stressed_notional_share_float: float,
    p95_auction_participation_float: float,
) -> str:
    if (
        not _is_finite_float(red_notional_share_float)
        or not _is_finite_float(capacity_stressed_notional_share_float)
        or not _is_finite_float(p95_auction_participation_float)
    ):
        return FRICTION_NO_DATA_VERDICT_STR

    if red_notional_share_float > 0.50 or p95_auction_participation_float > 0.25:
        return AUCTION_NOT_BELIEVABLE_VERDICT_STR
    if (
        red_notional_share_float > 0.20
        or capacity_stressed_notional_share_float > 0.40
        or p95_auction_participation_float > 0.10
    ):
        return AUCTION_STRESSED_VERDICT_STR
    if (
        red_notional_share_float <= 0.05
        and capacity_stressed_notional_share_float <= 0.15
        and p95_auction_participation_float <= 0.05
    ):
        return AUCTION_PASS_VERDICT_STR
    if red_notional_share_float <= 0.20 and capacity_stressed_notional_share_float <= 0.40:
        return AUCTION_WATCH_VERDICT_STR
    return AUCTION_STRESSED_VERDICT_STR


def combine_friction_verdict_str(cost_verdict_str: str, auction_verdict_str: str) -> str:
    if (
        cost_verdict_str == FRICTION_NO_DATA_VERDICT_STR
        or auction_verdict_str == FRICTION_NO_DATA_VERDICT_STR
    ):
        return FRICTION_NO_DATA_VERDICT_STR
    if (
        cost_verdict_str == COST_PROBLEM_VERDICT_STR
        or auction_verdict_str == AUCTION_NOT_BELIEVABLE_VERDICT_STR
    ):
        return FRICTION_NOT_BELIEVABLE_VERDICT_STR
    if (
        auction_verdict_str == AUCTION_STRESSED_VERDICT_STR
        and cost_verdict_str in {COST_CONSERVATIVE_VERDICT_STR, COST_REASONABLE_VERDICT_STR}
    ):
        return FRICTION_WATCH_VERDICT_STR
    if (
        cost_verdict_str == COST_OPTIMISTIC_VERDICT_STR
        or auction_verdict_str == AUCTION_STRESSED_VERDICT_STR
    ):
        return FRICTION_OPTIMISTIC_VERDICT_STR
    if auction_verdict_str == AUCTION_WATCH_VERDICT_STR:
        return FRICTION_WATCH_VERDICT_STR
    if (
        cost_verdict_str == COST_REASONABLE_VERDICT_STR
        and auction_verdict_str == AUCTION_PASS_VERDICT_STR
    ):
        return FRICTION_REASONABLE_VERDICT_STR
    if (
        cost_verdict_str == COST_CONSERVATIVE_VERDICT_STR
        and auction_verdict_str == AUCTION_PASS_VERDICT_STR
    ):
        return FRICTION_CONSERVATIVE_VERDICT_STR
    return FRICTION_WATCH_VERDICT_STR


def save_friction_analysis_results(
    friction_result_obj: FrictionAnalysisResult,
    output_dir_str: str = "results",
) -> Path:
    output_dir_path = build_research_output_path(
        output_dir_str,
        "strategy",
        friction_result_obj.strategy_name_str,
        FRICTION_ANALYSIS_TYPE_STR,
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)

    friction_result_obj.friction_order_df.to_csv(
        output_dir_path / FRICTION_ORDER_CSV_FILENAME_STR,
        index=False,
        date_format="%Y-%m-%d",
    )
    friction_result_obj.friction_summary_df.to_csv(
        output_dir_path / FRICTION_SUMMARY_CSV_FILENAME_STR,
        index=False,
    )
    _write_json_file(
        output_dir_path / SUMMARY_FILENAME_STR,
        friction_result_obj.summary_dict,
    )
    _write_json_file(
        output_dir_path / RUN_INFO_FILENAME_STR,
        _build_run_info_dict(friction_result_obj),
    )
    _write_json_file(
        output_dir_path / METADATA_FILENAME_STR,
        _build_metadata_dict(friction_result_obj),
    )
    (output_dir_path / REPORT_FILENAME_STR).write_text(
        _build_report_html_str(friction_result_obj),
        encoding="utf-8",
    )

    friction_result_obj.output_dir_path = output_dir_path
    return output_dir_path


class FrictionAnalysis:
    """
    Assess completed orders against a nominal MOO/MOC auction-liquidity proxy.

    The analyzer intentionally takes a completed Strategy object. It never
    mutates the strategy, reroutes orders, or changes transaction-cost logic.
    """

    def __init__(
        self,
        strategy_obj: Strategy,
        pricing_data_df: pd.DataFrame,
        execution_policy_str: str = MOO_EXECUTION_POLICY_STR,
        output_dir_str: str = "results",
        save_output_bool: bool = True,
        adv_lookback_int: int = ADV_LOOKBACK_INT,
        default_slippage_bps_float: float = DEFAULT_SLIPPAGE_BPS_FLOAT,
    ):
        self.strategy_obj = strategy_obj
        self.pricing_data_df = pricing_data_df.sort_index().copy()
        self.execution_policy_str = normalize_execution_policy_str(execution_policy_str)
        self.auction_fraction_float = auction_fraction_for_policy_float(
            self.execution_policy_str
        )
        self.output_dir_str = str(output_dir_str)
        self.save_output_bool = bool(save_output_bool)
        self.adv_lookback_int = int(adv_lookback_int)
        self.default_slippage_bps_float = float(default_slippage_bps_float)
        if self.adv_lookback_int <= 0:
            raise ValueError("adv_lookback_int must be positive.")
        if self.default_slippage_bps_float < 0.0:
            raise ValueError("default_slippage_bps_float must be non-negative.")

    def run(self) -> FrictionAnalysisResult:
        transaction_df = _get_completed_transaction_df(self.strategy_obj)
        _validate_transaction_columns(transaction_df)

        if len(transaction_df) == 0:
            friction_order_df = _empty_friction_order_df()
            friction_summary_df = _build_bucket_summary_df(friction_order_df)
            summary_dict = _build_summary_dict(
                friction_order_df,
                self.execution_policy_str,
                self.auction_fraction_float,
                self.adv_lookback_int,
                strategy_name_str=self.strategy_obj.name,
                default_slippage_bps_float=self.default_slippage_bps_float,
                strategy_obj=self.strategy_obj,
            )
            friction_result_obj = FrictionAnalysisResult(
                strategy_name_str=self.strategy_obj.name,
                friction_order_df=friction_order_df,
                friction_summary_df=friction_summary_df,
                summary_dict=summary_dict,
                execution_policy_str=self.execution_policy_str,
                auction_fraction_float=self.auction_fraction_float,
                adv_lookback_int=self.adv_lookback_int,
                default_slippage_bps_float=self.default_slippage_bps_float,
            )
            if self.save_output_bool:
                save_friction_analysis_results(friction_result_obj, self.output_dir_str)
            return friction_result_obj

        transaction_df = transaction_df.copy()
        transaction_df["bar"] = pd.to_datetime(transaction_df["bar"])
        _validate_pricing_data_columns(self.pricing_data_df, transaction_df)

        adv20_dollar_lagged_map = _build_lagged_adv_map(
            self.pricing_data_df,
            transaction_df["asset"].astype(str).unique().tolist(),
            self.adv_lookback_int,
        )
        friction_order_df = _build_friction_order_df(
            transaction_df,
            adv20_dollar_lagged_map,
            self.execution_policy_str,
            self.auction_fraction_float,
            self.default_slippage_bps_float,
        )
        friction_summary_df = _build_bucket_summary_df(friction_order_df)
        summary_dict = _build_summary_dict(
            friction_order_df,
            self.execution_policy_str,
            self.auction_fraction_float,
            self.adv_lookback_int,
            strategy_name_str=self.strategy_obj.name,
            default_slippage_bps_float=self.default_slippage_bps_float,
            strategy_obj=self.strategy_obj,
        )
        friction_result_obj = FrictionAnalysisResult(
            strategy_name_str=self.strategy_obj.name,
            friction_order_df=friction_order_df,
            friction_summary_df=friction_summary_df,
            summary_dict=summary_dict,
            execution_policy_str=self.execution_policy_str,
            auction_fraction_float=self.auction_fraction_float,
            adv_lookback_int=self.adv_lookback_int,
            default_slippage_bps_float=self.default_slippage_bps_float,
        )

        if self.save_output_bool:
            save_friction_analysis_results(friction_result_obj, self.output_dir_str)

        return friction_result_obj


def normalize_execution_policy_str(execution_policy_str: str) -> str:
    normalized_policy_str = str(execution_policy_str).strip().lower()
    if normalized_policy_str in _MOO_POLICY_ALIAS_SET:
        return MOO_EXECUTION_POLICY_STR
    if normalized_policy_str in _MOC_POLICY_ALIAS_SET:
        return MOC_EXECUTION_POLICY_STR
    raise ValueError(
        f"Unsupported execution_policy_str '{execution_policy_str}'. "
        f"Use one of: {MOO_EXECUTION_POLICY_STR}, {MOC_EXECUTION_POLICY_STR}."
    )


def auction_fraction_for_policy_float(execution_policy_str: str) -> float:
    normalized_policy_str = normalize_execution_policy_str(execution_policy_str)
    if normalized_policy_str == MOO_EXECUTION_POLICY_STR:
        return MOO_AUCTION_FRACTION_FLOAT
    if normalized_policy_str == MOC_EXECUTION_POLICY_STR:
        return MOC_AUCTION_FRACTION_FLOAT
    raise ValueError(f"Unsupported execution_policy_str '{execution_policy_str}'.")


def _get_completed_transaction_df(strategy_obj: Strategy) -> pd.DataFrame:
    if not hasattr(strategy_obj, "get_transactions"):
        raise TypeError("strategy_obj must expose get_transactions().")
    transaction_df = strategy_obj.get_transactions()
    if transaction_df is None:
        return Strategy.initialize_transactions()
    return pd.DataFrame(transaction_df).copy()


def _validate_transaction_columns(transaction_df: pd.DataFrame) -> None:
    required_column_set = {
        "bar",
        "asset",
        "amount",
        "price",
        "total_value",
        "trade_id",
        "order_id",
        "commission",
    }
    missing_column_list = sorted(required_column_set.difference(transaction_df.columns))
    if missing_column_list:
        raise ValueError(
            "transactions_df is missing required columns: "
            f"{missing_column_list}."
        )


def _validate_pricing_data_columns(
    pricing_data_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
) -> None:
    if not isinstance(pricing_data_df.columns, pd.MultiIndex):
        raise ValueError("pricing_data_df must have MultiIndex columns.")

    missing_close_asset_list: list[str] = []
    missing_volume_asset_list: list[str] = []
    traded_asset_list = sorted(transaction_df["asset"].astype(str).unique().tolist())
    for asset_str in traded_asset_list:
        if (asset_str, "Close") not in pricing_data_df.columns:
            missing_close_asset_list.append(asset_str)
        if (asset_str, "Volume") not in pricing_data_df.columns:
            missing_volume_asset_list.append(asset_str)

    error_part_list: list[str] = []
    if missing_close_asset_list:
        error_part_list.append(f"Close: {missing_close_asset_list}")
    if missing_volume_asset_list:
        error_part_list.append(f"Volume: {missing_volume_asset_list}")
    if error_part_list:
        raise ValueError(
            "pricing_data_df is missing required auction-liquidity columns for "
            + "; ".join(error_part_list)
        )


def _build_lagged_adv_map(
    pricing_data_df: pd.DataFrame,
    traded_asset_list: list[str],
    adv_lookback_int: int,
) -> dict[str, pd.Series]:
    adv20_dollar_lagged_map: dict[str, pd.Series] = {}
    for asset_str in traded_asset_list:
        close_ser = pricing_data_df[(asset_str, "Close")].astype(float)
        volume_ser = pricing_data_df[(asset_str, "Volume")].astype(float)
        # *** CRITICAL *** lookahead-sensitive: MOO/MOC auction capacity at T may
        # only use volume known through T-1. Same-day final volume is not known
        # before the auction decision/cutoff.
        dollar_volume_ser = close_ser * volume_ser
        adv20_dollar_lagged_ser = (
            dollar_volume_ser.shift(1)
            .rolling(adv_lookback_int, min_periods=adv_lookback_int)
            .median()
        )
        adv20_dollar_lagged_map[asset_str] = adv20_dollar_lagged_ser
    return adv20_dollar_lagged_map


def _build_friction_order_df(
    transaction_df: pd.DataFrame,
    adv20_dollar_lagged_map: dict[str, pd.Series],
    execution_policy_str: str,
    auction_fraction_float: float,
    default_slippage_bps_float: float,
) -> pd.DataFrame:
    friction_row_dict_list: list[dict[str, object]] = []
    auction_impact_lambda_bps_float = auction_impact_lambda_for_policy_float(
        execution_policy_str
    )

    for _, transaction_ser in transaction_df.iterrows():
        bar_ts = pd.Timestamp(transaction_ser["bar"])
        asset_str = str(transaction_ser["asset"])
        amount_float = _coerce_float(transaction_ser["amount"])
        price_float = _coerce_float(transaction_ser["price"])
        signed_total_value_float = _coerce_float(transaction_ser["total_value"])
        order_dollar_float = _order_dollar_float(
            amount_float,
            price_float,
            signed_total_value_float,
        )
        adv20_dollar_lagged_float = _lookup_bar_value_float(
            adv20_dollar_lagged_map[asset_str],
            bar_ts,
        )

        unavailable_reason_str = ""
        if not _is_finite_float(order_dollar_float):
            unavailable_reason_str = "missing_order_notional"
        elif not _is_finite_float(adv20_dollar_lagged_float):
            unavailable_reason_str = "insufficient_lagged_adv_history"
        elif adv20_dollar_lagged_float <= 0.0:
            unavailable_reason_str = "non_positive_lagged_adv"

        if unavailable_reason_str:
            auction_proxy_dollar_float = np.nan
            auction_participation_float = np.nan
            adv_participation_float = np.nan
            adv_bucket_str = UNAVAILABLE_BUCKET_STR
            adv_rule_slippage_bps_value_float = np.nan
            adv_rule_slippage_dollar_float = np.nan
            default_slippage_dollar_float = np.nan
            adv_rule_minus_default_bps_float = np.nan
            adv_rule_minus_default_dollar_float = np.nan
            auction_impact_bps_value_float = np.nan
            auction_impact_dollar_float = np.nan
            auction_impact_minus_default_bps_float = np.nan
            auction_impact_minus_default_dollar_float = np.nan
            capacity_bucket_str = UNAVAILABLE_BUCKET_STR
            assessed_bool = False
        else:
            auction_proxy_dollar_float = (
                adv20_dollar_lagged_float * auction_fraction_float
            )
            if auction_proxy_dollar_float <= 0.0:
                auction_participation_float = np.nan
                adv_participation_float = np.nan
                adv_bucket_str = UNAVAILABLE_BUCKET_STR
                adv_rule_slippage_bps_value_float = np.nan
                adv_rule_slippage_dollar_float = np.nan
                default_slippage_dollar_float = np.nan
                adv_rule_minus_default_bps_float = np.nan
                adv_rule_minus_default_dollar_float = np.nan
                auction_impact_bps_value_float = np.nan
                auction_impact_dollar_float = np.nan
                auction_impact_minus_default_bps_float = np.nan
                auction_impact_minus_default_dollar_float = np.nan
                capacity_bucket_str = UNAVAILABLE_BUCKET_STR
                assessed_bool = False
                unavailable_reason_str = "non_positive_auction_proxy"
            else:
                auction_participation_float = (
                    order_dollar_float / auction_proxy_dollar_float
                )
                adv_participation_float = (
                    order_dollar_float / adv20_dollar_lagged_float
                )
                adv_bucket_str = classify_adv_participation_bucket(
                    adv_participation_float
                )
                adv_rule_slippage_bps_value_float = adv_rule_slippage_bps_float(
                    adv_participation_float
                )
                adv_rule_slippage_dollar_float = (
                    order_dollar_float
                    * adv_rule_slippage_bps_value_float
                    * 0.0001
                )
                default_slippage_dollar_float = (
                    order_dollar_float * default_slippage_bps_float * 0.0001
                )
                adv_rule_minus_default_bps_float = (
                    adv_rule_slippage_bps_value_float - default_slippage_bps_float
                )
                adv_rule_minus_default_dollar_float = (
                    adv_rule_slippage_dollar_float - default_slippage_dollar_float
                )
                auction_impact_bps_value_float = auction_impact_bps_float(
                    adv_participation_float,
                    auction_impact_lambda_bps_float,
                )
                auction_impact_dollar_float = (
                    order_dollar_float * auction_impact_bps_value_float * 0.0001
                )
                auction_impact_minus_default_bps_float = (
                    auction_impact_bps_value_float - default_slippage_bps_float
                )
                auction_impact_minus_default_dollar_float = (
                    auction_impact_dollar_float - default_slippage_dollar_float
                )
                capacity_bucket_str = classify_auction_participation_bucket(
                    auction_participation_float
                )
                assessed_bool = True

        friction_row_dict_list.append(
            {
                "bar": bar_ts.normalize(),
                "year_int": int(bar_ts.year),
                "month_str": bar_ts.strftime("%Y-%m"),
                "asset": asset_str,
                "side": _side_str(amount_float),
                "execution_policy_str": execution_policy_str,
                "trade_id": transaction_ser.get("trade_id"),
                "order_id": transaction_ser.get("order_id"),
                "amount_float": amount_float,
                "price_float": price_float,
                "signed_total_value_float": signed_total_value_float,
                "order_dollar_float": order_dollar_float,
                "commission_float": _coerce_float(transaction_ser.get("commission")),
                "adv20_dollar_lagged_float": adv20_dollar_lagged_float,
                "auction_fraction_float": auction_fraction_float,
                "auction_proxy_dollar_float": auction_proxy_dollar_float,
                "auction_participation_float": auction_participation_float,
                "auction_participation_pct_float": (
                    auction_participation_float * 100.0
                    if _is_finite_float(auction_participation_float)
                    else np.nan
                ),
                "adv_participation_float": adv_participation_float,
                "adv_participation_pct_float": (
                    adv_participation_float * 100.0
                    if _is_finite_float(adv_participation_float)
                    else np.nan
                ),
                "adv_bucket_str": adv_bucket_str,
                "adv_rule_slippage_bps_float": adv_rule_slippage_bps_value_float,
                "adv_rule_slippage_dollar_float": adv_rule_slippage_dollar_float,
                "default_slippage_bps_float": default_slippage_bps_float,
                "default_slippage_dollar_float": default_slippage_dollar_float,
                "adv_rule_minus_default_bps_float": adv_rule_minus_default_bps_float,
                "adv_rule_minus_default_dollar_float": adv_rule_minus_default_dollar_float,
                "auction_impact_lambda_bps_float": auction_impact_lambda_bps_float,
                "auction_impact_bps_float": auction_impact_bps_value_float,
                "auction_impact_dollar_float": auction_impact_dollar_float,
                "auction_impact_minus_default_bps_float": auction_impact_minus_default_bps_float,
                "auction_impact_minus_default_dollar_float": auction_impact_minus_default_dollar_float,
                "capacity_bucket_str": capacity_bucket_str,
                "assessed_bool": bool(assessed_bool),
                "unavailable_reason_str": unavailable_reason_str,
            }
        )

    friction_order_df = pd.DataFrame(friction_row_dict_list)
    if len(friction_order_df) == 0:
        return _empty_friction_order_df()
    return friction_order_df


def _empty_friction_order_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bar",
            "year_int",
            "month_str",
            "asset",
            "side",
            "execution_policy_str",
            "trade_id",
            "order_id",
            "amount_float",
            "price_float",
            "signed_total_value_float",
            "order_dollar_float",
            "commission_float",
            "adv20_dollar_lagged_float",
            "auction_fraction_float",
            "auction_proxy_dollar_float",
            "auction_participation_float",
            "auction_participation_pct_float",
            "adv_participation_float",
            "adv_participation_pct_float",
            "adv_bucket_str",
            "adv_rule_slippage_bps_float",
            "adv_rule_slippage_dollar_float",
            "default_slippage_bps_float",
            "default_slippage_dollar_float",
            "adv_rule_minus_default_bps_float",
            "adv_rule_minus_default_dollar_float",
            "auction_impact_lambda_bps_float",
            "auction_impact_bps_float",
            "auction_impact_dollar_float",
            "auction_impact_minus_default_bps_float",
            "auction_impact_minus_default_dollar_float",
            "capacity_bucket_str",
            "assessed_bool",
            "unavailable_reason_str",
        ]
    )


def _build_bucket_summary_df(friction_order_df: pd.DataFrame) -> pd.DataFrame:
    if len(friction_order_df) == 0:
        total_order_count_int = 0
        total_notional_float = 0.0
    else:
        total_order_count_int = int(len(friction_order_df))
        total_notional_float = float(
            friction_order_df["order_dollar_float"].fillna(0.0).sum()
        )

    summary_row_dict_list: list[dict[str, object]] = []
    for bucket_str in _BUCKET_ORDER_LIST:
        bucket_order_df = friction_order_df[
            friction_order_df.get("capacity_bucket_str", pd.Series(dtype=str))
            == bucket_str
        ]
        bucket_order_count_int = int(len(bucket_order_df))
        bucket_notional_float = float(
            bucket_order_df.get("order_dollar_float", pd.Series(dtype=float))
            .fillna(0.0)
            .sum()
        )
        summary_row_dict_list.append(
            {
                "capacity_bucket_str": bucket_str,
                "order_count_int": bucket_order_count_int,
                "order_share_float": _safe_divide_float(
                    bucket_order_count_int,
                    total_order_count_int,
                ),
                "notional_float": bucket_notional_float,
                "notional_share_float": _safe_divide_float(
                    bucket_notional_float,
                    total_notional_float,
                ),
                "assessed_bucket_bool": bucket_str != UNAVAILABLE_BUCKET_STR,
            }
        )
    return pd.DataFrame(summary_row_dict_list)


def _build_summary_dict(
    friction_order_df: pd.DataFrame,
    execution_policy_str: str,
    auction_fraction_float: float,
    adv_lookback_int: int,
    strategy_name_str: str,
    default_slippage_bps_float: float,
    strategy_obj: Strategy | None = None,
) -> dict[str, object]:
    total_order_count_int = int(len(friction_order_df))
    total_notional_float = float(
        friction_order_df.get("order_dollar_float", pd.Series(dtype=float))
        .fillna(0.0)
        .sum()
    )
    assessed_order_df = friction_order_df[
        friction_order_df.get("assessed_bool", pd.Series(dtype=bool)) == True
    ]
    assessed_order_count_int = int(len(assessed_order_df))
    assessed_notional_float = float(
        assessed_order_df.get("order_dollar_float", pd.Series(dtype=float))
        .fillna(0.0)
        .sum()
    )
    unavailable_order_count_int = total_order_count_int - assessed_order_count_int
    participation_ser = assessed_order_df.get(
        "auction_participation_float",
        pd.Series(dtype=float),
    ).astype(float)
    adv_participation_ser = assessed_order_df.get(
        "adv_participation_float",
        pd.Series(dtype=float),
    ).astype(float)
    auction_impact_bps_ser = assessed_order_df.get(
        "auction_impact_bps_float",
        pd.Series(dtype=float),
    ).astype(float)
    red_order_df = assessed_order_df[
        assessed_order_df.get("capacity_bucket_str", pd.Series(dtype=str))
        == RED_BUCKET_STR
    ]
    stressed_order_df = assessed_order_df[
        assessed_order_df.get("capacity_bucket_str", pd.Series(dtype=str)).isin(
            [CAPACITY_SENSITIVE_BUCKET_STR, RED_BUCKET_STR]
        )
    ]
    pass_order_df = assessed_order_df[
        assessed_order_df.get("capacity_bucket_str", pd.Series(dtype=str)).isin(
            [FINE_BUCKET_STR, WATCH_BUCKET_STR]
        )
    ]
    adv_red_order_df = assessed_order_df[
        assessed_order_df.get("adv_bucket_str", pd.Series(dtype=str))
        == ADV_RED_BUCKET_STR
    ]
    red_notional_float = float(
        red_order_df.get("order_dollar_float", pd.Series(dtype=float)).fillna(0.0).sum()
    )
    stressed_notional_float = float(
        stressed_order_df.get("order_dollar_float", pd.Series(dtype=float))
        .fillna(0.0)
        .sum()
    )
    pass_notional_float = float(
        pass_order_df.get("order_dollar_float", pd.Series(dtype=float)).fillna(0.0).sum()
    )
    adv_red_notional_float = float(
        adv_red_order_df.get("order_dollar_float", pd.Series(dtype=float))
        .fillna(0.0)
        .sum()
    )
    adv_rule_slippage_dollar_float = float(
        assessed_order_df.get("adv_rule_slippage_dollar_float", pd.Series(dtype=float))
        .fillna(0.0)
        .sum()
    )
    default_slippage_dollar_float = float(
        assessed_order_df.get("default_slippage_dollar_float", pd.Series(dtype=float))
        .fillna(0.0)
        .sum()
    )
    adv_rule_minus_default_dollar_float = (
        adv_rule_slippage_dollar_float - default_slippage_dollar_float
    )
    auction_impact_dollar_float = float(
        assessed_order_df.get("auction_impact_dollar_float", pd.Series(dtype=float))
        .fillna(0.0)
        .sum()
    )
    auction_impact_minus_default_dollar_float = (
        auction_impact_dollar_float - default_slippage_dollar_float
    )
    auction_impact_minus_default_bps_float = _notional_to_bps_float(
        auction_impact_minus_default_dollar_float,
        assessed_notional_float,
    )
    auction_impact_delta_label_str = classify_auction_impact_delta_label_str(
        auction_impact_minus_default_bps_float
    )
    performance_impact_dict = _build_performance_impact_summary_dict(
        strategy_obj,
        assessed_order_df,
    )
    summary_dict = {
        "strategy_name_str": strategy_name_str,
        "analysis_type_str": FRICTION_ANALYSIS_TYPE_STR,
        "execution_policy_str": execution_policy_str,
        "auction_fraction_float": auction_fraction_float,
        "auction_impact_lambda_bps_float": auction_impact_lambda_for_policy_float(
            execution_policy_str
        ),
        "auction_impact_model_label_str": AUCTION_IMPACT_MODEL_LABEL_STR,
        "adv_lookback_int": int(adv_lookback_int),
        "total_order_count_int": total_order_count_int,
        "total_notional_float": total_notional_float,
        "assessed_order_count_int": assessed_order_count_int,
        "assessed_notional_float": assessed_notional_float,
        "unavailable_order_count_int": unavailable_order_count_int,
        "p50_participation_float": _series_quantile_float(participation_ser, 0.50),
        "p95_participation_float": _series_quantile_float(participation_ser, 0.95),
        "max_participation_float": _series_max_float(participation_ser),
        "red_order_count_int": int(len(red_order_df)),
        "red_order_share_float": _safe_divide_float(
            len(red_order_df),
            assessed_order_count_int,
        ),
        "red_notional_float": red_notional_float,
        "red_notional_share_float": _safe_divide_float(
            red_notional_float,
            assessed_notional_float,
        ),
        "capacity_stressed_order_count_int": int(len(stressed_order_df)),
        "capacity_stressed_order_share_float": _safe_divide_float(
            len(stressed_order_df),
            assessed_order_count_int,
        ),
        "capacity_stressed_notional_float": stressed_notional_float,
        "capacity_stressed_notional_share_float": _safe_divide_float(
            stressed_notional_float,
            assessed_notional_float,
        ),
        "capacity_pass_order_rate_float": _safe_divide_float(
            len(pass_order_df),
            assessed_order_count_int,
        ),
        "capacity_pass_notional_rate_float": _safe_divide_float(
            pass_notional_float,
            assessed_notional_float,
        ),
        "adv_p50_participation_float": _series_quantile_float(adv_participation_ser, 0.50),
        "adv_p95_participation_float": _series_quantile_float(adv_participation_ser, 0.95),
        "adv_max_participation_float": _series_max_float(adv_participation_ser),
        "adv_red_order_count_int": int(len(adv_red_order_df)),
        "adv_red_order_share_float": _safe_divide_float(
            len(adv_red_order_df),
            assessed_order_count_int,
        ),
        "adv_red_notional_float": adv_red_notional_float,
        "adv_red_notional_share_float": _safe_divide_float(
            adv_red_notional_float,
            assessed_notional_float,
        ),
        "adv_rule_slippage_dollar_float": adv_rule_slippage_dollar_float,
        "adv_rule_slippage_blended_bps_float": _notional_to_bps_float(
            adv_rule_slippage_dollar_float,
            assessed_notional_float,
        ),
        "default_slippage_bps_float": default_slippage_bps_float,
        "default_slippage_dollar_float": default_slippage_dollar_float,
        "default_slippage_blended_bps_float": _notional_to_bps_float(
            default_slippage_dollar_float,
            assessed_notional_float,
        ),
        "adv_rule_minus_default_dollar_float": adv_rule_minus_default_dollar_float,
        "adv_rule_minus_default_bps_float": _notional_to_bps_float(
            adv_rule_minus_default_dollar_float,
            assessed_notional_float,
        ),
        "auction_impact_p50_bps_float": _series_quantile_float(
            auction_impact_bps_ser,
            0.50,
        ),
        "auction_impact_p95_bps_float": _series_quantile_float(
            auction_impact_bps_ser,
            0.95,
        ),
        "auction_impact_max_bps_float": _series_max_float(auction_impact_bps_ser),
        "auction_impact_dollar_float": auction_impact_dollar_float,
        "auction_impact_blended_bps_float": _notional_to_bps_float(
            auction_impact_dollar_float,
            assessed_notional_float,
        ),
        "auction_impact_minus_default_dollar_float": auction_impact_minus_default_dollar_float,
        "auction_impact_minus_default_bps_float": auction_impact_minus_default_bps_float,
        "auction_impact_delta_label_str": auction_impact_delta_label_str,
        "auction_impact_interpretation_str": _build_auction_impact_interpretation_str(
            default_slippage_bps_float,
            auction_impact_delta_label_str,
        ),
        **performance_impact_dict,
        "bucket_thresholds_dict": {
            FINE_BUCKET_STR: f"participation <= {FINE_MAX_PARTICIPATION_FLOAT:.3f}",
            WATCH_BUCKET_STR: (
                f"{FINE_MAX_PARTICIPATION_FLOAT:.3f} < participation <= "
                f"{WATCH_MAX_PARTICIPATION_FLOAT:.3f}"
            ),
            CAPACITY_SENSITIVE_BUCKET_STR: (
                f"{WATCH_MAX_PARTICIPATION_FLOAT:.3f} < participation <= "
                f"{CAPACITY_SENSITIVE_MAX_PARTICIPATION_FLOAT:.3f}"
            ),
            RED_BUCKET_STR: f"participation > {CAPACITY_SENSITIVE_MAX_PARTICIPATION_FLOAT:.3f}",
        },
        "adv_rule_thresholds_dict": {
            ADV_FINE_BUCKET_STR: (
                f"ADV participation < {ADV_FINE_MAX_PARTICIPATION_FLOAT:.3f} "
                f"=> {ADV_FINE_SLIPPAGE_BPS_FLOAT:.1f} bps"
            ),
            ADV_WATCH_BUCKET_STR: (
                f"{ADV_FINE_MAX_PARTICIPATION_FLOAT:.3f} <= ADV participation "
                f"< {ADV_WATCH_MAX_PARTICIPATION_FLOAT:.3f} "
                f"=> {ADV_WATCH_SLIPPAGE_BPS_FLOAT:.1f} bps"
            ),
            ADV_RED_BUCKET_STR: (
                f"ADV participation >= {ADV_WATCH_MAX_PARTICIPATION_FLOAT:.3f} "
                f"=> {ADV_RED_SLIPPAGE_BPS_FLOAT:.1f} bps"
            ),
        },
    }
    summary_dict.update(_build_decision_summary_dict(summary_dict, assessed_order_df))
    return summary_dict


def _build_performance_impact_summary_dict(
    strategy_obj: Strategy | None,
    assessed_order_df: pd.DataFrame,
) -> dict[str, object]:
    empty_performance_dict = {
        "current_final_value_float": np.nan,
        "auction_impact_estimated_final_value_float": np.nan,
        "auction_impact_estimated_cumulative_extra_drag_float": np.nan,
        "current_annual_return_float": np.nan,
        "auction_impact_estimated_annual_return_float": np.nan,
        "auction_impact_estimated_annual_return_delta_float": np.nan,
        "current_sharpe_float": np.nan,
        "auction_impact_estimated_sharpe_float": np.nan,
        "auction_impact_estimated_sharpe_delta_float": np.nan,
        "auction_impact_performance_note_str": (
            "Performance estimate unavailable because the strategy result equity curve "
            "was not available."
        ),
    }
    if strategy_obj is None or not hasattr(strategy_obj, "results"):
        return empty_performance_dict

    result_df = getattr(strategy_obj, "results")
    if not isinstance(result_df, pd.DataFrame) or "total_value" not in result_df.columns:
        return empty_performance_dict

    current_total_value_ser = (
        result_df["total_value"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(current_total_value_ser) < 2:
        return empty_performance_dict

    current_total_value_ser.index = pd.to_datetime(current_total_value_ser.index).normalize()
    current_total_value_ser = current_total_value_ser.groupby(current_total_value_ser.index).last()
    portfolio_value_ser = _strategy_portfolio_value_ser(strategy_obj, current_total_value_ser.index)

    extra_drag_by_bar_ser = pd.Series(dtype=float)
    if (
        len(assessed_order_df) > 0
        and "bar" in assessed_order_df.columns
        and "auction_impact_minus_default_dollar_float" in assessed_order_df.columns
    ):
        extra_drag_df = assessed_order_df[
            ["bar", "auction_impact_minus_default_dollar_float"]
        ].copy()
        extra_drag_df["bar"] = pd.to_datetime(extra_drag_df["bar"]).dt.normalize()
        extra_drag_df["auction_impact_minus_default_dollar_float"] = (
            extra_drag_df["auction_impact_minus_default_dollar_float"]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        extra_drag_by_bar_ser = extra_drag_df.groupby("bar")[
            "auction_impact_minus_default_dollar_float"
        ].sum()

    daily_extra_drag_ser = extra_drag_by_bar_ser.reindex(
        current_total_value_ser.index,
        fill_value=0.0,
    )
    cumulative_extra_drag_ser = daily_extra_drag_ser.cumsum()

    # *** CRITICAL *** This is a report-only diagnostic overlay. It subtracts
    # the signed auction-impact delta from the completed equity curve; it does
    # not rerun sizing, rebalance weights, fills, or transaction semantics.
    auction_impact_estimated_total_value_ser = (
        current_total_value_ser - cumulative_extra_drag_ser
    )

    current_annual_return_float = _annualized_return_from_total_value_float(
        current_total_value_ser
    )
    estimated_annual_return_float = _annualized_return_from_total_value_float(
        auction_impact_estimated_total_value_ser
    )
    current_sharpe_float = _sharpe_from_total_value_float(
        current_total_value_ser,
        portfolio_value_ser,
    )
    estimated_sharpe_float = _sharpe_from_total_value_float(
        auction_impact_estimated_total_value_ser,
        portfolio_value_ser,
    )

    return {
        "current_final_value_float": float(current_total_value_ser.iloc[-1]),
        "auction_impact_estimated_final_value_float": float(
            auction_impact_estimated_total_value_ser.iloc[-1]
        ),
        "auction_impact_estimated_cumulative_extra_drag_float": float(
            cumulative_extra_drag_ser.iloc[-1]
        ),
        "current_annual_return_float": current_annual_return_float,
        "auction_impact_estimated_annual_return_float": estimated_annual_return_float,
        "auction_impact_estimated_annual_return_delta_float": (
            estimated_annual_return_float - current_annual_return_float
        ),
        "current_sharpe_float": current_sharpe_float,
        "auction_impact_estimated_sharpe_float": estimated_sharpe_float,
        "auction_impact_estimated_sharpe_delta_float": (
            estimated_sharpe_float - current_sharpe_float
        ),
        "auction_impact_performance_note_str": (
            "Estimated from completed order drag only: current equity minus cumulative "
            "auction-impact delta vs default. No rescale, no retrade, no fill changes."
        ),
    }


def _annualized_return_from_total_value_float(total_value_ser: pd.Series) -> float:
    clean_total_value_ser = (
        pd.Series(total_value_ser, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(clean_total_value_ser) < 2:
        return np.nan
    start_value_float = float(clean_total_value_ser.iloc[0])
    end_value_float = float(clean_total_value_ser.iloc[-1])
    if start_value_float <= 0.0 or end_value_float <= 0.0:
        return np.nan
    return float(
        (end_value_float / start_value_float)
        ** (TRADING_DAYS_PER_YEAR_FLOAT / len(clean_total_value_ser))
        - 1.0
    )


def _strategy_portfolio_value_ser(
    strategy_obj: Strategy,
    target_index: pd.Index,
) -> pd.Series | None:
    result_df = getattr(strategy_obj, "results", None)
    if not isinstance(result_df, pd.DataFrame) or "portfolio_value" not in result_df.columns:
        return None
    portfolio_value_ser = (
        result_df["portfolio_value"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(portfolio_value_ser) == 0:
        return None
    portfolio_value_ser.index = pd.to_datetime(portfolio_value_ser.index).normalize()
    portfolio_value_ser = portfolio_value_ser.groupby(portfolio_value_ser.index).last()
    return portfolio_value_ser.reindex(target_index)


def _sharpe_from_total_value_float(
    total_value_ser: pd.Series,
    portfolio_value_ser: pd.Series | None = None,
) -> float:
    clean_total_value_ser = (
        pd.Series(total_value_ser, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(clean_total_value_ser) < 3:
        return np.nan

    # *** CRITICAL *** This diagnostic Sharpe uses only realized/estimated
    # equity history after the completed backtest. It uses the same repo
    # Sharpe helper as the vanilla report so active/cash periods are filtered
    # the same way; it must not feed any signal or order decision path.
    daily_return_ser = clean_total_value_ser.pct_change(fill_method=None).dropna()
    aligned_portfolio_value_ser = None
    if portfolio_value_ser is not None:
        aligned_portfolio_value_ser = portfolio_value_ser.reindex(daily_return_ser.index)
    if len(daily_return_ser) < 2:
        return np.nan
    return float(
        sharpe_ratio(
            daily_return_ser,
            aligned_portfolio_value_ser,
            days_in_year=int(TRADING_DAYS_PER_YEAR_FLOAT),
        )
    )


def _build_auction_impact_interpretation_str(
    default_slippage_bps_float: float,
    auction_impact_delta_label_str: str,
) -> str:
    default_bps_str = f"{float(default_slippage_bps_float):.1f} bps"
    if auction_impact_delta_label_str == AUCTION_IMPACT_LOWER_THAN_DEFAULT_LABEL_STR:
        return (
            f"This suggests current {default_bps_str} is likely conservative "
            "versus the auction-impact proxy."
        )
    if auction_impact_delta_label_str == AUCTION_IMPACT_REASONABLE_LABEL_STR:
        return (
            f"This suggests current {default_bps_str} is roughly reasonable "
            "versus the auction-impact proxy."
        )
    if auction_impact_delta_label_str == AUCTION_IMPACT_MILDLY_OPTIMISTIC_LABEL_STR:
        return (
            f"This suggests current {default_bps_str} is mildly optimistic "
            "versus the auction-impact proxy."
        )
    if auction_impact_delta_label_str == AUCTION_IMPACT_MATERIALLY_OPTIMISTIC_LABEL_STR:
        return (
            f"This suggests current {default_bps_str} is materially optimistic "
            "versus the auction-impact proxy."
        )
    return "Not enough assessed orders to compare current default cost to auction impact."


def _build_decision_summary_dict(
    summary_dict: dict[str, object],
    assessed_order_df: pd.DataFrame,
) -> dict[str, object]:
    cost_verdict_str = classify_cost_verdict_str(
        summary_dict.get("adv_rule_minus_default_bps_float")
    )
    auction_verdict_str = classify_auction_verdict_str(
        summary_dict.get("red_notional_share_float"),
        summary_dict.get("capacity_stressed_notional_share_float"),
        summary_dict.get("p95_participation_float"),
    )
    friction_verdict_str = combine_friction_verdict_str(
        cost_verdict_str,
        auction_verdict_str,
    )
    concentration_dict = _build_concentration_summary_dict(assessed_order_df)
    scale_limit_dict = _build_scale_limit_summary_dict(summary_dict)
    verdict_explanation_str = _build_verdict_explanation_str(
        summary_dict,
        cost_verdict_str,
        auction_verdict_str,
        friction_verdict_str,
    )

    return {
        "cost_verdict_str": cost_verdict_str,
        "auction_verdict_str": auction_verdict_str,
        "friction_verdict_str": friction_verdict_str,
        "verdict_explanation_str": verdict_explanation_str,
        **concentration_dict,
        **scale_limit_dict,
    }


def _build_concentration_summary_dict(assessed_order_df: pd.DataFrame) -> dict[str, object]:
    empty_concentration_dict = {
        "top_red_asset_str": None,
        "top_red_asset_red_notional_float": 0.0,
        "top_red_asset_red_notional_share_float": np.nan,
        "top5_red_asset_red_notional_share_float": np.nan,
        "worst_year_by_red_notional_share_int": None,
        "worst_year_red_notional_share_float": np.nan,
        "worst_year_red_notional_float": 0.0,
    }
    if len(assessed_order_df) == 0:
        return empty_concentration_dict

    red_order_df = assessed_order_df[
        assessed_order_df.get("capacity_bucket_str", pd.Series(dtype=str)) == RED_BUCKET_STR
    ]
    red_notional_total_float = float(
        red_order_df.get("order_dollar_float", pd.Series(dtype=float)).fillna(0.0).sum()
    )
    if red_notional_total_float <= 0.0:
        return empty_concentration_dict

    red_asset_notional_ser = (
        red_order_df.groupby("asset")["order_dollar_float"]
        .sum()
        .sort_values(ascending=False)
    )
    top_red_asset_str = str(red_asset_notional_ser.index[0])
    top_red_asset_red_notional_float = float(red_asset_notional_ser.iloc[0])
    top5_red_asset_red_notional_float = float(red_asset_notional_ser.head(5).sum())

    year_total_notional_ser = assessed_order_df.groupby("year_int")["order_dollar_float"].sum()
    year_red_notional_ser = red_order_df.groupby("year_int")["order_dollar_float"].sum()
    year_red_share_ser = (year_red_notional_ser / year_total_notional_ser).dropna()
    if len(year_red_share_ser) == 0:
        worst_year_by_red_notional_share_int = None
        worst_year_red_notional_share_float = np.nan
        worst_year_red_notional_float = 0.0
    else:
        worst_year_by_red_notional_share_int = int(year_red_share_ser.idxmax())
        worst_year_red_notional_share_float = float(year_red_share_ser.max())
        worst_year_red_notional_float = float(
            year_red_notional_ser.loc[worst_year_by_red_notional_share_int]
        )

    return {
        "top_red_asset_str": top_red_asset_str,
        "top_red_asset_red_notional_float": top_red_asset_red_notional_float,
        "top_red_asset_red_notional_share_float": _safe_divide_float(
            top_red_asset_red_notional_float,
            red_notional_total_float,
        ),
        "top5_red_asset_red_notional_share_float": _safe_divide_float(
            top5_red_asset_red_notional_float,
            red_notional_total_float,
        ),
        "worst_year_by_red_notional_share_int": worst_year_by_red_notional_share_int,
        "worst_year_red_notional_share_float": worst_year_red_notional_share_float,
        "worst_year_red_notional_float": worst_year_red_notional_float,
    }


def _build_scale_limit_summary_dict(summary_dict: dict[str, object]) -> dict[str, object]:
    return {
        "scale_to_p95_auction_red_threshold_float": _scale_to_threshold_float(
            summary_dict.get("p95_participation_float"),
            CAPACITY_SENSITIVE_MAX_PARTICIPATION_FLOAT,
        ),
        "scale_to_max_auction_proxy_float": _scale_to_threshold_float(
            summary_dict.get("max_participation_float"),
            1.0,
        ),
        "scale_to_p95_adv_5pct_float": _scale_to_threshold_float(
            summary_dict.get("adv_p95_participation_float"),
            ADV_FINE_MAX_PARTICIPATION_FLOAT,
        ),
    }


def _build_verdict_explanation_str(
    summary_dict: dict[str, object],
    cost_verdict_str: str,
    auction_verdict_str: str,
    friction_verdict_str: str,
) -> str:
    delta_bps_float = _coerce_float(summary_dict.get("adv_rule_minus_default_bps_float"))
    red_notional_share_float = _coerce_float(summary_dict.get("red_notional_share_float"))
    stressed_notional_share_float = _coerce_float(
        summary_dict.get("capacity_stressed_notional_share_float")
    )
    if friction_verdict_str == FRICTION_NO_DATA_VERDICT_STR:
        return "Not enough assessed orders to produce a friction realism verdict."

    if _is_finite_float(delta_bps_float):
        cost_phrase_str = (
            f"Cost model is {cost_verdict_str.lower()} by "
            f"{abs(delta_bps_float):.2f} bps versus the ADV-rule estimate"
        )
    else:
        cost_phrase_str = "Cost model could not be classified"

    if _is_finite_float(red_notional_share_float) and _is_finite_float(stressed_notional_share_float):
        auction_phrase_str = (
            f"{red_notional_share_float * 100.0:.2f}% of assessed notional is Red "
            f"and {stressed_notional_share_float * 100.0:.2f}% is auction-stressed "
            "versus nominal auction capacity"
        )
    else:
        auction_phrase_str = "auction capacity could not be classified"

    return (
        f"{cost_phrase_str}, but {auction_phrase_str}; "
        f"combined verdict is {friction_verdict_str}."
    )


def _build_run_info_dict(friction_result_obj: FrictionAnalysisResult) -> dict[str, object]:
    return {
        "entity_type": "strategy",
        "entity_id": friction_result_obj.strategy_name_str,
        "analysis_type": FRICTION_ANALYSIS_TYPE_STR,
        "parameters": {
            "execution_policy_str": friction_result_obj.execution_policy_str,
            "auction_fraction_float": friction_result_obj.auction_fraction_float,
            "auction_impact_lambda_bps_float": friction_result_obj.summary_dict.get(
                "auction_impact_lambda_bps_float"
            ),
            "adv_lookback_int": friction_result_obj.adv_lookback_int,
            "default_slippage_bps_float": friction_result_obj.default_slippage_bps_float,
        },
    }


def _build_metadata_dict(friction_result_obj: FrictionAnalysisResult) -> dict[str, object]:
    return {
        "artifact_type": FRICTION_ANALYSIS_TYPE_STR,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_name_str": friction_result_obj.strategy_name_str,
        "execution_policy_str": friction_result_obj.execution_policy_str,
        "auction_fraction_float": friction_result_obj.auction_fraction_float,
        "auction_proxy_description_str": (
            "auction-liquidity proxy, not observed auction volume"
        ),
        "adv_rule_description_str": (
            "full-day ADV tiered rule-of-thumb, not an auction-capacity estimate"
        ),
        "auction_impact_description_str": AUCTION_IMPACT_MODEL_LABEL_STR,
        "moo_auction_fraction_float": MOO_AUCTION_FRACTION_FLOAT,
        "moc_auction_fraction_float": MOC_AUCTION_FRACTION_FLOAT,
        "moo_auction_impact_lambda_bps_float": MOO_AUCTION_IMPACT_LAMBDA_BPS_FLOAT,
        "moc_auction_impact_lambda_bps_float": MOC_AUCTION_IMPACT_LAMBDA_BPS_FLOAT,
        "adv_lookback_int": friction_result_obj.adv_lookback_int,
        "default_slippage_bps_float": friction_result_obj.default_slippage_bps_float,
    }


def _build_report_html_str(friction_result_obj: FrictionAnalysisResult) -> str:
    summary_dict = friction_result_obj.summary_dict
    strategy_name_html = html.escape(friction_result_obj.strategy_name_str)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy_name_html} FrictionAnalysis</title>
<style>
{_report_css_str()}
</style>
</head>
<body>
<main>
<header class="page-header">
<div>
<p class="eyebrow">FrictionAnalysis V1</p>
<h1>{strategy_name_html}</h1>
<p class="subtitle">Capacity check, ADV-rule cost, and adverse auction-impact estimate.</p>
</div>
</header>
{_verdict_panel_html_str(summary_dict)}
{_auction_impact_card_html_str(summary_dict)}
{_kpi_strip_html_str(summary_dict)}
<section class="section-grid">
<article class="panel">
<h2>Bucket Distribution</h2>
{_bucket_distribution_html_str(friction_result_obj.friction_summary_df)}
</article>
<article class="panel">
<h2>Estimated Impact Details</h2>
{_impact_panel_html_str(friction_result_obj)}
</article>
</section>
<section class="section-grid">
<article class="panel">
<h2>Assumptions</h2>
{_assumption_panel_html_str(friction_result_obj)}
</article>
<article class="panel">
<h2>Interpretation</h2>
{_interpretation_panel_html_str()}
</article>
</section>
<section class="panel">
<h2>Worst Orders</h2>
{_order_table_html_str(_worst_order_df(friction_result_obj.friction_order_df))}
</section>
<section class="section-grid">
<article class="panel">
<h2>Worst Assets</h2>
{_asset_table_html_str(_worst_asset_df(friction_result_obj.friction_order_df))}
</article>
<article class="panel">
<h2>Worst Years</h2>
{_year_table_html_str(_worst_year_df(friction_result_obj.friction_order_df))}
</article>
</section>
</main>
</body>
</html>"""


def _report_css_str() -> str:
    return """
:root {
  --ink: #172b4d;
  --muted: #626f86;
  --page: #f7f8fa;
  --panel: #ffffff;
  --border: #dfe1e6;
  --fine: #22a06b;
  --watch: #f5a524;
  --sensitive: #e06c00;
  --red: #c9372c;
  --unavailable: #8590a2;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font-family: "Segoe UI", Arial, sans-serif;
  font-size: 14px;
  line-height: 1.45;
}
main {
  max-width: 1320px;
  margin: 0 auto;
  padding: 28px;
}
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 18px;
}
.eyebrow {
  margin: 0 0 4px 0;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
h1 {
  margin: 0;
  font-size: 30px;
  letter-spacing: 0;
}
h2 {
  margin: 0 0 14px 0;
  font-size: 16px;
  letter-spacing: 0;
}
.subtitle {
  margin: 6px 0 0 0;
  color: var(--muted);
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
  gap: 10px;
  margin: 18px 0;
}
.verdict-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-left: 5px solid var(--watch);
  border-radius: 8px;
  padding: 16px;
  margin: 18px 0;
}
.verdict-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}
.verdict-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  background: #fbfcfe;
}
.verdict-label {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.verdict-value {
  margin: 4px 0 0 0;
  font-size: 22px;
  font-weight: 800;
}
.verdict-text {
  margin: 0;
  color: var(--muted);
}
.impact-callout {
  background: var(--panel);
  border: 1px solid var(--border);
  border-left: 5px solid var(--red);
  border-radius: 8px;
  padding: 16px;
  margin: 18px 0;
}
.impact-callout h2 {
  margin-bottom: 10px;
}
.impact-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}
.impact-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  background: #fbfcfe;
}
.impact-label {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.impact-value {
  margin: 4px 0 0 0;
  font-size: 20px;
  font-weight: 800;
}
.impact-note {
  margin: 0;
  color: var(--muted);
}
.verdict-Conservative,
.verdict-Reasonable,
.verdict-Pass { color: var(--fine); }
.verdict-Watch,
.verdict-Optimistic,
.verdict-Stressed { color: #8f5c00; }
.verdict-Cost-Problem,
.verdict-Not-Believable { color: var(--red); }
.verdict-No-Data { color: var(--unavailable); }
.kpi {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
}
.kpi-label {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
}
.kpi-value {
  margin: 4px 0 0 0;
  font-size: 20px;
  font-weight: 700;
}
.section-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.8fr);
  gap: 14px;
  margin-bottom: 14px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 14px;
}
.bucket-row {
  display: grid;
  grid-template-columns: 160px minmax(180px, 1fr) minmax(180px, 1fr);
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}
.bucket-name {
  font-weight: 700;
}
.bar-track {
  height: 18px;
  background: #eef1f5;
  border-radius: 4px;
  overflow: hidden;
}
.bar {
  height: 100%;
  min-width: 2px;
}
.bar-fine { background: var(--fine); }
.bar-watch { background: var(--watch); }
.bar-sensitive { background: var(--sensitive); }
.bar-red { background: var(--red); }
.bar-unavailable { background: var(--unavailable); }
.bar-label {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
}
.table-wrap {
  width: 100%;
  max-height: 460px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th, td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  white-space: nowrap;
}
th {
  background: #f3f6fa;
  position: sticky;
  top: 0;
  z-index: 1;
}
.assumption-list {
  margin: 0;
  padding-left: 18px;
}
.assumption-list li {
  margin-bottom: 8px;
}
.tag {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 999px;
  background: #eef1f5;
  color: var(--ink);
  font-size: 12px;
  font-weight: 700;
}
.bucket-Fine { color: var(--fine); font-weight: 700; }
.bucket-Watch { color: #8f5c00; font-weight: 700; }
.bucket-Capacity-sensitive { color: var(--sensitive); font-weight: 700; }
.bucket-Red { color: var(--red); font-weight: 700; }
.bucket-Unavailable { color: var(--unavailable); font-weight: 700; }
@media (max-width: 980px) {
  main { padding: 16px; }
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .section-grid { grid-template-columns: 1fr; }
  .bucket-row { grid-template-columns: 1fr; gap: 4px; }
}
"""


def _kpi_strip_html_str(summary_dict: dict[str, object]) -> str:
    kpi_tuple_list = [
        ("Total assessed orders", _fmt_int_str(summary_dict["assessed_order_count_int"])),
        ("Assessed notional", _fmt_dollar_str(summary_dict["assessed_notional_float"])),
        (
            "P50 / P95 / Max participation",
            (
                f"{_fmt_pct_from_decimal_str(summary_dict['p50_participation_float'])} / "
                f"{_fmt_pct_from_decimal_str(summary_dict['p95_participation_float'])} / "
                f"{_fmt_pct_from_decimal_str(summary_dict['max_participation_float'])}"
            ),
        ),
        (
            "Auction red orders",
            (
                f"{_fmt_int_str(summary_dict['red_order_count_int'])} "
                f"({_fmt_pct_from_decimal_str(summary_dict['red_order_share_float'])})"
            ),
        ),
        (
            "Auction stressed orders",
            (
                f"{_fmt_int_str(summary_dict['capacity_stressed_order_count_int'])} "
                f"({_fmt_pct_from_decimal_str(summary_dict['capacity_stressed_order_share_float'])})"
            ),
        ),
        (
            "Red notional share",
            _fmt_pct_from_decimal_str(summary_dict["red_notional_share_float"]),
        ),
        (
            "Capacity pass rate",
            _fmt_pct_from_decimal_str(summary_dict["capacity_pass_order_rate_float"]),
        ),
        (
            "ADV P95 / Max",
            (
                f"{_fmt_pct_from_decimal_str(summary_dict['adv_p95_participation_float'])} / "
                f"{_fmt_pct_from_decimal_str(summary_dict['adv_max_participation_float'])}"
            ),
        ),
        (
            "ADV-rule blended cost",
            _fmt_bps_str(summary_dict["adv_rule_slippage_blended_bps_float"]),
        ),
        (
            "ADV-rule delta vs default",
            _fmt_bps_signed_str(summary_dict["adv_rule_minus_default_bps_float"]),
        ),
    ]
    card_html_list = [
        (
            "<div class=\"kpi\">"
            f"<p class=\"kpi-label\">{html.escape(label_str)}</p>"
            f"<p class=\"kpi-value\">{html.escape(value_str)}</p>"
            "</div>"
        )
        for label_str, value_str in kpi_tuple_list
    ]
    return f"<section class=\"kpi-grid\">{''.join(card_html_list)}</section>"


def _auction_impact_card_html_str(summary_dict: dict[str, object]) -> str:
    card_tuple_list = [
        (
            "Auction impact estimate",
            (
                f"{_fmt_bps_str(summary_dict.get('auction_impact_blended_bps_float'))} / "
                f"{_fmt_dollar_str(summary_dict.get('auction_impact_dollar_float'))}"
            ),
        ),
        (
            "Current default",
            (
                f"{_fmt_bps_str(summary_dict.get('default_slippage_blended_bps_float'))} / "
                f"{_fmt_dollar_str(summary_dict.get('default_slippage_dollar_float'))}"
            ),
        ),
        (
            "Extra drag vs default",
            (
                f"{_fmt_bps_signed_str(summary_dict.get('auction_impact_minus_default_bps_float'))} / "
                f"{_fmt_dollar_signed_str(summary_dict.get('auction_impact_minus_default_dollar_float'))}"
            ),
        ),
        (
            "Estimated ann. return",
            (
                f"{_fmt_pct_from_decimal_str(summary_dict.get('current_annual_return_float'))} -> "
                f"{_fmt_pct_from_decimal_str(summary_dict.get('auction_impact_estimated_annual_return_float'))} "
                f"({_fmt_pp_signed_str(summary_dict.get('auction_impact_estimated_annual_return_delta_float'))})"
            ),
        ),
        (
            "Estimated Sharpe",
            (
                f"{_fmt_float_str(summary_dict.get('current_sharpe_float'))} -> "
                f"{_fmt_float_str(summary_dict.get('auction_impact_estimated_sharpe_float'))} "
                f"({_fmt_float_signed_str(summary_dict.get('auction_impact_estimated_sharpe_delta_float'))})"
            ),
        ),
        (
            "Readout",
            str(summary_dict.get("auction_impact_delta_label_str", "N/A")),
        ),
    ]
    card_html_list = [
        (
            "<div class=\"impact-card\">"
            f"<p class=\"impact-label\">{html.escape(label_str)}</p>"
            f"<p class=\"impact-value\">{html.escape(value_str)}</p>"
            "</div>"
        )
        for label_str, value_str in card_tuple_list
    ]
    return (
        "<section class=\"impact-callout\">"
        "<h2>Auction Impact</h2>"
        f"<div class=\"impact-grid\">{''.join(card_html_list)}</div>"
        f"<p class=\"impact-note\">{html.escape(str(summary_dict.get('auction_impact_interpretation_str', '')))}</p>"
        f"<p class=\"impact-note\">{html.escape(str(summary_dict.get('auction_impact_performance_note_str', '')))}</p>"
        "<p class=\"impact-note\">Estimated adverse auction impact, not observed auction slippage.</p>"
        "</section>"
    )


def _verdict_panel_html_str(summary_dict: dict[str, object]) -> str:
    verdict_tuple_list = [
        ("Friction Verdict", str(summary_dict.get("friction_verdict_str", "N/A"))),
        ("Cost Verdict", str(summary_dict.get("cost_verdict_str", "N/A"))),
        ("Auction Verdict", str(summary_dict.get("auction_verdict_str", "N/A"))),
    ]
    verdict_card_html_list = [
        (
            "<div class=\"verdict-card\">"
            f"<p class=\"verdict-label\">{html.escape(label_str)}</p>"
            f"<p class=\"verdict-value verdict-{_verdict_css_token_str(value_str)}\">"
            f"{html.escape(value_str)}</p>"
            "</div>"
        )
        for label_str, value_str in verdict_tuple_list
    ]
    return (
        "<section class=\"verdict-panel\">"
        f"<div class=\"verdict-grid\">{''.join(verdict_card_html_list)}</div>"
        f"<p class=\"verdict-text\">{html.escape(str(summary_dict.get('verdict_explanation_str', '')))}</p>"
        f"{_decision_diagnostic_html_str(summary_dict)}"
        "</section>"
    )


def _decision_diagnostic_html_str(summary_dict: dict[str, object]) -> str:
    diagnostic_row_tuple_list = [
        ("Top red asset", str(summary_dict.get("top_red_asset_str") or "N/A")),
        (
            "Top red asset share",
            _fmt_pct_from_decimal_str(summary_dict.get("top_red_asset_red_notional_share_float")),
        ),
        (
            "Top 5 red assets share",
            _fmt_pct_from_decimal_str(summary_dict.get("top5_red_asset_red_notional_share_float")),
        ),
        (
            "Worst red year",
            _fmt_optional_int_str(summary_dict.get("worst_year_by_red_notional_share_int")),
        ),
        (
            "Worst year red share",
            _fmt_pct_from_decimal_str(summary_dict.get("worst_year_red_notional_share_float")),
        ),
        (
            "Scale to p95 auction Red",
            _fmt_scale_str(summary_dict.get("scale_to_p95_auction_red_threshold_float")),
        ),
        (
            "Scale to p95 ADV = 5%",
            _fmt_scale_str(summary_dict.get("scale_to_p95_adv_5pct_float")),
        ),
    ]
    row_html_list = [
        (
            "<tr>"
            f"<td>{html.escape(label_str)}</td>"
            f"<td>{html.escape(value_str)}</td>"
            "</tr>"
        )
        for label_str, value_str in diagnostic_row_tuple_list
    ]
    return (
        "<div class=\"table-wrap\" style=\"margin-top: 12px; max-height: none;\">"
        "<table><tbody>"
        f"{''.join(row_html_list)}"
        "</tbody></table></div>"
    )


def _bucket_distribution_html_str(friction_summary_df: pd.DataFrame) -> str:
    if len(friction_summary_df) == 0:
        return "<p>No orders.</p>"
    max_order_count_int = max(
        1,
        int(friction_summary_df["order_count_int"].max()),
    )
    max_notional_float = max(
        1.0,
        float(friction_summary_df["notional_float"].max()),
    )
    row_html_list: list[str] = []
    for _, summary_ser in friction_summary_df.iterrows():
        bucket_str = str(summary_ser["capacity_bucket_str"])
        bar_class_str = _bucket_bar_class_str(bucket_str)
        count_width_float = (
            float(summary_ser["order_count_int"]) / max_order_count_int * 100.0
        )
        notional_width_float = (
            float(summary_ser["notional_float"]) / max_notional_float * 100.0
        )
        row_html_list.append(
            "<div class=\"bucket-row\">"
            f"<div class=\"bucket-name bucket-{_bucket_css_token_str(bucket_str)}\">"
            f"{html.escape(bucket_str)}</div>"
            "<div>"
            "<div class=\"bar-track\">"
            f"<div class=\"bar {bar_class_str}\" style=\"width:{count_width_float:.1f}%\"></div>"
            "</div>"
            f"<div class=\"bar-label\">Count: {_fmt_int_str(summary_ser['order_count_int'])} "
            f"({_fmt_pct_from_decimal_str(summary_ser['order_share_float'])})</div>"
            "</div>"
            "<div>"
            "<div class=\"bar-track\">"
            f"<div class=\"bar {bar_class_str}\" style=\"width:{notional_width_float:.1f}%\"></div>"
            "</div>"
            f"<div class=\"bar-label\">Notional: {_fmt_dollar_str(summary_ser['notional_float'])} "
            f"({_fmt_pct_from_decimal_str(summary_ser['notional_share_float'])})</div>"
            "</div>"
            "</div>"
        )
    return "".join(row_html_list)


def _assumption_panel_html_str(friction_result_obj: FrictionAnalysisResult) -> str:
    thresholds_html_str = (
        f"{FINE_BUCKET_STR}: <= 1%; "
        f"{WATCH_BUCKET_STR}: > 1% to <= 2.5%; "
        f"{CAPACITY_SENSITIVE_BUCKET_STR}: > 2.5% to <= 5%; "
        f"{RED_BUCKET_STR}: > 5%"
    )
    return (
        "<ul class=\"assumption-list\">"
        "<li><span class=\"tag\">MOO proxy</span> "
        f"{_fmt_pct_from_decimal_str(MOO_AUCTION_FRACTION_FLOAT)} * lagged "
        f"{friction_result_obj.adv_lookback_int}d median dollar ADV.</li>"
        "<li><span class=\"tag\">MOC proxy</span> "
        f"{_fmt_pct_from_decimal_str(MOC_AUCTION_FRACTION_FLOAT)} * lagged "
        f"{friction_result_obj.adv_lookback_int}d median dollar ADV.</li>"
        "<li><span class=\"tag\">Selected policy</span> "
        f"{html.escape(friction_result_obj.execution_policy_str)} at "
        f"{_fmt_pct_from_decimal_str(friction_result_obj.auction_fraction_float)}.</li>"
        "<li><span class=\"tag\">Thresholds</span> "
        f"{html.escape(thresholds_html_str)}</li>"
        "<li><span class=\"tag\">ADV rule</span> "
        f"<5% ADV = {ADV_FINE_SLIPPAGE_BPS_FLOAT:.1f} bp; "
        f"5-10% ADV = {ADV_WATCH_SLIPPAGE_BPS_FLOAT:.1f} bps; "
        f">=10% ADV = {ADV_RED_SLIPPAGE_BPS_FLOAT:.1f} bps.</li>"
        "<li><span class=\"tag\">Auction impact</span> "
        "adverse proxy: bps = lambda * sqrt(ADV participation / 1% ADV); "
        f"MOO lambda = {MOO_AUCTION_IMPACT_LAMBDA_BPS_FLOAT:.1f}; "
        f"MOC lambda = {MOC_AUCTION_IMPACT_LAMBDA_BPS_FLOAT:.1f}.</li>"
        "<li><span class=\"tag\">Default model</span> "
        f"{_fmt_bps_str(friction_result_obj.default_slippage_bps_float)} "
        "per assessed order notional.</li>"
        "<li>This is an auction-liquidity proxy, not observed auction volume.</li>"
        "</ul>"
    )


def _impact_panel_html_str(friction_result_obj: FrictionAnalysisResult) -> str:
    summary_dict = friction_result_obj.summary_dict
    impact_row_tuple_list = [
        (
            "Auction-impact estimate",
            _fmt_dollar_str(summary_dict["auction_impact_dollar_float"]),
        ),
        (
            "Auction-impact blended bps",
            _fmt_bps_str(summary_dict["auction_impact_blended_bps_float"]),
        ),
        (
            "Auction-impact delta vs default",
            _fmt_dollar_signed_str(
                summary_dict["auction_impact_minus_default_dollar_float"]
            ),
        ),
        (
            "Auction-impact delta bps",
            _fmt_bps_signed_str(summary_dict["auction_impact_minus_default_bps_float"]),
        ),
        (
            "Auction-impact P50 / P95 / Max",
            (
                f"{_fmt_bps_str(summary_dict['auction_impact_p50_bps_float'])} / "
                f"{_fmt_bps_str(summary_dict['auction_impact_p95_bps_float'])} / "
                f"{_fmt_bps_str(summary_dict['auction_impact_max_bps_float'])}"
            ),
        ),
        (
            "Current ann. return",
            _fmt_pct_from_decimal_str(summary_dict["current_annual_return_float"]),
        ),
        (
            "Auction-impact est. ann. return",
            _fmt_pct_from_decimal_str(
                summary_dict["auction_impact_estimated_annual_return_float"]
            ),
        ),
        (
            "Estimated ann. return change",
            _fmt_pp_signed_str(
                summary_dict["auction_impact_estimated_annual_return_delta_float"]
            ),
        ),
        (
            "Current Sharpe",
            _fmt_float_str(summary_dict["current_sharpe_float"]),
        ),
        (
            "Auction-impact est. Sharpe",
            _fmt_float_str(summary_dict["auction_impact_estimated_sharpe_float"]),
        ),
        (
            "Estimated Sharpe change",
            _fmt_float_signed_str(
                summary_dict["auction_impact_estimated_sharpe_delta_float"]
            ),
        ),
        (
            "ADV-rule slippage cost",
            _fmt_dollar_str(summary_dict["adv_rule_slippage_dollar_float"]),
        ),
        (
            "ADV-rule blended bps",
            _fmt_bps_str(summary_dict["adv_rule_slippage_blended_bps_float"]),
        ),
        (
            "Default 2.5 bps cost",
            _fmt_dollar_str(summary_dict["default_slippage_dollar_float"]),
        ),
        (
            "Delta vs default cost",
            _fmt_dollar_signed_str(summary_dict["adv_rule_minus_default_dollar_float"]),
        ),
        (
            "Delta vs default bps",
            _fmt_bps_signed_str(summary_dict["adv_rule_minus_default_bps_float"]),
        ),
        (
            "ADV red orders",
            (
                f"{_fmt_int_str(summary_dict['adv_red_order_count_int'])} "
                f"({_fmt_pct_from_decimal_str(summary_dict['adv_red_order_share_float'])})"
            ),
        ),
    ]
    row_html_list = [
        (
            "<tr>"
            f"<td>{html.escape(label_str)}</td>"
            f"<td>{html.escape(value_str)}</td>"
            "</tr>"
        )
        for label_str, value_str in impact_row_tuple_list
    ]
    return (
        "<div class=\"table-wrap\"><table><tbody>"
        f"{''.join(row_html_list)}"
        "</tbody></table></div>"
    )


def _interpretation_panel_html_str() -> str:
    return (
        "<ul class=\"assumption-list\">"
        "<li><span class=\"tag\">Capacity check</span> asks whether the order is "
        "large versus the selected MOO/MOC auction proxy.</li>"
        "<li><span class=\"tag\">ADV-rule cost</span> is the simple full-day "
        "liquidity rule-of-thumb.</li>"
        "<li><span class=\"tag\">Auction-impact estimate</span> is an adverse "
        "MOO/MOC impact proxy. It is not realized broker slippage.</li>"
        "<li><span class=\"tag\">Estimated ann. return / Sharpe</span> subtracts "
        "the cumulative auction-impact delta from the completed equity curve. "
        "It does not rescale, rerun, or change fills.</li>"
        "<li>Delta vs default compares the ADV rule to the current default "
        "slippage assumption. Negative means the default is harsher than this "
        "rule; positive means this rule estimates extra drag.</li>"
        "</ul>"
    )


def _worst_order_df(friction_order_df: pd.DataFrame) -> pd.DataFrame:
    if len(friction_order_df) == 0:
        return pd.DataFrame()
    assessed_order_df = friction_order_df[friction_order_df["assessed_bool"] == True]
    if len(assessed_order_df) == 0:
        return pd.DataFrame()
    column_list = [
        "bar",
        "asset",
        "side",
        "order_dollar_float",
        "adv20_dollar_lagged_float",
        "auction_proxy_dollar_float",
        "auction_participation_float",
        "adv_participation_float",
        "adv_rule_slippage_bps_float",
        "adv_rule_minus_default_bps_float",
        "auction_impact_bps_float",
        "auction_impact_minus_default_bps_float",
        "capacity_bucket_str",
        "adv_bucket_str",
    ]
    return (
        assessed_order_df.sort_values("auction_participation_float", ascending=False)
        .head(20)[column_list]
        .copy()
    )


def _worst_asset_df(friction_order_df: pd.DataFrame) -> pd.DataFrame:
    if len(friction_order_df) == 0:
        return pd.DataFrame()
    assessed_order_df = friction_order_df[friction_order_df["assessed_bool"] == True]
    if len(assessed_order_df) == 0:
        return pd.DataFrame()
    asset_group_obj = assessed_order_df.groupby("asset", dropna=False)
    worst_asset_df = asset_group_obj.agg(
        order_count_int=("asset", "size"),
        notional_float=("order_dollar_float", "sum"),
        p95_participation_float=("auction_participation_float", lambda value_ser: value_ser.quantile(0.95)),
        max_participation_float=("auction_participation_float", "max"),
        adv_p95_participation_float=("adv_participation_float", lambda value_ser: value_ser.quantile(0.95)),
        adv_max_participation_float=("adv_participation_float", "max"),
        adv_rule_slippage_dollar_float=("adv_rule_slippage_dollar_float", "sum"),
        auction_impact_dollar_float=("auction_impact_dollar_float", "sum"),
        red_order_count_int=("capacity_bucket_str", lambda value_ser: int((value_ser == RED_BUCKET_STR).sum())),
    ).reset_index()
    return worst_asset_df.sort_values(
        ["p95_participation_float", "max_participation_float"],
        ascending=False,
    ).head(25)


def _worst_year_df(friction_order_df: pd.DataFrame) -> pd.DataFrame:
    if len(friction_order_df) == 0:
        return pd.DataFrame()
    assessed_order_df = friction_order_df[friction_order_df["assessed_bool"] == True]
    if len(assessed_order_df) == 0:
        return pd.DataFrame()
    year_group_obj = assessed_order_df.groupby("year_int", dropna=False)
    worst_year_df = year_group_obj.agg(
        order_count_int=("year_int", "size"),
        notional_float=("order_dollar_float", "sum"),
        p95_participation_float=("auction_participation_float", lambda value_ser: value_ser.quantile(0.95)),
        max_participation_float=("auction_participation_float", "max"),
        adv_p95_participation_float=("adv_participation_float", lambda value_ser: value_ser.quantile(0.95)),
        adv_max_participation_float=("adv_participation_float", "max"),
        adv_rule_slippage_dollar_float=("adv_rule_slippage_dollar_float", "sum"),
        auction_impact_dollar_float=("auction_impact_dollar_float", "sum"),
        red_order_count_int=("capacity_bucket_str", lambda value_ser: int((value_ser == RED_BUCKET_STR).sum())),
    ).reset_index()
    return worst_year_df.sort_values(
        ["p95_participation_float", "max_participation_float"],
        ascending=False,
    )


def _order_table_html_str(order_df: pd.DataFrame) -> str:
    if len(order_df) == 0:
        return "<p>No assessed orders.</p>"
    table_row_html_list: list[str] = []
    for _, order_ser in order_df.iterrows():
        bucket_str = str(order_ser["capacity_bucket_str"])
        table_row_html_list.append(
            "<tr>"
            f"<td>{html.escape(_date_str(order_ser['bar']))}</td>"
            f"<td>{html.escape(str(order_ser['asset']))}</td>"
            f"<td>{html.escape(str(order_ser['side']))}</td>"
            f"<td>{_fmt_dollar_str(order_ser['order_dollar_float'])}</td>"
            f"<td>{_fmt_dollar_str(order_ser['adv20_dollar_lagged_float'])}</td>"
            f"<td>{_fmt_dollar_str(order_ser['auction_proxy_dollar_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(order_ser['auction_participation_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(order_ser['adv_participation_float'])}</td>"
            f"<td>{_fmt_bps_str(order_ser['adv_rule_slippage_bps_float'])}</td>"
            f"<td>{_fmt_bps_signed_str(order_ser['adv_rule_minus_default_bps_float'])}</td>"
            f"<td>{_fmt_bps_str(order_ser['auction_impact_bps_float'])}</td>"
            f"<td>{_fmt_bps_signed_str(order_ser['auction_impact_minus_default_bps_float'])}</td>"
            f"<td class=\"bucket-{_bucket_css_token_str(bucket_str)}\">{html.escape(bucket_str)}</td>"
            f"<td>{html.escape(str(order_ser['adv_bucket_str']))}</td>"
            "</tr>"
        )
    return (
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Date</th><th>Asset</th><th>Side</th><th>Order $</th>"
        "<th>Lagged 20d ADV $</th><th>Auction proxy $</th>"
        "<th>Auction part</th><th>ADV part</th><th>ADV bps</th>"
        "<th>ADV delta</th><th>Auction impact</th><th>Impact delta</th>"
        "<th>Auction bucket</th><th>ADV bucket</th>"
        "</tr></thead><tbody>"
        f"{''.join(table_row_html_list)}"
        "</tbody></table></div>"
    )


def _asset_table_html_str(asset_df: pd.DataFrame) -> str:
    if len(asset_df) == 0:
        return "<p>No assessed assets.</p>"
    table_row_html_list: list[str] = []
    for _, asset_ser in asset_df.iterrows():
        table_row_html_list.append(
            "<tr>"
            f"<td>{html.escape(str(asset_ser['asset']))}</td>"
            f"<td>{_fmt_int_str(asset_ser['order_count_int'])}</td>"
            f"<td>{_fmt_dollar_str(asset_ser['notional_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(asset_ser['p95_participation_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(asset_ser['max_participation_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(asset_ser['adv_p95_participation_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(asset_ser['adv_max_participation_float'])}</td>"
            f"<td>{_fmt_dollar_str(asset_ser['adv_rule_slippage_dollar_float'])}</td>"
            f"<td>{_fmt_dollar_str(asset_ser['auction_impact_dollar_float'])}</td>"
            f"<td>{_fmt_int_str(asset_ser['red_order_count_int'])}</td>"
            "</tr>"
        )
    return (
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Asset</th><th>Orders</th><th>Notional</th><th>Auction P95</th>"
        "<th>Auction Max</th><th>ADV P95</th><th>ADV Max</th>"
        "<th>ADV cost</th><th>Auction impact</th><th>Auction Red</th>"
        "</tr></thead><tbody>"
        f"{''.join(table_row_html_list)}"
        "</tbody></table></div>"
    )


def _year_table_html_str(year_df: pd.DataFrame) -> str:
    if len(year_df) == 0:
        return "<p>No assessed years.</p>"
    table_row_html_list: list[str] = []
    for _, year_ser in year_df.iterrows():
        table_row_html_list.append(
            "<tr>"
            f"<td>{_fmt_int_str(year_ser['year_int'])}</td>"
            f"<td>{_fmt_int_str(year_ser['order_count_int'])}</td>"
            f"<td>{_fmt_dollar_str(year_ser['notional_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(year_ser['p95_participation_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(year_ser['max_participation_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(year_ser['adv_p95_participation_float'])}</td>"
            f"<td>{_fmt_pct_from_decimal_str(year_ser['adv_max_participation_float'])}</td>"
            f"<td>{_fmt_dollar_str(year_ser['adv_rule_slippage_dollar_float'])}</td>"
            f"<td>{_fmt_dollar_str(year_ser['auction_impact_dollar_float'])}</td>"
            f"<td>{_fmt_int_str(year_ser['red_order_count_int'])}</td>"
            "</tr>"
        )
    return (
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Year</th><th>Orders</th><th>Notional</th><th>Auction P95</th>"
        "<th>Auction Max</th><th>ADV P95</th><th>ADV Max</th>"
        "<th>ADV cost</th><th>Auction impact</th><th>Auction Red</th>"
        "</tr></thead><tbody>"
        f"{''.join(table_row_html_list)}"
        "</tbody></table></div>"
    )


def _write_json_file(json_path: Path, data_dict: dict[str, object]) -> None:
    json_path.write_text(
        json.dumps(
            _sanitize_json_obj(data_dict),
            indent=2,
            sort_keys=True,
            default=_json_default_obj,
        ),
        encoding="utf-8",
    )


def _sanitize_json_obj(value_obj):
    if isinstance(value_obj, dict):
        return {
            str(key_obj): _sanitize_json_obj(nested_value_obj)
            for key_obj, nested_value_obj in value_obj.items()
        }
    if isinstance(value_obj, (list, tuple)):
        return [_sanitize_json_obj(nested_value_obj) for nested_value_obj in value_obj]
    if isinstance(value_obj, Path):
        return str(value_obj)
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.isoformat()
    if isinstance(value_obj, np.integer):
        return int(value_obj)
    if isinstance(value_obj, np.floating):
        value_float = float(value_obj)
        if np.isnan(value_float):
            return None
        return value_float
    if isinstance(value_obj, np.bool_):
        return bool(value_obj)
    if isinstance(value_obj, float) and np.isnan(value_obj):
        return None
    return value_obj


def _json_default_obj(value_obj):
    if isinstance(value_obj, Path):
        return str(value_obj)
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.isoformat()
    if isinstance(value_obj, np.integer):
        return int(value_obj)
    if isinstance(value_obj, np.floating):
        value_float = float(value_obj)
        if np.isnan(value_float):
            return None
        return value_float
    if isinstance(value_obj, np.bool_):
        return bool(value_obj)
    if isinstance(value_obj, float) and np.isnan(value_obj):
        return None
    return value_obj


def _lookup_bar_value_float(value_ser: pd.Series, bar_ts: pd.Timestamp) -> float:
    if bar_ts in value_ser.index:
        return _coerce_float(value_ser.loc[bar_ts])
    normalized_bar_ts = bar_ts.normalize()
    if normalized_bar_ts in value_ser.index:
        return _coerce_float(value_ser.loc[normalized_bar_ts])
    return np.nan


def _order_dollar_float(
    amount_float: float,
    price_float: float,
    signed_total_value_float: float,
) -> float:
    if _is_finite_float(signed_total_value_float):
        return abs(signed_total_value_float)
    if _is_finite_float(amount_float) and _is_finite_float(price_float):
        return abs(amount_float * price_float)
    return np.nan


def _side_str(amount_float: float) -> str:
    if not _is_finite_float(amount_float):
        return "Unknown"
    if amount_float > 0:
        return "Buy"
    if amount_float < 0:
        return "Sell"
    return "Flat"


def _coerce_float(value_obj) -> float:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(value_float):
        return np.nan
    return value_float


def _is_finite_float(value_obj) -> bool:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value_float))


def _safe_divide_float(numerator_obj, denominator_obj) -> float:
    numerator_float = _coerce_float(numerator_obj)
    denominator_float = _coerce_float(denominator_obj)
    if not _is_finite_float(numerator_float) or not _is_finite_float(denominator_float):
        return np.nan
    if denominator_float == 0.0:
        return np.nan
    return float(numerator_float / denominator_float)


def _scale_to_threshold_float(current_value_obj, threshold_float: float) -> float:
    current_value_float = _coerce_float(current_value_obj)
    if not _is_finite_float(current_value_float):
        return np.nan
    if current_value_float <= 0.0:
        return np.nan
    return float(threshold_float / current_value_float)


def _notional_to_bps_float(cost_dollar_obj, notional_dollar_obj) -> float:
    cost_dollar_float = _coerce_float(cost_dollar_obj)
    notional_dollar_float = _coerce_float(notional_dollar_obj)
    if not _is_finite_float(cost_dollar_float) or not _is_finite_float(notional_dollar_float):
        return np.nan
    if notional_dollar_float == 0.0:
        return np.nan
    return float(cost_dollar_float / notional_dollar_float * 10_000.0)


def _series_quantile_float(value_ser: pd.Series, quantile_float: float) -> float:
    clean_value_ser = pd.Series(value_ser, dtype=float).dropna()
    if len(clean_value_ser) == 0:
        return np.nan
    return float(clean_value_ser.quantile(quantile_float))


def _series_max_float(value_ser: pd.Series) -> float:
    clean_value_ser = pd.Series(value_ser, dtype=float).dropna()
    if len(clean_value_ser) == 0:
        return np.nan
    return float(clean_value_ser.max())


def _fmt_int_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    return f"{int(round(value_float)):,}"


def _fmt_optional_int_str(value_obj) -> str:
    if value_obj is None:
        return "N/A"
    return _fmt_int_str(value_obj)


def _fmt_dollar_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    return f"${value_float:,.0f}"


def _fmt_dollar_signed_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    sign_str = "+" if value_float >= 0.0 else "-"
    return f"{sign_str}${abs(value_float):,.0f}"


def _fmt_bps_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    return f"{value_float:,.2f} bps"


def _fmt_bps_signed_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    return f"{value_float:+,.2f} bps"


def _fmt_scale_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    return f"{value_float:,.2f}x"


def _fmt_float_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    return f"{value_float:,.2f}"


def _fmt_float_signed_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    return f"{value_float:+,.2f}"


def _fmt_pct_from_decimal_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    return f"{value_float * 100.0:,.2f}%"


def _fmt_pp_signed_str(value_obj) -> str:
    value_float = _coerce_float(value_obj)
    if not _is_finite_float(value_float):
        return "N/A"
    return f"{value_float * 100.0:+,.2f} pp"


def _date_str(value_obj) -> str:
    try:
        timestamp_obj = pd.Timestamp(value_obj)
    except (TypeError, ValueError):
        return str(value_obj)
    if pd.isna(timestamp_obj):
        return "N/A"
    return timestamp_obj.date().isoformat()


def _bucket_css_token_str(bucket_str: str) -> str:
    return str(bucket_str).replace(" ", "-")


def _verdict_css_token_str(verdict_str: str) -> str:
    return str(verdict_str).replace(" ", "-")


def _bucket_bar_class_str(bucket_str: str) -> str:
    if bucket_str == FINE_BUCKET_STR:
        return "bar-fine"
    if bucket_str == WATCH_BUCKET_STR:
        return "bar-watch"
    if bucket_str == CAPACITY_SENSITIVE_BUCKET_STR:
        return "bar-sensitive"
    if bucket_str == RED_BUCKET_STR:
        return "bar-red"
    return "bar-unavailable"
