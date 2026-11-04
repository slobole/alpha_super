"""Month-End Flow Vote: hold what month-end rebalancers must buy, SPY or TLT, by the vote of three readings of the month.

Same calendar, data roles, sizing and costs as `strategy_taa_month_end_rebalancing_flow`. What differs: the final-week
position is the mean of three published readings instead of one, and the next-month leg and the fill session are set by
the frozen rules of scripts/research/eom_tradable_20261101/SPEC_FROZEN.md. Study: docs/research/EOM_TRADABLE_20261101.md.

Decision at the close of the seventh-last session of month m (dtme 7), total-return closes only:
    P = 10000 * (0.40 - bond_weight), bond_weight = 0.4 (1 + R_IEF) / [0.6 (1 + R_SPY) + 0.4 (1 + R_IEF)], returns since the last close of m-1
    votes for SPY: P is in the bottom 20% of all prior months' P; P < 0; SPY lagged TLT at session min(15, N-6)
    final week: SPY = votes / 3, TLT = 1 - SPY, gross 100%
Next month, by month m's state: see `early_weight_tuple`.
Fills: see `MonthEndVoteFlowConfig.fill_str`; quantities always from the close before the fill.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import is_snapshot_mode_enabled_bool, norgatedata
from strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow import (
    HISTORY_START_DATE_STR,
    SIGNAL_NAMESPACE_STR,
    TRADED_ASSET_TUPLE,
    MonthEndRebalancingFlowStrategy,
    causal_bucket_float,
    exchange_session_idx,
)

STRATEGY_NAME_STR = "strategy_taa_month_end_flow_vote"
SIGNAL_ASSET_TUPLE = ("SPY", "IEF", "TLT")
FILL_TUPLE = ("close", "open", "open_late")
EARLY_RULE_TUPLE = ("flat", "pair_top")
PAIR_WEIGHT_TUPLE = (0.5, -0.5)
FLAT_WEIGHT_TUPLE = (0.0, 0.0)


@dataclass(frozen=True)
class MonthEndVoteFlowConfig:
    early_rule_str: str = "pair_top"
    fill_str: str = "close"
    backtest_start_date_str: str = "2003-01-02"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0

    def __post_init__(self) -> None:
        if self.early_rule_str not in EARLY_RULE_TUPLE or self.fill_str not in FILL_TUPLE:
            raise ValueError("Unknown next-month rule or fill.")
        if not np.isfinite(self.capital_base_float) or self.capital_base_float <= 0:
            raise ValueError("capital_base_float must be positive and finite.")
        if pd.Timestamp(self.backtest_start_date_str) < pd.Timestamp("2003-01-01"):
            raise ValueError("Scoring begins in January 2003 or later.")


DEFAULT_CONFIG = MonthEndVoteFlowConfig()


def spy_vote_count_int(bucket_float: float, pressure_float: float, half_spread_float: float) -> int:
    """A missing rank (fewer than 24 prior months) casts no vote for SPY."""
    return int(bucket_float == 1) + int(pressure_float < 0.0) + int(half_spread_float < 0.0)


def final_weight_tuple(bucket_float: float, pressure_float: float, half_spread_float: float) -> tuple[float, float]:
    spy_float = spy_vote_count_int(bucket_float, pressure_float, half_spread_float) / 3.0
    return (spy_float, 1.0 - spy_float)


def early_weight_tuple(early_rule_str: str, bucket_float: float) -> tuple[float, float]:
    """Top 40% of the prior-month drift rank: +50% SPY, -50% TLT. Otherwise flat."""
    if early_rule_str == "flat":
        return FLAT_WEIGHT_TUPLE
    return PAIR_WEIGHT_TUPLE if bucket_float in (4, 5) else FLAT_WEIGHT_TUPLE


def build_month_table_df(total_return_close_df: pd.DataFrame) -> pd.DataFrame:
    """One row per month, from observed TR closes, with no filling and no full-sample threshold."""
    price_idx = total_return_close_df.index
    if price_idx[0] > pd.Timestamp("2002-07-31"):
        raise ValueError("Full pressure history requires the July 2002 month-end close.")
    session_idx = exchange_session_idx(price_idx[-1])
    month_period_idx = session_idx.to_period("M")
    prior_pressure_list: list[float] = []
    month_row_list: list[dict] = []
    for month_period in pd.period_range("2002-08", price_idx[-1].to_period("M"), freq="M"):
        month_session_idx = session_idx[month_period_idx == month_period]
        if len(month_session_idx) < 9:
            continue
        # *** CRITICAL*** Offsets index the exchange calendar, never a price-truncated month.
        measure_ts = month_session_idx[-7]
        if measure_ts > price_idx[-1]:
            continue
        previous_close_ts = session_idx[month_period_idx == month_period - 1][-1]
        session_count_int = len(month_session_idx)
        # *** CRITICAL*** Session min(15, N-6) is at or before the measure close, so the decision never reads a later close.
        half_ts = month_session_idx[min(15, session_count_int - 6) - 1]
        endpoint_df = total_return_close_df.reindex(index=[previous_close_ts, measure_ts, half_ts], columns=list(SIGNAL_ASSET_TUPLE)).astype(float)
        if not np.isfinite(endpoint_df.to_numpy()).all() or (endpoint_df <= 0).any().any():
            raise ValueError(f"Missing/invalid TOTALRETURN signal endpoint in {month_period}.")
        base_ser = endpoint_df.iloc[0]
        spy_growth_float, ief_growth_float = float(endpoint_df.iloc[1]["SPY"] / base_ser["SPY"]), float(endpoint_df.iloc[1]["IEF"] / base_ser["IEF"])
        # wB' = .4*(1+R_IEF) / [.6*(1+R_SPY)+.4*(1+R_IEF)]; pressure_bps = 10000*(.4-wB'). Positive means buy bonds.
        bond_weight_float = 0.4 * ief_growth_float / (0.6 * spy_growth_float + 0.4 * ief_growth_float)
        pressure_float = 10_000.0 * (0.4 - bond_weight_float)
        bucket_float = causal_bucket_float(pressure_float, prior_pressure_list)
        half_spread_float = float(endpoint_df.iloc[2]["SPY"] / base_ser["SPY"] - endpoint_df.iloc[2]["TLT"] / base_ser["TLT"])
        next_month_idx = session_idx[month_period_idx == month_period + 1]
        month_row_list.append({
            "month_period": str(month_period), "prev_eom_date": previous_close_ts, "measure_date": measure_ts,
            "final_fill_date": month_session_idx[-6], "switch_close_date": month_session_idx[-1], "next_first_date": next_month_idx[0],
            "exit_fill_date": next_month_idx[4], "pressure_bps_float": pressure_float, "bucket_float": bucket_float,
            "half_spread_float": half_spread_float, "spy_vote_count_int": spy_vote_count_int(bucket_float, pressure_float, half_spread_float),
        })
        prior_pressure_list.append(pressure_float)
    return pd.DataFrame(month_row_list)


def build_order_schedule_df(month_table_df: pd.DataFrame, config_obj: MonthEndVoteFlowConfig) -> pd.DataFrame:
    """Target weights by fill session. `open_late` ends the final week at the first open of the new month."""
    switch_column_str = "next_first_date" if config_obj.fill_str == "open_late" else "switch_close_date"
    order_row_list: list[dict] = []
    for month_row in month_table_df.itertuples(index=False):
        final_tuple = final_weight_tuple(month_row.bucket_float, month_row.pressure_bps_float, month_row.half_spread_float)
        early_tuple = early_weight_tuple(config_obj.early_rule_str, month_row.bucket_float)
        for leg_str, fill_ts, weight_tuple in (("final", month_row.final_fill_date, final_tuple), ("early", getattr(month_row, switch_column_str), early_tuple),
                                               ("exit", month_row.exit_fill_date, FLAT_WEIGHT_TUPLE)):
            order_row_list.append({"fill_date": fill_ts, "measure_date": month_row.measure_date, "leg_str": leg_str, "SPY": weight_tuple[0], "TLT": weight_tuple[1]})
    return pd.DataFrame(order_row_list).set_index("fill_date").sort_index()


FLOW_SOURCE_TUPLE = (
    ("SPY", "SPY", "CAPITALSPECIAL"), ("TLT", "TLT", "CAPITALSPECIAL"),
    ("SPY", "FLOW_TR_SPY", "TOTALRETURN"), ("IEF", "FLOW_TR_IEF", "TOTALRETURN"), ("TLT", "FLOW_TR_TLT", "TOTALRETURN"),
    ("$SPXTR", "$SPX", "TOTALRETURN"),
)


def _flow_frame_df(source_price_df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """One frame from the six source series, in FLOW_SOURCE_TUPLE order: the same steps for direct Norgate and for the snapshot."""
    price_frame_list: list[pd.DataFrame] = []
    adjustment_dict: dict[str, str] = {}
    for (symbol_str, namespace_str, adjustment_str), price_df in zip(FLOW_SOURCE_TUPLE, source_price_df_list, strict=True):
        if price_df is None or price_df.empty:
            raise RuntimeError(f"No observed Norgate prices for {symbol_str}.")
        if namespace_str.startswith("FLOW_TR_"):
            price_df = price_df.loc[:, ["Close"]].copy()
        price_df.columns = pd.MultiIndex.from_tuples([(namespace_str, field_str) for field_str in price_df.columns])
        price_frame_list.append(price_df)
        adjustment_dict[namespace_str] = adjustment_str
    pricing_data_df = pd.concat(price_frame_list, axis=1).sort_index()
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = adjustment_dict
    pricing_data_df.attrs["benchmark_data_symbol_dict"] = {"$SPX": "$SPXTR"}
    pricing_data_df.attrs["price_padding_policy_str"] = "NONE"
    return pricing_data_df


def _get_month_end_flow_snapshot_data(config_obj: MonthEndVoteFlowConfig) -> pd.DataFrame:
    """The same six series from the qualified Flow Vote snapshot: observed rows, native fields and dtypes (data/flow_vote_snapshot_contract.py)."""
    from data.flow_vote_snapshot_contract import FLOW_VOTE_PROFILE_STR
    from data.norgate_loader import load_price_timeseries
    from data.norgate_snapshot_store import NorgateSnapshotValidationError, load_valid_snapshot_manifest, use_norgate_data_profile

    with use_norgate_data_profile(FLOW_VOTE_PROFILE_STR):
        initial_manifest_obj = load_valid_snapshot_manifest(FLOW_VOTE_PROFILE_STR)
        source_price_df_list = [
            load_price_timeseries(symbol_str, adjustment_str=adjustment_str, start_date_str=HISTORY_START_DATE_STR,
                                  end_date_str=config_obj.end_date_str, data_profile_str=FLOW_VOTE_PROFILE_STR)
            for symbol_str, _, adjustment_str in FLOW_SOURCE_TUPLE
        ]
        final_manifest_obj = load_valid_snapshot_manifest(FLOW_VOTE_PROFILE_STR)
    # *** CRITICAL*** A publication between the six reads must not mix two data vintages in one decision.
    if (initial_manifest_obj.snapshot_dir_path_obj != final_manifest_obj.snapshot_dir_path_obj
            or initial_manifest_obj.manifest_hash_str != final_manifest_obj.manifest_hash_str):
        raise NorgateSnapshotValidationError("Flow Vote snapshot changed during the data load; retry with one complete snapshot.")
    return _flow_frame_df(source_price_df_list)


def get_month_end_flow_data(config_obj: MonthEndVoteFlowConfig) -> pd.DataFrame:
    if is_snapshot_mode_enabled_bool():
        return _get_month_end_flow_snapshot_data(config_obj)
    # *** CRITICAL*** Do not use the shared ALLMARKETDAYS padded loader: a synthetic close must not become a signal endpoint or a fill.
    return _flow_frame_df([
        norgatedata.price_timeseries(
            symbol_str, stock_price_adjustment_setting=getattr(norgatedata.StockPriceAdjustmentType, adjustment_str),
            padding_setting=norgatedata.PaddingType.NONE, start_date=HISTORY_START_DATE_STR, end_date=config_obj.end_date_str,
            timeseriesformat="pandas-dataframe",
        )
        for symbol_str, _, adjustment_str in FLOW_SOURCE_TUPLE
    ])


class MonthEndVoteFlowStrategy(MonthEndRebalancingFlowStrategy):
    """The parent's prior-close share orders, borrow ledger and close-and-reopen; a new schedule and a choice of fill."""

    def __init__(self, config_obj: MonthEndVoteFlowConfig = DEFAULT_CONFIG):
        super().__init__(config_obj)
        self.name = STRATEGY_NAME_STR
        self._accounting_policy_dict["execution_policy_str"] = "MOC" if config_obj.fill_str == "close" else "MOO"

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        if pricing_data_df.attrs.get("price_padding_policy_str") != "NONE":
            raise ValueError("Observed-price provenance requires price_padding_policy_str=NONE.")
        expected_adjustment_dict = {**{a: "CAPITALSPECIAL" for a in TRADED_ASSET_TUPLE}, **{f"FLOW_TR_{a}": "TOTALRETURN" for a in SIGNAL_ASSET_TUPLE}}
        actual_adjustment_dict = pricing_data_df.attrs.get("norgate_adjustment_by_symbol_dict", {})
        for asset_str, adjustment_str in expected_adjustment_dict.items():
            if actual_adjustment_dict.get(asset_str) != adjustment_str:
                raise ValueError(f"Adjustment provenance requires {asset_str}={adjustment_str}.")
        total_return_close_df = pd.DataFrame({a: pricing_data_df[(f"FLOW_TR_{a}", "Close")] for a in SIGNAL_ASSET_TUPLE})
        month_table_df = build_month_table_df(total_return_close_df)
        order_schedule_df = build_order_schedule_df(month_table_df, self.config_obj)
        session_idx = exchange_session_idx(pricing_data_df.index[-1])
        feature_df = pd.DataFrame(0.0, index=pricing_data_df.index, columns=["rebalance_bool", "SPY", "TLT"])
        for fill_ts, order_row in order_schedule_df.iterrows():
            # *** CRITICAL*** Map a calendar fill date to the PRIOR session: iterate sees only that prior close, for an open or a close fill alike.
            decision_ts = session_idx[session_idx.get_loc(fill_ts) - 1]
            if decision_ts not in feature_df.index:
                continue
            if order_row["measure_date"] > decision_ts:
                raise RuntimeError("Signal is not known a full session before its fill.")
            feature_df.loc[decision_ts] = [1.0, order_row["SPY"], order_row["TLT"]]
        feature_df.columns = pd.MultiIndex.from_tuples([(SIGNAL_NAMESPACE_STR, field_str) for field_str in feature_df.columns])
        if len(month_table_df) >= len(self.month_table_df):
            self.month_table_df = month_table_df
            self.order_schedule_df = order_schedule_df
            self.session_idx = session_idx
        return pd.concat([pricing_data_df, feature_df], axis=1)

    def process_orders(self, prices: pd.DataFrame) -> None:
        if self.config_obj.fill_str == "close":
            super().process_orders(prices)
            return
        # Open fills are the engine's own contract. The missing-session guard and the borrow ledger stay the parent's.
        if self.previous_bar != self.session_idx[self.session_idx.get_loc(self.current_bar) - 1]:
            raise RuntimeError("Missing XNYS session; refuse to move a fill silently.")
        Strategy.process_orders(self, prices)
        self.apply_post_mark_accounting(prices)


class MonthEndVoteFlowTimingStrategy(MonthEndVoteFlowStrategy):
    """Keep prior-close sizing intact when the Timing analyzer posts today's dividends early, as the parent module's timing class does."""

    def __init__(self, config_obj: MonthEndVoteFlowConfig):
        super().__init__(config_obj)
        self.prior_close_nav_float = float(config_obj.capital_base_float)

    @property
    def previous_total_value(self) -> float:
        # *** CRITICAL*** Timing credits current ex-date dividends before iterate(), unlike Vanilla. Size on the last completed close's NAV.
        return self.prior_close_nav_float

    def apply_post_mark_accounting(self, prices: pd.DataFrame) -> None:
        super().apply_post_mark_accounting(prices)
        self.prior_close_nav_float = float(self.total_value)


def _build_analysis_context_dict(config_obj: MonthEndVoteFlowConfig = DEFAULT_CONFIG, pricing_data_df: pd.DataFrame | None = None) -> dict[str, object]:
    if pricing_data_df is None:
        pricing_data_df = get_month_end_flow_data(config_obj)
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    if len(calendar_idx) < 2:
        raise ValueError("At least two scoring sessions are required.")
    # *** CRITICAL*** The timing engine bypasses the local fill adapter. Validate the path before it can substitute a stale close or cancel a
    # missing fill. This is data validation, never a signal selection filter.
    expected_idx = exchange_session_idx(calendar_idx[-1])
    expected_idx = expected_idx[(expected_idx >= calendar_idx[0]) & (expected_idx <= calendar_idx[-1])]
    if not calendar_idx.equals(expected_idx):
        raise ValueError("Analysis data is missing an XNYS scoring session.")
    for asset_str in TRADED_ASSET_TUPLE:
        for field_str in ("Open", "Close", "Dividend"):
            price_ser = pricing_data_df.loc[calendar_idx, (asset_str, field_str)].astype(float)
            if not np.isfinite(price_ser).all() or (field_str != "Dividend" and (price_ser <= 0).any()):
                raise ValueError(f"Invalid analysis {field_str} for {asset_str}.")
    return {"strategy_name_str": STRATEGY_NAME_STR, "capital_base_float": config_obj.capital_base_float, "config_obj": config_obj,
            "pricing_data_df": pricing_data_df, "calendar_idx": calendar_idx}


@lru_cache(maxsize=2)
def _capacity_pricing_df(end_date_str: str | None) -> pd.DataFrame:
    """Freeze one input frame per endpoint across the AUM grid in this process."""
    return get_month_end_flow_data(replace(DEFAULT_CONFIG, end_date_str=end_date_str))


def build_capacity_analysis_inputs(capital_base_float: float, show_display_bool: bool = False, backtest_start_date_str: str | None = None, end_date_str: str | None = None) -> dict[str, object]:
    config_obj = replace(DEFAULT_CONFIG, capital_base_float=capital_base_float, end_date_str=end_date_str,
                         backtest_start_date_str=backtest_start_date_str or DEFAULT_CONFIG.backtest_start_date_str)
    context_dict = _build_analysis_context_dict(config_obj, _capacity_pricing_df(end_date_str).copy(deep=True))
    strategy_obj = MonthEndVoteFlowStrategy(config_obj)
    run_daily(strategy_obj, context_dict["pricing_data_df"], calendar=context_dict["calendar_idx"], show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    return {"strategy_obj": strategy_obj, "pricing_data_df": context_dict["pricing_data_df"], "execution_policy_str": "MOC" if config_obj.fill_str == "close" else "MOO"}


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    context_dict = _build_analysis_context_dict()

    def strategy_factory_fn() -> MonthEndVoteFlowTimingStrategy:
        strategy_obj = MonthEndVoteFlowTimingStrategy(context_dict["config_obj"])
        # The analyzer computes features on a separate base object. Each fresh cell also needs its own calendar and report state.
        strategy_obj.compute_signals(context_dict["pricing_data_df"])
        return strategy_obj

    default_timing_str = "same_close_moc" if DEFAULT_CONFIG.fill_str == "close" else "same_open"
    return {"strategy_factory_fn": strategy_factory_fn, "pricing_data_df": context_dict["pricing_data_df"], "calendar_idx": context_dict["calendar_idx"],
            "order_generation_mode_str": "vanilla_current_bar", "risk_model_str": "taa_rebalance",
            "entry_timing_str_tuple": ("same_open", "same_close_moc", "next_open", "next_close"), "exit_timing_str_tuple": ("same_open", "same_close_moc", "next_open", "next_close"),
            "default_entry_timing_str": default_timing_str, "default_exit_timing_str": default_timing_str}


def build_stress_test_context_dict() -> dict[str, object]:
    return _build_analysis_context_dict()


def build_stress_test_strategy_obj(context_dict: dict[str, object]) -> MonthEndVoteFlowStrategy:
    return MonthEndVoteFlowStrategy(context_dict["config_obj"])


def run_variant(show_display_bool: bool = False, save_results_bool: bool = True, output_dir_str: str = "results",
                backtest_start_date_str: str | None = None, end_date_str: str | None = None,
                capital_base_float: float = DEFAULT_CONFIG.capital_base_float, config_obj: MonthEndVoteFlowConfig | None = None) -> MonthEndVoteFlowStrategy:
    config_obj = replace(config_obj or DEFAULT_CONFIG, capital_base_float=capital_base_float, end_date_str=end_date_str,
                         backtest_start_date_str=backtest_start_date_str or DEFAULT_CONFIG.backtest_start_date_str)
    pricing_data_df = get_month_end_flow_data(config_obj)
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    if len(calendar_idx) < 2:
        raise ValueError("At least two scoring sessions are required.")
    strategy_obj = MonthEndVoteFlowStrategy(config_obj)
    run_daily(strategy_obj, pricing_data_df, calendar=calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    if show_display_bool:
        print(strategy_obj.summary.to_string())
    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        strategy_obj.output_path_str = str(output_path.resolve())
        strategy_obj.month_table_df.to_csv(output_path / "month_table.csv", index=False)
        strategy_obj.order_schedule_df.to_csv(output_path / "order_schedule.csv")
    return strategy_obj


if __name__ == "__main__":
    run_variant(show_display_bool=True)
