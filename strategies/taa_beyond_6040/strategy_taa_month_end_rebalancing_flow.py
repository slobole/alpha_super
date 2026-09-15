"""Month-End Rebalancing Flow: frozen C_main calendar strategy on SPY/TLT.

Measure the 60/40 SPY/IEF drift at dtme=7 close, enter the final five
sessions at dtme=6 MOC, reverse at month-end MOC, exit at session 5 MOC.
The expanding quintile uses strictly prior months since August 2002.

Research/BENCH only. A local execution adapter preserves causal MOC fills.
Whole shares are sized from the previous close/NAV and held between auctions;
the source's daily constant-weight TOTALRETURN ledger is a separate reference.
See docs/research/month_end_rebalancing_flow.md for the accounting bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import is_snapshot_mode_enabled_bool, norgatedata


STRATEGY_NAME_STR = "strategy_taa_month_end_rebalancing_flow"
TRADED_ASSET_TUPLE = ("SPY", "TLT")
SIGNAL_ASSET_TUPLE = ("SPY", "IEF")
SIGNAL_NAMESPACE_STR = "MonthEndFlow"
HISTORY_START_DATE_STR = "2002-07-26"
ANNUAL_BORROW_RATE_FLOAT = 0.01


@dataclass(frozen=True)
class MonthEndRebalancingFlowConfig:
    backtest_start_date_str: str = "2003-01-02"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.capital_base_float) or self.capital_base_float <= 0:
            raise ValueError("capital_base_float must be positive and finite.")
        if pd.Timestamp(self.backtest_start_date_str) < pd.Timestamp("2003-01-01"):
            raise ValueError("Scoring begins in January 2003 or later.")


DEFAULT_CONFIG = MonthEndRebalancingFlowConfig()


def exchange_session_idx(end_date_ts: pd.Timestamp) -> pd.DatetimeIndex:
    """Include the complete next month; truncated prices never define month-end."""
    calendar_end_ts = (pd.Timestamp(end_date_ts).to_period("M") + 1).end_time.normalize()
    calendar_obj = xcals.get_calendar(
        "XNYS", start="2002-07-01", end=calendar_end_ts.strftime("%Y-%m-%d")
    )
    return calendar_obj.sessions.tz_localize(None)


def causal_bucket_float(pressure_float: float, prior_pressure_list: list[float]) -> float:
    if len(prior_pressure_list) < 24:
        return float("nan")
    # *** CRITICAL*** Prior months only, before appending the current pressure.
    # F_m = count(P_prior <= P_m) / n_prior; q_m = min(5, floor(5 F_m) + 1).
    cdf_float = float(np.mean(np.asarray(prior_pressure_list) <= pressure_float))
    return float(min(5, int(np.floor(5.0 * cdf_float)) + 1))


def target_weight_tuple(bucket_float: float, leg_str: str) -> tuple[float, float]:
    if leg_str == "final":
        return (1.0, 0.0) if bucket_float == 1 else (0.0, 1.0)
    if leg_str == "early":
        if bucket_float == 1:
            return (0.0, 0.0)
        return (0.5, -0.5) if bucket_float in (4, 5) else (0.0, -1.0)
    if leg_str == "exit":
        return (0.0, 0.0)
    raise ValueError(f"Unknown leg: {leg_str}")


def build_month_table_df(total_return_close_df: pd.DataFrame) -> pd.DataFrame:
    """Use observed TR endpoints, with no filling or full-sample thresholds."""
    price_idx = total_return_close_df.index
    if not isinstance(price_idx, pd.DatetimeIndex) or len(price_idx) == 0:
        raise ValueError("A nonempty DatetimeIndex is required.")
    if not price_idx.is_monotonic_increasing or not price_idx.is_unique:
        raise ValueError("Signal dates must be unique and increasing.")
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
        # *** CRITICAL*** These offsets index the exchange calendar, never a
        # price-truncated month. Half-days count; historical closures use XNYS.
        measure_ts = month_session_idx[-7]
        if measure_ts > price_idx[-1]:
            continue
        previous_close_ts = session_idx[month_period_idx == month_period - 1][-1]
        endpoint_df = total_return_close_df.reindex(
            index=[previous_close_ts, measure_ts], columns=list(SIGNAL_ASSET_TUPLE)
        ).astype(float)
        if not np.isfinite(endpoint_df.to_numpy()).all() or (endpoint_df <= 0).any().any():
            raise ValueError(f"Missing/invalid TOTALRETURN signal endpoint in {month_period}.")
        spy_growth_float = float(endpoint_df.loc[measure_ts, "SPY"] / endpoint_df.loc[previous_close_ts, "SPY"])
        ief_growth_float = float(endpoint_df.loc[measure_ts, "IEF"] / endpoint_df.loc[previous_close_ts, "IEF"])
        # wB' = .4*(1+R_IEF) / [.6*(1+R_SPY)+.4*(1+R_IEF)]
        # pressure_bps = 10000*(.4-wB'). Positive means buy bonds.
        bond_weight_float = 0.4 * ief_growth_float / (0.6 * spy_growth_float + 0.4 * ief_growth_float)
        pressure_float = 10_000.0 * (0.4 - bond_weight_float)
        bucket_float = causal_bucket_float(pressure_float, prior_pressure_list)
        next_month_idx = session_idx[month_period_idx == month_period + 1]
        month_row_list.append({
            "month_period": str(month_period),
            "prev_eom_date": previous_close_ts,
            "measure_date": measure_ts,
            "final_fill_date": month_session_idx[-6],
            "early_fill_date": month_session_idx[-1],
            "exit_fill_date": next_month_idx[4],
            "pressure_ief_measure_bps_float": pressure_float,
            "bucket_ief_measure_causal_int": bucket_float,
            "prior_month_count_int": len(prior_pressure_list),
        })
        prior_pressure_list.append(pressure_float)
    return pd.DataFrame(month_row_list, columns=[
        "month_period", "prev_eom_date", "measure_date", "final_fill_date",
        "early_fill_date", "exit_fill_date", "pressure_ief_measure_bps_float",
        "bucket_ief_measure_causal_int", "prior_month_count_int",
    ])


def build_order_schedule_df(month_table_df: pd.DataFrame) -> pd.DataFrame:
    order_row_list: list[dict] = []
    for month_row in month_table_df.itertuples(index=False):
        for leg_str in ("final", "early", "exit"):
            spy_weight_float, tlt_weight_float = target_weight_tuple(
                month_row.bucket_ief_measure_causal_int, leg_str
            )
            order_row_list.append({
                "fill_date": getattr(month_row, f"{leg_str}_fill_date"),
                "measure_date": month_row.measure_date,
                "leg_str": leg_str,
                "SPY": spy_weight_float,
                "TLT": tlt_weight_float,
            })
    return pd.DataFrame(order_row_list, columns=[
        "fill_date", "measure_date", "leg_str", "SPY", "TLT",
    ]).set_index("fill_date").sort_index()


def build_reference_hold_weights_df(
    month_table_df: pd.DataFrame, hold_session_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Source fixture convention: exposure from previous close to this close."""
    hold_weight_df = pd.DataFrame(0.0, index=hold_session_idx, columns=TRADED_ASSET_TUPLE)
    for month_row in month_table_df.itertuples(index=False):
        for leg_str, entry_ts, exit_ts in (
            ("final", month_row.final_fill_date, month_row.early_fill_date),
            ("early", month_row.early_fill_date, month_row.exit_fill_date),
        ):
            # *** CRITICAL*** Entry-close return is excluded; exit-close return
            # is included. Neither interval uses a future price for the signal.
            held_bool_vec = (hold_session_idx > entry_ts) & (hold_session_idx <= exit_ts)
            hold_weight_df.loc[held_bool_vec, :] = target_weight_tuple(
                month_row.bucket_ief_measure_causal_int, leg_str
            )
    return hold_weight_df


def get_month_end_flow_data(config_obj: MonthEndRebalancingFlowConfig) -> pd.DataFrame:
    if is_snapshot_mode_enabled_bool():
        raise RuntimeError("This research loader requires direct Norgate with PaddingType.NONE; snapshot observation provenance is not supported.")
    price_frame_list: list[pd.DataFrame] = []
    adjustment_dict: dict[str, str] = {}
    for symbol_str, namespace_str, adjustment_str in (
        ("SPY", "SPY", "CAPITALSPECIAL"), ("TLT", "TLT", "CAPITALSPECIAL"),
        ("SPY", "FLOW_TR_SPY", "TOTALRETURN"), ("IEF", "FLOW_TR_IEF", "TOTALRETURN"),
        ("$SPXTR", "$SPX", "TOTALRETURN"),
    ):
        # *** CRITICAL*** Do not use the shared ALLMARKETDAYS padded loader:
        # finite synthetic closes must not become signal endpoints or MOC fills.
        price_df = norgatedata.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=getattr(norgatedata.StockPriceAdjustmentType, adjustment_str),
            padding_setting=norgatedata.PaddingType.NONE,
            start_date=HISTORY_START_DATE_STR, end_date=config_obj.end_date_str,
            timeseriesformat="pandas-dataframe",
        )
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


class MonthEndRebalancingFlowStrategy(Strategy):
    """Causal prior-close share orders, executed in the following close auction."""

    def __init__(self, config_obj: MonthEndRebalancingFlowConfig = DEFAULT_CONFIG):
        super().__init__(
            name=STRATEGY_NAME_STR, benchmarks=["$SPX"],
            capital_base=config_obj.capital_base_float,
            performance_benchmark_symbol_str="$SPX",
            performance_benchmark_adjustment_str="TOTALRETURN",
        )
        self.config_obj = config_obj
        self.asset_list = list(TRADED_ASSET_TUPLE)
        self.month_table_df = pd.DataFrame()
        self.order_schedule_df = pd.DataFrame()
        self.borrow_fee_row_list: list[dict] = []
        self.decision_row_list: list[dict] = []
        self.position_row_list: list[dict] = []
        self.trade_id_int = 0
        self.trade_id_by_asset_dict: dict[str, int] = {}
        self.borrow_fee_total_float = 0.0
        self.run_calendar_idx = pd.DatetimeIndex([])
        self.configure_dividend_cash_ledger(enabled_bool=True)
        self._data_adjustment_policy_dict.update({
            "signal_adjustment_str": "TOTALRETURN",
            "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
            "performance_benchmark_adjustment_str": "TOTALRETURN",
            "price_padding_policy_str": "NONE",
        })
        self._accounting_policy_dict.update({
            "execution_policy_str": "MOC",
            "sizing_policy_str": "previous_close_NAV_and_prices_whole_shares",
            "performance_parity_status_str": "source_targets_only_not_return_parity",
            "annual_short_borrow_rate_float": ANNUAL_BORROW_RATE_FLOAT,
            "borrow_collateral_policy_str": "ceil_102pct_close_ACT_360_to_next_session",
            "short_proceeds_policy_str": "not_reinvested_zero_interest",
            "gross_exposure_policy_str": "100pct_target_realized_weights_can_drift",
            "initial_position_policy_str": "flat_until_first_scheduled_auction",
            "calendar_policy_str": "XNYS_historical_closures_not_notice_time_replay",
            "trade_statistics_policy_str": "trade_PnL_excludes_cash_dividends_and_borrow_NAV_includes_both",
        })

    def configure_run_calendar(self, calendar_idx: pd.DatetimeIndex) -> None:
        self.run_calendar_idx = pd.DatetimeIndex(calendar_idx).copy()
        if len(self.run_calendar_idx) < 2:
            raise ValueError("At least two scoring sessions are required.")

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        if pricing_data_df.attrs.get("price_padding_policy_str") != "NONE":
            raise ValueError("Observed-price provenance requires price_padding_policy_str=NONE.")
        expected_adjustment_dict = {
            **{asset_str: "CAPITALSPECIAL" for asset_str in TRADED_ASSET_TUPLE},
            **{f"FLOW_TR_{asset_str}": "TOTALRETURN" for asset_str in SIGNAL_ASSET_TUPLE},
        }
        actual_adjustment_dict = pricing_data_df.attrs.get("norgate_adjustment_by_symbol_dict", {})
        for asset_str, adjustment_str in expected_adjustment_dict.items():
            if actual_adjustment_dict.get(asset_str) != adjustment_str:
                raise ValueError(f"Adjustment provenance requires {asset_str}={adjustment_str}.")
        total_return_close_df = pd.DataFrame({
            asset_str: pricing_data_df[(f"FLOW_TR_{asset_str}", "Close")]
            for asset_str in SIGNAL_ASSET_TUPLE
        })
        month_table_df = build_month_table_df(total_return_close_df)
        order_schedule_df = build_order_schedule_df(month_table_df)
        session_idx = exchange_session_idx(pricing_data_df.index[-1])
        feature_df = pd.DataFrame(0.0, index=pricing_data_df.index, columns=["rebalance_bool", "SPY", "TLT"])
        for fill_ts, order_row in order_schedule_df.iterrows():
            # *** CRITICAL*** Map a calendar fill date to the PRIOR session.
            # iterate receives only that prior close; today's OHLC is ignored.
            decision_ts = session_idx[session_idx.get_loc(fill_ts) - 1]
            if decision_ts not in feature_df.index:
                continue
            if order_row["measure_date"] > decision_ts:
                raise RuntimeError("MOC signal is not known a full session before fill.")
            feature_df.loc[decision_ts] = [1.0, order_row["SPY"], order_row["TLT"]]
        feature_df.columns = pd.MultiIndex.from_tuples(
            [(SIGNAL_NAMESPACE_STR, field_str) for field_str in feature_df.columns]
        )
        if len(month_table_df) >= len(self.month_table_df):
            self.month_table_df = month_table_df
            self.order_schedule_df = order_schedule_df
            self.session_idx = session_idx
        return pd.concat([pricing_data_df, feature_df], axis=1)

    def iterate(self, data_df, close_row_ser, open_price_ser) -> None:
        if close_row_ser is None or not close_row_ser[(SIGNAL_NAMESPACE_STR, "rebalance_bool")]:
            return
        if self.get_orders():
            raise RuntimeError("A previous MOC order remains unresolved.")
        nav_float = float(self.previous_total_value)
        if not np.isfinite(nav_float) or nav_float <= 0:
            raise RuntimeError("MOC sizing requires positive prior-close NAV.")
        target_share_dict: dict[str, int] = {}
        for asset_str in TRADED_ASSET_TUPLE:
            close_float = float(close_row_ser[(asset_str, "Close")])
            if not np.isfinite(close_float) or close_float <= 0:
                raise RuntimeError(f"Invalid prior-close sizing price for {asset_str}.")
            weight_float = float(close_row_ser[(SIGNAL_NAMESPACE_STR, asset_str)])
            # q_i = trunc(w_i * NAV_previous_close / price_i_previous_close).
            target_share_dict[asset_str] = int(np.trunc(weight_float * nav_float / close_float))
        # Close and reopen opposing trades in the SAME auction, with separate
        # IDs so engine trade analytics remain correct. Each order pays fees.
        for asset_str in TRADED_ASSET_TUPLE:
            held_share_float = self.get_position(asset_str)
            if held_share_float:
                self.order(asset_str, -held_share_float, trade_id=self.trade_id_by_asset_dict[asset_str])
                self.get_orders()[-1].timing_order_kind_str = "exit"
        for asset_str, share_int in target_share_dict.items():
            if share_int:
                self.trade_id_int += 1
                self.trade_id_by_asset_dict[asset_str] = self.trade_id_int
                self.order(asset_str, share_int, trade_id=self.trade_id_int)
                self.get_orders()[-1].timing_order_kind_str = "entry"
        self.decision_row_list.append({
            "decision_date": self.previous_bar, "fill_date": self.current_bar,
            "sizing_nav_float": nav_float, **target_share_dict,
        })

    def process_orders(self, prices: pd.DataFrame) -> None:
        for order_obj in self.get_orders():
            if not isinstance(order_obj, MarketOrder) or order_obj.asset not in TRADED_ASSET_TUPLE:
                raise RuntimeError("The local MOC adapter supports SPY/TLT market orders only.")
        # Fail before cash/positions mutate: never invoke the engine's stale-
        # price liquidation fallback for a missing closing auction or session.
        if self.previous_bar != self.session_idx[self.session_idx.get_loc(self.current_bar) - 1]:
            raise RuntimeError("Missing XNYS session; refuse to move a MOC fill silently.")
        for asset_str in TRADED_ASSET_TUPLE:
            close_float = float(prices.loc[self.current_bar, (asset_str, "Close")])
            if not np.isfinite(close_float) or close_float <= 0:
                raise RuntimeError(f"Missing valid MOC close for {asset_str}.")
            if (asset_str, "Dividend") not in prices.columns:
                raise RuntimeError(f"Required gross Dividend field missing for {asset_str}.")
        # *** CRITICAL*** Execution-only adapter, AFTER fixed share orders are
        # formed from the previous close. Substitute auction prices in a COPY;
        # no fill-session close enters features or sizing, and source OHLC stay
        # unchanged. Dividends use pre-fill positions and actual entitlement T.
        execution_df = prices.copy()
        for asset_str in TRADED_ASSET_TUPLE:
            execution_df.loc[self.current_bar, (asset_str, "Open")] = prices.loc[self.current_bar, (asset_str, "Close")]
        super().process_orders(execution_df)
        self.apply_post_mark_accounting(prices)

    def apply_post_mark_accounting(self, prices: pd.DataFrame) -> None:
        if len(self.run_calendar_idx) == 0:
            raise RuntimeError("Configure the scoring calendar before borrow accounting.")
        if self.current_bar not in self.run_calendar_idx:
            return  # Timing visits a pre-start bar without holding a position.
        self.position_row_list.append({
            "date": self.current_bar,
            **{asset_str: self.get_position(asset_str) for asset_str in TRADED_ASSET_TUPLE},
        })

        held_share_float = float(self.get_position("TLT"))
        if held_share_float >= 0:
            return
        if self.current_bar == self.run_calendar_idx[-1]:
            return  # Do not prepay an interval beyond the requested NAV endpoint.
        next_session_ts = self.session_idx[self.session_idx.get_loc(self.current_bar) + 1]
        calendar_days_int = (next_session_ts - self.current_bar).days
        close_float = float(prices.loc[self.current_bar, ("TLT", "Close")])
        collateral_float = abs(held_share_float) * float(np.ceil(1.02 * close_float))
        # *** CRITICAL*** Post-MOC holdings owe the next overnight interval,
        # including weekends/holidays. A MOC cover owes no subsequent interval.
        # fee = abs(q_short) * ceil(1.02*Close) * .01 * calendar_days / 360.
        fee_float = collateral_float * ANNUAL_BORROW_RATE_FLOAT * calendar_days_int / 360.0
        self.cash -= fee_float
        self.total_value -= fee_float
        self.borrow_fee_total_float += fee_float
        self._accounting_policy_dict["borrow_fee_total_float"] = self.borrow_fee_total_float
        self.borrow_fee_row_list.append({
            "accrual_start_date": self.current_bar, "next_session_date": next_session_ts,
            "calendar_day_count_int": calendar_days_int, "short_share_float": held_share_float,
            "collateral_value_float": collateral_float, "borrow_fee_float": fee_float,
        })

    def finalize(self, current_data_df: pd.DataFrame) -> None:
        self.borrow_fee_df = pd.DataFrame(self.borrow_fee_row_list)
        self.decision_df = pd.DataFrame(self.decision_row_list)
        self.held_share_df = pd.DataFrame(self.position_row_list, columns=["date", *TRADED_ASSET_TUPLE]).set_index("date")

    def summarize(self, include_benchmarks=True):
        # Timing constructs completed results directly and does not call finalize.
        self.finalize(pd.DataFrame())
        super().summarize(include_benchmarks=include_benchmarks)


class MonthEndRebalancingFlowTimingStrategy(MonthEndRebalancingFlowStrategy):
    """Keep prior-close sizing intact when Timing posts today's dividends early."""

    def __init__(self, config_obj: MonthEndRebalancingFlowConfig):
        super().__init__(config_obj)
        self.prior_close_nav_float = float(config_obj.capital_base_float)

    @property
    def previous_total_value(self) -> float:
        # *** CRITICAL*** Timing credits current ex-date dividends before
        # iterate(), unlike Vanilla. Size using the last completed close's NAV,
        # excluding today's cash posting. Actual cash still receives the dividend.
        return self.prior_close_nav_float

    def apply_post_mark_accounting(self, prices: pd.DataFrame) -> None:
        super().apply_post_mark_accounting(prices)
        self.prior_close_nav_float = float(self.total_value)


def _build_analysis_context_dict(
    config_obj: MonthEndRebalancingFlowConfig = DEFAULT_CONFIG,
    pricing_data_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    if pricing_data_df is None:
        pricing_data_df = get_month_end_flow_data(config_obj)
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    if len(calendar_idx) < 2:
        raise ValueError("At least two scoring sessions are required.")
    # *** CRITICAL*** The timing engine bypasses the local MOC adapter. Validate
    # the requested path before it can substitute a stale close or cancel a
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
    return {
        "strategy_name_str": STRATEGY_NAME_STR,
        "capital_base_float": config_obj.capital_base_float,
        "config_obj": config_obj,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": calendar_idx,
    }


@lru_cache(maxsize=2)
def _capacity_pricing_df(end_date_str: str | None) -> pd.DataFrame:
    """Freeze one input frame per endpoint across the AUM grid in this process."""
    return get_month_end_flow_data(replace(DEFAULT_CONFIG, end_date_str=end_date_str))


def build_capacity_analysis_inputs(
    capital_base_float: float,
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = replace(
        DEFAULT_CONFIG, capital_base_float=capital_base_float,
        backtest_start_date_str=backtest_start_date_str or DEFAULT_CONFIG.backtest_start_date_str,
        end_date_str=end_date_str,
    )
    context_dict = _build_analysis_context_dict(config_obj, _capacity_pricing_df(end_date_str).copy(deep=True))
    strategy_obj = MonthEndRebalancingFlowStrategy(config_obj)
    run_daily(strategy_obj, context_dict["pricing_data_df"], calendar=context_dict["calendar_idx"],
              show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": context_dict["pricing_data_df"],
        "execution_policy_str": "MOC",
    }


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    context_dict = _build_analysis_context_dict()

    def strategy_factory_fn() -> MonthEndRebalancingFlowTimingStrategy:
        strategy_obj = MonthEndRebalancingFlowTimingStrategy(context_dict["config_obj"])
        # The analyzer computes features on a separate base object. Each fresh
        # cell also needs its own calendar/report state before post-mark accounting.
        strategy_obj.compute_signals(context_dict["pricing_data_df"])
        return strategy_obj

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": context_dict["pricing_data_df"],
        "calendar_idx": context_dict["calendar_idx"],
        "order_generation_mode_str": "vanilla_current_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_open", "same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_open", "same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "same_close_moc",
        "default_exit_timing_str": "same_close_moc",
    }


def build_stress_test_context_dict() -> dict[str, object]:
    return _build_analysis_context_dict()


def build_stress_test_strategy_obj(context_dict: dict[str, object]) -> MonthEndRebalancingFlowStrategy:
    return MonthEndRebalancingFlowStrategy(context_dict["config_obj"])


def run_variant(
    show_display_bool: bool = False, save_results_bool: bool = True,
    output_dir_str: str = "results", backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
) -> MonthEndRebalancingFlowStrategy:
    config_obj = replace(
        DEFAULT_CONFIG, capital_base_float=capital_base_float,
        backtest_start_date_str=backtest_start_date_str or DEFAULT_CONFIG.backtest_start_date_str,
        end_date_str=end_date_str,
    )
    pricing_data_df = get_month_end_flow_data(config_obj)
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    if len(calendar_idx) < 2:
        raise ValueError("At least two scoring sessions are required.")
    strategy_obj = MonthEndRebalancingFlowStrategy(config_obj)
    run_daily(strategy_obj, pricing_data_df, calendar=calendar_idx,
              show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    if show_display_bool:
        print(strategy_obj.summary.to_string())
    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        strategy_obj.output_path_str = str(output_path.resolve())
        strategy_obj.month_table_df.to_csv(output_path / "month_table.csv", index=False)
        strategy_obj.order_schedule_df.to_csv(output_path / "moc_schedule.csv")
        strategy_obj.decision_df.to_csv(output_path / "decisions.csv", index=False)
        strategy_obj.held_share_df.to_csv(output_path / "held_shares.csv")
        strategy_obj.borrow_fee_df.to_csv(output_path / "borrow_ledger.csv", index=False)
        input_path = output_path / "pricing_data.parquet"
        pricing_data_df.to_parquet(input_path)
        (output_path / "input_manifest.json").write_text(json.dumps({
            "pricing_data_sha256_str": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "strategy_source_sha256_str": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "observed_start_str": str(pricing_data_df.index[0].date()),
            "observed_end_str": str(pricing_data_df.index[-1].date()),
            "row_count_int": len(pricing_data_df),
            "data_provenance_dict": pricing_data_df.attrs,
        }, indent=2) + "\n", encoding="utf-8")
        build_reference_hold_weights_df(strategy_obj.month_table_df, calendar_idx).to_csv(
            output_path / "source_hold_targets.csv"
        )
        source_doc_path = Path(__file__).resolve().parents[2] / "docs/research/month_end_rebalancing_flow.md"
        (output_path / "implementation_contract.md").write_text(source_doc_path.read_text(encoding="utf-8"), encoding="utf-8")
    return strategy_obj


if __name__ == "__main__":
    run_variant(show_display_bool=True)
