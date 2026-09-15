"""QQQ Golden Cross: fixed SMA50/SMA200, long or cash, next-open execution.

SMA_n(T) = sum(Close_(T-i) for i in range(n)) / n; d_T = SMA50(T)-SMA200(T).
Flat: enter when d_(T-1) <= 0 < d_T. Long: exit when d_(T-1) >= 0 > d_T.
Equality holds the position. Start in cash and wait for a new crossing.
Target shares = floor(NAV_T / Close_T), filled at Open_(T+1) with costs.
This 100% target can overdraw cash on an opening gap; financing is not modeled.
See the adjacent Markdown contract for scope and accounting assumptions.
"""

from __future__ import annotations

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.plot import plot as plot_performance
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import is_snapshot_mode_enabled_bool, norgatedata


STRATEGY_NAME_STR = "strategy_mo_qqq_golden_cross"


def get_prices(history_start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """Load observed QQQ bars; snapshot observation provenance is not supported."""
    if is_snapshot_mode_enabled_bool():
        raise RuntimeError("QQQ Golden Cross requires direct Norgate with PaddingType.NONE.")
    # *** CRITICAL*** Synthetic padded closes must not enter SMA50/SMA200.
    # Signal and execution prices both use CAPITALSPECIAL; no dividend backfill.
    price_df = norgatedata.price_timeseries(
        "QQQ",
        stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
        padding_setting=norgatedata.PaddingType.NONE,
        start_date=history_start_date_str,
        end_date=end_date_str,
        timeseriesformat="pandas-dataframe",
    )
    if price_df is None or price_df.empty:
        raise ValueError("No observed QQQ prices returned.")
    pricing_data_df = price_df.copy()
    pricing_data_df.columns = pd.MultiIndex.from_tuples(
        [("QQQ", field_str) for field_str in pricing_data_df.columns]
    )
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {"QQQ": "CAPITALSPECIAL"}
    pricing_data_df.attrs["price_padding_policy_str"] = "NONE"
    return pricing_data_df


class QqqGoldenCrossStrategy(Strategy):
    """One fixed 50/200 crossing strategy; no daily resizing or parameter sweep."""

    enable_signal_audit = True

    def __init__(
        self,
        capital_base_float: float,
        slippage_float: float = 0.00025,
        commission_per_share_float: float = 0.005,
        commission_minimum_float: float = 1.0,
    ) -> None:
        if not np.isfinite(capital_base_float) or capital_base_float <= 0:
            raise ValueError("capital_base_float must be finite and positive.")
        for cost_float in (slippage_float, commission_per_share_float, commission_minimum_float):
            if not np.isfinite(cost_float) or cost_float < 0:
                raise ValueError("Trading costs must be finite and nonnegative.")
        super().__init__(
            name=STRATEGY_NAME_STR, benchmarks=[], capital_base=capital_base_float,
            slippage=slippage_float, commission_per_share=commission_per_share_float,
            commission_minimum=commission_minimum_float,
        )
        self.configure_dividend_cash_ledger(enabled_bool=True)
        self.trade_id_int = 0
        self.decision_start_ts: pd.Timestamp | None = None
        self._data_adjustment_policy_dict.update({
            "stock_signal_adjustment_str": "CAPITALSPECIAL",
            "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
        })

    def configure_run_calendar(self, calendar_idx: pd.DatetimeIndex) -> None:
        if calendar_idx.empty:
            raise ValueError("The execution calendar must not be empty.")
        self.decision_start_ts = pd.Timestamp(calendar_idx[0])

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        session_idx = pricing_data_df.index
        if (not isinstance(session_idx, pd.DatetimeIndex) or session_idx.empty
                or not session_idx.is_unique or not session_idx.is_monotonic_increasing
                or session_idx.tz is not None or not session_idx.equals(session_idx.normalize())):
            raise ValueError("Prices require unique, increasing, naive daily session dates.")
        exchange_obj = xcals.get_calendar(
            "XNYS", start=session_idx[0] - pd.Timedelta(days=1),
            end=session_idx[-1] + pd.Timedelta(days=1),
        )
        expected_idx = exchange_obj.sessions_in_range(session_idx[0], session_idx[-1])
        if not session_idx.equals(expected_idx):
            raise ValueError("QQQ prices must contain every exchange session; do not fill or drop gaps.")
        if pricing_data_df.attrs.get("norgate_adjustment_by_symbol_dict", {}).get("QQQ") != "CAPITALSPECIAL":
            raise ValueError("QQQ requires CAPITALSPECIAL price provenance.")
        if pricing_data_df.attrs.get("price_padding_policy_str") != "NONE":
            raise ValueError("QQQ requires unpadded price provenance (NONE).")
        required_field_list = ["Open", "High", "Low", "Close", "Volume", "Dividend"]
        missing_field_list = [
            field_str for field_str in required_field_list
            if ("QQQ", field_str) not in pricing_data_df.columns
        ]
        if missing_field_list:
            raise ValueError(f"QQQ is missing required fields: {missing_field_list}")
        observed_df = pricing_data_df.loc[:, [("QQQ", field_str) for field_str in required_field_list]]
        if not np.isfinite(observed_df.to_numpy(dtype=float)).all():
            raise ValueError("QQQ contains missing or nonfinite observations.")
        if (observed_df.loc[:, [("QQQ", field_str) for field_str in required_field_list[:-1]]] <= 0).any().any():
            raise ValueError("QQQ prices and volume must be positive; padded rows are forbidden.")

        signal_df = pricing_data_df.copy()
        close_ser = signal_df[("QQQ", "Close")]
        # *** CRITICAL*** Each window ends at Close_T, with all n observations
        # required. iterate reads only previous_bar=T and fills at Open_(T+1).
        fast_sma_ser = close_ser.rolling(50, min_periods=50).mean()
        slow_sma_ser = close_ser.rolling(200, min_periods=200).mean()
        spread_ser = fast_sma_ser - slow_sma_ser
        # *** CRITICAL*** Positive lag compares T with T-1, never with T+1.
        # Both SMA200 values must exist: first possible crossing is observation 201.
        previous_spread_ser = spread_ser.shift(1)
        signal_df[("QQQ", "sma_50")] = fast_sma_ser
        signal_df[("QQQ", "sma_200")] = slow_sma_ser
        signal_df[("QQQ", "entry_cross")] = (previous_spread_ser <= 0) & (spread_ser > 0)
        signal_df[("QQQ", "exit_cross")] = (previous_spread_ser >= 0) & (spread_ser < 0)
        return signal_df

    def iterate(
        self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series,
    ) -> None:
        if close_row_ser is None or self.previous_bar is None:
            return
        # *** CRITICAL*** Pre-start prices warm indicators only. A crossing
        # before the chosen start cannot trigger an opening trade on day one.
        if self.decision_start_ts is not None and self.previous_bar < self.decision_start_ts:
            return
        position_float = self.get_position("QQQ")
        if position_float > 0 and bool(close_row_ser[("QQQ", "exit_cross")]):
            self.order_target("QQQ", 0, trade_id=self.trade_id_int)
        elif position_float == 0 and bool(close_row_ser[("QQQ", "entry_cross")]):
            # *** CRITICAL*** q=floor(NAV_T/Close_T), before Open_(T+1) is known.
            # No open-price sizing or hidden cash buffer; gap/cost overdrafts remain possible.
            close_float = float(close_row_ser[("QQQ", "Close")])
            target_share_int = int(np.floor(self.previous_total_value / close_float))
            if target_share_int > 0:
                self.trade_id_int += 1
                self.order_target("QQQ", target_share_int, trade_id=self.trade_id_int)

    def plot(self, benchmark=None, benchmark_label=None, save_to=None):
        """Render the standard report without an unapproved benchmark."""
        if benchmark is not None:
            raise ValueError("No benchmark account is configured for this baseline.")
        plot_performance(
            self.results["total_value"], self.results["drawdown"],
            strategy_label="QQQ Golden Cross", save_to=save_to,
        )


def run_variant(
    *,
    history_start_date_str: str,
    backtest_start_date_str: str,
    end_date_str: str,
    capital_base_float: float,
    slippage_float: float = 0.00025,
    commission_per_share_float: float = 0.005,
    commission_minimum_float: float = 1.0,
    pricing_data_df: pd.DataFrame | None = None,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
) -> QqqGoldenCrossStrategy:
    """Run one explicitly dated baseline; there is no default historical window."""
    history_start_ts = pd.Timestamp(history_start_date_str)
    backtest_start_ts = pd.Timestamp(backtest_start_date_str)
    end_ts = pd.Timestamp(end_date_str)
    if not history_start_ts < backtest_start_ts <= end_ts:
        raise ValueError("Require history_start < backtest_start <= end.")
    if pricing_data_df is None:
        pricing_data_df = get_prices(history_start_date_str, end_date_str)
    # *** CRITICAL*** Bound the supplied history before calculating signals.
    # Earlier observations warm SMAs; only dates at/after start can form new trades.
    pricing_data_df = pricing_data_df.loc[history_start_ts:end_ts].copy()
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= backtest_start_ts]
    if len(pricing_data_df.index[pricing_data_df.index < backtest_start_ts]) < 200:
        raise ValueError("Provide at least 200 observed warmup sessions before the start.")
    strategy_obj = QqqGoldenCrossStrategy(
        capital_base_float, slippage_float, commission_per_share_float, commission_minimum_float,
    )
    run_daily(
        strategy_obj, pricing_data_df, calendar=calendar_idx,
        show_progress=show_display_bool, show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )
    if show_display_bool:
        print(strategy_obj.summary)
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj
