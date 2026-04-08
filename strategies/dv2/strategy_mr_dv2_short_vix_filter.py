from __future__ import annotations

from collections import defaultdict

import pandas as pd
import talib
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import Strategy
from alpha.indicators import dv2_indicator
from data.norgate_loader import build_index_constituent_matrix
from strategies.dv2.strategy_mr_dv2_vix_filter import (
    BENCHMARK_SYMBOL_STR,
    DEFAULT_MAX_POSITIONS_INT,
    DV2_LOOKBACK_DAY_INT,
    NATR_LOOKBACK_DAY_INT,
    P126_RETURN_MIN_FLOAT,
    SMA_WINDOW_DAY_INT,
    VIX_SIGNAL_SYMBOL_STR,
    compute_vix_regime_signal_df,
    get_dv2_vix_filter_prices,
)


DV2_SHORT_ENTRY_MIN_FLOAT = 90.0
P126_SHORT_RETURN_MAX_FLOAT = -P126_RETURN_MIN_FLOAT


def default_trade_id_int() -> int:
    return -1


class DV2ShortVixFilterStrategy(Strategy):
    """
    DV2 stock mean-reversion short variant with a VIX bear-market entry gate.

    Entry formula:

        entry_t
            = 1[dv2_t > 90]
            * 1[Close_t < SMA200_t]
            * 1[p126d_return_t < -0.05]
            * 1[vix_bear_bool_t = 1]

    Exit formula:

        cover_t
            = 1[Close_t < Low_{t-1}]
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        capital_base: float = 100_000.0,
        slippage: float = 0.0001,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        max_positions_int: int = DEFAULT_MAX_POSITIONS_INT,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        if max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")

        self.max_positions_int = int(max_positions_int)
        self.trade_id_int = 0
        self.current_trade_id_map = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        feature_col_map: dict[tuple[str, str], pd.Series] = {}
        symbol_list = signal_data_df.columns.get_level_values(0).unique().tolist()

        for symbol_str in self.signal_progress(
            symbol_list,
            desc_str="dv2 short signal precompute",
            total_int=len(symbol_list),
        ):
            if str(symbol_str).startswith("$") or (symbol_str, "Close") not in signal_data_df.columns:
                continue

            close_price_ser = signal_data_df[(symbol_str, "Close")].astype(float)
            high_price_ser = signal_data_df[(symbol_str, "High")].astype(float)
            low_price_ser = signal_data_df[(symbol_str, "Low")].astype(float)

            # *** CRITICAL*** The 126-day return must use only lagged closes.
            # Any forward-looking return would leak future path information
            # into the current short entry filter.
            p126d_return_ser = close_price_ser / close_price_ser.shift(DV2_LOOKBACK_DAY_INT) - 1.0
            natr_value_ser = pd.Series(
                talib.NATR(
                    high_price_ser.to_numpy(dtype=float),
                    low_price_ser.to_numpy(dtype=float),
                    close_price_ser.to_numpy(dtype=float),
                    NATR_LOOKBACK_DAY_INT,
                ),
                index=signal_data_df.index,
                dtype=float,
            )
            dv2_value_ser = dv2_indicator(
                close_price_ser,
                high_price_ser,
                low_price_ser,
                length_int=DV2_LOOKBACK_DAY_INT,
            )

            # *** CRITICAL*** The 200-day SMA must be strictly trailing.
            # Using future prices in this trend filter would invalidate the
            # short-side signal audit.
            sma_200_price_ser = close_price_ser.rolling(SMA_WINDOW_DAY_INT).mean()

            feature_col_map[(symbol_str, "p126d_return_ser")] = p126d_return_ser
            feature_col_map[(symbol_str, "natr_value_ser")] = natr_value_ser
            feature_col_map[(symbol_str, "dv2_value_ser")] = dv2_value_ser
            feature_col_map[(symbol_str, "sma_200_price_ser")] = sma_200_price_ser

        vix_close_key = (VIX_SIGNAL_SYMBOL_STR, "Close")
        if vix_close_key not in signal_data_df.columns:
            raise RuntimeError(f"Missing required VIX close column: {vix_close_key}")

        vix_regime_signal_df = compute_vix_regime_signal_df(signal_data_df[vix_close_key])
        for field_str in vix_regime_signal_df.columns:
            feature_col_map[(VIX_SIGNAL_SYMBOL_STR, field_str)] = vix_regime_signal_df[field_str]

        if len(feature_col_map) == 0:
            return signal_data_df

        feature_data_df = pd.DataFrame(feature_col_map, index=signal_data_df.index)
        return pd.concat([signal_data_df, feature_data_df], axis=1)

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ):
        if data_df is None or close_row_ser is None:
            return

        position_ser = self.get_positions()
        short_position_ser = position_ser[position_ser < 0]
        short_slot_int = self.max_positions_int - len(short_position_ser)

        for symbol_str in short_position_ser.index:
            close_price_float = float(close_row_ser.loc[(symbol_str, "Close")])

            # *** CRITICAL*** iterate() only sees data through previous_bar.
            # Using `.iloc[-2]` preserves the inherited "prior low" mirror of
            # the legacy DV2 exit timing instead of reading beyond the causal
            # information set.
            prior_low_float = float(data_df[(symbol_str, "Low")].iloc[-2])
            if close_price_float < prior_low_float:
                self.order_target_value(
                    symbol_str,
                    0,
                    trade_id=self.current_trade_id_map[symbol_str],
                )
                short_slot_int += 1

        vix_bear_key = (VIX_SIGNAL_SYMBOL_STR, "vix_bear_bool")
        if vix_bear_key not in close_row_ser.index:
            return

        vix_bear_value = close_row_ser.loc[vix_bear_key]
        if pd.isna(vix_bear_value):
            return
        if not bool(vix_bear_value):
            return

        capital_to_allocate_per_trade_float = self.previous_total_value / self.max_positions_int
        opportunity_list = self.get_opportunity_list(close_row_ser)

        while short_slot_int > 0 and len(opportunity_list) > 0:
            symbol_str = opportunity_list.pop(0)
            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id_int += 1
            self.current_trade_id_map[symbol_str] = self.trade_id_int
            self.order_value(
                symbol_str,
                -capital_to_allocate_per_trade_float,
                trade_id=self.trade_id_int,
            )
            short_slot_int -= 1

    def get_opportunity_list(self, close_row_ser: pd.Series) -> list[str]:
        """
        Rank eligible short candidates in descending NATR order.
        """
        if self.universe_df is None:
            raise RuntimeError("universe_df must be assigned before get_opportunity_list().")

        stock_mask = ~close_row_ser.index.get_level_values(0).astype(str).str.startswith("$")
        stock_row_ser = close_row_ser.loc[stock_mask]
        if len(stock_row_ser) == 0:
            return []

        stock_feature_df = stock_row_ser.unstack()
        required_col_list = [
            "Close",
            "p126d_return_ser",
            "natr_value_ser",
            "dv2_value_ser",
            "sma_200_price_ser",
        ]
        if any(field_str not in stock_feature_df.columns for field_str in required_col_list):
            return []

        stock_feature_df = stock_feature_df.dropna(subset=required_col_list)
        if len(stock_feature_df) == 0:
            return []

        eligible_feature_df = stock_feature_df[
            (stock_feature_df["dv2_value_ser"] > DV2_SHORT_ENTRY_MIN_FLOAT)
            & (stock_feature_df["Close"] < stock_feature_df["sma_200_price_ser"])
            & (stock_feature_df["p126d_return_ser"] < P126_SHORT_RETURN_MAX_FLOAT)
        ].sort_values("natr_value_ser", ascending=False)

        universe_row_ser = self.universe_df.loc[self.previous_bar]
        tradable_symbol_list = universe_row_ser[universe_row_ser == 1].index.tolist()
        return eligible_feature_df[eligible_feature_df.index.isin(tradable_symbol_list)].index.tolist()


if __name__ == "__main__":
    benchmark_list = [BENCHMARK_SYMBOL_STR]
    index_symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_dv2_vix_filter_prices(
        index_symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )

    strategy = DV2ShortVixFilterStrategy(
        name="strategy_mr_dv2_short_vix_filter",
        benchmarks=benchmark_list,
        capital_base=100_000.0,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.universe_df = universe_df

    calendar_idx = pricing_data_df.index
    calendar_idx = calendar_idx[calendar_idx.year >= 2004]

    run_daily(
        strategy,
        pricing_data_df,
        calendar=calendar_idx,
        audit_override_bool=None,
    )

    strategy.universe_df = None

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)

    from alpha.engine.report import save_results

    save_results(strategy)
