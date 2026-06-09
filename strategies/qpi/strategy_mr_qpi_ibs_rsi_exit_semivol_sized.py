import numpy as np
import pandas as pd

from IPython.display import display
from collections import defaultdict

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import build_index_constituent_matrix
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import (
    QPIIbsRsiExitStrategy,
    default_trade_id_int,
    get_prices,
)


ENTRY_SIZING_MODE_INVERSE_SEMIVOL_STR = "inverse_semivol"
ENTRY_SIZING_MODE_DIRECT_SEMIVOL_STR = "direct_semivol"
VALID_ENTRY_SIZING_MODE_SET = {
    ENTRY_SIZING_MODE_INVERSE_SEMIVOL_STR,
    ENTRY_SIZING_MODE_DIRECT_SEMIVOL_STR,
}
DEFAULT_SEMIVOLATILITY_LOOKBACK_INT = 63


def clean_entry_sizing_mode_str(entry_sizing_mode_str: str) -> str:
    clean_mode_str = str(entry_sizing_mode_str).strip().lower()
    if clean_mode_str not in VALID_ENTRY_SIZING_MODE_SET:
        raise ValueError(
            f"entry_sizing_mode_str must be one of {sorted(VALID_ENTRY_SIZING_MODE_SET)}, "
            f"got {entry_sizing_mode_str!r}."
        )
    return clean_mode_str


class QPIIbsRsiExitSemivolSizedStrategy(QPIIbsRsiExitStrategy):
    """
    QPI research variant.

    The baseline QPI/IBS/SMA/3-day-return filters, Turnover ranking, PIT
    universe, and IBS/RSI2 exits stay the same. Only new-entry dollar sizing
    changes from equal slot to downside-volatility sizing.
    """

    def __init__(
        self,
        name: str,
        benchmarks: list | tuple,
        capital_base=100_000,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        max_positions_int: int = 10,
        qpi_threshold_float: float = 30.0,
        sma_window_int: int = 200,
        qpi_window_int: int = 3,
        qpi_lookback_years_int: int = 5,
        return_lookback_days_int: int = 3,
        max_entry_ibs_float: float = 0.1,
        exit_ibs_threshold_float: float = 0.90,
        rsi_window_int: int = 2,
        exit_rsi2_threshold_float: float = 90.0,
        entry_sizing_mode_str: str = ENTRY_SIZING_MODE_INVERSE_SEMIVOL_STR,
        semivolatility_lookback_int: int = DEFAULT_SEMIVOLATILITY_LOOKBACK_INT,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
            max_positions_int=max_positions_int,
            qpi_threshold_float=qpi_threshold_float,
            sma_window_int=sma_window_int,
            qpi_window_int=qpi_window_int,
            qpi_lookback_years_int=qpi_lookback_years_int,
            return_lookback_days_int=return_lookback_days_int,
            max_entry_ibs_float=max_entry_ibs_float,
            exit_ibs_threshold_float=exit_ibs_threshold_float,
            rsi_window_int=rsi_window_int,
            exit_rsi2_threshold_float=exit_rsi2_threshold_float,
        )
        if semivolatility_lookback_int < 2:
            raise ValueError("semivolatility_lookback_int must be at least 2.")
        self.entry_sizing_mode_str = clean_entry_sizing_mode_str(entry_sizing_mode_str)
        self.semivolatility_lookback_int = int(semivolatility_lookback_int)
        self.semivolatility_field_str = f"semivolatility_{self.semivolatility_lookback_int}_ser"

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data_df)
        symbol_list = signal_data_df.columns.get_level_values(0).unique()
        close_price_map = {}

        for symbol_obj in symbol_list:
            symbol_str = str(symbol_obj)
            if symbol_str.startswith("$") or (symbol_obj, "Close") not in signal_data_df.columns:
                continue
            close_price_map[symbol_str] = signal_data_df[(symbol_obj, "Close")].astype(float)

        if not close_price_map:
            return signal_data_df

        close_price_df = pd.DataFrame(close_price_map, index=signal_data_df.index)

        # *** CRITICAL*** Entry sizing semivolatility must use trailing closes
        # only. The engine consumes this value at previous_bar close and fills
        # orders at the next open; future closes here would be lookahead.
        close_return_df = close_price_df.pct_change(fill_method=None)
        downside_return_df = close_return_df.clip(upper=0.0)
        semivolatility_df = (
            downside_return_df.pow(2.0)
            .rolling(
                window=self.semivolatility_lookback_int,
                min_periods=self.semivolatility_lookback_int,
            )
            .mean()
            .pow(0.5)
            * np.sqrt(252.0)
        )

        semivolatility_feature_df = semivolatility_df.copy()
        semivolatility_feature_df.columns = pd.MultiIndex.from_tuples(
            [
                (symbol_str, self.semivolatility_field_str)
                for symbol_str in semivolatility_feature_df.columns
            ]
        )
        return pd.concat([signal_data_df, semivolatility_feature_df], axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        long_slots_int = self.max_positions_int - len(long_position_ser)

        for symbol_str in long_position_ser.index:
            ibs_value_float = close_row_ser.get((symbol_str, "ibs_value_ser"), np.nan)
            rsi2_value_float = close_row_ser.get((symbol_str, "rsi2_value_ser"), np.nan)

            exit_for_ibs_bool = (
                pd.notna(ibs_value_float)
                and float(ibs_value_float) > self.exit_ibs_threshold_float
            )
            exit_for_rsi2_bool = (
                pd.notna(rsi2_value_float)
                and float(rsi2_value_float) > self.exit_rsi2_threshold_float
            )

            if exit_for_ibs_bool or exit_for_rsi2_bool:
                self.order_target_value(
                    symbol_str,
                    0.0,
                    trade_id=self.current_trade_map[symbol_str],
                )
                long_slots_int += 1

        if long_slots_int <= 0:
            return

        opportunity_symbol_list = self.get_opportunity_list(close_row_ser)
        selected_symbol_list = []
        selected_semivolatility_list = []

        while long_slots_int > 0 and len(opportunity_symbol_list) > 0:
            symbol_str = opportunity_symbol_list.pop(0)

            if self.get_position(symbol_str) != 0:
                continue

            semivolatility_float = float(
                close_row_ser.get((symbol_str, self.semivolatility_field_str), np.nan)
            )
            if not np.isfinite(semivolatility_float) or semivolatility_float <= 0.0:
                continue

            selected_symbol_list.append(symbol_str)
            selected_semivolatility_list.append(semivolatility_float)
            long_slots_int -= 1

        if len(selected_symbol_list) == 0:
            return

        semivolatility_ser = pd.Series(
            selected_semivolatility_list,
            index=selected_symbol_list,
            dtype=float,
        )
        if self.entry_sizing_mode_str == ENTRY_SIZING_MODE_INVERSE_SEMIVOL_STR:
            raw_weight_ser = 1.0 / semivolatility_ser
        else:
            raw_weight_ser = semivolatility_ser

        weight_ser = raw_weight_ser / raw_weight_ser.sum()
        entry_budget_float = self.previous_total_value * len(weight_ser) / float(self.max_positions_int)
        order_value_ser = entry_budget_float * weight_ser

        for symbol_str, order_value_float in order_value_ser.items():
            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_value(symbol_str, float(order_value_float), trade_id=self.trade_id_int)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    entry_sizing_mode_str: str = ENTRY_SIZING_MODE_INVERSE_SEMIVOL_STR,
):
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=end_date_str,
    )

    clean_mode_str = clean_entry_sizing_mode_str(entry_sizing_mode_str)
    strategy_obj = QPIIbsRsiExitSemivolSizedStrategy(
        name=f"strategy_mr_qpi_ibs_rsi_exit_semivol_sized_{clean_mode_str}",
        benchmarks=benchmark_list,
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
        entry_sizing_mode_str=clean_mode_str,
    )
    strategy_obj.universe_df = universe_df
    strategy_obj.trade_id_int = 0
    strategy_obj.current_trade_map = defaultdict(default_trade_id_int)

    # *** CRITICAL*** Keep full pre-start history for QPI/IBS/RSI and
    # semivolatility features, but execute only on the configured backtest calendar.
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
    )

    strategy_obj.universe_df = None

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
