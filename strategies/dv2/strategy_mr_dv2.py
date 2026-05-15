import pandas as pd

from IPython.display import display
import talib
from collections import defaultdict
from typing import List
from alpha.engine.strategy import Strategy
from alpha.engine.backtest import run_daily
from alpha.engine.friction_analysis import FrictionAnalysis
from alpha.engine.report import save_results
from alpha.indicators import dv2_indicator
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices


def get_prices(symbols: List[str], benchmarks: List[str], start_date: str = '1998-01-01', end_date: str = None) -> pd.DataFrame:
    return load_raw_prices(symbols, benchmarks, start_date, end_date)


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """
    Build inputs for ExecutionTimingAnalysis.

    Formula:

        signal_t = daily DV2 signal known after T close

        entry_fill = signal_t + entry_lag at entry_price_field
        exit_fill  = signal_t + exit_lag  at exit_price_field
    """
    benchmark_list = ['$SPX']
    symbol_list, universe_df = build_index_constituent_matrix(indexname='S&P 500')
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date='1998-01-01',
        end_date=None,
    )

    # *** CRITICAL*** Keep the same post-warmup calendar used by the Vanilla
    # strategy script. Changing this sample changes all annualized metrics.
    calendar_idx = pricing_data_df.index[pricing_data_df.index.year >= 2004]

    def strategy_factory_fn():
        strategy_obj = DVO2Strategy(
            name='strategy_mr_dv2',
            benchmarks=benchmark_list,
            capital_base=100_000,
            slippage=0.00025,
            commission_per_share=0.005,
            commission_minimum=1.0,
        )
        strategy_obj.universe_df = universe_df
        strategy_obj.trade_id = 0
        strategy_obj.current_trade = defaultdict(default_trade_id_int)
        return strategy_obj

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "daily_ohlc_signal",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }


def default_trade_id_int() -> int:
    return -1


def get_asof_universe_symbol_list(
    universe_df: pd.DataFrame | None,
    decision_date_ts: pd.Timestamp,
) -> list[str]:
    if universe_df is None or len(universe_df) == 0:
        return []

    sorted_universe_df = universe_df.sort_index()
    # *** CRITICAL*** PIT universe membership may lag the newest price date.
    # Use only the latest universe row available on or before decision_t; never
    # use a later row, because that would leak future index membership.
    universe_row_int = int(
        sorted_universe_df.index.searchsorted(pd.Timestamp(decision_date_ts), side="right")
    ) - 1
    if universe_row_int < 0:
        return []

    universe_membership_ser = sorted_universe_df.iloc[universe_row_int]
    return universe_membership_ser[universe_membership_ser == 1].index.astype(str).tolist()


def build_friction_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """
    Build the completed DV2 run needed by FrictionAnalysis.

    Formula:

        auction_proxy_dollar_t = lagged_20d_median_dollar_adv_t * auction_fraction
        participation_t = abs(order_dollar_t) / auction_proxy_dollar_t

    The actual auction proxy is computed inside alpha.engine.friction_analysis.
    This hook only preserves the existing DV2 data-loading and run path.
    """
    benchmark_list = ['$SPX']
    symbol_list, universe_df = build_index_constituent_matrix(indexname='S&P 500')
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date='1998-01-01',
        end_date=end_date_str,
    )

    strategy_obj = DVO2Strategy(
        name='strategy_mr_dv2',
        benchmarks=benchmark_list,
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy_obj.universe_df = universe_df
    strategy_obj.trade_id = 0
    strategy_obj.current_trade = defaultdict(default_trade_id_int)

    # *** CRITICAL*** FrictionAnalysis must assess the same completed order
    # ledger as the deployment-reference DV2 backtest. Keep pre-start history
    # for indicators, but execute only on the configured backtest calendar.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
    )

    strategy_obj.universe_df = None
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
    }


class DVO2Strategy(Strategy):
    max_positions = 10  # maximum number of positions to hold
    trade_id = 0  # intiliazing trade_id to 0
    current_trade = defaultdict(default_trade_id_int)  # initializing current_trade as a defaultdict with default value -1
    universe_df = None  # df storing the universe of stocks

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        symbols = signal_data.columns.get_level_values(0).unique()
        feature_cols = {}

        for symbol in symbols:
            if str(symbol).startswith('$') or (symbol, 'Close') not in signal_data.columns:
                continue

            close = signal_data[(symbol, 'Close')]
            high = signal_data[(symbol, 'High')]
            low = signal_data[(symbol, 'Low')]

            # *** CRITICAL*** All feature calculations below must be causal:
            # decision_t may use historical closes/rolling windows only through
            # previous_bar, and the engine fills resulting orders at Open_t.
            feature_cols[(symbol, 'p126d_return')] = close / close.shift(126) - 1
            feature_cols[(symbol, 'natr')] = talib.NATR(high, low, close, 14)
            feature_cols[(symbol, 'dv2')] = dv2_indicator(close, high, low, length_int=126)
            feature_cols[(symbol, 'sma_200')] = close.rolling(200).mean()

        if not feature_cols:
            return signal_data

        features = pd.DataFrame(feature_cols, index=signal_data.index)
        return pd.concat([signal_data, features], axis=1).copy()

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        """
        this method will contain the logic for executing trades based on
        the strategy rules. it must be implemented to place buy and sell orders
        according to QPI and price conditions.
        """

        # get current porfolio positions
        positions = self.get_positions()
        long_positions = positions[positions > 0]
        long_slots = self.max_positions - len(long_positions)  # calculate available slots for new positions

        # exit rules
        """ exit logic: sell if price > yesterday's high """
        for symbol in long_positions.index:
            c = close[(symbol, 'Close')]
            yh = data[(symbol, 'High')].iloc[-2]
            if c > yh:
                self.order_target_value(symbol, 0, trade_id=self.current_trade[symbol])
                long_slots += 1


        # entry rules
        capital_to_allocate_per_trade = self.previous_total_value / self.max_positions  # each trade receives an equal share of the portfolio
        long_opportunities = self.get_opportunities(close)

        while long_slots > 0 and len(long_opportunities) > 0:
            symbol = long_opportunities.pop(0)  # pips the top-ranked stock from opportunities

            # skip if already holding this stock (for safety)
            if self.get_position(symbol) != 0:
                continue

            # assign new trade_id and log it
            self.trade_id += 1
            self.current_trade[symbol] = self.trade_id

            # place buy order with equal capital allocation
            self.order_value(symbol, capital_to_allocate_per_trade, trade_id=self.trade_id)

            long_slots -= 1  # reduce available slots

    def get_opportunities(self, close) -> list:
        """
        identifying and ranking the best trade opportunities based on our entry criteria
        """
        # unstack multi-index DataFrame to have symbols as index and features as columns
        df = close.unstack().dropna()

        # remove benchmark symbols (e.g., $SPX) from the dataset
        df = df[~df.index.astype(str).str.startswith('$')]

        # apply entry filters:
        # - dv2<10
        # - Close[0] > sma(200) (uptrend condition)
        # - Return over 126 days > 0 (positive long-term momentum)
        # then sort by NATR in descending order for liquidity ranking
        df = df[
            (df['dv2'] < 10) &
            (df['Close'] > df['sma_200']) &
            (df['p126d_return'] > 0.05)
        ].sort_values('natr', ascending=False)

        # get the list of stocks in the universe on the previous trading day
        u = get_asof_universe_symbol_list(self.universe_df, pd.Timestamp(self.previous_bar))

        # return the filtered list of opportunity tickers that are also in the universe
        return df[df.index.isin(u)].index.tolist()


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    benchmarks = ['$SPX']
    index_symbols, universe_df = build_index_constituent_matrix(indexname='S&P 500')
    pricing_data = get_prices(index_symbols, benchmarks, start_date='1998-01-01', end_date=end_date_str)

    strategy = DVO2Strategy(
        name='strategy_mr_dv2',
        benchmarks=benchmarks,
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.universe_df = universe_df
    strategy.trade_id = 0
    strategy.current_trade = defaultdict(default_trade_id_int)

    # *** CRITICAL*** Deployment-reference backtests keep full pre-start
    # history for indicators, but the executable calendar starts at the first
    # deployment fill session.
    calendar = pricing_data.index[pricing_data.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(
        strategy,
        pricing_data,
        calendar,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
    )

    strategy.universe_df = None

    if show_display_bool:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        display(strategy.summary)
        display(strategy.summary_trades)

    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)

    return strategy


def run_friction_analysis(
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    friction_input_dict = build_friction_analysis_inputs(
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    friction_analysis_obj = FrictionAnalysis(
        strategy_obj=friction_input_dict["strategy_obj"],
        pricing_data_df=friction_input_dict["pricing_data_df"],
        execution_policy_str=friction_input_dict["execution_policy_str"],
        output_dir_str=output_dir_str,
        save_output_bool=save_results_bool,
    )
    return friction_analysis_obj.run()



if __name__ == "__main__":
    run_variant()

