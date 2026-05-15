import unittest
import warnings

import numpy as np
import pandas as pd

from strategies.crypto.strategy_btc_residual_upro_lookahead import (
    BtcResidualUproLookaheadResearchStrategy,
    build_btc_residual_signal_data_df,
    run_btc_residual_upro_lookahead_backtest,
    run_variant,
)


class BtcResidualUproLookaheadTests(unittest.TestCase):
    @staticmethod
    def make_close_price_ser(
        date_index: pd.DatetimeIndex,
        return_list: list[float],
        start_price_float: float,
    ) -> pd.Series:
        price_list = [float(start_price_float)]
        for return_float in return_list[1:]:
            price_list.append(float(price_list[-1] * (1.0 + return_float)))
        return pd.Series(price_list, index=date_index, dtype=float)

    @staticmethod
    def make_pricing_data_df(close_price_map: dict[str, pd.Series]) -> pd.DataFrame:
        frame_map: dict[tuple[str, str], pd.Series] = {}
        for symbol_str, close_price_ser in close_price_map.items():
            close_price_ser = close_price_ser.astype(float)
            frame_map[(symbol_str, "Open")] = close_price_ser
            frame_map[(symbol_str, "High")] = close_price_ser * 1.01
            frame_map[(symbol_str, "Low")] = close_price_ser * 0.99
            frame_map[(symbol_str, "Close")] = close_price_ser

        pricing_data_df = pd.DataFrame(frame_map, index=next(iter(close_price_map.values())).index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_signal_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=14, freq="B")
        risk_return_list = [
            0.0,
            0.010,
            -0.006,
            0.018,
            -0.012,
            0.015,
            0.004,
            -0.007,
            0.011,
            0.006,
            -0.004,
            0.013,
            -0.009,
            0.016,
        ]
        btc_return_list = [
            0.0,
            0.026,
            -0.010,
            0.031,
            -0.030,
            0.021,
            0.018,
            -0.020,
            0.039,
            -0.002,
            0.008,
            0.028,
            -0.025,
            0.050,
        ]
        upro_return_list = [
            0.0,
            0.012,
            -0.004,
            0.020,
            -0.008,
            0.016,
            0.003,
            -0.006,
            0.009,
            0.007,
            -0.003,
            0.011,
            -0.006,
            0.014,
        ]
        return self.make_pricing_data_df(
            {
                "BTC-USD": self.make_close_price_ser(date_index, btc_return_list, 100.0),
                "QQQ": self.make_close_price_ser(date_index, risk_return_list, 50.0),
                "UPRO": self.make_close_price_ser(date_index, upro_return_list, 25.0),
            }
        )

    def make_strategy(self, **kwargs) -> BtcResidualUproLookaheadResearchStrategy:
        base_kwargs = dict(
            name="BtcResidualUproLookaheadTest",
            benchmarks=[],
            btc_symbol_str="BTC-USD",
            risk_symbol_str="QQQ",
            trade_symbol_str="UPRO",
            beta_lookback_int=3,
            zscore_lookback_int=2,
            entry_zscore_float=1.5,
            target_notional_float=1_000.0,
            capital_base=10_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return BtcResidualUproLookaheadResearchStrategy(**base_kwargs)

    def test_build_signal_data_adds_domain_type_columns(self):
        pricing_data_df = self.make_signal_pricing_data_df()

        signal_data_df = build_btc_residual_signal_data_df(
            pricing_data_df=pricing_data_df,
            btc_symbol_str="BTC-USD",
            risk_symbol_str="QQQ",
            beta_lookback_int=3,
            zscore_lookback_int=2,
        )

        expected_col_list = [
            ("BTC-USD", "btc_return_ser"),
            ("QQQ", "risk_return_ser"),
            ("BTC-USD", "beta_ser"),
            ("BTC-USD", "expected_btc_return_ser"),
            ("BTC-USD", "residual_ser"),
            ("BTC-USD", "residual_zscore_ser"),
        ]
        for col_tuple in expected_col_list:
            self.assertIn(col_tuple, signal_data_df.columns)

    def test_beta_and_zscore_use_prior_rolling_windows(self):
        pricing_data_df = self.make_signal_pricing_data_df()

        signal_data_df = build_btc_residual_signal_data_df(
            pricing_data_df=pricing_data_df,
            btc_symbol_str="BTC-USD",
            risk_symbol_str="QQQ",
            beta_lookback_int=3,
            zscore_lookback_int=2,
        )
        btc_return_ser = signal_data_df[("BTC-USD", "btc_return_ser")].astype(float)
        risk_return_ser = signal_data_df[("QQQ", "risk_return_ser")].astype(float)
        residual_ser = signal_data_df[("BTC-USD", "residual_ser")].astype(float)

        expected_beta_ser = (
            btc_return_ser.rolling(3).cov(risk_return_ser).shift(1)
            / risk_return_ser.rolling(3).var().shift(1)
        )
        unshifted_beta_ser = (
            btc_return_ser.rolling(3).cov(risk_return_ser)
            / risk_return_ser.rolling(3).var()
        )
        expected_zscore_ser = residual_ser / residual_ser.rolling(2).std().shift(1)
        unshifted_zscore_ser = residual_ser / residual_ser.rolling(2).std()

        check_bar_ts = signal_data_df.index[8]
        actual_beta_float = float(signal_data_df.loc[check_bar_ts, ("BTC-USD", "beta_ser")])
        actual_zscore_float = float(signal_data_df.loc[check_bar_ts, ("BTC-USD", "residual_zscore_ser")])

        self.assertAlmostEqual(actual_beta_float, float(expected_beta_ser.loc[check_bar_ts]))
        self.assertAlmostEqual(actual_zscore_float, float(expected_zscore_ser.loc[check_bar_ts]))
        self.assertNotAlmostEqual(actual_beta_float, float(unshifted_beta_ser.loc[check_bar_ts]))
        self.assertNotAlmostEqual(actual_zscore_float, float(unshifted_zscore_ser.loc[check_bar_ts]))

    def test_backtest_enters_and_exits_at_same_signal_close_without_topping_up(self):
        date_index = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        )
        pricing_data_df = self.make_pricing_data_df(
            {
                "BTC-USD": pd.Series([100.0, 102.0, 103.0, 101.0, 100.0], index=date_index),
                "QQQ": pd.Series([50.0, 51.0, 51.5, 51.0, 50.5], index=date_index),
                "UPRO": pd.Series([100.0, 100.0, 110.0, 120.0, 130.0], index=date_index),
            }
        )
        signal_data_df = pricing_data_df.copy()
        signal_data_df[("BTC-USD", "residual_zscore_ser")] = [np.nan, 1.6, 1.7, 1.0, np.nan]
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)

        strategy = self.make_strategy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            run_btc_residual_upro_lookahead_backtest(
                strategy=strategy,
                pricing_data_df=pricing_data_df,
                signal_data_df=signal_data_df,
                backtest_start_date_str=None,
            )

        transaction_df = strategy.get_transactions().reset_index(drop=True)
        self.assertEqual(len(transaction_df), 2)

        entry_row_ser = transaction_df.iloc[0]
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), pd.Timestamp("2024-01-03"))
        self.assertEqual(entry_row_ser["asset"], "UPRO")
        self.assertEqual(float(entry_row_ser["price"]), 100.0)
        self.assertEqual(float(entry_row_ser["amount"]), 10.0)

        exit_row_ser = transaction_df.iloc[1]
        self.assertEqual(pd.Timestamp(exit_row_ser["bar"]), pd.Timestamp("2024-01-05"))
        self.assertEqual(float(exit_row_ser["price"]), 120.0)
        self.assertEqual(float(exit_row_ser["amount"]), -10.0)

    def test_backtest_does_not_trade_before_signal_warmup(self):
        date_index = pd.date_range("2024-01-02", periods=10, freq="B")
        pricing_data_df = self.make_pricing_data_df(
            {
                "BTC-USD": pd.Series(np.linspace(100.0, 110.0, len(date_index)), index=date_index),
                "QQQ": pd.Series(np.linspace(50.0, 55.0, len(date_index)), index=date_index),
                "UPRO": pd.Series(np.linspace(25.0, 30.0, len(date_index)), index=date_index),
            }
        )
        strategy = BtcResidualUproLookaheadResearchStrategy(
            name="BtcResidualWarmupTest",
            benchmarks=[],
            capital_base=10_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

        signal_data_df = strategy.compute_signals(pricing_data_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            run_btc_residual_upro_lookahead_backtest(
                strategy=strategy,
                pricing_data_df=pricing_data_df,
                signal_data_df=signal_data_df,
                backtest_start_date_str=None,
            )

        self.assertEqual(len(strategy.get_transactions()), 0)
        self.assertIsNotNone(strategy.summary)

    def test_run_variant_accepts_synthetic_pricing_data_and_returns_summary(self):
        pricing_data_df = self.make_signal_pricing_data_df()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            strategy = run_variant(
                show_display_bool=False,
                save_results_bool=False,
                pricing_data_df=pricing_data_df,
                benchmark_list=(),
                beta_lookback_int=3,
                zscore_lookback_int=2,
                entry_zscore_float=-999.0,
                target_notional_float=1_000.0,
                capital_base_float=10_000.0,
                slippage_float=0.0,
                commission_per_share_float=0.0,
                commission_minimum_float=0.0,
                backtest_start_date_str=str(pricing_data_df.index[0].date()),
            )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)
        self.assertGreater(len(strategy.signal_data_df), 0)


if __name__ == "__main__":
    unittest.main()
