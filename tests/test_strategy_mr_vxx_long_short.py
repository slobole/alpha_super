import os
import unittest
import warnings
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.vix_stuff.strategy_mr_vxx_long_short import (
    SIGNAL_NAMESPACE_STR,
    STITCHED_TRADE_SYMBOL_STR,
    VxxLongShortStrategy,
    build_stitched_trade_price_df,
    compute_vxx_long_short_signal_df,
)


class VxxLongShortStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> VxxLongShortStrategy:
        base_kwargs = dict(
            name="VxxLongShortTest",
            benchmarks=["SPY"],
            trade_symbol_str=STITCHED_TRADE_SYMBOL_STR,
            vix_symbol_str="$VIX",
            vix3m_symbol_str="$VIX3M",
            diff_float=0.0,
            target_weight_float=1.0 / 3.0,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return VxxLongShortStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=6, freq="B")

        pricing_data_df = pd.DataFrame(
            {
                (STITCHED_TRADE_SYMBOL_STR, "Open"): [40.0, 39.0, 38.0, 41.0, 43.0, 42.0],
                (STITCHED_TRADE_SYMBOL_STR, "High"): [41.0, 40.0, 39.0, 42.0, 44.0, 43.0],
                (STITCHED_TRADE_SYMBOL_STR, "Low"): [39.0, 38.0, 37.0, 40.0, 42.0, 41.0],
                (STITCHED_TRADE_SYMBOL_STR, "Close"): [39.5, 38.5, 41.5, 42.5, 41.0, 40.5],
                ("$VIX", "Open"): [20.0, 21.0, 18.0, 17.0, 22.0, 23.0],
                ("$VIX", "High"): [20.5, 21.5, 18.5, 17.5, 22.5, 23.5],
                ("$VIX", "Low"): [19.5, 20.5, 17.5, 16.5, 21.5, 22.5],
                ("$VIX", "Close"): [20.0, 21.0, 18.0, 17.0, 22.0, 23.0],
                ("$VIX3M", "Open"): [19.0, 20.0, 19.0, 18.0, 21.0, 24.0],
                ("$VIX3M", "High"): [19.5, 20.5, 19.5, 18.5, 21.5, 24.5],
                ("$VIX3M", "Low"): [18.5, 19.5, 18.5, 17.5, 20.5, 23.5],
                ("$VIX3M", "Close"): [19.0, 20.0, 19.0, 18.0, 21.0, 24.0],
                ("SPY", "Open"): [470.0, 471.0, 472.0, 473.0, 474.0, 475.0],
                ("SPY", "High"): [471.0, 472.0, 473.0, 474.0, 475.0, 476.0],
                ("SPY", "Low"): [469.0, 470.0, 471.0, 472.0, 473.0, 474.0],
                ("SPY", "Close"): [470.5, 471.5, 472.5, 473.5, 474.5, 475.5],
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_build_stitched_trade_price_df_uses_old_before_2019_and_new_from_2019(self):
        date_index = pd.to_datetime(["2018-12-31", "2019-01-02"])
        source_price_df = pd.DataFrame(
            {
                ("VXX-201901", "Open"): [100.0, 101.0],
                ("VXX-201901", "Close"): [100.5, 101.5],
                ("VXX", "Open"): [200.0, 201.0],
                ("VXX", "Close"): [200.5, 201.5],
            },
            index=date_index,
            dtype=float,
        )
        source_price_df.columns = pd.MultiIndex.from_tuples(source_price_df.columns)

        stitched_trade_price_df = build_stitched_trade_price_df(
            pricing_data_df=source_price_df,
            old_trade_symbol_str="VXX-201901",
            new_trade_symbol_str="VXX",
            stitched_trade_symbol_str=STITCHED_TRADE_SYMBOL_STR,
            splice_date_str="2019-01-01",
        )

        self.assertAlmostEqual(
            float(stitched_trade_price_df.loc[pd.Timestamp("2018-12-31"), (STITCHED_TRADE_SYMBOL_STR, "Open")]),
            100.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(stitched_trade_price_df.loc[pd.Timestamp("2019-01-02"), (STITCHED_TRADE_SYMBOL_STR, "Open")]),
            201.0,
            places=12,
        )

    def test_compute_vxx_long_short_signal_df_marks_only_regime_flips(self):
        date_index = pd.date_range("2024-01-02", periods=5, freq="B")

        signal_df = compute_vxx_long_short_signal_df(
            vix_close_ser=pd.Series([20.0, 21.0, 18.0, 17.0, 22.0], index=date_index, dtype=float),
            vix3m_close_ser=pd.Series([19.0, 20.0, 19.0, 18.0, 21.0], index=date_index, dtype=float),
            diff_float=0.0,
            target_weight_float=1.0 / 3.0,
        )

        self.assertEqual(signal_df["backwardation_bool_ser"].tolist(), [True, True, False, False, True])
        self.assertEqual(signal_df["regime_flip_bool_ser"].tolist(), [False, False, True, False, True])
        self.assertEqual(
            [round(x, 10) for x in signal_df["signed_regime_weight_ser"].tolist()],
            [round(x, 10) for x in [1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0]],
        )
        self.assertEqual(
            [round(x, 10) for x in signal_df["entry_target_weight_ser"].tolist()],
            [round(x, 10) for x in [0.0, 0.0, -1.0 / 3.0, 0.0, 1.0 / 3.0]],
        )

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn((SIGNAL_NAMESPACE_STR, "term_spread_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "backwardation_bool_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "prior_backwardation_bool_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "regime_flip_bool_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "signed_regime_weight_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "entry_target_weight_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_enters_short_when_regime_flips_to_contango(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "regime_flip_bool_ser"): True,
                (SIGNAL_NAMESPACE_STR, "entry_target_weight_ser"): -(1.0 / 3.0),
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy.iterate(
            pd.DataFrame(index=[strategy.previous_bar]),
            close_row_ser,
            pd.Series({STITCHED_TRADE_SYMBOL_STR: 40.0}, dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, STITCHED_TRADE_SYMBOL_STR)
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, -(1.0 / 3.0), places=12)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_reverses_from_short_to_long_on_regime_flip(self):
        strategy = self.make_strategy()
        strategy.trade_id_int = 4
        strategy.active_trade_id_int = 4
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.add_transaction(
            4,
            strategy.previous_bar,
            STITCHED_TRADE_SYMBOL_STR,
            -200,
            40.0,
            -8_000.0,
            1,
            0.0,
        )

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "regime_flip_bool_ser"): True,
                (SIGNAL_NAMESPACE_STR, "entry_target_weight_ser"): 1.0 / 3.0,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy.iterate(
            pd.DataFrame(index=[strategy.previous_bar]),
            close_row_ser,
            pd.Series({STITCHED_TRADE_SYMBOL_STR: 40.0}, dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)

        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, STITCHED_TRADE_SYMBOL_STR)
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 4)

        entry_order = order_list[1]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, STITCHED_TRADE_SYMBOL_STR)
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 1.0 / 3.0, places=12)
        self.assertEqual(entry_order.trade_id, 5)

    def test_iterate_submits_no_order_when_regime_does_not_flip(self):
        strategy = self.make_strategy()
        strategy.trade_id_int = 2
        strategy.active_trade_id_int = 2
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.add_transaction(
            2,
            strategy.previous_bar,
            STITCHED_TRADE_SYMBOL_STR,
            150,
            40.0,
            6_000.0,
            1,
            0.0,
        )

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "regime_flip_bool_ser"): False,
                (SIGNAL_NAMESPACE_STR, "entry_target_weight_ser"): 0.0,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy.iterate(
            pd.DataFrame(index=[strategy.previous_bar]),
            close_row_ser,
            pd.Series({STITCHED_TRADE_SYMBOL_STR: 40.0}, dtype=float),
        )

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_run_daily_smoke_generates_summary(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            run_daily(
                strategy,
                pricing_data_df,
                calendar=pricing_data_df.index,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=None,
            )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)


if __name__ == "__main__":
    unittest.main()
