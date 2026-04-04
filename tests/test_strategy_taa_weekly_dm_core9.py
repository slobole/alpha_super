import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.strategy_taa_weekly_dm_core9 import (
    WeeklyDualMomentumCore9Config,
    WeeklyDualMomentumCore9Strategy,
    compute_weekly_signal_tables,
    get_weekly_decision_close_df,
    map_weekly_decision_dates_to_rebalance_schedule_df,
)


class WeeklyDualMomentumCore9StrategyTests(unittest.TestCase):
    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-02-05",
        decision_date_str: str = "2024-02-02",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> WeeklyDualMomentumCore9Strategy:
        base_kwargs = dict(
            name="WeeklyDualMomentumCore9Test",
            benchmarks=["$SPX"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            risk_asset_list=("SPY", "VEA", "VWO", "IEF", "TLT", "GLD", "DBC", "VNQ"),
            cash_proxy_str="BIL",
            top_n_int=3,
            momentum_lookback_week_vec=(4, 13, 26, 52),
            trend_sma_week_int=40,
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return WeeklyDualMomentumCore9Strategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2022-01-03", periods=430, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        asset_param_map: dict[str, tuple[float, float, float, float]] = {
            "SPY": (100.0, 0.25, 2.0, 0.03),
            "VEA": (90.0, 0.18, 1.5, 0.04),
            "VWO": (80.0, 0.16, 2.2, 0.05),
            "IEF": (105.0, 0.05, 0.8, 0.02),
            "TLT": (110.0, 0.04, 1.0, 0.025),
            "GLD": (120.0, 0.10, 1.4, 0.035),
            "DBC": (70.0, 0.08, 1.8, 0.045),
            "VNQ": (95.0, 0.12, 1.1, 0.04),
            "BIL": (50.0, 0.01, 0.05, 0.01),
        }

        data_map: dict[tuple[str, str], np.ndarray] = {}
        for asset_str, (base_float, slope_float, amplitude_float, frequency_float) in asset_param_map.items():
            signal_close_vec = (
                base_float
                + (slope_float * step_vec)
                + (amplitude_float * np.sin(step_vec * frequency_float))
            )
            close_vec = signal_close_vec.copy()
            data_map[(asset_str, "Open")] = close_vec - 0.25
            data_map[(asset_str, "High")] = close_vec + 0.75
            data_map[(asset_str, "Low")] = close_vec - 0.75
            data_map[(asset_str, "Close")] = close_vec
            data_map[(asset_str, "SignalClose")] = signal_close_vec

        benchmark_close_vec = 4_000.0 + (3.0 * step_vec) + (20.0 * np.sin(step_vec * 0.025))
        data_map[("$SPX", "Open")] = benchmark_close_vec - 2.0
        data_map[("$SPX", "High")] = benchmark_close_vec + 4.0
        data_map[("$SPX", "Low")] = benchmark_close_vec - 4.0
        data_map[("$SPX", "Close")] = benchmark_close_vec

        pricing_data_df = pd.DataFrame(data_map, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_rotating_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2022-01-03", periods=430, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        asset_param_map: dict[str, tuple[float, float, float, float, float]] = {
            "SPY": (100.0, 0.08, 12.0, 0.05, 0.0),
            "VEA": (95.0, 0.06, 11.0, 0.05, 0.8),
            "VWO": (85.0, 0.04, 10.0, 0.05, 1.6),
            "IEF": (105.0, 0.02, 8.0, 0.04, 2.4),
            "TLT": (110.0, 0.01, 9.0, 0.04, 3.2),
            "GLD": (120.0, 0.03, 10.0, 0.045, 4.0),
            "DBC": (75.0, -0.01, 13.0, 0.05, 4.8),
            "VNQ": (90.0, 0.00, 12.0, 0.05, 5.6),
            "BIL": (50.0, 0.01, 0.05, 0.01, 0.0),
        }

        data_map: dict[tuple[str, str], np.ndarray] = {}
        for asset_str, (base_float, slope_float, amplitude_float, frequency_float, phase_float) in asset_param_map.items():
            signal_close_vec = (
                base_float
                + (slope_float * step_vec)
                + (amplitude_float * np.sin(step_vec * frequency_float + phase_float))
            )
            close_vec = signal_close_vec.copy()
            data_map[(asset_str, "Open")] = close_vec - 0.25
            data_map[(asset_str, "High")] = close_vec + 0.75
            data_map[(asset_str, "Low")] = close_vec - 0.75
            data_map[(asset_str, "Close")] = close_vec
            data_map[(asset_str, "SignalClose")] = signal_close_vec

        benchmark_close_vec = 4_000.0 + (2.0 * step_vec) + (40.0 * np.sin(step_vec * 0.03))
        data_map[("$SPX", "Open")] = benchmark_close_vec - 2.0
        data_map[("$SPX", "High")] = benchmark_close_vec + 4.0
        data_map[("$SPX", "Low")] = benchmark_close_vec - 4.0
        data_map[("$SPX", "Close")] = benchmark_close_vec

        pricing_data_df = pd.DataFrame(data_map, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_get_weekly_decision_close_df_uses_actual_last_tradable_close_of_week(self):
        signal_close_df = pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
                "BIL": [50.0, 50.01, 50.02, 50.03, 50.04],
            },
            index=pd.to_datetime(
                [
                    "2024-03-25",
                    "2024-03-26",
                    "2024-03-27",
                    "2024-03-28",
                    "2024-04-01",
                ]
            ),
        )

        weekly_decision_close_df = get_weekly_decision_close_df(signal_close_df)

        self.assertEqual(list(weekly_decision_close_df.index), [pd.Timestamp("2024-03-28")])

    def test_map_weekly_decision_dates_to_rebalance_schedule_df_uses_next_tradable_open(self):
        decision_date_index = pd.to_datetime(["2024-02-16", "2024-03-28"])
        execution_index = pd.to_datetime(
            [
                "2024-02-12",
                "2024-02-13",
                "2024-02-14",
                "2024-02-15",
                "2024-02-16",
                "2024-02-20",
                "2024-03-25",
                "2024-03-26",
                "2024-03-27",
                "2024-03-28",
                "2024-04-01",
                "2024-04-02",
            ]
        )

        rebalance_schedule_df = map_weekly_decision_dates_to_rebalance_schedule_df(
            decision_date_index=decision_date_index,
            execution_index=execution_index,
        )

        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-02-20"), "decision_date_ts"]),
            pd.Timestamp("2024-02-16"),
        )
        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-04-01"), "decision_date_ts"]),
            pd.Timestamp("2024-03-28"),
        )

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("SPY", "momentum_score_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "trend_pass_bool"), signal_data_df.columns)
        self.assertIn(("SPY", f"trend_sma_{strategy.trend_sma_week_int}w_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "target_weight_ser"), signal_data_df.columns)
        self.assertIn(("BIL", "target_weight_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_momentum_score_matches_explicit_4_13_26_52_week_formula(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()
        signal_data_df = strategy.compute_signals(pricing_data_df)
        signal_close_df = pricing_data_df.xs("SignalClose", axis=1, level=1)[strategy.tradeable_asset_list]
        weekly_close_df, _, trend_sma_df, _, _ = compute_weekly_signal_tables(
            signal_close_df=signal_close_df,
            config=WeeklyDualMomentumCore9Config(
                risk_asset_list=tuple(strategy.risk_asset_list),
                cash_proxy_str=strategy.cash_proxy_str,
                benchmark_list=tuple(strategy._benchmarks),
                top_n_int=strategy.top_n_int,
                momentum_lookback_week_vec=tuple(strategy.momentum_lookback_week_vec),
                trend_sma_week_int=strategy.trend_sma_week_int,
            ),
        )
        decision_date_ts = trend_sma_df.dropna().index[-1]
        spy_weekly_close_ser = weekly_close_df["SPY"]

        manual_momentum_float = 0.0
        for lookback_week_int in strategy.momentum_lookback_week_vec:
            manual_momentum_float += (
                float(spy_weekly_close_ser.loc[decision_date_ts] / spy_weekly_close_ser.shift(lookback_week_int).loc[decision_date_ts])
                - 1.0
            )
        manual_momentum_float /= float(len(strategy.momentum_lookback_week_vec))

        manual_trend_pass_bool = bool(
            spy_weekly_close_ser.loc[decision_date_ts] > spy_weekly_close_ser.rolling(strategy.trend_sma_week_int).mean().loc[decision_date_ts]
        )

        self.assertAlmostEqual(
            float(signal_data_df.loc[decision_date_ts, ("SPY", "momentum_score_ser")]),
            manual_momentum_float,
        )
        self.assertEqual(
            bool(signal_data_df.loc[decision_date_ts, ("SPY", "trend_pass_bool")]),
            manual_trend_pass_bool,
        )

    def test_compute_weekly_signal_tables_excludes_below_trend_assets_and_sends_residual_to_bil(self):
        config = WeeklyDualMomentumCore9Config(
            risk_asset_list=("AAA", "BBB", "CCC", "DDD"),
            cash_proxy_str="BIL",
            benchmark_list=("$SPX",),
            top_n_int=3,
            momentum_lookback_week_vec=(1, 2),
            trend_sma_week_int=3,
        )
        signal_close_df = pd.DataFrame(
            {
                "AAA": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "BBB": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
                "CCC": [10.0, 12.0, 14.0, 9.0, 8.0, 7.0],
                "DDD": [10.0, 9.5, 9.0, 8.5, 8.0, 7.5],
                "BIL": [50.0, 50.02, 50.04, 50.06, 50.08, 50.10],
            },
            index=pd.date_range("2024-01-05", periods=6, freq="W-FRI"),
        )

        _, _, _, _, decision_weight_df = compute_weekly_signal_tables(signal_close_df, config)
        target_weight_ser = decision_weight_df.iloc[-1]

        self.assertAlmostEqual(float(target_weight_ser.loc["AAA"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BBB"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["CCC"]), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["DDD"]), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BIL"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0)

    def test_compute_weekly_signal_tables_sends_full_weight_to_bil_when_no_assets_pass(self):
        config = WeeklyDualMomentumCore9Config(
            risk_asset_list=("AAA", "BBB", "CCC"),
            cash_proxy_str="BIL",
            benchmark_list=("$SPX",),
            top_n_int=3,
            momentum_lookback_week_vec=(1, 2),
            trend_sma_week_int=3,
        )
        signal_close_df = pd.DataFrame(
            {
                "AAA": [15.0, 14.0, 13.0, 12.0, 11.0, 10.0],
                "BBB": [14.0, 13.0, 12.0, 11.0, 10.0, 9.0],
                "CCC": [13.0, 12.0, 11.0, 10.0, 9.0, 8.0],
                "BIL": [50.0, 50.02, 50.04, 50.06, 50.08, 50.10],
            },
            index=pd.date_range("2024-01-05", periods=6, freq="W-FRI"),
        )

        _, _, _, _, decision_weight_df = compute_weekly_signal_tables(signal_close_df, config)
        target_weight_ser = decision_weight_df.iloc[-1]

        self.assertAlmostEqual(float(target_weight_ser.loc["AAA"]), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BBB"]), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["CCC"]), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BIL"]), 1.0)

    def test_iterate_submits_no_orders_on_non_rebalance_dates(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-02-02")
        strategy.current_bar = pd.Timestamp("2024-02-06")

        close_row_ser = self.make_close_row_ser(
            {
                ("SPY", "target_weight_ser"): 1.0 / 3.0,
                ("VEA", "target_weight_ser"): 1.0 / 3.0,
                ("BIL", "target_weight_ser"): 1.0 / 3.0,
            }
        )
        open_price_ser = pd.Series({"SPY": 100.0, "VEA": 80.0, "BIL": 50.0})

        strategy.iterate(pd.DataFrame(), close_row_ser, open_price_ser)

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_liquidates_then_resizes_and_opens_positions(self):
        strategy = self.make_strategy(
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            risk_asset_list=("AAA", "BBB", "CCC"),
            cash_proxy_str="BIL",
            momentum_lookback_week_vec=(1, 2),
            trend_sma_week_int=3,
        )
        strategy.previous_bar = pd.Timestamp("2024-02-02")
        strategy.current_bar = pd.Timestamp("2024-02-05")
        strategy.trade_id_int = 12

        strategy.add_transaction(11, strategy.previous_bar, "AAA", 100, 100.0, 10_000.0, 1, 0.0)
        strategy.add_transaction(12, strategy.previous_bar, "CCC", 50, 40.0, 2_000.0, 2, 0.0)
        strategy.current_trade_map["AAA"] = 11
        strategy.current_trade_map["CCC"] = 12

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "target_weight_ser"): 1.0 / 3.0,
                ("BBB", "target_weight_ser"): 1.0 / 3.0,
                ("CCC", "target_weight_ser"): 0.0,
                ("BIL", "target_weight_ser"): 1.0 / 3.0,
            }
        )
        open_price_ser = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 40.0, "BIL": 100.0})

        strategy.iterate(pd.DataFrame(), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 4)
        self.assertEqual([order.asset for order in order_list], ["CCC", "AAA", "BBB", "BIL"])

        liquidation_order = order_list[0]
        self.assertIsInstance(liquidation_order, MarketOrder)
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.unit, "shares")
        self.assertEqual(liquidation_order.amount, 0)
        self.assertEqual(liquidation_order.trade_id, 12)

        resize_order = order_list[1]
        self.assertIsInstance(resize_order, MarketOrder)
        self.assertTrue(resize_order.target)
        self.assertEqual(resize_order.unit, "percent")
        self.assertAlmostEqual(resize_order.amount, 1.0 / 3.0)
        self.assertEqual(resize_order.trade_id, 11)

        new_bbb_order = order_list[2]
        self.assertEqual(new_bbb_order.asset, "BBB")
        self.assertEqual(new_bbb_order.trade_id, 13)
        self.assertTrue(new_bbb_order.target)
        self.assertEqual(new_bbb_order.unit, "percent")

        new_bil_order = order_list[3]
        self.assertEqual(new_bil_order.asset, "BIL")
        self.assertEqual(new_bil_order.trade_id, 14)
        self.assertTrue(new_bil_order.target)
        self.assertEqual(new_bil_order.unit, "percent")

    def test_run_daily_smoke_generates_summary(self):
        pricing_data_df = self.make_rotating_pricing_data_df()
        signal_close_df = pricing_data_df.xs("SignalClose", axis=1, level=1)[
            ["SPY", "VEA", "VWO", "IEF", "TLT", "GLD", "DBC", "VNQ", "BIL"]
        ]
        config = WeeklyDualMomentumCore9Config()
        _, _, _, _, decision_weight_df = compute_weekly_signal_tables(signal_close_df, config)
        rebalance_schedule_df = map_weekly_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(decision_weight_df.index),
            execution_index=pricing_data_df.index,
        )

        strategy = self.make_strategy(rebalance_schedule_df=rebalance_schedule_df)
        calendar_idx = pricing_data_df.index[pricing_data_df.index >= rebalance_schedule_df.index[0]]

        run_daily(
            strategy,
            pricing_data_df,
            calendar=calendar_idx,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=None,
        )

        self.assertIsNotNone(strategy.summary)
        self.assertIsNotNone(strategy.summary_trades)
        self.assertGreater(len(strategy.get_transactions()), 0)


if __name__ == "__main__":
    unittest.main()
