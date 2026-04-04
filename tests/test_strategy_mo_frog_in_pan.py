import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.strategy_mo_frog_in_pan import (
    FrogInPanConfig,
    FrogInPanStrategy,
    compute_frog_in_pan_signal_tables,
    get_monthly_decision_close_df,
    map_quarter_end_decision_dates_to_rebalance_schedule_df,
)


class FrogInPanStrategyTests(unittest.TestCase):
    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-04-01",
        decision_date_str: str = "2024-03-28",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> FrogInPanStrategy:
        base_kwargs = dict(
            name="FrogInPanTest",
            benchmarks=["SPY"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str="SPY",
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            momentum_lookback_month_int=12,
            momentum_skip_month_int=1,
            regime_momentum_lookback_month_int=12,
            regime_momentum_skip_month_int=1,
            rebalance_month_interval_int=3,
            top_momentum_quantile_float=0.8,
            top_fip_fraction_float=0.5,
            max_holdings_int=10,
        )
        base_kwargs.update(kwargs)
        return FrogInPanStrategy(**base_kwargs)

    @staticmethod
    def make_close_vec(base_price_float: float, daily_return_vec: np.ndarray) -> np.ndarray:
        return base_price_float * np.cumprod(1.0 + daily_return_vec)

    def make_pricing_data_df(self, periods_int: int = 430) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=periods_int, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_return_vec = 0.0010 + 0.0001 * np.sin(step_vec * 0.05)
        bbb_return_vec = np.where((step_vec.astype(int) % 9) < 6, 0.0012, -0.0002) + 0.00005 * np.cos(step_vec * 0.03)
        ccc_return_vec = 0.0009 + 0.0006 * np.sin(step_vec * 0.18)
        ddd_return_vec = 0.0007 + 0.0008 * np.where((step_vec.astype(int) % 7) < 4, 1.0, -1.0)
        eee_return_vec = 0.0005 + 0.0003 * np.cos(step_vec * 0.11)
        fff_return_vec = -0.0001 + 0.0004 * np.sin(step_vec * 0.07)
        spy_return_vec = 0.0007 + 0.00005 * np.sin(step_vec * 0.03)

        close_map = {
            "AAA": self.make_close_vec(100.0, aaa_return_vec),
            "BBB": self.make_close_vec(95.0, bbb_return_vec),
            "CCC": self.make_close_vec(90.0, ccc_return_vec),
            "DDD": self.make_close_vec(85.0, ddd_return_vec),
            "EEE": self.make_close_vec(80.0, eee_return_vec),
            "FFF": self.make_close_vec(75.0, fff_return_vec),
            "SPY": self.make_close_vec(300.0, spy_return_vec),
        }
        turnover_multiplier_map = {
            "AAA": 650_000.0,
            "BBB": 600_000.0,
            "CCC": 500_000.0,
            "DDD": 350_000.0,
            "EEE": 800_000.0,
            "FFF": 250_000.0,
            "SPY": 2_500_000.0,
        }

        pricing_data_map: dict[tuple[str, str], np.ndarray] = {}
        for symbol_str, close_vec in close_map.items():
            pricing_data_map[(symbol_str, "Open")] = close_vec * 0.999
            pricing_data_map[(symbol_str, "High")] = close_vec * 1.002
            pricing_data_map[(symbol_str, "Low")] = close_vec * 0.998
            pricing_data_map[(symbol_str, "Close")] = close_vec
            pricing_data_map[(symbol_str, "Turnover")] = close_vec * turnover_multiplier_map[symbol_str]

        pricing_data_df = pd.DataFrame(pricing_data_map, index=date_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_map_quarter_end_decision_dates_uses_next_tradable_open(self):
        decision_date_index = pd.to_datetime(["2024-03-28", "2024-06-28"])
        execution_index = pd.to_datetime(
            [
                "2024-03-27",
                "2024-03-28",
                "2024-04-01",
                "2024-06-27",
                "2024-06-28",
                "2024-07-01",
            ]
        )

        rebalance_schedule_df = map_quarter_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=decision_date_index,
            execution_index=execution_index,
        )

        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-04-01"), "decision_date_ts"]),
            pd.Timestamp("2024-03-28"),
        )
        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-07-01"), "decision_date_ts"]),
            pd.Timestamp("2024-06-28"),
        )

    def test_compute_frog_in_pan_signal_tables_matches_momentum_and_fip_formula(self):
        pricing_data_df = self.make_pricing_data_df(periods_int=430)
        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )

        (
            monthly_decision_close_df,
            quarterly_decision_close_df,
            momentum_score_df,
            fip_score_df,
            regime_momentum_ser,
            regime_pass_bool_ser,
        ) = compute_frog_in_pan_signal_tables(
            price_close_df=price_close_df,
            regime_close_ser=pricing_data_df[("SPY", "Close")].astype(float),
            config=FrogInPanConfig(
                regime_symbol_str="SPY",
                benchmark_symbol_list=("SPY",),
                momentum_lookback_month_int=12,
                momentum_skip_month_int=1,
                regime_momentum_lookback_month_int=12,
                regime_momentum_skip_month_int=1,
                rebalance_month_interval_int=3,
                top_momentum_quantile_float=0.10,
                top_fip_fraction_float=0.50,
            ),
        )

        decision_date_ts = pd.Timestamp(quarterly_decision_close_df.index[-1])
        month_loc_int = int(monthly_decision_close_df.index.get_loc(decision_date_ts))
        window_start_ts = pd.Timestamp(monthly_decision_close_df.index[month_loc_int - 12])
        window_end_ts = pd.Timestamp(monthly_decision_close_df.index[month_loc_int - 1])

        expected_momentum_float = (
            float(monthly_decision_close_df.loc[window_end_ts, "AAA"])
            / float(monthly_decision_close_df.loc[window_start_ts, "AAA"])
            - 1.0
        )
        self.assertAlmostEqual(
            float(momentum_score_df.loc[decision_date_ts, "AAA"]),
            expected_momentum_float,
            places=12,
        )

        daily_return_df = price_close_df.pct_change(fill_method=None)
        window_return_df = daily_return_df.loc[
            (daily_return_df.index > window_start_ts) & (daily_return_df.index <= window_end_ts)
        ]
        expected_fip_float = (
            float((window_return_df["BBB"] > 0.0).sum())
            / float(window_return_df["BBB"].notna().sum())
        )
        self.assertAlmostEqual(
            float(fip_score_df.loc[decision_date_ts, "BBB"]),
            expected_fip_float,
            places=12,
        )
        self.assertGreater(float(fip_score_df.loc[decision_date_ts, "AAA"]), float(fip_score_df.loc[decision_date_ts, "DDD"]))
        regime_close_df = pd.DataFrame(
            {"SPY": pricing_data_df[("SPY", "Close")].astype(float)},
            index=pricing_data_df.index,
        )
        monthly_regime_close_df = get_monthly_decision_close_df(regime_close_df)
        expected_regime_momentum_float = (
            float(monthly_regime_close_df.loc[window_end_ts, "SPY"])
            / float(monthly_regime_close_df.loc[window_start_ts, "SPY"])
            - 1.0
        )
        self.assertAlmostEqual(float(regime_momentum_ser.loc[decision_date_ts]), expected_regime_momentum_float, places=12)
        self.assertTrue(bool(regime_pass_bool_ser.loc[decision_date_ts]))

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df(periods_int=430)

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "momentum_12_1_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "fip_score_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "regime_momentum_12_1_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "regime_pass_bool"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_get_target_weight_ser_selects_top_momentum_then_top_fip(self):
        strategy = self.make_strategy(
            top_momentum_quantile_float=0.8,
            top_fip_fraction_float=0.5,
        )
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "DDD": [1], "EEE": [1], "FFF": [1]},
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_12_1_ser"): 0.80,
                ("AAA", "fip_score_ser"): 0.40,
                ("BBB", "momentum_12_1_ser"): 0.70,
                ("BBB", "fip_score_ser"): 0.90,
                ("CCC", "momentum_12_1_ser"): 0.60,
                ("CCC", "fip_score_ser"): 0.80,
                ("DDD", "momentum_12_1_ser"): 0.50,
                ("DDD", "fip_score_ser"): 0.10,
                ("EEE", "momentum_12_1_ser"): 0.40,
                ("EEE", "fip_score_ser"): 0.95,
                ("EEE", "Turnover"): 95_000_000.0,
                ("FFF", "momentum_12_1_ser"): 0.30,
                ("FFF", "fip_score_ser"): 0.85,
                ("FFF", "Turnover"): 15_000_000.0,
                ("AAA", "Turnover"): 40_000_000.0,
                ("BBB", "Turnover"): 80_000_000.0,
                ("CCC", "Turnover"): 70_000_000.0,
                ("DDD", "Turnover"): 20_000_000.0,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["BBB", "CCC"])
        self.assertAlmostEqual(float(target_weight_ser.loc["BBB"]), 0.5)
        self.assertAlmostEqual(float(target_weight_ser.loc["CCC"]), 0.5)
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0)

    def test_get_target_weight_ser_caps_final_holdings_by_turnover(self):
        strategy = self.make_strategy(
            top_momentum_quantile_float=1.0,
            top_fip_fraction_float=1.0,
            max_holdings_int=2,
        )
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "DDD": [1], "EEE": [1]},
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_12_1_ser"): 0.80,
                ("AAA", "fip_score_ser"): 0.60,
                ("AAA", "Turnover"): 40_000_000.0,
                ("BBB", "momentum_12_1_ser"): 0.70,
                ("BBB", "fip_score_ser"): 0.70,
                ("BBB", "Turnover"): 90_000_000.0,
                ("CCC", "momentum_12_1_ser"): 0.60,
                ("CCC", "fip_score_ser"): 0.80,
                ("CCC", "Turnover"): 35_000_000.0,
                ("DDD", "momentum_12_1_ser"): 0.50,
                ("DDD", "fip_score_ser"): 0.90,
                ("DDD", "Turnover"): 25_000_000.0,
                ("EEE", "momentum_12_1_ser"): 0.40,
                ("EEE", "fip_score_ser"): 0.95,
                ("EEE", "Turnover"): 120_000_000.0,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["EEE", "BBB"])
        self.assertAlmostEqual(float(target_weight_ser.loc["EEE"]), 0.5)
        self.assertAlmostEqual(float(target_weight_ser.loc["BBB"]), 0.5)
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0)

    def test_iterate_liquidates_zero_target_assets_then_buys_selected_names(self):
        strategy = self.make_strategy(
            top_momentum_quantile_float=0.8,
            top_fip_fraction_float=0.5,
        )
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.current_bar = pd.Timestamp("2024-04-01")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "DDD": [1], "EEE": [1], "FFF": [1]},
            index=[strategy.previous_bar],
        )
        strategy.add_transaction(7, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.add_transaction(8, strategy.previous_bar, "BBB", 5, 100.0, 500.0, 2, 0.0)
        strategy.current_trade_map["AAA"] = 7
        strategy.current_trade_map["BBB"] = 8

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_12_1_ser"): 0.80,
                ("AAA", "fip_score_ser"): 0.20,
                ("BBB", "momentum_12_1_ser"): 0.70,
                ("BBB", "fip_score_ser"): 0.90,
                ("CCC", "momentum_12_1_ser"): 0.60,
                ("CCC", "fip_score_ser"): 0.80,
                ("DDD", "momentum_12_1_ser"): 0.50,
                ("DDD", "fip_score_ser"): 0.10,
                ("EEE", "momentum_12_1_ser"): 0.40,
                ("EEE", "fip_score_ser"): 0.95,
                ("FFF", "momentum_12_1_ser"): 0.30,
                ("FFF", "fip_score_ser"): 0.85,
                ("AAA", "Turnover"): 40_000_000.0,
                ("BBB", "Turnover"): 80_000_000.0,
                ("CCC", "Turnover"): 70_000_000.0,
                ("DDD", "Turnover"): 20_000_000.0,
                ("EEE", "Turnover"): 95_000_000.0,
                ("FFF", "Turnover"): 15_000_000.0,
                ("SPY", "regime_pass_bool"): True,
            }
        )
        open_price_ser = pd.Series(
            {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0, "DDD": 100.0, "EEE": 100.0, "FFF": 100.0, "SPY": 100.0},
            dtype=float,
        )

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 3)
        self.assertEqual(order_list[0].asset, "AAA")
        self.assertEqual(order_list[0].amount, 0)
        self.assertEqual(order_list[0].trade_id, 7)
        self.assertEqual(order_list[1].asset, "BBB")
        self.assertEqual(order_list[1].trade_id, 8)
        self.assertEqual(order_list[2].asset, "CCC")
        self.assertEqual(order_list[2].trade_id, 1)
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertAlmostEqual(float(order_list[1].amount), 0.5)
        self.assertAlmostEqual(float(order_list[2].amount), 0.5)
        self.assertEqual(strategy.current_trade_map["CCC"], 1)

    def test_get_target_weight_ser_returns_empty_when_regime_momentum_fails(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "DDD": [1], "EEE": [1], "FFF": [1]},
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_12_1_ser"): 0.80,
                ("AAA", "fip_score_ser"): 0.90,
                ("BBB", "momentum_12_1_ser"): 0.70,
                ("BBB", "fip_score_ser"): 0.80,
                ("CCC", "momentum_12_1_ser"): 0.60,
                ("CCC", "fip_score_ser"): 0.70,
                ("DDD", "momentum_12_1_ser"): 0.50,
                ("DDD", "fip_score_ser"): 0.60,
                ("EEE", "momentum_12_1_ser"): 0.40,
                ("EEE", "fip_score_ser"): 0.50,
                ("FFF", "momentum_12_1_ser"): 0.30,
                ("FFF", "fip_score_ser"): 0.40,
                ("AAA", "Turnover"): 40_000_000.0,
                ("BBB", "Turnover"): 80_000_000.0,
                ("CCC", "Turnover"): 70_000_000.0,
                ("DDD", "Turnover"): 20_000_000.0,
                ("EEE", "Turnover"): 95_000_000.0,
                ("FFF", "Turnover"): 15_000_000.0,
                ("SPY", "regime_pass_bool"): False,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(len(target_weight_ser), 0)

    def test_run_daily_smoke_generates_summary(self):
        pricing_data_df = self.make_pricing_data_df(periods_int=430).copy()
        late_rotation_index = pricing_data_df.index[-120:]
        ccc_close_vec = np.linspace(
            float(pricing_data_df.loc[late_rotation_index[0], ("CCC", "Close")]),
            70.0,
            len(late_rotation_index),
        )
        eee_close_vec = np.linspace(
            float(pricing_data_df.loc[late_rotation_index[0], ("EEE", "Close")]),
            float(pricing_data_df.loc[late_rotation_index[0], ("EEE", "Close")]) * 1.80,
            len(late_rotation_index),
        )
        pricing_data_df.loc[late_rotation_index, ("CCC", "Open")] = ccc_close_vec * 0.999
        pricing_data_df.loc[late_rotation_index, ("CCC", "High")] = ccc_close_vec * 1.002
        pricing_data_df.loc[late_rotation_index, ("CCC", "Low")] = ccc_close_vec * 0.998
        pricing_data_df.loc[late_rotation_index, ("CCC", "Close")] = ccc_close_vec
        pricing_data_df.loc[late_rotation_index, ("EEE", "Open")] = eee_close_vec * 0.999
        pricing_data_df.loc[late_rotation_index, ("EEE", "High")] = eee_close_vec * 1.002
        pricing_data_df.loc[late_rotation_index, ("EEE", "Low")] = eee_close_vec * 0.998
        pricing_data_df.loc[late_rotation_index, ("EEE", "Close")] = eee_close_vec

        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        _, quarterly_decision_close_df, _, _, _, _ = compute_frog_in_pan_signal_tables(
            price_close_df=price_close_df,
            regime_close_ser=pricing_data_df[("SPY", "Close")].astype(float),
            config=FrogInPanConfig(
                regime_symbol_str="SPY",
                benchmark_symbol_list=("SPY",),
                momentum_lookback_month_int=12,
                momentum_skip_month_int=1,
                regime_momentum_lookback_month_int=12,
                regime_momentum_skip_month_int=1,
                rebalance_month_interval_int=3,
                top_momentum_quantile_float=0.8,
                top_fip_fraction_float=0.5,
            ),
        )
        rebalance_schedule_df = map_quarter_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(quarterly_decision_close_df.index),
            execution_index=pricing_data_df.index,
        )

        strategy = self.make_strategy(rebalance_schedule_df=rebalance_schedule_df)
        strategy.universe_df = pd.DataFrame(
            {"AAA": 1, "BBB": 1, "CCC": 1, "DDD": 1, "EEE": 1, "FFF": 1},
            index=pricing_data_df.index,
        )

        calendar_idx = pricing_data_df.index[pricing_data_df.index >= rebalance_schedule_df.index[0]]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            run_daily(
                strategy,
                pricing_data_df,
                calendar=calendar_idx,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=None,
                audit_sample_size_int=5,
            )

        self.assertIsNotNone(strategy.summary)
        self.assertGreater(len(strategy.results), 0)
        self.assertGreater(len(strategy.get_transactions()), 0)


if __name__ == "__main__":
    unittest.main()
