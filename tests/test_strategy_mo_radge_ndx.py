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
from strategies.strategy_mo_radge_ndx import (
    ATR_WINDOW_INT,
    RadgeMomentumNdxStrategy,
    compute_radge_signal_tables,
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


class RadgeMomentumNdxStrategyTests(unittest.TestCase):
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

    def make_strategy(self, **kwargs) -> RadgeMomentumNdxStrategy:
        base_kwargs = dict(
            name="RadgeMomentumNdxTest",
            benchmarks=["SPY"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str="SPY",
            capital_base=10_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_month_int=12,
            index_trend_window_int=200,
            stock_trend_window_int=100,
            max_positions_int=5,
        )
        base_kwargs.update(kwargs)
        return RadgeMomentumNdxStrategy(**base_kwargs)

    @staticmethod
    def make_close_vec(base_price_float: float, daily_return_vec: np.ndarray) -> np.ndarray:
        return base_price_float * np.cumprod(1.0 + daily_return_vec)

    def make_pricing_data_df(self, periods_int: int = 460) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=periods_int, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_return_vec = 0.0012 + 0.0001 * np.sin(step_vec * 0.05)
        bbb_return_vec = 0.0010 + 0.0001 * np.cos(step_vec * 0.04)
        ccc_return_vec = 0.0008 + 0.0003 * np.sin(step_vec * 0.10)
        ddd_return_vec = 0.0005 + 0.0005 * np.where((step_vec.astype(int) % 8) < 4, 1.0, -1.0)
        eee_return_vec = 0.0004 + 0.0002 * np.cos(step_vec * 0.11)
        fff_return_vec = -0.0001 + 0.0005 * np.sin(step_vec * 0.07)
        spy_return_vec = 0.0006 + 0.00005 * np.sin(step_vec * 0.03)

        close_map = {
            "AAA": self.make_close_vec(100.0, aaa_return_vec),
            "BBB": self.make_close_vec(95.0, bbb_return_vec),
            "CCC": self.make_close_vec(90.0, ccc_return_vec),
            "DDD": self.make_close_vec(85.0, ddd_return_vec),
            "EEE": self.make_close_vec(80.0, eee_return_vec),
            "FFF": self.make_close_vec(75.0, fff_return_vec),
            "SPY": self.make_close_vec(300.0, spy_return_vec),
        }

        pricing_data_map: dict[tuple[str, str], np.ndarray] = {}
        for symbol_str, close_vec in close_map.items():
            pricing_data_map[(symbol_str, "Open")] = close_vec * 0.999
            pricing_data_map[(symbol_str, "High")] = close_vec * 1.010
            pricing_data_map[(symbol_str, "Low")] = close_vec * 0.990
            pricing_data_map[(symbol_str, "Close")] = close_vec

        pricing_data_df = pd.DataFrame(pricing_data_map, index=date_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_get_monthly_decision_close_df_uses_actual_last_tradable_close_and_drops_incomplete_month(self):
        date_index = pd.to_datetime(
            [
                "2024-03-27",
                "2024-03-28",
                "2024-04-01",
                "2024-04-02",
            ]
        )
        price_close_df = pd.DataFrame({"AAA": [10.0, 11.0, 12.0, 13.0]}, index=date_index)

        monthly_decision_close_df = get_monthly_decision_close_df(price_close_df)

        self.assertEqual(monthly_decision_close_df.index.tolist(), [pd.Timestamp("2024-03-28")])
        self.assertAlmostEqual(float(monthly_decision_close_df.loc[pd.Timestamp("2024-03-28"), "AAA"]), 11.0)

    def test_map_month_end_decision_dates_uses_next_tradable_open(self):
        decision_date_index = pd.to_datetime(["2024-03-28", "2024-04-30"])
        execution_index = pd.to_datetime(
            [
                "2024-03-27",
                "2024-03-28",
                "2024-04-01",
                "2024-04-29",
                "2024-04-30",
                "2024-05-01",
            ]
        )

        rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=decision_date_index,
            execution_index=execution_index,
        )

        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-04-01"), "decision_date_ts"]),
            pd.Timestamp("2024-03-28"),
        )
        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-05-01"), "decision_date_ts"]),
            pd.Timestamp("2024-04-30"),
        )

    def test_compute_radge_signal_tables_matches_monthly_roc_formula(self):
        pricing_data_df = self.make_pricing_data_df()
        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        price_high_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "High")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        price_low_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Low")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )

        (
            monthly_decision_close_df,
            monthly_roc_df,
            _atr_decision_df,
            _stock_trend_pass_df,
            _regime_sma_ser,
            _regime_pass_ser,
            _risk_adj_score_df,
        ) = compute_radge_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=pricing_data_df[("SPY", "Close")].astype(float),
        )

        decision_date_ts = pd.Timestamp(monthly_roc_df.index[-1])
        raw_monthly_decision_close_df = get_monthly_decision_close_df(price_close_df)
        expected_monthly_roc_float = (
            float(raw_monthly_decision_close_df.loc[decision_date_ts, "AAA"])
            / float(raw_monthly_decision_close_df.shift(12).loc[decision_date_ts, "AAA"])
            - 1.0
        )

        self.assertAlmostEqual(
            float(monthly_roc_df.loc[decision_date_ts, "AAA"]),
            expected_monthly_roc_float,
            places=12,
        )

    def test_compute_radge_signal_tables_matches_trailing_atr_formula(self):
        pricing_data_df = self.make_pricing_data_df()
        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        price_high_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "High")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        price_low_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Low")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )

        (
            _monthly_decision_close_df,
            monthly_roc_df,
            atr_decision_df,
            _stock_trend_pass_df,
            _regime_sma_ser,
            _regime_pass_ser,
            risk_adj_score_df,
        ) = compute_radge_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=pricing_data_df[("SPY", "Close")].astype(float),
        )

        decision_date_ts = pd.Timestamp(atr_decision_df.index[-1])
        prior_close_ser = price_close_df["AAA"].shift(1)
        true_range_df = pd.concat(
            [
                price_high_df["AAA"] - price_low_df["AAA"],
                (price_high_df["AAA"] - prior_close_ser).abs(),
                (price_low_df["AAA"] - prior_close_ser).abs(),
            ],
            axis=1,
        )
        expected_atr_float = float(
            true_range_df.max(axis=1).rolling(window=ATR_WINDOW_INT, min_periods=ATR_WINDOW_INT).mean().loc[decision_date_ts]
        )

        self.assertAlmostEqual(float(atr_decision_df.loc[decision_date_ts, "AAA"]), expected_atr_float, places=12)
        self.assertAlmostEqual(
            float(risk_adj_score_df.loc[decision_date_ts, "AAA"]),
            float(monthly_roc_df.loc[decision_date_ts, "AAA"]) / expected_atr_float,
            places=12,
        )

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "monthly_roc_12_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "atr_20_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "stock_trend_pass_bool"), signal_data_df.columns)
        self.assertIn(("AAA", "risk_adj_score_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "regime_sma_200_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "regime_pass_bool"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_compute_radge_signal_tables_matches_spy_above_sma_regime_formula(self):
        pricing_data_df = self.make_pricing_data_df()
        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        price_high_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "High")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        price_low_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Low")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        spy_close_ser = pricing_data_df[("SPY", "Close")].astype(float)

        (
            _monthly_decision_close_df,
            _monthly_roc_df,
            _atr_decision_df,
            _stock_trend_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            _risk_adj_score_df,
        ) = compute_radge_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=spy_close_ser,
        )

        decision_date_ts = pd.Timestamp(regime_sma_ser.index[-1])
        expected_regime_sma_float = float(
            spy_close_ser.rolling(window=200, min_periods=200).mean().loc[decision_date_ts]
        )
        expected_regime_pass_bool = bool(spy_close_ser.loc[decision_date_ts] > expected_regime_sma_float)

        self.assertAlmostEqual(float(regime_sma_ser.loc[decision_date_ts]), expected_regime_sma_float, places=12)
        self.assertEqual(bool(regime_pass_ser.loc[decision_date_ts]), expected_regime_pass_bool)

    def test_get_target_weight_ser_uses_pit_universe_trend_filter_and_deterministic_rank(self):
        strategy = self.make_strategy(max_positions_int=5)
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "EEE": [1],
                "FFF": [1],
                "GGG": [1],
                "OUT": [0],
            },
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 1.00,
                ("AAA", "stock_trend_pass_bool"): True,
                ("BBB", "risk_adj_score_ser"): 1.00,
                ("BBB", "stock_trend_pass_bool"): True,
                ("CCC", "risk_adj_score_ser"): 0.90,
                ("CCC", "stock_trend_pass_bool"): True,
                ("DDD", "risk_adj_score_ser"): 0.80,
                ("DDD", "stock_trend_pass_bool"): True,
                ("EEE", "risk_adj_score_ser"): 0.70,
                ("EEE", "stock_trend_pass_bool"): True,
                ("FFF", "risk_adj_score_ser"): 0.60,
                ("FFF", "stock_trend_pass_bool"): True,
                ("GGG", "risk_adj_score_ser"): 1.50,
                ("GGG", "stock_trend_pass_bool"): False,
                ("OUT", "risk_adj_score_ser"): 2.00,
                ("OUT", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["AAA", "BBB", "CCC", "DDD", "EEE"])
        self.assertTrue(np.allclose(target_weight_ser.to_numpy(dtype=float), 0.20))
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0)

    def test_get_target_weight_ser_leaves_cash_when_fewer_than_five_names_qualify(self):
        strategy = self.make_strategy(max_positions_int=5)
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "DDD": [1], "EEE": [1]},
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 1.00,
                ("AAA", "stock_trend_pass_bool"): True,
                ("BBB", "risk_adj_score_ser"): 0.90,
                ("BBB", "stock_trend_pass_bool"): True,
                ("CCC", "risk_adj_score_ser"): 0.80,
                ("CCC", "stock_trend_pass_bool"): True,
                ("DDD", "risk_adj_score_ser"): 0.70,
                ("DDD", "stock_trend_pass_bool"): False,
                ("EEE", "risk_adj_score_ser"): np.nan,
                ("EEE", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["AAA", "BBB", "CCC"])
        self.assertTrue(np.allclose(target_weight_ser.to_numpy(dtype=float), 0.20))
        self.assertAlmostEqual(float(target_weight_ser.sum()), 0.60)

    def test_iterate_submits_no_orders_on_non_rebalance_dates(self):
        strategy = self.make_strategy()
        strategy.current_bar = pd.Timestamp("2024-04-02")
        strategy.previous_bar = pd.Timestamp("2024-04-01")

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"AAA": 100.0}, dtype=float))

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_liquidates_all_holdings_when_regime_fails(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.current_bar = pd.Timestamp("2024-04-01")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1]},
            index=[strategy.previous_bar],
        )
        strategy.add_transaction(7, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.add_transaction(8, strategy.previous_bar, "BBB", 5, 100.0, 500.0, 2, 0.0)
        strategy.current_trade_map["AAA"] = 7
        strategy.current_trade_map["BBB"] = 8

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 1.0,
                ("AAA", "stock_trend_pass_bool"): True,
                ("BBB", "risk_adj_score_ser"): 0.9,
                ("BBB", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): False,
            }
        )

        strategy.iterate(
            pd.DataFrame(index=[strategy.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0, "SPY": 100.0}, dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["AAA", "BBB"])
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertTrue(all(order_obj.target for order_obj in order_list))
        self.assertTrue(all(order_obj.unit == "shares" for order_obj in order_list))
        self.assertTrue(all(order_obj.amount == 0 for order_obj in order_list))

    def test_iterate_submits_zero_target_liquidations_plus_fixed_slot_orders(self):
        strategy = self.make_strategy(max_positions_int=5)
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.current_bar = pd.Timestamp("2024-04-01")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "DDD": [1]},
            index=[strategy.previous_bar],
        )
        strategy.add_transaction(7, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.add_transaction(8, strategy.previous_bar, "BBB", 5, 100.0, 500.0, 2, 0.0)
        strategy.current_trade_map["AAA"] = 7
        strategy.current_trade_map["BBB"] = 8

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 1.00,
                ("AAA", "stock_trend_pass_bool"): True,
                ("BBB", "risk_adj_score_ser"): 0.50,
                ("BBB", "stock_trend_pass_bool"): False,
                ("CCC", "risk_adj_score_ser"): 0.90,
                ("CCC", "stock_trend_pass_bool"): True,
                ("DDD", "risk_adj_score_ser"): 0.80,
                ("DDD", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): True,
            }
        )
        open_price_ser = pd.Series(
            {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0, "DDD": 100.0, "SPY": 100.0},
            dtype=float,
        )

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["BBB", "AAA", "CCC", "DDD"])
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))

        liquidation_order = order_list[0]
        self.assertEqual(liquidation_order.amount, 0)
        self.assertEqual(liquidation_order.unit, "shares")
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.trade_id, 8)

        resize_order = order_list[1]
        self.assertEqual(resize_order.unit, "percent")
        self.assertTrue(resize_order.target)
        self.assertAlmostEqual(float(resize_order.amount), 0.20)
        self.assertEqual(resize_order.trade_id, 7)

        new_order_list = order_list[2:]
        self.assertTrue(all(order_obj.unit == "percent" for order_obj in new_order_list))
        self.assertTrue(all(order_obj.target for order_obj in new_order_list))
        self.assertTrue(all(np.isclose(float(order_obj.amount), 0.20) for order_obj in new_order_list))
        self.assertEqual([order_obj.trade_id for order_obj in new_order_list], [1, 2])
        self.assertEqual(strategy.current_trade_map["CCC"], 1)
        self.assertEqual(strategy.current_trade_map["DDD"], 2)

    def test_run_daily_smoke_generates_summary(self):
        pricing_data_df = self.make_pricing_data_df().copy()
        late_regime_index = pricing_data_df.index[-80:]
        start_spy_close_float = float(pricing_data_df.loc[late_regime_index[0], ("SPY", "Close")])
        spy_close_vec = np.linspace(start_spy_close_float, start_spy_close_float * 0.45, len(late_regime_index))
        pricing_data_df.loc[late_regime_index, ("SPY", "Open")] = spy_close_vec * 0.999
        pricing_data_df.loc[late_regime_index, ("SPY", "High")] = spy_close_vec * 1.010
        pricing_data_df.loc[late_regime_index, ("SPY", "Low")] = spy_close_vec * 0.990
        pricing_data_df.loc[late_regime_index, ("SPY", "Close")] = spy_close_vec

        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        price_high_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "High")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        price_low_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Low")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )
        (
            monthly_decision_close_df,
            _monthly_roc_df,
            _atr_decision_df,
            _stock_trend_pass_df,
            _regime_sma_ser,
            _regime_pass_ser,
            _risk_adj_score_df,
        ) = compute_radge_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=pricing_data_df[("SPY", "Close")].astype(float),
        )
        rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
            execution_index=pricing_data_df.index,
        )

        strategy = self.make_strategy(rebalance_schedule_df=rebalance_schedule_df)
        strategy.universe_df = pd.DataFrame(
            {"AAA": 1, "BBB": 1, "CCC": 1, "DDD": 1, "EEE": 1, "FFF": 1},
            index=pricing_data_df.index,
        )

        calendar_idx = pricing_data_df.index[pricing_data_df.index >= rebalance_schedule_df.index[0]]
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
