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
from strategies.strategy_taa_sector_rotation_buffered import (
    BufferedSectorRotationConfig,
    BufferedSectorRotationStrategy,
    compute_buffered_sector_signal_tables,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


class BufferedSectorRotationStrategyTests(unittest.TestCase):
    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-02-01",
        decision_date_str: str = "2024-01-31",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> BufferedSectorRotationStrategy:
        base_kwargs = dict(
            name="BufferedSectorRotationTest",
            benchmarks=[],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            sector_asset_list=("AAA", "BBB", "CCC", "DDD", "EEE"),
            market_asset_str="SPY",
            fallback_asset="BIL",
            top_n_int=3,
            hold_rank_threshold_int=4,
            mid_lookback_month_int=6,
            long_lookback_month_int=12,
            momentum_skip_month_int=1,
            trend_sma_month_int=10,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return BufferedSectorRotationStrategy(**base_kwargs)

    def make_monthly_signal_close_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-31", periods=13, freq="ME")
        signal_close_df = pd.DataFrame(
            {
                "AAA": [100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 145, 150],
                "BBB": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 160, 165],
                "CCC": [100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 150, 155],
                "DDD": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 125, 130],
                "EEE": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 115, 80],
                "SPY": [100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 135, 140],
                "BIL": [100.0] * 13,
            },
            index=date_index,
            dtype=float,
        )
        return signal_close_df

    def make_pricing_data_df(self, periods_int: int = 420) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=periods_int, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 100.0 + 0.20 * step_vec + 0.60 * np.sin(step_vec * 0.03)
        bbb_close_vec = 95.0 + 0.22 * step_vec + 0.55 * np.cos(step_vec * 0.028)
        ccc_close_vec = 90.0 + 0.18 * step_vec + 0.50 * np.sin(step_vec * 0.024)
        ddd_close_vec = 85.0 + 0.16 * step_vec + 0.40 * np.cos(step_vec * 0.020)
        eee_close_vec = 80.0 + 0.14 * step_vec + 0.35 * np.sin(step_vec * 0.018)
        spy_close_vec = 300.0 + 0.30 * step_vec + 0.50 * np.sin(step_vec * 0.015)
        bil_close_vec = 100.0 + 0.01 * step_vec

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): aaa_close_vec - 0.3,
                ("AAA", "High"): aaa_close_vec + 0.6,
                ("AAA", "Low"): aaa_close_vec - 0.6,
                ("AAA", "Close"): aaa_close_vec,
                ("AAA", "SignalClose"): aaa_close_vec,
                ("BBB", "Open"): bbb_close_vec - 0.3,
                ("BBB", "High"): bbb_close_vec + 0.6,
                ("BBB", "Low"): bbb_close_vec - 0.6,
                ("BBB", "Close"): bbb_close_vec,
                ("BBB", "SignalClose"): bbb_close_vec,
                ("CCC", "Open"): ccc_close_vec - 0.3,
                ("CCC", "High"): ccc_close_vec + 0.6,
                ("CCC", "Low"): ccc_close_vec - 0.6,
                ("CCC", "Close"): ccc_close_vec,
                ("CCC", "SignalClose"): ccc_close_vec,
                ("DDD", "Open"): ddd_close_vec - 0.3,
                ("DDD", "High"): ddd_close_vec + 0.6,
                ("DDD", "Low"): ddd_close_vec - 0.6,
                ("DDD", "Close"): ddd_close_vec,
                ("DDD", "SignalClose"): ddd_close_vec,
                ("EEE", "Open"): eee_close_vec - 0.3,
                ("EEE", "High"): eee_close_vec + 0.6,
                ("EEE", "Low"): eee_close_vec - 0.6,
                ("EEE", "Close"): eee_close_vec,
                ("EEE", "SignalClose"): eee_close_vec,
                ("SPY", "Open"): spy_close_vec - 0.4,
                ("SPY", "High"): spy_close_vec + 0.8,
                ("SPY", "Low"): spy_close_vec - 0.8,
                ("SPY", "Close"): spy_close_vec,
                ("SPY", "SignalClose"): spy_close_vec,
                ("BIL", "Open"): bil_close_vec - 0.05,
                ("BIL", "High"): bil_close_vec + 0.10,
                ("BIL", "Low"): bil_close_vec - 0.10,
                ("BIL", "Close"): bil_close_vec,
                ("BIL", "SignalClose"): bil_close_vec,
            },
            index=date_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_buffered_sector_signal_tables_matches_mixed_formula(self):
        config = BufferedSectorRotationConfig(
            sector_asset_list=("AAA", "BBB", "CCC", "DDD", "EEE"),
            market_asset_str="SPY",
            fallback_asset="BIL",
            benchmark_list=(),
            top_n_int=3,
            hold_rank_threshold_int=4,
            mid_lookback_month_int=6,
            long_lookback_month_int=12,
            momentum_skip_month_int=1,
            trend_sma_month_int=10,
        )
        signal_close_df = self.make_monthly_signal_close_df()

        (
            monthly_sector_close_df,
            momentum_score_df,
            sector_trend_sma_df,
            sector_trend_pass_bool_df,
            market_trend_sma_ser,
            market_trend_pass_bool_ser,
        ) = compute_buffered_sector_signal_tables(signal_close_df=signal_close_df, config=config)

        decision_date_ts = pd.Timestamp("2024-01-31")
        expected_bbb_score_float = 0.5 * ((160.0 / 130.0) - 1.0) + 0.5 * ((160.0 / 100.0) - 1.0)
        expected_eee_score_float = 0.5 * ((115.0 / 106.0) - 1.0) + 0.5 * ((115.0 / 100.0) - 1.0)

        self.assertAlmostEqual(float(momentum_score_df.loc[decision_date_ts, "BBB"]), expected_bbb_score_float)
        self.assertAlmostEqual(float(momentum_score_df.loc[decision_date_ts, "EEE"]), expected_eee_score_float)
        self.assertFalse(bool(sector_trend_pass_bool_df.loc[decision_date_ts, "EEE"]))
        self.assertTrue(bool(market_trend_pass_bool_ser.loc[decision_date_ts]))
        self.assertGreater(
            float(monthly_sector_close_df.loc[decision_date_ts, "BBB"]),
            float(sector_trend_sma_df.loc[decision_date_ts, "BBB"]),
        )
        self.assertGreater(
            float(signal_close_df.loc[decision_date_ts, "SPY"]),
            float(market_trend_sma_ser.loc[decision_date_ts]),
        )

    def test_map_month_end_decision_dates_uses_next_tradable_open(self):
        decision_date_index = pd.to_datetime(["2024-01-31", "2024-02-29"])
        execution_index = pd.to_datetime(
            [
                "2024-01-31",
                "2024-02-01",
                "2024-02-02",
                "2024-02-29",
                "2024-03-01",
            ]
        )

        rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=decision_date_index,
            execution_index=execution_index,
        )

        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-02-01"), "decision_date_ts"]),
            pd.Timestamp("2024-01-31"),
        )
        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-03-01"), "decision_date_ts"]),
            pd.Timestamp("2024-02-29"),
        )

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "momentum_score_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "trend_sma_10m_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "trend_pass_bool"), signal_data_df.columns)
        self.assertIn(("SPY", "trend_sma_10m_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "trend_pass_bool"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_get_target_weight_ser_keeps_rank4_incumbent_due_to_buffer(self):
        strategy = self.make_strategy()
        current_position_ser = pd.Series({"AAA": 10.0, "BBB": 10.0, "DDD": 10.0, "BIL": 0.0}, dtype=float)

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_score_ser"): 0.50,
                ("BBB", "momentum_score_ser"): 0.40,
                ("CCC", "momentum_score_ser"): 0.45,
                ("DDD", "momentum_score_ser"): 0.30,
                ("EEE", "momentum_score_ser"): 0.20,
                ("AAA", "trend_sma_10m_ser"): 100.0,
                ("BBB", "trend_sma_10m_ser"): 100.0,
                ("CCC", "trend_sma_10m_ser"): 100.0,
                ("DDD", "trend_sma_10m_ser"): 100.0,
                ("EEE", "trend_sma_10m_ser"): 100.0,
                ("AAA", "trend_pass_bool"): True,
                ("BBB", "trend_pass_bool"): True,
                ("CCC", "trend_pass_bool"): True,
                ("DDD", "trend_pass_bool"): True,
                ("EEE", "trend_pass_bool"): True,
                ("SPY", "trend_sma_10m_ser"): 100.0,
                ("SPY", "trend_pass_bool"): True,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(
            close_row_ser=close_row_ser,
            current_position_ser=current_position_ser,
        )

        self.assertAlmostEqual(float(target_weight_ser.loc["AAA"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BBB"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["DDD"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["CCC"]), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["EEE"]), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BIL"]), 0.0)

    def test_get_target_weight_ser_redirects_all_to_bil_when_market_filter_fails(self):
        strategy = self.make_strategy()
        current_position_ser = pd.Series({"AAA": 10.0, "BBB": 10.0, "DDD": 10.0, "BIL": 0.0}, dtype=float)

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_score_ser"): 0.50,
                ("BBB", "momentum_score_ser"): 0.40,
                ("CCC", "momentum_score_ser"): 0.45,
                ("DDD", "momentum_score_ser"): 0.20,
                ("EEE", "momentum_score_ser"): 0.30,
                ("AAA", "trend_sma_10m_ser"): 100.0,
                ("BBB", "trend_sma_10m_ser"): 100.0,
                ("CCC", "trend_sma_10m_ser"): 100.0,
                ("DDD", "trend_sma_10m_ser"): 100.0,
                ("EEE", "trend_sma_10m_ser"): 100.0,
                ("AAA", "trend_pass_bool"): True,
                ("BBB", "trend_pass_bool"): True,
                ("CCC", "trend_pass_bool"): True,
                ("DDD", "trend_pass_bool"): True,
                ("EEE", "trend_pass_bool"): True,
                ("SPY", "trend_sma_10m_ser"): 100.0,
                ("SPY", "trend_pass_bool"): False,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(
            close_row_ser=close_row_ser,
            current_position_ser=current_position_ser,
        )

        self.assertAlmostEqual(float(target_weight_ser.loc["BIL"]), 1.0)
        self.assertAlmostEqual(float(target_weight_ser.drop(labels=["BIL"]).sum()), 0.0)

    def test_iterate_liquidates_and_opens_positions_using_buffered_weights(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.current_bar = pd.Timestamp("2024-02-01")
        strategy.trade_id_int = 7
        strategy.add_transaction(6, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.add_transaction(7, strategy.previous_bar, "BBB", 10, 100.0, 1_000.0, 2, 0.0)
        strategy.current_trade_map["AAA"] = 6
        strategy.current_trade_map["BBB"] = 7

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_score_ser"): 0.50,
                ("BBB", "momentum_score_ser"): 0.10,
                ("CCC", "momentum_score_ser"): 0.45,
                ("DDD", "momentum_score_ser"): 0.40,
                ("EEE", "momentum_score_ser"): 0.30,
                ("AAA", "trend_sma_10m_ser"): 100.0,
                ("BBB", "trend_sma_10m_ser"): 100.0,
                ("CCC", "trend_sma_10m_ser"): 100.0,
                ("DDD", "trend_sma_10m_ser"): 100.0,
                ("EEE", "trend_sma_10m_ser"): 100.0,
                ("AAA", "trend_pass_bool"): True,
                ("BBB", "trend_pass_bool"): True,
                ("CCC", "trend_pass_bool"): False,
                ("DDD", "trend_pass_bool"): True,
                ("EEE", "trend_pass_bool"): True,
                ("SPY", "trend_sma_10m_ser"): 100.0,
                ("SPY", "trend_pass_bool"): True,
            }
        )
        open_price_ser = pd.Series(
            {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0, "DDD": 100.0, "EEE": 100.0, "BIL": 100.0},
            dtype=float,
        )

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 4)
        self.assertEqual(order_list[0].asset, "BBB")
        self.assertEqual(order_list[0].amount, 0)
        self.assertEqual(order_list[1].asset, "AAA")
        self.assertEqual(order_list[1].trade_id, 6)
        self.assertEqual(order_list[2].asset, "DDD")
        self.assertEqual(order_list[2].trade_id, 8)
        self.assertEqual(order_list[3].asset, "BIL")
        self.assertEqual(order_list[3].trade_id, 9)
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))

    def test_run_daily_smoke_generates_summary(self):
        pricing_data_df = self.make_pricing_data_df().copy()
        late_spy_break_index = pricing_data_df.index[-35:]
        spy_close_vec = np.linspace(
            float(pricing_data_df.loc[late_spy_break_index[0], ("SPY", "Close")]),
            250.0,
            len(late_spy_break_index),
        )
        pricing_data_df.loc[late_spy_break_index, ("SPY", "Open")] = spy_close_vec - 0.4
        pricing_data_df.loc[late_spy_break_index, ("SPY", "High")] = spy_close_vec + 0.8
        pricing_data_df.loc[late_spy_break_index, ("SPY", "Low")] = spy_close_vec - 0.8
        pricing_data_df.loc[late_spy_break_index, ("SPY", "Close")] = spy_close_vec
        pricing_data_df.loc[late_spy_break_index, ("SPY", "SignalClose")] = spy_close_vec

        signal_close_df = pd.DataFrame(
            {
                asset_str: pricing_data_df[(asset_str, "SignalClose")].astype(float)
                for asset_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "SPY", "BIL")
            },
            index=pricing_data_df.index,
        )
        (
            _,
            momentum_score_df,
            sector_trend_sma_df,
            _,
            market_trend_sma_ser,
            _,
        ) = compute_buffered_sector_signal_tables(
            signal_close_df=signal_close_df,
            config=BufferedSectorRotationConfig(
                sector_asset_list=("AAA", "BBB", "CCC", "DDD", "EEE"),
                market_asset_str="SPY",
                fallback_asset="BIL",
                benchmark_list=(),
                top_n_int=3,
                hold_rank_threshold_int=4,
                mid_lookback_month_int=6,
                long_lookback_month_int=12,
                momentum_skip_month_int=1,
                trend_sma_month_int=10,
            ),
        )
        valid_momentum_bool_ser = momentum_score_df.notna().all(axis=1)
        valid_sector_trend_bool_ser = sector_trend_sma_df.notna().all(axis=1)
        valid_market_trend_bool_ser = market_trend_sma_ser.notna()
        valid_decision_index = momentum_score_df.index[
            valid_momentum_bool_ser & valid_sector_trend_bool_ser & valid_market_trend_bool_ser
        ]
        rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(valid_decision_index),
            execution_index=pricing_data_df.index,
        )

        strategy = self.make_strategy(rebalance_schedule_df=rebalance_schedule_df)
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
