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
from strategies.strategy_taa_sector_rotation import (
    SectorRotationConfig,
    SectorRotationStrategy,
    compute_sector_rotation_signal_tables,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


class SectorRotationStrategyTests(unittest.TestCase):
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

    def make_strategy(self, **kwargs) -> SectorRotationStrategy:
        base_kwargs = dict(
            name="SectorRotationTest",
            benchmarks=[],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            sector_asset_list=("AAA", "BBB", "CCC", "DDD"),
            fallback_asset="BIL",
            top_n_int=3,
            momentum_lookback_month_int=12,
            momentum_skip_month_int=1,
            trend_sma_month_int=10,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return SectorRotationStrategy(**base_kwargs)

    def make_monthly_signal_close_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-31", periods=13, freq="ME")
        signal_close_df = pd.DataFrame(
            {
                "AAA": [100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 150, 160],
                "BBB": [100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 140, 145],
                "CCC": [100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 145, 90],
                "DDD": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 130, 135],
                "BIL": [100.0] * 13,
            },
            index=date_index,
            dtype=float,
        )
        return signal_close_df

    def make_pricing_data_df(self, periods_int: int = 420) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=periods_int, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 100.0 + 0.25 * step_vec + 0.60 * np.sin(step_vec * 0.03)
        bbb_close_vec = 90.0 + 0.22 * step_vec + 0.50 * np.cos(step_vec * 0.025)
        ccc_close_vec = 80.0 + 0.18 * step_vec + 0.45 * np.sin(step_vec * 0.02)
        ddd_close_vec = 70.0 + 0.12 * step_vec + 0.35 * np.cos(step_vec * 0.018)
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

    def test_compute_sector_rotation_signal_tables_matches_formula_and_redirects_failed_filter(self):
        config = SectorRotationConfig(
            sector_asset_list=("AAA", "BBB", "CCC", "DDD"),
            fallback_asset="BIL",
            benchmark_list=(),
            top_n_int=3,
            momentum_lookback_month_int=12,
            momentum_skip_month_int=1,
            trend_sma_month_int=10,
        )
        signal_close_df = self.make_monthly_signal_close_df()

        (
            monthly_close_df,
            momentum_score_df,
            trend_sma_df,
            trend_pass_bool_df,
            decision_weight_df,
        ) = compute_sector_rotation_signal_tables(signal_close_df=signal_close_df, config=config)

        decision_date_ts = pd.Timestamp("2024-01-31")
        self.assertAlmostEqual(float(momentum_score_df.loc[decision_date_ts, "AAA"]), 0.50)
        self.assertAlmostEqual(float(momentum_score_df.loc[decision_date_ts, "CCC"]), 0.45)
        self.assertAlmostEqual(float(momentum_score_df.loc[decision_date_ts, "BBB"]), 0.40)
        self.assertFalse(bool(trend_pass_bool_df.loc[decision_date_ts, "CCC"]))
        self.assertGreater(float(trend_sma_df.loc[decision_date_ts, "CCC"]), float(monthly_close_df.loc[decision_date_ts, "CCC"]))

        target_weight_ser = decision_weight_df.loc[decision_date_ts]
        self.assertAlmostEqual(float(target_weight_ser.loc["AAA"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BBB"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["CCC"]), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["DDD"]), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BIL"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0)

    def test_map_month_end_decision_dates_uses_next_tradable_open(self):
        decision_date_index = pd.to_datetime(["2024-01-31", "2024-02-29"])
        execution_index = pd.to_datetime(
            [
                "2024-01-30",
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
        pricing_data_df = self.make_pricing_data_df(periods_int=420)

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "momentum_12_1_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "trend_sma_10m_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "trend_pass_bool"), signal_data_df.columns)
        self.assertIn(("AAA", "target_weight_ser"), signal_data_df.columns)
        self.assertIn(("BIL", "target_weight_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_liquidates_zero_target_assets_then_buys_new_targets(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.current_bar = pd.Timestamp("2024-02-01")
        strategy.trade_id_int = 7
        strategy.add_transaction(6, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.add_transaction(7, strategy.previous_bar, "CCC", 5, 100.0, 500.0, 2, 0.0)
        strategy.current_trade_map["AAA"] = 6
        strategy.current_trade_map["CCC"] = 7

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "target_weight_ser"): 1.0 / 3.0,
                ("BBB", "target_weight_ser"): 1.0 / 3.0,
                ("CCC", "target_weight_ser"): 0.0,
                ("DDD", "target_weight_ser"): 0.0,
                ("BIL", "target_weight_ser"): 1.0 / 3.0,
            }
        )
        open_price_ser = pd.Series(
            {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0, "DDD": 100.0, "BIL": 100.0},
            dtype=float,
        )

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 4)
        self.assertEqual(order_list[0].asset, "CCC")
        self.assertEqual(order_list[0].amount, 0)
        self.assertEqual(order_list[0].trade_id, 7)
        self.assertEqual(order_list[1].asset, "AAA")
        self.assertEqual(order_list[1].trade_id, 6)
        self.assertEqual(order_list[2].asset, "BBB")
        self.assertEqual(order_list[2].trade_id, 8)
        self.assertEqual(order_list[3].asset, "BIL")
        self.assertEqual(order_list[3].trade_id, 9)
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertEqual(strategy.current_trade_map["BBB"], 8)
        self.assertEqual(strategy.current_trade_map["BIL"], 9)

    def test_iterate_skips_non_rebalance_dates(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.current_bar = pd.Timestamp("2024-02-02")

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "target_weight_ser"): 1.0 / 3.0,
                ("BBB", "target_weight_ser"): 1.0 / 3.0,
                ("CCC", "target_weight_ser"): 0.0,
                ("DDD", "target_weight_ser"): 0.0,
                ("BIL", "target_weight_ser"): 1.0 / 3.0,
            }
        )
        strategy.iterate(
            pd.DataFrame(index=[strategy.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0, "DDD": 100.0, "BIL": 100.0}, dtype=float),
        )

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_run_daily_smoke_generates_summary(self):
        pricing_data_df = self.make_pricing_data_df(periods_int=420).copy()
        late_decline_index = pricing_data_df.index[-40:]
        ccc_close_vec = np.linspace(
            float(pricing_data_df.loc[late_decline_index[0], ("CCC", "Close")]),
            60.0,
            len(late_decline_index),
        )
        pricing_data_df.loc[late_decline_index, ("CCC", "Open")] = ccc_close_vec - 0.3
        pricing_data_df.loc[late_decline_index, ("CCC", "High")] = ccc_close_vec + 0.6
        pricing_data_df.loc[late_decline_index, ("CCC", "Low")] = ccc_close_vec - 0.6
        pricing_data_df.loc[late_decline_index, ("CCC", "Close")] = ccc_close_vec
        pricing_data_df.loc[late_decline_index, ("CCC", "SignalClose")] = ccc_close_vec

        signal_close_df = pd.DataFrame(
            {
                asset_str: pricing_data_df[(asset_str, "SignalClose")].astype(float)
                for asset_str in ("AAA", "BBB", "CCC", "DDD", "BIL")
            },
            index=pricing_data_df.index,
        )
        _, _, _, _, decision_weight_df = compute_sector_rotation_signal_tables(
            signal_close_df=signal_close_df,
            config=SectorRotationConfig(
                sector_asset_list=("AAA", "BBB", "CCC", "DDD"),
                fallback_asset="BIL",
                benchmark_list=(),
                top_n_int=3,
                momentum_lookback_month_int=12,
                momentum_skip_month_int=1,
                trend_sma_month_int=10,
            ),
        )
        rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(decision_weight_df.index),
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
