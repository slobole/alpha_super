import os
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.data import FredSeriesSnapshot
from alpha.engine.order import MarketOrder
from strategies.taa_df.strategy_taa_df import (
    DefenseFirstStrategy,
    load_cash_return_ser,
    map_month_end_weights_to_rebalance_open_df,
)


class DefenseFirstStrategyTests(unittest.TestCase):
    def make_strategy(self, rebalance_weight_df: pd.DataFrame, tradeable_asset_list: list[str]) -> DefenseFirstStrategy:
        return DefenseFirstStrategy(
            name="DefenseFirstTest",
            benchmarks=[],
            rebalance_weight_df=rebalance_weight_df,
            tradeable_asset_list=tradeable_asset_list,
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_map_month_end_weights_to_rebalance_open_df_uses_first_trading_day_of_next_month(self):
        month_end_weight_df = pd.DataFrame(
            {"GLD": [0.40, 0.20], "SPY": [0.60, 0.80]},
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        )
        execution_index = pd.to_datetime(["2024-02-01", "2024-02-02", "2024-03-01", "2024-03-04"])

        rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(month_end_weight_df, execution_index)

        expected_weight_df = pd.DataFrame(
            {"GLD": [0.40, 0.20], "SPY": [0.60, 0.80]},
            index=pd.to_datetime(["2024-02-01", "2024-03-01"]),
        )
        expected_weight_df.index.name = "rebalance_date"

        pd.testing.assert_frame_equal(rebalance_weight_df, expected_weight_df)

    def test_iterate_submits_no_orders_on_non_rebalance_dates(self):
        rebalance_weight_df = pd.DataFrame(
            {"GLD": [0.25], "SPY": [0.75]},
            index=pd.to_datetime(["2024-02-01"]),
        )
        strategy_obj = self.make_strategy(rebalance_weight_df, ["GLD", "SPY"])
        strategy_obj.current_bar = pd.Timestamp("2024-02-02")
        strategy_obj.previous_bar = pd.Timestamp("2024-02-01")

        open_price_ser = pd.Series({"GLD": 100.0, "SPY": 200.0})

        strategy_obj.iterate(pd.DataFrame(), pd.Series(dtype=float), open_price_ser)

        self.assertEqual(len(strategy_obj.get_orders()), 0)

    def test_iterate_uses_generic_target_helpers_on_rebalance_dates(self):
        rebalance_date_ts = pd.Timestamp("2024-02-01")
        rebalance_weight_df = pd.DataFrame(
            {"GLD": [0.25], "UUP": [0.0], "SPY": [0.75], "DBC": [0.0]},
            index=pd.to_datetime([rebalance_date_ts]),
        )
        strategy_obj = self.make_strategy(rebalance_weight_df, ["GLD", "UUP", "SPY", "DBC"])
        strategy_obj.current_bar = rebalance_date_ts
        strategy_obj.previous_bar = pd.Timestamp("2024-01-31")
        strategy_obj.trade_id_int = 22

        strategy_obj.add_transaction(21, strategy_obj.previous_bar, "GLD", 10, 100.0, 1000.0, 1, 0.0)
        strategy_obj.add_transaction(22, strategy_obj.previous_bar, "UUP", 5, 50.0, 250.0, 2, 0.0)
        strategy_obj.current_trade_map["GLD"] = 21
        strategy_obj.current_trade_map["UUP"] = 22

        open_price_ser = pd.Series({"GLD": 100.0, "UUP": 50.0, "SPY": 200.0, "DBC": 25.0})
        close_price_ser = pd.Series(
            {
                ("GLD", "Close"): 100.0,
                ("UUP", "Close"): 50.0,
                ("SPY", "Close"): 200.0,
                ("DBC", "Close"): 25.0,
            }
        )

        strategy_obj.iterate(pd.DataFrame(), close_price_ser, open_price_ser)

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 3)
        self.assertEqual([order_obj.asset for order_obj in order_list], ["UUP", "GLD", "SPY"])

        liquidation_order_obj = order_list[0]
        self.assertIsInstance(liquidation_order_obj, MarketOrder)
        self.assertEqual(liquidation_order_obj.unit, "shares")
        self.assertTrue(liquidation_order_obj.target)
        self.assertEqual(liquidation_order_obj.amount, 0)
        self.assertEqual(liquidation_order_obj.trade_id, 22)

        resize_order_obj = order_list[1]
        self.assertIsInstance(resize_order_obj, MarketOrder)
        self.assertEqual(resize_order_obj.unit, "percent")
        self.assertTrue(resize_order_obj.target)
        self.assertEqual(resize_order_obj.amount, 0.25)
        self.assertEqual(resize_order_obj.trade_id, 21)

        new_position_order_obj = order_list[2]
        self.assertIsInstance(new_position_order_obj, MarketOrder)
        self.assertEqual(new_position_order_obj.unit, "percent")
        self.assertTrue(new_position_order_obj.target)
        self.assertEqual(new_position_order_obj.amount, 0.75)
        self.assertEqual(new_position_order_obj.trade_id, 23)

        self.assertEqual(strategy_obj.current_trade_map["SPY"], 23)
        self.assertEqual(strategy_obj.trade_id_int, 23)
        self.assertNotIn("DBC", [order_obj.asset for order_obj in order_list])

    def test_load_cash_return_ser_preserves_formula_and_month_end_last_sampling(self):
        dtb3_value_ser = pd.Series(
            [5.00, 6.00, 6.50],
            index=pd.to_datetime(["2024-01-31", "2024-02-28", "2024-02-29"]),
            name="DTB3",
        )
        dtb3_value_ser.index.name = "DATE"
        dtb3_snapshot_obj = FredSeriesSnapshot(
            value_ser=dtb3_value_ser,
            source_name_str="FRED",
            series_id_str="DTB3",
            download_attempt_timestamp_ts=datetime(2024, 3, 1, tzinfo=UTC),
            download_status_str="download_success",
            latest_observation_date_ts=pd.Timestamp("2024-02-29"),
            used_cache_bool=False,
            freshness_business_days_int=0,
        )

        with patch(
            "strategies.taa_df.strategy_taa_df.load_daily_fred_series_snapshot",
            return_value=dtb3_snapshot_obj,
        ) as load_snapshot_mock_obj:
            cash_return_ser = load_cash_return_ser(
                dtb3_csv_path_str="unused.csv",
                dtb3_series_id_str="DTB3",
                as_of_ts=datetime(2024, 3, 1, tzinfo=UTC),
                mode_str="backtest",
            )

        load_snapshot_mock_obj.assert_called_once()

        expected_cash_return_ser = (1.0 + dtb3_value_ser / 100.0) ** (1.0 / 12.0) - 1.0
        expected_cash_return_ser.name = "cash_return"
        pd.testing.assert_series_equal(cash_return_ser, expected_cash_return_ser)

        monthly_cash_return_ser = cash_return_ser.resample("ME").last()
        expected_monthly_cash_return_ser = pd.Series(
            [
                float(expected_cash_return_ser.loc[pd.Timestamp("2024-01-31")]),
                float(expected_cash_return_ser.loc[pd.Timestamp("2024-02-29")]),
            ],
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
            name="cash_return",
        )
        expected_monthly_cash_return_ser.index.name = "DATE"
        pd.testing.assert_series_equal(
            monthly_cash_return_ser,
            expected_monthly_cash_return_ser,
            check_freq=False,
        )


if __name__ == "__main__":
    unittest.main()
