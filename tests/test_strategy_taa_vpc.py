import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.strategy_taa_vpc import (
    VaradiPercentileChannelsConfig,
    VaradiPercentileChannelsStrategy,
    _compute_varadi_target_weight_ser,
    compute_varadi_percentile_channel_signal_dfs,
    map_month_end_weights_to_rebalance_open_df,
)


class VaradiPercentileChannelsStrategyTests(unittest.TestCase):
    def make_strategy(
        self,
        rebalance_weight_df: pd.DataFrame,
        tradeable_asset_list: list[str],
    ) -> VaradiPercentileChannelsStrategy:
        return VaradiPercentileChannelsStrategy(
            name="VaradiPercentileChannelsTest",
            benchmarks=[],
            rebalance_weight_df=rebalance_weight_df,
            tradeable_asset_list=tradeable_asset_list,
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_pricing_data(self, asset_list: list[str]) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-29", periods=8, freq="D")
        base_close_vec = np.linspace(100.0, 107.0, len(date_index))
        price_field_map: dict[tuple[str, str], np.ndarray] = {}

        for asset_idx_int, asset_str in enumerate(asset_list):
            asset_offset_float = float(asset_idx_int * 5)
            close_vec = base_close_vec + asset_offset_float
            price_field_map[(asset_str, "Open")] = close_vec - 0.5
            price_field_map[(asset_str, "High")] = close_vec + 0.5
            price_field_map[(asset_str, "Low")] = close_vec - 1.0
            price_field_map[(asset_str, "Close")] = close_vec

        pricing_data = pd.DataFrame(price_field_map, index=date_index)
        pricing_data.columns = pd.MultiIndex.from_tuples(pricing_data.columns)
        return pricing_data

    def test_hysteresis_state_switches_only_on_channel_crossings(self):
        config = VaradiPercentileChannelsConfig(
            risk_asset_list=("SPY",),
            cash_proxy_asset="SHY",
            benchmark_list=(),
            channel_lookback_day_vec=(4,),
            volatility_lookback_day_int=2,
        )
        date_index = pd.date_range("2024-01-01", periods=8, freq="D")
        signal_close_df = pd.DataFrame(
            {"SPY": [10.0, 11.0, 12.0, 13.0, 14.0, 13.5, 13.0, 12.0]},
            index=date_index,
        )

        signal_state_df, channel_score_df, volatility_df = compute_varadi_percentile_channel_signal_dfs(
            signal_close_df,
            config,
        )

        self.assertEqual(signal_state_df.loc[date_index[0], ("SPY", 4)], -1.0)
        self.assertEqual(signal_state_df.loc[date_index[3], ("SPY", 4)], 1.0)
        self.assertEqual(signal_state_df.loc[date_index[5], ("SPY", 4)], 1.0)
        self.assertEqual(signal_state_df.loc[date_index[7], ("SPY", 4)], -1.0)
        self.assertTrue(np.isnan(channel_score_df.loc[date_index[2], "SPY"]))
        self.assertTrue(np.isfinite(volatility_df.loc[date_index[3], "SPY"]))

    def test_initial_state_stays_negative_without_upper_channel_breakout(self):
        config = VaradiPercentileChannelsConfig(
            risk_asset_list=("SPY",),
            cash_proxy_asset="SHY",
            benchmark_list=(),
            channel_lookback_day_vec=(4,),
            volatility_lookback_day_int=2,
        )
        date_index = pd.date_range("2024-01-01", periods=7, freq="D")
        signal_close_df = pd.DataFrame(
            {"SPY": [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0]},
            index=date_index,
        )

        signal_state_df, _, _ = compute_varadi_percentile_channel_signal_dfs(signal_close_df, config)

        expected_state_ser = pd.Series(-1.0, index=date_index)
        pd.testing.assert_series_equal(
            signal_state_df.loc[:, ("SPY", 4)],
            expected_state_ser,
            check_names=False,
        )

    def test_target_weight_formula_matches_long_only_inverse_vol_math(self):
        channel_score_ser = pd.Series({"SPY": 1.0, "VNQ": 0.5, "LQD": 0.0, "DBC": -0.5}, dtype=float)
        volatility_ser = pd.Series({"SPY": 0.10, "VNQ": 0.20, "LQD": 0.25, "DBC": 0.25}, dtype=float)

        raw_score_ser, target_weight_ser = _compute_varadi_target_weight_ser(
            channel_score_ser,
            volatility_ser,
            "SHY",
        )

        self.assertAlmostEqual(raw_score_ser.loc["SPY"], 10.0)
        self.assertAlmostEqual(raw_score_ser.loc["VNQ"], 2.5)
        self.assertAlmostEqual(raw_score_ser.loc["LQD"], 0.0)
        self.assertAlmostEqual(raw_score_ser.loc["DBC"], -2.0)
        self.assertAlmostEqual(target_weight_ser.loc["SPY"], 10.0 / 14.5)
        self.assertAlmostEqual(target_weight_ser.loc["VNQ"], 2.5 / 14.5)
        self.assertAlmostEqual(target_weight_ser.loc["LQD"], 0.0)
        self.assertAlmostEqual(target_weight_ser.loc["DBC"], 0.0)
        self.assertAlmostEqual(target_weight_ser.loc["SHY"], 2.0 / 14.5)
        self.assertTrue(np.isclose(float(target_weight_ser.sum()), 1.0, atol=1e-12))

    def test_negative_raw_scores_raise_cash_weight_via_absolute_value_denominator(self):
        channel_score_ser = pd.Series({"SPY": 1.0, "VNQ": -1.0}, dtype=float)
        volatility_ser = pd.Series({"SPY": 0.20, "VNQ": 0.20}, dtype=float)

        _, target_weight_ser = _compute_varadi_target_weight_ser(channel_score_ser, volatility_ser, "SHY")

        self.assertAlmostEqual(target_weight_ser.loc["SPY"], 0.5)
        self.assertAlmostEqual(target_weight_ser.loc["VNQ"], 0.0)
        self.assertAlmostEqual(target_weight_ser.loc["SHY"], 0.5)

    def test_invalid_or_zero_volatility_allocates_fully_to_cash_proxy(self):
        channel_score_ser = pd.Series({"SPY": 1.0, "VNQ": -1.0}, dtype=float)
        volatility_ser = pd.Series({"SPY": 0.0, "VNQ": np.nan}, dtype=float)

        raw_score_ser, target_weight_ser = _compute_varadi_target_weight_ser(channel_score_ser, volatility_ser, "SHY")

        self.assertTrue((raw_score_ser == 0.0).all())
        self.assertAlmostEqual(target_weight_ser.loc["SPY"], 0.0)
        self.assertAlmostEqual(target_weight_ser.loc["VNQ"], 0.0)
        self.assertAlmostEqual(target_weight_ser.loc["SHY"], 1.0)

    def test_map_month_end_weights_to_rebalance_open_df_uses_first_trading_day_of_next_month(self):
        month_end_weight_df = pd.DataFrame(
            {"SPY": [0.60, 0.20], "SHY": [0.40, 0.80]},
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        )
        execution_index = pd.to_datetime(["2024-02-01", "2024-02-05", "2024-03-01", "2024-03-04"])

        rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(month_end_weight_df, execution_index)

        expected_weight_df = pd.DataFrame(
            {"SPY": [0.60, 0.20], "SHY": [0.40, 0.80]},
            index=pd.to_datetime(["2024-02-01", "2024-03-01"]),
        )
        expected_weight_df.index.name = "rebalance_date"

        pd.testing.assert_frame_equal(rebalance_weight_df, expected_weight_df)

    def test_compute_signals_passes_signal_audit_without_future_weight_backfill(self):
        rebalance_weight_df = pd.DataFrame(
            {"SPY": [0.60], "SHY": [0.40]},
            index=pd.to_datetime(["2024-02-01"]),
        )
        strategy = self.make_strategy(rebalance_weight_df, ["SPY", "SHY"])
        pricing_data = self.make_pricing_data(["SPY", "SHY"])

        signal_data = strategy.compute_signals(pricing_data)

        self.assertIn(("SPY", "target_weight"), signal_data.columns)
        self.assertIn(("SHY", "target_weight"), signal_data.columns)

        strategy.audit_signals(pricing_data, signal_data)

    def test_iterate_submits_no_orders_on_non_rebalance_dates(self):
        rebalance_weight_df = pd.DataFrame(
            {"SPY": [0.25], "SHY": [0.75]},
            index=pd.to_datetime(["2024-02-01"]),
        )
        strategy = self.make_strategy(rebalance_weight_df, ["SPY", "SHY"])
        strategy.current_bar = pd.Timestamp("2024-02-02")
        strategy.previous_bar = pd.Timestamp("2024-02-01")

        open_price_ser = pd.Series({"SPY": 100.0, "SHY": 80.0})

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), open_price_ser)

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_liquidates_first_then_submits_positive_target_weight_orders(self):
        rebalance_date = pd.Timestamp("2024-02-01")
        rebalance_weight_df = pd.DataFrame(
            {"SPY": [0.25], "VNQ": [0.0], "SHY": [0.75]},
            index=pd.to_datetime([rebalance_date]),
        )
        strategy = self.make_strategy(rebalance_weight_df, ["SPY", "VNQ", "SHY"])
        strategy.current_bar = rebalance_date
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.trade_id_int = 40

        strategy.add_transaction(31, strategy.previous_bar, "SPY", 10, 100.0, 1000.0, 1, 0.0)
        strategy.add_transaction(40, strategy.previous_bar, "VNQ", 5, 50.0, 250.0, 2, 0.0)
        strategy.current_trade_map["SPY"] = 31
        strategy.current_trade_map["VNQ"] = 40

        open_price_ser = pd.Series({"SPY": 100.0, "VNQ": 50.0, "SHY": 80.0})

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 3)
        self.assertEqual([order.asset for order in order_list], ["VNQ", "SPY", "SHY"])

        liquidation_order = order_list[0]
        self.assertIsInstance(liquidation_order, MarketOrder)
        self.assertEqual(liquidation_order.unit, "shares")
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.amount, 0)
        self.assertEqual(liquidation_order.trade_id, 40)

        resize_order = order_list[1]
        self.assertIsInstance(resize_order, MarketOrder)
        self.assertEqual(resize_order.unit, "percent")
        self.assertTrue(resize_order.target)
        self.assertEqual(resize_order.amount, 0.25)
        self.assertEqual(resize_order.trade_id, 31)

        new_position_order = order_list[2]
        self.assertIsInstance(new_position_order, MarketOrder)
        self.assertEqual(new_position_order.unit, "percent")
        self.assertTrue(new_position_order.target)
        self.assertEqual(new_position_order.amount, 0.75)
        self.assertEqual(new_position_order.trade_id, 41)

        self.assertEqual(strategy.current_trade_map["SHY"], 41)
        self.assertEqual(strategy.trade_id_int, 41)


if __name__ == "__main__":
    unittest.main()
