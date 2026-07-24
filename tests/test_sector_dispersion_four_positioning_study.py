import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from scripts.research.run_sector_dispersion_four_positioning_study import (
    STUDY_UNIVERSE_ID_TUPLE,
    VARIANT_SPEC_TUPLE,
    FourPositioningConfig,
    SectorDispersionFourPositioningStrategy,
    compute_inverse_vol_target_weight_df,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import DEFAULT_CONFIG


class SectorDispersionFourPositioningStudyTests(unittest.TestCase):
    def make_config_obj(
        self,
        variant_id_str: str,
        slippage_float: float = 0.0,
        commission_per_share_float: float = 0.0,
    ) -> FourPositioningConfig:
        base_config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB"),
            benchmark_symbol_str="$SPX",
            history_start_date_str="2023-12-01",
            backtest_start_date_str="2024-01-02",
            range_vol_lookback_day_int=3,
            capital_base_float=100_000.0,
            portfolio_leverage_float=1.0,
            slippage_float=slippage_float,
            commission_per_share_float=commission_per_share_float,
            commission_minimum_float=0.0,
        )
        variant_spec_obj = next(
            spec_obj
            for spec_obj in VARIANT_SPEC_TUPLE
            if spec_obj.variant_id_str == variant_id_str
        )
        return FourPositioningConfig(
            base_config_obj=base_config_obj,
            variant_spec_obj=variant_spec_obj,
            inverse_vol_lookback_day_int=3,
            spy_sma_lookback_day_int=3,
        )

    def make_strategy_obj(
        self,
        variant_id_str: str,
        slippage_float: float = 0.0,
        commission_per_share_float: float = 0.0,
    ) -> SectorDispersionFourPositioningStrategy:
        return SectorDispersionFourPositioningStrategy(
            name=f"test_{variant_id_str.lower()}",
            benchmarks=[],
            positioning_config_obj=self.make_config_obj(
                variant_id_str=variant_id_str,
                slippage_float=slippage_float,
                commission_per_share_float=commission_per_share_float,
            ),
        )

    def test_variant_and_universe_manifests_are_frozen(self):
        self.assertEqual(
            tuple(spec_obj.variant_id_str for spec_obj in VARIANT_SPEC_TUPLE),
            (
                "B0_REF",
                "P0_STRICT",
                "P1_INVOL20",
                "P2_SOFT200",
                "P3_INVOL20_SOFT200",
            ),
        )
        self.assertEqual(STUDY_UNIVERSE_ID_TUPLE, ("spdr_9", "vanguard_11", "spdr_11"))
        strict_cap_dict = {
            spec_obj.variant_id_str: spec_obj.strict_cash_cap_bool
            for spec_obj in VARIANT_SPEC_TUPLE
        }
        self.assertFalse(strict_cap_dict["B0_REF"])
        self.assertTrue(all(strict_cap_dict[variant_id_str] for variant_id_str in strict_cap_dict if variant_id_str != "B0_REF"))

    def test_inverse_vol_weights_are_causal_full_universe_weights(self):
        date_index = pd.bdate_range("2024-01-02", periods=8)
        close_price_df = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0],
                "BBB": [100.0, 104.0, 96.0, 105.0, 95.0, 106.0, 94.0, 107.0],
            },
            index=date_index,
        )

        volatility_ann_df, target_weight_df = compute_inverse_vol_target_weight_df(
            close_price_df=close_price_df,
            lookback_day_int=3,
        )

        self.assertGreater(
            float(volatility_ann_df.loc[date_index[-1], "BBB"]),
            float(volatility_ann_df.loc[date_index[-1], "AAA"]),
        )
        self.assertGreater(
            float(target_weight_df.loc[date_index[-1], "AAA"]),
            float(target_weight_df.loc[date_index[-1], "BBB"]),
        )
        self.assertAlmostEqual(float(target_weight_df.loc[date_index[-1]].sum()), 1.0)

        changed_close_price_df = close_price_df.copy()
        changed_close_price_df.loc[date_index[-1], "BBB"] = 500.0
        _, changed_target_weight_df = compute_inverse_vol_target_weight_df(
            close_price_df=changed_close_price_df,
            lookback_day_int=3,
        )
        pd.testing.assert_frame_equal(
            target_weight_df.iloc[:-1],
            changed_target_weight_df.iloc[:-1],
        )

    def test_soft_sma_and_inverse_vol_features_use_completed_close(self):
        date_index = pd.bdate_range("2024-01-02", periods=8)
        pricing_column_dict: dict[tuple[str, str], pd.Series] = {}
        aaa_close_ser = pd.Series([100.0, 99.0, 101.0, 100.0, 102.0, 101.0, 103.0, 102.0], index=date_index)
        bbb_close_ser = pd.Series([50.0, 52.0, 49.0, 53.0, 48.0, 54.0, 47.0, 55.0], index=date_index)
        for symbol_str, close_ser in {"AAA": aaa_close_ser, "BBB": bbb_close_ser}.items():
            pricing_column_dict[(symbol_str, "Open")] = close_ser
            pricing_column_dict[(symbol_str, "High")] = close_ser + 2.0
            pricing_column_dict[(symbol_str, "Low")] = close_ser - 2.0
            pricing_column_dict[(symbol_str, "Close")] = close_ser
            pricing_column_dict[(symbol_str, "Dividend")] = pd.Series(0.0, index=date_index)
        spy_close_ser = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0], index=date_index)
        pricing_column_dict[("SPY", "Close")] = spy_close_ser
        pricing_column_dict[("$SPX", "Close")] = spy_close_ser
        pricing_data_df = pd.DataFrame(pricing_column_dict, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

        strategy_obj = self.make_strategy_obj("P3_INVOL20_SOFT200")
        signal_data_df = strategy_obj.compute_signals(pricing_data_df)

        expected_sma_float = float(spy_close_ser.iloc[-3:].mean())
        self.assertAlmostEqual(
            float(signal_data_df.loc[date_index[-1], ("SPY", "sma_3_ser")]),
            expected_sma_float,
        )
        self.assertAlmostEqual(
            float(signal_data_df.loc[date_index[-1], ("SPY", "market_scale_ser")]),
            0.5,
        )
        self.assertAlmostEqual(
            float(
                signal_data_df.loc[
                    date_index[-1],
                    [("AAA", "inverse_vol_target_weight_ser"), ("BBB", "inverse_vol_target_weight_ser")],
                ].sum()
            ),
            1.0,
        )

    def test_entry_target_weights_match_all_four_positioning_rules(self):
        entry_row_ser = pd.Series(
            {
                ("AAA", "entry_signal_bool"): True,
                ("AAA", "Close"): 100.0,
                ("AAA", "inverse_vol_target_weight_ser"): 0.8,
                ("BBB", "inverse_vol_target_weight_ser"): 0.2,
                ("SPY", "market_scale_ser"): 0.5,
            }
        )
        expected_weight_dict = {
            "P0_STRICT": 0.50,
            "P1_INVOL20": 0.80,
            "P2_SOFT200": 0.25,
            "P3_INVOL20_SOFT200": 0.40,
        }

        for variant_id_str, expected_weight_float in expected_weight_dict.items():
            with self.subTest(variant_id_str=variant_id_str):
                strategy_obj = self.make_strategy_obj(variant_id_str)
                strategy_obj.iterate(pd.DataFrame(), entry_row_ser, pd.Series(dtype=float))
                self.assertEqual(len(strategy_obj.get_orders()), 1)
                self.assertAlmostEqual(
                    float(strategy_obj.get_orders()[0].amount),
                    100_000.0 * expected_weight_float / 100.0,
                )

    def test_strict_cash_cap_scales_gap_entries_after_costs(self):
        strategy_obj = self.make_strategy_obj(
            "P0_STRICT",
            slippage_float=0.00025,
            commission_per_share_float=0.00525,
        )
        strategy_obj.current_bar = pd.Timestamp("2024-01-03")
        strategy_obj.order_target("AAA", 500.0, trade_id=1)
        strategy_obj.order_target("BBB", 500.0, trade_id=2)
        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): [120.0],
                ("AAA", "High"): [121.0],
                ("AAA", "Low"): [119.0],
                ("AAA", "Close"): [120.0],
                ("AAA", "Dividend"): [0.0],
                ("BBB", "Open"): [120.0],
                ("BBB", "High"): [121.0],
                ("BBB", "Low"): [119.0],
                ("BBB", "Close"): [120.0],
                ("BBB", "Dividend"): [0.0],
            },
            index=pd.DatetimeIndex([strategy_obj.current_bar]),
        )

        strategy_obj.process_orders(pricing_data_df)

        expected_per_share_cash_float = 120.0 * 1.00025 + 0.00525
        expected_scale_float = 100_000.0 / (1_000.0 * expected_per_share_cash_float)
        transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
        self.assertEqual(len(transaction_df), 2)
        self.assertAlmostEqual(float(transaction_df.iloc[0]["amount"]), 500.0 * expected_scale_float, places=7)
        self.assertAlmostEqual(float(transaction_df.iloc[1]["amount"]), 500.0 * expected_scale_float, places=7)
        self.assertGreaterEqual(float(strategy_obj.cash), -1e-7)
        self.assertLessEqual(float(strategy_obj.cash), 1e-5)
        self.assertEqual(len(strategy_obj.cash_cap_event_df), 1)
        self.assertAlmostEqual(
            float(strategy_obj.cash_cap_event_df.iloc[0]["cash_cap_scale_float"]),
            expected_scale_float,
            places=7,
        )

    def test_reference_row_preserves_uncapped_negative_cash_behavior(self):
        strategy_obj = self.make_strategy_obj("B0_REF")
        strategy_obj.current_bar = pd.Timestamp("2024-01-03")
        strategy_obj.order_target("AAA", 500.0, trade_id=1)
        strategy_obj.order_target("BBB", 500.0, trade_id=2)
        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): [120.0],
                ("AAA", "High"): [121.0],
                ("AAA", "Low"): [119.0],
                ("AAA", "Close"): [120.0],
                ("AAA", "Dividend"): [0.0],
                ("BBB", "Open"): [120.0],
                ("BBB", "High"): [121.0],
                ("BBB", "Low"): [119.0],
                ("BBB", "Close"): [120.0],
                ("BBB", "Dividend"): [0.0],
            },
            index=pd.DatetimeIndex([strategy_obj.current_bar]),
        )

        strategy_obj.process_orders(pricing_data_df)

        self.assertAlmostEqual(float(strategy_obj.cash), -20_000.0)
        self.assertTrue(strategy_obj.cash_cap_event_df.empty)

    def test_tiny_cash_residue_without_buy_orders_is_not_a_cap_event(self):
        strategy_obj = self.make_strategy_obj("P0_STRICT")
        strategy_obj.current_bar = pd.Timestamp("2024-01-03")
        strategy_obj.cash = -1e-10
        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): [100.0],
                ("AAA", "High"): [101.0],
                ("AAA", "Low"): [99.0],
                ("AAA", "Close"): [100.0],
                ("AAA", "Dividend"): [0.0],
                ("BBB", "Open"): [50.0],
                ("BBB", "High"): [51.0],
                ("BBB", "Low"): [49.0],
                ("BBB", "Close"): [50.0],
                ("BBB", "Dividend"): [0.0],
            },
            index=pd.DatetimeIndex([strategy_obj.current_bar]),
        )

        strategy_obj.process_orders(pricing_data_df)

        self.assertTrue(strategy_obj.cash_cap_event_df.empty)
        self.assertAlmostEqual(float(strategy_obj.cash), -1e-10)

    def test_strict_cap_uses_dividend_and_same_open_exit_to_fund_buy(self):
        strategy_obj = self.make_strategy_obj("P0_STRICT")
        strategy_obj.current_bar = pd.Timestamp("2024-01-03")
        strategy_obj.cash = 0.0
        strategy_obj._position_amount_map = {"AAA": 500.0}
        strategy_obj.order_target("AAA", 0.0, trade_id=1)
        strategy_obj.order_target("BBB", 510.0, trade_id=2)
        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): [100.0],
                ("AAA", "High"): [101.0],
                ("AAA", "Low"): [99.0],
                ("AAA", "Close"): [100.0],
                ("AAA", "Dividend"): [1.0],
                ("BBB", "Open"): [100.0],
                ("BBB", "High"): [101.0],
                ("BBB", "Low"): [99.0],
                ("BBB", "Close"): [100.0],
                ("BBB", "Dividend"): [0.0],
            },
            index=pd.DatetimeIndex([strategy_obj.current_bar]),
        )

        strategy_obj.process_orders(pricing_data_df)

        expected_available_cash_float = 500.0 + 500.0 * 100.0
        expected_scale_float = expected_available_cash_float / (510.0 * 100.0)
        transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
        self.assertEqual(len(transaction_df), 2)
        self.assertAlmostEqual(float(transaction_df.iloc[0]["amount"]), -500.0)
        self.assertAlmostEqual(
            float(transaction_df.iloc[1]["amount"]),
            510.0 * expected_scale_float,
        )
        self.assertAlmostEqual(float(strategy_obj.dividend_cash_total_float), 500.0)
        self.assertAlmostEqual(float(strategy_obj.cash), 0.0, places=8)
        self.assertEqual(len(strategy_obj.cash_cap_event_df), 1)
        self.assertAlmostEqual(
            float(
                strategy_obj.cash_cap_event_df.iloc[0][
                    "cash_after_exits_before_buys_float"
                ]
            ),
            expected_available_cash_float,
        )
        self.assertAlmostEqual(
            float(strategy_obj.cash_cap_event_df.iloc[0]["cash_cap_scale_float"]),
            expected_scale_float,
        )

    def test_soft_position_signal_fills_half_slot_at_next_open(self):
        date_index = pd.bdate_range("2024-01-02", periods=8)
        signal_day_int = 4
        fill_day_int = 5
        log_range_vec = np.array([0.010, 0.011, 0.012, 0.013, 0.120, 0.010, 0.011, 0.012])
        low_vec = np.full(len(date_index), 100.0)
        high_vec = low_vec * np.exp(log_range_vec)
        aaa_ibs_vec = np.full(len(date_index), 0.50)
        aaa_ibs_vec[signal_day_int] = 0.05
        aaa_close_vec = low_vec + aaa_ibs_vec * (high_vec - low_vec)
        aaa_open_vec = aaa_close_vec.copy()
        aaa_open_vec[fill_day_int] = 110.0
        bbb_close_vec = low_vec + 0.50 * (high_vec - low_vec)
        pricing_column_dict: dict[tuple[str, str], pd.Series] = {}
        for symbol_str, open_vec, close_vec in (
            ("AAA", aaa_open_vec, aaa_close_vec),
            ("BBB", bbb_close_vec, bbb_close_vec),
        ):
            pricing_column_dict[(symbol_str, "Open")] = pd.Series(open_vec, index=date_index)
            pricing_column_dict[(symbol_str, "High")] = pd.Series(high_vec, index=date_index)
            pricing_column_dict[(symbol_str, "Low")] = pd.Series(low_vec, index=date_index)
            pricing_column_dict[(symbol_str, "Close")] = pd.Series(close_vec, index=date_index)
            pricing_column_dict[(symbol_str, "Dividend")] = pd.Series(0.0, index=date_index)
        spy_close_ser = pd.Series(np.linspace(108.0, 101.0, len(date_index)), index=date_index)
        for field_str in ("Open", "High", "Low", "Close"):
            pricing_column_dict[("SPY", field_str)] = spy_close_ser
            pricing_column_dict[("$SPX", field_str)] = spy_close_ser
        pricing_data_df = pd.DataFrame(pricing_column_dict, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        strategy_obj = self.make_strategy_obj("P2_SOFT200")

        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=date_index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=False,
        )

        transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
        expected_share_float = 100_000.0 * 0.25 / float(aaa_close_vec[signal_day_int])
        self.assertEqual(len(transaction_df), 1)
        self.assertEqual(pd.Timestamp(transaction_df.iloc[0]["bar"]), date_index[fill_day_int])
        self.assertAlmostEqual(float(transaction_df.iloc[0]["price"]), 110.0)
        self.assertAlmostEqual(float(transaction_df.iloc[0]["amount"]), expected_share_float)
        self.assertGreaterEqual(float(strategy_obj.results["cash"].min()), -0.01)


if __name__ == "__main__":
    unittest.main()
