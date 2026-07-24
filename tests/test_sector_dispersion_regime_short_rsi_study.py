import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
import talib

from alpha.engine.backtest import run_daily
from scripts.research.run_sector_dispersion_regime_short_rsi_study import (
    STUDY_UNIVERSE_ID_TUPLE,
    VARIANT_SPEC_TUPLE,
    SectorDispersionRegimeShortRsiConfig,
    SectorDispersionRegimeShortRsiStrategy,
    compute_hac_mean_test_dict,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import DEFAULT_CONFIG


class SectorDispersionRegimeShortRsiStudyTests(unittest.TestCase):
    def make_config_obj(
        self,
        variant_id_str: str,
        spy_sma_lookback_day_int: int = 3,
    ) -> SectorDispersionRegimeShortRsiConfig:
        base_config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB"),
            benchmark_symbol_str="$SPX",
            history_start_date_str="2023-12-01",
            backtest_start_date_str="2024-01-02",
            range_vol_lookback_day_int=3,
            capital_base_float=100_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )
        variant_spec_obj = next(
            spec_obj
            for spec_obj in VARIANT_SPEC_TUPLE
            if spec_obj.variant_id_str == variant_id_str
        )
        return SectorDispersionRegimeShortRsiConfig(
            base_config_obj=base_config_obj,
            variant_spec_obj=variant_spec_obj,
            spy_sma_override_day_int=spy_sma_lookback_day_int,
        )

    def make_strategy_obj(
        self,
        variant_id_str: str,
    ) -> SectorDispersionRegimeShortRsiStrategy:
        config_obj = self.make_config_obj(variant_id_str=variant_id_str)
        return SectorDispersionRegimeShortRsiStrategy(
            name=f"test_{variant_id_str.lower()}",
            benchmarks=[],
            study_config_obj=config_obj,
        )

    def test_variant_and_universe_manifests_are_frozen(self):
        self.assertEqual(
            tuple(spec_obj.variant_id_str for spec_obj in VARIANT_SPEC_TUPLE),
            ("B0", "L200", "S0", "S200", "S100", "L200_RSI", "S200_RSI"),
        )
        self.assertEqual(STUDY_UNIVERSE_ID_TUPLE, ("spdr_9", "vanguard_11", "spdr_11"))
        control_id_dict = {
            spec_obj.variant_id_str: spec_obj.control_variant_id_str
            for spec_obj in VARIANT_SPEC_TUPLE
        }
        self.assertEqual(control_id_dict["L200"], "B0")
        self.assertEqual(control_id_dict["S0"], "CASH")
        self.assertEqual(control_id_dict["S200_RSI"], "S200")

    def test_signal_features_use_completed_spy_close_and_talib_rsi2(self):
        date_index = pd.bdate_range("2024-01-02", periods=10)
        aaa_close_ser = pd.Series(
            [100.0, 99.0, 101.0, 98.0, 102.0, 97.0, 103.0, 96.0, 104.0, 95.0],
            index=date_index,
        )
        pricing_column_dict: dict[tuple[str, str], pd.Series] = {}
        for symbol_str, close_ser in {
            "AAA": aaa_close_ser,
            "BBB": pd.Series(np.linspace(50.0, 52.0, len(date_index)), index=date_index),
        }.items():
            pricing_column_dict[(symbol_str, "Open")] = close_ser
            pricing_column_dict[(symbol_str, "High")] = close_ser + 2.0
            pricing_column_dict[(symbol_str, "Low")] = close_ser - 2.0
            pricing_column_dict[(symbol_str, "Close")] = close_ser
            pricing_column_dict[(symbol_str, "Dividend")] = pd.Series(0.0, index=date_index)
        spy_close_ser = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            index=date_index,
        )
        pricing_column_dict[("SPY", "Close")] = spy_close_ser
        pricing_column_dict[("$SPX", "Close")] = spy_close_ser
        pricing_data_df = pd.DataFrame(pricing_column_dict, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

        strategy_obj = self.make_strategy_obj(variant_id_str="L200_RSI")
        signal_data_df = strategy_obj.compute_signals(pricing_data_df)

        expected_sma_float = float(spy_close_ser.iloc[-3:].mean())
        expected_rsi_vec = talib.RSI(aaa_close_ser.to_numpy(dtype=float), timeperiod=2)
        self.assertAlmostEqual(
            float(signal_data_df.loc[date_index[-1], ("SPY", "sma_3_ser")]),
            expected_sma_float,
        )
        self.assertTrue(
            bool(signal_data_df.loc[date_index[-1], ("SPY", "above_sma_regime_bool")])
        )
        self.assertAlmostEqual(
            float(signal_data_df.loc[date_index[-1], ("AAA", "rsi2_value_ser")]),
            float(expected_rsi_vec[-1]),
        )

    def test_long_sma_gate_filters_entries_and_rsi_is_and_exit(self):
        entry_strategy_obj = self.make_strategy_obj(variant_id_str="L200")
        blocked_entry_ser = pd.Series(
            {
                ("SPY", "above_sma_regime_bool"): False,
                ("AAA", "entry_signal_bool"): True,
                ("AAA", "Close"): 100.0,
            }
        )
        entry_strategy_obj.iterate(pd.DataFrame(), blocked_entry_ser, pd.Series(dtype=float))
        self.assertEqual(len(entry_strategy_obj.get_orders()), 0)

        allowed_entry_ser = blocked_entry_ser.copy()
        allowed_entry_ser[("SPY", "above_sma_regime_bool")] = True
        entry_strategy_obj.iterate(pd.DataFrame(), allowed_entry_ser, pd.Series(dtype=float))
        self.assertEqual(len(entry_strategy_obj.get_orders()), 1)
        self.assertGreater(float(entry_strategy_obj.get_orders()[0].amount), 0.0)

        held_strategy_obj = self.make_strategy_obj(variant_id_str="L200_RSI")
        held_strategy_obj._position_amount_map = {"AAA": 10.0}
        held_strategy_obj.current_trade_map["AAA"] = 1
        delayed_exit_ser = pd.Series(
            {
                ("SPY", "above_sma_regime_bool"): False,
                ("AAA", "exit_signal_bool"): True,
                ("AAA", "rsi2_value_ser"): 85.0,
            }
        )
        held_strategy_obj.iterate(pd.DataFrame(), delayed_exit_ser, pd.Series(dtype=float))
        self.assertEqual(len(held_strategy_obj.get_orders()), 0)

        confirmed_exit_ser = delayed_exit_ser.copy()
        confirmed_exit_ser[("AAA", "rsi2_value_ser")] = 95.0
        held_strategy_obj.iterate(pd.DataFrame(), confirmed_exit_ser, pd.Series(dtype=float))
        self.assertEqual(len(held_strategy_obj.get_orders()), 1)
        self.assertEqual(float(held_strategy_obj.get_orders()[0].amount), 0.0)

    def test_short_sma_gate_filters_entries_and_rsi_is_and_cover(self):
        entry_strategy_obj = self.make_strategy_obj(variant_id_str="S200")
        blocked_entry_ser = pd.Series(
            {
                ("SPY", "below_sma_regime_bool"): False,
                ("AAA", "exit_signal_bool"): True,
                ("AAA", "Close"): 100.0,
            }
        )
        entry_strategy_obj.iterate(pd.DataFrame(), blocked_entry_ser, pd.Series(dtype=float))
        self.assertEqual(len(entry_strategy_obj.get_orders()), 0)

        allowed_entry_ser = blocked_entry_ser.copy()
        allowed_entry_ser[("SPY", "below_sma_regime_bool")] = True
        entry_strategy_obj.iterate(pd.DataFrame(), allowed_entry_ser, pd.Series(dtype=float))
        self.assertEqual(len(entry_strategy_obj.get_orders()), 1)
        self.assertLess(float(entry_strategy_obj.get_orders()[0].amount), 0.0)

        held_strategy_obj = self.make_strategy_obj(variant_id_str="S200_RSI")
        held_strategy_obj._position_amount_map = {"AAA": -10.0}
        held_strategy_obj.current_trade_map["AAA"] = 1
        delayed_cover_ser = pd.Series(
            {
                ("SPY", "below_sma_regime_bool"): False,
                ("AAA", "entry_signal_bool"): True,
                ("AAA", "rsi2_value_ser"): 15.0,
            }
        )
        held_strategy_obj.iterate(pd.DataFrame(), delayed_cover_ser, pd.Series(dtype=float))
        self.assertEqual(len(held_strategy_obj.get_orders()), 0)

        confirmed_cover_ser = delayed_cover_ser.copy()
        confirmed_cover_ser[("AAA", "rsi2_value_ser")] = 5.0
        held_strategy_obj.iterate(pd.DataFrame(), confirmed_cover_ser, pd.Series(dtype=float))
        self.assertEqual(len(held_strategy_obj.get_orders()), 1)
        self.assertEqual(float(held_strategy_obj.get_orders()[0].amount), 0.0)

    def test_short_dividend_and_calendar_day_borrow_are_cash_debits(self):
        date_index = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")])
        strategy_obj = self.make_strategy_obj(variant_id_str="S0")
        strategy_obj.cash = 100_000.0
        strategy_obj.current_bar = date_index[1]
        strategy_obj._position_amount_map = {"AAA": -10.0}
        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): [100.0, 99.0],
                ("AAA", "High"): [101.0, 100.0],
                ("AAA", "Low"): [99.0, 98.0],
                ("AAA", "Close"): [100.0, 99.0],
                ("AAA", "Dividend"): [0.0, 1.0],
                ("BBB", "Open"): [50.0, 50.0],
                ("BBB", "High"): [51.0, 51.0],
                ("BBB", "Low"): [49.0, 49.0],
                ("BBB", "Close"): [50.0, 50.0],
                ("BBB", "Dividend"): [0.0, 0.0],
            },
            index=date_index,
        )

        strategy_obj.process_orders(pricing_data_df)

        expected_borrow_fee_float = 10.0 * 100.0 * 0.01 * 3.0 / 365.0
        self.assertAlmostEqual(
            strategy_obj.cash,
            100_000.0 - 10.0 - expected_borrow_fee_float,
        )
        self.assertAlmostEqual(strategy_obj.dividend_cash_total_float, -10.0)
        self.assertAlmostEqual(strategy_obj.borrow_fee_total_float, expected_borrow_fee_float)
        self.assertEqual(len(strategy_obj.borrow_fee_df), 1)

    def test_mirrored_short_signal_fills_at_next_open(self):
        date_index = pd.bdate_range("2024-01-02", periods=8)
        signal_day_int = 4
        fill_day_int = 5
        log_range_vec = np.array([0.010, 0.011, 0.012, 0.013, 0.120, 0.010, 0.011, 0.012])
        low_vec = np.full(len(date_index), 100.0)
        high_vec = low_vec * np.exp(log_range_vec)
        aaa_ibs_vec = np.full(len(date_index), 0.50)
        aaa_ibs_vec[signal_day_int] = 0.95
        aaa_close_vec = low_vec + aaa_ibs_vec * (high_vec - low_vec)
        aaa_open_vec = aaa_close_vec.copy()
        aaa_open_vec[fill_day_int] = 111.0
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
        spy_close_ser = pd.Series(np.linspace(100.0, 107.0, len(date_index)), index=date_index)
        for field_str in ("Open", "High", "Low", "Close"):
            pricing_column_dict[("SPY", field_str)] = spy_close_ser
            pricing_column_dict[("$SPX", field_str)] = spy_close_ser
        pricing_data_df = pd.DataFrame(pricing_column_dict, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        strategy_obj = self.make_strategy_obj(variant_id_str="S0")

        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=date_index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=False,
        )

        transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
        expected_share_float = -100_000.0 * 0.5 / float(aaa_close_vec[signal_day_int])
        self.assertEqual(len(transaction_df), 1)
        self.assertEqual(pd.Timestamp(transaction_df.iloc[0]["bar"]), date_index[fill_day_int])
        self.assertAlmostEqual(float(transaction_df.iloc[0]["price"]), 111.0)
        self.assertAlmostEqual(float(transaction_df.iloc[0]["amount"]), expected_share_float)

    def test_hac_mean_test_applies_full_eighteen_test_bonferroni_count(self):
        difference_return_ser = pd.Series(
            [0.0010, 0.0020, -0.0005, 0.0015, 0.0002, 0.0011, -0.0002, 0.0014],
            index=pd.bdate_range("2024-01-02", periods=8),
        )

        test_metric_dict = compute_hac_mean_test_dict(
            difference_return_ser=difference_return_ser,
            family_test_count_int=18,
        )

        self.assertGreater(float(test_metric_dict["hac_mean_t_stat_float"]), 0.0)
        self.assertAlmostEqual(
            float(test_metric_dict["bonferroni_p_value_float"]),
            min(1.0, float(test_metric_dict["raw_p_value_float"]) * 18.0),
        )


if __name__ == "__main__":
    unittest.main()
