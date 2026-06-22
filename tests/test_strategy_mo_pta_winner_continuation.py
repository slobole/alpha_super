import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from strategies.momentum.strategy_mo_pta_winner_continuation import (
    DEFAULT_CONFIG,
    RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG,
    RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_STOCK_SMA100_IVOL63_CONFIG,
    RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_TOP10_CONFIG,
    RUSSELL3000_LONG_SHORT_10X10_CONFIG,
    PtaWinnerContinuationStrategy,
    assign_quantile_bucket_ser,
    compute_index_above_sma_filter_ser,
    compute_stock_above_sma_filter_df,
    compute_pta_winner_continuation_signal_tables,
)


class PtaWinnerContinuationTests(unittest.TestCase):
    def make_test_config(self):
        return replace(
            DEFAULT_CONFIG,
            variant_key_str="test",
            indexname_str="S&P 500",
            benchmark_symbol_str="$SPX",
            pta_lookback_trading_day_int=3,
        )

    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-04-01",
        decision_date_str: str = "2024-03-29",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> PtaWinnerContinuationStrategy:
        config_obj = kwargs.pop("config", self.make_test_config())
        base_kwargs = dict(
            name="PtaWinnerContinuationTest",
            benchmarks=list(
                dict.fromkeys(
                    [
                        symbol_str
                        for symbol_str in [config_obj.benchmark_symbol_str, config_obj.regime_symbol_str]
                        if symbol_str is not None
                    ]
                )
            ),
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            config=config_obj,
        )
        base_kwargs.update(kwargs)
        return PtaWinnerContinuationStrategy(**base_kwargs)

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_default_config_is_russell3000_long_short_10x10(self):
        self.assertEqual(RUSSELL3000_LONG_SHORT_10X10_CONFIG.indexname_str, "Russell 3000")
        self.assertEqual(RUSSELL3000_LONG_SHORT_10X10_CONFIG.variant_key_str, "russell3000_long_short_10x10")
        self.assertEqual(RUSSELL3000_LONG_SHORT_10X10_CONFIG.bucket_count_int, 10)
        self.assertEqual(RUSSELL3000_LONG_SHORT_10X10_CONFIG.winner_return_bucket_int, 10)
        self.assertEqual(RUSSELL3000_LONG_SHORT_10X10_CONFIG.long_pta_bucket_int, 10)
        self.assertEqual(RUSSELL3000_LONG_SHORT_10X10_CONFIG.short_pta_bucket_int, 1)
        self.assertAlmostEqual(RUSSELL3000_LONG_SHORT_10X10_CONFIG.long_gross_exposure_float, 1.0)
        self.assertAlmostEqual(RUSSELL3000_LONG_SHORT_10X10_CONFIG.short_gross_exposure_float, 1.0)

    def test_long_only_sma200_config_uses_rua_regime_and_no_short_gross(self):
        self.assertEqual(RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG.indexname_str, "Russell 3000")
        self.assertEqual(
            RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG.variant_key_str,
            "russell3000_long_only_10x10_rua_sma200",
        )
        self.assertEqual(RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG.benchmark_symbol_str, "$RUA")
        self.assertEqual(RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG.regime_symbol_str, "$RUA")
        self.assertTrue(RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG.index_sma_filter_enabled_bool)
        self.assertEqual(RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG.index_sma_window_int, 200)
        self.assertAlmostEqual(RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG.long_gross_exposure_float, 1.0)
        self.assertAlmostEqual(RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG.short_gross_exposure_float, 0.0)

    def test_long_only_sma100_ivol63_config_enables_stock_filter_and_inverse_vol(self):
        config_obj = RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_STOCK_SMA100_IVOL63_CONFIG

        self.assertEqual(config_obj.indexname_str, "Russell 3000")
        self.assertEqual(
            config_obj.variant_key_str,
            "russell3000_long_only_10x10_rua_sma200_stock_sma100_ivol63",
        )
        self.assertTrue(config_obj.index_sma_filter_enabled_bool)
        self.assertTrue(config_obj.stock_sma_filter_enabled_bool)
        self.assertEqual(config_obj.stock_sma_window_int, 100)
        self.assertTrue(config_obj.inverse_vol_weighting_enabled_bool)
        self.assertEqual(config_obj.inverse_vol_window_int, 63)
        self.assertAlmostEqual(config_obj.short_gross_exposure_float, 0.0)

    def test_long_only_sma200_top10_config_caps_long_positions(self):
        config_obj = RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_TOP10_CONFIG

        self.assertEqual(config_obj.indexname_str, "Russell 3000")
        self.assertEqual(
            config_obj.variant_key_str,
            "russell3000_long_only_10x10_rua_sma200_top10",
        )
        self.assertTrue(config_obj.index_sma_filter_enabled_bool)
        self.assertEqual(config_obj.max_long_positions_int, 10)
        self.assertAlmostEqual(config_obj.short_gross_exposure_float, 0.0)

    def test_assign_quantile_bucket_places_highest_values_in_top_bucket(self):
        value_ser = pd.Series(np.arange(1, 101, dtype=float), index=[f"S{i:03d}" for i in range(1, 101)])

        bucket_ser = assign_quantile_bucket_ser(value_ser=value_ser, bucket_count_int=10)

        self.assertEqual(bucket_ser.loc["S001"], 1)
        self.assertEqual(bucket_ser.loc["S010"], 1)
        self.assertEqual(bucket_ser.loc["S091"], 10)
        self.assertEqual(bucket_ser.loc["S100"], 10)

    def test_index_sma_filter_uses_trailing_window(self):
        index_close_ser = pd.Series(
            [100.0, 101.0, 102.0, 103.0],
            index=pd.bdate_range("2024-01-01", periods=4),
        )

        pass_ser = compute_index_above_sma_filter_ser(
            index_close_ser=index_close_ser,
            window_int=3,
        )

        self.assertFalse(bool(pass_ser.iloc[0]))
        self.assertFalse(bool(pass_ser.iloc[1]))
        self.assertTrue(bool(pass_ser.iloc[2]))
        self.assertTrue(bool(pass_ser.iloc[3]))

    def test_stock_sma_filter_uses_trailing_window(self):
        date_index = pd.bdate_range("2024-01-01", periods=4)
        price_close_df = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 102.0, 103.0],
                "BBB": [103.0, 102.0, 101.0, 100.0],
            },
            index=date_index,
        )

        pass_df = compute_stock_above_sma_filter_df(
            price_close_df=price_close_df,
            window_int=3,
        )

        self.assertFalse(bool(pass_df.loc[date_index[1], "AAA"]))
        self.assertTrue(bool(pass_df.loc[date_index[2], "AAA"]))
        self.assertFalse(bool(pass_df.loc[date_index[2], "BBB"]))

    def test_signal_tables_use_prior_month_pta_not_same_month_pta(self):
        config_obj = self.make_test_config()
        date_index = pd.bdate_range("2024-01-02", "2024-03-29")
        price_close_df = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 30.0, len(date_index)),
                "BBB": np.linspace(20.0, 25.0, len(date_index)),
                "CCC": np.linspace(30.0, 20.0, len(date_index)),
                "DDD": np.linspace(40.0, 42.0, len(date_index)),
            },
            index=date_index,
        )

        monthly_decision_close_df, monthly_return_df, prior_pta_df = (
            compute_pta_winner_continuation_signal_tables(
                price_close_df=price_close_df,
                config=config_obj,
            )
        )

        decision_index = monthly_decision_close_df.index
        self.assertIn(pd.Timestamp("2024-02-29"), decision_index)
        self.assertIn(pd.Timestamp("2024-03-29"), decision_index)
        expected_feb_return_float = (
            price_close_df.loc[pd.Timestamp("2024-02-29"), "AAA"]
            / price_close_df.loc[pd.Timestamp("2024-01-31"), "AAA"]
        ) - 1.0
        self.assertAlmostEqual(
            float(monthly_return_df.loc[pd.Timestamp("2024-02-29"), "AAA"]),
            float(expected_feb_return_float),
        )

        # *** CRITICAL*** The prior PTA row for March must use the PTA that
        # could be known after February close. Changing March prices must not
        # change the PTA input used by the March winner decision.
        self.assertFalse(prior_pta_df.loc[pd.Timestamp("2024-02-29")].isna().all())
        self.assertFalse(prior_pta_df.loc[pd.Timestamp("2024-03-29")].isna().all())
        modified_price_close_df = price_close_df.copy()
        modified_price_close_df.loc[modified_price_close_df.index >= pd.Timestamp("2024-03-01"), "AAA"] *= 10.0
        _modified_monthly_decision_close_df, _modified_monthly_return_df, modified_prior_pta_df = (
            compute_pta_winner_continuation_signal_tables(
                price_close_df=modified_price_close_df,
                config=config_obj,
            )
        )
        np.testing.assert_allclose(
            prior_pta_df.loc[pd.Timestamp("2024-03-29")].to_numpy(dtype=float),
            modified_prior_pta_df.loc[pd.Timestamp("2024-03-29")].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_target_weights_select_winner_high_pta_long_and_winner_low_pta_short(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        symbol_list = [f"S{i:03d}" for i in range(1, 101)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )
        prior_pta_by_symbol_dict = {}
        for symbol_int in range(1, 91):
            prior_pta_by_symbol_dict[f"S{symbol_int:03d}"] = float(symbol_int + 5)
        for symbol_int in range(91, 96):
            prior_pta_by_symbol_dict[f"S{symbol_int:03d}"] = float(symbol_int - 90)
        for symbol_int in range(96, 101):
            prior_pta_by_symbol_dict[f"S{symbol_int:03d}"] = float(symbol_int)

        close_row_map = {}
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            close_row_map[(symbol_str, strategy_obj.monthly_return_field_str)] = float(symbol_int)
            close_row_map[(symbol_str, strategy_obj.prior_pta_field_str)] = prior_pta_by_symbol_dict[symbol_str]

        target_weight_ser = strategy_obj.get_target_weight_ser(
            close_row_ser=self.make_close_row_ser(close_row_map)
        )

        expected_long_symbol_list = [f"S{i:03d}" for i in range(96, 101)]
        expected_short_symbol_list = [f"S{i:03d}" for i in range(91, 96)]
        self.assertEqual(target_weight_ser.loc[expected_long_symbol_list].tolist(), [0.2] * 5)
        self.assertEqual(target_weight_ser.loc[expected_short_symbol_list].tolist(), [-0.2] * 5)
        self.assertEqual(set(target_weight_ser.index), set(expected_long_symbol_list + expected_short_symbol_list))

    def test_sma_filter_blocks_targets_when_index_below_sma(self):
        config_obj = replace(
            self.make_test_config(),
            regime_symbol_str="$RUA",
            index_sma_filter_enabled_bool=True,
            short_gross_exposure_float=0.0,
        )
        strategy_obj = self.make_strategy(config=config_obj)
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        strategy_obj.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy_obj.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.monthly_return_field_str): 1.0,
                ("AAA", strategy_obj.prior_pta_field_str): 1.0,
                ("$RUA", strategy_obj.index_sma_pass_field_str): False,
            }
        )

        target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(len(target_weight_ser), 0)

    def test_long_only_sma_filter_keeps_only_long_high_pta_winners_when_regime_passes(self):
        config_obj = replace(
            self.make_test_config(),
            regime_symbol_str="$RUA",
            index_sma_filter_enabled_bool=True,
            short_gross_exposure_float=0.0,
        )
        strategy_obj = self.make_strategy(config=config_obj)
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        symbol_list = [f"S{i:03d}" for i in range(1, 101)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )
        close_row_map = {
            ("$RUA", strategy_obj.index_sma_pass_field_str): True,
        }
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            close_row_map[(symbol_str, strategy_obj.monthly_return_field_str)] = float(symbol_int)
            close_row_map[(symbol_str, strategy_obj.prior_pta_field_str)] = float(symbol_int)

        target_weight_ser = strategy_obj.get_target_weight_ser(
            close_row_ser=self.make_close_row_ser(close_row_map)
        )

        expected_long_symbol_list = [f"S{i:03d}" for i in range(91, 101)]
        self.assertEqual(target_weight_ser.index.tolist(), expected_long_symbol_list)
        self.assertEqual(target_weight_ser.tolist(), [0.1] * 10)

    def test_max_long_positions_keeps_highest_prior_pta_candidates(self):
        config_obj = replace(
            self.make_test_config(),
            regime_symbol_str="$RUA",
            index_sma_filter_enabled_bool=True,
            bucket_count_int=5,
            winner_return_bucket_int=5,
            long_pta_bucket_int=5,
            short_pta_bucket_int=1,
            short_gross_exposure_float=0.0,
            max_long_positions_int=10,
        )
        strategy_obj = self.make_strategy(config=config_obj)
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        symbol_list = [f"S{i:03d}" for i in range(1, 101)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )
        close_row_map = {
            ("$RUA", strategy_obj.index_sma_pass_field_str): True,
        }
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            close_row_map[(symbol_str, strategy_obj.monthly_return_field_str)] = float(symbol_int)
            close_row_map[(symbol_str, strategy_obj.prior_pta_field_str)] = float(symbol_int)

        target_weight_ser = strategy_obj.get_target_weight_ser(
            close_row_ser=self.make_close_row_ser(close_row_map)
        )

        expected_long_symbol_list = [f"S{i:03d}" for i in range(91, 101)]
        self.assertEqual(target_weight_ser.index.tolist(), expected_long_symbol_list)
        self.assertEqual(target_weight_ser.tolist(), [0.1] * 10)

    def test_inverse_vol63_weights_selected_longs(self):
        config_obj = replace(
            self.make_test_config(),
            regime_symbol_str="$RUA",
            index_sma_filter_enabled_bool=True,
            stock_sma_filter_enabled_bool=True,
            short_gross_exposure_float=0.0,
            inverse_vol_weighting_enabled_bool=True,
            inverse_vol_window_int=63,
        )
        strategy_obj = self.make_strategy(config=config_obj)
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        symbol_list = [f"S{i:03d}" for i in range(1, 101)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )
        close_row_map = {
            ("$RUA", strategy_obj.index_sma_pass_field_str): True,
        }
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            close_row_map[(symbol_str, strategy_obj.monthly_return_field_str)] = float(symbol_int)
            close_row_map[(symbol_str, strategy_obj.prior_pta_field_str)] = float(symbol_int)
            close_row_map[(symbol_str, strategy_obj.stock_sma_pass_field_str)] = True
            close_row_map[(symbol_str, strategy_obj.inverse_vol_field_str)] = float(symbol_int)

        target_weight_ser = strategy_obj.get_target_weight_ser(
            close_row_ser=self.make_close_row_ser(close_row_map)
        )

        expected_long_symbol_list = [f"S{i:03d}" for i in range(91, 101)]
        expected_inverse_vol_ser = pd.Series(
            {symbol_str: 1.0 / float(int(symbol_str[1:])) for symbol_str in expected_long_symbol_list},
            dtype=float,
        )
        expected_weight_ser = expected_inverse_vol_ser / float(expected_inverse_vol_ser.sum())
        pd.testing.assert_series_equal(target_weight_ser, expected_weight_ser)


if __name__ == "__main__":
    unittest.main()
