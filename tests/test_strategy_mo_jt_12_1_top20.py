import os
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.momentum.strategy_mo_jt_12_1_top20 import (
    DEFAULT_CONFIG,
    NASDAQ100_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG,
    NASDAQ100_SMA100_CONFIG,
    NASDAQ100_SMA100_VOL_TARGET_CONFIG,
    NASDAQ100_VOL_TARGET_CONFIG,
    RUSSELL1000_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG,
    SMA100_INDEX_SMA200_PN_RANKING_CONFIG_BY_VARIANT_KEY_DICT,
    SMA100_INDEX_SMA200_RANKING_SWEEP_CONFIG_BY_VARIANT_KEY_DICT,
    RUSSELL1000_SMA100_CONFIG,
    RUSSELL1000_SMA100_VOL_TARGET_CONFIG,
    RUSSELL1000_VOL_TARGET_CONFIG,
    SP500_CONFIG,
    SP500_SMA100_INDEX_SMA200_CONFIG,
    SP500_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG,
    SP500_SMA100_CONFIG,
    SP500_SMA100_VOL_TARGET_CONFIG,
    SP500_VOL_TARGET_CONFIG,
    MULTI_HORIZON_Z_RANKING_METHOD_STR,
    PN_EV_MULTI_WINDOW_Z_RANKING_METHOD_STR,
    PN_LRB_MULTI_WINDOW_Z_RANKING_METHOD_STR,
    RAW_12_1_RANKING_METHOD_STR,
    RESIDUAL_12_1_RANKING_METHOD_STR,
    TREND_QUALITY_RANKING_METHOD_STR,
    VOL_NORMALIZED_12_1_RANKING_METHOD_STR,
    Jt121Top20Strategy,
    compute_multi_horizon_z_score_df,
    compute_index_above_sma_filter_ser,
    compute_jt_12_1_momentum_score_df,
    compute_pn_indicator_score_df,
    compute_residual_12_1_score_df,
    compute_trend_quality_score_df,
    compute_vol_normalized_12_1_score_df,
    compute_stock_above_sma_filter_df,
)


class Jt121Top20Tests(unittest.TestCase):
    def make_test_config(self):
        return replace(
            DEFAULT_CONFIG,
            variant_key_str="test",
            benchmark_symbol_str="SPY",
            lookback_trading_day_int=5,
            skip_trading_day_int=2,
            max_positions_int=2,
        )

    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-01-10",
        decision_date_str: str = "2024-01-09",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> Jt121Top20Strategy:
        config_obj = kwargs.pop("config_obj", self.make_test_config())
        base_kwargs = dict(
            name="Jt121Top20Test",
            benchmarks=[config_obj.benchmark_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            config_obj=config_obj,
        )
        base_kwargs.update(kwargs)
        return Jt121Top20Strategy(**base_kwargs)

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_default_configs_use_requested_universes(self):
        self.assertEqual(SP500_CONFIG.indexname_str, "S&P 500")
        self.assertEqual(SP500_CONFIG.benchmark_symbol_str, "$SPX")
        self.assertEqual(NASDAQ100_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(NASDAQ100_CONFIG.benchmark_symbol_str, "$NDX")
        self.assertEqual(RUSSELL1000_CONFIG.indexname_str, "Russell 1000")
        self.assertEqual(RUSSELL1000_CONFIG.benchmark_symbol_str, "$RUI")
        self.assertTrue(SP500_VOL_TARGET_CONFIG.volatility_target_enabled_bool)
        self.assertTrue(NASDAQ100_VOL_TARGET_CONFIG.volatility_target_enabled_bool)
        self.assertTrue(RUSSELL1000_VOL_TARGET_CONFIG.volatility_target_enabled_bool)
        self.assertTrue(SP500_SMA100_CONFIG.stock_sma_filter_enabled_bool)
        self.assertTrue(NASDAQ100_SMA100_CONFIG.stock_sma_filter_enabled_bool)
        self.assertTrue(RUSSELL1000_SMA100_CONFIG.stock_sma_filter_enabled_bool)
        self.assertTrue(SP500_SMA100_VOL_TARGET_CONFIG.volatility_target_enabled_bool)
        self.assertTrue(NASDAQ100_SMA100_VOL_TARGET_CONFIG.volatility_target_enabled_bool)
        self.assertTrue(RUSSELL1000_SMA100_VOL_TARGET_CONFIG.volatility_target_enabled_bool)
        self.assertTrue(SP500_SMA100_VOL_TARGET_CONFIG.stock_sma_filter_enabled_bool)
        self.assertTrue(NASDAQ100_SMA100_VOL_TARGET_CONFIG.stock_sma_filter_enabled_bool)
        self.assertTrue(RUSSELL1000_SMA100_VOL_TARGET_CONFIG.stock_sma_filter_enabled_bool)
        self.assertTrue(SP500_SMA100_INDEX_SMA200_CONFIG.stock_sma_filter_enabled_bool)
        self.assertTrue(NASDAQ100_SMA100_INDEX_SMA200_CONFIG.stock_sma_filter_enabled_bool)
        self.assertTrue(RUSSELL1000_SMA100_INDEX_SMA200_CONFIG.stock_sma_filter_enabled_bool)
        self.assertTrue(SP500_SMA100_INDEX_SMA200_CONFIG.index_sma_filter_enabled_bool)
        self.assertTrue(NASDAQ100_SMA100_INDEX_SMA200_CONFIG.index_sma_filter_enabled_bool)
        self.assertTrue(RUSSELL1000_SMA100_INDEX_SMA200_CONFIG.index_sma_filter_enabled_bool)
        self.assertTrue(SP500_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG.volatility_target_enabled_bool)
        self.assertTrue(NASDAQ100_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG.volatility_target_enabled_bool)
        self.assertTrue(RUSSELL1000_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG.volatility_target_enabled_bool)
        self.assertTrue(SP500_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG.index_sma_filter_enabled_bool)
        self.assertTrue(NASDAQ100_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG.index_sma_filter_enabled_bool)
        self.assertTrue(RUSSELL1000_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG.index_sma_filter_enabled_bool)

    def test_ranking_sweep_configs_cover_four_requested_methods_plus_baseline(self):
        ranking_method_set = {
            config_obj.ranking_method_str
            for config_obj in SMA100_INDEX_SMA200_RANKING_SWEEP_CONFIG_BY_VARIANT_KEY_DICT.values()
        }

        self.assertEqual(
            ranking_method_set,
            {
                RAW_12_1_RANKING_METHOD_STR,
                VOL_NORMALIZED_12_1_RANKING_METHOD_STR,
                TREND_QUALITY_RANKING_METHOD_STR,
                MULTI_HORIZON_Z_RANKING_METHOD_STR,
                RESIDUAL_12_1_RANKING_METHOD_STR,
            },
        )
        self.assertEqual(len(SMA100_INDEX_SMA200_RANKING_SWEEP_CONFIG_BY_VARIANT_KEY_DICT), 15)
        self.assertTrue(
            all(
                config_obj.stock_sma_filter_enabled_bool and config_obj.index_sma_filter_enabled_bool
                for config_obj in SMA100_INDEX_SMA200_RANKING_SWEEP_CONFIG_BY_VARIANT_KEY_DICT.values()
            )
        )

    def test_pn_ranking_configs_cover_ev_and_lrb_plus_baseline(self):
        ranking_method_set = {
            config_obj.ranking_method_str
            for config_obj in SMA100_INDEX_SMA200_PN_RANKING_CONFIG_BY_VARIANT_KEY_DICT.values()
        }

        self.assertEqual(
            ranking_method_set,
            {
                RAW_12_1_RANKING_METHOD_STR,
                PN_EV_MULTI_WINDOW_Z_RANKING_METHOD_STR,
                PN_LRB_MULTI_WINDOW_Z_RANKING_METHOD_STR,
            },
        )
        self.assertEqual(len(SMA100_INDEX_SMA200_PN_RANKING_CONFIG_BY_VARIANT_KEY_DICT), 9)
        self.assertTrue(
            all(
                config_obj.stock_sma_filter_enabled_bool and config_obj.index_sma_filter_enabled_bool
                for config_obj in SMA100_INDEX_SMA200_PN_RANKING_CONFIG_BY_VARIANT_KEY_DICT.values()
            )
        )

    def test_compute_signal_matches_12_1_formula(self):
        config_obj = self.make_test_config()
        date_index = pd.bdate_range("2024-01-01", periods=23)
        price_close_df = pd.DataFrame(
            {
                "AAA": [100.0 + float(day_int) for day_int in range(23)],
                "BBB": [200.0 - float(day_int) for day_int in range(23)],
            },
            index=date_index,
        )

        momentum_score_df = compute_jt_12_1_momentum_score_df(
            price_close_df=price_close_df,
            config_obj=config_obj,
        )

        decision_date_ts = pd.Timestamp(date_index[-1])
        # *** CRITICAL*** Manual expected value uses t-2 over t-5, matching
        # the configured 12-1 skip/lookback analogue in this small test.
        expected_aaa_float = (
            price_close_df.loc[date_index[-3], "AAA"] / price_close_df.loc[date_index[-6], "AAA"]
        ) - 1.0
        self.assertAlmostEqual(
            float(momentum_score_df.loc[decision_date_ts, "AAA"]),
            expected_aaa_float,
        )

    def test_compute_stock_sma_filter_uses_trailing_window(self):
        config_obj = replace(
            self.make_test_config(),
            stock_sma_filter_enabled_bool=True,
            stock_sma_window_int=3,
        )
        date_index = pd.bdate_range("2024-01-01", periods=5)
        price_close_df = pd.DataFrame(
            {
                "AAA": [10.0, 11.0, 12.0, 13.0, 14.0],
                "BBB": [10.0, 9.0, 8.0, 7.0, 6.0],
            },
            index=date_index,
        )

        stock_sma_pass_df = compute_stock_above_sma_filter_df(
            price_close_df=price_close_df,
            config_obj=config_obj,
        )

        # *** CRITICAL*** The expected values compare each close to the
        # trailing 3-close mean ending on the same decision close.
        self.assertFalse(bool(stock_sma_pass_df.loc[date_index[1], "AAA"]))
        self.assertTrue(bool(stock_sma_pass_df.loc[date_index[-1], "AAA"]))
        self.assertFalse(bool(stock_sma_pass_df.loc[date_index[-1], "BBB"]))

    def test_compute_index_sma_filter_uses_trailing_window(self):
        config_obj = replace(
            self.make_test_config(),
            index_sma_filter_enabled_bool=True,
            index_sma_window_int=3,
        )
        date_index = pd.bdate_range("2024-01-01", periods=5)
        index_close_ser = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=date_index)

        index_sma_pass_ser = compute_index_above_sma_filter_ser(
            index_close_ser=index_close_ser,
            config_obj=config_obj,
        )

        # *** CRITICAL*** The expected values compare index close to the
        # trailing 3-close mean ending on the same decision close.
        self.assertFalse(bool(index_sma_pass_ser.loc[date_index[1]]))
        self.assertTrue(bool(index_sma_pass_ser.loc[date_index[-1]]))

    def test_compute_vol_normalized_score_uses_skipped_trailing_volatility(self):
        config_obj = replace(
            self.make_test_config(),
            ranking_method_str=VOL_NORMALIZED_12_1_RANKING_METHOD_STR,
        )
        date_index = pd.bdate_range("2024-01-01", periods=23)
        price_close_df = pd.DataFrame(
            {
                "AAA": [100.0 + float(day_int) for day_int in range(23)],
                "BBB": [100.0, 110.0, 95.0, 115.0, 100.0] + [100.0 + float(day_int) for day_int in range(18)],
            },
            index=date_index,
        )

        score_df = compute_vol_normalized_12_1_score_df(
            price_close_df=price_close_df,
            config_obj=config_obj,
        )

        decision_date_ts = pd.Timestamp(score_df.index[-1])
        raw_momentum_df = compute_jt_12_1_momentum_score_df(
            price_close_df=price_close_df,
            config_obj=config_obj,
        )
        skipped_return_df = price_close_df.pct_change(fill_method=None).shift(config_obj.skip_trading_day_int)
        trailing_vol_df = skipped_return_df.rolling(window=3, min_periods=3).std(ddof=1) * (252.0 ** 0.5)
        expected_score_float = (
            raw_momentum_df.loc[decision_date_ts, "AAA"] / trailing_vol_df.loc[decision_date_ts, "AAA"]
        )
        self.assertAlmostEqual(float(score_df.loc[decision_date_ts, "AAA"]), float(expected_score_float))

    def test_compute_multi_horizon_z_score_returns_monthly_cross_sectional_scores(self):
        config_obj = replace(
            DEFAULT_CONFIG,
            ranking_method_str=MULTI_HORIZON_Z_RANKING_METHOD_STR,
        )
        date_index = pd.bdate_range("2023-01-02", periods=280)
        price_close_df = pd.DataFrame(
            {
                "AAA": [100.0 + float(day_int) for day_int in range(280)],
                "BBB": [120.0 + 0.4 * float(day_int) for day_int in range(280)],
                "CCC": [90.0 + 0.1 * float(day_int) for day_int in range(280)],
            },
            index=date_index,
        )

        score_df = compute_multi_horizon_z_score_df(
            price_close_df=price_close_df,
            config_obj=config_obj,
        )

        self.assertGreater(len(score_df.dropna(how="all")), 0)
        self.assertTrue(score_df.dropna(how="all").iloc[-1].notna().all())

    def test_compute_residual_score_subtracts_beta_scaled_index_return(self):
        config_obj = replace(
            self.make_test_config(),
            ranking_method_str=RESIDUAL_12_1_RANKING_METHOD_STR,
        )
        date_index = pd.bdate_range("2024-01-01", periods=23)
        index_close_ser = pd.Series([100.0 + float(day_int) for day_int in range(23)], index=date_index)
        price_close_df = pd.DataFrame(
            {
                "AAA": [100.0 + 2.0 * float(day_int) for day_int in range(23)],
                "BBB": [100.0 + 0.5 * float(day_int) for day_int in range(23)],
            },
            index=date_index,
        )

        score_df = compute_residual_12_1_score_df(
            price_close_df=price_close_df,
            index_close_ser=index_close_ser,
            config_obj=config_obj,
        )

        self.assertTrue(score_df.iloc[-1].notna().all())

    def test_compute_trend_quality_score_prefers_cleaner_uptrend(self):
        config_obj = replace(
            DEFAULT_CONFIG,
            ranking_method_str=TREND_QUALITY_RANKING_METHOD_STR,
        )
        date_index = pd.bdate_range("2023-01-02", periods=160)
        price_close_df = pd.DataFrame(
            {
                "AAA": [100.0 * (1.002 ** day_int) for day_int in range(160)],
                "BBB": [100.0 * (1.002 ** day_int) * (1.05 if day_int % 2 == 0 else 0.95) for day_int in range(160)],
            },
            index=date_index,
        )

        score_df = compute_trend_quality_score_df(
            price_close_df=price_close_df,
            config_obj=config_obj,
        )

        decision_date_ts = pd.Timestamp(score_df.index[-1])
        self.assertGreater(float(score_df.loc[decision_date_ts, "AAA"]), float(score_df.loc[decision_date_ts, "BBB"]))

    def test_compute_pn_ev_score_prefers_more_positive_return_balance(self):
        date_index = pd.bdate_range("2023-01-02", periods=280)
        price_close_df = pd.DataFrame(
            {
                "AAA": [100.0 * (1.001 ** day_int) for day_int in range(280)],
                "BBB": [100.0 * (1.0002 ** day_int) * (1.02 if day_int % 2 == 0 else 0.98) for day_int in range(280)],
                "CCC": [100.0 * (0.999 ** day_int) for day_int in range(280)],
            },
            index=date_index,
        )

        score_df = compute_pn_indicator_score_df(price_close_df=price_close_df, score_name_str="ev")

        decision_date_ts = pd.Timestamp(score_df.index[-1])
        self.assertGreater(float(score_df.loc[decision_date_ts, "AAA"]), float(score_df.loc[decision_date_ts, "CCC"]))

    def test_compute_pn_lrb_score_prefers_higher_positive_negative_ratio(self):
        date_index = pd.bdate_range("2023-01-02", periods=280)
        price_close_df = pd.DataFrame(
            {
                "AAA": [100.0 * (1.001 ** day_int) * (1.003 if day_int % 5 != 0 else 0.997) for day_int in range(280)],
                "BBB": [100.0 * (1.0002 ** day_int) * (1.02 if day_int % 2 == 0 else 0.98) for day_int in range(280)],
                "CCC": [100.0 * (0.999 ** day_int) for day_int in range(280)],
            },
            index=date_index,
        )

        score_df = compute_pn_indicator_score_df(price_close_df=price_close_df, score_name_str="lrb")

        decision_date_ts = pd.Timestamp(score_df.index[-1])
        self.assertGreater(float(score_df.loc[decision_date_ts, "AAA"]), float(score_df.loc[decision_date_ts, "BBB"]))

    def test_get_selected_symbol_list_uses_top_n_pit_members_without_positive_floor(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-09")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "OUT": [0],
            },
            index=[strategy_obj.previous_bar],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.momentum_score_field_str): -0.10,
                ("BBB", strategy_obj.momentum_score_field_str): -0.05,
                ("CCC", strategy_obj.momentum_score_field_str): -0.20,
                ("OUT", strategy_obj.momentum_score_field_str): 9.0,
            }
        )

        selected_symbol_list = strategy_obj.get_selected_symbol_list(close_row_ser=close_row_ser)

        self.assertEqual(selected_symbol_list, ["BBB", "AAA"])

    def test_stock_sma_filter_blocks_high_momentum_name_below_average(self):
        config_obj = replace(
            self.make_test_config(),
            stock_sma_filter_enabled_bool=True,
            stock_sma_window_int=3,
        )
        strategy_obj = self.make_strategy(config_obj=config_obj)
        strategy_obj.previous_bar = pd.Timestamp("2024-01-09")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
            },
            index=[strategy_obj.previous_bar],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.momentum_score_field_str): 0.30,
                ("BBB", strategy_obj.momentum_score_field_str): 0.20,
                ("CCC", strategy_obj.momentum_score_field_str): 0.10,
                ("AAA", strategy_obj.stock_sma_pass_field_str): False,
                ("BBB", strategy_obj.stock_sma_pass_field_str): True,
                ("CCC", strategy_obj.stock_sma_pass_field_str): True,
            }
        )

        selected_symbol_list = strategy_obj.get_selected_symbol_list(close_row_ser=close_row_ser)

        self.assertEqual(selected_symbol_list, ["BBB", "CCC"])

    def test_index_sma_filter_blocks_new_buys_and_liquidates_existing_name(self):
        config_obj = replace(
            self.make_test_config(),
            index_sma_filter_enabled_bool=True,
            index_sma_window_int=3,
        )
        strategy_obj = self.make_strategy(config_obj=config_obj)
        strategy_obj.previous_bar = pd.Timestamp("2024-01-09")
        strategy_obj.current_bar = pd.Timestamp("2024-01-10")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
            },
            index=[strategy_obj.previous_bar],
        )
        strategy_obj.add_transaction(7, strategy_obj.previous_bar, "CCC", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.current_trade_map["CCC"] = 7
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.momentum_score_field_str): 0.30,
                ("BBB", strategy_obj.momentum_score_field_str): 0.20,
                ("CCC", strategy_obj.momentum_score_field_str): 0.10,
                (config_obj.benchmark_symbol_str, strategy_obj.index_sma_pass_field_str): False,
            }
        )

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["CCC"])
        self.assertEqual(order_list[0].unit, "shares")
        self.assertEqual(order_list[0].amount, 0)
        self.assertTrue(order_list[0].target)
        self.assertEqual(order_list[0].trade_id, 7)

    def test_iterate_rebalances_to_equal_weight_top_n_and_sells_dropped_name(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-09")
        strategy_obj.current_bar = pd.Timestamp("2024-01-10")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
            },
            index=[strategy_obj.previous_bar],
        )
        strategy_obj.add_transaction(7, strategy_obj.previous_bar, "CCC", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.current_trade_map["CCC"] = 7
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.momentum_score_field_str): 0.30,
                ("BBB", strategy_obj.momentum_score_field_str): 0.20,
                ("CCC", strategy_obj.momentum_score_field_str): 0.10,
            }
        )

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["CCC", "AAA", "BBB"])
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertEqual(order_list[0].unit, "shares")
        self.assertEqual(order_list[0].amount, 0)
        self.assertTrue(order_list[0].target)
        self.assertEqual(order_list[0].trade_id, 7)
        self.assertTrue(all(order_obj.unit == "percent" for order_obj in order_list[1:]))
        self.assertTrue(all(order_obj.target for order_obj in order_list[1:]))
        self.assertTrue(all(float(order_obj.amount) == 0.5 for order_obj in order_list[1:]))

    def test_volatility_target_scales_top_n_weights_down(self):
        config_obj = replace(
            self.make_test_config(),
            volatility_target_enabled_bool=True,
            target_annual_volatility_float=0.12,
            realized_vol_window_int=3,
            max_gross_exposure_float=1.0,
        )
        strategy_obj = self.make_strategy(config_obj=config_obj)
        strategy_obj.previous_bar = pd.Timestamp("2024-01-09")
        strategy_obj.current_bar = pd.Timestamp("2024-01-10")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
            },
            index=[strategy_obj.previous_bar],
        )
        strategy_obj._daily_return_history_list = [0.02, -0.02, 0.02]
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.momentum_score_field_str): 0.30,
                ("BBB", strategy_obj.momentum_score_field_str): 0.20,
            }
        )

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0}, dtype=float),
        )

        realized_volatility_float = pd.Series([0.02, -0.02, 0.02], dtype=float).std(ddof=1) * (252.0 ** 0.5)
        expected_gross_exposure_float = 0.12 / realized_volatility_float
        expected_weight_float = expected_gross_exposure_float / 2.0
        order_list = strategy_obj.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["AAA", "BBB"])
        self.assertTrue(all(order_obj.unit == "percent" for order_obj in order_list))
        self.assertTrue(all(order_obj.target for order_obj in order_list))
        self.assertTrue(all(abs(float(order_obj.amount) - expected_weight_float) < 1e-12 for order_obj in order_list))

    def test_volatility_target_caps_low_vol_scale_at_one_gross(self):
        config_obj = replace(
            self.make_test_config(),
            volatility_target_enabled_bool=True,
            target_annual_volatility_float=0.12,
            realized_vol_window_int=3,
            max_gross_exposure_float=1.0,
        )
        strategy_obj = self.make_strategy(config_obj=config_obj)
        strategy_obj._daily_return_history_list = [0.001, -0.001, 0.001]

        gross_exposure_float = strategy_obj.get_gross_exposure_target_float()

        self.assertEqual(gross_exposure_float, 1.0)

    def test_volatility_target_reenters_after_all_cash_window(self):
        config_obj = replace(
            self.make_test_config(),
            volatility_target_enabled_bool=True,
            target_annual_volatility_float=0.12,
            realized_vol_window_int=3,
            max_gross_exposure_float=1.0,
        )
        strategy_obj = self.make_strategy(config_obj=config_obj)
        strategy_obj._daily_return_history_list = [0.0, 0.0, 0.0]

        gross_exposure_float = strategy_obj.get_gross_exposure_target_float()

        self.assertEqual(gross_exposure_float, 1.0)


if __name__ == "__main__":
    unittest.main()
