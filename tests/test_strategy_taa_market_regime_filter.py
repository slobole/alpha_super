import unittest

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.taa_traditional.strategy_taa_market_regime_filter import (
    DEFAULT_TRADE_ID_INT,
    SIGNAL_NAMESPACE_STR,
    MarketRegimeFilterConfig,
    MarketRegimeFilterStrategy,
    build_execution_target_weight_df,
    compute_market_regime_signal_df,
)
from strategies.taa_traditional.strategy_taa_market_regime_filter_sso_green import (
    SSO_GREEN_CONFIG,
)
from strategies.taa_traditional.strategy_taa_market_regime_filter_score1_ief import (
    SCORE1_IEF_CONFIG,
)
from strategies.taa_traditional.strategy_taa_market_regime_filter_score2_spy_ief import (
    SCORE2_SPY_IEF_CONFIG,
)


class MarketRegimeFilterStrategyTests(unittest.TestCase):
    def make_config(self) -> MarketRegimeFilterConfig:
        return MarketRegimeFilterConfig(
            strategy_name_str="strategy_taa_market_regime_filter_test",
            trend_window_day_int=3,
            credit_z_window_day_int=3,
            credit_z_min_float=-1.0,
            history_start_date_str="2024-01-01",
            backtest_start_date_str="2024-01-04",
            capital_base_float=100_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )

    def make_strategy(self, config: MarketRegimeFilterConfig | None = None) -> MarketRegimeFilterStrategy:
        config_obj = self.make_config() if config is None else config
        return MarketRegimeFilterStrategy(
            name=config_obj.strategy_name_str,
            config=config_obj,
        )

    def make_pricing_data_df(self) -> pd.DataFrame:
        trading_index = pd.date_range("2024-01-01", periods=8, freq="B")
        close_map = {
            "SPY": [10.0, 10.0, 10.0, 12.0, 12.0, 9.0, 8.0, 13.0],
            "SSO": [20.0, 20.0, 20.0, 24.0, 24.0, 18.0, 16.0, 26.0],
            "$VIX": [20.0, 19.0, 18.0, 17.0, 17.0, 18.0, 30.0, 16.0],
            "$VIX3M": [21.0, 21.0, 21.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "HYG": [100.0, 101.0, 102.0, 98.0, 103.0, 90.0, 80.0, 105.0],
            "IEF": [100.0] * 8,
            "$SPXTR": [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0],
        }
        pricing_data_map: dict[tuple[str, str], list[float]] = {}
        for symbol_str, close_list in close_map.items():
            close_arr = np.array(close_list, dtype=float)
            pricing_data_map[(symbol_str, "Open")] = (close_arr * 0.999).tolist()
            pricing_data_map[(symbol_str, "High")] = (close_arr * 1.001).tolist()
            pricing_data_map[(symbol_str, "Low")] = (close_arr * 0.998).tolist()
            pricing_data_map[(symbol_str, "Close")] = close_arr.tolist()

        pricing_data_df = pd.DataFrame(pricing_data_map, index=trading_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_compute_market_regime_signal_df_matches_formula(self):
        config_obj = self.make_config()
        pricing_data_df = self.make_pricing_data_df()

        signal_df = compute_market_regime_signal_df(
            spy_close_ser=pricing_data_df[("SPY", "Close")],
            vix_close_ser=pricing_data_df[("$VIX", "Close")],
            vix3m_close_ser=pricing_data_df[("$VIX3M", "Close")],
            hyg_close_ser=pricing_data_df[("HYG", "Close")],
            ief_close_ser=pricing_data_df[("IEF", "Close")],
            config=config_obj,
        )

        green_ts = pd.Timestamp("2024-01-05")
        risk_ts = pd.Timestamp("2024-01-08")
        cash_ts = pd.Timestamp("2024-01-09")

        self.assertTrue(bool(signal_df.loc[green_ts, "trend_pass_bool"]))
        self.assertTrue(bool(signal_df.loc[green_ts, "volatility_pass_bool"]))
        self.assertTrue(bool(signal_df.loc[green_ts, "credit_pass_bool"]))
        self.assertEqual(int(signal_df.loc[green_ts, "detector_score_int"]), 3)
        self.assertAlmostEqual(float(signal_df.loc[green_ts, "target_weight_ser"]), 1.0)
        self.assertAlmostEqual(float(signal_df.loc[green_ts, "target_weight_spy_ser"]), 1.0)
        self.assertEqual(signal_df.loc[green_ts, "target_map_str"], "SPY:1")

        credit_ratio_window_ser = pd.Series([0.98, 1.03, 0.90], dtype=float)
        expected_credit_z_float = float(
            (0.90 - credit_ratio_window_ser.mean()) / credit_ratio_window_ser.std(ddof=0)
        )
        self.assertAlmostEqual(float(signal_df.loc[risk_ts, "credit_z_ser"]), expected_credit_z_float)
        self.assertFalse(bool(signal_df.loc[risk_ts, "trend_pass_bool"]))
        self.assertTrue(bool(signal_df.loc[risk_ts, "volatility_pass_bool"]))
        self.assertFalse(bool(signal_df.loc[risk_ts, "credit_pass_bool"]))
        self.assertEqual(int(signal_df.loc[risk_ts, "detector_score_int"]), 1)
        self.assertAlmostEqual(float(signal_df.loc[risk_ts, "target_weight_ser"]), 0.0)
        self.assertAlmostEqual(float(signal_df.loc[risk_ts, "target_weight_spy_ser"]), 0.0)
        self.assertEqual(signal_df.loc[risk_ts, "target_map_str"], "cash")

        self.assertEqual(int(signal_df.loc[cash_ts, "detector_score_int"]), 0)
        self.assertAlmostEqual(float(signal_df.loc[cash_ts, "target_weight_ser"]), 0.0)
        self.assertEqual(signal_df.loc[cash_ts, "target_map_str"], "cash")

    def test_build_execution_target_weight_df_shifts_close_signal_to_next_bar(self):
        trading_index = pd.date_range("2024-01-01", periods=4, freq="B")
        signal_data_df = pd.DataFrame(
            {
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): [np.nan, 1.0, 0.5, 0.0],
            },
            index=trading_index,
            dtype=float,
        )
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)

        daily_target_weight_df = build_execution_target_weight_df(
            signal_data_df=signal_data_df,
            trade_symbol_str="SPY",
            result_index=trading_index,
        )

        self.assertTrue(np.allclose(daily_target_weight_df["SPY"].to_numpy(), [0.0, 0.0, 1.0, 0.5]))
        self.assertTrue(np.allclose(daily_target_weight_df.sum(axis=1).to_numpy(), 1.0))

    def test_build_execution_target_weight_df_supports_sso_green_spy_caution(self):
        config_obj = MarketRegimeFilterConfig(
            **{
                **self.make_config().__dict__,
                "green_trade_symbol_str": "SSO",
                "caution_trade_symbol_str": "SPY",
            }
        )
        trading_index = pd.date_range("2024-01-01", periods=4, freq="B")
        signal_data_df = pd.DataFrame(
            {
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): [np.nan, 3.0, 2.0, 1.0],
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): [np.nan, 1.0, 0.5, 0.0],
            },
            index=trading_index,
            dtype=float,
        )
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)

        daily_target_weight_df = build_execution_target_weight_df(
            signal_data_df=signal_data_df,
            config=config_obj,
            result_index=trading_index,
        )

        self.assertEqual(list(daily_target_weight_df.columns), ["SSO", "SPY", "Cash"])
        self.assertTrue(np.allclose(daily_target_weight_df["SSO"].to_numpy(), [0.0, 0.0, 1.0, 0.0]))
        self.assertTrue(np.allclose(daily_target_weight_df["SPY"].to_numpy(), [0.0, 0.0, 0.0, 0.5]))
        self.assertTrue(np.allclose(daily_target_weight_df.sum(axis=1).to_numpy(), 1.0))

    def test_sso_green_wrapper_config_maps_score_three_to_sso_and_score_two_to_spy(self):
        self.assertEqual(SSO_GREEN_CONFIG.green_trade_symbol, "SSO")
        self.assertEqual(SSO_GREEN_CONFIG.caution_trade_symbol, "SPY")
        self.assertEqual(SSO_GREEN_CONFIG.trade_symbol_tuple, ("SSO", "SPY"))

    def test_score1_ief_wrapper_config_maps_score_one_to_ief(self):
        self.assertEqual(SCORE1_IEF_CONFIG.green_trade_symbol, "SPY")
        self.assertEqual(SCORE1_IEF_CONFIG.caution_trade_symbol, "SPY")
        self.assertEqual(SCORE1_IEF_CONFIG.risk_trade_symbol, "IEF")
        self.assertAlmostEqual(SCORE1_IEF_CONFIG.risk_target_weight_float, 1.0)
        self.assertEqual(SCORE1_IEF_CONFIG.trade_symbol_tuple, ("SPY", "IEF"))

    def test_score2_spy_ief_wrapper_config_splits_score_two_between_spy_and_ief(self):
        self.assertEqual(SCORE2_SPY_IEF_CONFIG.green_trade_symbol, "SPY")
        self.assertEqual(SCORE2_SPY_IEF_CONFIG.caution_trade_symbol, "SPY")
        self.assertEqual(SCORE2_SPY_IEF_CONFIG.caution_defensive_trade_symbol, "IEF")
        self.assertAlmostEqual(SCORE2_SPY_IEF_CONFIG.caution_target_weight_float, 0.5)
        self.assertAlmostEqual(SCORE2_SPY_IEF_CONFIG.caution_defensive_target_weight_float, 0.5)
        self.assertEqual(SCORE2_SPY_IEF_CONFIG.trade_symbol_tuple, ("SPY", "IEF"))

    def test_build_execution_target_weight_df_supports_score_one_ief_defensive_leg(self):
        config_obj = MarketRegimeFilterConfig(
            **{
                **self.make_config().__dict__,
                "risk_trade_symbol_str": "IEF",
                "risk_target_weight_float": 1.0,
            }
        )
        trading_index = pd.date_range("2024-01-01", periods=6, freq="B")
        signal_data_df = pd.DataFrame(
            {
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): [np.nan, 3.0, 2.0, 1.0, 0.0, 3.0],
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): [np.nan, 1.0, 0.5, 1.0, 0.0, 1.0],
            },
            index=trading_index,
            dtype=float,
        )
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)

        daily_target_weight_df = build_execution_target_weight_df(
            signal_data_df=signal_data_df,
            config=config_obj,
            result_index=trading_index,
        )

        self.assertEqual(list(daily_target_weight_df.columns), ["SPY", "IEF", "Cash"])
        self.assertTrue(np.allclose(daily_target_weight_df["SPY"].to_numpy(), [0.0, 0.0, 1.0, 0.5, 0.0, 0.0]))
        self.assertTrue(np.allclose(daily_target_weight_df["IEF"].to_numpy(), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.assertTrue(np.allclose(daily_target_weight_df["Cash"].to_numpy(), [1.0, 1.0, 0.0, 0.5, 0.0, 1.0]))
        self.assertTrue(np.allclose(daily_target_weight_df.sum(axis=1).to_numpy(), 1.0))

    def test_build_execution_target_weight_df_supports_score_two_spy_ief_split(self):
        config_obj = MarketRegimeFilterConfig(
            **{
                **self.make_config().__dict__,
                "caution_defensive_trade_symbol_str": "IEF",
                "caution_defensive_target_weight_float": 0.5,
            }
        )
        trading_index = pd.date_range("2024-01-01", periods=5, freq="B")
        signal_data_df = pd.DataFrame(
            {
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): [np.nan, 3.0, 2.0, 1.0, 0.0],
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): [np.nan, 1.0, 1.0, 0.0, 0.0],
            },
            index=trading_index,
            dtype=float,
        )
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)

        daily_target_weight_df = build_execution_target_weight_df(
            signal_data_df=signal_data_df,
            config=config_obj,
            result_index=trading_index,
        )

        self.assertEqual(list(daily_target_weight_df.columns), ["SPY", "IEF", "Cash"])
        self.assertTrue(np.allclose(daily_target_weight_df["SPY"].to_numpy(), [0.0, 0.0, 1.0, 0.5, 0.0]))
        self.assertTrue(np.allclose(daily_target_weight_df["IEF"].to_numpy(), [0.0, 0.0, 0.0, 0.5, 0.0]))
        self.assertTrue(np.allclose(daily_target_weight_df["Cash"].to_numpy(), [1.0, 1.0, 0.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(daily_target_weight_df.sum(axis=1).to_numpy(), 1.0))

    def test_build_execution_target_weight_df_rejects_multi_asset_target_weight_fallback(self):
        config_obj = MarketRegimeFilterConfig(
            **{
                **self.make_config().__dict__,
                "caution_defensive_trade_symbol_str": "IEF",
                "caution_defensive_target_weight_float": 0.5,
            }
        )
        trading_index = pd.date_range("2024-01-01", periods=4, freq="B")
        signal_data_df = pd.DataFrame(
            {
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): [np.nan, 1.0, 1.0, 0.0],
            },
            index=trading_index,
            dtype=float,
        )
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)

        with self.assertRaisesRegex(RuntimeError, "requires detector_score_int"):
            build_execution_target_weight_df(
                signal_data_df=signal_data_df,
                config=config_obj,
                result_index=trading_index,
            )

    def test_iterate_submits_target_percent_when_detector_allows_exposure(self):
        strategy_obj = self.make_strategy()
        strategy_obj.current_bar = pd.Timestamp("2024-01-05")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-04")
        strategy_obj._total_value_history_list = [100_000.0]

        close_ser = pd.Series(
            {
                ("SPY", "Close"): 100.0,
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): 2,
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 0.5,
            }
        )
        close_ser.index = pd.MultiIndex.from_tuples(close_ser.index)
        open_price_ser = pd.Series({"SPY": 101.0}, dtype=float)

        strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_ser, open_price_ser)

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 1)
        order_obj = order_list[0]
        self.assertIsInstance(order_obj, MarketOrder)
        self.assertEqual(order_obj.asset, "SPY")
        self.assertEqual(order_obj.unit, "percent")
        self.assertTrue(order_obj.target)
        self.assertAlmostEqual(float(order_obj.amount), 0.5)
        self.assertEqual(strategy_obj.current_trade_id_int, 1)

    def test_iterate_skips_when_target_share_count_is_unchanged(self):
        strategy_obj = self.make_strategy()
        strategy_obj.current_bar = pd.Timestamp("2024-01-05")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-04")
        strategy_obj._total_value_history_list = [100_000.0]
        strategy_obj.add_transaction(1, strategy_obj.previous_bar, "SPY", 500, 100.0, 50_000.0, 1, 0.0)
        strategy_obj.current_trade_id_int = 1

        close_ser = pd.Series(
            {
                ("SPY", "Close"): 100.0,
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): 2,
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 0.5,
            }
        )
        close_ser.index = pd.MultiIndex.from_tuples(close_ser.index)
        open_price_ser = pd.Series({"SPY": 101.0}, dtype=float)

        strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_ser, open_price_ser)

        self.assertEqual(len(strategy_obj.get_orders()), 0)

    def test_iterate_liquidates_when_regime_turns_risk_off(self):
        strategy_obj = self.make_strategy()
        strategy_obj.current_bar = pd.Timestamp("2024-01-05")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-04")
        strategy_obj._total_value_history_list = [100_000.0]
        strategy_obj.add_transaction(7, strategy_obj.previous_bar, "SPY", 500, 100.0, 50_000.0, 1, 0.0)
        strategy_obj.current_trade_id_int = 7

        close_ser = pd.Series(
            {
                ("SPY", "Close"): 100.0,
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): 1,
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 0.0,
            }
        )
        close_ser.index = pd.MultiIndex.from_tuples(close_ser.index)
        open_price_ser = pd.Series({"SPY": 101.0}, dtype=float)

        strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_ser, open_price_ser)

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 1)
        liquidation_order_obj = order_list[0]
        self.assertEqual(liquidation_order_obj.asset, "SPY")
        self.assertEqual(liquidation_order_obj.unit, "shares")
        self.assertTrue(liquidation_order_obj.target)
        self.assertEqual(liquidation_order_obj.amount, 0.0)
        self.assertEqual(liquidation_order_obj.trade_id, 7)
        self.assertEqual(strategy_obj.current_trade_id_int, DEFAULT_TRADE_ID_INT)

    def test_iterate_fails_loud_when_detector_score_is_missing(self):
        strategy_obj = self.make_strategy()
        strategy_obj.current_bar = pd.Timestamp("2024-01-05")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-04")
        strategy_obj._total_value_history_list = [100_000.0]

        close_ser = pd.Series({("SPY", "Close"): 100.0})
        close_ser.index = pd.MultiIndex.from_tuples(close_ser.index)
        open_price_ser = pd.Series({"SPY": 101.0}, dtype=float)

        with self.assertRaisesRegex(RuntimeError, "Missing market-regime detector score"):
            strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_ser, open_price_ser)

    def test_iterate_fails_loud_when_detector_score_is_nan(self):
        strategy_obj = self.make_strategy()
        strategy_obj.current_bar = pd.Timestamp("2024-01-05")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-04")
        strategy_obj._total_value_history_list = [100_000.0]

        close_ser = pd.Series(
            {
                ("SPY", "Close"): 100.0,
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): np.nan,
            }
        )
        close_ser.index = pd.MultiIndex.from_tuples(close_ser.index)
        open_price_ser = pd.Series({"SPY": 101.0}, dtype=float)

        with self.assertRaisesRegex(RuntimeError, "detector score is NaN"):
            strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_ser, open_price_ser)

    def test_iterate_switches_from_sso_green_to_spy_caution(self):
        config_obj = MarketRegimeFilterConfig(
            **{
                **self.make_config().__dict__,
                "green_trade_symbol_str": "SSO",
                "caution_trade_symbol_str": "SPY",
            }
        )
        strategy_obj = self.make_strategy(config=config_obj)
        strategy_obj.current_bar = pd.Timestamp("2024-01-05")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-04")
        strategy_obj._total_value_history_list = [100_000.0]
        strategy_obj.trade_id_int = 4
        strategy_obj.add_transaction(4, strategy_obj.previous_bar, "SSO", 400, 50.0, 20_000.0, 1, 0.0)
        strategy_obj.current_trade_id_map["SSO"] = 4

        close_ser = pd.Series(
            {
                ("SPY", "Close"): 100.0,
                ("SSO", "Close"): 50.0,
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): 2,
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 0.5,
            }
        )
        close_ser.index = pd.MultiIndex.from_tuples(close_ser.index)
        open_price_ser = pd.Series({"SPY": 101.0, "SSO": 51.0}, dtype=float)

        strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_ser, open_price_ser)

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 2)
        self.assertEqual([order.asset for order in order_list], ["SSO", "SPY"])
        self.assertEqual(order_list[0].unit, "shares")
        self.assertTrue(order_list[0].target)
        self.assertEqual(order_list[0].amount, 0.0)
        self.assertEqual(order_list[0].trade_id, 4)
        self.assertEqual(order_list[1].unit, "percent")
        self.assertTrue(order_list[1].target)
        self.assertAlmostEqual(float(order_list[1].amount), 0.5)
        self.assertEqual(order_list[1].trade_id, 5)

    def test_iterate_submits_two_target_orders_for_score_two_spy_ief_split(self):
        config_obj = MarketRegimeFilterConfig(
            **{
                **self.make_config().__dict__,
                "caution_defensive_trade_symbol_str": "IEF",
                "caution_defensive_target_weight_float": 0.5,
            }
        )
        strategy_obj = self.make_strategy(config=config_obj)
        strategy_obj.current_bar = pd.Timestamp("2024-01-05")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-04")
        strategy_obj._total_value_history_list = [100_000.0]

        close_ser = pd.Series(
            {
                ("SPY", "Close"): 100.0,
                ("IEF", "Close"): 95.0,
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): 2,
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 1.0,
            }
        )
        close_ser.index = pd.MultiIndex.from_tuples(close_ser.index)
        open_price_ser = pd.Series({"SPY": 101.0, "IEF": 96.0}, dtype=float)

        strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_ser, open_price_ser)

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 2)
        self.assertEqual([order.asset for order in order_list], ["SPY", "IEF"])
        self.assertTrue(all(order.unit == "percent" for order in order_list))
        self.assertTrue(all(order.target for order in order_list))
        self.assertTrue(np.allclose([float(order.amount) for order in order_list], [0.5, 0.5]))

    def test_iterate_submits_score_one_ief_target_order(self):
        config_obj = MarketRegimeFilterConfig(
            **{
                **self.make_config().__dict__,
                "risk_trade_symbol_str": "IEF",
                "risk_target_weight_float": 1.0,
            }
        )
        strategy_obj = self.make_strategy(config=config_obj)
        strategy_obj.current_bar = pd.Timestamp("2024-01-05")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-04")
        strategy_obj._total_value_history_list = [100_000.0]

        close_ser = pd.Series(
            {
                ("SPY", "Close"): 100.0,
                ("IEF", "Close"): 95.0,
                (SIGNAL_NAMESPACE_STR, "detector_score_int"): 1,
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 1.0,
            }
        )
        close_ser.index = pd.MultiIndex.from_tuples(close_ser.index)
        open_price_ser = pd.Series({"SPY": 101.0, "IEF": 96.0}, dtype=float)

        strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_ser, open_price_ser)

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 1)
        order_obj = order_list[0]
        self.assertEqual(order_obj.asset, "IEF")
        self.assertEqual(order_obj.unit, "percent")
        self.assertTrue(order_obj.target)
        self.assertAlmostEqual(float(order_obj.amount), 1.0)

    def test_run_daily_smoke_generates_summary_and_target_weights(self):
        pricing_data_df = self.make_pricing_data_df()
        strategy_obj = self.make_strategy()
        calendar_index = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp("2024-01-04")]

        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=calendar_index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=None,
        )

        self.assertIsNotNone(strategy_obj.summary)
        self.assertIn("Strategy", strategy_obj.summary.columns)
        self.assertGreater(len(strategy_obj.results), 0)
        self.assertGreater(len(strategy_obj.regime_signal_df), 0)
        self.assertTrue({"SPY", "Cash"}.issubset(strategy_obj.daily_target_weights.columns))
        self.assertTrue(np.allclose(strategy_obj.daily_target_weights.sum(axis=1).to_numpy(), 1.0))


if __name__ == "__main__":
    unittest.main()
