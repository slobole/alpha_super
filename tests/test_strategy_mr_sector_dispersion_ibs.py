import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.mean_reversion import strategy_mr_sector_dispersion_ibs_kie as kie_module
from strategies.mean_reversion import strategy_mr_sector_dispersion_ibs_kie_ihi as kie_ihi_module
from strategies.mean_reversion import (
    strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200 as kie_ihi_asset_sma_module,
)
from strategies.mean_reversion import strategy_mr_sector_dispersion_ibs_kie_ihi_xlc as kie_ihi_xlc_module
from strategies.mean_reversion import (
    strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200 as asset_sma_module,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    DEFAULT_CONFIG,
    ORIGINAL_SYMBOL_TUPLE,
    SectorDispersionIbsConfig,
    SectorDispersionIbsStrategy,
    UNIVERSE_A_SYMBOL_TUPLE,
    UNIVERSE_B_SYMBOL_TUPLE,
    UNIVERSE_C_SYMBOL_TUPLE,
    compute_sector_dispersion_ibs_signal_df,
    normalize_universe_name_str,
    resolve_effective_backtest_start_date_str,
    resolve_full_basket_calendar_idx,
    resolve_history_start_date_str,
    resolve_universe_symbol_tuple,
    run_variant,
)


class SectorDispersionIbsStrategyTests(unittest.TestCase):
    def make_pricing_data_df(
        self,
        price_map_dict: dict[str, dict[str, list[float]]],
        date_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        column_map: dict[tuple[str, str], pd.Series] = {}
        for symbol_str, field_map_dict in price_map_dict.items():
            for field_str, value_list in field_map_dict.items():
                column_map[(symbol_str, field_str)] = pd.Series(
                    value_list,
                    index=date_index,
                    dtype=float,
                )
        pricing_data_df = pd.DataFrame(column_map, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], object]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def make_symbol_ohlc_map(
        self,
        log_range_list: list[float],
        ibs_list: list[float],
        open_list: list[float] | None = None,
    ) -> dict[str, list[float]]:
        low_vec = np.full(len(log_range_list), 100.0)
        high_vec = low_vec * np.exp(np.array(log_range_list, dtype=float))
        close_vec = low_vec + np.array(ibs_list, dtype=float) * (high_vec - low_vec)
        if open_list is None:
            open_vec = close_vec.copy()
        else:
            open_vec = np.array(open_list, dtype=float)
        return {
            "Open": open_vec.tolist(),
            "High": high_vec.tolist(),
            "Low": low_vec.tolist(),
            "Close": close_vec.tolist(),
        }

    def make_strategy(self, config_obj: SectorDispersionIbsConfig | None = None) -> SectorDispersionIbsStrategy:
        if config_obj is None:
            config_obj = replace(
                DEFAULT_CONFIG,
                symbol_tuple=("AAA", "BBB", "CCC"),
                history_start_date_str="2023-12-01",
                backtest_start_date_str="2024-01-01",
                range_vol_lookback_day_int=3,
                capital_base_float=1_000.0,
                slippage_float=0.0,
                commission_per_share_float=0.0,
                commission_minimum_float=0.0,
            )
        return SectorDispersionIbsStrategy(
            name="SectorDispersionIbsTest",
            benchmarks=[],
            config_obj=config_obj,
        )

    def test_default_config_uses_unlevered_article_signal_translation(self):
        self.assertEqual(DEFAULT_CONFIG.symbol_tuple, ORIGINAL_SYMBOL_TUPLE)
        self.assertEqual(DEFAULT_CONFIG.entry_ibs_max_float, 0.10)
        self.assertEqual(DEFAULT_CONFIG.exit_ibs_min_float, 0.90)
        self.assertEqual(DEFAULT_CONFIG.range_vol_lookback_day_int, 21)
        self.assertEqual(DEFAULT_CONFIG.min_relative_range_float, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.portfolio_leverage_float, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.slippage_float, 0.00025)
        self.assertAlmostEqual(DEFAULT_CONFIG.commission_per_share_float, 0.00525)
        self.assertEqual(DEFAULT_CONFIG.commission_minimum_float, 0.0)

    def test_universe_variants_resolve_expected_baskets(self):
        self.assertEqual(
            UNIVERSE_A_SYMBOL_TUPLE,
            (
                "SOXX",
                "IGV",
                "IBB",
                "XLF",
                "XLE",
                "XLI",
                "XLY",
                "XLP",
                "XLU",
                "XLRE",
                "XLB",
                "XLC",
            ),
        )
        self.assertEqual(
            UNIVERSE_B_SYMBOL_TUPLE,
            UNIVERSE_A_SYMBOL_TUPLE + ("KRE", "XOP", "ITA", "XRT", "ITB", "XME", "IHI"),
        )
        self.assertEqual(
            UNIVERSE_C_SYMBOL_TUPLE,
            UNIVERSE_B_SYMBOL_TUPLE + ("XBI", "KIE", "IAI", "IYT", "IHF", "IHE", "XHB", "XAR", "XES"),
        )
        self.assertEqual(normalize_universe_name_str("Universe A"), "a")
        self.assertEqual(resolve_universe_symbol_tuple("B"), UNIVERSE_B_SYMBOL_TUPLE)

        config_obj = SectorDispersionIbsConfig(universe_name_str="C")
        self.assertEqual(config_obj.universe_name_str, "c")
        self.assertEqual(config_obj.symbol_tuple, UNIVERSE_C_SYMBOL_TUPLE)

    def test_resolve_history_start_moves_before_portfolio_manager_start_override(self):
        history_start_date_str = resolve_history_start_date_str(
            config_obj=DEFAULT_CONFIG,
            backtest_start_date_str="2004-01-01",
        )

        self.assertLess(
            pd.Timestamp(history_start_date_str),
            pd.Timestamp("2004-01-01"),
        )

    def test_effective_start_honors_explicit_caller_boundary(self):
        self.assertEqual(
            resolve_effective_backtest_start_date_str(DEFAULT_CONFIG, "2004-01-01"),
            "2004-01-01",
        )
        self.assertEqual(
            resolve_effective_backtest_start_date_str(DEFAULT_CONFIG, "2020-01-02"),
            "2020-01-02",
        )
        self.assertEqual(
            resolve_effective_backtest_start_date_str(DEFAULT_CONFIG, None),
            "2004-01-01",
        )

    def test_every_fixed_wrapper_honors_explicit_start_and_earlier_default(self):
        date_index = pd.bdate_range("2015-01-02", periods=280)
        neutral_ohlc_dict = self.make_symbol_ohlc_map(
            log_range_list=[0.01] * len(date_index),
            ibs_list=[0.50] * len(date_index),
        )
        price_map_dict = {
            symbol_str: neutral_ohlc_dict
            for symbol_str in UNIVERSE_C_SYMBOL_TUPLE
        }
        price_map_dict["$SPX"] = {
            "Close": [5000.0 + index_int for index_int in range(len(date_index))]
        }
        pricing_data_df = self.make_pricing_data_df(price_map_dict, date_index)
        wrapper_case_list = [
            ("base", run_variant, False),
            ("kie", kie_module.run_variant, False),
            ("kie_ihi", kie_ihi_module.run_variant, False),
            ("kie_ihi_asset_sma200", kie_ihi_asset_sma_module.run_variant, True),
            ("kie_ihi_xlc", kie_ihi_xlc_module.run_variant, False),
            ("asset_sma200", asset_sma_module.run_variant, True),
        ]

        for wrapper_name_str, run_variant_func, requires_sma200_bool in wrapper_case_list:
            with self.subTest(wrapper_name_str=wrapper_name_str, mode_str="explicit"):
                explicit_strategy_obj = run_variant_func(
                    show_display_bool=False,
                    save_results_bool=False,
                    backtest_start_date_str="2004-01-01",
                    pricing_data_df=pricing_data_df,
                    audit_override_bool=False,
                )
                expected_explicit_start_ts = date_index[
                    199 if requires_sma200_bool else 21
                ]
                self.assertEqual(
                    explicit_strategy_obj.config_obj.backtest_start_date_str,
                    "2004-01-01",
                )
                self.assertEqual(
                    explicit_strategy_obj.results.index[0],
                    expected_explicit_start_ts,
                )

            with self.subTest(wrapper_name_str=wrapper_name_str, mode_str="default"):
                default_strategy_obj = run_variant_func(
                    show_display_bool=False,
                    save_results_bool=False,
                    pricing_data_df=pricing_data_df,
                    audit_override_bool=False,
                )
                self.assertEqual(
                    default_strategy_obj.config_obj.backtest_start_date_str,
                    "2004-01-01",
                )
                self.assertEqual(
                    default_strategy_obj.config_obj.history_start_date_str,
                    "2003-01-01",
                )
                self.assertEqual(
                    default_strategy_obj.results.index[0],
                    date_index[199 if requires_sma200_bool else 21],
                )

    def test_full_basket_calendar_waits_for_every_symbol_warmup(self):
        date_index = pd.bdate_range("2024-01-02", periods=10)
        neutral_ohlc_dict = self.make_symbol_ohlc_map(
            log_range_list=[0.01] * len(date_index),
            ibs_list=[0.50] * len(date_index),
        )
        delayed_ohlc_dict = {
            field_str: [np.nan] * 4 + value_list[4:]
            for field_str, value_list in neutral_ohlc_dict.items()
        }
        pricing_data_df = self.make_pricing_data_df(
            {
                "AAA": neutral_ohlc_dict,
                "BBB": neutral_ohlc_dict,
                "CCC": delayed_ohlc_dict,
            },
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB", "CCC"),
            backtest_start_date_str=date_index[0].date().isoformat(),
            range_vol_lookback_day_int=2,
        )

        calendar_idx = resolve_full_basket_calendar_idx(
            pricing_data_df=pricing_data_df,
            config_obj=config_obj,
        )

        self.assertEqual(calendar_idx[0], date_index[6])

    def test_full_basket_calendar_fails_when_one_etf_never_becomes_ready(self):
        date_index = pd.bdate_range("2024-01-02", periods=6)
        neutral_ohlc_dict = self.make_symbol_ohlc_map(
            log_range_list=[0.01] * len(date_index),
            ibs_list=[0.50] * len(date_index),
        )
        missing_ohlc_dict = {
            field_str: [np.nan] * len(date_index)
            for field_str in neutral_ohlc_dict
        }
        pricing_data_df = self.make_pricing_data_df(
            {"AAA": neutral_ohlc_dict, "BBB": missing_ohlc_dict},
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB"),
            backtest_start_date_str=date_index[0].date().isoformat(),
            range_vol_lookback_day_int=2,
        )

        with self.assertRaisesRegex(ValueError, "No full-basket Sector Dispersion start"):
            resolve_full_basket_calendar_idx(
                pricing_data_df=pricing_data_df,
                config_obj=config_obj,
            )

    def test_full_basket_calendar_fails_on_post_start_invalid_ohlc(self):
        date_index = pd.bdate_range("2024-01-02", periods=8)
        neutral_ohlc_dict = self.make_symbol_ohlc_map(
            log_range_list=[0.01] * len(date_index),
            ibs_list=[0.50] * len(date_index),
        )
        broken_ohlc_dict = {
            field_str: list(value_list)
            for field_str, value_list in neutral_ohlc_dict.items()
        }
        broken_ohlc_dict["Open"][6] = np.nan
        pricing_data_df = self.make_pricing_data_df(
            {"AAA": neutral_ohlc_dict, "BBB": broken_ohlc_dict},
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB"),
            backtest_start_date_str=date_index[0].date().isoformat(),
            range_vol_lookback_day_int=2,
        )

        with self.assertRaisesRegex(ValueError, "became invalid after the effective start"):
            resolve_full_basket_calendar_idx(
                pricing_data_df=pricing_data_df,
                config_obj=config_obj,
            )

    def test_full_basket_calendar_allows_integrity_valid_flat_bar_after_start(self):
        date_index = pd.bdate_range("2024-01-02", periods=8)
        neutral_ohlc_dict = self.make_symbol_ohlc_map(
            log_range_list=[0.01] * len(date_index),
            ibs_list=[0.50] * len(date_index),
        )
        flat_ohlc_dict = {
            field_str: list(value_list)
            for field_str, value_list in neutral_ohlc_dict.items()
        }
        for field_str in ("Open", "High", "Low", "Close"):
            flat_ohlc_dict[field_str][6] = 100.0
        pricing_data_df = self.make_pricing_data_df(
            {"AAA": neutral_ohlc_dict, "BBB": flat_ohlc_dict},
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB"),
            backtest_start_date_str=date_index[0].date().isoformat(),
            range_vol_lookback_day_int=2,
        )

        calendar_idx = resolve_full_basket_calendar_idx(pricing_data_df, config_obj)

        self.assertEqual(calendar_idx[0], date_index[2])
        self.assertIn(date_index[6], calendar_idx)

    def test_full_basket_calendar_rejects_malformed_ohlc_after_start(self):
        date_index = pd.bdate_range("2024-01-02", periods=8)
        neutral_ohlc_dict = self.make_symbol_ohlc_map(
            log_range_list=[0.01] * len(date_index),
            ibs_list=[0.50] * len(date_index),
        )
        malformed_ohlc_dict = {
            field_str: list(value_list)
            for field_str, value_list in neutral_ohlc_dict.items()
        }
        malformed_ohlc_dict["Close"][6] = malformed_ohlc_dict["High"][6] + 1.0
        pricing_data_df = self.make_pricing_data_df(
            {"AAA": neutral_ohlc_dict, "BBB": malformed_ohlc_dict},
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB"),
            backtest_start_date_str=date_index[0].date().isoformat(),
            range_vol_lookback_day_int=2,
        )

        with self.assertRaisesRegex(ValueError, "became invalid after the effective start"):
            resolve_full_basket_calendar_idx(pricing_data_df, config_obj)

    def test_full_basket_calendar_waits_for_readiness_after_configured_start_gap(self):
        date_index = pd.bdate_range("2024-01-02", periods=12)
        neutral_ohlc_dict = self.make_symbol_ohlc_map(
            log_range_list=[0.01] * len(date_index),
            ibs_list=[0.50] * len(date_index),
        )
        gapped_ohlc_dict = {
            field_str: list(value_list)
            for field_str, value_list in neutral_ohlc_dict.items()
        }
        for field_str in gapped_ohlc_dict:
            gapped_ohlc_dict[field_str][5] = np.nan
        pricing_data_df = self.make_pricing_data_df(
            {"AAA": neutral_ohlc_dict, "BBB": gapped_ohlc_dict},
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB"),
            backtest_start_date_str=date_index[5].date().isoformat(),
            range_vol_lookback_day_int=2,
        )

        calendar_idx = resolve_full_basket_calendar_idx(pricing_data_df, config_obj)

        self.assertEqual(calendar_idx[0], date_index[8])

    def test_extra_close_warmup_does_not_require_old_range_rows(self):
        date_index = pd.bdate_range("2024-01-02", periods=205)
        neutral_ohlc_dict = self.make_symbol_ohlc_map(
            log_range_list=[0.01] * len(date_index),
            ibs_list=[0.50] * len(date_index),
        )
        old_flat_ohlc_dict = {
            field_str: list(value_list)
            for field_str, value_list in neutral_ohlc_dict.items()
        }
        old_flat_ohlc_dict["High"][:100] = old_flat_ohlc_dict["Low"][:100]
        pricing_data_df = self.make_pricing_data_df(
            {"AAA": neutral_ohlc_dict, "BBB": old_flat_ohlc_dict},
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB"),
            backtest_start_date_str=date_index[0].date().isoformat(),
            range_vol_lookback_day_int=2,
        )

        calendar_idx = resolve_full_basket_calendar_idx(
            pricing_data_df=pricing_data_df,
            config_obj=config_obj,
            required_close_history_observation_count_int=200,
        )

        self.assertEqual(calendar_idx[0], date_index[199])

    def test_original_and_named_universes_apply_full_basket_gate_in_run_variant(self):
        date_index = pd.bdate_range("2015-12-01", periods=60)
        neutral_ohlc_dict = self.make_symbol_ohlc_map(
            log_range_list=[0.01] * len(date_index),
            ibs_list=[0.50] * len(date_index),
        )
        delayed_symbol_by_universe_dict = {
            "original": None,
            "a": "XLC",
            "b": "IHI",
            "c": "KIE",
        }
        for universe_name_str, delayed_symbol_str in delayed_symbol_by_universe_dict.items():
            with self.subTest(universe_name_str=universe_name_str):
                price_map_dict = {}
                for symbol_str in UNIVERSE_C_SYMBOL_TUPLE:
                    if symbol_str == delayed_symbol_str:
                        price_map_dict[symbol_str] = {
                            field_str: [np.nan] * 4 + value_list[4:]
                            for field_str, value_list in neutral_ohlc_dict.items()
                        }
                    else:
                        price_map_dict[symbol_str] = neutral_ohlc_dict
                price_map_dict["$SPX"] = {
                    "Close": [5000.0 + index_int for index_int in range(len(date_index))]
                }
                pricing_data_df = self.make_pricing_data_df(price_map_dict, date_index)
                strategy_obj = run_variant(
                    show_display_bool=False,
                    save_results_bool=False,
                    backtest_start_date_str="2004-01-01",
                    universe_name_str=universe_name_str,
                    pricing_data_df=pricing_data_df,
                    audit_override_bool=False,
                )
                expected_start_ts = (
                    date_index[21]
                    if universe_name_str == "original"
                    else date_index[25]
                )
                self.assertEqual(strategy_obj.results.index[0], expected_start_ts)

    def test_compute_signals_uses_lagged_range_volatility(self):
        date_index = pd.bdate_range("2024-01-02", periods=6)
        log_range_list = [0.01, 0.02, 0.03, 0.04, 0.10, 0.12]
        ibs_list = [0.50, 0.40, 0.30, 0.20, 0.05, 0.95]
        pricing_data_df = self.make_pricing_data_df(
            {
                "AAA": self.make_symbol_ohlc_map(log_range_list, ibs_list),
                "BBB": self.make_symbol_ohlc_map(log_range_list, [0.50] * 6),
                "CCC": self.make_symbol_ohlc_map(log_range_list, [0.50] * 6),
            },
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB", "CCC"),
            history_start_date_str="2023-12-01",
            backtest_start_date_str="2024-01-01",
            range_vol_lookback_day_int=3,
        )

        signal_data_df = compute_sector_dispersion_ibs_signal_df(
            pricing_data_df=pricing_data_df,
            config_obj=config_obj,
        )

        check_ts = date_index[4]
        expected_prior_vol_float = float(pd.Series(log_range_list[1:4]).std())
        expected_relative_range_float = float(log_range_list[4] / expected_prior_vol_float)
        unshifted_vol_float = float(pd.Series(log_range_list[2:5]).std())

        # *** CRITICAL*** expected_relative_range_float uses ranges ending at
        # T-1. The unshifted denominator would include today's large range and
        # dampen the signal with same-day information.
        self.assertAlmostEqual(
            float(signal_data_df.loc[check_ts, ("AAA", "range_vol_3_ser")]),
            expected_prior_vol_float,
        )
        self.assertAlmostEqual(
            float(signal_data_df.loc[check_ts, ("AAA", "relative_range_ser")]),
            expected_relative_range_float,
        )
        self.assertNotAlmostEqual(expected_prior_vol_float, unshifted_vol_float)
        self.assertAlmostEqual(float(signal_data_df.loc[check_ts, ("AAA", "ibs_value_ser")]), 0.05)
        self.assertTrue(bool(signal_data_df.loc[check_ts, ("AAA", "entry_signal_bool")]))
        self.assertFalse(bool(signal_data_df.loc[check_ts, ("AAA", "exit_signal_bool")]))
        self.assertTrue(bool(signal_data_df.loc[date_index[5], ("AAA", "exit_signal_bool")]))

    def test_iterate_exits_held_symbol_and_enters_new_signal(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-08")
        strategy_obj.current_bar = pd.Timestamp("2024-01-09")
        strategy_obj.add_transaction(7, pd.Timestamp("2024-01-05"), "AAA", 10.0, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.current_trade_map["AAA"] = 7

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "exit_signal_bool"): True,
                ("AAA", "entry_signal_bool"): False,
                ("BBB", "exit_signal_bool"): False,
                ("BBB", "entry_signal_bool"): True,
                ("BBB", "Close"): 100.0,
                ("CCC", "exit_signal_bool"): False,
                ("CCC", "entry_signal_bool"): False,
            }
        )

        strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_row_ser, pd.Series(dtype=float))

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 2)
        self.assertIsInstance(order_list[0], MarketOrder)
        self.assertEqual(order_list[0].asset, "AAA")
        self.assertEqual(order_list[0].amount, 0.0)
        self.assertTrue(order_list[0].target)
        self.assertEqual(order_list[0].trade_id, 7)
        self.assertEqual(order_list[1].asset, "BBB")
        self.assertEqual(order_list[1].unit, "shares")
        self.assertTrue(order_list[1].target)
        expected_target_share_float = (
            float(strategy_obj.previous_total_value)
            * strategy_obj.target_weight_float
            / 100.0
        )
        self.assertAlmostEqual(order_list[1].amount, expected_target_share_float)

    def test_iterate_does_not_rebalance_held_position_without_exit(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-08")
        strategy_obj.current_bar = pd.Timestamp("2024-01-09")
        strategy_obj.add_transaction(11, pd.Timestamp("2024-01-05"), "AAA", 10.0, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.current_trade_map["AAA"] = 11

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "exit_signal_bool"): False,
                ("AAA", "entry_signal_bool"): True,
                ("BBB", "exit_signal_bool"): False,
                ("BBB", "entry_signal_bool"): False,
                ("CCC", "exit_signal_bool"): False,
                ("CCC", "entry_signal_bool"): False,
            }
        )

        strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_row_ser, pd.Series(dtype=float))

        self.assertEqual(strategy_obj.get_orders(), [])

    def test_run_daily_fills_entry_at_next_open(self):
        date_index = pd.bdate_range("2024-01-02", periods=6)
        log_range_list = [0.01, 0.02, 0.03, 0.04, 0.10, 0.02]
        aaa_ibs_list = [0.50, 0.50, 0.50, 0.50, 0.05, 0.50]
        pricing_data_df = self.make_pricing_data_df(
            {
                "AAA": self.make_symbol_ohlc_map(
                    log_range_list,
                    aaa_ibs_list,
                    open_list=[100.0, 101.0, 102.0, 103.0, 104.0, 110.0],
                ),
                "BBB": self.make_symbol_ohlc_map(log_range_list, [0.50] * 6),
                "CCC": self.make_symbol_ohlc_map(log_range_list, [0.50] * 6),
            },
            date_index=date_index,
        )
        strategy_obj = self.make_strategy()

        run_daily(
            strategy_obj,
            pricing_data_df,
            date_index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=False,
        )

        transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
        self.assertEqual(len(transaction_df), 1)
        entry_row_ser = transaction_df.iloc[0]
        signal_ts = date_index[4]
        fill_ts = date_index[5]
        expected_amount_float = (
            strategy_obj._capital_base
            * strategy_obj.target_weight_float
            / float(pricing_data_df.loc[signal_ts, ("AAA", "Close")])
        )

        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), fill_ts)
        self.assertEqual(entry_row_ser["asset"], "AAA")
        self.assertAlmostEqual(float(entry_row_ser["price"]), 110.0)
        self.assertAlmostEqual(float(entry_row_ser["amount"]), expected_amount_float)


if __name__ == "__main__":
    unittest.main()
