import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from alpha.engine.capacity_analysis import CapacityAnalysis

from strategies.mean_reversion import strategy_mr_sector_dispersion_ibs as base_module
from strategies.mean_reversion import strategy_mr_sector_dispersion_ibs_kie_ihi as kie_ihi_module
from strategies.mean_reversion import strategy_mr_sector_dispersion_ibs_kie_ihi_xlc as kie_ihi_xlc_module


class SectorDispersionIbsKieIhiVariantTests(unittest.TestCase):
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
        high_vec = np.maximum(high_vec, open_vec)
        low_vec = np.minimum(low_vec, open_vec)
        return {
            "Open": open_vec.tolist(),
            "High": high_vec.tolist(),
            "Low": low_vec.tolist(),
            "Close": close_vec.tolist(),
        }

    def make_pricing_data_df(
        self,
        signal_symbol_str: str,
        date_index: pd.DatetimeIndex,
        signal_day_int: int,
        fill_day_int: int,
    ) -> pd.DataFrame:
        log_range_list = [0.010 + 0.001 * index_int for index_int in range(len(date_index))]
        log_range_list[signal_day_int] = 0.120
        neutral_ibs_list = [0.50] * len(date_index)
        signal_ibs_list = [0.50] * len(date_index)
        signal_ibs_list[signal_day_int] = 0.05
        signal_open_list = [100.0] * len(date_index)
        signal_open_list[fill_day_int] = 111.0

        column_map: dict[tuple[str, str], pd.Series] = {}
        for symbol_str in ("SOXX", "IGV", "IBB", "KIE", "IHI", "XLC"):
            if symbol_str == signal_symbol_str:
                field_map_dict = self.make_symbol_ohlc_map(
                    log_range_list=log_range_list,
                    ibs_list=signal_ibs_list,
                    open_list=signal_open_list,
                )
            else:
                field_map_dict = self.make_symbol_ohlc_map(
                    log_range_list=log_range_list,
                    ibs_list=neutral_ibs_list,
                )
            for field_str, value_list in field_map_dict.items():
                column_map[(symbol_str, field_str)] = pd.Series(
                    value_list,
                    index=date_index,
                    dtype=float,
                )
        column_map[("$SPX", "Close")] = pd.Series(
            [5000.0 + index_int for index_int in range(len(date_index))],
            index=date_index,
            dtype=float,
        )
        for symbol_str in ("SOXX", "IGV", "IBB", "KIE", "IHI", "XLC"):
            column_map[(symbol_str, "Volume")] = pd.Series(
                [1_000_000.0] * len(date_index),
                index=date_index,
                dtype=float,
            )
            column_map[(symbol_str, "Turnover")] = pd.Series(
                [100_000_000.0] * len(date_index),
                index=date_index,
                dtype=float,
            )

        pricing_data_df = pd.DataFrame(column_map, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_default_configs_use_fixed_recommended_baskets(self):
        self.assertEqual(
            kie_ihi_module.STRATEGY_SYMBOL_TUPLE,
            ("SOXX", "IGV", "IBB", "KIE", "IHI"),
        )
        self.assertEqual(
            kie_ihi_xlc_module.STRATEGY_SYMBOL_TUPLE,
            ("SOXX", "IGV", "IBB", "KIE", "IHI", "XLC"),
        )
        self.assertEqual(kie_ihi_module.DEFAULT_CONFIG.symbol_tuple, kie_ihi_module.STRATEGY_SYMBOL_TUPLE)
        self.assertEqual(kie_ihi_xlc_module.DEFAULT_CONFIG.symbol_tuple, kie_ihi_xlc_module.STRATEGY_SYMBOL_TUPLE)
        self.assertAlmostEqual(kie_ihi_module.DEFAULT_CONFIG.portfolio_leverage_float, 1.0)
        self.assertAlmostEqual(kie_ihi_xlc_module.DEFAULT_CONFIG.portfolio_leverage_float, 1.0)
        self.assertAlmostEqual(kie_ihi_module.DEFAULT_CONFIG.slippage_float, 0.00025)
        self.assertAlmostEqual(kie_ihi_xlc_module.DEFAULT_CONFIG.slippage_float, 0.00025)

    def test_kie_ihi_capacity_builder_runs_with_etf_proxy_profile(self):
        date_index = pd.bdate_range("2024-01-02", periods=25)
        pricing_data_df = self.make_pricing_data_df(
            signal_symbol_str="KIE",
            date_index=date_index,
            signal_day_int=22,
            fill_day_int=23,
        )
        with patch.object(
            base_module,
            "get_sector_dispersion_ibs_data",
            return_value=pricing_data_df,
        ):
            capacity_input_dict = kie_ihi_module.build_capacity_analysis_inputs(
                capital_base_float=250_000.0,
            )

        self.assertEqual(capacity_input_dict["execution_policy_str"], "MOO")
        self.assertEqual(capacity_input_dict["impact_profile_str"], "MOO_ETF_PROXY")
        capacity_result_obj = CapacityAnalysis(**capacity_input_dict).run()
        self.assertEqual(capacity_result_obj.impact_profile_str, "MOO_ETF_PROXY")
        self.assertTrue(
            (
                capacity_result_obj.equity_curve_df["stress_equity_float"]
                <= capacity_result_obj.equity_curve_df["central_equity_float"] + 1e-12
            ).all()
        )

    def test_run_variant_uses_fixed_basket_and_next_open_fill(self):
        date_index = pd.bdate_range("2024-01-02", periods=25)
        signal_day_int = 22
        fill_day_int = 23
        case_tuple = (
            (kie_ihi_module, "IHI"),
            (kie_ihi_xlc_module, "XLC"),
        )

        for module_obj, signal_symbol_str in case_tuple:
            with self.subTest(strategy_name_str=module_obj.STRATEGY_NAME_STR):
                pricing_data_df = self.make_pricing_data_df(
                    signal_symbol_str=signal_symbol_str,
                    date_index=date_index,
                    signal_day_int=signal_day_int,
                    fill_day_int=fill_day_int,
                )

                strategy_obj = module_obj.run_variant(
                    show_display_bool=False,
                    save_results_bool=False,
                    backtest_start_date_str=date_index[0].date().isoformat(),
                    capital_base_float=100_000.0,
                    pricing_data_df=pricing_data_df,
                    audit_override_bool=False,
                )

                transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
                expected_target_share_float = (
                    100_000.0
                    * (module_obj.DEFAULT_CONFIG.portfolio_leverage_float / len(module_obj.STRATEGY_SYMBOL_TUPLE))
                    / float(pricing_data_df.loc[date_index[signal_day_int], (signal_symbol_str, "Close")])
                )
                expected_fill_price_float = 111.0 * (
                    1.0 + module_obj.DEFAULT_CONFIG.slippage_float
                )

                self.assertEqual(strategy_obj.name, module_obj.STRATEGY_NAME_STR)
                self.assertEqual(strategy_obj.symbol_tuple, module_obj.STRATEGY_SYMBOL_TUPLE)
                self.assertEqual(strategy_obj.results.index[0], date_index[21])
                self.assertEqual(len(transaction_df), 1)
                entry_row_ser = transaction_df.iloc[0]
                self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), date_index[fill_day_int])
                self.assertEqual(entry_row_ser["asset"], signal_symbol_str)
                self.assertAlmostEqual(float(entry_row_ser["price"]), expected_fill_price_float)
                self.assertAlmostEqual(float(entry_row_ser["amount"]), expected_target_share_float)

    def test_saved_metadata_points_to_each_wrapper_module(self):
        date_index = pd.bdate_range("2024-01-02", periods=25)
        case_tuple = (
            (kie_ihi_module, "IHI", "SectorDispersionIbsKieIhiStrategy"),
            (kie_ihi_xlc_module, "XLC", "SectorDispersionIbsKieIhiXlcStrategy"),
        )

        for module_obj, signal_symbol_str, class_name_str in case_tuple:
            with self.subTest(strategy_name_str=module_obj.STRATEGY_NAME_STR):
                pricing_data_df = self.make_pricing_data_df(
                    signal_symbol_str=signal_symbol_str,
                    date_index=date_index,
                    signal_day_int=22,
                    fill_day_int=23,
                )

                with tempfile.TemporaryDirectory() as temp_dir_str:
                    module_obj.run_variant(
                        show_display_bool=False,
                        save_results_bool=True,
                        output_dir_str=temp_dir_str,
                        backtest_start_date_str=date_index[0].date().isoformat(),
                        capital_base_float=100_000.0,
                        pricing_data_df=pricing_data_df,
                        audit_override_bool=False,
                    )
                    metadata_path_list = sorted(
                        (
                            Path(temp_dir_str)
                            / "research"
                            / "strategy"
                            / module_obj.STRATEGY_NAME_STR
                            / "vanilla_backtest"
                        ).glob("*/metadata.json")
                    )
                    self.assertEqual(len(metadata_path_list), 1)
                    metadata_dict = json.loads(metadata_path_list[0].read_text(encoding="utf-8"))

                self.assertEqual(metadata_dict["class_module"], module_obj.__name__)
                self.assertEqual(metadata_dict["class_name"], class_name_str)

    def test_fixed_wrappers_wait_for_their_latest_constituent(self):
        date_index = pd.bdate_range("2024-01-02", periods=30)
        case_tuple = (
            (kie_ihi_module, "IHI"),
            (kie_ihi_xlc_module, "XLC"),
        )

        for module_obj, delayed_symbol_str in case_tuple:
            with self.subTest(strategy_name_str=module_obj.STRATEGY_NAME_STR):
                pricing_data_df = self.make_pricing_data_df(
                    signal_symbol_str=delayed_symbol_str,
                    date_index=date_index,
                    signal_day_int=26,
                    fill_day_int=27,
                )
                for field_str in ("Open", "High", "Low", "Close"):
                    pricing_data_df.loc[
                        date_index[:4],
                        (delayed_symbol_str, field_str),
                    ] = np.nan

                strategy_obj = module_obj.run_variant(
                    show_display_bool=False,
                    save_results_bool=False,
                    backtest_start_date_str=date_index[0].date().isoformat(),
                    capital_base_float=100_000.0,
                    pricing_data_df=pricing_data_df,
                    audit_override_bool=False,
                )

                self.assertEqual(strategy_obj.results.index[0], date_index[25])


if __name__ == "__main__":
    unittest.main()
