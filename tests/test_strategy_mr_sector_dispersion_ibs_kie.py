import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie import (
    DEFAULT_CONFIG,
    SectorDispersionIbsKieStrategy,
    STRATEGY_NAME_STR,
    STRATEGY_SYMBOL_TUPLE,
    run_variant,
)


class SectorDispersionIbsKieStrategyTests(unittest.TestCase):
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

    def make_kie_signal_pricing_data_tuple(self) -> tuple[pd.DataFrame, pd.DatetimeIndex, int, int]:
        date_index = pd.bdate_range("2024-01-02", periods=25)
        log_range_list = [0.010 + 0.001 * index_int for index_int in range(len(date_index))]
        signal_day_int = 22
        fill_day_int = 23
        log_range_list[signal_day_int] = 0.120
        neutral_ibs_list = [0.50] * len(date_index)
        kie_ibs_list = [0.50] * len(date_index)
        kie_ibs_list[signal_day_int] = 0.05
        kie_open_list = [100.0] * len(date_index)
        kie_open_list[fill_day_int] = 111.0

        price_map_dict = {
            "SOXX": self.make_symbol_ohlc_map(log_range_list, neutral_ibs_list),
            "IGV": self.make_symbol_ohlc_map(log_range_list, neutral_ibs_list),
            "IBB": self.make_symbol_ohlc_map(log_range_list, neutral_ibs_list),
            "KIE": self.make_symbol_ohlc_map(
                log_range_list,
                kie_ibs_list,
                open_list=kie_open_list,
            ),
            "$SPX": {
                "Close": [5000.0 + index_int for index_int in range(len(date_index))],
            },
        }
        return (
            self.make_pricing_data_df(price_map_dict, date_index),
            date_index,
            signal_day_int,
            fill_day_int,
        )

    def test_default_config_is_original_plus_kie(self):
        self.assertEqual(STRATEGY_NAME_STR, "strategy_mr_sector_dispersion_ibs_kie")
        self.assertEqual(STRATEGY_SYMBOL_TUPLE, ("SOXX", "IGV", "IBB", "KIE"))
        self.assertEqual(DEFAULT_CONFIG.symbol_tuple, STRATEGY_SYMBOL_TUPLE)
        self.assertEqual(DEFAULT_CONFIG.benchmark_symbol_str, "$SPX")
        self.assertAlmostEqual(DEFAULT_CONFIG.portfolio_leverage_float, 1.5)

    def test_run_variant_uses_kie_basket_and_next_open_fill(self):
        pricing_data_df, date_index, signal_day_int, fill_day_int = (
            self.make_kie_signal_pricing_data_tuple()
        )

        strategy_obj = run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str=date_index[0].date().isoformat(),
            capital_base_float=100_000.0,
            pricing_data_df=pricing_data_df,
            audit_override_bool=False,
        )

        transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
        self.assertEqual(strategy_obj.name, STRATEGY_NAME_STR)
        self.assertEqual(strategy_obj.symbol_tuple, STRATEGY_SYMBOL_TUPLE)
        self.assertEqual(len(transaction_df), 1)
        entry_row_ser = transaction_df.iloc[0]
        expected_target_share_float = (
            100_000.0
            * (DEFAULT_CONFIG.portfolio_leverage_float / len(STRATEGY_SYMBOL_TUPLE))
            / float(pricing_data_df.loc[date_index[signal_day_int], ("KIE", "Close")])
        )
        expected_fill_price_float = 111.0 * (1.0 + DEFAULT_CONFIG.slippage_float)

        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), date_index[fill_day_int])
        self.assertEqual(entry_row_ser["asset"], "KIE")
        self.assertAlmostEqual(float(entry_row_ser["price"]), expected_fill_price_float)
        self.assertAlmostEqual(float(entry_row_ser["amount"]), expected_target_share_float)

    def test_saved_metadata_points_to_kie_wrapper_module(self):
        pricing_data_df, date_index, _signal_day_int, _fill_day_int = (
            self.make_kie_signal_pricing_data_tuple()
        )

        with tempfile.TemporaryDirectory() as temp_dir_str:
            strategy_obj = run_variant(
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
                    / STRATEGY_NAME_STR
                    / "vanilla_backtest"
                ).glob("*/metadata.json")
            )
            self.assertEqual(len(metadata_path_list), 1)
            metadata_dict = json.loads(metadata_path_list[0].read_text(encoding="utf-8"))

        self.assertIsInstance(strategy_obj, SectorDispersionIbsKieStrategy)
        self.assertEqual(
            metadata_dict["class_module"],
            "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie",
        )
        self.assertEqual(metadata_dict["class_name"], "SectorDispersionIbsKieStrategy")


if __name__ == "__main__":
    unittest.main()
