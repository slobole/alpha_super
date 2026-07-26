from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.research.run_sector_dispersion_family_universe_study import (
    STUDY_END_DATE_STR,
    SectorDispersionDividendStrategy,
    build_execution_calendar_idx,
    build_exposure_diagnostic_dict,
    build_universe_config_obj,
    build_universe_manifest_df,
    compute_equal_weight_benchmark_return_ser,
    prepare_isolated_no_print_sessions,
)


class SectorDispersionFamilyUniverseStudyTests(unittest.TestCase):
    def test_universe_manifest_is_frozen(self):
        manifest_df = build_universe_manifest_df()

        self.assertEqual(
            manifest_df["universe_id_str"].tolist(),
            [
                "spdr_11",
                "spdr_9",
                "vanguard_11",
                "spdr_proxy_11",
                "ishares_us_11",
                "ishares_global_11",
            ],
        )
        self.assertEqual(manifest_df["priority_int"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(manifest_df["symbol_count_int"].tolist(), [11, 9, 11, 11, 11, 11])
        self.assertEqual(
            manifest_df.loc[
                manifest_df["universe_id_str"].eq("spdr_proxy_11"),
                "symbol_tuple_str",
            ].iloc[0],
            "XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,VOX,IYR",
        )
        self.assertEqual(STUDY_END_DATE_STR, "2026-07-17")

        for symbol_tuple_str in manifest_df["symbol_tuple_str"]:
            symbol_tuple = tuple(symbol_tuple_str.split(","))
            self.assertEqual(len(symbol_tuple), len(set(symbol_tuple)))

    def test_universe_config_and_strategy_use_unlevered_one_over_n_sizing(self):
        manifest_row_ser = build_universe_manifest_df().iloc[0]
        config_obj = build_universe_config_obj(manifest_row_ser=manifest_row_ser)
        strategy_obj = SectorDispersionDividendStrategy(
            name="test_unlevered",
            benchmarks=[config_obj.benchmark_symbol_str],
            config_obj=config_obj,
        )

        self.assertEqual(config_obj.portfolio_leverage_float, 1.0)
        self.assertEqual(len(config_obj.symbol_tuple), 11)
        self.assertAlmostEqual(strategy_obj.target_weight_float, 1.0 / 11.0)

    def test_full_basket_calendar_waits_for_every_symbol_history(self):
        date_index = pd.bdate_range("2024-01-02", periods=30)
        pricing_data_df = pd.DataFrame(index=date_index)
        for symbol_str in ("AAA", "BBB"):
            pricing_data_df[(symbol_str, "Open")] = 100.0
            pricing_data_df[(symbol_str, "High")] = 101.0
            pricing_data_df[(symbol_str, "Low")] = 99.0
            pricing_data_df[(symbol_str, "Close")] = 100.0
            pricing_data_df[(symbol_str, "Dividend")] = 0.0
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        missing_column_list = [
            ("BBB", field_str)
            for field_str in ("Open", "High", "Low", "Close")
        ]
        pricing_data_df.loc[date_index[:3], missing_column_list] = np.nan

        manifest_row_ser = pd.Series(
            {
                "universe_id_str": "synthetic",
                "symbol_tuple_str": "AAA,BBB",
                "raw_common_start_date_str": date_index[0].date().isoformat(),
            }
        )
        config_obj = build_universe_config_obj(
            manifest_row_ser=manifest_row_ser,
            end_date_str=date_index[-1].date().isoformat(),
        )

        calendar_idx = build_execution_calendar_idx(
            pricing_data_df=pricing_data_df,
            config_obj=config_obj,
        )

        self.assertEqual(calendar_idx[0], date_index[24])

    def test_dividend_credit_uses_preopen_positions(self):
        current_bar_ts = pd.Timestamp("2024-01-03")
        manifest_row_ser = pd.Series(
            {
                "universe_id_str": "synthetic",
                "symbol_tuple_str": "AAA,BBB",
                "raw_common_start_date_str": "2024-01-01",
            }
        )
        config_obj = build_universe_config_obj(
            manifest_row_ser=manifest_row_ser,
            end_date_str=current_bar_ts.date().isoformat(),
        )
        strategy_obj = SectorDispersionDividendStrategy(
            name="test_dividends",
            benchmarks=[],
            config_obj=config_obj,
        )
        strategy_obj.cash = 0.0
        strategy_obj.current_bar = current_bar_ts
        strategy_obj._position_amount_map = {"AAA": 10.0, "BBB": 5.0}

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): [100.0],
                ("AAA", "High"): [101.0],
                ("AAA", "Low"): [99.0],
                ("AAA", "Close"): [100.0],
                ("AAA", "Dividend"): [1.0],
                ("BBB", "Open"): [50.0],
                ("BBB", "High"): [51.0],
                ("BBB", "Low"): [49.0],
                ("BBB", "Close"): [50.0],
                ("BBB", "Dividend"): [2.0],
            },
            index=[current_bar_ts],
        )

        strategy_obj.process_orders(pricing_data_df)

        self.assertAlmostEqual(strategy_obj.cash, 20.0)
        self.assertAlmostEqual(strategy_obj.dividend_cash_total_float, 20.0)
        self.assertEqual(len(strategy_obj.dividend_credit_df), 2)
        self.assertEqual(len(strategy_obj.get_dividend_ledger()), 0)
        self.assertEqual(
            strategy_obj._accounting_policy_dict["dividend_data_status_str"],
            "disabled_explicitly",
        )

    def test_manual_dividend_strategy_does_not_double_credit_on_next_bar(self):
        date_index = pd.bdate_range("2024-01-02", periods=2)
        manifest_row_ser = pd.Series(
            {
                "universe_id_str": "synthetic",
                "symbol_tuple_str": "AAA",
                "raw_common_start_date_str": date_index[0].date().isoformat(),
            }
        )
        config_obj = build_universe_config_obj(
            manifest_row_ser=manifest_row_ser,
            end_date_str=date_index[-1].date().isoformat(),
        )
        strategy_obj = SectorDispersionDividendStrategy(
            name="test_no_double_credit",
            benchmarks=[],
            config_obj=config_obj,
        )
        strategy_obj.cash = 0.0
        strategy_obj._position_amount_map = {"AAA": 10.0}
        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): [100.0, 99.0],
                ("AAA", "High"): [100.0, 99.0],
                ("AAA", "Low"): [100.0, 99.0],
                ("AAA", "Close"): [100.0, 99.0],
                ("AAA", "Dividend"): [1.0, 0.0],
            },
            index=date_index,
        )

        strategy_obj.current_bar = date_index[0]
        strategy_obj.process_orders(pricing_data_df)
        strategy_obj.previous_bar = date_index[0]
        strategy_obj.current_bar = date_index[1]
        strategy_obj.process_orders(pricing_data_df)

        self.assertAlmostEqual(strategy_obj.cash, 10.0)
        self.assertAlmostEqual(strategy_obj.dividend_cash_total_float, 10.0)
        self.assertEqual(len(strategy_obj.dividend_credit_df), 1)
        self.assertEqual(len(strategy_obj.get_dividend_ledger()), 0)

    def test_isolated_no_print_session_is_stale_valued_and_order_is_canceled(self):
        date_index = pd.bdate_range("2024-01-02", periods=3)
        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): [100.0, np.nan, 102.0],
                ("AAA", "High"): [101.0, np.nan, 103.0],
                ("AAA", "Low"): [99.0, np.nan, 101.0],
                ("AAA", "Close"): [100.0, np.nan, 102.0],
                ("AAA", "Dividend"): [0.0, np.nan, 0.0],
            },
            index=date_index,
        )
        manifest_df = pd.DataFrame(
            [
                {
                    "universe_id_str": "synthetic",
                    "symbol_tuple_str": "AAA",
                }
            ]
        )

        prepared_data_df, stale_session_df = prepare_isolated_no_print_sessions(
            pricing_data_df=pricing_data_df,
            universe_manifest_df=manifest_df,
        )

        self.assertEqual(len(stale_session_df), 1)
        self.assertEqual(stale_session_df["symbol_str"].iloc[0], "AAA")
        for field_str in ("Open", "High", "Low", "Close"):
            self.assertAlmostEqual(
                float(prepared_data_df.loc[date_index[1], ("AAA", field_str)]),
                100.0,
            )
        self.assertTrue(
            bool(prepared_data_df.loc[date_index[1], ("AAA", "stale_no_print_bool")])
        )

        manifest_row_ser = pd.Series(
            {
                "universe_id_str": "synthetic",
                "symbol_tuple_str": "AAA",
                "raw_common_start_date_str": date_index[0].date().isoformat(),
            }
        )
        config_obj = build_universe_config_obj(
            manifest_row_ser=manifest_row_ser,
            end_date_str=date_index[-1].date().isoformat(),
        )
        strategy_obj = SectorDispersionDividendStrategy(
            name="test_stale_cancel",
            benchmarks=[],
            config_obj=config_obj,
        )
        strategy_obj.current_bar = date_index[1]
        strategy_obj.order_target("AAA", 10.0)

        strategy_obj.process_orders(prepared_data_df)

        self.assertEqual(len(strategy_obj.get_transactions()), 0)
        self.assertEqual(strategy_obj.stale_order_cancellation_count_int, 1)

    def test_equal_weight_benchmark_includes_cash_dividends(self):
        date_index = pd.bdate_range("2024-01-02", periods=2)
        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Close"): [100.0, 99.0],
                ("AAA", "Dividend"): [0.0, 2.0],
                ("BBB", "Close"): [100.0, 100.0],
                ("BBB", "Dividend"): [0.0, 0.0],
            },
            index=date_index,
        )

        benchmark_return_ser = compute_equal_weight_benchmark_return_ser(
            pricing_data_df=pricing_data_df,
            symbol_tuple=("AAA", "BBB"),
            calendar_idx=date_index,
        )

        self.assertAlmostEqual(float(benchmark_return_ser.iloc[0]), 0.0)
        self.assertAlmostEqual(float(benchmark_return_ser.iloc[1]), 0.005)

    def test_exposure_diagnostics_report_gap_drift_and_negative_cash(self):
        date_index = pd.bdate_range("2024-01-02", periods=3)
        realized_weight_df = pd.DataFrame(
            {
                "AAA": [0.50, 0.60, 0.0],
                "BBB": [0.40, 0.45, 0.0],
                "Cash": [0.10, -0.05, 1.0],
            },
            index=date_index,
        )
        result_df = pd.DataFrame(
            {
                "cash": [10_000.0, -5_000.0, 100_000.0],
                "total_value": [100_000.0, 100_000.0, 100_000.0],
            },
            index=date_index,
        )

        diagnostic_dict = build_exposure_diagnostic_dict(
            realized_weight_df=realized_weight_df,
            result_df=result_df,
            symbol_tuple=("AAA", "BBB"),
        )

        self.assertAlmostEqual(diagnostic_dict["max_gross_exposure_pct_float"], 105.0)
        self.assertAlmostEqual(diagnostic_dict["minimum_cash_float"], -5_000.0)
        self.assertEqual(diagnostic_dict["negative_cash_day_count_int"], 1)
        self.assertAlmostEqual(diagnostic_dict["average_active_position_count_float"], 4.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
