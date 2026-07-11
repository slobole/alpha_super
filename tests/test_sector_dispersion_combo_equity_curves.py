import unittest

import pandas as pd

from scripts.research.plot_sector_dispersion_combo_equity_curves import (
    DEFAULT_COMPARISON_VARIANT_SPEC_TUPLE,
    FULL_UNIVERSE_COMPARISON_VARIANT_SPEC_TUPLE,
    build_comparison_manifest_df,
    build_variant_spec_tuple,
    compute_drawdown_curve_df,
    compute_market_correlation_summary_df,
    normalize_equity_curve_df,
)


class SectorDispersionComboEquityCurveTests(unittest.TestCase):
    def test_default_manifest_contains_selected_finalists(self):
        manifest_df = build_comparison_manifest_df()

        self.assertEqual(len(manifest_df), len(DEFAULT_COMPARISON_VARIANT_SPEC_TUPLE))
        self.assertEqual(manifest_df["variant_label_str"].iloc[0], "Base")
        self.assertIn("Base+KIE", set(manifest_df["variant_label_str"]))
        self.assertIn("Base+KIE+IHI", set(manifest_df["variant_label_str"]))
        self.assertIn("Base+KIE+IHI+XLC", set(manifest_df["variant_label_str"]))
        self.assertIn("Base+KIE+XLRE+IHI", set(manifest_df["variant_label_str"]))
        self.assertIn("Base+KIE+XLRE", set(manifest_df["variant_label_str"]))
        self.assertIn("Base+XLRE", set(manifest_df["variant_label_str"]))
        self.assertEqual(len(manifest_df["variant_label_str"]), len(set(manifest_df["variant_label_str"])))
        self.assertEqual(len(manifest_df["addition_tuple_str"]), len(set(manifest_df["addition_tuple_str"])))

    def test_full_universe_manifest_is_optional(self):
        variant_spec_tuple = build_variant_spec_tuple(include_full_universes_bool=True)
        manifest_df = build_comparison_manifest_df(variant_spec_tuple=variant_spec_tuple)

        self.assertEqual(
            len(manifest_df),
            len(DEFAULT_COMPARISON_VARIANT_SPEC_TUPLE) + len(FULL_UNIVERSE_COMPARISON_VARIANT_SPEC_TUPLE),
        )
        self.assertIn("Full Universe A", set(manifest_df["variant_label_str"]))
        self.assertIn("Full Universe B", set(manifest_df["variant_label_str"]))
        self.assertIn("Full Universe C", set(manifest_df["variant_label_str"]))
        self.assertGreater(
            int(manifest_df.loc[manifest_df["variant_label_str"].eq("Full Universe C"), "addition_count_int"].iloc[0]),
            int(manifest_df.loc[manifest_df["variant_label_str"].eq("Full Universe A"), "addition_count_int"].iloc[0]),
        )

    def test_normalize_equity_curve_df_scales_each_column_from_own_start(self):
        date_index = pd.bdate_range("2024-01-02", periods=3)
        equity_curve_df = pd.DataFrame(
            {
                "A": [100.0, 110.0, 121.0],
                "B": [200.0, 180.0, 220.0],
            },
            index=date_index,
            dtype=float,
        )

        normalized_equity_curve_df = normalize_equity_curve_df(equity_curve_df)

        self.assertAlmostEqual(float(normalized_equity_curve_df.loc[date_index[0], "A"]), 1.0)
        self.assertAlmostEqual(float(normalized_equity_curve_df.loc[date_index[-1], "A"]), 1.21)
        self.assertAlmostEqual(float(normalized_equity_curve_df.loc[date_index[-1], "B"]), 1.10)

    def test_compute_drawdown_curve_df_uses_column_running_peak(self):
        date_index = pd.bdate_range("2024-01-02", periods=4)
        normalized_equity_curve_df = pd.DataFrame(
            {
                "A": [1.0, 1.2, 0.9, 1.3],
                "B": [1.0, 0.8, 0.7, 0.9],
            },
            index=date_index,
            dtype=float,
        )

        drawdown_curve_df = compute_drawdown_curve_df(normalized_equity_curve_df)

        self.assertAlmostEqual(float(drawdown_curve_df.loc[date_index[0], "A"]), 0.0)
        self.assertAlmostEqual(float(drawdown_curve_df.loc[date_index[2], "A"]), -0.25)
        self.assertAlmostEqual(float(drawdown_curve_df.loc[date_index[2], "B"]), -0.30)
        self.assertAlmostEqual(float(drawdown_curve_df.loc[date_index[3], "B"]), -0.10)

    def test_compute_market_correlation_summary_df_measures_down_market_correlation_and_beta(self):
        date_index = pd.bdate_range("2024-01-02", periods=6)
        equity_curve_df = pd.DataFrame(
            {
                "Variant": [100.0, 80.0, 88.0, 52.8, 63.36, 57.024],
                "Benchmark $SPX": [100.0, 90.0, 94.5, 75.6, 83.16, 79.002],
            },
            index=date_index,
            dtype=float,
        )

        market_correlation_summary_df = compute_market_correlation_summary_df(
            equity_curve_df=equity_curve_df,
            benchmark_label_str="Benchmark $SPX",
            market_tail_quantile_float=0.60,
        )
        row_ser = market_correlation_summary_df.set_index("variant_label_str").loc["Variant"]

        self.assertEqual(int(row_ser["market_down_day_count_int"]), 3)
        self.assertEqual(int(row_ser["market_tail_day_count_int"]), 3)
        self.assertAlmostEqual(float(row_ser["corr_to_spx_float"]), 1.0)
        self.assertAlmostEqual(float(row_ser["market_down_corr_to_spx_float"]), 1.0)
        self.assertAlmostEqual(float(row_ser["market_down_beta_to_spx_float"]), 2.0)
        self.assertAlmostEqual(float(row_ser["market_tail_corr_to_spx_float"]), 1.0)
        self.assertAlmostEqual(float(row_ser["market_tail_beta_to_spx_float"]), 2.0)


if __name__ == "__main__":
    unittest.main()
