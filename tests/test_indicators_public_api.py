import unittest

import numpy as np
import pandas as pd

from alpha import indicators as public_indicators
from alpha.engine.strategy import Strategy


class IndicatorImportSmokeStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        close_price_ser = signal_data[("TEST", "Close")]
        turnover_dollar_ser = signal_data[("TEST", "Turnover")]
        high_price_ser = signal_data[("TEST", "High")]
        low_price_ser = signal_data[("TEST", "Low")]

        signal_data[("TEST", "adv_2")] = public_indicators.adv_dollar_indicator(
            turnover_dollar_ser,
            window_int=2,
        )
        signal_data[("TEST", "ibs")] = public_indicators.ibs_indicator(
            close_price_ser,
            high_price_ser,
            low_price_ser,
        )
        return signal_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


class PublicIndicatorApiTests(unittest.TestCase):
    def test_public_exports_match_expected_order(self):
        expected_export_list = [
            "dv2_indicator",
            "qp_indicator",
            "adv_dollar_indicator",
            "ibs_indicator",
            "dv2_indicator_fast",
            "qp_indicator_fast",
            "dv2_indicator_reference",
            "qp_indicator_reference",
        ]
        self.assertEqual(public_indicators.__all__, expected_export_list)

    def test_adv_dollar_indicator_preserves_warmup_and_formula(self):
        date_index = pd.date_range("2024-01-01", periods=4, freq="D")
        turnover_dollar_ser = pd.Series([10.0, 20.0, 30.0, 50.0], index=date_index, dtype=float)

        adv_value_ser = public_indicators.adv_dollar_indicator(turnover_dollar_ser, window_int=2)
        expected_value_ser = pd.Series([np.nan, 15.0, 25.0, 40.0], index=date_index, dtype=float)

        pd.testing.assert_series_equal(
            adv_value_ser,
            expected_value_ser,
            check_names=False,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )

    def test_adv_dollar_indicator_supports_dataframe_inputs(self):
        date_index = pd.date_range("2024-01-01", periods=4, freq="D")
        turnover_dollar_df = pd.DataFrame(
            {
                "AAA": [10.0, 20.0, 30.0, 50.0],
                "BBB": [12.0, 18.0, 30.0, 42.0],
            },
            index=date_index,
            dtype=float,
        )

        adv_value_df = public_indicators.adv_dollar_indicator(turnover_dollar_df, window_int=2)
        expected_value_df = pd.DataFrame(
            {
                "AAA": [np.nan, 15.0, 25.0, 40.0],
                "BBB": [np.nan, 15.0, 24.0, 36.0],
            },
            index=date_index,
            dtype=float,
        )

        pd.testing.assert_frame_equal(
            adv_value_df,
            expected_value_df,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )

    def test_ibs_indicator_returns_nan_for_zero_range(self):
        date_index = pd.date_range("2024-01-01", periods=3, freq="D")
        close_price_ser = pd.Series([9.0, 11.5, 10.0], index=date_index, dtype=float)
        high_price_ser = pd.Series([10.0, 12.0, 10.0], index=date_index, dtype=float)
        low_price_ser = pd.Series([8.0, 11.0, 10.0], index=date_index, dtype=float)

        ibs_value_ser = public_indicators.ibs_indicator(
            close_price_ser,
            high_price_ser,
            low_price_ser,
        )
        expected_value_ser = pd.Series([0.5, 0.5, np.nan], index=date_index, dtype=float)

        pd.testing.assert_series_equal(
            ibs_value_ser,
            expected_value_ser,
            check_names=False,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )

    def test_ibs_indicator_supports_dataframe_inputs(self):
        date_index = pd.date_range("2024-01-01", periods=3, freq="D")
        close_price_df = pd.DataFrame(
            {
                "AAA": [9.0, 11.5, 10.0],
                "BBB": [19.0, 21.5, 22.0],
            },
            index=date_index,
            dtype=float,
        )
        high_price_df = pd.DataFrame(
            {
                "AAA": [10.0, 12.0, 10.0],
                "BBB": [20.0, 22.0, 22.0],
            },
            index=date_index,
            dtype=float,
        )
        low_price_df = pd.DataFrame(
            {
                "AAA": [8.0, 11.0, 10.0],
                "BBB": [18.0, 21.0, 22.0],
            },
            index=date_index,
            dtype=float,
        )

        ibs_value_df = public_indicators.ibs_indicator(
            close_price_df,
            high_price_df,
            low_price_df,
        )
        expected_value_df = pd.DataFrame(
            {
                "AAA": [0.5, 0.5, np.nan],
                "BBB": [0.5, 0.5, np.nan],
            },
            index=date_index,
            dtype=float,
        )

        pd.testing.assert_frame_equal(
            ibs_value_df,
            expected_value_df,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )

    def test_research_style_import_executes(self):
        namespace_dict: dict[str, object] = {}
        exec(
            (
                "from alpha.indicators import "
                "dv2_indicator, qp_indicator, adv_dollar_indicator, ibs_indicator, "
                "dv2_indicator_fast, qp_indicator_fast, dv2_indicator_reference, qp_indicator_reference"
            ),
            namespace_dict,
        )

        for indicator_name_str in [
            "dv2_indicator",
            "qp_indicator",
            "adv_dollar_indicator",
            "ibs_indicator",
            "dv2_indicator_fast",
            "qp_indicator_fast",
            "dv2_indicator_reference",
            "qp_indicator_reference",
        ]:
            self.assertIn(indicator_name_str, namespace_dict)
            self.assertTrue(callable(namespace_dict[indicator_name_str]))

    def test_strategy_style_import_executes(self):
        date_index = pd.date_range("2024-01-01", periods=5, freq="D")
        pricing_data = pd.DataFrame(
            {
                ("TEST", "Open"): [10.0, 10.2, 10.4, 10.6, 10.8],
                ("TEST", "High"): [10.5, 10.7, 10.9, 11.1, 11.3],
                ("TEST", "Low"): [9.8, 10.0, 10.2, 10.4, 10.6],
                ("TEST", "Close"): [10.1, 10.3, 10.5, 10.7, 10.9],
                ("TEST", "Turnover"): [1_000.0, 1_200.0, 1_400.0, 1_600.0, 1_800.0],
            },
            index=date_index,
        )
        pricing_data.columns = pd.MultiIndex.from_tuples(pricing_data.columns)

        strategy = IndicatorImportSmokeStrategy(
            name="IndicatorImportSmoke",
            benchmarks=[],
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        signal_data = strategy.compute_signals(pricing_data)

        self.assertIn(("TEST", "adv_2"), signal_data.columns)
        self.assertIn(("TEST", "ibs"), signal_data.columns)


if __name__ == "__main__":
    unittest.main()
