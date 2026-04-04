import unittest

import numpy as np
import pandas as pd

from strategies.eom_tlt_vs_spy.strategy_eom_tlt_spy_upro_vix_variant import (
    DEFAULT_CONFIG,
    build_month_signal_df,
)


class EomTltSpyUproVixVariantTest(unittest.TestCase):
    def build_price_frame_pair(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        trading_index = pd.bdate_range("2024-01-01", "2024-03-31")
        open_price_df = pd.DataFrame(index=trading_index, dtype=float)
        close_price_df = pd.DataFrame(index=trading_index, dtype=float)

        open_price_df["SPY"] = 100.0
        open_price_df["TLT"] = 100.0
        open_price_df["^VIX"] = 20.0

        close_price_df["SPY"] = 100.0
        close_price_df["TLT"] = 100.0
        close_price_df["^VIX"] = 20.0

        january_index = trading_index[trading_index.to_period("M") == pd.Period("2024-01", freq="M")]
        february_index = trading_index[trading_index.to_period("M") == pd.Period("2024-02", freq="M")]

        january_signal_end_bar_ts = pd.Timestamp(january_index[DEFAULT_CONFIG.signal_day_count_int - 1])
        january_reversal_vix_bar_ts = pd.Timestamp(
            january_index[len(january_index) - DEFAULT_CONFIG.eom_hold_day_count_int - 1]
        )
        february_signal_end_bar_ts = pd.Timestamp(february_index[DEFAULT_CONFIG.signal_day_count_int - 1])
        february_month_end_bar_ts = pd.Timestamp(february_index[-1])

        close_price_df.loc[january_signal_end_bar_ts, "SPY"] = 95.0
        close_price_df.loc[january_signal_end_bar_ts, "TLT"] = 105.0
        close_price_df.loc[january_reversal_vix_bar_ts, "^VIX"] = 14.0

        close_price_df.loc[february_signal_end_bar_ts, "SPY"] = 110.0
        close_price_df.loc[february_signal_end_bar_ts, "TLT"] = 90.0
        close_price_df.loc[february_month_end_bar_ts, "^VIX"] = 16.0

        return open_price_df, close_price_df

    def test_month_signal_df_uses_prior_close_vix_for_asset_choice(self):
        open_price_df, close_price_df = self.build_price_frame_pair()

        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=DEFAULT_CONFIG,
        )

        january_signal_row_ser = month_signal_df.loc["2024-01"]
        february_signal_row_ser = month_signal_df.loc["2024-02"]

        self.assertEqual(january_signal_row_ser["reversal_asset_str"], "UPRO")
        self.assertAlmostEqual(float(january_signal_row_ser["reversal_vix_close_float"]), 14.0)
        self.assertEqual(
            pd.Timestamp(january_signal_row_ser["reversal_vix_decision_bar_ts"]),
            pd.Timestamp(january_signal_row_ser["reversal_entry_bar_ts"]) - pd.offsets.BDay(1),
        )

        self.assertEqual(february_signal_row_ser["pair_long_asset_str"], "SPY")
        self.assertAlmostEqual(float(february_signal_row_ser["pair_vix_close_float"]), 16.0)
        self.assertEqual(pd.Timestamp(february_signal_row_ser["pair_vix_decision_bar_ts"]).month, 2)
        self.assertEqual(pd.Timestamp(february_signal_row_ser["pair_entry_bar_ts"]).month, 3)

    def test_nan_vix_defaults_to_spy(self):
        open_price_df, close_price_df = self.build_price_frame_pair()
        january_index = open_price_df.index[open_price_df.index.to_period("M") == pd.Period("2024-01", freq="M")]
        january_reversal_vix_bar_ts = pd.Timestamp(
            january_index[len(january_index) - DEFAULT_CONFIG.eom_hold_day_count_int - 1]
        )
        close_price_df.loc[january_reversal_vix_bar_ts, "^VIX"] = np.nan

        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=DEFAULT_CONFIG,
        )

        january_signal_row_ser = month_signal_df.loc["2024-01"]
        self.assertEqual(january_signal_row_ser["reversal_asset_str"], "SPY")
        self.assertTrue(np.isnan(float(january_signal_row_ser["reversal_vix_close_float"])))


if __name__ == "__main__":
    unittest.main()
