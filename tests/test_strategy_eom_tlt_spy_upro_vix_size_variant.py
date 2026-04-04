import unittest

import numpy as np
import pandas as pd

from strategies.eom_tlt_vs_spy.strategy_eom_tlt_spy_upro_vix_size_variant import (
    DEFAULT_CONFIG,
    build_month_signal_df,
    build_trade_leg_plan_df,
    compute_long_weight_scale_float,
)


class EomTltSpyUproVixSizeVariantTests(unittest.TestCase):
    @staticmethod
    def make_open_close_df(
        trading_index: pd.DatetimeIndex,
        spy_close_list: list[float],
        tlt_close_list: list[float],
        vix_close_list: list[float],
        open_price_float: float = 100.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        open_price_df = pd.DataFrame(
            {
                "SPY": pd.Series(open_price_float, index=trading_index, dtype=float),
                "TLT": pd.Series(open_price_float, index=trading_index, dtype=float),
                "^VIX": pd.Series(20.0, index=trading_index, dtype=float),
            }
        )
        close_price_df = pd.DataFrame(
            {
                "SPY": pd.Series(spy_close_list, index=trading_index, dtype=float),
                "TLT": pd.Series(tlt_close_list, index=trading_index, dtype=float),
                "^VIX": pd.Series(vix_close_list, index=trading_index, dtype=float),
            }
        )
        return open_price_df, close_price_df

    def test_compute_long_weight_scale_float_clips_inverse_vix_rule(self):
        self.assertAlmostEqual(compute_long_weight_scale_float(10.0, DEFAULT_CONFIG), 1.5)
        self.assertAlmostEqual(compute_long_weight_scale_float(20.0, DEFAULT_CONFIG), 1.0)
        self.assertAlmostEqual(compute_long_weight_scale_float(40.0, DEFAULT_CONFIG), 0.5)
        self.assertAlmostEqual(compute_long_weight_scale_float(np.nan, DEFAULT_CONFIG), 1.0)

    def test_reversal_upro_weight_uses_prior_close_vix(self):
        trading_index = pd.bdate_range("2024-01-01", "2024-02-29")
        january_index = trading_index[trading_index.to_period("M") == pd.Period("2024-01", freq="M")]

        spy_close_list = [100.0] * len(trading_index)
        tlt_close_list = [100.0] * len(trading_index)
        vix_close_list = [20.0] * len(trading_index)

        january_signal_end_pos_int = DEFAULT_CONFIG.signal_day_count_int - 1
        january_signal_end_bar_ts = pd.Timestamp(january_index[january_signal_end_pos_int])
        january_reversal_vix_bar_ts = pd.Timestamp(
            january_index[len(january_index) - DEFAULT_CONFIG.eom_hold_day_count_int - 1]
        )

        spy_close_list[trading_index.get_loc(january_signal_end_bar_ts)] = 95.0
        tlt_close_list[trading_index.get_loc(january_signal_end_bar_ts)] = 105.0
        vix_close_list[trading_index.get_loc(january_reversal_vix_bar_ts)] = 10.0

        open_price_df, close_price_df = self.make_open_close_df(
            trading_index=trading_index,
            spy_close_list=spy_close_list,
            tlt_close_list=tlt_close_list,
            vix_close_list=vix_close_list,
        )

        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=DEFAULT_CONFIG,
        )

        january_signal_row_ser = month_signal_df.loc["2024-01"]
        self.assertEqual(january_signal_row_ser["reversal_asset_str"], "UPRO")
        self.assertAlmostEqual(float(january_signal_row_ser["reversal_vix_close_float"]), 10.0)
        self.assertAlmostEqual(float(january_signal_row_ser["reversal_long_weight_scale_float"]), 1.5)
        self.assertAlmostEqual(float(january_signal_row_ser["reversal_signed_weight_float"]), 1.5)

    def test_pair_upro_weight_scales_only_upro_leg(self):
        trading_index = pd.bdate_range("2024-01-01", "2024-03-31")
        january_index = trading_index[trading_index.to_period("M") == pd.Period("2024-01", freq="M")]

        spy_close_list = [100.0] * len(trading_index)
        tlt_close_list = [100.0] * len(trading_index)
        vix_close_list = [20.0] * len(trading_index)

        january_signal_end_pos_int = DEFAULT_CONFIG.signal_day_count_int - 1
        january_signal_end_bar_ts = pd.Timestamp(january_index[january_signal_end_pos_int])
        january_month_end_bar_ts = pd.Timestamp(january_index[-1])

        spy_close_list[trading_index.get_loc(january_signal_end_bar_ts)] = 110.0
        tlt_close_list[trading_index.get_loc(january_signal_end_bar_ts)] = 90.0
        vix_close_list[trading_index.get_loc(january_month_end_bar_ts)] = 40.0

        open_price_df, close_price_df = self.make_open_close_df(
            trading_index=trading_index,
            spy_close_list=spy_close_list,
            tlt_close_list=tlt_close_list,
            vix_close_list=vix_close_list,
        )

        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=DEFAULT_CONFIG,
        )
        trade_leg_plan_df = build_trade_leg_plan_df(
            month_signal_df=month_signal_df,
            config=DEFAULT_CONFIG,
        )

        january_signal_row_ser = month_signal_df.loc["2024-01"]
        self.assertEqual(january_signal_row_ser["pair_long_asset_str"], "UPRO")
        self.assertAlmostEqual(float(january_signal_row_ser["pair_long_weight_scale_float"]), 0.5)
        self.assertAlmostEqual(float(january_signal_row_ser["pair_long_signed_weight_float"]), 0.25)

        pair_long_upro_row_ser = trade_leg_plan_df[trade_leg_plan_df["leg_type_str"] == "pair_long_upro"].iloc[0]
        pair_short_tlt_row_ser = trade_leg_plan_df[trade_leg_plan_df["leg_type_str"] == "pair_short_tlt"].iloc[0]
        self.assertEqual(str(pair_long_upro_row_ser["asset_str"]), "UPRO")
        self.assertAlmostEqual(float(pair_long_upro_row_ser["signed_weight_float"]), 0.25)
        self.assertAlmostEqual(float(pair_short_tlt_row_ser["signed_weight_float"]), -0.5)


if __name__ == "__main__":
    unittest.main()
