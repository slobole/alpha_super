import math
import unittest
import warnings

import numpy as np
import pandas as pd

from strategies.eom_tlt_vs_spy.strategy_eom_zroz_spy_sso_l2_norm_variant import (
    EomZrozSpySsoL2NormVariantConfig,
    EomZrozSpySsoL2NormVariantResearchStrategy,
    build_daily_target_weight_df,
    build_month_signal_df,
    build_trade_leg_plan_df,
    run_variant_research_backtest,
)


class EomZrozSpySsoL2NormVariantTests(unittest.TestCase):
    @staticmethod
    def make_l2_index() -> pd.DatetimeIndex:
        prior_index = pd.bdate_range(end="2023-12-29", periods=64)
        signal_index = pd.bdate_range(start="2024-01-01", end="2024-02-08")
        return prior_index.append(signal_index)

    @staticmethod
    def make_prior_close_ser(
        trading_index: pd.DatetimeIndex,
        daily_abs_log_return_float: float,
        signal_close_float: float,
    ) -> pd.Series:
        close_value_float = 100.0
        direction_sign_float = 1.0
        close_value_list: list[float] = []
        for bar_ts in trading_index:
            if pd.Timestamp(bar_ts) <= pd.Timestamp("2023-12-29"):
                if len(close_value_list) > 0:
                    close_value_float *= math.exp(direction_sign_float * daily_abs_log_return_float)
                    direction_sign_float *= -1.0
                close_value_list.append(close_value_float)
            else:
                close_value_list.append(100.0)

        close_ser = pd.Series(close_value_list, index=trading_index, dtype=float)
        january_index = trading_index[trading_index.to_period("M") == pd.Period("2024-01")]
        signal_end_bar_ts = pd.Timestamp(january_index[14])
        close_ser.loc[signal_end_bar_ts] = float(signal_close_float)
        return close_ser

    def make_open_close_df(
        self,
        spy_daily_abs_log_return_float: float,
        zroz_daily_abs_log_return_float: float,
        spy_signal_close_float: float,
        zroz_signal_close_float: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        trading_index = self.make_l2_index()
        open_price_df = pd.DataFrame(
            {
                "SPY": pd.Series(100.0, index=trading_index, dtype=float),
                "ZROZ": pd.Series(100.0, index=trading_index, dtype=float),
                "SSO": pd.Series(100.0, index=trading_index, dtype=float),
            }
        )
        close_price_df = pd.DataFrame(
            {
                "SPY": self.make_prior_close_ser(
                    trading_index=trading_index,
                    daily_abs_log_return_float=spy_daily_abs_log_return_float,
                    signal_close_float=spy_signal_close_float,
                ),
                "ZROZ": self.make_prior_close_ser(
                    trading_index=trading_index,
                    daily_abs_log_return_float=zroz_daily_abs_log_return_float,
                    signal_close_float=zroz_signal_close_float,
                ),
                "SSO": pd.Series(100.0, index=trading_index, dtype=float),
            }
        )
        return open_price_df, close_price_df

    @staticmethod
    def compute_expected_l2_vol_float(
        close_ser: pd.Series,
        vol_window_day_count_int: int,
    ) -> float:
        daily_log_return_ser = np.log(close_ser / close_ser.shift(1))
        vol_window_log_return_ser = daily_log_return_ser.loc[: pd.Timestamp("2023-12-29")].tail(
            vol_window_day_count_int
        )
        return float(vol_window_log_return_ser.std(ddof=1))

    def make_strategy(
        self,
        trade_leg_plan_df: pd.DataFrame,
        daily_target_weight_df: pd.DataFrame,
        capital_base_float: float = 1_000.0,
    ) -> EomZrozSpySsoL2NormVariantResearchStrategy:
        return EomZrozSpySsoL2NormVariantResearchStrategy(
            name="EomZrozSpySsoL2NormVariantTest",
            benchmarks=[],
            tradeable_asset_list=["SSO", "ZROZ"],
            trade_leg_plan_df=trade_leg_plan_df,
            daily_target_weight_df=daily_target_weight_df,
            capital_base=capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_build_month_signal_df_uses_l2_normalized_log_returns_and_sets_zroz_pair(self):
        open_price_df, close_price_df = self.make_open_close_df(
            spy_daily_abs_log_return_float=0.010,
            zroz_daily_abs_log_return_float=0.005,
            spy_signal_close_float=116.0,
            zroz_signal_close_float=106.5,
        )

        config = EomZrozSpySsoL2NormVariantConfig()
        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=config,
        )
        trade_leg_plan_df = build_trade_leg_plan_df(
            month_signal_df=month_signal_df,
            config=config,
        )

        self.assertEqual(list(month_signal_df.index), ["2024-01"])
        signal_row_ser = month_signal_df.loc["2024-01"]
        expected_spy_l2_daily_vol_float = self.compute_expected_l2_vol_float(
            close_ser=close_price_df["SPY"],
            vol_window_day_count_int=config.vol_window_day_count_int,
        )
        expected_zroz_l2_daily_vol_float = self.compute_expected_l2_vol_float(
            close_ser=close_price_df["ZROZ"],
            vol_window_day_count_int=config.vol_window_day_count_int,
        )
        expected_spy_log_return_float = math.log(116.0 / 100.0)
        expected_zroz_log_return_float = math.log(106.5 / 100.0)
        expected_spy_score_float = expected_spy_log_return_float / (
            expected_spy_l2_daily_vol_float * math.sqrt(config.signal_day_count_int)
        )
        expected_zroz_score_float = expected_zroz_log_return_float / (
            expected_zroz_l2_daily_vol_float * math.sqrt(config.signal_day_count_int)
        )

        self.assertAlmostEqual(float(signal_row_ser["spy_first15_log_return_float"]), expected_spy_log_return_float)
        self.assertAlmostEqual(float(signal_row_ser["zroz_first15_log_return_float"]), expected_zroz_log_return_float)
        self.assertAlmostEqual(float(signal_row_ser["spy_l2_daily_vol_float"]), expected_spy_l2_daily_vol_float)
        self.assertAlmostEqual(float(signal_row_ser["zroz_l2_daily_vol_float"]), expected_zroz_l2_daily_vol_float)
        self.assertAlmostEqual(float(signal_row_ser["spy_l2_score_float"]), expected_spy_score_float)
        self.assertAlmostEqual(float(signal_row_ser["zroz_l2_score_float"]), expected_zroz_score_float)
        self.assertTrue(bool(signal_row_ser["spy_outperformed_bool"]))
        self.assertEqual(str(signal_row_ser["reversal_asset_str"]), "ZROZ")
        self.assertEqual(str(trade_leg_plan_df.iloc[1]["asset_str"]), "SSO")
        self.assertEqual(str(trade_leg_plan_df.iloc[2]["asset_str"]), "ZROZ")

    def test_l2_normalization_can_flip_raw_return_winner(self):
        open_price_df, close_price_df = self.make_open_close_df(
            spy_daily_abs_log_return_float=0.020,
            zroz_daily_abs_log_return_float=0.005,
            spy_signal_close_float=104.0,
            zroz_signal_close_float=103.5,
        )

        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=EomZrozSpySsoL2NormVariantConfig(),
        )

        signal_row_ser = month_signal_df.loc["2024-01"]
        self.assertGreater(
            float(signal_row_ser["spy_first15_log_return_float"]),
            float(signal_row_ser["zroz_first15_log_return_float"]),
        )
        self.assertGreater(
            float(signal_row_ser["zroz_l2_score_float"]),
            float(signal_row_ser["spy_l2_score_float"]),
        )
        self.assertTrue(bool(signal_row_ser["zroz_outperformed_bool"]))
        self.assertEqual(str(signal_row_ser["reversal_asset_str"]), "SSO")
        self.assertTrue(pd.isna(signal_row_ser["pair_entry_bar_ts"]))
        self.assertTrue(pd.isna(signal_row_ser["pair_exit_bar_ts"]))

    def test_month_to_date_signal_mode_uses_signal_window_returns_without_prior_history(self):
        trading_index = pd.bdate_range(start="2024-01-01", end="2024-02-08")
        open_price_df = pd.DataFrame(
            {
                "SPY": pd.Series(100.0, index=trading_index, dtype=float),
                "ZROZ": pd.Series(100.0, index=trading_index, dtype=float),
                "SSO": pd.Series(100.0, index=trading_index, dtype=float),
            }
        )
        close_price_df = open_price_df.copy()
        signal_end_bar_ts = pd.Timestamp(trading_index[14])
        close_price_df.loc[signal_end_bar_ts, "SPY"] = 116.0
        close_price_df.loc[signal_end_bar_ts, "ZROZ"] = 106.5

        config = EomZrozSpySsoL2NormVariantConfig(vol_window_mode_str="month_to_date_signal")
        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=config,
        )

        signal_row_ser = month_signal_df.loc["2024-01"]
        expected_spy_log_return_float = math.log(116.0 / 100.0)
        expected_zroz_log_return_float = math.log(106.5 / 100.0)
        expected_spy_log_return_ser = pd.Series([0.0] * 14 + [expected_spy_log_return_float], dtype=float)
        expected_zroz_log_return_ser = pd.Series([0.0] * 14 + [expected_zroz_log_return_float], dtype=float)

        self.assertEqual(str(signal_row_ser["vol_window_mode_str"]), "month_to_date_signal")
        self.assertAlmostEqual(float(signal_row_ser["spy_l2_daily_vol_float"]), float(expected_spy_log_return_ser.std(ddof=1)))
        self.assertAlmostEqual(float(signal_row_ser["zroz_l2_daily_vol_float"]), float(expected_zroz_log_return_ser.std(ddof=1)))

    def test_build_month_signal_df_skips_when_trailing_l2_window_is_missing(self):
        trading_index = pd.bdate_range(start="2024-01-01", end="2024-02-08")
        open_price_df = pd.DataFrame(
            {
                "SPY": pd.Series(100.0, index=trading_index, dtype=float),
                "ZROZ": pd.Series(100.0, index=trading_index, dtype=float),
                "SSO": pd.Series(100.0, index=trading_index, dtype=float),
            }
        )
        close_price_df = open_price_df.copy()

        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=EomZrozSpySsoL2NormVariantConfig(),
        )

        self.assertEqual(len(month_signal_df), 0)

    def test_build_daily_target_weight_df_and_backtest_cover_reversal_and_pair_legs(self):
        trading_index = pd.to_datetime(
            [
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
                "2024-02-02",
                "2024-02-05",
                "2024-02-06",
                "2024-02-07",
                "2024-02-08",
            ]
        )
        trade_leg_plan_df = pd.DataFrame(
            [
                {
                    "trade_id_int": 1,
                    "leg_type_str": "reversal",
                    "signal_month_period_str": "2024-01",
                    "asset_str": "ZROZ",
                    "signed_weight_float": 1.0,
                    "entry_bar_ts": pd.Timestamp("2024-01-30"),
                    "exit_bar_ts": pd.Timestamp("2024-01-31"),
                    "rel_l2_score_float": 0.05,
                },
                {
                    "trade_id_int": 2,
                    "leg_type_str": "pair_long_sso",
                    "signal_month_period_str": "2024-01",
                    "asset_str": "SSO",
                    "signed_weight_float": 0.5,
                    "entry_bar_ts": pd.Timestamp("2024-02-01"),
                    "exit_bar_ts": pd.Timestamp("2024-02-07"),
                    "rel_l2_score_float": 0.05,
                },
                {
                    "trade_id_int": 3,
                    "leg_type_str": "pair_short_zroz",
                    "signal_month_period_str": "2024-01",
                    "asset_str": "ZROZ",
                    "signed_weight_float": -0.5,
                    "entry_bar_ts": pd.Timestamp("2024-02-01"),
                    "exit_bar_ts": pd.Timestamp("2024-02-07"),
                    "rel_l2_score_float": 0.05,
                },
            ]
        ).set_index("trade_id_int", drop=True)
        daily_target_weight_df = build_daily_target_weight_df(
            trading_index=trading_index,
            trade_leg_plan_df=trade_leg_plan_df,
            asset_list=["SSO", "ZROZ"],
        )

        pricing_data_df = pd.DataFrame(
            {
                ("SSO", "Open"): pd.Series(100.0, index=trading_index, dtype=float),
                ("SSO", "High"): pd.Series([101.0, 101.0, 111.0, 121.0, 126.0, 131.0, 141.0, 141.0], index=trading_index, dtype=float),
                ("SSO", "Low"): pd.Series([99.0, 99.0, 109.0, 119.0, 124.0, 129.0, 139.0, 139.0], index=trading_index, dtype=float),
                ("SSO", "Close"): pd.Series([100.0, 100.0, 110.0, 120.0, 125.0, 130.0, 140.0, 140.0], index=trading_index, dtype=float),
                ("ZROZ", "Open"): pd.Series(100.0, index=trading_index, dtype=float),
                ("ZROZ", "High"): pd.Series([101.0, 111.0, 101.0, 99.0, 97.0, 95.0, 91.0, 91.0], index=trading_index, dtype=float),
                ("ZROZ", "Low"): pd.Series([99.0, 109.0, 99.0, 97.0, 95.0, 93.0, 89.0, 89.0], index=trading_index, dtype=float),
                ("ZROZ", "Close"): pd.Series([100.0, 110.0, 100.0, 98.0, 96.0, 94.0, 90.0, 90.0], index=trading_index, dtype=float),
            },
            index=trading_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

        strategy = self.make_strategy(
            trade_leg_plan_df=trade_leg_plan_df,
            daily_target_weight_df=daily_target_weight_df,
            capital_base_float=1_000.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            run_variant_research_backtest(
                strategy=strategy,
                pricing_data_df=pricing_data_df,
            )

        self.assertEqual(len(strategy._transactions), 6)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-01-31"), "total_value"]), 1_100.0)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-02-07"), "total_value"]), 1_350.0)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-02-08"), "total_value"]), 1_350.0)
        self.assertAlmostEqual(float(strategy.summary.loc["Final [$]", "Strategy"]), 1_350.0)
        self.assertAlmostEqual(float(strategy.summary.loc["Return [%]", "Strategy"]), 35.0)


if __name__ == "__main__":
    unittest.main()
