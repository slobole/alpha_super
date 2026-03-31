import unittest
import warnings

import pandas as pd

from strategies.strategy_eom_tlt_spy_sso_variant import (
    EomTltSpySsoVariantConfig,
    EomTltSpySsoVariantResearchStrategy,
    build_daily_target_weight_df,
    build_month_signal_df,
    build_trade_leg_plan_df,
    run_variant_research_backtest,
)


class EomTltSpySsoVariantTests(unittest.TestCase):
    @staticmethod
    def make_index(date_str_list: list[str]) -> pd.DatetimeIndex:
        return pd.to_datetime(date_str_list)

    @staticmethod
    def make_open_close_df(
        trading_index: pd.DatetimeIndex,
        spy_close_list: list[float],
        tlt_close_list: list[float],
        sso_close_list: list[float],
        open_price_float: float = 100.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        open_price_df = pd.DataFrame(
            {
                "SPY": pd.Series(open_price_float, index=trading_index, dtype=float),
                "TLT": pd.Series(open_price_float, index=trading_index, dtype=float),
                "SSO": pd.Series(open_price_float, index=trading_index, dtype=float),
            }
        )
        close_price_df = pd.DataFrame(
            {
                "SPY": pd.Series(spy_close_list, index=trading_index, dtype=float),
                "TLT": pd.Series(tlt_close_list, index=trading_index, dtype=float),
                "SSO": pd.Series(sso_close_list, index=trading_index, dtype=float),
            }
        )
        return open_price_df, close_price_df

    def make_strategy(
        self,
        trade_leg_plan_df: pd.DataFrame,
        daily_target_weight_df: pd.DataFrame,
        capital_base_float: float = 1_000.0,
    ) -> EomTltSpySsoVariantResearchStrategy:
        return EomTltSpySsoVariantResearchStrategy(
            name="EomTltSpySsoVariantTest",
            benchmarks=[],
            tradeable_asset_list=["SSO", "TLT"],
            trade_leg_plan_df=trade_leg_plan_df,
            daily_target_weight_df=daily_target_weight_df,
            capital_base=capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_build_month_signal_df_sets_tlt_reversal_and_sso_pair_when_spy_outperforms(self):
        trading_index = self.make_index(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
                "2024-01-12",
                "2024-01-16",
                "2024-01-17",
                "2024-01-18",
                "2024-01-19",
                "2024-01-22",
                "2024-01-23",
                "2024-01-24",
                "2024-01-25",
                "2024-01-26",
                "2024-01-29",
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
        spy_close_list = list(range(101, 123)) + [100.0, 100.0, 100.0, 100.0, 100.0]
        tlt_close_list = [100.0 + 0.5 * idx_int for idx_int in range(1, 23)] + [100.0, 100.0, 100.0, 100.0, 100.0]
        sso_close_list = list(range(102, 124)) + [102.0, 103.0, 104.0, 105.0, 106.0]
        open_price_df, close_price_df = self.make_open_close_df(
            trading_index=trading_index,
            spy_close_list=spy_close_list,
            tlt_close_list=tlt_close_list,
            sso_close_list=sso_close_list,
        )

        config = EomTltSpySsoVariantConfig()
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
        self.assertAlmostEqual(float(signal_row_ser["spy_first15_return_float"]), 0.15)
        self.assertAlmostEqual(float(signal_row_ser["tlt_first15_return_float"]), 0.075)
        self.assertAlmostEqual(float(signal_row_ser["rel_15_return_float"]), 0.075)
        self.assertTrue(bool(signal_row_ser["spy_outperformed_bool"]))
        self.assertEqual(str(signal_row_ser["reversal_asset_str"]), "TLT")
        self.assertEqual(pd.Timestamp(signal_row_ser["pair_entry_bar_ts"]), pd.Timestamp("2024-02-01"))
        self.assertEqual(pd.Timestamp(signal_row_ser["pair_exit_bar_ts"]), pd.Timestamp("2024-02-07"))
        self.assertEqual(len(trade_leg_plan_df), 3)
        self.assertEqual(str(trade_leg_plan_df.iloc[1]["asset_str"]), "SSO")
        self.assertAlmostEqual(float(trade_leg_plan_df.iloc[1]["signed_weight_float"]), 0.5)
        self.assertEqual(str(trade_leg_plan_df.iloc[2]["asset_str"]), "TLT")
        self.assertAlmostEqual(float(trade_leg_plan_df.iloc[2]["signed_weight_float"]), -0.5)

    def test_build_month_signal_df_sets_sso_reversal_and_no_pair_when_tlt_outperforms(self):
        trading_index = self.make_index(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
                "2024-01-12",
                "2024-01-16",
                "2024-01-17",
                "2024-01-18",
                "2024-01-19",
                "2024-01-22",
                "2024-01-23",
                "2024-01-24",
                "2024-01-25",
                "2024-01-26",
                "2024-01-29",
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
                "2024-02-02",
                "2024-02-05",
                "2024-02-06",
                "2024-02-07",
            ]
        )
        spy_close_list = [100.0 + 0.5 * idx_int for idx_int in range(1, 22)] + [100.0, 100.0, 100.0, 100.0, 100.0]
        tlt_close_list = list(range(101, 122)) + [100.0, 100.0, 100.0, 100.0, 100.0]
        sso_close_list = list(range(102, 123)) + [101.0, 101.0, 101.0, 101.0, 101.0]
        open_price_df, close_price_df = self.make_open_close_df(
            trading_index=trading_index,
            spy_close_list=spy_close_list,
            tlt_close_list=tlt_close_list,
            sso_close_list=sso_close_list,
        )

        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=EomTltSpySsoVariantConfig(),
        )

        signal_row_ser = month_signal_df.loc["2024-01"]
        self.assertTrue(bool(signal_row_ser["tlt_outperformed_bool"]))
        self.assertEqual(str(signal_row_ser["reversal_asset_str"]), "SSO")
        self.assertTrue(pd.isna(signal_row_ser["pair_entry_bar_ts"]))
        self.assertTrue(pd.isna(signal_row_ser["pair_exit_bar_ts"]))

    def test_build_month_signal_df_skips_reversal_when_signal_window_overlaps_hold_window(self):
        trading_index = self.make_index(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
                "2024-01-12",
                "2024-01-16",
                "2024-01-17",
                "2024-01-18",
                "2024-01-19",
                "2024-01-22",
                "2024-01-23",
                "2024-01-24",
                "2024-01-25",
                "2024-01-26",
                "2024-01-29",
                "2024-02-01",
            ]
        )
        spy_close_list = list(range(101, 121))
        tlt_close_list = list(range(100, 120))
        sso_close_list = list(range(102, 122))
        open_price_df, close_price_df = self.make_open_close_df(
            trading_index=trading_index,
            spy_close_list=spy_close_list,
            tlt_close_list=tlt_close_list,
            sso_close_list=sso_close_list,
        )

        month_signal_df = build_month_signal_df(
            open_price_df=open_price_df,
            close_price_df=close_price_df,
            config=EomTltSpySsoVariantConfig(),
        )

        signal_row_ser = month_signal_df.loc["2024-01"]
        self.assertIsNone(signal_row_ser["reversal_asset_str"])
        self.assertTrue(pd.isna(signal_row_ser["reversal_entry_bar_ts"]))
        self.assertTrue(pd.isna(signal_row_ser["reversal_exit_bar_ts"]))

    def test_build_daily_target_weight_df_and_backtest_cover_reversal_and_pair_legs(self):
        trading_index = self.make_index(
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
                    "asset_str": "TLT",
                    "signed_weight_float": 1.0,
                    "entry_bar_ts": pd.Timestamp("2024-01-30"),
                    "exit_bar_ts": pd.Timestamp("2024-01-31"),
                    "rel_15_return_float": 0.05,
                },
                {
                    "trade_id_int": 2,
                    "leg_type_str": "pair_long_sso",
                    "signal_month_period_str": "2024-01",
                    "asset_str": "SSO",
                    "signed_weight_float": 0.5,
                    "entry_bar_ts": pd.Timestamp("2024-02-01"),
                    "exit_bar_ts": pd.Timestamp("2024-02-07"),
                    "rel_15_return_float": 0.05,
                },
                {
                    "trade_id_int": 3,
                    "leg_type_str": "pair_short_tlt",
                    "signal_month_period_str": "2024-01",
                    "asset_str": "TLT",
                    "signed_weight_float": -0.5,
                    "entry_bar_ts": pd.Timestamp("2024-02-01"),
                    "exit_bar_ts": pd.Timestamp("2024-02-07"),
                    "rel_15_return_float": 0.05,
                },
            ]
        ).set_index("trade_id_int", drop=True)
        daily_target_weight_df = build_daily_target_weight_df(
            trading_index=trading_index,
            trade_leg_plan_df=trade_leg_plan_df,
            asset_list=["SSO", "TLT"],
        )

        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-01-30"), "SSO"]), 0.0)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-01-30"), "TLT"]), 1.0)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-02-01"), "SSO"]), 0.5)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-02-01"), "TLT"]), -0.5)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-02-08"), "SSO"]), 0.0)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-02-08"), "TLT"]), 0.0)

        pricing_data_df = pd.DataFrame(
            {
                ("SSO", "Open"): pd.Series(100.0, index=trading_index, dtype=float),
                ("SSO", "High"): pd.Series([101.0, 101.0, 111.0, 121.0, 126.0, 131.0, 141.0, 141.0], index=trading_index, dtype=float),
                ("SSO", "Low"): pd.Series([99.0, 99.0, 109.0, 119.0, 124.0, 129.0, 139.0, 139.0], index=trading_index, dtype=float),
                ("SSO", "Close"): pd.Series([100.0, 100.0, 110.0, 120.0, 125.0, 130.0, 140.0, 140.0], index=trading_index, dtype=float),
                ("TLT", "Open"): pd.Series(100.0, index=trading_index, dtype=float),
                ("TLT", "High"): pd.Series([101.0, 111.0, 101.0, 99.0, 97.0, 95.0, 91.0, 91.0], index=trading_index, dtype=float),
                ("TLT", "Low"): pd.Series([99.0, 109.0, 99.0, 97.0, 95.0, 93.0, 89.0, 89.0], index=trading_index, dtype=float),
                ("TLT", "Close"): pd.Series([100.0, 110.0, 100.0, 98.0, 96.0, 94.0, 90.0, 90.0], index=trading_index, dtype=float),
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
        self.assertEqual(pd.Timestamp(strategy._transactions.iloc[0]["bar"]), pd.Timestamp("2024-01-30"))
        self.assertEqual(pd.Timestamp(strategy._transactions.iloc[5]["bar"]), pd.Timestamp("2024-02-07"))
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-01-31"), "total_value"]), 1_100.0)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-02-07"), "total_value"]), 1_350.0)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-02-08"), "total_value"]), 1_350.0)
        self.assertAlmostEqual(float(strategy.summary.loc["Final [$]", "Strategy"]), 1_350.0)
        self.assertAlmostEqual(float(strategy.summary.loc["Return [%]", "Strategy"]), 35.0)


if __name__ == "__main__":
    unittest.main()
