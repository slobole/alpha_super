import unittest
import warnings

import pandas as pd

from strategies.strategy_eom_trend_ibit import (
    EomTrendIbitCloseResearchStrategy,
    build_daily_target_weight_ser,
    build_eom_trend_trade_plan_df,
    run_eom_trend_close_research_backtest,
)


class EomTrendIbitStrategyTests(unittest.TestCase):
    @staticmethod
    def make_month_index(date_str_list: list[str]) -> pd.DatetimeIndex:
        return pd.to_datetime(date_str_list)

    def make_strategy(
        self,
        trade_plan_df: pd.DataFrame,
        daily_target_weight_ser: pd.Series,
        capital_base_float: float = 1_000.0,
    ) -> EomTrendIbitCloseResearchStrategy:
        return EomTrendIbitCloseResearchStrategy(
            name="EomTrendIbitTest",
            benchmarks=[],
            trade_symbol_str="IBIT",
            trade_plan_df=trade_plan_df,
            daily_target_weight_ser=daily_target_weight_ser,
            capital_base=capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_build_eom_trend_trade_plan_df_skips_overlap_months(self):
        trading_index = self.make_month_index(
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
        open_price_ser = pd.Series(100.0, index=trading_index, dtype=float)
        close_price_ser = pd.Series(101.0, index=trading_index, dtype=float)

        trade_plan_df = build_eom_trend_trade_plan_df(
            open_price_ser=open_price_ser,
            close_price_ser=close_price_ser,
            signal_day_count_int=15,
            hold_day_count_int=5,
        )

        self.assertEqual(len(trade_plan_df), 0)

    def test_build_eom_trend_trade_plan_df_computes_positive_first_15_return(self):
        trading_index = self.make_month_index(
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
            ]
        )
        open_price_ser = pd.Series(100.0, index=trading_index, dtype=float)
        close_price_ser = pd.Series(range(101, 101 + len(trading_index)), index=trading_index, dtype=float)

        trade_plan_df = build_eom_trend_trade_plan_df(
            open_price_ser=open_price_ser,
            close_price_ser=close_price_ser,
            signal_day_count_int=15,
            hold_day_count_int=5,
        )

        self.assertEqual(len(trade_plan_df), 1)
        plan_row_ser = trade_plan_df.iloc[0]
        self.assertEqual(pd.Timestamp(plan_row_ser["month_start_bar_ts"]), pd.Timestamp("2024-01-02"))
        self.assertEqual(pd.Timestamp(plan_row_ser["signal_end_bar_ts"]), pd.Timestamp("2024-01-23"))
        self.assertEqual(pd.Timestamp(plan_row_ser["entry_bar_ts"]), pd.Timestamp("2024-01-25"))
        self.assertEqual(pd.Timestamp(trade_plan_df.index[0]), pd.Timestamp("2024-01-31"))
        self.assertAlmostEqual(float(plan_row_ser["first_15_return_float"]), 0.15)
        self.assertTrue(bool(plan_row_ser["eligible_bool"]))

    def test_build_daily_target_weight_ser_marks_only_eligible_last_five_days(self):
        trading_index = self.make_month_index(
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
            ]
        )
        open_price_ser = pd.Series(100.0, index=trading_index, dtype=float)
        close_price_ser = pd.Series(range(101, 101 + len(trading_index)), index=trading_index, dtype=float)

        trade_plan_df = build_eom_trend_trade_plan_df(
            open_price_ser=open_price_ser,
            close_price_ser=close_price_ser,
            signal_day_count_int=15,
            hold_day_count_int=5,
        )
        daily_target_weight_ser = build_daily_target_weight_ser(
            trading_index=trading_index,
            trade_plan_df=trade_plan_df,
        )

        self.assertAlmostEqual(float(daily_target_weight_ser.loc[pd.Timestamp("2024-01-24")]), 0.0)
        self.assertAlmostEqual(float(daily_target_weight_ser.loc[pd.Timestamp("2024-01-25")]), 1.0)
        self.assertAlmostEqual(float(daily_target_weight_ser.loc[pd.Timestamp("2024-01-31")]), 1.0)
        self.assertAlmostEqual(float(daily_target_weight_ser.loc[pd.Timestamp("2024-02-01")]), 0.0)

    def test_build_eom_trend_trade_plan_df_marks_nonpositive_month_ineligible(self):
        trading_index = self.make_month_index(
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
            ]
        )
        open_price_ser = pd.Series(100.0, index=trading_index, dtype=float)
        close_price_ser = pd.Series(99.0, index=trading_index, dtype=float)

        trade_plan_df = build_eom_trend_trade_plan_df(
            open_price_ser=open_price_ser,
            close_price_ser=close_price_ser,
            signal_day_count_int=15,
            hold_day_count_int=5,
        )

        self.assertEqual(len(trade_plan_df), 1)
        self.assertFalse(bool(trade_plan_df.iloc[0]["eligible_bool"]))

    def test_run_eom_trend_close_research_backtest_enters_at_open_and_exits_at_month_close(self):
        trading_index = self.make_month_index(
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
            ]
        )
        open_price_ser = pd.Series(100.0, index=trading_index, dtype=float)
        close_price_ser = pd.Series(
            [
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                115.0,
                116.0,
                117.0,
                118.0,
                119.0,
                120.0,
                121.0,
                122.0,
            ],
            index=trading_index,
            dtype=float,
        )

        trade_plan_df = build_eom_trend_trade_plan_df(
            open_price_ser=open_price_ser,
            close_price_ser=close_price_ser,
            signal_day_count_int=15,
            hold_day_count_int=5,
        )
        daily_target_weight_ser = build_daily_target_weight_ser(
            trading_index=trading_index,
            trade_plan_df=trade_plan_df,
        )
        strategy = self.make_strategy(
            trade_plan_df=trade_plan_df,
            daily_target_weight_ser=daily_target_weight_ser,
            capital_base_float=1_000.0,
        )

        pricing_data_df = pd.DataFrame(
            {
                ("IBIT", "Open"): open_price_ser,
                ("IBIT", "High"): close_price_ser + 1.0,
                ("IBIT", "Low"): close_price_ser - 1.0,
                ("IBIT", "Close"): close_price_ser,
            },
            index=trading_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            run_eom_trend_close_research_backtest(
                strategy=strategy,
                pricing_data_df=pricing_data_df,
            )

        self.assertEqual(len(strategy._transactions), 2)
        self.assertEqual(pd.Timestamp(strategy._transactions.iloc[0]["bar"]), pd.Timestamp("2024-01-25"))
        self.assertEqual(pd.Timestamp(strategy._transactions.iloc[1]["bar"]), pd.Timestamp("2024-01-31"))
        self.assertAlmostEqual(float(strategy._transactions.iloc[0]["price"]), 100.0)
        self.assertAlmostEqual(float(strategy._transactions.iloc[1]["price"]), 121.0)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-01-31"), "total_value"]), 1_210.0)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-02-01"), "total_value"]), 1_210.0)
        self.assertAlmostEqual(float(strategy.summary.loc["Final [$]", "Strategy"]), 1_210.0)
        self.assertAlmostEqual(float(strategy.summary.loc["Return [%]", "Strategy"]), 21.0)


if __name__ == "__main__":
    unittest.main()
