import unittest
import warnings

import pandas as pd

from strategies.eom_bond_basket.strategy_eom_bond_basket import (
    EomBondBasketCloseResearchStrategy,
    EomBondBasketConfig,
    DEFAULT_CONFIG,
    build_daily_target_weight_df,
    build_month_trade_plan_df,
    build_trade_leg_plan_df,
    run_eom_bond_basket_close_research_backtest,
)


class EomBondBasketTests(unittest.TestCase):
    @staticmethod
    def make_index(date_str_list: list[str]) -> pd.DatetimeIndex:
        return pd.to_datetime(date_str_list)

    def make_strategy(
        self,
        trade_leg_plan_df: pd.DataFrame,
        daily_target_weight_df: pd.DataFrame,
        capital_base_float: float = 900.0,
    ) -> EomBondBasketCloseResearchStrategy:
        return EomBondBasketCloseResearchStrategy(
            name="EomBondBasketTest",
            benchmarks=[],
            tradeable_asset_list=["TLT", "EDV", "ZROZ"],
            trade_leg_plan_df=trade_leg_plan_df,
            daily_target_weight_df=daily_target_weight_df,
            capital_base=capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_default_config_uses_three_bond_assets(self):
        self.assertEqual(DEFAULT_CONFIG.trade_symbol_list, ("TLT", "EDV", "ZROZ"))
        self.assertAlmostEqual(float(DEFAULT_CONFIG.slippage_float), 0.00025)

    def test_strategy_constructor_default_slippage_uses_house_default(self):
        strategy = EomBondBasketCloseResearchStrategy(
            name="EomBondBasketDefaultSlippageTest",
            benchmarks=[],
            tradeable_asset_list=["TLT", "EDV", "ZROZ"],
            trade_leg_plan_df=pd.DataFrame(),
            daily_target_weight_df=pd.DataFrame(),
        )

        self.assertAlmostEqual(float(strategy._slippage), 0.00025)

    def test_build_month_trade_plan_df_maps_b3_entry_to_b1_exit(self):
        trading_index = self.make_index(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-02-01",
            ]
        )
        close_price_df = pd.DataFrame(
            {
                "TLT": pd.Series(100.0, index=trading_index, dtype=float),
                "EDV": pd.Series(100.0, index=trading_index, dtype=float),
                "ZROZ": pd.Series(100.0, index=trading_index, dtype=float),
            }
        )

        config = EomBondBasketConfig(trade_symbol_list=("TLT", "EDV", "ZROZ"))
        month_trade_plan_df = build_month_trade_plan_df(
            close_price_df=close_price_df,
            config=config,
        )
        trade_leg_plan_df = build_trade_leg_plan_df(
            month_trade_plan_df=month_trade_plan_df,
            config=config,
        )

        self.assertEqual(list(month_trade_plan_df["signal_month_period_str"]), ["2024-01"])
        month_trade_row_ser = month_trade_plan_df.iloc[0]
        self.assertEqual(pd.Timestamp(month_trade_row_ser["entry_bar_ts"]), pd.Timestamp("2024-01-05"))
        self.assertEqual(pd.Timestamp(month_trade_row_ser["exit_bar_ts"]), pd.Timestamp("2024-01-09"))
        self.assertEqual(int(month_trade_row_ser["entry_bday_count_int"]), 3)
        self.assertEqual(int(month_trade_row_ser["exit_bday_count_int"]), 1)

        self.assertEqual(len(trade_leg_plan_df), 3)
        self.assertEqual(list(trade_leg_plan_df["asset_str"]), ["TLT", "EDV", "ZROZ"])
        for signed_weight_float in trade_leg_plan_df["signed_weight_float"]:
            self.assertAlmostEqual(float(signed_weight_float), 1.0 / 3.0)

    def test_build_daily_target_weight_df_is_inclusive_for_close_hold_window(self):
        trading_index = self.make_index(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
        trade_leg_plan_df = pd.DataFrame(
            [
                {
                    "trade_id_int": 1,
                    "basket_trade_id_int": 1,
                    "signal_month_period_str": "2024-01",
                    "asset_str": "TLT",
                    "signed_weight_float": 1.0 / 3.0,
                    "entry_bar_ts": pd.Timestamp("2024-01-05"),
                    "exit_bar_ts": pd.Timestamp("2024-01-09"),
                },
                {
                    "trade_id_int": 2,
                    "basket_trade_id_int": 1,
                    "signal_month_period_str": "2024-01",
                    "asset_str": "EDV",
                    "signed_weight_float": 1.0 / 3.0,
                    "entry_bar_ts": pd.Timestamp("2024-01-05"),
                    "exit_bar_ts": pd.Timestamp("2024-01-09"),
                },
                {
                    "trade_id_int": 3,
                    "basket_trade_id_int": 1,
                    "signal_month_period_str": "2024-01",
                    "asset_str": "ZROZ",
                    "signed_weight_float": 1.0 / 3.0,
                    "entry_bar_ts": pd.Timestamp("2024-01-05"),
                    "exit_bar_ts": pd.Timestamp("2024-01-09"),
                },
            ]
        ).set_index("trade_id_int", drop=True)

        daily_target_weight_df = build_daily_target_weight_df(
            trading_index=trading_index,
            trade_leg_plan_df=trade_leg_plan_df,
            asset_list=["TLT", "EDV", "ZROZ"],
        )

        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-01-04")].sum()), 0.0)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-01-05")].sum()), 1.0)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-01-08")].sum()), 1.0)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-01-09")].sum()), 1.0)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-01-10")].sum()), 0.0)

    def test_close_research_backtest_enters_and_exits_basket_at_close(self):
        trading_index = self.make_index(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
        trade_leg_plan_df = pd.DataFrame(
            [
                {
                    "trade_id_int": 1,
                    "basket_trade_id_int": 1,
                    "signal_month_period_str": "2024-01",
                    "asset_str": "TLT",
                    "signed_weight_float": 1.0 / 3.0,
                    "entry_bar_ts": pd.Timestamp("2024-01-05"),
                    "exit_bar_ts": pd.Timestamp("2024-01-09"),
                },
                {
                    "trade_id_int": 2,
                    "basket_trade_id_int": 1,
                    "signal_month_period_str": "2024-01",
                    "asset_str": "EDV",
                    "signed_weight_float": 1.0 / 3.0,
                    "entry_bar_ts": pd.Timestamp("2024-01-05"),
                    "exit_bar_ts": pd.Timestamp("2024-01-09"),
                },
                {
                    "trade_id_int": 3,
                    "basket_trade_id_int": 1,
                    "signal_month_period_str": "2024-01",
                    "asset_str": "ZROZ",
                    "signed_weight_float": 1.0 / 3.0,
                    "entry_bar_ts": pd.Timestamp("2024-01-05"),
                    "exit_bar_ts": pd.Timestamp("2024-01-09"),
                },
            ]
        ).set_index("trade_id_int", drop=True)
        daily_target_weight_df = build_daily_target_weight_df(
            trading_index=trading_index,
            trade_leg_plan_df=trade_leg_plan_df,
            asset_list=["TLT", "EDV", "ZROZ"],
        )
        pricing_data_df = pd.DataFrame(
            {
                ("TLT", "Open"): pd.Series(100.0, index=trading_index, dtype=float),
                ("TLT", "High"): pd.Series([101.0, 111.0, 121.0, 121.0], index=trading_index, dtype=float),
                ("TLT", "Low"): pd.Series([99.0, 109.0, 119.0, 119.0], index=trading_index, dtype=float),
                ("TLT", "Close"): pd.Series([100.0, 110.0, 120.0, 120.0], index=trading_index, dtype=float),
                ("EDV", "Open"): pd.Series(100.0, index=trading_index, dtype=float),
                ("EDV", "High"): pd.Series([101.0, 91.0, 81.0, 81.0], index=trading_index, dtype=float),
                ("EDV", "Low"): pd.Series([99.0, 89.0, 79.0, 79.0], index=trading_index, dtype=float),
                ("EDV", "Close"): pd.Series([100.0, 90.0, 80.0, 80.0], index=trading_index, dtype=float),
                ("ZROZ", "Open"): pd.Series(100.0, index=trading_index, dtype=float),
                ("ZROZ", "High"): pd.Series([101.0, 101.0, 111.0, 111.0], index=trading_index, dtype=float),
                ("ZROZ", "Low"): pd.Series([99.0, 99.0, 109.0, 109.0], index=trading_index, dtype=float),
                ("ZROZ", "Close"): pd.Series([100.0, 100.0, 110.0, 110.0], index=trading_index, dtype=float),
            },
            index=trading_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

        strategy = self.make_strategy(
            trade_leg_plan_df=trade_leg_plan_df,
            daily_target_weight_df=daily_target_weight_df,
            capital_base_float=900.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            run_eom_bond_basket_close_research_backtest(
                strategy=strategy,
                pricing_data_df=pricing_data_df,
            )

        self.assertEqual(len(strategy._transactions), 6)
        self.assertEqual(pd.Timestamp(strategy._transactions.iloc[0]["bar"]), pd.Timestamp("2024-01-05"))
        self.assertEqual(pd.Timestamp(strategy._transactions.iloc[5]["bar"]), pd.Timestamp("2024-01-09"))
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-01-05"), "total_value"]), 900.0)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-01-08"), "total_value"]), 900.0)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-01-09"), "total_value"]), 930.0)
        self.assertAlmostEqual(float(strategy.results.loc[pd.Timestamp("2024-01-10"), "total_value"]), 930.0)
        self.assertAlmostEqual(float(strategy.summary.loc["Final [$]", "Strategy"]), 930.0)
        self.assertAlmostEqual(float(strategy.summary.loc["Return [%]", "Strategy"]), 100.0 * (930.0 / 900.0 - 1.0))


if __name__ == "__main__":
    unittest.main()
