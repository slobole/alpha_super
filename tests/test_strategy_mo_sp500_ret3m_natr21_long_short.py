import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.momentum.strategy_mo_sp500_ret3m_natr21_long_short import (
    NATR_WINDOW_INT,
    REBALANCE_FREQUENCY_ANNUAL_STR,
    REBALANCE_FREQUENCY_MONTHLY_STR,
    REBALANCE_FREQUENCY_QUARTERLY_STR,
    RETURN_LOOKBACK_DAY_INT,
    Sp500Ret3mNatr21LongShortConfig,
    Sp500Ret3mNatr21LongShortStrategy,
    compute_ret3m_natr21_signal_tables,
    get_rebalance_decision_close_df,
    map_decision_dates_to_next_open_schedule_df,
)


class Sp500Ret3mNatr21LongShortTests(unittest.TestCase):
    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-04-01",
        decision_date_str: str = "2024-03-28",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> Sp500Ret3mNatr21LongShortStrategy:
        base_kwargs = dict(
            name="Sp500Ret3mNatr21LongShortTest",
            benchmarks=["SPY"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            benchmark_symbol_str="SPY",
            capital_base=10_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            return_lookback_day_int=RETURN_LOOKBACK_DAY_INT,
            natr_window_int=NATR_WINDOW_INT,
            quantile_fraction_float=0.25,
            gross_exposure_float=1.0,
            rebalance_frequency_str=REBALANCE_FREQUENCY_MONTHLY_STR,
        )
        base_kwargs.update(kwargs)
        return Sp500Ret3mNatr21LongShortStrategy(**base_kwargs)

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    @staticmethod
    def make_price_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        date_index = pd.bdate_range("2023-01-02", "2023-06-30")
        step_vec = np.arange(len(date_index), dtype=float)
        close_vec = 100.0 + step_vec
        price_close_df = pd.DataFrame(
            {
                "AAA": close_vec,
                "BBB": 90.0 + (0.5 * step_vec),
            },
            index=date_index,
        )
        price_high_df = price_close_df * 1.01
        price_low_df = price_close_df * 0.99
        return price_close_df, price_high_df, price_low_df

    def test_decision_close_frequency_uses_last_completed_period(self):
        date_index = pd.to_datetime(
            [
                "2023-12-29",
                "2024-01-31",
                "2024-02-29",
                "2024-03-28",
                "2024-04-30",
                "2024-05-22",
            ]
        )
        price_close_df = pd.DataFrame({"AAA": np.arange(len(date_index), dtype=float)}, index=date_index)

        monthly_decision_close_df = get_rebalance_decision_close_df(
            price_close_df=price_close_df,
            rebalance_frequency_str=REBALANCE_FREQUENCY_MONTHLY_STR,
        )
        quarterly_decision_close_df = get_rebalance_decision_close_df(
            price_close_df=price_close_df,
            rebalance_frequency_str=REBALANCE_FREQUENCY_QUARTERLY_STR,
        )
        annual_decision_close_df = get_rebalance_decision_close_df(
            price_close_df=price_close_df,
            rebalance_frequency_str=REBALANCE_FREQUENCY_ANNUAL_STR,
        )

        self.assertEqual(pd.Timestamp(monthly_decision_close_df.index[-1]), pd.Timestamp("2024-04-30"))
        self.assertEqual(
            quarterly_decision_close_df.index.tolist(),
            [pd.Timestamp("2023-12-29"), pd.Timestamp("2024-03-28")],
        )
        self.assertEqual(annual_decision_close_df.index.tolist(), [pd.Timestamp("2023-12-29")])

    def test_map_decision_dates_to_next_open_schedule(self):
        rebalance_schedule_df = map_decision_dates_to_next_open_schedule_df(
            decision_date_index=pd.to_datetime(["2024-03-28", "2024-04-30"]),
            execution_index=pd.to_datetime(["2024-03-28", "2024-04-01", "2024-04-30", "2024-05-01"]),
        )

        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-04-01"), "decision_date_ts"]),
            pd.Timestamp("2024-03-28"),
        )
        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-05-01"), "decision_date_ts"]),
            pd.Timestamp("2024-04-30"),
        )

    def test_compute_signal_tables_match_return_natr_and_score_formulas(self):
        price_close_df, price_high_df, price_low_df = self.make_price_frames()
        config = Sp500Ret3mNatr21LongShortConfig(
            rebalance_frequency_str=REBALANCE_FREQUENCY_MONTHLY_STR,
        )

        (
            _decision_close_df,
            return_3m_decision_df,
            natr_21_decision_df,
            rank_score_decision_df,
        ) = compute_ret3m_natr21_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            config=config,
        )

        decision_date_ts = pd.Timestamp(rank_score_decision_df.index[-1])
        prior_close_ser = price_close_df["AAA"].shift(1)
        true_range_df = pd.concat(
            [
                price_high_df["AAA"] - price_low_df["AAA"],
                (price_high_df["AAA"] - prior_close_ser).abs(),
                (price_low_df["AAA"] - prior_close_ser).abs(),
            ],
            axis=1,
        )
        expected_return_float = (
            float(price_close_df.loc[decision_date_ts, "AAA"])
            / float(price_close_df.shift(RETURN_LOOKBACK_DAY_INT).loc[decision_date_ts, "AAA"])
            - 1.0
        )
        expected_atr_float = float(
            true_range_df.max(axis=1)
            .rolling(window=NATR_WINDOW_INT, min_periods=NATR_WINDOW_INT)
            .mean()
            .loc[decision_date_ts]
        )
        expected_natr_float = 100.0 * expected_atr_float / float(price_close_df.loc[decision_date_ts, "AAA"])

        self.assertAlmostEqual(float(return_3m_decision_df.loc[decision_date_ts, "AAA"]), expected_return_float)
        self.assertAlmostEqual(float(natr_21_decision_df.loc[decision_date_ts, "AAA"]), expected_natr_float)
        self.assertAlmostEqual(
            float(rank_score_decision_df.loc[decision_date_ts, "AAA"]),
            expected_return_float / expected_natr_float,
        )

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        price_close_df, price_high_df, price_low_df = self.make_price_frames()
        pricing_data_map: dict[tuple[str, str], pd.Series] = {}
        for symbol_str in ("AAA", "BBB"):
            pricing_data_map[(symbol_str, "Open")] = price_close_df[symbol_str] * 0.999
            pricing_data_map[(symbol_str, "High")] = price_high_df[symbol_str]
            pricing_data_map[(symbol_str, "Low")] = price_low_df[symbol_str]
            pricing_data_map[(symbol_str, "Close")] = price_close_df[symbol_str]
        pricing_data_map[("SPY", "Open")] = pd.Series(300.0, index=price_close_df.index)
        pricing_data_map[("SPY", "High")] = pd.Series(301.0, index=price_close_df.index)
        pricing_data_map[("SPY", "Low")] = pd.Series(299.0, index=price_close_df.index)
        pricing_data_map[("SPY", "Close")] = pd.Series(300.0, index=price_close_df.index)
        pricing_data_df = pd.DataFrame(pricing_data_map, index=price_close_df.index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

        strategy = self.make_strategy()
        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "return_63d_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "natr_21_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "rank_score_ser"), signal_data_df.columns)
        strategy.audit_signals(pricing_data_df, signal_data_df, sample_size=4)

    def test_target_weights_are_net_neutral_top_and_bottom_quantiles(self):
        strategy = self.make_strategy(quantile_fraction_float=0.25, gross_exposure_float=1.0)
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")},
            index=[strategy.previous_bar],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "rank_score_ser"): 8.0,
                ("BBB", "rank_score_ser"): 7.0,
                ("CCC", "rank_score_ser"): 6.0,
                ("DDD", "rank_score_ser"): 5.0,
                ("EEE", "rank_score_ser"): 4.0,
                ("FFF", "rank_score_ser"): 3.0,
                ("GGG", "rank_score_ser"): 2.0,
                ("HHH", "rank_score_ser"): 1.0,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.to_dict(), {"AAA": 0.25, "BBB": 0.25, "GGG": -0.25, "HHH": -0.25})
        self.assertAlmostEqual(float(target_weight_ser.sum()), 0.0)
        self.assertAlmostEqual(float(target_weight_ser.abs().sum()), 1.0)

    def test_target_weights_use_latest_prior_pit_universe_row(self):
        strategy = self.make_strategy(quantile_fraction_float=0.25, gross_exposure_float=1.0)
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "EEE": [0],
            },
            index=[pd.Timestamp("2024-03-27")],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "rank_score_ser"): 4.0,
                ("BBB", "rank_score_ser"): 3.0,
                ("CCC", "rank_score_ser"): 2.0,
                ("DDD", "rank_score_ser"): 1.0,
                ("EEE", "rank_score_ser"): 99.0,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.to_dict(), {"AAA": 0.5, "DDD": -0.5})

    def test_iterate_closes_sign_flip_before_opening_new_side(self):
        strategy = self.make_strategy(quantile_fraction_float=0.25, gross_exposure_float=1.0)
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.current_bar = pd.Timestamp("2024-04-01")
        strategy.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in ("AAA", "BBB", "CCC", "DDD")},
            index=[strategy.previous_bar],
        )
        strategy.add_transaction(7, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 7
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "rank_score_ser"): 1.0,
                ("BBB", "rank_score_ser"): 2.0,
                ("CCC", "rank_score_ser"): 3.0,
                ("DDD", "rank_score_ser"): 4.0,
            }
        )

        strategy.iterate(
            pd.DataFrame(index=[strategy.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0, "DDD": 100.0}, dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["AAA", "AAA", "DDD"])
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertEqual(order_list[0].amount, 0)
        self.assertEqual(order_list[0].unit, "shares")
        self.assertTrue(order_list[0].target)
        self.assertEqual(order_list[0].trade_id, 7)
        self.assertEqual(order_list[1].unit, "percent")
        self.assertTrue(order_list[1].target)
        self.assertAlmostEqual(float(order_list[1].amount), -0.5)
        self.assertNotEqual(order_list[1].trade_id, 7)
        self.assertAlmostEqual(float(order_list[2].amount), 0.5)


if __name__ == "__main__":
    unittest.main()
