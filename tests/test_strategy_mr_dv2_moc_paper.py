import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.dv2.strategy_mr_dv2_moc_paper import (
    DV2MocPaperResearchStrategy,
    build_dv2_moc_paper_signal_data_df,
    get_dv2_opportunity_list,
    run_dv2_moc_paper_backtest,
)


class _SignalAuditHarnessStrategy(DV2MocPaperResearchStrategy):
    def __init__(self):
        super().__init__(
            name="DV2MocAuditHarness",
            benchmarks=[],
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            max_positions_int=2,
        )


class DV2MocPaperStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> DV2MocPaperResearchStrategy:
        base_kwargs = dict(
            name="DV2MocPaperTest",
            benchmarks=["$SPX"],
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            max_positions_int=2,
        )
        base_kwargs.update(kwargs)
        return DV2MocPaperResearchStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=260, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 50.0 + 0.18 * step_vec + 1.8 * np.sin(step_vec * 0.05)
        bbb_close_vec = 40.0 + 0.14 * step_vec + 1.5 * np.cos(step_vec * 0.06)
        spx_close_vec = 4000.0 + 3.5 * step_vec + 18.0 * np.sin(step_vec * 0.03)

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): aaa_close_vec - 0.25,
                ("AAA", "High"): aaa_close_vec + 0.60,
                ("AAA", "Low"): aaa_close_vec - 0.60,
                ("AAA", "Close"): aaa_close_vec,
                ("BBB", "Open"): bbb_close_vec - 0.20,
                ("BBB", "High"): bbb_close_vec + 0.55,
                ("BBB", "Low"): bbb_close_vec - 0.55,
                ("BBB", "Close"): bbb_close_vec,
                ("$SPX", "Open"): spx_close_vec - 4.0,
                ("$SPX", "High"): spx_close_vec + 8.0,
                ("$SPX", "Low"): spx_close_vec - 8.0,
                ("$SPX", "Close"): spx_close_vec,
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map, dtype=float)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def make_small_signal_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=4, freq="B")
        signal_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): [9.5, 10.0, 10.5, 13.0],
                ("AAA", "High"): [10.2, 10.5, 12.0, 13.5],
                ("AAA", "Low"): [9.0, 9.8, 10.0, 12.5],
                ("AAA", "Close"): [10.0, 10.0, 11.5, 13.1],
                ("AAA", "p126d_return_ser"): [0.10, 0.10, 0.10, 0.10],
                ("AAA", "natr_value_ser"): [2.0, 2.0, 2.0, 2.0],
                ("AAA", "dv2_value_ser"): [50.0, 5.0, 50.0, 50.0],
                ("AAA", "sma_200_price_ser"): [9.0, 9.0, 9.0, 9.0],
                ("BBB", "Open"): [19.5, 20.0, 20.5, 21.0],
                ("BBB", "High"): [20.2, 20.4, 21.0, 21.5],
                ("BBB", "Low"): [19.0, 19.8, 20.0, 20.5],
                ("BBB", "Close"): [20.0, 20.0, 20.0, 21.2],
                ("BBB", "p126d_return_ser"): [0.10, 0.10, 0.10, 0.10],
                ("BBB", "natr_value_ser"): [1.0, 1.0, 5.0, 1.0],
                ("BBB", "dv2_value_ser"): [50.0, 50.0, 6.0, 50.0],
                ("BBB", "sma_200_price_ser"): [19.0, 19.0, 19.0, 19.0],
                ("$SPX", "Open"): [4000.0, 4010.0, 4020.0, 4030.0],
                ("$SPX", "High"): [4010.0, 4020.0, 4030.0, 4040.0],
                ("$SPX", "Low"): [3990.0, 4000.0, 4010.0, 4020.0],
                ("$SPX", "Close"): [4005.0, 4015.0, 4025.0, 4035.0],
            },
            index=date_index,
            dtype=float,
        )
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)
        return signal_data_df

    def make_small_universe_df(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "AAA": np.ones(len(date_index), dtype=int),
                "BBB": np.ones(len(date_index), dtype=int),
            },
            index=date_index,
        )

    def test_build_dv2_moc_paper_signal_data_df_adds_expected_features_and_passes_signal_audit(self):
        harness_strategy = _SignalAuditHarnessStrategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = build_dv2_moc_paper_signal_data_df(pricing_data_df)

        self.assertIn(("AAA", "p126d_return_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "natr_value_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "dv2_value_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "sma_200_price_ser"), signal_data_df.columns)
        self.assertIn(("BBB", "p126d_return_ser"), signal_data_df.columns)

        harness_strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_get_dv2_opportunity_list_filters_and_sorts_by_natr(self):
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 100.0,
                ("AAA", "p126d_return_ser"): 0.08,
                ("AAA", "natr_value_ser"): 3.0,
                ("AAA", "dv2_value_ser"): 8.0,
                ("AAA", "sma_200_price_ser"): 90.0,
                ("BBB", "Close"): 120.0,
                ("BBB", "p126d_return_ser"): 0.09,
                ("BBB", "natr_value_ser"): 5.0,
                ("BBB", "dv2_value_ser"): 7.0,
                ("BBB", "sma_200_price_ser"): 110.0,
                ("CCC", "Close"): 80.0,
                ("CCC", "p126d_return_ser"): 0.02,
                ("CCC", "natr_value_ser"): 9.0,
                ("CCC", "dv2_value_ser"): 4.0,
                ("CCC", "sma_200_price_ser"): 70.0,
            }
        )
        universe_row_ser = pd.Series({"AAA": 1, "BBB": 1, "CCC": 1}, dtype=int)

        opportunity_list = get_dv2_opportunity_list(close_row_ser, universe_row_ser)

        self.assertEqual(opportunity_list, ["BBB", "AAA"])

    def test_run_dv2_moc_paper_backtest_enters_at_close_and_exits_next_open_without_slot_reuse(self):
        strategy = self.make_strategy(max_positions_int=1)
        signal_data_df = self.make_small_signal_data_df()
        strategy.universe_df = self.make_small_universe_df(signal_data_df.index)

        run_dv2_moc_paper_backtest(
            strategy=strategy,
            pricing_data_df=signal_data_df,
            signal_data_df=signal_data_df,
        )

        transaction_df = strategy.get_transactions().reset_index(drop=True)
        self.assertEqual(len(transaction_df), 2)

        entry_row_ser = transaction_df.iloc[0]
        self.assertEqual(entry_row_ser["asset"], "AAA")
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), pd.Timestamp("2024-01-03"))
        self.assertEqual(float(entry_row_ser["price"]), 10.0)
        self.assertGreater(float(entry_row_ser["amount"]), 0.0)

        exit_row_ser = transaction_df.iloc[1]
        self.assertEqual(exit_row_ser["asset"], "AAA")
        self.assertEqual(pd.Timestamp(exit_row_ser["bar"]), pd.Timestamp("2024-01-05"))
        self.assertEqual(float(exit_row_ser["price"]), 13.0)
        self.assertLess(float(exit_row_ser["amount"]), 0.0)

        self.assertNotIn("BBB", transaction_df["asset"].tolist())
        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in scalar divide",
            category=RuntimeWarning,
        )
        unittest.main()
