import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.execution_timing import (
    ExecutionTimingAnalysis,
    compute_cvar_5_pct_float,
)
from alpha.engine.strategy import Strategy
from execution_timing_analysis import _strategy_module_name_str


def make_timing_pricing_data_df() -> pd.DataFrame:
    calendar_idx = pd.date_range("2024-01-02", periods=5, freq="B")
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): [9.8, 10.5, 11.0, 12.0, 12.4],
            ("AAA", "High"): [10.4, 11.0, 12.0, 12.6, 12.8],
            ("AAA", "Low"): [9.6, 10.1, 10.8, 11.8, 12.1],
            ("AAA", "Close"): [10.0, 10.8, 11.6, 12.2, 12.5],
            ("BBB", "Open"): [19.8, 20.2, 20.5, 20.8, 21.0],
            ("BBB", "High"): [20.4, 20.8, 21.0, 21.3, 21.5],
            ("BBB", "Low"): [19.6, 20.0, 20.2, 20.5, 20.8],
            ("BBB", "Close"): [20.0, 20.4, 20.7, 21.0, 21.2],
        },
        index=calendar_idx,
        dtype=float,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def make_timing_pricing_data_with_benchmark_df() -> pd.DataFrame:
    pricing_data_df = make_timing_pricing_data_df()
    benchmark_df = pd.DataFrame(
        {
            ("$SPX", "Open"): [100.0, 101.0, 102.0, 103.0, 104.0],
            ("$SPX", "High"): [101.0, 102.0, 103.0, 104.0, 105.0],
            ("$SPX", "Low"): [99.0, 100.0, 101.0, 102.0, 103.0],
            ("$SPX", "Close"): [100.5, 101.5, 102.5, 103.5, 104.5],
        },
        index=pricing_data_df.index,
        dtype=float,
    )
    benchmark_df.columns = pd.MultiIndex.from_tuples(benchmark_df.columns)
    return pd.concat([pricing_data_df, benchmark_df], axis=1)


def make_timing_pricing_data_with_spy_df() -> pd.DataFrame:
    pricing_data_df = make_timing_pricing_data_df()
    spy_df = pd.DataFrame(
        {
            ("SPY", "Open"): [100.0, 101.0, 102.0, 103.0, 104.0],
            ("SPY", "High"): [101.0, 102.0, 103.0, 104.0, 105.0],
            ("SPY", "Low"): [99.0, 100.0, 101.0, 102.0, 103.0],
            ("SPY", "Close"): [100.5, 101.5, 102.5, 103.5, 104.5],
        },
        index=pricing_data_df.index,
        dtype=float,
    )
    spy_df.columns = pd.MultiIndex.from_tuples(spy_df.columns)
    return pd.concat([pricing_data_df, spy_df], axis=1)


class TimingToyStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data.copy()

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None:
            return

        close_price_float = float(close[("AAA", "Close")])
        if self.get_position("AAA") == 0 and close_price_float <= 10.0:
            self.order("AAA", 1, trade_id=1)
        elif self.get_position("AAA") > 0 and close_price_float >= 11.6:
            self.order_target("AAA", 0, trade_id=1)


class DelayedSlotToyStrategy(Strategy):
    max_positions = 1

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data.copy()

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None:
            return

        aaa_close_float = float(close[("AAA", "Close")])
        if self.get_position("AAA") == 0 and self.get_position("BBB") == 0 and aaa_close_float <= 10.0:
            self.order("AAA", 1, trade_id=1)
            return

        if self.get_position("AAA") > 0 and aaa_close_float >= 10.8:
            self.order_target("AAA", 0, trade_id=1)
            self.order("BBB", 1, trade_id=2)


class CurrentBarRebalanceToyStrategy(Strategy):
    def __init__(self, rebalance_weight_df: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebalance_weight_df = rebalance_weight_df.copy()

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data.copy()

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None:
            return
        if self.current_bar not in self.rebalance_weight_df.index:
            return

        target_weight_float = float(self.rebalance_weight_df.loc[self.current_bar, "AAA"])
        self.order_target_percent("AAA", target_weight_float, trade_id=1)


class ExecutionTimingAnalysisTests(unittest.TestCase):
    def make_timing_strategy(self) -> TimingToyStrategy:
        return TimingToyStrategy(
            name="TimingToy",
            benchmarks=[],
            capital_base=1_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_delayed_slot_strategy(self) -> DelayedSlotToyStrategy:
        return DelayedSlotToyStrategy(
            name="DelayedSlotToy",
            benchmarks=[],
            capital_base=1_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_current_bar_rebalance_strategy(self) -> CurrentBarRebalanceToyStrategy:
        rebalance_weight_df = pd.DataFrame(
            {"AAA": [0.5, 0.0]},
            index=pd.to_datetime(["2024-01-03", "2024-01-05"]),
            dtype=float,
        )
        return CurrentBarRebalanceToyStrategy(
            rebalance_weight_df=rebalance_weight_df,
            name="CurrentBarRebalanceToy",
            benchmarks=[],
            capital_base=1_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_next_open_next_open_cell_matches_vanilla_backtest(self):
        pricing_data_df = make_timing_pricing_data_df()
        vanilla_strategy_obj = self.make_timing_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                vanilla_strategy_obj,
                pricing_data_df,
                calendar=pricing_data_df.index,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_timing_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("next_open",),
            exit_timing_str_tuple=("next_open",),
            save_output_bool=False,
        ).run()
        timing_strategy_obj = timing_result_obj.strategy_map[("next_open", "next_open")]

        pd.testing.assert_series_equal(
            vanilla_strategy_obj.results["total_value"],
            timing_strategy_obj.results["total_value"],
            check_names=False,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            vanilla_strategy_obj.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
            timing_strategy_obj.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
        )

    def test_same_close_moc_entry_fills_at_signal_bar_close(self):
        pricing_data_df = make_timing_pricing_data_df()

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_timing_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("same_close_moc",),
            exit_timing_str_tuple=("next_open",),
            save_output_bool=False,
        ).run()
        transaction_df = timing_result_obj.strategy_map[
            ("same_close_moc", "next_open")
        ].get_transactions().reset_index(drop=True)

        entry_row_ser = transaction_df.iloc[0]
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), pd.Timestamp("2024-01-02"))
        self.assertAlmostEqual(float(entry_row_ser["price"]), 10.0)

    def test_next_close_exit_fills_at_next_bar_close(self):
        pricing_data_df = make_timing_pricing_data_df()

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_timing_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("same_close_moc",),
            exit_timing_str_tuple=("next_close",),
            save_output_bool=False,
        ).run()
        transaction_df = timing_result_obj.strategy_map[
            ("same_close_moc", "next_close")
        ].get_transactions().reset_index(drop=True)

        exit_row_ser = transaction_df.iloc[1]
        self.assertEqual(pd.Timestamp(exit_row_ser["bar"]), pd.Timestamp("2024-01-05"))
        self.assertAlmostEqual(float(exit_row_ser["price"]), 12.2)

    def test_delayed_exit_does_not_free_slot_before_earlier_entry_fill(self):
        pricing_data_df = make_timing_pricing_data_df().iloc[:3].copy()

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_delayed_slot_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("same_close_moc",),
            exit_timing_str_tuple=("next_open",),
            save_output_bool=False,
        ).run()
        transaction_df = timing_result_obj.strategy_map[
            ("same_close_moc", "next_open")
        ].get_transactions().reset_index(drop=True)

        self.assertEqual(transaction_df["asset"].tolist(), ["AAA", "AAA"])
        self.assertNotIn("BBB", transaction_df["asset"].tolist())

    def test_missing_next_open_entry_price_cancels_order_like_vanilla(self):
        pricing_data_df = make_timing_pricing_data_df()
        pricing_data_df.loc[pd.Timestamp("2024-01-03"), ("AAA", "Open")] = np.nan

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_timing_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("next_open",),
            exit_timing_str_tuple=("next_open",),
            save_output_bool=False,
        ).run()
        timing_strategy_obj = timing_result_obj.strategy_map[("next_open", "next_open")]

        self.assertEqual(len(timing_strategy_obj.get_transactions()), 0)
        self.assertAlmostEqual(
            float(timing_strategy_obj.results["total_value"].iloc[-1]),
            1_000.0,
        )

    def test_missing_next_open_exit_price_liquidates_stale_position_before_order(self):
        pricing_data_df = make_timing_pricing_data_df()
        pricing_data_df.loc[pd.Timestamp("2024-01-05"), ("AAA", "Open")] = np.nan
        pricing_data_df.loc[pd.Timestamp("2024-01-05"), ("AAA", "Close")] = np.nan

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_timing_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("same_close_moc",),
            exit_timing_str_tuple=("next_open",),
            save_output_bool=False,
        ).run()
        transaction_df = timing_result_obj.strategy_map[
            ("same_close_moc", "next_open")
        ].get_transactions().reset_index(drop=True)

        self.assertEqual(len(transaction_df), 2)
        stale_exit_row_ser = transaction_df.iloc[1]
        self.assertEqual(pd.Timestamp(stale_exit_row_ser["bar"]), pd.Timestamp("2024-01-05"))
        self.assertAlmostEqual(float(stale_exit_row_ser["price"]), 11.6)

    def test_cvar_5_matches_tail_formula(self):
        daily_return_ser = pd.Series([-0.10, -0.05, 0.0, 0.05, 0.10], dtype=float)

        cvar_5_pct_float = compute_cvar_5_pct_float(daily_return_ser)

        self.assertAlmostEqual(cvar_5_pct_float, -10.0)

    def test_risk_labels_are_assigned_by_timing_mode(self):
        pricing_data_df = make_timing_pricing_data_df()

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_timing_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            save_output_bool=False,
        ).run()
        metric_df = timing_result_obj.metric_df.set_index(["Entry Timing", "Exit Timing"])

        self.assertNotIn("Risk Note", timing_result_obj.metric_df.columns)
        self.assertEqual(timing_result_obj.metric_df.iloc[0]["Scenario"], "Default")
        self.assertEqual(timing_result_obj.metric_df.iloc[0]["Entry Timing"], "T+1 Open")
        self.assertEqual(timing_result_obj.metric_df.iloc[0]["Exit Timing"], "T+1 Open")
        self.assertNotIn("Trades", timing_result_obj.metric_df.columns)
        self.assertEqual(metric_df.loc[("T+1 Open", "T+1 Close (MOC)"), "Risk Label"], "Clean")
        self.assertEqual(metric_df.loc[("T Close (Biased/MOC)", "T+1 Open"), "Risk Label"], "MOC Assumption")

        diagnostic_timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_timing_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("same_open",),
            exit_timing_str_tuple=("next_open",),
            save_output_bool=False,
        ).run()
        diagnostic_metric_df = diagnostic_timing_result_obj.metric_df.set_index(["Entry Timing", "Exit Timing"])
        self.assertEqual(
            diagnostic_metric_df.loc[("T Open (Diagnostic)", "T+1 Open"), "Risk Label"],
            "Diagnostic Only",
        )

    def test_save_output_writes_expected_artifacts(self):
        pricing_data_df = make_timing_pricing_data_df()

        with tempfile.TemporaryDirectory() as temp_dir_str:
            timing_result_obj = ExecutionTimingAnalysis(
                strategy_factory_fn=self.make_timing_strategy,
                pricing_data_df=pricing_data_df,
                calendar_idx=pricing_data_df.index,
                entry_timing_str_tuple=("next_open",),
                exit_timing_str_tuple=("next_open",),
                output_dir_str=temp_dir_str,
                save_output_bool=True,
            ).run()

            output_path = Path(timing_result_obj.output_dir_path)
            self.assertTrue((output_path / "execution_timing_metrics.csv").exists())
            self.assertTrue((output_path / "ann_return_matrix.csv").exists())
            self.assertTrue((output_path / "cvar_5_matrix.csv").exists())
            self.assertTrue((output_path / "sharpe_matrix.csv").exists())
            self.assertTrue((output_path / "max_drawdown_matrix.csv").exists())
            self.assertTrue((output_path / "metadata.json").exists())
            self.assertTrue((output_path / "report.html").exists())

    def test_vanilla_current_bar_same_open_cell_matches_vanilla_rebalance_strategy(self):
        pricing_data_df = make_timing_pricing_data_df()
        vanilla_strategy_obj = self.make_current_bar_rebalance_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                vanilla_strategy_obj,
                pricing_data_df,
                calendar=pricing_data_df.index,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_current_bar_rebalance_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("same_open",),
            exit_timing_str_tuple=("same_open",),
            save_output_bool=False,
            order_generation_mode_str="vanilla_current_bar",
            risk_model_str="taa_rebalance",
        ).run()
        timing_strategy_obj = timing_result_obj.strategy_map[("same_open", "same_open")]

        pd.testing.assert_series_equal(
            vanilla_strategy_obj.results["total_value"],
            timing_strategy_obj.results["total_value"],
            check_names=False,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            vanilla_strategy_obj.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
            timing_strategy_obj.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
        )

    def test_taa_rebalance_risk_model_labels_same_open_as_clean(self):
        pricing_data_df = make_timing_pricing_data_df()

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_current_bar_rebalance_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("same_open", "same_close_moc"),
            exit_timing_str_tuple=("same_open", "next_open"),
            save_output_bool=False,
            order_generation_mode_str="vanilla_current_bar",
            risk_model_str="taa_rebalance",
        ).run()
        metric_df = timing_result_obj.metric_df.set_index(["Entry Timing", "Exit Timing"])

        self.assertEqual(metric_df.loc[("T+1 Open", "T+1 Open"), "Risk Label"], "Clean")
        self.assertEqual(metric_df.loc[("T+1 Close (MOC)", "T+1 Open"), "Risk Label"], "MOC Assumption")
        self.assertEqual(metric_df.loc[("T+1 Open", "T+2 Open"), "Risk Label"], "Funding Assumption")

    def test_taa_signal_bar_t_close_cell_is_visible_and_biased(self):
        pricing_data_df = make_timing_pricing_data_df()

        timing_result_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=self.make_current_bar_rebalance_strategy,
            pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index,
            entry_timing_str_tuple=("same_close_moc", "next_open"),
            exit_timing_str_tuple=("same_close_moc", "next_open"),
            save_output_bool=False,
            order_generation_mode_str="signal_bar",
            risk_model_str="taa_rebalance",
            default_entry_timing_str="next_open",
            default_exit_timing_str="next_open",
        ).run()
        metric_df = timing_result_obj.metric_df.set_index(["Entry Timing", "Exit Timing"])

        self.assertEqual(timing_result_obj.metric_df.iloc[0]["Scenario"], "Default")
        self.assertEqual(metric_df.loc[("T Close (Biased/MOC)", "T+1 Open"), "Risk Label"], "Biased MOC")
        self.assertEqual(metric_df.loc[("T+1 Open", "T+1 Open"), "Risk Label"], "Clean")

    def test_cli_accepts_dotted_module_and_repo_relative_python_path(self):
        self.assertEqual(
            _strategy_module_name_str("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"),
            "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        )
        self.assertEqual(
            _strategy_module_name_str(r".\strategies\taa_df\strategy_taa_df_btal_fallback_tqqq_vix_cash.py"),
            "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        )

    def test_dv2_module_exposes_execution_timing_cli_hook(self):
        import strategies.dv2.strategy_mr_dv2 as dv2_module

        pricing_data_df = make_timing_pricing_data_with_benchmark_df()
        universe_df = pd.DataFrame(
            1,
            index=pricing_data_df.index,
            columns=["AAA", "BBB"],
            dtype=int,
        )

        with patch.object(
            dv2_module,
            "build_index_constituent_matrix",
            return_value=(["AAA", "BBB"], universe_df),
        ):
            with patch.object(dv2_module, "get_prices", return_value=pricing_data_df):
                strategy_input_dict = dv2_module.build_execution_timing_analysis_inputs()

        self.assertEqual(strategy_input_dict["order_generation_mode_str"], "signal_bar")
        self.assertEqual(strategy_input_dict["risk_model_str"], "daily_ohlc_signal")
        self.assertEqual(strategy_input_dict["default_entry_timing_str"], "next_open")
        self.assertEqual(strategy_input_dict["default_exit_timing_str"], "next_open")

        strategy_obj = strategy_input_dict["strategy_factory_fn"]()
        self.assertEqual(strategy_obj.name, "strategy_mr_dv2")
        self.assertIs(strategy_obj.universe_df, universe_df)

    def test_qpi_modules_expose_execution_timing_cli_hooks(self):
        import strategies.qpi.strategy_mr_qpi as qpi_module
        import strategies.qpi.strategy_mr_qpi_ibs_rsi_exit as qpi_ibs_module

        pricing_data_df = make_timing_pricing_data_with_benchmark_df()
        universe_df = pd.DataFrame(
            1,
            index=pricing_data_df.index,
            columns=["AAA", "BBB"],
            dtype=int,
        )

        for module_obj, expected_strategy_name_str in (
            (qpi_module, "strategy_mr_qpi"),
            (qpi_ibs_module, "strategy_mr_qpi_ibs_rsi_exit"),
        ):
            with patch.object(
                module_obj,
                "build_index_constituent_matrix",
                return_value=(["AAA", "BBB"], universe_df),
            ):
                with patch.object(module_obj, "get_prices", return_value=pricing_data_df):
                    strategy_input_dict = module_obj.build_execution_timing_analysis_inputs()

            self.assertEqual(strategy_input_dict["order_generation_mode_str"], "signal_bar")
            self.assertEqual(strategy_input_dict["risk_model_str"], "daily_ohlc_signal")
            self.assertEqual(strategy_input_dict["entry_timing_str_tuple"], ("same_close_moc", "next_open", "next_close"))
            self.assertEqual(strategy_input_dict["default_entry_timing_str"], "next_open")

            strategy_obj = strategy_input_dict["strategy_factory_fn"]()
            self.assertEqual(strategy_obj.name, expected_strategy_name_str)
            self.assertIs(strategy_obj.universe_df, universe_df)

    def test_atr_normalized_ndx_module_exposes_decision_close_timing_hook(self):
        import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module

        pricing_data_df = make_timing_pricing_data_with_spy_df()
        universe_df = pd.DataFrame(
            1,
            index=pricing_data_df.index,
            columns=["AAA", "BBB"],
            dtype=int,
        )
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-01-03")]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-04")], name="execution_date_ts"),
        )

        with patch.object(
            atr_module,
            "get_atr_normalized_ndx_data",
            return_value=(pricing_data_df, universe_df, rebalance_schedule_df),
        ):
            strategy_input_dict = atr_module.build_execution_timing_analysis_inputs()

        self.assertEqual(strategy_input_dict["order_generation_mode_str"], "signal_bar")
        self.assertEqual(strategy_input_dict["risk_model_str"], "taa_rebalance")
        self.assertEqual(strategy_input_dict["entry_timing_str_tuple"], ("same_close_moc", "next_open", "next_close"))
        self.assertEqual(strategy_input_dict["default_entry_timing_str"], "next_open")

        strategy_obj = strategy_input_dict["strategy_factory_fn"]()
        self.assertEqual(strategy_obj.name, "strategy_mo_atr_normalized_ndx")
        self.assertIs(strategy_obj.universe_df, universe_df)
        self.assertIn(pd.Timestamp("2024-01-03"), strategy_obj.rebalance_schedule_df.index)
        self.assertEqual(
            pd.Timestamp(strategy_obj.rebalance_schedule_df.loc[pd.Timestamp("2024-01-03"), "decision_date_ts"]),
            pd.Timestamp("2024-01-03"),
        )


if __name__ == "__main__":
    unittest.main()
