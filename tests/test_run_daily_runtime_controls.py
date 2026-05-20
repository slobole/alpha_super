import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import Strategy


def make_pricing_data() -> pd.DataFrame:
    date_index = pd.date_range("2024-01-01", periods=6, freq="D")
    columns = pd.MultiIndex.from_product([["TEST"], ["Open", "High", "Low", "Close"]])
    value_list = [
        [10.0, 10.5, 9.5, 10.0],
        [10.2, 10.8, 9.9, 10.4],
        [10.5, 11.0, 10.1, 10.8],
        [11.0, 11.4, 10.8, 11.2],
        [11.4, 11.9, 11.1, 11.7],
        [11.8, 12.2, 11.5, 12.0],
    ]
    return pd.DataFrame(value_list, index=date_index, columns=columns)


def make_missing_price_pricing_data() -> pd.DataFrame:
    date_index = pd.date_range("2024-01-01", periods=4, freq="D")
    columns = pd.MultiIndex.from_product([["TEST"], ["Open", "High", "Low", "Close"]])
    value_list = [
        [10.0, 10.5, 9.5, 10.0],
        [10.5, 10.8, 10.3, 10.6],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ]
    return pd.DataFrame(value_list, index=date_index, columns=columns)


class AuditTrackingStrategy(Strategy):
    enable_signal_audit = True
    signal_audit_sample_size = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audit_call_count_int = 0
        self.audit_sample_size_arg = None

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        # *** CRITICAL*** The lagged close feature must remain trailing only.
        signal_data[("TEST", "lagged_close")] = signal_data[("TEST", "Close")].shift(1)
        return signal_data

    def audit_signals(self, pricing_data: pd.DataFrame, signal_data: pd.DataFrame, sample_size: int | None = None):
        self.audit_call_count_int += 1
        self.audit_sample_size_arg = sample_size
        return super().audit_signals(pricing_data, signal_data, sample_size=sample_size)

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None

    def finalize(self, current_data: pd.DataFrame):
        return None

    def summarize(self, include_benchmarks=True):
        self.summary = pd.DataFrame()
        self.summary_trades = pd.DataFrame()


class ProgressSmokeStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        for symbol_str in self.signal_progress(["TEST"], desc_str="test signal", total_int=1):
            # *** CRITICAL*** The lagged close feature must remain trailing only.
            signal_data[(symbol_str, "lagged_close")] = signal_data[(symbol_str, "Close")].shift(1)
        return signal_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        if close is None:
            return

        close_price_float = float(close[("TEST", "Close")])
        if self.get_position("TEST") == 0 and close_price_float <= 10.4:
            self.order("TEST", 1, trade_id=1)
        elif self.get_position("TEST") > 0 and close_price_float >= 11.2:
            self.order_target("TEST", 0, trade_id=1)

    def finalize(self, current_data: pd.DataFrame):
        return None


class MissingPriceLiquidationStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data.copy()

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        if close is None:
            return
        if self.get_position("TEST") == 0:
            self.order("TEST", 10, trade_id=1)

    def finalize(self, current_data: pd.DataFrame):
        return None


class RunDailyRuntimeControlTests(unittest.TestCase):
    def make_audit_strategy(self) -> AuditTrackingStrategy:
        return AuditTrackingStrategy(
            name="AuditTracking",
            benchmarks=[],
            capital_base=1000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_progress_strategy(self) -> ProgressSmokeStrategy:
        return ProgressSmokeStrategy(
            name="ProgressSmoke",
            benchmarks=[],
            capital_base=1000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_missing_price_strategy(self) -> MissingPriceLiquidationStrategy:
        return MissingPriceLiquidationStrategy(
            name="MissingPriceLiquidation",
            benchmarks=[],
            capital_base=1000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_run_daily_skips_audit_when_override_is_false(self):
        pricing_data = make_pricing_data()
        strategy = self.make_audit_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                strategy,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )

        self.assertEqual(strategy.audit_call_count_int, 0)

    def test_run_daily_passes_audit_sample_size_override(self):
        pricing_data = make_pricing_data()
        strategy = self.make_audit_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                strategy,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=None,
                audit_sample_size_int=3,
            )

        self.assertEqual(strategy.audit_call_count_int, 1)
        self.assertEqual(strategy.audit_sample_size_arg, 3)

    def test_progress_flags_do_not_change_trades_or_equity(self):
        pricing_data = make_pricing_data()
        strategy_without_progress = self.make_progress_strategy()
        strategy_with_progress = self.make_progress_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                strategy_without_progress,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )
        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                strategy_with_progress,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=True,
                audit_override_bool=False,
            )

        pd.testing.assert_frame_equal(
            strategy_without_progress.results,
            strategy_with_progress.results,
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            strategy_without_progress.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
            strategy_with_progress.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            strategy_without_progress.summary,
            strategy_with_progress.summary,
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            strategy_without_progress.summary_trades,
            strategy_with_progress.summary_trades,
            check_like=True,
        )

    def test_run_daily_force_closes_positions_when_current_bar_prices_disappear(self):
        pricing_data = make_missing_price_pricing_data()
        strategy = self.make_missing_price_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                strategy,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )

        transaction_df = strategy.get_transactions().copy()
        self.assertEqual(len(transaction_df), 2)
        self.assertEqual(float(strategy.get_position("TEST")), 0.0)
        self.assertAlmostEqual(float(strategy.total_value), 1001.0)
        self.assertEqual(len(strategy._open_trades), 0)

        closing_trade_ser = transaction_df.iloc[-1]
        self.assertEqual(pd.Timestamp(closing_trade_ser["bar"]), pd.Timestamp("2024-01-03"))
        self.assertEqual(int(closing_trade_ser["trade_id"]), 1)
        self.assertEqual(float(closing_trade_ser["amount"]), -10.0)
        self.assertAlmostEqual(float(closing_trade_ser["price"]), 10.6)

    def test_run_daily_structured_logging_is_opt_in_and_preserves_results(self):
        pricing_data = make_pricing_data()
        baseline_strategy = self.make_progress_strategy()
        logged_strategy = self.make_progress_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                baseline_strategy,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir_path_obj = Path(temp_dir_str)
            audit_log_path_obj = temp_dir_path_obj / "engine_events.jsonl"
            trace_root_path_obj = temp_dir_path_obj / "pods"
            with contextlib.redirect_stdout(io.StringIO()):
                run_daily(
                    logged_strategy,
                    pricing_data,
                    show_progress=False,
                    show_signal_progress_bool=False,
                    audit_override_bool=False,
                    run_id_str="run_001",
                    audit_log_path_str=str(audit_log_path_obj),
                    trace_enabled_bool=True,
                    trace_log_root_path_str=str(trace_root_path_obj),
                )

            pd.testing.assert_frame_equal(
                baseline_strategy.results,
                logged_strategy.results,
                check_like=True,
            )
            pd.testing.assert_frame_equal(
                baseline_strategy.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
                logged_strategy.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
                check_like=True,
            )

            audit_event_name_list = [
                json.loads(line_str)["event_name_str"]
                for line_str in audit_log_path_obj.read_text(encoding="utf-8").splitlines()
                if line_str.strip() != ""
            ]
            self.assertIn("engine.signal_precompute.started", audit_event_name_list)
            self.assertIn("engine.signal_precompute.completed", audit_event_name_list)
            self.assertIn("engine.signal_audit.skipped", audit_event_name_list)
            self.assertIn("engine.backtest_loop.started", audit_event_name_list)
            self.assertIn("engine.backtest_loop.completed", audit_event_name_list)
            self.assertIn("engine.order.submitted", audit_event_name_list)
            self.assertIn("engine.order.executed", audit_event_name_list)

            trace_log_path_obj = trace_root_path_obj / "ProgressSmoke" / "run_001" / "trace_events.jsonl"
            trace_event_name_list = [
                json.loads(line_str)["event_name_str"]
                for line_str in trace_log_path_obj.read_text(encoding="utf-8").splitlines()
                if line_str.strip() != ""
            ]
            self.assertIn("engine.bar.completed", trace_event_name_list)

    def test_run_daily_structured_logging_failure_does_not_change_results(self):
        pricing_data = make_pricing_data()
        baseline_strategy = self.make_progress_strategy()
        logged_strategy = self.make_progress_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                baseline_strategy,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )

        with (
            mock.patch("alpha.engine.strategy.log_event", side_effect=OSError("audit log unavailable")),
            mock.patch("alpha.engine.strategy.log_trace_event", side_effect=OSError("trace log unavailable")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_daily(
                logged_strategy,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
                run_id_str="run_001",
                audit_log_path_str="unused.jsonl",
                trace_enabled_bool=True,
            )

        pd.testing.assert_frame_equal(
            baseline_strategy.results,
            logged_strategy.results,
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            baseline_strategy.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
            logged_strategy.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
            check_like=True,
        )
        self.assertGreater(logged_strategy._structured_logging_error_count_int, 0)


if __name__ == "__main__":
    unittest.main()
