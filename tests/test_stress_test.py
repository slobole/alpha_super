import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alpha.engine.crisis import (
    CrisisPeriodConfig,
    CrisisStrategySpec,
    run_crisis_replay_suite,
)
from alpha.engine.strategy import Strategy
from alpha.engine.stress_test import (
    StressTestAnalyzer,
    _build_verdict_html,
    _heatmap_cell_style_str,
    _heatmap_scale_max_float,
    resolve_stress_launch_window,
    run_stress_test_suite,
    save_stress_test_results,
)


class BuyAndHoldOneShareStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None:
            return
        if self.get_position("AAA") != 0:
            return
        self.order("AAA", 1, trade_id=1)


class SellOnEventStartStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None:
            return
        current_ts = pd.Timestamp(self.current_bar).normalize()
        if self.get_position("AAA") == 0 and current_ts < pd.Timestamp("2020-02-20"):
            self.order("AAA", 1, trade_id=1)
            return
        if self.get_position("AAA") != 0 and current_ts >= pd.Timestamp("2020-02-20"):
            self.order("AAA", -1, trade_id=1)


def build_stress_pricing_data_df() -> pd.DataFrame:
    calendar_idx = pd.DatetimeIndex(
        ["2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21"]
    )
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): [10.0, 10.0, 8.0, 12.0],
            ("AAA", "High"): [10.5, 10.5, 8.5, 12.5],
            ("AAA", "Low"): [9.5, 9.5, 7.5, 11.5],
            ("AAA", "Close"): [10.0, 10.0, 8.0, 12.0],
            ("$SPX", "Open"): [100.0, 100.0, 90.0, 95.0],
            ("$SPX", "High"): [101.0, 101.0, 91.0, 96.0],
            ("$SPX", "Low"): [99.0, 99.0, 89.0, 94.0],
            ("$SPX", "Close"): [100.0, 100.0, 90.0, 95.0],
        },
        index=calendar_idx,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def load_buy_hold_context_dict() -> dict[str, object]:
    pricing_data_df = build_stress_pricing_data_df()
    return {
        "strategy_name_str": "ToyStressBuyHold",
        "capital_base_float": 100.0,
        "benchmark_list": ["$SPX"],
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pricing_data_df.index,
    }


def build_buy_hold_strategy_obj(context_dict: dict[str, object]) -> Strategy:
    return BuyAndHoldOneShareStrategy(
        name=str(context_dict["strategy_name_str"]),
        benchmarks=list(context_dict["benchmark_list"]),
        capital_base=float(context_dict["capital_base_float"]),
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )


def build_event_exit_strategy_obj(context_dict: dict[str, object]) -> Strategy:
    return SellOnEventStartStrategy(
        name=str(context_dict["strategy_name_str"]),
        benchmarks=list(context_dict["benchmark_list"]),
        capital_base=float(context_dict["capital_base_float"]),
        slippage=0.0,
        commission_per_share=0.10,
        commission_minimum=0.0,
    )


BUY_HOLD_SPEC = CrisisStrategySpec(
    strategy_key_str="toy_stress_buy_hold",
    load_context_fn=load_buy_hold_context_dict,
    build_strategy_fn=build_buy_hold_strategy_obj,
)

EVENT_EXIT_SPEC = CrisisStrategySpec(
    strategy_key_str="toy_stress_event_exit",
    load_context_fn=load_buy_hold_context_dict,
    build_strategy_fn=build_event_exit_strategy_obj,
)


class StressTestAnalyzerTests(unittest.TestCase):
    def test_resolve_stress_launch_window_uses_pre_event_trading_bars(self):
        calendar_idx = pd.DatetimeIndex(
            ["2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21"]
        )

        launch_ts, event_entry_ts, skip_reason_str = resolve_stress_launch_window(
            event_start_ts=pd.Timestamp("2020-02-20"),
            calendar_idx=calendar_idx,
            launch_offset_int=1,
        )

        self.assertEqual(skip_reason_str, "")
        self.assertEqual(launch_ts, pd.Timestamp("2020-02-19"))
        self.assertEqual(event_entry_ts, pd.Timestamp("2020-02-19"))

    def test_stress_test_launches_before_event_and_enters_with_position(self):
        stress_result_obj = run_stress_test_suite(
            crisis_period_list=[
                CrisisPeriodConfig(
                    crisis_name_str="toy_crisis",
                    start_date_str="2020-02-20",
                    end_date_str="2020-02-21",
                )
            ],
            launch_offset_tuple=(1,),
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )

        metric_row_ser = stress_result_obj.stress_metric_df.iloc[0]
        entry_position_df = stress_result_obj.stress_entry_position_df
        aaa_position_ser = entry_position_df[entry_position_df["asset_str"] == "AAA"].iloc[0]

        self.assertEqual(metric_row_ser["launch_ts"], pd.Timestamp("2020-02-19"))
        self.assertEqual(metric_row_ser["event_entry_ts"], pd.Timestamp("2020-02-19"))
        self.assertAlmostEqual(float(metric_row_ser["event_return_pct_float"]), 2.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser["benchmark_event_return_pct_float"]), -5.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser["relative_event_return_pct_float"]), 7.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser["event_max_drawdown_pct_float"]), -2.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser["worst_event_day_pct_float"]), -2.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser["first_event_day_return_pct_float"]), -2.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser["entry_to_trough_pct_float"]), -2.0, places=12)
        self.assertEqual(int(metric_row_ser["time_to_trough_day_count_int"]), 0)
        self.assertEqual(bool(metric_row_ser["recovered_by_event_end_bool"]), True)
        self.assertEqual(int(metric_row_ser["time_to_recover_day_count_int"]), 1)
        self.assertAlmostEqual(float(metric_row_ser["gross_exposure_float"]), 0.10, places=12)
        self.assertAlmostEqual(float(metric_row_ser["cash_weight_float"]), 0.90, places=12)
        self.assertAlmostEqual(float(metric_row_ser["entry_top1_weight_float"]), 0.10, places=12)
        self.assertAlmostEqual(float(metric_row_ser["entry_top5_weight_float"]), 0.10, places=12)
        self.assertAlmostEqual(float(metric_row_ser["event_turnover_float"]), 0.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser["event_commission_pct_float"]), 0.0, places=12)
        expected_volatility_float = float(
            pd.Series([-0.02, 102.0 / 98.0 - 1.0]).std(ddof=1) * math.sqrt(252.0) * 100.0
        )
        self.assertAlmostEqual(
            float(metric_row_ser["event_volatility_ann_pct_float"]),
            expected_volatility_float,
            places=12,
        )
        self.assertAlmostEqual(float(metric_row_ser["event_down_capture_float"]), 0.20, places=12)
        self.assertAlmostEqual(float(aaa_position_ser["entry_close_price_float"]), 10.0, places=12)
        self.assertAlmostEqual(float(aaa_position_ser["position_weight_float"]), 0.10, places=12)

    def test_event_trade_metrics_include_turnover_and_commission(self):
        stress_result_obj = run_stress_test_suite(
            crisis_period_list=[
                CrisisPeriodConfig(
                    crisis_name_str="toy_crisis",
                    start_date_str="2020-02-20",
                    end_date_str="2020-02-21",
                )
            ],
            launch_offset_tuple=(1,),
            strategy_spec_obj=EVENT_EXIT_SPEC,
            save_output_bool=False,
        )

        metric_row_ser = stress_result_obj.stress_metric_df.iloc[0]
        event_entry_value_float = 100.0 - 0.10
        event_end_value_float = 100.0 - 0.10 - 2.0 - 0.10

        self.assertAlmostEqual(
            float(metric_row_ser["event_return_pct_float"]),
            (event_end_value_float / event_entry_value_float - 1.0) * 100.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(metric_row_ser["event_turnover_float"]),
            8.0 / event_entry_value_float,
            places=12,
        )
        self.assertAlmostEqual(
            float(metric_row_ser["event_commission_pct_float"]),
            0.10 / event_entry_value_float,
            places=12,
        )
        self.assertAlmostEqual(
            float(metric_row_ser["entry_top1_weight_float"]),
            10.0 / event_entry_value_float,
            places=12,
        )
        self.assertEqual(bool(metric_row_ser["recovered_by_event_end_bool"]), False)
        self.assertTrue(pd.isna(metric_row_ser["time_to_recover_day_count_int"]))

    def test_stress_test_differs_from_fresh_crisis_replay(self):
        crisis_period_list = [
            CrisisPeriodConfig(
                crisis_name_str="toy_crisis",
                start_date_str="2020-02-20",
                end_date_str="2020-02-21",
            )
        ]
        stress_result_obj = run_stress_test_suite(
            crisis_period_list=crisis_period_list,
            launch_offset_tuple=(1,),
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )
        crisis_replay_result = run_crisis_replay_suite(
            crisis_period_list=crisis_period_list,
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )

        stress_event_return_float = float(
            stress_result_obj.stress_metric_df.iloc[0]["event_return_pct_float"]
        )
        fresh_replay_return_float = float(
            crisis_replay_result.crisis_metric_df.iloc[0]["strategy_return_pct_float"]
        )

        self.assertAlmostEqual(stress_event_return_float, 2.0, places=12)
        self.assertAlmostEqual(fresh_replay_return_float, 4.0, places=12)
        self.assertNotEqual(stress_event_return_float, fresh_replay_return_float)

    def test_insufficient_pre_event_history_skips_scenario(self):
        stress_result_obj = run_stress_test_suite(
            crisis_period_list=[
                CrisisPeriodConfig(
                    crisis_name_str="toy_crisis",
                    start_date_str="2020-02-20",
                    end_date_str="2020-02-21",
                )
            ],
            launch_offset_tuple=(3,),
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )

        self.assertEqual(len(stress_result_obj.stress_metric_df), 0)
        self.assertEqual(len(stress_result_obj.stress_path_df), 0)

    def test_save_stress_test_results_writes_expected_artifacts_and_html(self):
        stress_result_obj = StressTestAnalyzer(
            strategy_spec_obj=BUY_HOLD_SPEC,
            crisis_period_list=[
                CrisisPeriodConfig(
                    crisis_name_str="toy_crisis",
                    start_date_str="2020-02-20",
                    end_date_str="2020-02-21",
                )
            ],
            launch_offset_tuple=(1,),
            save_output_bool=False,
        ).run()

        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_path = save_stress_test_results(
                stress_result_obj,
                output_dir_str=temp_dir_str,
            )
            relative_output_path = output_path.relative_to(Path(temp_dir_str))

            self.assertEqual(
                relative_output_path.parts[:4],
                ("research", "strategy", "toy_stress_buy_hold", "stress_test"),
            )
            self.assertTrue((output_path / "stress_metrics.csv").exists())
            self.assertTrue((output_path / "stress_paths.csv").exists())
            self.assertTrue((output_path / "stress_entry_positions.csv").exists())
            self.assertTrue((output_path / "stress_transactions.csv").exists())
            self.assertTrue((output_path / "metadata.json").exists())
            self.assertTrue((output_path / "run_info.json").exists())
            self.assertTrue((output_path / "summary.json").exists())
            self.assertTrue((output_path / "report.html").exists())

            run_info_dict = json.loads((output_path / "run_info.json").read_text(encoding="utf-8"))
            summary_dict = json.loads((output_path / "summary.json").read_text(encoding="utf-8"))
            report_html_str = (output_path / "report.html").read_text(encoding="utf-8")

            self.assertEqual(run_info_dict["analysis_type"], "stress_test")
            self.assertEqual(run_info_dict["parameters"]["stress_type"], "historical_pre_crisis_launch")
            self.assertEqual(summary_dict["scenario_count_int"], 1)
            self.assertIn("worst_first_event_day_return_pct_float", summary_dict)
            self.assertIn("max_entry_top1_weight_float", summary_dict)
            self.assertEqual(summary_dict["unrecovered_scenario_count_int"], 0)
            self.assertIn("StressTestAnalyzer Report", report_html_str)
            self.assertIn("Risk Flags", report_html_str)
            self.assertIn("Heatmap Dashboard", report_html_str)
            self.assertIn("Event Return Heatmap", report_html_str)
            self.assertIn("From entering the crisis to event end.", report_html_str)
            self.assertIn("How exposed the strategy was before the event.", report_html_str)
            self.assertIn("Rows are crises, columns are launch offsets", report_html_str)
            self.assertIn("Higher values mean more single-position concentration risk", report_html_str)
            self.assertIn("kpi-meaning", report_html_str)
            self.assertIn("heatmap-meaning", report_html_str)
            self.assertIn("heatmap-label-col", report_html_str)
            self.assertIn("heatmap-value-cell", report_html_str)
            self.assertIn("heatmap-row-label", report_html_str)
            self.assertIn("heatmap-matrix-wrap", report_html_str)
            self.assertNotIn("heatmap-scroll", report_html_str)
            self.assertIn("toy crisis", report_html_str)
            self.assertIn("Stress Matrix", report_html_str)
            self.assertIn("Worst Cases", report_html_str)
            self.assertIn("Event Chapter", report_html_str)
            self.assertIn("Combined Launch-to-End Chart", report_html_str)
            self.assertIn("event-chart-stack", report_html_str)
            self.assertIn("event-chart-panel", report_html_str)
            self.assertNotIn('<div class="chart-grid">', report_html_str)
            self.assertIn("Exposure By Offset", report_html_str)
            self.assertIn("Crisis start marker", report_html_str)
            self.assertIn("Entering Positions", report_html_str)
            self.assertIn("Raw Detail Appendix", report_html_str)
            self.assertIn("Transaction Summary", report_html_str)
            self.assertIn("Historical pre-crisis launch stress", report_html_str)
            # Plain-English verdict leads the report.
            self.assertIn("stress-verdict", report_html_str)
            self.assertIn(">Verdict<", report_html_str)
            self.assertIn("worst entering-event return", report_html_str)
            # Heatmaps carry a color-scale legend.
            self.assertIn("heatmap-legend", report_html_str)
            self.assertIn("legend-swatch", report_html_str)
            self.assertIn("Color scale", report_html_str)
            # CSS palette tokens are actually defined (var(--color-*) refs resolve).
            self.assertIn(":root {", report_html_str)
            self.assertIn("--color-border:", report_html_str)
            self.assertIn("--color-benchmark-dark:", report_html_str)
            # Stress Matrix is split into themed, collapsible groups.
            self.assertIn("appendix-details", report_html_str)
            self.assertIn("stress-matrix-groups", report_html_str)
            self.assertIn("<h4>Returns</h4>", report_html_str)
            self.assertIn("<h4>Risk</h4>", report_html_str)
            self.assertIn("<h4>Entry Exposure</h4>", report_html_str)
            self.assertIn("<h4>Trading</h4>", report_html_str)

    def test_heatmap_scale_is_data_driven_and_not_clipped(self):
        # A -60% drawdown must not be color-clipped to look the same as -35%.
        scale_max_float = _heatmap_scale_max_float(
            pd.DataFrame({"event_max_drawdown_pct_float": [-35.0, -60.0, -5.0]}),
            "event_max_drawdown_pct_float",
        )
        self.assertAlmostEqual(scale_max_float, 60.0, places=9)
        style_35_str = _heatmap_cell_style_str("event_max_drawdown_pct_float", -35.0, scale_max_float)
        style_60_str = _heatmap_cell_style_str("event_max_drawdown_pct_float", -60.0, scale_max_float)
        self.assertNotEqual(style_35_str, style_60_str)

    def test_heatmap_scale_respects_floor_and_nan(self):
        # A calm sample uses the per-metric floor so cells are not all saturated.
        calm_scale_float = _heatmap_scale_max_float(
            pd.DataFrame({"event_max_drawdown_pct_float": [-2.0, -3.0]}),
            "event_max_drawdown_pct_float",
        )
        self.assertAlmostEqual(calm_scale_float, 10.0, places=9)
        # An all-NaN column falls back to the floor instead of raising.
        nan_scale_float = _heatmap_scale_max_float(
            pd.DataFrame({"event_max_drawdown_pct_float": [float("nan")]}),
            "event_max_drawdown_pct_float",
        )
        self.assertAlmostEqual(nan_scale_float, 10.0, places=9)

    def test_verdict_handles_empty_metric_frame(self):
        empty_result_obj = run_stress_test_suite(
            crisis_period_list=[
                CrisisPeriodConfig(
                    crisis_name_str="toy_crisis",
                    start_date_str="2020-02-20",
                    end_date_str="2020-02-21",
                )
            ],
            launch_offset_tuple=(3,),  # insufficient history -> zero scenarios
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )
        self.assertEqual(len(empty_result_obj.stress_metric_df), 0)
        verdict_html_str = _build_verdict_html(empty_result_obj)
        self.assertIn("stress-verdict", verdict_html_str)
        self.assertIn("No stress scenarios were evaluated", verdict_html_str)


if __name__ == "__main__":
    unittest.main()
