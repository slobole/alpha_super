import json

import numpy as np
import pandas as pd
import pytest

from alpha.engine.friction_analysis import (
    CAPACITY_SENSITIVE_BUCKET_STR,
    ADV_FINE_BUCKET_STR,
    ADV_RED_BUCKET_STR,
    ADV_WATCH_BUCKET_STR,
    AUCTION_IMPACT_LOWER_THAN_DEFAULT_LABEL_STR,
    AUCTION_IMPACT_MATERIALLY_OPTIMISTIC_LABEL_STR,
    AUCTION_IMPACT_MILDLY_OPTIMISTIC_LABEL_STR,
    AUCTION_IMPACT_REASONABLE_LABEL_STR,
    AUCTION_NOT_BELIEVABLE_VERDICT_STR,
    AUCTION_PASS_VERDICT_STR,
    AUCTION_STRESSED_VERDICT_STR,
    AUCTION_WATCH_VERDICT_STR,
    COST_CONSERVATIVE_VERDICT_STR,
    COST_OPTIMISTIC_VERDICT_STR,
    COST_PROBLEM_VERDICT_STR,
    COST_REASONABLE_VERDICT_STR,
    DEFAULT_SLIPPAGE_BPS_FLOAT,
    FINE_BUCKET_STR,
    FRICTION_NOT_BELIEVABLE_VERDICT_STR,
    FRICTION_REASONABLE_VERDICT_STR,
    FRICTION_WATCH_VERDICT_STR,
    FRICTION_ORDER_CSV_FILENAME_STR,
    FRICTION_SUMMARY_CSV_FILENAME_STR,
    MOC_AUCTION_FRACTION_FLOAT,
    MOC_AUCTION_IMPACT_LAMBDA_BPS_FLOAT,
    MOO_AUCTION_FRACTION_FLOAT,
    MOO_AUCTION_IMPACT_LAMBDA_BPS_FLOAT,
    RED_BUCKET_STR,
    REPORT_FILENAME_STR,
    RUN_INFO_FILENAME_STR,
    SUMMARY_FILENAME_STR,
    METADATA_FILENAME_STR,
    UNAVAILABLE_BUCKET_STR,
    WATCH_BUCKET_STR,
    FrictionAnalysis,
    FrictionAnalysisResult,
    adv_rule_slippage_bps_float,
    auction_impact_bps_float,
    auction_impact_lambda_for_policy_float,
    classify_auction_impact_delta_label_str,
    classify_auction_verdict_str,
    classify_adv_participation_bucket,
    classify_auction_participation_bucket,
    classify_cost_verdict_str,
    combine_friction_verdict_str,
)
from alpha.engine.strategy import Strategy


class ToyStrategy(Strategy):
    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


def _pricing_data_df(volume_float_list: list[float], close_float: float = 10.0) -> pd.DataFrame:
    date_idx = pd.date_range("2024-01-02", periods=len(volume_float_list), freq="B")
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Close"): [close_float] * len(volume_float_list),
            ("AAA", "Volume"): volume_float_list,
        },
        index=date_idx,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def _multi_asset_pricing_data_df(asset_list: list[str], bar_count_int: int = 21) -> pd.DataFrame:
    date_idx = pd.date_range("2024-01-02", periods=bar_count_int, freq="B")
    column_data_dict = {}
    for asset_str in asset_list:
        column_data_dict[(asset_str, "Close")] = [10.0] * bar_count_int
        column_data_dict[(asset_str, "Volume")] = [1_000.0] * bar_count_int
    pricing_data_df = pd.DataFrame(column_data_dict, index=date_idx)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def _missing_volume_pricing_data_df(bar_count_int: int = 21) -> pd.DataFrame:
    date_idx = pd.date_range("2024-01-02", periods=bar_count_int, freq="B")
    pricing_data_df = pd.DataFrame(
        {("AAA", "Close"): [10.0] * bar_count_int},
        index=date_idx,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def _toy_strategy_obj(bar_ts: pd.Timestamp, total_value_float: float = 100.0) -> ToyStrategy:
    strategy_obj = ToyStrategy(
        name="toy_strategy",
        benchmarks=[],
        capital_base=100_000,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy_obj._transactions = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "bar": bar_ts,
                "asset": "AAA",
                "amount": total_value_float / 10.0,
                "price": 10.0,
                "total_value": total_value_float,
                "order_id": 7,
                "commission": 0.0,
            }
        ]
    )
    return strategy_obj


def _attach_total_value_curve(strategy_obj: ToyStrategy, date_idx: pd.DatetimeIndex) -> None:
    total_value_ser = pd.Series(
        np.linspace(1_000.0, 1_200.0, len(date_idx)),
        index=date_idx,
        dtype=float,
    )
    strategy_obj.results = pd.DataFrame(
        {
            "total_value": total_value_ser,
            "portfolio_value": total_value_ser,
            "daily_returns": total_value_ser.pct_change(fill_method=None).fillna(0.0),
        }
    )


def _toy_strategy_with_transactions_obj(transaction_dict_list: list[dict[str, object]]) -> ToyStrategy:
    strategy_obj = ToyStrategy(
        name="toy_strategy",
        benchmarks=[],
        capital_base=100_000,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy_obj._transactions = pd.DataFrame(transaction_dict_list)
    return strategy_obj


def test_lagged_adv_uses_shifted_volume_not_same_day_volume():
    volume_float_list = ([100.0] * 10) + ([1_000.0] * 10) + [1_000.0]
    pricing_data_df = _pricing_data_df(volume_float_list)
    bar_ts = pricing_data_df.index[20]
    strategy_obj = _toy_strategy_obj(bar_ts=bar_ts, total_value_float=55.0)

    friction_result_obj = FrictionAnalysis(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        save_output_bool=False,
    ).run()

    order_ser = friction_result_obj.friction_order_df.iloc[0]
    assert order_ser["adv20_dollar_lagged_float"] == pytest.approx(5_500.0)
    assert order_ser["auction_proxy_dollar_float"] == pytest.approx(110.0)
    assert order_ser["auction_participation_float"] == pytest.approx(0.5)
    assert order_ser["adv_participation_float"] == pytest.approx(0.01)
    assert order_ser["adv_rule_slippage_bps_float"] == pytest.approx(1.0)
    assert order_ser["default_slippage_bps_float"] == pytest.approx(DEFAULT_SLIPPAGE_BPS_FLOAT)


def test_moo_and_moc_use_nominal_auction_fractions():
    pricing_data_df = _pricing_data_df([1_000.0] * 21)
    bar_ts = pricing_data_df.index[20]
    strategy_obj = _toy_strategy_obj(bar_ts=bar_ts, total_value_float=100.0)

    moo_result_obj = FrictionAnalysis(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        execution_policy_str="MOO",
        save_output_bool=False,
    ).run()
    moc_result_obj = FrictionAnalysis(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        execution_policy_str="MOC",
        save_output_bool=False,
    ).run()

    assert moo_result_obj.auction_fraction_float == MOO_AUCTION_FRACTION_FLOAT
    assert moc_result_obj.auction_fraction_float == MOC_AUCTION_FRACTION_FLOAT
    assert moo_result_obj.friction_order_df.iloc[0]["auction_proxy_dollar_float"] == pytest.approx(200.0)
    assert moc_result_obj.friction_order_df.iloc[0]["auction_proxy_dollar_float"] == pytest.approx(1_000.0)


def test_auction_impact_formula_policy_lambdas_and_delta_labels():
    assert auction_impact_bps_float(0.01, 35.0) == pytest.approx(35.0)
    assert auction_impact_bps_float(0.0025, 35.0) == pytest.approx(17.5)
    assert np.isnan(auction_impact_bps_float(np.nan, 35.0))

    assert auction_impact_lambda_for_policy_float("MOO") == pytest.approx(
        MOO_AUCTION_IMPACT_LAMBDA_BPS_FLOAT
    )
    assert auction_impact_lambda_for_policy_float("MOC") == pytest.approx(
        MOC_AUCTION_IMPACT_LAMBDA_BPS_FLOAT
    )

    assert (
        classify_auction_impact_delta_label_str(-0.51)
        == AUCTION_IMPACT_LOWER_THAN_DEFAULT_LABEL_STR
    )
    assert (
        classify_auction_impact_delta_label_str(-0.50)
        == AUCTION_IMPACT_REASONABLE_LABEL_STR
    )
    assert (
        classify_auction_impact_delta_label_str(0.50)
        == AUCTION_IMPACT_REASONABLE_LABEL_STR
    )
    assert (
        classify_auction_impact_delta_label_str(np.nextafter(0.50, 1.0))
        == AUCTION_IMPACT_MILDLY_OPTIMISTIC_LABEL_STR
    )
    assert (
        classify_auction_impact_delta_label_str(2.50)
        == AUCTION_IMPACT_MILDLY_OPTIMISTIC_LABEL_STR
    )
    assert (
        classify_auction_impact_delta_label_str(np.nextafter(2.50, 3.0))
        == AUCTION_IMPACT_MATERIALLY_OPTIMISTIC_LABEL_STR
    )


def test_missing_volume_fails_loud():
    pricing_data_df = _missing_volume_pricing_data_df()
    bar_ts = pricing_data_df.index[20]
    strategy_obj = _toy_strategy_obj(bar_ts=bar_ts)

    with pytest.raises(ValueError, match="Volume"):
        FrictionAnalysis(
            strategy_obj=strategy_obj,
            pricing_data_df=pricing_data_df,
            save_output_bool=False,
        ).run()


def test_orders_with_insufficient_history_are_unavailable_not_safe():
    pricing_data_df = _pricing_data_df([1_000.0] * 10)
    bar_ts = pricing_data_df.index[9]
    strategy_obj = _toy_strategy_obj(bar_ts=bar_ts, total_value_float=1.0)

    friction_result_obj = FrictionAnalysis(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        save_output_bool=False,
    ).run()

    order_ser = friction_result_obj.friction_order_df.iloc[0]
    assert order_ser["capacity_bucket_str"] == UNAVAILABLE_BUCKET_STR
    assert bool(order_ser["assessed_bool"]) is False
    assert order_ser["unavailable_reason_str"] == "insufficient_lagged_adv_history"
    assert np.isnan(order_ser["auction_impact_bps_float"])
    assert np.isnan(order_ser["auction_impact_dollar_float"])


def test_bucket_threshold_boundaries_are_inclusive_on_upper_bounds():
    assert classify_auction_participation_bucket(0.01) == FINE_BUCKET_STR
    assert classify_auction_participation_bucket(np.nextafter(0.01, 1.0)) == WATCH_BUCKET_STR
    assert classify_auction_participation_bucket(0.025) == WATCH_BUCKET_STR
    assert classify_auction_participation_bucket(np.nextafter(0.025, 1.0)) == CAPACITY_SENSITIVE_BUCKET_STR
    assert classify_auction_participation_bucket(0.05) == CAPACITY_SENSITIVE_BUCKET_STR
    assert classify_auction_participation_bucket(np.nextafter(0.05, 1.0)) == RED_BUCKET_STR
    assert classify_auction_participation_bucket(np.nan) == UNAVAILABLE_BUCKET_STR


def test_adv_frame_rule_of_thumb_and_default_delta():
    assert classify_adv_participation_bucket(np.nextafter(0.05, 0.0)) == ADV_FINE_BUCKET_STR
    assert classify_adv_participation_bucket(0.05) == ADV_WATCH_BUCKET_STR
    assert classify_adv_participation_bucket(np.nextafter(0.10, 0.0)) == ADV_WATCH_BUCKET_STR
    assert classify_adv_participation_bucket(0.10) == ADV_RED_BUCKET_STR
    assert adv_rule_slippage_bps_float(0.01) == pytest.approx(1.0)
    assert adv_rule_slippage_bps_float(0.06) == pytest.approx(5.0)
    assert adv_rule_slippage_bps_float(0.10) == pytest.approx(10.0)

    pricing_data_df = _pricing_data_df([1_000.0] * 21)
    bar_ts = pricing_data_df.index[20]
    strategy_obj = _toy_strategy_obj(bar_ts=bar_ts, total_value_float=100.0)

    friction_result_obj = FrictionAnalysis(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        save_output_bool=False,
    ).run()

    order_ser = friction_result_obj.friction_order_df.iloc[0]
    summary_dict = friction_result_obj.summary_dict
    assert order_ser["adv_participation_float"] == pytest.approx(0.01)
    assert order_ser["adv_rule_slippage_dollar_float"] == pytest.approx(0.01)
    assert order_ser["default_slippage_dollar_float"] == pytest.approx(0.025)
    assert order_ser["adv_rule_minus_default_dollar_float"] == pytest.approx(-0.015)
    assert order_ser["auction_impact_lambda_bps_float"] == pytest.approx(
        MOO_AUCTION_IMPACT_LAMBDA_BPS_FLOAT
    )
    assert order_ser["auction_impact_bps_float"] == pytest.approx(35.0)
    assert order_ser["auction_impact_dollar_float"] == pytest.approx(0.35)
    assert order_ser["auction_impact_minus_default_bps_float"] == pytest.approx(32.5)
    assert order_ser["auction_impact_minus_default_dollar_float"] == pytest.approx(0.325)
    assert summary_dict["red_order_share_float"] == pytest.approx(1.0)
    assert summary_dict["adv_rule_slippage_blended_bps_float"] == pytest.approx(1.0)
    assert summary_dict["adv_rule_minus_default_bps_float"] == pytest.approx(-1.5)
    assert summary_dict["auction_impact_blended_bps_float"] == pytest.approx(35.0)
    assert summary_dict["auction_impact_minus_default_bps_float"] == pytest.approx(32.5)
    assert summary_dict["auction_impact_p50_bps_float"] == pytest.approx(35.0)
    assert summary_dict["auction_impact_p95_bps_float"] == pytest.approx(35.0)
    assert summary_dict["auction_impact_max_bps_float"] == pytest.approx(35.0)
    assert (
        summary_dict["auction_impact_delta_label_str"]
        == AUCTION_IMPACT_MATERIALLY_OPTIMISTIC_LABEL_STR
    )
    assert "current 2.5 bps is materially optimistic" in summary_dict[
        "auction_impact_interpretation_str"
    ]


def test_auction_impact_estimated_ann_return_and_sharpe_overlay():
    pricing_data_df = _pricing_data_df([1_000.0] * 21)
    bar_ts = pricing_data_df.index[20]
    strategy_obj = _toy_strategy_obj(bar_ts=bar_ts, total_value_float=100.0)
    _attach_total_value_curve(strategy_obj, pricing_data_df.index)

    friction_result_obj = FrictionAnalysis(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        save_output_bool=False,
    ).run()

    summary_dict = friction_result_obj.summary_dict
    current_total_value_ser = strategy_obj.results["total_value"].astype(float)
    extra_drag_float = 0.325
    estimated_total_value_ser = current_total_value_ser.copy()
    estimated_total_value_ser.iloc[-1] = estimated_total_value_ser.iloc[-1] - extra_drag_float
    current_daily_return_ser = current_total_value_ser.pct_change(fill_method=None).dropna()
    estimated_daily_return_ser = estimated_total_value_ser.pct_change(fill_method=None).dropna()

    expected_current_annual_return_float = (
        (current_total_value_ser.iloc[-1] / current_total_value_ser.iloc[0])
        ** (252.0 / len(current_total_value_ser))
        - 1.0
    )
    expected_estimated_annual_return_float = (
        (estimated_total_value_ser.iloc[-1] / estimated_total_value_ser.iloc[0])
        ** (252.0 / len(estimated_total_value_ser))
        - 1.0
    )
    expected_current_sharpe_float = (
        current_daily_return_ser.mean() / current_daily_return_ser.std() * np.sqrt(252.0)
    )
    expected_estimated_sharpe_float = (
        estimated_daily_return_ser.mean()
        / estimated_daily_return_ser.std()
        * np.sqrt(252.0)
    )

    assert summary_dict["auction_impact_estimated_cumulative_extra_drag_float"] == pytest.approx(extra_drag_float)
    assert summary_dict["current_annual_return_float"] == pytest.approx(expected_current_annual_return_float)
    assert summary_dict["auction_impact_estimated_annual_return_float"] == pytest.approx(expected_estimated_annual_return_float)
    assert summary_dict["auction_impact_estimated_annual_return_delta_float"] == pytest.approx(
        expected_estimated_annual_return_float - expected_current_annual_return_float
    )
    assert summary_dict["current_sharpe_float"] == pytest.approx(expected_current_sharpe_float)
    assert summary_dict["auction_impact_estimated_sharpe_float"] == pytest.approx(expected_estimated_sharpe_float)
    assert summary_dict["auction_impact_estimated_sharpe_delta_float"] == pytest.approx(
        expected_estimated_sharpe_float - expected_current_sharpe_float
    )
    assert "No rescale, no retrade, no fill changes" in summary_dict[
        "auction_impact_performance_note_str"
    ]


def test_cost_auction_and_combined_verdict_thresholds():
    assert classify_cost_verdict_str(-0.50) == COST_CONSERVATIVE_VERDICT_STR
    assert classify_cost_verdict_str(np.nextafter(-0.50, 0.0)) == COST_REASONABLE_VERDICT_STR
    assert classify_cost_verdict_str(0.50) == COST_REASONABLE_VERDICT_STR
    assert classify_cost_verdict_str(np.nextafter(0.50, 1.0)) == COST_OPTIMISTIC_VERDICT_STR
    assert classify_cost_verdict_str(2.50) == COST_OPTIMISTIC_VERDICT_STR
    assert classify_cost_verdict_str(np.nextafter(2.50, 3.0)) == COST_PROBLEM_VERDICT_STR

    assert classify_auction_verdict_str(0.05, 0.15, 0.05) == AUCTION_PASS_VERDICT_STR
    assert classify_auction_verdict_str(0.10, 0.20, 0.08) == AUCTION_WATCH_VERDICT_STR
    assert classify_auction_verdict_str(np.nextafter(0.20, 1.0), 0.20, 0.08) == AUCTION_STRESSED_VERDICT_STR
    assert classify_auction_verdict_str(0.10, np.nextafter(0.40, 1.0), 0.08) == AUCTION_STRESSED_VERDICT_STR
    assert classify_auction_verdict_str(0.10, 0.20, np.nextafter(0.10, 1.0)) == AUCTION_STRESSED_VERDICT_STR
    assert classify_auction_verdict_str(np.nextafter(0.50, 1.0), 0.20, 0.08) == AUCTION_NOT_BELIEVABLE_VERDICT_STR
    assert classify_auction_verdict_str(0.10, 0.20, np.nextafter(0.25, 1.0)) == AUCTION_NOT_BELIEVABLE_VERDICT_STR

    assert combine_friction_verdict_str(
        COST_CONSERVATIVE_VERDICT_STR,
        AUCTION_STRESSED_VERDICT_STR,
    ) == FRICTION_WATCH_VERDICT_STR
    assert combine_friction_verdict_str(
        COST_REASONABLE_VERDICT_STR,
        AUCTION_PASS_VERDICT_STR,
    ) == FRICTION_REASONABLE_VERDICT_STR
    assert combine_friction_verdict_str(
        COST_PROBLEM_VERDICT_STR,
        AUCTION_PASS_VERDICT_STR,
    ) == FRICTION_NOT_BELIEVABLE_VERDICT_STR


def test_concentration_scale_limit_and_verdict_summary_metrics():
    pricing_data_df = _multi_asset_pricing_data_df(["AAA", "BBB"])
    bar_ts = pricing_data_df.index[20]
    transaction_dict_list = [
        {
            "trade_id": 1,
            "bar": bar_ts,
            "asset": "AAA",
            "amount": 10.0,
            "price": 10.0,
            "total_value": 100.0,
            "order_id": 1,
            "commission": 0.0,
        },
        {
            "trade_id": 2,
            "bar": bar_ts,
            "asset": "AAA",
            "amount": 10.0,
            "price": 10.0,
            "total_value": 100.0,
            "order_id": 2,
            "commission": 0.0,
        },
        {
            "trade_id": 3,
            "bar": bar_ts,
            "asset": "BBB",
            "amount": 10.0,
            "price": 10.0,
            "total_value": 100.0,
            "order_id": 3,
            "commission": 0.0,
        },
    ]
    strategy_obj = _toy_strategy_with_transactions_obj(transaction_dict_list)

    friction_result_obj = FrictionAnalysis(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        save_output_bool=False,
    ).run()

    summary_dict = friction_result_obj.summary_dict
    assert summary_dict["cost_verdict_str"] == COST_CONSERVATIVE_VERDICT_STR
    assert summary_dict["auction_verdict_str"] == AUCTION_NOT_BELIEVABLE_VERDICT_STR
    assert summary_dict["friction_verdict_str"] == FRICTION_NOT_BELIEVABLE_VERDICT_STR
    assert summary_dict["top_red_asset_str"] == "AAA"
    assert summary_dict["top_red_asset_red_notional_share_float"] == pytest.approx(2 / 3)
    assert summary_dict["top5_red_asset_red_notional_share_float"] == pytest.approx(1.0)
    assert summary_dict["worst_year_by_red_notional_share_int"] == 2024
    assert summary_dict["worst_year_red_notional_share_float"] == pytest.approx(1.0)
    assert summary_dict["scale_to_p95_auction_red_threshold_float"] == pytest.approx(0.10)
    assert summary_dict["scale_to_max_auction_proxy_float"] == pytest.approx(2.0)
    assert summary_dict["scale_to_p95_adv_5pct_float"] == pytest.approx(5.0)
    assert "combined verdict is Not Believable" in summary_dict["verdict_explanation_str"]


def test_save_output_writes_csv_json_and_report_html(tmp_path):
    pricing_data_df = _pricing_data_df([1_000.0] * 21)
    bar_ts = pricing_data_df.index[20]
    strategy_obj = _toy_strategy_obj(bar_ts=bar_ts, total_value_float=100.0)

    friction_result_obj = FrictionAnalysis(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        output_dir_str=str(tmp_path),
        save_output_bool=True,
    ).run()

    output_dir_path = friction_result_obj.output_dir_path
    assert output_dir_path is not None
    assert "research" in output_dir_path.parts
    assert "strategy" in output_dir_path.parts
    assert "friction_analysis" in output_dir_path.parts
    assert (output_dir_path / FRICTION_ORDER_CSV_FILENAME_STR).exists()
    assert (output_dir_path / FRICTION_SUMMARY_CSV_FILENAME_STR).exists()
    assert (output_dir_path / SUMMARY_FILENAME_STR).exists()
    assert (output_dir_path / RUN_INFO_FILENAME_STR).exists()
    assert (output_dir_path / METADATA_FILENAME_STR).exists()
    assert (output_dir_path / REPORT_FILENAME_STR).exists()

    summary_dict = json.loads((output_dir_path / SUMMARY_FILENAME_STR).read_text(encoding="utf-8"))
    saved_order_df = pd.read_csv(output_dir_path / FRICTION_ORDER_CSV_FILENAME_STR)
    report_html_str = (output_dir_path / REPORT_FILENAME_STR).read_text(encoding="utf-8")
    assert summary_dict["analysis_type_str"] == "friction_analysis"
    assert "auction_impact_bps_float" in saved_order_df.columns
    assert "auction_impact_dollar_float" in saved_order_df.columns
    assert "auction_impact_blended_bps_float" in summary_dict
    assert "auction_impact_minus_default_dollar_float" in summary_dict
    assert "auction_impact_estimated_annual_return_float" in summary_dict
    assert "auction_impact_estimated_sharpe_float" in summary_dict
    assert "FrictionAnalysis V1" in report_html_str
    assert "auction-liquidity proxy, not observed auction volume" in report_html_str
    assert "Auction Impact" in report_html_str
    assert "Extra drag vs default" in report_html_str
    assert "Estimated ann. return" in report_html_str
    assert "Estimated Sharpe" in report_html_str
    assert "current 2.5 bps is materially optimistic" in report_html_str
    assert "Estimated adverse auction impact, not observed auction slippage" in report_html_str
    assert "Estimated Impact Details" in report_html_str
    assert "ADV-rule slippage cost" in report_html_str
    assert "Friction Verdict" in report_html_str
    assert "Cost Verdict" in report_html_str
    assert "Auction Verdict" in report_html_str
    assert "combined verdict is" in report_html_str


def test_dv2_hook_returns_result_and_writes_under_friction_analysis(tmp_path, monkeypatch):
    from strategies.dv2 import strategy_mr_dv2 as dv2_module

    pricing_data_df = _pricing_data_df([1_000.0] * 21)
    bar_ts = pricing_data_df.index[20]
    strategy_obj = _toy_strategy_obj(bar_ts=bar_ts, total_value_float=100.0)
    strategy_obj.name = "strategy_mr_dv2"

    def fake_build_friction_analysis_inputs(**kwargs):
        return {
            "strategy_obj": strategy_obj,
            "pricing_data_df": pricing_data_df,
            "execution_policy_str": "MOO",
        }

    monkeypatch.setattr(
        dv2_module,
        "build_friction_analysis_inputs",
        fake_build_friction_analysis_inputs,
    )

    friction_result_obj = dv2_module.run_friction_analysis(
        save_results_bool=True,
        output_dir_str=str(tmp_path),
    )

    assert isinstance(friction_result_obj, FrictionAnalysisResult)
    assert friction_result_obj.strategy_name_str == "strategy_mr_dv2"
    assert friction_result_obj.output_dir_path is not None
    assert "friction_analysis" in friction_result_obj.output_dir_path.parts
