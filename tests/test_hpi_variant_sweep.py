import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import talib


TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from scripts.research.run_hpi_variant_sweep import (
    ADV_63_FIELD_STR,
    DECLARED_VARIANT_COUNT_INT,
    ENTRY_HORIZON_VOTE_STR,
    EXIT_IBS_STR,
    EXIT_RSI_STR,
    HPI_2D_FIELD_STR,
    HPI_5D_FIELD_STR,
    LIQUIDITY_FIXED_STR,
    LIQUIDITY_RELATIVE_STR,
    NATR_10_FIELD_STR,
    NATR_20_FIELD_STR,
    NATR_ENSEMBLE_FIELD_STR,
    NATR_RANK_FIELD_STR,
    RAW_PRICE_FIELD_STR,
    RANK_NATR14_STR,
    RANK_NATR_ENSEMBLE_STR,
    STATUS_ROW_DICT_LIST,
    VARIANT_SPEC_TUPLE,
    VariantSpec,
    HPIResearchSweepStrategy,
    build_baseline_comparison_df,
    build_variant_summary_dict,
    capped_entry_order_value_ser,
    compute_hpi_breadth_ser,
    compute_hpi_sweep_signal_data_df,
    parse_args,
    resolve_selected_variant_spec_list,
    slippage_bps_to_float,
    validate_variant_feature_contract,
)
from alpha.engine.backtest import run_daily
from strategies.hpi.stateful_long import (
    HPIStatefulLongStrategy,
    NATR_FIELD_STR,
    TURNOVER_FIELD_STR,
)


def make_sweep_strategy(
    variant_spec_obj: VariantSpec,
) -> HPIResearchSweepStrategy:
    return HPIResearchSweepStrategy(
        variant_spec_obj=variant_spec_obj,
        benchmark_symbol_str="$SPX",
        signal_data_df=pd.DataFrame(),
        capital_base_float=100_000.0,
    )


def test_sweep_strategy_accepts_explicit_slippage():
    strategy_obj = HPIResearchSweepStrategy(
        variant_spec_obj=VariantSpec("slippage", "sp500"),
        benchmark_symbol_str="$SPX",
        signal_data_df=pd.DataFrame(),
        capital_base_float=100_000.0,
        slippage_float=0.002,
    )

    assert strategy_obj._slippage == pytest.approx(0.002)


def test_variant_summary_labels_synthetic_forced_liquidations_truthfully():
    metric_value_dict = {
        "Alpha HAC t-stat": 1.0,
        "Return (Ann.) [%]": 5.0,
        "Volatility (Ann.) [%]": 10.0,
        "Sharpe Ratio": 0.5,
        "Max. Drawdown [%]": -12.0,
        "MAR Ratio": 0.4,
        "Exposure Time [%]": 70.0,
        "Turnover (Ann.) [%]": 80.0,
        "Cost Drag (Ann.) [%]": 0.2,
        "Final [$]": 105_000.0,
        "Alpha (Ann.) [%]": 1.0,
    }
    summary_df = pd.DataFrame(
        {"Strategy": pd.Series(metric_value_dict, dtype=float)}
    )
    transaction_df = pd.DataFrame({"order_id": [-1, 3, -1]})
    strategy_obj = SimpleNamespace(
        summary=summary_df,
        results=pd.DataFrame(
            index=pd.to_datetime(["2024-01-02", "2024-01-03"])
        ),
        _trades=pd.DataFrame(index=range(2)),
        _accounting_policy_dict={},
        get_transactions=lambda: transaction_df,
    )

    summary_dict = build_variant_summary_dict(
        strategy_obj,
        VariantSpec("baseline", "sp500"),
        multiple_test_count_int=1,
    )

    assert summary_dict["synthetic_forced_liquidation_count_int"] == 2
    assert "terminal_liquidation_count_int" not in summary_dict


def test_slippage_cli_default_and_bps_conversion(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_hpi_variant_sweep.py"])

    assert parse_args().slippage_bps == pytest.approx(2.5)
    assert slippage_bps_to_float(2.5) == pytest.approx(0.00025)
    assert slippage_bps_to_float(20.0) == pytest.approx(0.002)


@pytest.mark.parametrize(
    "slippage_bps_float",
    [-1.0, float("nan"), float("inf"), 10_000.0],
)
def test_slippage_bps_conversion_rejects_invalid_values(
    slippage_bps_float: float,
):
    with pytest.raises(ValueError, match="finite"):
        slippage_bps_to_float(slippage_bps_float)


def make_close_row_ser(
    row_value_dict: dict[tuple[str, str], float],
) -> pd.Series:
    close_row_ser = pd.Series(row_value_dict, dtype=float)
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
    return close_row_ser


def eligible_field_dict(
    symbol_str: str,
    *,
    turnover_float: float = 10.0,
    natr_float: float = 4.0,
    adv_63_float: float = 10_000_000.0,
    raw_price_float: float = 20.0,
) -> dict[tuple[str, str], float]:
    return {
        (symbol_str, "Close"): 105.0,
        (symbol_str, "sma_200_price_ser"): 100.0,
        (symbol_str, "ibs_value_ser"): 0.05,
        (symbol_str, "rsi2_value_ser"): 20.0,
        (symbol_str, "return_2d_ser"): -0.02,
        (symbol_str, "return_3d_ser"): -0.03,
        (symbol_str, "return_5d_ser"): -0.04,
        (symbol_str, "hpi_2d_ser"): 20.0,
        (symbol_str, "hpi_value_ser"): 20.0,
        (symbol_str, "hpi_5d_ser"): 20.0,
        (symbol_str, TURNOVER_FIELD_STR): turnover_float,
        (symbol_str, NATR_FIELD_STR): natr_float,
        (symbol_str, NATR_10_FIELD_STR): natr_float - 1.0,
        (symbol_str, NATR_20_FIELD_STR): natr_float + 1.0,
        (symbol_str, ADV_63_FIELD_STR): adv_63_float,
        (symbol_str, RAW_PRICE_FIELD_STR): raw_price_float,
    }


def test_sweep_scope_has_20_executed_rows_and_excludes_portfolio_variants():
    variant_key_list = [
        variant_spec_obj.key_str for variant_spec_obj in VARIANT_SPEC_TUPLE
    ]
    threshold_status_dict = next(
        status_row_dict
        for status_row_dict in STATUS_ROW_DICT_LIST
        if status_row_dict["variant_str"] == "hpi_threshold_24_30_36_vote"
    )

    assert len(variant_key_list) == 20
    assert DECLARED_VARIANT_COUNT_INT == 20
    assert len(set(variant_key_list)) == 20
    assert not any("portfolio" in variant_key_str for variant_key_str in variant_key_list)
    assert threshold_status_dict["status_str"] == "not_run_algebraic_duplicate"

    combined_variant_spec_obj = next(
        variant_spec_obj
        for variant_spec_obj in VARIANT_SPEC_TUPLE
        if variant_spec_obj.key_str
        == "sp500_hpi_2_3_5_vote_liquidity_relative"
    )
    assert combined_variant_spec_obj.entry_mode_str == ENTRY_HORIZON_VOTE_STR
    assert combined_variant_spec_obj.liquidity_mode_str == LIQUIDITY_RELATIVE_STR


def test_selected_liquidity_variant_fails_loudly_without_raw_features():
    signal_data_df = pd.DataFrame(
        {
            ("AAA", TURNOVER_FIELD_STR): [10.0],
            ("AAA", "Close"): [100.0],
        },
        index=[pd.Timestamp("2024-03-07")],
    )
    signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)
    liquidity_variant_spec_obj = VariantSpec(
        "liquidity",
        "sp500",
        liquidity_mode_str=LIQUIDITY_FIXED_STR,
    )

    with pytest.raises(RuntimeError, match=ADV_63_FIELD_STR):
        validate_variant_feature_contract(
            signal_data_df,
            [liquidity_variant_spec_obj],
            "2024-01-01",
        )


def test_variant_subset_automatically_includes_its_universe_baseline():
    selected_variant_spec_list = resolve_selected_variant_spec_list(
        {"sp500_liquidity_fixed"}
    )

    assert [
        variant_spec_obj.key_str
        for variant_spec_obj in selected_variant_spec_list
    ] == [
        "sp500_baseline",
        "sp500_liquidity_fixed",
    ]


def test_baseline_comparison_uses_paired_returns_and_frozen_family_correction(
    tmp_path: Path,
):
    run_dir_obj = tmp_path / "sweep"
    result_dir_obj = run_dir_obj / "sp500" / "daily_results"
    result_dir_obj.mkdir(parents=True)
    date_index = pd.bdate_range("2020-01-02", periods=300)
    baseline_return_ser = pd.Series(
        0.0002 + 0.001 * np.sin(np.arange(300, dtype=float)),
        index=date_index,
    )
    variant_return_ser = baseline_return_ser + (
        0.0001 + 0.0002 * np.cos(np.arange(300, dtype=float))
    )
    pd.DataFrame({"daily_returns": baseline_return_ser}).to_csv(
        result_dir_obj / "sp500_baseline.csv"
    )
    pd.DataFrame({"daily_returns": variant_return_ser}).to_csv(
        result_dir_obj / "sp500_variant.csv"
    )
    variant_summary_df = pd.DataFrame(
        [
            {
                "key_str": "sp500_baseline",
                "universe_key_str": "sp500",
                "annual_return_pct": 5.0,
            },
            {
                "key_str": "sp500_variant",
                "universe_key_str": "sp500",
                "annual_return_pct": 6.0,
            },
        ]
    )

    comparison_df = build_baseline_comparison_df(
        run_dir_obj,
        variant_summary_df,
    )
    variant_comparison_ser = comparison_df.set_index("key_str").loc[
        "sp500_variant"
    ]

    assert variant_comparison_ser["cagr_delta_vs_baseline_pct"] == pytest.approx(
        1.0
    )
    assert variant_comparison_ser["mean_return_delta_annual_pct"] > 0.0
    assert variant_comparison_ser["paired_hac_t_float"] > 0.0
    assert variant_comparison_ser[
        "paired_hac_p_bonferroni_float"
    ] == pytest.approx(
        min(
            float(variant_comparison_ser["paired_hac_p_float"])
            * DECLARED_VARIANT_COUNT_INT,
            1.0,
        )
    )


def test_horizon_vote_requires_two_complete_negative_hpi_events():
    strategy_obj = make_sweep_strategy(
        VariantSpec(
            "vote",
            "sp500",
            entry_mode_str=ENTRY_HORIZON_VOTE_STR,
        )
    )
    two_vote_field_dict = eligible_field_dict("TWO", turnover_float=20.0)
    two_vote_field_dict[("TWO", HPI_5D_FIELD_STR)] = 60.0
    one_vote_field_dict = eligible_field_dict("ONE", turnover_float=30.0)
    one_vote_field_dict[("ONE", HPI_2D_FIELD_STR)] = 60.0
    one_vote_field_dict[("ONE", HPI_5D_FIELD_STR)] = 60.0

    opportunity_df = strategy_obj._opportunity_df(
        make_close_row_ser(
            {
                **two_vote_field_dict,
                **one_vote_field_dict,
            }
        ),
        {"TWO", "ONE"},
    )

    assert opportunity_df.index.tolist() == ["TWO"]


@pytest.mark.parametrize(
    ("liquidity_mode_str", "expected_symbol_list"),
    [
        (LIQUIDITY_FIXED_STR, ["HIGH"]),
        (LIQUIDITY_RELATIVE_STR, ["HIGH"]),
    ],
)
def test_liquidity_controls_filter_before_final_ranking(
    liquidity_mode_str: str,
    expected_symbol_list: list[str],
):
    strategy_obj = make_sweep_strategy(
        VariantSpec(
            "liquidity",
            "sp500",
            liquidity_mode_str=liquidity_mode_str,
        )
    )
    high_field_dict = eligible_field_dict(
        "HIGH",
        turnover_float=10.0,
        adv_63_float=20_000_000.0,
    )
    low_field_dict = eligible_field_dict(
        "LOW",
        turnover_float=100.0,
        adv_63_float=2_000_000.0,
    )

    opportunity_df = strategy_obj._opportunity_df(
        make_close_row_ser({**high_field_dict, **low_field_dict}),
        {"HIGH", "LOW"},
    )

    assert opportunity_df.index.tolist() == expected_symbol_list


def test_combined_vote_and_relative_liquidity_requires_both_conditions():
    combined_variant_spec_obj = next(
        variant_spec_obj
        for variant_spec_obj in VARIANT_SPEC_TUPLE
        if variant_spec_obj.key_str
        == "sp500_hpi_2_3_5_vote_liquidity_relative"
    )
    strategy_obj = make_sweep_strategy(combined_variant_spec_obj)
    both_field_dict = eligible_field_dict("BOTH", adv_63_float=40_000_000.0)
    vote_only_field_dict = eligible_field_dict(
        "VOTE_ONLY",
        adv_63_float=20_000_000.0,
    )
    liquidity_only_field_dict = eligible_field_dict(
        "LIQ_ONLY",
        adv_63_float=30_000_000.0,
    )
    neither_field_dict = eligible_field_dict(
        "NEITHER",
        adv_63_float=10_000_000.0,
    )
    for symbol_str, field_dict in (
        ("LIQ_ONLY", liquidity_only_field_dict),
        ("NEITHER", neither_field_dict),
    ):
        field_dict[(symbol_str, HPI_2D_FIELD_STR)] = 60.0
        field_dict[(symbol_str, HPI_5D_FIELD_STR)] = 60.0

    opportunity_df = strategy_obj._opportunity_df(
        make_close_row_ser(
            {
                **both_field_dict,
                **vote_only_field_dict,
                **liquidity_only_field_dict,
                **neither_field_dict,
            }
        ),
        {"BOTH", "VOTE_ONLY", "LIQ_ONLY", "NEITHER"},
    )

    assert opportunity_df.index.tolist() == ["BOTH"]


def test_natr_ensemble_uses_same_date_cross_sectional_percentile_ranks():
    strategy_obj = make_sweep_strategy(
        VariantSpec(
            "natr_ensemble",
            "nasdaq100",
            ranking_mode_str=RANK_NATR_ENSEMBLE_STR,
        )
    )
    first_field_dict = eligible_field_dict("FIRST")
    second_field_dict = eligible_field_dict("SECOND")
    first_field_dict[("FIRST", NATR_10_FIELD_STR)] = 9.0
    first_field_dict[("FIRST", NATR_FIELD_STR)] = 1.0
    first_field_dict[("FIRST", NATR_20_FIELD_STR)] = 9.0
    second_field_dict[("SECOND", NATR_10_FIELD_STR)] = 1.0
    second_field_dict[("SECOND", NATR_FIELD_STR)] = 9.0
    second_field_dict[("SECOND", NATR_20_FIELD_STR)] = 1.0

    opportunity_df = strategy_obj._opportunity_df(
        make_close_row_ser({**first_field_dict, **second_field_dict}),
        {"FIRST", "SECOND"},
    )

    assert opportunity_df.index.tolist() == ["FIRST", "SECOND"]
    assert opportunity_df.loc["FIRST", NATR_ENSEMBLE_FIELD_STR] == pytest.approx(
        5.0 / 6.0
    )
    assert opportunity_df.loc["SECOND", NATR_ENSEMBLE_FIELD_STR] == pytest.approx(
        2.0 / 3.0
    )


@pytest.mark.parametrize(
    ("exit_mode_str", "ibs_value_float", "rsi2_value_float", "expected_bool"),
    [
        (EXIT_IBS_STR, 0.95, 20.0, True),
        (EXIT_IBS_STR, 0.20, 95.0, False),
        (EXIT_RSI_STR, 0.95, 20.0, False),
        (EXIT_RSI_STR, 0.20, 95.0, True),
    ],
)
def test_single_rule_exit_variants_are_isolated(
    exit_mode_str: str,
    ibs_value_float: float,
    rsi2_value_float: float,
    expected_bool: bool,
):
    strategy_obj = make_sweep_strategy(
        VariantSpec(
            "exit",
            "sp500",
            exit_mode_str=exit_mode_str,
        )
    )

    assert strategy_obj._exit_signal_bool(
        ibs_value_float,
        rsi2_value_float,
    ) is expected_bool


def test_capped_sizing_redistributes_budget_without_breaching_position_cap():
    raw_weight_ser = pd.Series(
        {"A": 100.0, "B": 1.0, "C": 1.0},
        dtype=float,
    )

    order_value_ser = capped_entry_order_value_ser(
        raw_weight_ser,
        entry_budget_float=30_000.0,
        previous_total_value_float=100_000.0,
    )

    assert order_value_ser.sum() == pytest.approx(30_000.0)
    assert order_value_ser.max() == pytest.approx(15_000.0)
    assert order_value_ser.loc["B"] == pytest.approx(7_500.0)
    assert order_value_ser.loc["C"] == pytest.approx(7_500.0)


def test_restrict_data_exposes_previous_close_and_current_open_only():
    date_index = pd.bdate_range("2024-03-07", periods=3)
    full_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): [99.0, 100.0, 101.0],
            ("AAA", "Close"): [100.0, 101.0, 999.0],
        },
        index=date_index,
    )
    full_data_df.columns = pd.MultiIndex.from_tuples(full_data_df.columns)
    strategy_obj = make_sweep_strategy(
        VariantSpec(
            "timing",
            "nasdaq100",
            ranking_mode_str=RANK_NATR14_STR,
        )
    )
    strategy_obj.previous_bar = date_index[1]
    strategy_obj.current_bar = date_index[2]

    previous_data_df, close_row_ser, open_price_ser = strategy_obj.restrict_data(
        full_data_df
    )

    assert previous_data_df.index.tolist() == [date_index[1]]
    assert close_row_ser[("AAA", "Close")] == pytest.approx(101.0)
    assert open_price_ser["AAA"] == pytest.approx(101.0)
    assert 999.0 not in previous_data_df.to_numpy()


def test_sweep_baseline_matches_stateful_hpi_order_and_execution_contract():
    date_index = pd.bdate_range("2024-03-07", periods=4)
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): [100.0, 101.0, 102.0, 103.0],
            ("AAA", "High"): [101.0, 102.0, 103.0, 104.0],
            ("AAA", "Low"): [99.0, 100.0, 101.0, 102.0],
            ("AAA", "Close"): [100.0, 101.0, 102.0, 103.0],
            ("AAA", "Dividend"): [0.0, 0.0, 0.0, 0.0],
            ("$SPX", "Open"): [5_000.0, 5_010.0, 5_020.0, 5_030.0],
            ("$SPX", "High"): [5_005.0, 5_015.0, 5_025.0, 5_035.0],
            ("$SPX", "Low"): [4_995.0, 5_005.0, 5_015.0, 5_025.0],
            ("$SPX", "Close"): [5_000.0, 5_010.0, 5_020.0, 5_030.0],
            ("$SPX", "Dividend"): [0.0, 0.0, 0.0, 0.0],
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "AAA": "CAPITALSPECIAL",
        "$SPX": "TOTALRETURN",
    }
    signal_data_df = pricing_data_df.copy()
    signal_field_value_dict = {
        TURNOVER_FIELD_STR: [20.0, 20.0, 20.0, 20.0],
        "return_3d_ser": [-0.03, -0.03, 0.02, 0.02],
        "hpi_value_ser": [20.0, 20.0, 80.0, 80.0],
        "sma_200_price_ser": [90.0, 90.0, 90.0, 90.0],
        "ibs_value_ser": [0.05, 0.95, 0.50, 0.50],
        "rsi2_value_ser": [20.0, 20.0, 20.0, 20.0],
    }
    for field_str, value_list in signal_field_value_dict.items():
        signal_data_df[("AAA", field_str)] = value_list
    signal_data_df = signal_data_df.sort_index(axis=1)
    universe_df = pd.DataFrame({"AAA": 1}, index=date_index)

    baseline_strategy_obj = HPIStatefulLongStrategy(
        name="baseline",
        benchmarks=["$SPX"],
        ranking_field_str=TURNOVER_FIELD_STR,
    )
    baseline_strategy_obj.universe_df = universe_df
    baseline_strategy_obj.compute_signals = (
        lambda _pricing_data_df: signal_data_df
    )
    sweep_strategy_obj = HPIResearchSweepStrategy(
        variant_spec_obj=VariantSpec("sp500_baseline", "sp500"),
        benchmark_symbol_str="$SPX",
        signal_data_df=signal_data_df,
        capital_base_float=100_000.0,
    )
    sweep_strategy_obj.universe_df = universe_df

    run_daily(
        baseline_strategy_obj,
        pricing_data_df,
        date_index,
        show_progress=False,
        show_signal_progress_bool=False,
    )
    run_daily(
        sweep_strategy_obj,
        pricing_data_df,
        date_index,
        show_progress=False,
        show_signal_progress_bool=False,
    )

    comparison_column_list = [
        "trade_id",
        "bar",
        "asset",
        "amount",
        "price",
        "total_value",
        "commission",
    ]
    pd.testing.assert_frame_equal(
        sweep_strategy_obj.get_transactions()[comparison_column_list].reset_index(
            drop=True
        ),
        baseline_strategy_obj.get_transactions()[comparison_column_list].reset_index(
            drop=True
        ),
    )
    pd.testing.assert_series_equal(
        sweep_strategy_obj.results["total_value"],
        baseline_strategy_obj.results["total_value"],
    )


def test_feature_superset_matches_literal_horizons_natr_and_raw_adv():
    date_index = pd.bdate_range("2018-01-02", periods=1_266)
    step_vec = np.arange(len(date_index), dtype=float)
    close_vec = 100.0 + 0.04 * step_vec + 2.0 * np.sin(step_vec * 0.05)
    high_vec = close_vec + 1.0
    low_vec = close_vec - 1.0
    raw_close_vec = close_vec * 1.3
    volume_vec = 1_000_000.0 + step_vec
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): close_vec - 0.1,
            ("AAA", "High"): high_vec,
            ("AAA", "Low"): low_vec,
            ("AAA", "Close"): close_vec,
            ("AAA", "Unadjusted Close"): raw_close_vec,
            ("AAA", "Volume"): volume_vec,
            ("AAA", TURNOVER_FIELD_STR): raw_close_vec * volume_vec,
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

    signal_data_df = compute_hpi_sweep_signal_data_df(pricing_data_df)
    last_date_ts = date_index[-1]

    assert isinstance(signal_data_df.columns, pd.MultiIndex)
    assert signal_data_df.loc[
        last_date_ts,
        ("AAA", "return_2d_ser"),
    ] == pytest.approx(close_vec[-1] / close_vec[-3] - 1.0)
    assert signal_data_df.loc[
        last_date_ts,
        ("AAA", "return_5d_ser"),
    ] == pytest.approx(close_vec[-1] / close_vec[-6] - 1.0)
    assert signal_data_df.loc[
        last_date_ts,
        ("AAA", NATR_10_FIELD_STR),
    ] == pytest.approx(
        talib.NATR(high_vec, low_vec, close_vec, timeperiod=10)[-1]
    )
    assert signal_data_df.loc[
        last_date_ts,
        ("AAA", NATR_20_FIELD_STR),
    ] == pytest.approx(
        talib.NATR(high_vec, low_vec, close_vec, timeperiod=20)[-1]
    )
    assert signal_data_df.loc[
        last_date_ts,
        ("AAA", ADV_63_FIELD_STR),
    ] == pytest.approx(float(np.mean(raw_close_vec[-63:] * volume_vec[-63:])))


def test_breadth_uses_same_date_point_in_time_membership():
    date_index = pd.bdate_range("2024-03-07", periods=2)
    signal_data_df = pd.DataFrame(
        {
            ("AAA", "hpi_value_ser"): [20.0, 20.0],
            ("AAA", "return_3d_ser"): [-0.02, -0.02],
            ("BBB", "hpi_value_ser"): [50.0, 20.0],
            ("BBB", "return_3d_ser"): [-0.02, -0.02],
        },
        index=date_index,
    )
    signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)
    universe_df = pd.DataFrame(
        {
            "AAA": [1, 0],
            "BBB": [1, 1],
        },
        index=date_index,
    )

    breadth_ser = compute_hpi_breadth_ser(signal_data_df, universe_df)

    assert breadth_ser.iloc[0] == pytest.approx(0.5)
    assert breadth_ser.iloc[1] == pytest.approx(1.0)
