import pandas as pd
import pytest

from alpha.bench import catalog
from strategies.momentum import (
    strategy_mo_atr_normalized_ndx_vxn_scaled_paper_b_short as short_module,
)


def _make_strategy() -> short_module.VxnScaledAtrNormalizedNdxPaperBShortStrategy:
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [pd.Timestamp("2024-03-28")]},
        index=pd.to_datetime(["2024-04-01"]),
    )
    vxn_scale_signal_df = pd.DataFrame(
        {"vxn_exposure_scale_float": [0.5]},
        index=pd.to_datetime(["2024-03-28"]),
    )
    strategy_obj = short_module.VxnScaledAtrNormalizedNdxPaperBShortStrategy(
        name="paper_b_short_test",
        benchmarks=["SPY"],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        roc_mode_str=short_module.ROC_MODE_PAPER_B_STR,
        max_positions_int=2,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1], "BBB": [1], "CCC": [1], "OUT": [0]},
        index=[strategy_obj.previous_bar],
    )
    return strategy_obj


def test_short_variant_selects_bottom_eligible_names_and_applies_vxn_scale():
    strategy_obj = _make_strategy()
    close_row_ser = pd.Series(
        {
            ("AAA", "risk_adj_score_ser"): -0.50,
            ("AAA", "stock_trend_pass_bool"): False,
            ("BBB", "risk_adj_score_ser"): -0.20,
            ("BBB", "stock_trend_pass_bool"): False,
            ("CCC", "risk_adj_score_ser"): -0.80,
            ("CCC", "stock_trend_pass_bool"): True,
            ("OUT", "risk_adj_score_ser"): -1.00,
            ("OUT", "stock_trend_pass_bool"): False,
            ("SPY", "regime_pass_bool"): False,
        }
    )
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

    target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser=close_row_ser)

    assert target_weight_ser.index.tolist() == ["AAA", "BBB"]
    assert target_weight_ser.tolist() == pytest.approx([-0.25, -0.25])
    assert target_weight_ser.sum() == pytest.approx(-0.50)


def test_short_variant_is_in_cash_when_spy_is_above_sma200():
    strategy_obj = _make_strategy()
    close_row_ser = pd.Series(
        {
            ("AAA", "risk_adj_score_ser"): -0.50,
            ("AAA", "stock_trend_pass_bool"): False,
            ("SPY", "regime_pass_bool"): True,
        }
    )
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

    target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser=close_row_ser)

    assert target_weight_ser.empty


def test_run_variant_fixes_paper_b_and_disables_signal_audit(monkeypatch):
    observed_run_kwarg_dict = {}
    observed_strategy_dict = {}
    date_index = pd.to_datetime(["2000-01-03", "2000-01-04"])
    pricing_data_df = pd.DataFrame(index=date_index)
    universe_df = pd.DataFrame(index=date_index)
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [date_index[0]]},
        index=[date_index[1]],
    )
    vxn_scale_signal_df = pd.DataFrame(
        {"vxn_exposure_scale_float": [0.5]},
        index=[date_index[0]],
    )

    monkeypatch.setattr(
        short_module,
        "get_vxn_scaled_atr_normalized_ndx_roc_variant_data",
        lambda _config_obj: (
            pricing_data_df,
            universe_df,
            rebalance_schedule_df,
            vxn_scale_signal_df,
        ),
    )

    def fake_run_daily(strategy_obj, _pricing_data_df, **run_kwarg_dict):
        observed_strategy_dict["strategy_obj"] = strategy_obj
        observed_run_kwarg_dict.update(run_kwarg_dict)

    monkeypatch.setattr(short_module, "run_daily", fake_run_daily)

    strategy_obj = short_module.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2000-01-03",
    )

    assert strategy_obj is observed_strategy_dict["strategy_obj"]
    assert strategy_obj.roc_mode_str == short_module.ROC_MODE_PAPER_B_STR
    assert strategy_obj.max_positions_int == 10
    assert observed_run_kwarg_dict["audit_override_bool"] is False


def test_bench_discovers_paper_b_short_as_research_only():
    entry_obj = next(
        entry_obj
        for entry_obj in catalog.list_strategies()
        if entry_obj.stem_str
        == "strategy_mo_atr_normalized_ndx_vxn_scaled_paper_b_short"
    )

    assert entry_obj.has_run_variant_bool is True
    assert entry_obj.is_wired_bool is False
    assert entry_obj.category_str == "momentum"
    assert entry_obj.subcategory_str == "atr_normalized_rotation"
