import numpy as np
import pandas as pd
import pytest

from alpha.bench import catalog
from strategies.momentum import strategy_mo_atr_normalized_ndx_vxn_scaled_paper_b as wrapper
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_roc_variants import (
    ROC_MODE_LAST_12M_STR,
    ROC_MODE_PAPER_B_STR,
    compute_monthly_roc_variant_df,
    get_required_roc_history_month_int,
)


def test_paper_b_numerator_matches_reversal_adjusted_formula():
    decision_date_idx = pd.date_range("2023-01-31", periods=13, freq="BME")
    close_price_vec = np.full(len(decision_date_idx), 100.0)
    close_price_vec[-2] = 70.0
    close_price_vec[-1] = 77.0
    monthly_decision_close_df = pd.DataFrame(
        {"LOSER": close_price_vec},
        index=decision_date_idx,
    )

    paper_b_df = compute_monthly_roc_variant_df(
        monthly_decision_close_df=monthly_decision_close_df,
        roc_mode_str=ROC_MODE_PAPER_B_STR,
    )

    assert paper_b_df.iloc[-1, 0] == pytest.approx(-0.33)
    assert get_required_roc_history_month_int(ROC_MODE_PAPER_B_STR) == 12


def test_paper_b_mode_does_not_change_existing_last_12m_numerator():
    decision_date_idx = pd.date_range("2023-01-31", periods=13, freq="BME")
    monthly_decision_close_df = pd.DataFrame(
        {"WINNER": np.linspace(100.0, 160.0, len(decision_date_idx))},
        index=decision_date_idx,
    )

    last_12m_df = compute_monthly_roc_variant_df(
        monthly_decision_close_df=monthly_decision_close_df,
        roc_mode_str=ROC_MODE_LAST_12M_STR,
    )

    assert last_12m_df.iloc[-1, 0] == pytest.approx(0.60)


def test_wrapper_runs_only_the_paper_b_mode(monkeypatch):
    observed_kwarg_dict = {}
    expected_strategy_obj = object()

    def fake_run_roc_variant(**run_kwarg_dict):
        observed_kwarg_dict.update(run_kwarg_dict)
        return expected_strategy_obj

    monkeypatch.setattr(wrapper, "run_roc_variant", fake_run_roc_variant)

    strategy_obj = wrapper.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2001-01-01",
        capital_base_float=250_000.0,
        end_date_str="2025-12-31",
    )

    assert strategy_obj is expected_strategy_obj
    assert observed_kwarg_dict["roc_mode_str"] == ROC_MODE_PAPER_B_STR
    assert observed_kwarg_dict["atr_window_int"] == 20
    assert observed_kwarg_dict["show_display_bool"] is False
    assert observed_kwarg_dict["save_results_bool"] is False
    assert observed_kwarg_dict["backtest_start_date_str"] == "2001-01-01"
    assert observed_kwarg_dict["capital_base_float"] == 250_000.0
    assert observed_kwarg_dict["end_date_str"] == "2025-12-31"


def test_bench_discovers_paper_b_ndx_variant_as_research_only():
    entry_obj = next(
        entry_obj
        for entry_obj in catalog.list_strategies()
        if entry_obj.stem_str == "strategy_mo_atr_normalized_ndx_vxn_scaled_paper_b"
    )

    assert entry_obj.has_run_variant_bool is True
    assert entry_obj.is_wired_bool is False
    assert entry_obj.category_str == "momentum"
    assert entry_obj.subcategory_str == "atr_normalized_rotation"
