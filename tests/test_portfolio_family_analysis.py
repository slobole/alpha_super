"""Synthetic contract checks; never read portfolio-family performance artifacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.research.portfolio_family_20260923 import analyze as analysis_module


def test_independent_sleeves_drift_instead_of_hidden_daily_rebalance():
    date_idx = pd.to_datetime(["2024-06-03", "2024-06-04"])
    return_df = pd.DataFrame({"growth": [1.0, 0.0], "defensive": [0.0, 1.0]}, index=date_idx)
    nav_series, weight_df, contribution_df, cost_series = analysis_module.simulate_book(
        return_df, {"growth": .5, "defensive": .5}, pd.Timestamp("2024-05-31"),
        "none_drift", outer_cost_float=0,
    )
    assert nav_series.tolist() == pytest.approx([1.0, 1.5, 2.0])
    assert nav_series.iloc[-1] != pytest.approx(2.25)  # hidden daily 50/50 reset
    assert weight_df.iloc[1].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert cost_series.eq(0).all()
    np.testing.assert_allclose(contribution_df.sum(axis=1), [.5, 1 / 3], atol=1e-14)


def test_annual_reset_and_fee_use_prior_close_and_contributions_reconcile():
    date_idx = pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03"])
    return_df = pd.DataFrame({"growth": [.5, 2.0, 0.0], "defensive": [0.0, -.2, 0.0]}, index=date_idx)
    nav_series, weight_df, contribution_df, cost_series = analysis_module.simulate_book(
        return_df, {"growth": .5, "defensive": .5}, pd.Timestamp("2024-12-30"),
        "annual_fixed", outer_cost_float=.01,
    )
    # Initial 1% purchase proxy leaves .495 per sleeve. The following reset
    # sees .7425/.495, so prior weights are 60/40 and gross turnover is 20%.
    assert cost_series.tolist() == pytest.approx([.01, .002, 0.0])
    assert nav_series.iloc[1] == pytest.approx(1.2375)
    assert weight_df.iloc[1].tolist() == pytest.approx([.5, .5])
    assert nav_series.iloc[2] == pytest.approx(1.2375 * .998 * 1.9)
    book_return_vec = nav_series.to_numpy()[1:] / nav_series.to_numpy()[:-1] - 1
    np.testing.assert_allclose(contribution_df.sum(axis=1), book_return_vec, atol=1e-14)
    changed_df = return_df.copy()
    changed_df.loc[date_idx[1], ["growth", "defensive"]] = [-.8, 4.0]
    _, changed_weight_df, _, changed_cost_series = analysis_module.simulate_book(
        changed_df, {"growth": .5, "defensive": .5}, pd.Timestamp("2024-12-30"),
        "annual_fixed", outer_cost_float=.01,
    )
    np.testing.assert_allclose(changed_weight_df.iloc[1], weight_df.iloc[1])
    assert changed_cost_series.iloc[1] == pytest.approx(cost_series.iloc[1])


def test_initial_loss_and_fee_survive_drawdown_anchor():
    date_idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    metric_dict = analysis_module.metrics_dict(pd.Series([-.20, .10], index=date_idx))
    assert metric_dict["max_drawdown"] == pytest.approx(-.20)
    assert metric_dict["max_underwater_sessions"] == 2
    nav_series, _, _, _ = analysis_module.simulate_book(
        pd.DataFrame({"flat": [0., 0.]}, index=date_idx), {"flat": 1.},
        pd.Timestamp("2024-12-31"), "none_drift", outer_cost_float=.01,
    )
    return_series = pd.Series(nav_series.to_numpy()[1:] / nav_series.to_numpy()[:-1] - 1, index=date_idx)
    assert analysis_module.metrics_dict(return_series)["max_drawdown"] == pytest.approx(-.01)


def test_monthly_metrics_compound_daily_returns():
    date_idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-03", "2025-02-04"])
    metric_dict = analysis_module.metrics_dict(pd.Series([.10, -.10, -.005, 0.], index=date_idx))
    assert metric_dict["worst_month"] == pytest.approx(-.01)
    assert metric_dict["negative_month_fraction"] == 1.
    assert metric_dict["total_return"] == pytest.approx(.99 * .995 - 1)


@pytest.mark.parametrize("bad_case_str", ["internal_hole", "missing_anchor", "missing_endpoint", "missing_return"])
def test_strict_panel_rejects_incomplete_sessions_or_returns(bad_case_str):
    date_idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    component_df = pd.DataFrame({"native": [np.nan, .01, -.01, .02]}, index=date_idx)
    bad_df = component_df.copy()
    if bad_case_str == "internal_hole":
        bad_df = bad_df.drop(date_idx[1])
    elif bad_case_str == "missing_anchor":
        bad_df = bad_df.iloc[1:]
    elif bad_case_str == "missing_endpoint":
        bad_df = bad_df.iloc[:-1]
    else:
        bad_df.loc[date_idx[2], "native"] = np.nan
    with pytest.raises(ValueError):
        analysis_module.strict_return_panel(
            {"complete": component_df, "bad": bad_df}, ["complete", "bad"],
            "2025-01-02", "2025-01-07", "native",
        )


def test_strict_panel_discards_only_anchor_and_preserves_first_actual_loss():
    date_idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    component_df = pd.DataFrame({"native": [np.nan, -.15, .03]}, index=date_idx)
    panel_df = analysis_module.strict_return_panel(
        {"pod": component_df}, ["pod"], "2025-01-02", "2025-01-06", "native",
    )
    assert panel_df.index.equals(date_idx[1:])
    assert panel_df.pod.tolist() == pytest.approx([-.15, .03])


def test_holm_correction_preserves_input_order_and_stepdown_monotonicity():
    np.testing.assert_allclose(analysis_module.holm_adjust(np.array([.04, .01, .20, .03])), [.09, .04, .20, .09])
    np.testing.assert_allclose(analysis_module.holm_adjust(np.array([.01, .8, .01])), [.03, .8, .03])


def test_paired_bootstrap_is_deterministic_and_cancels_identical_paths():
    date_idx = pd.bdate_range("2024-01-02", periods=80)
    common_vec = np.sin(np.arange(80)) * .05
    return_df = pd.DataFrame({"control": common_vec, "same": common_vec,
                              "increment": common_vec + .001}, index=date_idx)
    test_list = [{"candidate": "same", "comparator": "control"},
                 {"candidate": "increment", "comparator": "control"}]
    first_df = analysis_module.paired_bootstrap(return_df, test_list, 123, 200, 7)
    second_df = analysis_module.paired_bootstrap(return_df, test_list, 123, 200, 7)
    pd.testing.assert_frame_equal(first_df, second_df)
    identical_series = first_df.iloc[0]
    assert identical_series.annualized_paired_mean == 0
    assert identical_series.ci95_low == identical_series.ci95_high == 0
    assert identical_series.p_one_sided == identical_series.p_holm == 1
    assert first_df.iloc[1].annualized_paired_mean == pytest.approx(.252)
    assert first_df.iloc[1].ci95_low == pytest.approx(.252)
    assert first_df.iloc[1].ci95_high == pytest.approx(.252)


@pytest.fixture
def short_source_path(tmp_path):
    source_path = tmp_path / "synthetic_short"
    source_path.mkdir()
    date_idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    metadata_dict = {"source_id_str": "synthetic_short", "accounting_policy_dict": {
        "dividend_withholding_rate_float": 0.0,
    }}
    (source_path / "source_metadata.json").write_text(json.dumps(metadata_dict), encoding="utf-8")
    pd.DataFrame({"total_value": [100.] * 4, "portfolio_value": [90.] * 4, "cash": [10.] * 4},
                 index=date_idx).to_csv(source_path / "nav.csv.gz", index_label="date")
    pd.DataFrame({"LONG": [1.1] * 4, "SHORT": [-.2] * 4, "Cash": [.1] * 4},
                 index=date_idx).to_csv(source_path / "realized_weights.csv.gz", index_label="date")
    pd.DataFrame({"accrual_start_date_ts": date_idx[:3], "collateral_value_float": [30.] * 3,
                  "annual_borrow_rate_float": [.01] * 3,
                  "borrow_fee_float": np.array([1., 3., 1.]) * 30 * .01 / 360,
                  }).to_csv(source_path / "borrow.csv.gz", index=False)
    pd.DataFrame({"ex_date": [date_idx[1], date_idx[1]],
                  "gross_dividend_cash_float": [4., -2.],
                  }).to_csv(source_path / "dividends.csv.gz", index=False)
    pd.DataFrame({"bar": [date_idx[1]], "amount": [2.], "price": [10.], "commission": [.05],
                  }).to_csv(source_path / "transactions.csv.gz", index=False)
    return source_path


def test_source_costs_use_exact_collateral_calendar_days_and_incremental_charges(short_source_path):
    component_df, _ = analysis_module.source_components(short_source_path)
    friday_series = component_df.loc["2025-01-03"]
    assert friday_series.gross == pytest.approx(1.3)  # Cash is not gross exposure.
    assert friday_series.short == pytest.approx(.2)
    assert friday_series.funding_base_weight == pytest.approx(.2)  # exact30 collateral minus10 cash
    assert friday_series.tax_adjustment == pytest.approx(.01)  # long dividend only
    assert friday_series.funding_common == pytest.approx(20 * .05 * 3 / 360 / 100)
    assert friday_series.native_borrow_rate == pytest.approx(30 * .01 * 3 / 360 / 100)
    assert friday_series.borrow_extra == pytest.approx(30 * .04 * 3 / 360 / 100)
    assert friday_series.turnover == pytest.approx(.2)
    assert friday_series.slippage_extra == pytest.approx(.0002)
    assert friday_series.commission_rate == pytest.approx(.0005)
    assert friday_series.common_account == pytest.approx(-.01 - 20 * .05 * 3 / 360 / 100)
    assert friday_series.conservative == pytest.approx(-.01 - 20 * .08 * 3 / 360 / 100 - .0002 - .0001)
    assert component_df.iloc[-1].funding_common == 0
    assert component_df.iloc[-1].funding_conservative == 0
    assert component_df.iloc[-1].borrow_extra == 0


def test_terminal_mark_must_not_add_forward_borrow_interval(short_source_path):
    borrow_path = short_source_path / "borrow.csv.gz"
    borrow_df = pd.read_csv(borrow_path)
    terminal_df = pd.DataFrame({"accrual_start_date_ts": ["2025-01-07"], "collateral_value_float": [30.],
                               "annual_borrow_rate_float": [.01], "borrow_fee_float": [30 * .01 / 360]})
    pd.concat([borrow_df, terminal_df], ignore_index=True).to_csv(borrow_path, index=False)
    component_df, _ = analysis_module.source_components(short_source_path)
    # Native history is retained; adjusted reporting refunds a known prepaid
    # interval beyond its terminal mark and removes its incremental stress fee.
    assert component_df.iloc[-1].native_borrow_rate > 0
    native_panel_df = analysis_module.strict_return_panel(
        {"source": component_df}, ["source"], "2025-01-02", "2025-01-07", "native",
    )
    adjusted_panel_df = analysis_module.strict_return_panel(
        {"source": component_df}, ["source"], "2025-01-02", "2025-01-07", "conservative",
    )
    assert native_panel_df.iloc[-1, 0] == 0
    assert adjusted_panel_df.iloc[-1, 0] == pytest.approx(30 * .01 / 360 / 100)


def test_missing_nonterminal_short_collateral_is_rejected(short_source_path):
    borrow_path = short_source_path / "borrow.csv.gz"
    borrow_df = pd.read_csv(borrow_path)
    borrow_df.iloc[[0, 2]].to_csv(borrow_path, index=False)
    with pytest.raises(ValueError, match="Missing collateral"):
        analysis_module.source_components(short_source_path)


@pytest.mark.parametrize("scenario_str", ["common_account", "conservative"])
def test_selected_window_terminal_cost_does_not_depend_on_later_native_history(short_source_path, scenario_str):
    full_component_df, _ = analysis_module.source_components(short_source_path)
    nav_path = short_source_path / "nav.csv.gz"
    nav_df = pd.read_csv(nav_path, index_col="date", parse_dates=["date"]).iloc[:-1]
    # A source genuinely ending Jan6 would never prepay Jan6-to-Jan7 borrow.
    # Its native cash/NAV therefore retain the fee the longer source prepaid.
    avoided_fee_float = 30 * .01 / 360
    nav_df.loc[nav_df.index[-1], ["total_value", "cash"]] += avoided_fee_float
    nav_df.to_csv(nav_path, index_label="date")
    weight_path = short_source_path / "realized_weights.csv.gz"
    weight_df = pd.read_csv(weight_path, index_col="date", parse_dates=["date"]).iloc[:-1]
    terminal_nav_float = 100 + avoided_fee_float
    weight_df.loc[weight_df.index[-1], ["LONG", "SHORT", "Cash"]] = [
        110 / terminal_nav_float, -20 / terminal_nav_float, (10 + avoided_fee_float) / terminal_nav_float,
    ]
    weight_df.to_csv(weight_path, index_label="date")
    borrow_path = short_source_path / "borrow.csv.gz"
    borrow_df = pd.read_csv(borrow_path)
    borrow_df.iloc[:-1].to_csv(borrow_path, index=False)
    ended_component_df, _ = analysis_module.source_components(short_source_path)
    full_panel_df = analysis_module.strict_return_panel(
        {"source": full_component_df}, ["source"], "2025-01-02", "2025-01-06", scenario_str,
    )
    ended_panel_df = analysis_module.strict_return_panel(
        {"source": ended_component_df}, ["source"], "2025-01-02", "2025-01-06", scenario_str,
    )
    # An evaluated portfolio ends Jan6 in both cases. An extra native Jan7 mark
    # must not make the Jan6 reporting cutoff buy one more funding/borrow day.
    pd.testing.assert_frame_equal(full_panel_df, ended_panel_df, rtol=1e-12, atol=1e-14)


def test_current_day_borrow_is_not_refunded_at_reporting_cutoff(short_source_path):
    borrow_path = short_source_path / "borrow.csv.gz"
    borrow_df = pd.read_csv(borrow_path).rename(columns={"accrual_start_date_ts": "date_ts"})
    terminal_df = pd.DataFrame({"date_ts": ["2025-01-07"], "collateral_value_float": [30.],
                               "annual_borrow_rate_float": [.01], "borrow_fee_float": [30 * .01 / 360]})
    pd.concat([borrow_df, terminal_df], ignore_index=True).to_csv(borrow_path, index=False)
    component_df, _ = analysis_module.source_components(short_source_path)
    panel_df = analysis_module.strict_return_panel(
        {"source": component_df}, ["source"], "2025-01-02", "2025-01-07", "conservative",
    )
    assert component_df.iloc[-1].future_native_borrow == 0
    assert panel_df.iloc[-1, 0] == pytest.approx(-30 * .04 / 360 / 100)


def test_transaction_event_outside_nav_calendar_is_rejected(short_source_path):
    transaction_path = short_source_path / "transactions.csv.gz"
    transaction_df = pd.read_csv(transaction_path)
    transaction_df.loc[0, "bar"] = "2025-01-04"
    transaction_df.to_csv(transaction_path, index=False)
    with pytest.raises(ValueError, match="Event outside native NAV calendar"):
        analysis_module.source_components(short_source_path)
