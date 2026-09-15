"""Restart boundary, terminal valuation and accounting-bridge regression tests."""
import numpy as np
import pandas as pd
import pytest

from alpha.engine.strategy import Strategy
from alpha.engine.backtest import run_daily
from scripts.research.run_mr_deeper_dip_study import DeeperDipResearchMixin
from scripts.research.run_hpi_deeper_dip_path_study import window_calendar, YEAR_TUPLE
from scripts.research.analyze_hpi_deeper_dip_path_study import attribution, continuous_window_returns, validate_receipt_grid
from scripts.research.analyze_mr_deeper_dip_study import make_trade_table


def trade_row(asset_str, entry_date_str, shares_float=10., entry_float=100., exit_float=105.,
              dividend_float=0., commission_float=2., friction_float=0., terminal_bool=False, trade_id_int=1):
    return {"asset": asset_str, "entry_date": entry_date_str, "shares": shares_float,
            "entry_price": entry_float, "exit_equivalent_price": exit_float,
            "dividends": dividend_float, "native_commission": commission_float,
            "research_friction": friction_float, "terminal_mark_value": shares_float * exit_float if terminal_bool else 0.,
            "net_pnl": shares_float * (exit_float - entry_float) + dividend_float - commission_float - friction_float,
            "trade_id": trade_id_int, "exit_date": "2020-01-06", "closed": not terminal_bool}


def test_accounting_bridge_matches_by_asset_date_and_counts_terminal_once():
    moo_df = pd.DataFrame([trade_row("AAA", "2020-01-02", terminal_bool=True),
                           trade_row("BBB", "2020-01-02", exit_float=110.),
                           trade_row("CCC", "2020-01-02", exit_float=95.)])
    limit_df = pd.DataFrame([trade_row("AAA", "2020-01-02", shares_float=12., entry_float=99., exit_float=106.,
        dividend_float=3., commission_float=2.2, friction_float=1., terminal_bool=True, trade_id_int=99),
        trade_row("DDD", "2020-01-03", exit_float=108.)])
    bridge_df, group_df, shared_df, effect_ser = attribution(moo_df, limit_df)
    assert group_df["trade_count"].to_dict() == {"shared": 1, "moo_only": 2, "limit_only": 1}
    assert bridge_df["delta_pnl_contribution"].sum() == pytest.approx(limit_df["net_pnl"].sum() - moo_df["net_pnl"].sum())
    assert shared_df["delta_sale_cash"].iloc[0] == 0.
    assert shared_df["delta_terminal_value"].iloc[0] == 12. * 106. - 10. * 105.
    assert effect_ser.sum() == pytest.approx(shared_df["delta_pnl_contribution"].sum())
    assert bridge_df.loc[bridge_df["asset"] == "CCC", "delta_pnl_contribution"].iloc[0] > 0.


def test_duplicate_matching_key_is_rejected():
    trade_df = pd.DataFrame([trade_row("AAA", "2020-01-02")])
    with pytest.raises(AssertionError, match="Duplicate"):
        attribution(pd.concat([trade_df, trade_df]), trade_df)


def test_missing_present_arm_price_is_not_silently_zeroed():
    trade_df = pd.DataFrame([trade_row("AAA", "2020-01-02")])
    damaged_df = trade_df.copy()
    damaged_df["entry_price"] = np.nan
    with pytest.raises(AssertionError, match="Nonfinite"):
        attribution(trade_df, damaged_df)


def test_missing_present_quantity_is_not_an_absent_trade():
    trade_df = pd.DataFrame([trade_row("AAA", "2020-01-02")])
    damaged_df = trade_df.copy()
    damaged_df["shares"] = np.nan
    with pytest.raises(AssertionError, match="finite positive shares"):
        attribution(trade_df, damaged_df)


def test_continuous_slice_keeps_first_day_gain():
    date_idx = pd.date_range("2019-12-30", periods=4, freq="B")
    nav_ser = pd.Series([100000., 110000., 99000., 108900.], index=date_idx)
    return_ser = continuous_window_returns(nav_ser, date_idx[2:])
    assert return_ser.iloc[0] == pytest.approx(-.1)
    assert (1. + return_ser).prod() - 1. == pytest.approx(-.01)


def test_fourteen_fixed_calendar_windows_keep_original2010_start():
    calendar_idx = pd.bdate_range("2010-01-05", "2026-09-14")
    assert len(YEAR_TUPLE) == 14
    for year_int in YEAR_TUPLE:
        selected_idx = window_calendar(calendar_idx, year_int)
        assert selected_idx[0].year == year_int
        assert selected_idx[-1].year == year_int + 2
        assert selected_idx[-1] < pd.Timestamp(f"{year_int+3}-01-01")
    assert window_calendar(calendar_idx, 2010)[0] == pd.Timestamp("2010-01-05")


def test_receipts_reject_duplicate_cell_and_stale_spec():
    spec_dict = {"windows": [{"start_year": year_int, "start": f"{year_int}-01-05",
                  "end": f"{year_int+2}-12-31", "sessions": 753} for year_int in YEAR_TUPLE],
                 "layers": ["central", "stress"], "policies": ["moo", "limit_0.5pct"], "runner_sha256": "runner"}
    receipt_list = [{**window_dict, "layer": layer_str, "policy": policy_str,
                     "spec_sha256": "spec", "runner_sha256": "runner"}
                    for window_dict in spec_dict["windows"] for layer_str in spec_dict["layers"]
                    for policy_str in spec_dict["policies"]]
    validate_receipt_grid(receipt_list, spec_dict, "spec")
    with pytest.raises(AssertionError, match="grid"):
        validate_receipt_grid(receipt_list[:-1] + [receipt_list[0]], spec_dict, "spec")
    with pytest.raises(AssertionError, match="lineage"):
        validate_receipt_grid(receipt_list, spec_dict, "different_spec")
    wrong_window_list = [dict(receipt_dict) for receipt_dict in receipt_list]
    wrong_window_list[0]["end"] = "2026-09-14"
    with pytest.raises(AssertionError, match="calendar"):
        validate_receipt_grid(wrong_window_list, spec_dict, "spec")


class RestartFixtureStrategy(DeeperDipResearchMixin, Strategy):
    def compute_signals(self, pricing_data):
        return pricing_data

    def iterate(self, data, close, open_prices):
        if self.get_position("AAA") == 0. and close[("AAA", "Close")] == 100.:
            self.order_value("AAA", 1000., trade_id=1)


def test_restart_uses_preceding_close_and_ignores_post_window_prices():
    date_idx = pd.date_range("2020-01-02", periods=4, freq="B")
    pricing_df = pd.DataFrame({("AAA", "Open"): [100., 100., 101., 102.],
        ("AAA", "High"): [101., 102., 103., 104.], ("AAA", "Low"): [99., 98., 100., 101.],
        ("AAA", "Close"): [100., 101., 102., 103.], ("AAA", "Volume"): [100000.] * 4,
        ("AAA", "Dividend"): [1., 1., 0., 0.]}, index=date_idx)
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {"AAA": "CAPITALSPECIAL"}
    result_list = []
    for future_price_float in (103., 100000.):
        pricing_df.loc[date_idx[-1], ("AAA", "Close")] = future_price_float
        strategy_obj = RestartFixtureStrategy("restart", [], capital_base=100000., slippage=.00025,
                                             commission_per_share=.005, commission_minimum=1.)
        strategy_obj.configure_research(.005)
        window_df = pricing_df.loc[:date_idx[2]]
        run_daily(strategy_obj, window_df, date_idx[1:3], show_progress=False, show_signal_progress_bool=False)
        entry_dict = strategy_obj.entry_row_list[0]
        assert entry_dict["decision_date"] == "2020-01-02"
        assert entry_dict["date"] == "2020-01-03"
        assert entry_dict["shares"] == 10.
        dividend_df = strategy_obj.get_dividend_ledger()
        assert dividend_df["ex_date"].min() == date_idx[2]  # no inherited entitlement on first session
        trade_df = make_trade_table(strategy_obj.get_transactions(), dividend_df,
                                    pd.DataFrame(strategy_obj.friction_row_list), window_df)
        assert trade_df["terminal_mark_value"].iloc[0] == 1020.
        assert not trade_df["closed"].iloc[0]
        result_list.append(strategy_obj.total_value)
        assert trade_df["net_pnl"].sum() == pytest.approx(strategy_obj.total_value - 100000.)
    assert result_list[0] == result_list[1]
