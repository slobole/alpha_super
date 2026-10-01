"""Scout weights engine (alpha/scout/engines/weights.py): hand-computed cases, then the real-data identity gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha.scout.engines.weights import CostModel, simulate

DATE_INDEX = pd.bdate_range("2024-01-01", periods=6)
NO_COST = CostModel(slippage_float=0.0, fee_per_share_float=0.0, min_fee_float=0.0, dividend_withholding_float=0.25)


def _frames(open_rows, close_rows, dividend_rows=None, columns=("A", "B")):
    open_df = pd.DataFrame(open_rows, index=DATE_INDEX, columns=list(columns), dtype=float)
    close_df = pd.DataFrame(close_rows, index=DATE_INDEX, columns=list(columns), dtype=float)
    dividend_df = pd.DataFrame(dividend_rows if dividend_rows is not None else 0.0, index=DATE_INDEX, columns=list(columns), dtype=float)
    return open_df, close_df, dividend_df


def test_sizing_uses_previous_close_and_previous_total_fills_at_open():
    open_df, close_df, dividend_df = _frames([[10, 20]] * 6, [[10, 20], [11, 20], [12, 20], [12, 20], [12, 20], [12, 20]])
    weight_df = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[DATE_INDEX[1]])
    result = simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], capital_float=1000.0, cost_model=NO_COST)
    # Target on day 1 from V = 1000 and Close(day 0): A = trunc(500/10) = 50, B = trunc(500/20) = 25, filled at Open = 10, 20.
    np.testing.assert_allclose(result.position_after_rebalance_df.loc[DATE_INDEX[1]], [50, 25])
    # Cash = 1000 − 500 − 500 = 0; close of day 1: 50 × 11 + 25 × 20 = 1050.
    assert result.total_value_ser.loc[DATE_INDEX[1]] == pytest.approx(1050.0)
    assert result.daily_return_ser.iloc[0] == pytest.approx(0.05)


def test_costs_slippage_and_minimum_fee():
    open_df, close_df, dividend_df = _frames([[10, 20]] * 6, [[10, 20]] * 6)
    weight_df = pd.DataFrame({"A": [0.5], "B": [0.0]}, index=[DATE_INDEX[1]])
    cost_model = CostModel(slippage_float=0.001, fee_per_share_float=0.005, min_fee_float=1.0)
    result = simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], capital_float=1000.0, cost_model=cost_model)
    # Buy 50 A at 10.01; fee max(1, 0.25) = 1. Cash = 1000 − 500.5 − 1 = 498.5; mark 50 × 10 = 500.
    assert result.total_value_ser.loc[DATE_INDEX[1]] == pytest.approx(998.5)
    assert result.trade_df["fee_float"].tolist() == [1.0]


def test_dividends_are_credited_net_of_withholding_before_the_next_open():
    open_df, close_df, dividend_df = _frames([[10, 20]] * 6, [[10, 20]] * 6, [[0, 0], [0, 0], [2, 0], [0, 0], [0, 0], [0, 0]])
    weight_df = pd.DataFrame({"A": [0.5], "B": [0.0]}, index=[DATE_INDEX[1]])
    result = simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], capital_float=1000.0, cost_model=NO_COST)
    # 50 shares held at the close of day 2 with Dividend 2 -> +100 × 0.75 credited on day 3 (not day 2).
    assert result.total_value_ser.loc[DATE_INDEX[2]] == pytest.approx(1000.0)
    assert result.total_value_ser.loc[DATE_INDEX[3]] == pytest.approx(1075.0)
    bad_dividend_df = dividend_df.copy()
    bad_dividend_df.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="NaN dividend"):
        simulate(open_df, close_df, bad_dividend_df, weight_df, DATE_INDEX[1], capital_float=1000.0, cost_model=NO_COST)


def test_missing_price_liquidation_and_cancelled_orders():
    open_df, close_df, dividend_df = _frames(
        [[10, 20], [10, np.nan], [10, 20], [np.nan, 20], [10, 20], [10, 20]],
        [[10, 20], [10, np.nan], [10, 20], [9, 20], [10, 20], [10, 20]],
    )
    weight_df = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=[DATE_INDEX[1], DATE_INDEX[3]])
    result = simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], capital_float=1000.0, cost_model=NO_COST)
    # Day 1: B has no Open -> its buy is cancelled, only A is bought.
    np.testing.assert_allclose(result.position_after_rebalance_df.loc[DATE_INDEX[1]], [50, 0])
    # Day 3: A has NaN Open -> liquidated at its last close (day 2 = 10), and its new buy is cancelled.
    assert result.position_after_rebalance_df.loc[DATE_INDEX[3], "A"] == 0.0
    assert "liquidation" in set(result.trade_df["kind_str"])


def test_historical_share_units():
    open_df, close_df, dividend_df = _frames([[10, 20]] * 6, [[10, 20]] * 6)
    raw_close_df = close_df * 2.0  # a later 2:1 split: raw price is twice the adjusted price, k = 2
    weight_df = pd.DataFrame({"A": [0.5], "B": [0.0]}, index=[DATE_INDEX[1]])
    cost_model = CostModel(slippage_float=0.0, fee_per_share_float=0.1, min_fee_float=0.0)
    result = simulate(
        open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], capital_float=1000.0,
        share_unit_mode_str="historical", unadjusted_close_df=raw_close_df, cost_model=cost_model,
    )
    # Raw shares = trunc(500 / 20) = 25 -> ledger units 25 × 2 = 50; fee on raw shares 0.1 × 50 / 2 = 2.5.
    assert result.position_after_rebalance_df.loc[DATE_INDEX[1], "A"] == 50.0
    assert result.trade_df["fee_float"].iloc[0] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="unadjusted_close_df"):
        simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], share_unit_mode_str="historical")


def test_sell_side_slippage_and_historical_liquidation_fee():
    open_df, close_df, dividend_df = _frames(
        [[10, 20], [10, 20], [10, 20], [10, np.nan], [10, 20], [10, 20]],
        [[10, 20], [10, 20], [12, 20], [10, np.nan], [10, 20], [10, 20]],
    )
    raw_close_df = close_df * 4.0  # k = 4
    weight_df = pd.DataFrame({"A": [0.5, 0.0], "B": [0.5, 0.5]}, index=[DATE_INDEX[1], DATE_INDEX[2]])
    cost_model = CostModel(slippage_float=0.01, fee_per_share_float=0.1, min_fee_float=0.0)
    result = simulate(
        open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], capital_float=1000.0,
        share_unit_mode_str="historical", unadjusted_close_df=raw_close_df, cost_model=cost_model,
    )
    trade_df = result.trade_df.set_index(["date", "asset"])
    # Day 2: A is sold at Open × (1 − slippage) = 9.9 (a sell gets the lower price).
    assert trade_df.loc[(DATE_INDEX[2], "A"), "price_float"] == pytest.approx(9.9)
    # Day 3: B has no prices -> liquidated at the day-2 close 20 with a fee on raw shares (ledger / k at that bar).
    liquidation_row = trade_df.loc[(DATE_INDEX[3], "B")]
    assert liquidation_row["kind_str"] == "liquidation" and liquidation_row["price_float"] == pytest.approx(20.0)
    assert liquidation_row["fee_float"] == pytest.approx(0.1 * abs(liquidation_row["delta_float"]) / 4.0)


def test_monthly_drift_is_traded_back_to_target():
    open_df, close_df, dividend_df = _frames([[10, 10]] * 6, [[10, 10], [10, 10], [20, 10], [20, 10], [20, 10], [20, 10]])
    weight_df = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=[DATE_INDEX[1], DATE_INDEX[3]])
    result = simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], capital_float=1000.0, cost_model=NO_COST)
    # Day 2 close: A doubled -> V = 1500; day 3 targets trunc(750/20) = 37 A and trunc(750/10) = 75 B.
    np.testing.assert_allclose(result.position_after_rebalance_df.loc[DATE_INDEX[3]], [37, 75])


def test_short_split_sign_flip_and_borrow_fee():
    """CORE5's short contract: truncation toward zero, a flip as two fee-paying legs, borrow accrued on calendar days."""
    from alpha.scout.engines.weights import BorrowModel

    open_df, close_df, dividend_df = _frames([[10, 20]] * 6, [[10, 20]] * 6)
    weight_df = pd.DataFrame({"A": [0.5, 0.0], "B": [-0.2, 0.5]}, index=[DATE_INDEX[1], DATE_INDEX[5]])
    cost_model = CostModel(slippage_float=0.0, fee_per_share_float=0.005, min_fee_float=1.0)
    with pytest.raises(ValueError, match="allow_short_bool"):
        simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], capital_float=1000.0, cost_model=cost_model)
    kwarg_dict = {"capital_float": 1000.0, "cost_model": cost_model, "allow_short_bool": True, "borrow_model": BorrowModel(annual_rate_float=0.36)}
    result = simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], split_sign_flip_bool=True, **kwarg_dict)
    # Tue: +50 A, -10 B, fees 1 + 1 -> 998. Borrow per calendar day: 10 x ceil(1.02 x 20) = 210 x 0.36 / 360 = 0.21;
    # Tue, Wed, Thu 1 day each, Fri 3 days (to Mon) -> 1.26, so Fri closes at 996.74.
    np.testing.assert_allclose(result.position_after_rebalance_df.loc[DATE_INDEX[1]], [50, -10])
    assert result.total_value_ser.loc[DATE_INDEX[4]] == pytest.approx(996.74)
    assert result.borrow_fee_df["fee_float"].tolist() == pytest.approx([0.21, 0.21, 0.21, 0.63])
    # Mon: B goes -10 -> trunc(996.74 x 0.5 / 20) = 24 as two legs (+10, +24), each paying the $1 minimum; A sold (fee 1).
    assert result.trade_df.loc[result.trade_df["date"] == DATE_INDEX[5], "delta_float"].tolist() == [-50.0, 10.0, 24.0]
    assert result.total_value_ser.loc[DATE_INDEX[5]] == pytest.approx(993.74)
    one_leg_result = simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], **kwarg_dict)
    assert one_leg_result.total_value_ser.loc[DATE_INDEX[5]] == pytest.approx(994.74)  # one +34 order, one fee


def test_fractional_entries_and_untouched_holdings():
    """The sector IBS contract (opt-in): untruncated share targets, NaN = leave the asset alone, tiny deltas cancelled."""
    open_df, close_df, dividend_df = _frames([[10, 20]] * 6, [[10, 20], [11, 20], [12, 20], [12, 20], [12, 20], [12, 20]])
    # Day 1: A 0.5, B 0.25. Day 3: A untouched (NaN), B sold. Day 4: A at exactly its current weight (no order).
    weight_df = pd.DataFrame({"A": [0.5, np.nan, 50.0 * 12.0 / 1100.0], "B": [0.25, 0.0, np.nan]}, index=DATE_INDEX[[1, 3, 4]])
    kwarg_dict = {"capital_float": 1000.0, "cost_model": NO_COST}
    result = simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], fractional_shares_bool=True, hold_nan_bool=True, **kwarg_dict)
    # Day 1: 1000 x 0.5 / 10 = 50 A and 1000 x 0.25 / 20 = 12.5 B (whole shares would give 12); cash 250.
    # Close day 1: 550 + 250 + 250 = 1050; day 2: 1100; day 3: B sold at 20 -> cash 500, A still 50 -> 1100.
    np.testing.assert_allclose(result.position_after_rebalance_df.loc[DATE_INDEX[1]], [50.0, 12.5])
    np.testing.assert_allclose(result.daily_position_df["A"], [50.0] * 5)
    np.testing.assert_allclose(result.total_value_ser.to_numpy(), [1050.0, 1100.0, 1100.0, 1100.0, 1100.0])
    assert result.trade_df["date"].tolist() == [DATE_INDEX[1], DATE_INDEX[1], DATE_INDEX[3]]  # nothing on day 4
    # Without the flags: trunc(12.5) = 12 B, and the NaN cell is a 0 weight, so A is sold on day 3.
    plain_result = simulate(open_df, close_df, dividend_df, weight_df.iloc[:2], DATE_INDEX[1], **kwarg_dict)
    assert plain_result.position_after_rebalance_df.loc[DATE_INDEX[1], "B"] == 12.0
    assert plain_result.position_after_rebalance_df.loc[DATE_INDEX[3], "A"] == 0.0
    # A NaN from a decision hook without hold_nan_bool still fails loudly; fractional needs adjusted units.
    with pytest.raises(ValueError):
        simulate(open_df, close_df, dividend_df, weight_df.iloc[:0], DATE_INDEX[1], **kwarg_dict,
                 decision_fn=lambda t_idx_int, position_vec, total_float: np.array([np.nan, 0.5]))
    with pytest.raises(ValueError, match="adjusted"):
        simulate(open_df, close_df, dividend_df, weight_df, DATE_INDEX[1], share_unit_mode_str="historical",
                 unadjusted_close_df=close_df, fractional_shares_bool=True)


# ---------------------------------------------------------------------------------------------- real-data gates
def _norgate_running_bool() -> bool:
    """True only when the Norgate Data Updater is running (the package alone is always installed)."""
    try:
        import norgatedata

        return bool(norgatedata.status())
    except Exception:
        return False


NORGATE_AVAILABLE_BOOL = _norgate_running_bool()


def _saved_run_exists(spec_name_str: str) -> bool:
    from alpha.scout.gate.run import GATED_SPEC_DICT
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

    return any(Path(MAIN_CHECKOUT_ROOT_PATH).glob(GATED_SPEC_DICT[spec_name_str].pickle_glob_str))


@pytest.mark.skipif(not NORGATE_AVAILABLE_BOOL, reason="Norgate data not available")
@pytest.mark.parametrize("spec_name_str", ["taa_3x", "ndx_vxn"])
def test_identity_gate_passes_against_the_saved_engine_run(spec_name_str):
    if not _saved_run_exists(spec_name_str):
        pytest.skip("No saved engine run")
    from alpha.scout.gate.run import run_gate

    report = run_gate(spec_name_str)
    assert report.passed_bool, report.summary_str()


@pytest.mark.skipif(not NORGATE_AVAILABLE_BOOL, reason="Norgate data not available")
def test_identity_gate_blocks_a_deliberately_broken_spec(monkeypatch):
    if not _saved_run_exists("taa_3x"):
        pytest.skip("No saved engine run")
    from alpha.scout.gate.run import run_gate
    from alpha.scout.specs import taa_3x

    # A plausible-looking bug: the rank weights reversed (the worst-ranked asset gets the largest slot).
    monkeypatch.setattr(taa_3x, "RANK_WEIGHT_VEC", taa_3x.RANK_WEIGHT_VEC[::-1].copy())
    report = run_gate("taa_3x")
    assert not report.passed_bool
    assert len(report.mismatch_df) > 0


@pytest.mark.skipif(not NORGATE_AVAILABLE_BOOL, reason="Norgate data not available")
def test_identity_gate_blocks_a_missing_minimum_fee(monkeypatch):
    """Near miss: 0.4 bps a year, which the tolerance tier let through; the exact tier must not."""
    if not _saved_run_exists("taa_3x"):
        pytest.skip("No saved engine run")
    from alpha.scout.gate import run as gate_run

    monkeypatch.setattr(gate_run, "COST_MODEL", CostModel(min_fee_float=0.0))
    report = gate_run.run_gate("taa_3x")
    assert not report.passed_bool
    assert not report.check_dict["largest daily return difference"]["pass_bool"]


@pytest.mark.skipif(not NORGATE_AVAILABLE_BOOL, reason="Norgate data not available")
def test_identity_gate_blocks_a_one_session_membership_look_ahead(monkeypatch):
    """Near miss: membership read at the execution session instead of the decision session (4 bps, 99.7% cells)."""
    if not _saved_run_exists("ndx_vxn"):
        pytest.skip("No saved engine run")
    import dataclasses

    from alpha.scout.gate.run import run_gate
    from alpha.scout.specs import ndx_vxn

    original_load_inputs = ndx_vxn.load_inputs

    def look_ahead_inputs():
        inputs = original_load_inputs()
        return dataclasses.replace(inputs, member_df=inputs.member_df.shift(-1).ffill().astype(int))

    monkeypatch.setattr(ndx_vxn, "load_inputs", look_ahead_inputs)
    report = run_gate("ndx_vxn")
    assert not report.passed_bool
