"""MR capsule pods with liquidity sizing: the size rule by hand, causality, the order arithmetic, parity with 1/N at m = 1."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_strategy_mr_capsule_pods import _dv2, _hpi, _run, _synthetic_market, _trades

from alpha.bench import catalog
from alpha.engine.portfolio_manager import PortfolioManager
from alpha.strategy_registry import MaturityTier, tier_for
from strategies.hpi.stateful_long import ENTRY_HORIZON_VOTE_STR, TURNOVER_FIELD_STR, get_asof_universe_symbol_set
from strategies.mr_capsule import dv2_vix_gated_liq, hpi_vote_vix_gated_liq, liquidity_sizing
from strategies.mr_capsule.dv2_vix_gated import DV2VixGatedStrategy, build_dv2_capsule_strategy
from strategies.mr_capsule.dv2_vix_gated_liq import DV2VixGatedLiquidityStrategy
from strategies.mr_capsule.hpi_vote_vix_gated import HPIVoteVixGatedStrategy, build_hpi_capsule_strategy
from strategies.mr_capsule.hpi_vote_vix_gated_liq import HPIVoteVixGatedLiquidityStrategy
from strategies.mr_capsule.liquidity_sizing import LIQUIDITY_SIZE_FIELD_STR, LiquiditySizedPodMixin, liquidity_size_and_reference, member_at_each_session_df


def _frames(turnover_dict: dict[str, float], row_int: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2020-01-01", periods=row_int)
    turnover_df = pd.DataFrame({symbol_str: float(value_float) for symbol_str, value_float in turnover_dict.items()}, index=index)
    return turnover_df, pd.DataFrame(True, index=index, columns=turnover_df.columns)


# ----------------------------------------------------------------------------------------------- the size rule
def test_size_by_hand_and_the_observation_minimums():
    turnover_df, candidate_df = _frames({"AAA": 4e6, "BBB": 1e6}, 130)
    size_df, reference_ser = liquidity_size_and_reference(turnover_df, candidate_df)
    # ADV63 needs 42 rows (row 41 is the first); the reference needs 60 of those (row 100 is the first).
    assert reference_ser.iloc[:100].isna().all()
    assert reference_ser.iloc[100] == pytest.approx(1500.0)
    assert (size_df.iloc[:100] == 1.0).all().all()
    assert size_df["AAA"].iloc[100] == pytest.approx(2000.0 / 1500.0)
    assert size_df["BBB"].iloc[100] == pytest.approx(1000.0 / 1500.0)


def test_size_is_clipped_between_half_and_double():
    turnover_df, candidate_df = _frames({"AAA": 1e10, "BBB": 1e6, "CCC": 1e2}, 130)
    size_df, _ = liquidity_size_and_reference(turnover_df, candidate_df)
    assert size_df.iloc[-1].to_dict() == {"AAA": 2.0, "BBB": 1.0, "CCC": 0.5}


def test_reference_uses_candidates_only():
    turnover_df, candidate_df = _frames({"AAA": 4e6, "BBB": 1e6, "BIG": 1e8}, 130)
    candidate_df["BIG"] = False  # full history, never a candidate: it must not move the typical candidate
    size_df, reference_ser = liquidity_size_and_reference(turnover_df, candidate_df)
    assert reference_ser.iloc[-1] == pytest.approx(1500.0)
    assert size_df["BIG"].iloc[-1] == 2.0


def test_a_name_without_history_is_one_and_stays_out_of_the_median():
    turnover_df, candidate_df = _frames({"AAA": 4e6, "BBB": 1e6, "NEW": 1e8}, 130)
    turnover_df.iloc[:-5, turnover_df.columns.get_loc("NEW")] = np.nan  # listed five rows ago: under 42 observations
    size_df, reference_ser = liquidity_size_and_reference(turnover_df, candidate_df)
    assert reference_ser.iloc[-1] == pytest.approx(1500.0)
    assert size_df["NEW"].iloc[-1] == 1.0


def test_adv_is_a_mean_over_63_rows():
    index = pd.bdate_range("2020-01-01", periods=200)
    turnover_df = pd.DataFrame({"AAA": 100.0, "BBB": 10_000.0}, index=index)
    turnover_df.iloc[100, 0] = 1e6
    candidate_df = pd.DataFrame({"AAA": False, "BBB": True}, index=index)
    size_df, _ = liquidity_size_and_reference(turnover_df, candidate_df)
    # sqrt((62 x 100 + 1,000,000) / 63) over a typical sqrt(10,000): a median would ignore the spike, a longer window would hold it past row 162.
    assert size_df["AAA"].iloc[162] == pytest.approx(((62 * 100.0 + 1e6) / 63.0) ** 0.5 / 100.0)
    assert size_df["AAA"].iloc[163] == 0.5


def test_adv_needs_42_observations():
    index = pd.bdate_range("2020-01-01", periods=300)
    turnover_df = pd.DataFrame({"AAA": 2.25e6, "BBB": 1e6}, index=index)
    candidate_df = pd.DataFrame({"AAA": False, "BBB": True}, index=index)
    turnover_df.iloc[:-42, 0] = np.nan
    assert liquidity_size_and_reference(turnover_df, candidate_df)[0]["AAA"].iloc[-1] == pytest.approx(1.5)
    turnover_df.iloc[-42, 0] = np.nan
    assert liquidity_size_and_reference(turnover_df, candidate_df)[0]["AAA"].iloc[-1] == 1.0


def test_reference_is_a_median_over_252_rows():
    index = pd.bdate_range("2019-01-01", periods=800)
    turnover_df = pd.DataFrame({"BBB": 10_000.0}, index=index)
    turnover_df.iloc[400:, 0] = 1_000_000.0  # typical 100, a 63-row ramp, then 1,000
    _, reference_ser = liquidity_size_and_reference(turnover_df, pd.DataFrame(True, index=index, columns=["BBB"]))
    assert reference_ser.iloc[500] == pytest.approx(100.0)  # most of the window is still at 100: a mean would sit far above it
    assert reference_ser.iloc[700] == pytest.approx(1000.0)  # the window has passed the old level: a 504-row window would not have


def test_member_at_each_session_is_as_of_and_never_future():
    universe_df = pd.DataFrame({"AAA": [1, 0], "BBB": [0, 1]}, index=pd.to_datetime(["2024-01-03", "2024-01-10"]))
    member_df = member_at_each_session_df(universe_df, pd.bdate_range("2024-01-01", "2024-01-12"), ["AAA", "BBB", "CCC"])
    member_dict = {date_str: member_df.loc[pd.Timestamp(date_str)].to_dict() for date_str in ("2024-01-02", "2024-01-03", "2024-01-09", "2024-01-10")}
    assert member_dict["2024-01-02"] == {"AAA": False, "BBB": False, "CCC": False}  # before the first row nobody is a member
    assert member_dict["2024-01-03"] == {"AAA": True, "BBB": False, "CCC": False}
    assert member_dict["2024-01-09"] == {"AAA": True, "BBB": False, "CCC": False}  # the later row is not visible yet
    assert member_dict["2024-01-10"] == {"AAA": False, "BBB": True, "CCC": False}


def test_reference_carries_over_days_without_candidates():
    turnover_df, candidate_df = _frames({"AAA": 4e6, "BBB": 1e6}, 500)
    candidate_df.iloc[121:] = False
    size_df, reference_ser = liquidity_size_and_reference(turnover_df, candidate_df)
    assert reference_ser.iloc[-1] == pytest.approx(1500.0)  # the 252-row window is empty by then
    assert size_df["AAA"].iloc[-1] == pytest.approx(2000.0 / 1500.0)


def test_size_at_t_never_reads_a_later_row():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2015-01-01", periods=700)
    turnover_df = pd.DataFrame(np.exp(rng.normal(14.0, 1.5, (700, 20))), index=index, columns=[f"S{i}" for i in range(20)])
    turnover_df = turnover_df.mask(rng.random((700, 20)) < 0.05)
    candidate_df = pd.DataFrame(rng.random((700, 20)) < 0.1, index=index, columns=turnover_df.columns)
    full_size_df, full_reference_ser = liquidity_size_and_reference(turnover_df, candidate_df)
    # *** CRITICAL*** the same m and reference from a frame cut at T as from the whole history.
    for row_int in (150, 300, 451, 699):
        cut_size_df, cut_reference_ser = liquidity_size_and_reference(turnover_df.iloc[: row_int + 1], candidate_df.iloc[: row_int + 1])
        pd.testing.assert_series_equal(full_size_df.iloc[row_int], cut_size_df.iloc[row_int])
        assert full_reference_ser.iloc[row_int] == cut_reference_ser.iloc[row_int]


def test_future_turnover_does_not_change_the_size_at_t():
    turnover_df, candidate_df = _frames({"AAA": 4e6, "BBB": 1e6}, 200)
    before_df, _ = liquidity_size_and_reference(turnover_df, candidate_df)
    turnover_df.iloc[151:] = 1e12
    after_df, _ = liquidity_size_and_reference(turnover_df, candidate_df)
    pd.testing.assert_frame_equal(before_df.iloc[:151], after_df.iloc[:151])
    assert not before_df.iloc[151:].equals(after_df.iloc[151:])


# ----------------------------------------------------------------------------------------------- the order arithmetic
class _FakePod:
    """Only what the mixin calls on its parent: records each order."""

    def __init__(self, *args, **kwargs):
        self.previous_total_value = 100_000.0
        self.current_bar = pd.Timestamp("2024-01-03")
        self.position_ser = pd.Series(dtype=float)
        self.call_list: list[tuple] = []
        self._accounting_policy_dict = {}

    def get_positions(self):
        return self.position_ser

    def compute_signals(self, pricing_data):
        return pricing_data

    def iterate(self, data, close, open_prices):
        pass

    def order_value(self, asset, value, **kwargs):
        self.call_list.append(("value", asset, value))
        return 1

    def order_target_value(self, asset, target, **kwargs):
        self.call_list.append(("target", asset, target))
        return 1

    def _place_parking_orders(self, data_df, close_row_ser, stock_value_after_orders_float, gate_open_bool):
        self.call_list.append(("parking", stock_value_after_orders_float, gate_open_bool))


class _SizedFakePod(LiquiditySizedPodMixin, _FakePod):
    def _liquidity_candidate_df(self, signal_df, stock_list):
        return pd.DataFrame(True, index=signal_df.index, columns=stock_list)


def _row(size_dict: dict[str, float], close_dict: dict[str, float] | None = None) -> pd.Series:
    row_dict = {(symbol_str, LIQUIDITY_SIZE_FIELD_STR): value_float for symbol_str, value_float in size_dict.items()}
    row_dict.update({(symbol_str, "Close"): value_float for symbol_str, value_float in (close_dict or {}).items()})
    return pd.Series(row_dict)


def _pod(row_ser: pd.Series, position_dict: dict[str, float] | None = None) -> _SizedFakePod:
    pod = _SizedFakePod()
    pod.position_ser = pd.Series(position_dict or {}, dtype=float)
    pod.iterate(None, row_ser, None)
    return pod


def test_entry_dollars_are_the_slot_times_the_size():
    pod = _pod(_row({"AAA": 1.5, "BBB": 0.5}))
    pod.order_value("AAA", 10_000.0, trade_id=1)
    pod.order_value("BBB", 10_000.0, trade_id=2)
    assert pod.call_list == [("value", "AAA", 15_000.0), ("value", "BBB", 5_000.0)]
    assert pod.entry_weight_by_date_dict == {("2024-01-03", "AAA"): 0.15, ("2024-01-03", "BBB"): 0.05}


def test_a_name_without_a_size_column_is_one_over_n():
    pod = _pod(_row({"AAA": 1.5}))
    pod.order_value("ZZZ", 10_000.0)
    assert pod.call_list == [("value", "ZZZ", 10_000.0)]


def test_stocks_stop_at_99_percent_and_the_recorded_weight_is_before_the_cap():
    pod = _pod(_row({"AAA": 2.0}, {"HELD": 100.0, "BIL": 90.0}), {"HELD": 850.0, "BIL": 1000.0})
    pod.order_value("AAA", 10_000.0)
    # Room = 99,000 - 85,000 held; BIL does not count against the stock cap.
    assert pod.call_list == [("value", "AAA", pytest.approx(14_000.0))]
    assert pod.entry_weight_by_date_dict == {("2024-01-03", "AAA"): 0.2}


def test_placed_dollars_use_up_the_room():
    pod = _pod(_row({"AAA": 2.0, "BBB": 1.0}, {"HELD": 100.0}), {"HELD": 780.0})
    pod.order_value("AAA", 10_000.0)  # 20,000 of the 21,000 room
    pod.order_value("BBB", 10_000.0)  # capped at the 1,000 left
    assert pod.call_list == [("value", "AAA", 20_000.0), ("value", "BBB", pytest.approx(1_000.0))]
    assert pod.entry_weight_by_date_dict == {("2024-01-03", "AAA"): 0.2, ("2024-01-03", "BBB"): 0.1}


def test_an_entry_under_half_a_percent_of_nav_is_skipped():
    pod = _pod(_row({"AAA": 2.0, "BBB": 1.0}, {"HELD": 100.0}), {"HELD": 788.0})
    pod.order_value("AAA", 10_000.0)  # 20,000 of the 20,200 room
    assert pod.order_value("BBB", 10_000.0) is None  # 200 left, under 500
    assert [call[1] for call in pod.call_list] == ["AAA"]
    assert pod.skipped_entry_symbol_list == ["BBB"] and ("2024-01-03", "BBB") not in pod.entry_weight_by_date_dict


def test_an_exit_of_the_same_day_frees_its_room():
    pod = _pod(_row({"AAA": 2.0}, {"HELD": 100.0, "GONE": 100.0}), {"HELD": 800.0, "GONE": 180.0})
    pod.order_target_value("GONE", 0, trade_id=9)
    pod.order_value("AAA", 10_000.0)
    assert pod.call_list == [("target", "GONE", 0), ("value", "AAA", pytest.approx(19_000.0))]


def test_a_held_name_with_no_close_counts_as_zero_like_the_research_pods():
    pod = _pod(_row({"AAA": 1.0}, {"HELD": 100.0, "GAP": np.nan}), {"HELD": 900.0, "GAP": 800.0})
    pod.order_value("AAA", 10_000.0)
    # Room = 99,000 - 90,000. Without the guard the room is NaN and min() keeps the whole 10,000 past the cap.
    assert pod.call_list == [("value", "AAA", pytest.approx(9_000.0))]


def test_parking_sees_the_true_entry_dollars():
    pod = _pod(_row({"AAA": 1.5, "BBB": 0.5, "CCC": 2.0}))
    pod.order_value("AAA", 10_000.0)
    pod.order_value("BBB", 10_000.0)
    pod.order_value("CCC", 10_000.0)
    # The pod passes its kept value plus 3 x 10,000; the true entries are 40,000.
    pod._place_parking_orders(None, None, 50_000.0 + 30_000.0, True)
    assert pod.call_list[-1] == ("parking", pytest.approx(90_000.0), True)


def test_a_new_session_starts_a_new_count():
    row_ser = _row({"AAA": 2.0})
    pod = _pod(row_ser)
    pod.order_value("AAA", 10_000.0)
    pod.iterate(None, row_ser, None)
    assert pod._placed_dollar_float == pod._base_dollar_float == 0.0 and pod._exit_symbol_set == set()


def test_the_signal_frame_keeps_the_pricing_attrs():
    # The engine's dividend guard reads the adjustment map from these attrs.
    pricing_df = pd.DataFrame({("AAA", "Turnover"): 1e6, ("BIL", "Close"): 91.0}, index=pd.bdate_range("2020-01-01", periods=5))
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {"AAA": "CAPITALSPECIAL"}
    signal_df = _SizedFakePod().compute_signals(pricing_df)
    assert signal_df.attrs == pricing_df.attrs
    assert ("AAA", LIQUIDITY_SIZE_FIELD_STR) in signal_df.columns


# ----------------------------------------------------------------------------------------------- the pods in the engine
def _flat_turnover(pricing_df: pd.DataFrame) -> pd.DataFrame:
    pricing_df = pricing_df.copy()
    for symbol_str in pricing_df.columns.get_level_values(0).unique():
        pricing_df[(symbol_str, "Turnover")] = 50_000_000.0
    return pricing_df


def _stock_buys_sized_by_the_rule(strategy) -> int:
    """Stock buys placed once the reference exists: before that m is 1 by default, and parity there proves nothing about the rule."""
    first_reference_ts = strategy.liquidity_reference_ser.first_valid_index()
    trade_df = _trades(strategy)
    return int(((trade_df["asset"] != "BIL") & (trade_df["amount"] > 0) & (trade_df["bar"] > first_reference_ts)).sum())


def test_dv2_at_constant_liquidity_equals_the_one_over_n_pod():
    """m = 1 everywhere and a book that never fills: every order, BIL included, is the 1/N pod's."""
    pricing_df = _flat_turnover(_synthetic_market(8, 700, 11, with_parking_bool=True))
    calendar = pricing_df.index[400:]
    gate_ser = pd.Series(True, index=pricing_df.index)
    plain = _run(_dv2(DV2VixGatedStrategy, pricing_df, gate_ser=gate_ser, parking_bool=True), pricing_df, calendar)
    sized = _run(_dv2(DV2VixGatedLiquidityStrategy, pricing_df, gate_ser=gate_ser, parking_bool=True), pricing_df, calendar)
    assert _stock_buys_sized_by_the_rule(sized) > 5
    pd.testing.assert_frame_equal(_trades(plain), _trades(sized))
    np.testing.assert_allclose(plain.results["total_value"].astype(float), sized.results["total_value"].astype(float))


class _EveryNameACandidateHPI(HPIVoteVixGatedLiquidityStrategy):
    """A market this small has too few HPI candidates for a reference; this gives the size a real one and changes no order rule."""

    def _liquidity_candidate_df(self, signal_df, stock_list):
        return pd.DataFrame(True, index=signal_df.index, columns=stock_list)


def test_hpi_at_constant_liquidity_equals_the_one_over_n_pod():
    pricing_df = _flat_turnover(_synthetic_market(8, 1700, 21, with_parking_bool=True))
    calendar = pricing_df.index[1330:]
    gate_ser = pd.Series(True, index=pricing_df.index)
    plain = _run(_hpi(HPIVoteVixGatedStrategy, pricing_df, gate_ser=gate_ser, parking_bool=True), pricing_df, calendar)
    sized = _run(_hpi(_EveryNameACandidateHPI, pricing_df, gate_ser=gate_ser, parking_bool=True), pricing_df, calendar)
    assert _stock_buys_sized_by_the_rule(sized) > 5
    pd.testing.assert_frame_equal(_trades(plain), _trades(sized))
    np.testing.assert_allclose(plain.results["total_value"].astype(float), sized.results["total_value"].astype(float))


def _varied_market(session_int: int, seed_int: int) -> pd.DataFrame:
    pricing_df = _synthetic_market(40, session_int, seed_int, with_parking_bool=True, drift_float=0.0010, market_vol_float=0.012)
    rng = np.random.default_rng(seed_int)
    for symbol_str in pricing_df.columns.get_level_values(0).unique():
        pricing_df[(symbol_str, "Turnover")] = pricing_df[(symbol_str, "Turnover")] * float(np.exp(rng.normal(0.0, 1.5)))
    return pricing_df


def test_dv2_records_each_entry_at_its_size_over_ten():
    pricing_df = _varied_market(700, 12)
    calendar = pricing_df.index[400:]
    gate_ser = pd.Series(True, index=pricing_df.index)
    sized = _dv2(DV2VixGatedLiquidityStrategy, pricing_df, gate_ser=gate_ser, parking_bool=True)
    signal_df = sized.compute_signals(pricing_df)
    _run(sized, pricing_df, calendar)
    weight_ser = pd.Series(sized.entry_weight_by_date_dict)
    assert len(weight_ser) > 20
    assert weight_ser.nunique() > 20  # not an equal slot
    assert weight_ser.between(0.05 - 1e-12, 0.20 + 1e-12).all()
    # Each recorded weight is m at the decision close over 10.
    for (fill_str, symbol_str), weight_float in list(weight_ser.items())[:20]:
        position_int = int(pricing_df.index.get_loc(pd.Timestamp(fill_str)))
        assert weight_float == pytest.approx(signal_df[(symbol_str, LIQUIDITY_SIZE_FIELD_STR)].iloc[position_int - 1] / 10.0)
    stock_buy_df = _trades(sized).query("asset != 'BIL' and amount > 0")
    assert {(pd.Timestamp(bar).date().isoformat(), str(asset)) for bar, asset in weight_ser.index} <= {
        (pd.Timestamp(bar).date().isoformat(), str(asset)) for bar, asset in zip(stock_buy_df["bar"], stock_buy_df["asset"])}


def _broad_case(session_int: int, name_int: int, seed_int: int, pod_str: str):
    """Enough names for both pods to have many candidates, with a late listing, an early delisting, Turnover and Volume gaps and membership changes."""
    pricing_df = _synthetic_market(name_int, session_int, seed_int, with_parking_bool=True, drift_float=0.0010, market_vol_float=0.012)
    rng = np.random.default_rng(seed_int)
    stock_list = [f"S{i:03d}" for i in range(name_int)]
    for symbol_str in stock_list:
        pricing_df[(symbol_str, "Turnover")] = pricing_df[(symbol_str, "Turnover")] * float(np.exp(rng.normal(0.0, 1.5)))
    field_list = ["Open", "High", "Low", "Close", "Volume", "Turnover", "Unadjusted Close", "Dividend"]
    pricing_df.loc[pricing_df.index[: session_int // 2], [("S001", field_str) for field_str in field_list]] = np.nan
    pricing_df.loc[pricing_df.index[-session_int // 4:], [("S002", field_str) for field_str in field_list]] = np.nan
    for symbol_str in stock_list[3:8]:
        pricing_df.loc[pricing_df.index[rng.integers(0, session_int, 40)], (symbol_str, "Turnover")] = np.nan
        pricing_df.loc[pricing_df.index[rng.integers(0, session_int, 20)], (symbol_str, "Volume")] = np.nan
    row_idx = pricing_df.index[::5]
    universe_df = pd.DataFrame(1, index=row_idx, columns=stock_list)
    universe_df.iloc[: len(row_idx) // 3, universe_df.columns.get_loc("S003")] = 0
    universe_df.iloc[len(row_idx) // 2:, universe_df.columns.get_loc("S004")] = 0
    gate_ser = pd.Series(True, index=pricing_df.index)
    strategy = (_dv2(DV2VixGatedLiquidityStrategy, pricing_df, gate_ser=gate_ser, parking_bool=True) if pod_str == "dv2"
                else _hpi(HPIVoteVixGatedLiquidityStrategy, pricing_df, gate_ser=gate_ser, parking_bool=True))
    strategy.universe_df = universe_df
    return strategy, pricing_df, universe_df, strategy.compute_signals(pricing_df)


@pytest.fixture(scope="module")
def dv2_case():
    return _broad_case(700, 40, 3, "dv2")


@pytest.fixture(scope="module")
def hpi_case():
    return _broad_case(2200, 160, 4, "hpi")


def _candidate_df(strategy, signal_df):
    # As compute_signals calls it: the frame before the size columns exist.
    plain_df = signal_df.drop(columns=[col for col in signal_df.columns if col[1] == LIQUIDITY_SIZE_FIELD_STR])
    return strategy._liquidity_candidate_df(plain_df, sorted({str(symbol_obj) for symbol_obj, _ in plain_df.columns if not str(symbol_obj).startswith("$")}))


def test_dv2_candidates_equal_the_pods_own_list_on_every_session(dv2_case):
    strategy, _, _, signal_df = dv2_case
    candidate_df = _candidate_df(strategy, signal_df)
    assert candidate_df.to_numpy().sum() > 100
    for date_ts in signal_df.index:
        strategy.previous_bar = date_ts
        assert set(strategy.get_opportunities(signal_df.loc[date_ts])) == set(candidate_df.columns[candidate_df.loc[date_ts].to_numpy()]), date_ts


def test_hpi_candidates_equal_the_pods_own_list_on_every_session(hpi_case):
    strategy, _, universe_df, signal_df = hpi_case
    candidate_df = _candidate_df(strategy, signal_df)
    assert candidate_df.to_numpy().sum() > 100
    for date_ts in signal_df.index:
        strategy.previous_bar = date_ts
        own_set = set(strategy.get_opportunity_list(signal_df.loc[date_ts], get_asof_universe_symbol_set(universe_df, date_ts)))
        assert own_set == set(candidate_df.columns[candidate_df.loc[date_ts].to_numpy()]), date_ts


@pytest.mark.parametrize("fixture_str", ["dv2_case", "hpi_case"])
def test_the_engine_audit_finds_no_lookahead_in_the_size(fixture_str, request, monkeypatch):
    strategy, pricing_df, _, signal_df = request.getfixturevalue(fixture_str)
    size_df = signal_df.xs(LIQUIDITY_SIZE_FIELD_STR, axis=1, level=1)
    # The audit must compare real sizes at its last sampled row, not 1.0 with 1.0.
    assert np.isfinite(strategy.liquidity_reference_ser.iloc[-1]) and size_df.iloc[-1].nunique() > 10
    audit_col_list = [col for col in strategy.signal_audit_fields(pricing_df, signal_df) if col[1] == LIQUIDITY_SIZE_FIELD_STR]
    assert len(audit_col_list) == size_df.shape[1]
    # Only the size: on a truncated frame the parent pod leaves out its own fields for a name not listed yet.
    monkeypatch.setattr(strategy, "signal_audit_fields", lambda *_: audit_col_list)
    strategy.audit_signals(pricing_df, signal_df, sample_size=6)


def test_one_stock_without_turnover_is_simply_not_a_candidate(dv2_case):
    _, pricing_df, universe_df, _ = dv2_case
    framed_df = pricing_df.drop(columns=[("S010", "Turnover")])
    strategy = _dv2(DV2VixGatedLiquidityStrategy, framed_df, gate_ser=pd.Series(True, index=framed_df.index), parking_bool=True)
    strategy.universe_df = universe_df
    signal_df = strategy.compute_signals(framed_df)
    assert (signal_df[("S010", LIQUIDITY_SIZE_FIELD_STR)] == 1.0).all()
    assert not _candidate_df(strategy, signal_df)["S010"].any()
    for date_ts in signal_df.index[300:]:
        strategy.previous_bar = date_ts
        assert "S010" not in strategy.get_opportunities(signal_df.loc[date_ts])


@pytest.mark.parametrize("pod_str", ["dv2", "hpi"])
def test_a_frame_without_turnover_stops_the_pod_instead_of_sizing_at_one(pod_str):
    pricing_df = _synthetic_market(4, 300, 15, with_parking_bool=True)
    pricing_df = pricing_df.drop(columns=[(f"S{i:03d}", "Turnover") for i in range(4)])
    gate_ser = pd.Series(True, index=pricing_df.index)
    strategy = (_dv2(DV2VixGatedLiquidityStrategy, pricing_df, gate_ser=gate_ser, parking_bool=True) if pod_str == "dv2"
                else _hpi(HPIVoteVixGatedLiquidityStrategy, pricing_df, gate_ser=gate_ser, parking_bool=True))
    with pytest.raises(ValueError, match="requires Turnover"):
        strategy.compute_signals(pricing_df)


def test_parking_etfs_do_not_need_turnover_and_are_not_candidates():
    pricing_df = _synthetic_market(4, 300, 16, with_parking_bool=True)
    pricing_df = pricing_df.drop(columns=[("BIL", "Turnover"), ("SPMO", "Turnover")])
    strategy = _dv2(DV2VixGatedLiquidityStrategy, pricing_df, gate_ser=pd.Series(True, index=pricing_df.index), parking_bool=True)
    signal_df = strategy.compute_signals(pricing_df)
    assert (signal_df[("BIL", LIQUIDITY_SIZE_FIELD_STR)] == 1.0).all()
    assert not _candidate_df(strategy, signal_df)[["BIL", "SPMO"]].any().any()


def test_a_ready_check_fails_closed_without_a_reference():
    pricing_df = _synthetic_market(4, 80, 17, with_parking_bool=True)
    strategy = _dv2(DV2VixGatedLiquidityStrategy, pricing_df, gate_ser=pd.Series(True, index=pricing_df.index), parking_bool=True)
    signal_df = strategy.compute_signals(pricing_df)
    with pytest.raises(ValueError, match="no usable reference"):
        strategy.validate_liquidity_sizing_ready(signal_df, pricing_df.index[-1])


def _ready_case(row_int: int = 200, name_int: int = 10):
    index = pd.bdate_range(end="2024-06-28", periods=row_int)
    stock_list = [f"S{i}" for i in range(name_int)]
    signal_df = pd.DataFrame({(symbol_str, "Turnover"): 1e6 for symbol_str in stock_list}, index=index)
    strategy = DV2VixGatedLiquidityStrategy(name="ready", benchmarks=[], capital_base=100_000.0)
    strategy.universe_df = pd.DataFrame(1, index=index, columns=stock_list)
    strategy.liquidity_reference_ser = pd.Series(1000.0, index=index)
    return strategy, signal_df, index[-1]


def test_ready_check_passes_a_clean_decision_and_reports_its_coverage():
    strategy, signal_df, date_ts = _ready_case()
    assert strategy.validate_liquidity_sizing_ready(signal_df, date_ts) == 1.0
    signal_df.iloc[-1, 0] = np.nan
    assert strategy.validate_liquidity_sizing_ready(signal_df, date_ts) == pytest.approx(0.9)


@pytest.mark.parametrize("bad_int,blocks_bool", [(1, False), (2, False), (3, True)])
def test_ready_check_blocks_under_80_percent_of_members(bad_int, blocks_bool):
    strategy, signal_df, date_ts = _ready_case()
    signal_df.iloc[-31:-1, :bad_int] = np.nan  # 33 of the last 63 rows are left, under the 42 that m needs; the last row is fine
    if blocks_bool:
        with pytest.raises(ValueError, match="it has 70%"):
            strategy.validate_liquidity_sizing_ready(signal_df, date_ts)
    else:
        strategy.validate_liquidity_sizing_ready(signal_df, date_ts)


@pytest.mark.parametrize("row_slice,value_float", [
    pytest.param(slice(-63, None), 0.0, id="zero_window"),
    pytest.param(slice(-63, -1), 0.0, id="zero_window_with_a_good_last_row"),
    pytest.param(slice(-31, -1), np.inf, id="infinite_rows"),
    pytest.param(slice(-31, -1), -1.0, id="negative_rows"),
    pytest.param(slice(-1, None), np.nan, id="missing_last_row"),
    pytest.param(slice(-1, None), 0.0, id="zero_last_row"),
    pytest.param(slice(-1, None), np.inf, id="infinite_last_row"),
])
def test_ready_check_counts_only_finite_positive_turnover(row_slice, value_float):
    strategy, signal_df, date_ts = _ready_case()
    signal_df.iloc[row_slice] = value_float
    with pytest.raises(ValueError, match="needs usable Turnover"):
        strategy.validate_liquidity_sizing_ready(signal_df, date_ts)


def test_ready_check_needs_42_of_the_last_63_rows():
    strategy, signal_df, date_ts = _ready_case()
    signal_df.iloc[-22:-1] = 0.0  # exactly 42 positive rows are left
    strategy.validate_liquidity_sizing_ready(signal_df, date_ts)
    signal_df.iloc[-23] = 0.0
    with pytest.raises(ValueError, match="needs usable Turnover"):
        strategy.validate_liquidity_sizing_ready(signal_df, date_ts)


def test_ready_check_counts_a_stock_without_the_column_as_missing():
    strategy, signal_df, date_ts = _ready_case()
    strategy.universe_df["GONE0"] = 1  # members with no column at all: 10 of 13 have usable Turnover, under 80%
    strategy.universe_df["GONE1"] = 1
    strategy.universe_df["GONE2"] = 1
    signal_df[[("GONE0", "Close"), ("GONE1", "Close"), ("GONE2", "Close")]] = 50.0
    with pytest.raises(ValueError, match="it has 77%"):
        strategy.validate_liquidity_sizing_ready(signal_df, date_ts)


def test_ready_check_reads_the_decision_date_and_rejects_a_zero_reference():
    strategy, signal_df, date_ts = _ready_case()
    strategy.liquidity_reference_ser.iloc[-1] = 0.0
    with pytest.raises(ValueError, match="no usable reference"):
        strategy.validate_liquidity_sizing_ready(signal_df, date_ts)
    signal_df.iloc[-1] = np.nan
    strategy.validate_liquidity_sizing_ready(signal_df, signal_df.index[-2])  # one row earlier the reference is fine, and a later row does not count
    strategy.liquidity_reference_ser = pd.Series(1000.0, index=signal_df.index[:-1])
    with pytest.raises(ValueError, match="no usable reference"):
        strategy.validate_liquidity_sizing_ready(signal_df, date_ts)  # an older row does not stand in for T


def test_ready_check_counts_only_members_at_t():
    strategy, signal_df, date_ts = _ready_case()
    strategy.universe_df.iloc[-1, :3] = 0  # three former members, whose Turnover no longer matters
    signal_df.iloc[-31:-1, :3] = np.nan
    strategy.validate_liquidity_sizing_ready(signal_df, date_ts)
    strategy.universe_df.iloc[-1, :] = 0
    with pytest.raises(ValueError, match="no members"):
        strategy.validate_liquidity_sizing_ready(signal_df, date_ts)


@pytest.mark.parametrize("sized_fn,plain_fn", [
    (dv2_vix_gated_liq.build_dv2_capsule_liquidity_strategy, build_dv2_capsule_strategy),
    (hpi_vote_vix_gated_liq.build_hpi_capsule_liquidity_strategy, build_hpi_capsule_strategy),
])
def test_sized_builder_configures_what_the_one_over_n_builder_does(sized_fn, plain_fn):
    kwarg_dict = dict(strategy_name_str="same", parking_enabled_bool=True, spmo_parking_enabled_bool=False, universe_df=pd.DataFrame(), vix_close_ser=pd.Series(dtype=float), capital_base_float=250_000.0, slippage_float=0.0007)
    sized, plain = sized_fn(**kwarg_dict), plain_fn(**kwarg_dict)
    sizing_key_set = set(vars(_SizedFakePod())) - set(vars(_FakePod()))

    def config(strategy):
        return {key_str: value_obj for key_str, value_obj in vars(strategy).items()
                if key_str not in sizing_key_set and isinstance(value_obj, (bool, int, float, str, list, tuple, dict, type(None)))}

    assert config(sized) == config(plain)
    assert config(sized)["_slippage"] == 0.0007 and config(sized)["_capital_base"] == 250_000.0


@pytest.mark.parametrize("pod_module,run_str,loader_str", [
    (dv2_vix_gated_liq, "run_dv2_capsule_liquidity_pod", "load_pricing_data"),
    (hpi_vote_vix_gated_liq, "run_hpi_capsule_liquidity_pod", "load_hpi_capsule_pricing_data"),
])
def test_run_keeps_bil_parking_and_the_requested_window(monkeypatch, pod_module, run_str, loader_str):
    pricing_df = pd.DataFrame({("S0", "Close"): 1.0}, index=pd.bdate_range("2004-01-01", "2006-12-29"))
    captured_dict = {}
    monkeypatch.setattr(pod_module, loader_str, lambda end_date_str: (captured_dict.update(end_str=end_date_str) or pricing_df, pd.DataFrame()))
    monkeypatch.setattr(pod_module, "load_vix_close_ser", lambda end_date_str: pd.Series(dtype=float))
    monkeypatch.setattr(pod_module, "run_daily", lambda strategy, frame_df, calendar_idx, **_: captured_dict.update(strategy=strategy, frame_df=frame_df, calendar_idx=calendar_idx))
    strategy = getattr(pod_module, run_str)(strategy_name_str="window", show_display_bool=False, save_results_bool=False, backtest_start_date_str="2005-01-03", capital_base_float=250_000.0, end_date_str="2006-06-30", slippage_float=0.0007)
    assert captured_dict["strategy"] is strategy and captured_dict["end_str"] == "2006-06-30"
    assert strategy.parking_enabled_bool and not strategy.spmo_parking_enabled_bool
    # Pre-start history stays in the frame for m's windows; trading starts at the requested session.
    assert captured_dict["frame_df"].index[0] == pd.Timestamp("2004-01-01") and captured_dict["calendar_idx"][0] == pd.Timestamp("2005-01-03")
    assert strategy._capital_base == 250_000.0 and strategy._slippage == 0.0007


# ----------------------------------------------------------------------------------------------- entry points and registry
ENTRY_POINT_LIST = [("strategy_mr_dv2_vix_gated_liq_bil", "dv2", dv2_vix_gated_liq), ("strategy_mr_hpi_vote_vix_gated_liq_bil", "hpi", hpi_vote_vix_gated_liq)]


@pytest.mark.parametrize("module_name_str,pod_str,pod_module", ENTRY_POINT_LIST)
def test_entry_points_fix_the_name_and_pass_the_run_arguments(monkeypatch, module_name_str, pod_str, pod_module):
    module_obj = importlib.import_module(f"strategies.mr_capsule.{module_name_str}")
    captured_dict, capacity_dict, timing_dict = {}, {}, {}
    monkeypatch.setattr(module_obj, f"run_{pod_str}_capsule_liquidity_pod", lambda **kwarg_dict: captured_dict.update(kwarg_dict) or "ran")
    monkeypatch.setattr(module_obj, f"build_{pod_str}_capsule_liquidity_capacity_analysis_inputs", lambda **kwarg_dict: capacity_dict.update(kwarg_dict) or "capacity")
    monkeypatch.setattr(module_obj, f"build_{pod_str}_capsule_liquidity_execution_timing_analysis_inputs", lambda **kwarg_dict: timing_dict.update(kwarg_dict) or "timing")
    assert module_obj.run_variant(show_display_bool=False, save_results_bool=False, end_date_str="2026-01-02", capital_base_float=250_000.0, backtest_start_date_str="2005-01-03", slippage_float=0.0007) == "ran"
    assert captured_dict["strategy_name_str"] == module_name_str == module_obj.STRATEGY_NAME_STR
    assert captured_dict["end_date_str"] == "2026-01-02" and captured_dict["capital_base_float"] == 250_000.0
    assert captured_dict["backtest_start_date_str"] == "2005-01-03" and captured_dict["slippage_float"] == 0.0007
    assert module_obj.build_capacity_analysis_inputs(capital_base_float=250_000.0) == "capacity"
    assert module_obj.build_execution_timing_analysis_inputs() == "timing"
    assert capacity_dict["strategy_name_str"] == timing_dict["strategy_name_str"] == module_name_str


@pytest.mark.parametrize("module_name_str,pod_str,pod_module", ENTRY_POINT_LIST)
def test_entry_points_are_wired_and_listed_in_bench(module_name_str, pod_str, pod_module):
    module_str = f"strategies.mr_capsule.{module_name_str}"
    assert tier_for(module_str) is MaturityTier.WIRED
    assert catalog.get_strategy_by_module(module_str) is not None


@pytest.mark.parametrize("module_str", ["liquidity_sizing", "dv2_vix_gated_liq", "hpi_vote_vix_gated_liq"])
def test_shared_modules_are_not_bench_strategies(module_str):
    assert catalog.get_strategy_by_module(f"strategies.mr_capsule.{module_str}") is None


def test_book_loads_in_the_portfolio_manager():
    config_obj = PortfolioManager.from_yaml(Path(__file__).resolve().parents[1] / "portfolios" / "mr_capsule_liq_bil.yaml").config
    assert [pod_obj.strategy_import_str for pod_obj in config_obj.pod_config_list] == [
        "strategies.mr_capsule.strategy_mr_dv2_vix_gated_liq_bil",
        "strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_liq_bil",
    ]
    assert config_obj.weight_list == [0.5, 0.5] and config_obj.max_workers_int == 1


def test_the_one_over_n_modules_load_without_the_sizing():
    # A fresh interpreter: this test module has already imported the sizing here.
    code_str = (
        "import sys; import alpha.live.runner, alpha.live.mr_capsule_adapter, strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil, strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_bil;"
        "print([name_str for name_str in sys.modules if name_str.endswith(('liquidity_sizing', 'vix_gated_liq', 'liq_bil', 'timing_hook'))])"
    )
    completed_obj = subprocess.run([sys.executable, "-c", code_str], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1], check=True)
    assert completed_obj.stdout.strip().splitlines()[-1] == "[]"


def test_the_one_over_n_pods_import_nothing_of_the_sizing():
    for module_name_str in ("dv2_vix_gated", "hpi_vote_vix_gated", "capsule_pod", "strategy_mr_dv2_vix_gated_bil", "strategy_mr_hpi_vote_vix_gated_bil"):
        source_str = (Path(liquidity_sizing.__file__).parent / f"{module_name_str}.py").read_text(encoding="utf-8")
        assert "liquidity_sizing" not in source_str and "vix_gated_liq" not in source_str
