"""The momentum capsule legs' rules on a synthetic panel: the two-book calendar, no look-ahead, the rank buffer, the weights and the replay."""
from dataclasses import replace

import exchange_calendars as calendar_module
import numpy as np
import pandas as pd
import pytest

from alpha.live import scheduler_utils
from strategies.momentum import momentum_capsule_legs as legs_module
from strategies.momentum.momentum_capsule_legs import (
    LEG1_CONFIG,
    LEG2_CONFIG,
    LEG3_CONFIG,
    LegPanel,
    TargetPath,
    build_decision_df,
    build_rebalance_schedule_df,
    build_target_path,
    daily_return_mat,
    decision_date_by_book_dict,
    inverse_vol_weight_vec,
    pick_with_rank_buffer,
    traded_pricing_df,
)

SECTOR_LABEL_LIST = ["45000000", "35000000", "40000000", "25000000"]
CONFIG_DICT = {"leg1": LEG1_CONFIG, "leg2": LEG2_CONFIG, "leg3": LEG3_CONFIG}


def sessions(start_str, end_str):
    return pd.DatetimeIndex(calendar_module.get_calendar("XNYS").sessions_in_range(start_str, end_str)).tz_localize(None)


def make_panel(seed_int=3, column_count_int=48, start_str="2022-01-03", end_str="2024-03-28"):
    date_index = sessions(start_str, end_str)
    rng = np.random.default_rng(seed_int)
    row_count_int = len(date_index)
    drift_vec = np.linspace(-.0008, .0014, column_count_int)
    return_mat = drift_vec + .015 * rng.standard_normal((row_count_int, column_count_int))
    close_mat = 100.0 * np.exp(np.cumsum(return_mat, axis=0))
    symbol_vec = np.array([f"S{column_int:02d}" for column_int in range(column_count_int)], dtype=object)
    sector_label_vec = np.array([SECTOR_LABEL_LIST[column_int % 4] for column_int in range(column_count_int)], dtype=object)
    group_label_vec = np.array([f"{SECTOR_LABEL_LIST[column_int % 4][:2]}10" if column_int % 8 < 4 else f"{SECTOR_LABEL_LIST[column_int % 4][:2]}20" for column_int in range(column_count_int)], dtype=object)
    sector_ret_df = pd.DataFrame(.0004 + .01 * rng.standard_normal((row_count_int, 4)), index=date_index, columns=SECTOR_LABEL_LIST)
    return LegPanel(
        date_index=date_index,
        symbol_vec=symbol_vec,
        close_mat=close_mat,
        high_mat=close_mat * 1.01,
        low_mat=close_mat * .99,
        raw_close_mat=close_mat * 2.0,
        dollar_volume_mat=np.full(close_mat.shape, 5e7),
        member_mat=np.ones(close_mat.shape, dtype=bool),
        sector_label_vec=sector_label_vec,
        group_label_vec=group_label_vec,
        gate_close_vec=100.0 * np.exp(np.cumsum(.0009 + .004 * rng.standard_normal(row_count_int))),
        market_ret_vec=.0005 + .009 * rng.standard_normal(row_count_int),
        sector_ret_df=sector_ret_df,
        vxn_vec=np.full(row_count_int, 25.0),
    )


def cut_panel(panel, last_row_int):
    stop_int = last_row_int + 1
    return LegPanel(
        date_index=panel.date_index[:stop_int], symbol_vec=panel.symbol_vec, close_mat=panel.close_mat[:stop_int], high_mat=panel.high_mat[:stop_int],
        low_mat=panel.low_mat[:stop_int], raw_close_mat=panel.raw_close_mat[:stop_int], dollar_volume_mat=panel.dollar_volume_mat[:stop_int],
        member_mat=panel.member_mat[:stop_int], sector_label_vec=panel.sector_label_vec, group_label_vec=panel.group_label_vec,
        gate_close_vec=panel.gate_close_vec[:stop_int], market_ret_vec=panel.market_ret_vec[:stop_int],
        sector_ret_df=panel.sector_ret_df.iloc[:stop_int], vxn_vec=panel.vxn_vec[:stop_int],
    )


def test_decisions_fall_on_the_last_session_and_ten_sessions_before_it():
    date_index = sessions("2026-08-03", "2026-10-30")
    decision_df = build_decision_df(date_index)
    assert [str(date_ts.date()) for date_ts in decision_df["decision_date_ts"]] == ["2026-08-17", "2026-08-31", "2026-09-16", "2026-09-30", "2026-10-16", "2026-10-30"]
    assert decision_df["book_offset_int"].tolist() == [10, 0, 10, 0, 10, 0]
    assert pd.isna(decision_df["execution_date_ts"].iloc[-1])
    schedule_df = build_rebalance_schedule_df(date_index, "2026-08-01")
    assert schedule_df.loc[pd.Timestamp("2026-10-19"), "decision_date_ts"] == pd.Timestamp("2026-10-16")
    assert pd.Timestamp("2026-10-30") not in schedule_df["decision_date_ts"].to_list()


def test_an_unfinished_month_has_its_t10_decision_from_the_schedule_and_no_t0():
    date_dict = decision_date_by_book_dict(sessions("2026-08-03", "2026-10-16"))
    assert date_dict[10][-1] == pd.Timestamp("2026-10-16")
    assert date_dict[0][-1] == pd.Timestamp("2026-09-30")
    date_dict = decision_date_by_book_dict(sessions("2026-08-03", "2026-10-15"))
    assert date_dict[10][-1] == pd.Timestamp("2026-09-16")


def test_the_live_clock_agrees_with_the_strategy_calendar_on_every_session():
    calendar_obj = calendar_module.get_calendar("XNYS")
    first_ts = max(pd.Timestamp(calendar_obj.first_session), pd.Timestamp("2007-01-01"))
    session_idx = sessions((first_ts + pd.offsets.MonthBegin(1)).normalize(), "2026-10-30")
    decision_set = set(build_decision_df(session_idx)["decision_date_ts"])
    clock_set = {session_ts for session_ts in session_idx if scheduler_utils.is_month_swap_session_bool(session_ts, "XNYS")}
    assert clock_set == decision_set
    assert len(clock_set) == 2 * len(session_idx.to_period("M").unique())


@pytest.mark.parametrize("leg_str", list(CONFIG_DICT))
def test_a_decision_does_not_change_when_later_rows_arrive(leg_str):
    config_obj = CONFIG_DICT[leg_str]
    panel = make_panel()
    full_path_obj = build_target_path(config_obj, panel)
    decision_list = sorted(full_path_obj.target_by_decision_dict)[-8:]
    assert sum(len(full_path_obj.book_by_decision_dict[decision_ts]) > 0 for decision_ts in decision_list) >= 4
    for decision_ts in decision_list:
        row_int = panel.date_index.get_loc(decision_ts)
        cut_path_obj = build_target_path(config_obj, cut_panel(panel, row_int))
        pd.testing.assert_series_equal(cut_path_obj.target_by_decision_dict[decision_ts], full_path_obj.target_by_decision_dict[decision_ts])
        pd.testing.assert_series_equal(cut_path_obj.book_by_decision_dict[decision_ts], full_path_obj.book_by_decision_dict[decision_ts])


@pytest.mark.parametrize("leg_str", list(CONFIG_DICT))
def test_changing_prices_after_t_cannot_change_decisions_through_t(leg_str):
    config_obj = CONFIG_DICT[leg_str]
    panel = make_panel()
    full_path_obj = build_target_path(config_obj, panel)
    decision_ts = sorted(full_path_obj.target_by_decision_dict)[-5]
    row_int = panel.date_index.get_loc(decision_ts)
    for mat in (panel.close_mat, panel.high_mat, panel.low_mat, panel.raw_close_mat):
        mat[row_int + 1:] *= 3.0
    panel.gate_close_vec[row_int + 1:] *= .1
    panel.market_ret_vec[row_int + 1:] = .05
    panel.sector_ret_df.iloc[row_int + 1:] = -.05
    panel.vxn_vec[row_int + 1:] = 90.0
    shocked_path_obj = build_target_path(config_obj, panel)
    for earlier_ts in [date_ts for date_ts in full_path_obj.target_by_decision_dict if date_ts <= decision_ts]:
        pd.testing.assert_series_equal(shocked_path_obj.target_by_decision_dict[earlier_ts], full_path_obj.target_by_decision_dict[earlier_ts])


@pytest.mark.parametrize("leg_str", list(CONFIG_DICT))
def test_a_later_split_or_special_dividend_cannot_change_a_decision(leg_str):
    """*** CRITICAL *** Future-action invariance: a corporate action after T rescales one name's whole adjusted history before it. Scores read ratios, and the raw $10 and dollar volume filters read unadjusted fields, so no decision may move."""
    config_obj = CONFIG_DICT[leg_str]
    panel = make_panel()
    full_path_obj = build_target_path(config_obj, panel)
    for column_int in (3, 17, 40):
        for mat in (panel.close_mat, panel.high_mat, panel.low_mat):
            mat[:, column_int] *= 0.37
    rescaled_path_obj = build_target_path(config_obj, panel)
    for decision_ts, target_ser in full_path_obj.target_by_decision_dict.items():
        pd.testing.assert_series_equal(rescaled_path_obj.target_by_decision_dict[decision_ts], target_ser, check_exact=False, atol=1e-9)


@pytest.mark.parametrize("leg_str", list(CONFIG_DICT))
def test_each_book_reads_only_its_own_earlier_picks(leg_str):
    """Removing every decision of one book from consideration must leave the other book's picks unchanged: the buffers do not mix."""
    config_obj = CONFIG_DICT[leg_str]
    panel = make_panel()
    path_obj = build_target_path(config_obj, panel)
    decision_list = sorted(path_obj.book_by_decision_dict)
    assert {path_obj.book_offset_by_decision_dict[decision_ts] for decision_ts in decision_list} == {0, 10}
    t10_book_dict = {decision_ts: book_ser for decision_ts, book_ser in path_obj.book_by_decision_dict.items() if path_obj.book_offset_by_decision_dict[decision_ts] == 10}
    held_list = []
    real_pick_fn = legs_module.pick_with_rank_buffer

    def spy_pick_fn(ranked_vec, held_set, symbol_vec, label_vec, max_positions_int, sector_cap_int, hold_rank_int):
        held_list.append(frozenset(held_set))
        return real_pick_fn(ranked_vec, held_set, symbol_vec, label_vec, max_positions_int, sector_cap_int, hold_rank_int)
    legs_module.pick_with_rank_buffer = spy_pick_fn
    try:
        build_target_path(config_obj, panel)
    finally:
        legs_module.pick_with_rank_buffer = real_pick_fn
    # The n-th pick call of the run: a T10 decision sees the previous T10 book, not the T0 book between them.
    open_decision_list = [decision_ts for decision_ts in decision_list if path_obj.evidence_by_decision_dict[decision_ts]["gate_open_bool"] and path_obj.evidence_by_decision_dict[decision_ts]["finite_score_member_count_int"] > 0]
    assert len(open_decision_list) == len(held_list)
    previous_ts = None
    for decision_ts, held_set in zip(open_decision_list, held_list):
        if decision_ts in t10_book_dict and previous_ts is not None:
            assert held_set == frozenset(t10_book_dict[previous_ts].index)
        if decision_ts in t10_book_dict:
            previous_ts = decision_ts


def test_nasdaq_leg_weights_follow_vxn_as_of_t_and_floor_at_a_quarter():
    panel = make_panel()
    panel.vxn_vec[:] = 44.0
    path_obj = build_target_path(LEG1_CONFIG, panel)
    book_ser = next(book_ser for book_ser in path_obj.book_by_decision_dict.values() if len(book_ser))
    assert book_ser.iloc[0] == pytest.approx(0.5 / 10.0)
    panel.vxn_vec[:] = 200.0
    book_ser = next(book_ser for book_ser in build_target_path(LEG1_CONFIG, panel).book_by_decision_dict.values() if len(book_ser))
    assert book_ser.iloc[0] == pytest.approx(0.25 / 10.0)
    panel.vxn_vec[:] = 12.0
    book_ser = next(book_ser for book_ser in build_target_path(LEG1_CONFIG, panel).book_by_decision_dict.values() if len(book_ser))
    assert book_ser.iloc[0] == pytest.approx(1.0 / 10.0)


def test_leg3_filters_read_raw_price_and_dollar_volume_at_t():
    panel = make_panel()
    path_obj = build_target_path(LEG3_CONFIG, panel)
    decision_ts = next(date_ts for date_ts in sorted(path_obj.book_by_decision_dict, reverse=True) if len(path_obj.book_by_decision_dict[date_ts]))
    row_int = panel.date_index.get_loc(decision_ts)
    picked_str = path_obj.book_by_decision_dict[decision_ts].index[0]
    column_int = list(panel.symbol_vec).index(picked_str)
    raw_panel = make_panel()
    raw_panel.raw_close_mat[row_int, column_int] = 9.99
    assert picked_str not in build_target_path(LEG3_CONFIG, raw_panel).book_by_decision_dict[decision_ts].index
    volume_panel = make_panel()
    volume_panel.dollar_volume_mat[row_int - 19:row_int + 1, column_int] = 1e6
    assert picked_str not in build_target_path(LEG3_CONFIG, volume_panel).book_by_decision_dict[decision_ts].index
    member_panel = make_panel()
    member_panel.member_mat[row_int, column_int] = False
    assert picked_str not in build_target_path(LEG3_CONFIG, member_panel).book_by_decision_dict[decision_ts].index


def test_traded_frame_keeps_held_names_and_the_benchmark_and_shares_its_provenance():
    date_index = sessions("2026-09-01", "2026-09-30")
    column_idx = pd.MultiIndex.from_product([["AAA", "BBB", "$SPXTR"], ["Open", "Close"]])
    frame_df = pd.DataFrame(1.0, index=date_index, columns=column_idx)
    frame_df.attrs["norgate_adjustment_by_symbol_dict"] = {"AAA": "CAPITALSPECIAL", "BBB": "CAPITALSPECIAL", "$SPXTR": "TOTALRETURN"}
    universe_df = pd.DataFrame(1, index=date_index, columns=["AAA", "BBB"])
    path_obj = TargetPath({date_index[0]: pd.Series({"AAA": .1})}, {}, {}, {})
    traded_df = traded_pricing_df(frame_df, universe_df, path_obj)
    assert traded_df.columns.get_level_values(0).unique().tolist() == ["AAA", "$SPXTR"]
    assert traded_df.attrs["norgate_adjustment_by_symbol_dict"]["$SPXTR"] == "TOTALRETURN"
    assert traded_df.loc[:, ("AAA", "Close")].attrs["norgate_adjustment_by_symbol_dict"] is traded_df.attrs["norgate_adjustment_by_symbol_dict"]
    assert frame_df.columns.get_level_values(0).unique().tolist() == ["AAA", "BBB", "$SPXTR"]


@pytest.mark.parametrize("leg_str", list(CONFIG_DICT))
def test_the_account_targets_the_average_of_the_two_latest_books(leg_str):
    path_obj = build_target_path(CONFIG_DICT[leg_str], make_panel())
    decision_list = sorted(path_obj.target_by_decision_dict)
    latest_dict = {0: pd.Series(dtype=float), 10: pd.Series(dtype=float)}
    for decision_ts in decision_list:
        latest_dict[path_obj.book_offset_by_decision_dict[decision_ts]] = path_obj.book_by_decision_dict[decision_ts]
        expected_ser = .5 * latest_dict[0].add(latest_dict[10], fill_value=0.0)
        pd.testing.assert_series_equal(path_obj.target_by_decision_dict[decision_ts], expected_ser.sort_index())


def test_book_size_scale_and_sector_cap_of_the_nasdaq_legs():
    panel = make_panel()
    path_obj = build_target_path(LEG1_CONFIG, panel)
    decision_list = [decision_ts for decision_ts in sorted(path_obj.book_by_decision_dict) if len(path_obj.book_by_decision_dict[decision_ts])]
    assert decision_list
    label_dict = dict(zip(panel.symbol_vec, panel.sector_label_vec))
    for decision_ts in decision_list:
        book_ser = path_obj.book_by_decision_dict[decision_ts]
        assert len(book_ser) <= 10
        assert book_ser.round(12).nunique() == 1 and book_ser.iloc[0] == pytest.approx((22.0 / 25.0) / 10.0)
        assert pd.Series([label_dict[symbol_str] for symbol_str in book_ser.index]).value_counts().max() <= 4


def test_a_shut_gate_empties_both_books_at_their_own_decisions():
    panel = make_panel()
    panel.gate_close_vec[-90:] = panel.gate_close_vec[-91] * np.linspace(.8, .6, 90)
    path_obj = build_target_path(LEG1_CONFIG, panel)
    last_list = sorted(path_obj.target_by_decision_dict)[-2:]
    assert all(path_obj.evidence_by_decision_dict[decision_ts]["gate_open_bool"] is False for decision_ts in last_list)
    assert path_obj.target_by_decision_dict[last_list[-1]].empty


def test_a_name_that_stops_trading_between_decisions_leaves_its_share_in_cash():
    panel = make_panel()
    path_obj = build_target_path(LEG1_CONFIG, panel)
    decision_list = [decision_ts for decision_ts in sorted(path_obj.book_by_decision_dict) if len(path_obj.book_by_decision_dict[decision_ts])]
    t10_ts = next(decision_ts for decision_ts in reversed(decision_list[:-1]) if path_obj.book_offset_by_decision_dict[decision_ts] == 10)
    t0_ts = min(decision_ts for decision_ts in path_obj.book_by_decision_dict if decision_ts > t10_ts)
    book_ser = path_obj.book_by_decision_dict[t10_ts]
    dead_str = book_ser.index[0]
    column_int = list(panel.symbol_vec).index(dead_str)
    t10_row_int = panel.date_index.get_loc(t10_ts)
    for mat in (panel.close_mat, panel.high_mat, panel.low_mat, panel.raw_close_mat):
        mat[t10_row_int + 3:, column_int] = np.nan
    dead_path_obj = build_target_path(LEG1_CONFIG, panel)
    pd.testing.assert_series_equal(dead_path_obj.book_by_decision_dict[t10_ts], book_ser)
    assert dead_str not in dead_path_obj.book_by_decision_dict[t0_ts].index
    assert dead_str not in dead_path_obj.target_by_decision_dict[t0_ts].index
    expected_float = .5 * (dead_path_obj.book_by_decision_dict[t0_ts].sum() + book_ser.drop(dead_str).sum())
    assert dead_path_obj.target_by_decision_dict[t0_ts].sum() == pytest.approx(expected_float)


def test_a_later_backtest_window_reads_the_same_path():
    panel = make_panel()
    full_path_obj = build_target_path(LEG2_CONFIG, panel)
    later_path_obj = build_target_path(replace(LEG2_CONFIG, backtest_start_date_str="2023-11-01"), panel)
    assert later_path_obj.target_by_decision_dict.keys() == full_path_obj.target_by_decision_dict.keys()
    for decision_ts, target_ser in full_path_obj.target_by_decision_dict.items():
        pd.testing.assert_series_equal(later_path_obj.target_by_decision_dict[decision_ts], target_ser)


def test_the_rank_buffer_keeps_held_names_within_the_limit_then_fills_by_rank():
    symbol_vec = np.array([f"S{column_int}" for column_int in range(8)], dtype=object)
    label_vec = np.array(["A", "A", "A", "B", "B", "B", "C", "C"], dtype=object)
    picked_vec = pick_with_rank_buffer(np.arange(8), {"S5", "S7"}, symbol_vec, label_vec, 3, 2, 6)
    assert picked_vec.tolist() == [5, 0, 1]
    picked_vec = pick_with_rank_buffer(np.arange(8), {"S1"}, symbol_vec, label_vec, 4, 2, 6)
    assert picked_vec.tolist() == [1, 0, 3, 4]
    picked_vec = pick_with_rank_buffer(np.arange(8), {"S0", "S1", "S2"}, symbol_vec, label_vec, 3, 2, 6)
    assert picked_vec.tolist() == [0, 1, 3]


def test_inverse_vol_weights_keep_the_slots_gross_the_cap_and_the_floor():
    weight_vec = inverse_vol_weight_vec(np.array([.2, .4, .3, .05, np.nan]), .985, 20)
    slot_float = .985 / 20
    assert weight_vec.sum() == pytest.approx(5 * slot_float)
    assert weight_vec.max() <= 1.5 * slot_float + 1e-12
    # A volatility at or under the floor, or missing, reads the median of the others.
    assert weight_vec[3] == pytest.approx(weight_vec[2]) and weight_vec[4] == pytest.approx(weight_vec[2])
    capped_vec = inverse_vol_weight_vec(np.array([.09, 1.0, 1.0, 1.0]), .985, 20)
    assert capped_vec[0] == pytest.approx(1.5 * slot_float)
    assert capped_vec.sum() == pytest.approx(4 * slot_float)


# ---------------------------------------------------------------- named books, scores against their formulas, the panel's wiring
def _rising_panel():
    """Every name above its average, gate open after 200 sessions, plenty of dollar volume."""
    panel = make_panel()
    trend_vec = np.linspace(100.0, 200.0, len(panel.date_index))
    panel.gate_close_vec[:] = trend_vec
    panel.close_mat[:] = trend_vec[:, None]
    panel.high_mat[:] = trend_vec[:, None] * 1.01
    panel.low_mat[:] = trend_vec[:, None] * .99
    panel.raw_close_mat[:] = trend_vec[:, None]
    return panel


def _scripted(monkeypatch, config_obj, score_fn):
    monkeypatch.setitem(legs_module.SCORE_FUNCTION_DICT, config_obj.score_kind_str, lambda _c, _p, row_vec: {int(row_int): score_fn(int(row_int)) for row_int in row_vec})


def _names(book_ser):
    return sorted(book_ser.index)


def test_book_rules_on_scripted_scores(monkeypatch):
    panel = _rising_panel()
    decision_df = build_decision_df(panel.date_index)
    open_df = decision_df.loc[decision_df["decision_row_int"] >= 210]
    t10_row_list = open_df.loc[open_df["book_offset_int"] == 10, "decision_row_int"].tolist()
    t0_row_list = open_df.loc[open_df["book_offset_int"] == 0, "decision_row_int"].tolist()
    first_t10, second_t10, third_t10 = t10_row_list[:3]
    t0_between = [row_int for row_int in t0_row_list if first_t10 < row_int < second_t10][0]
    base_vec = -np.arange(48.0)

    def score_fn(row_int):
        score_vec = base_vec.copy()
        if row_int == first_t10:
            score_vec[15] = 100.0  # S15 enters the T10 book only
        return score_vec
    _scripted(monkeypatch, LEG1_CONFIG, score_fn)
    # S00 leaves the index for one decision; S01 falls under its average for one decision.
    panel.member_mat[second_t10, 0] = False
    panel.close_mat[t0_between, 1] = 1.0
    path_obj = build_target_path(LEG1_CONFIG, panel)
    by_ts = lambda row_int: path_obj.book_by_decision_dict[panel.date_index[row_int]]
    top10 = [f"S{k:02d}" for k in range(10)]
    # The best score enters; the best names are taken, not the worst.
    assert _names(by_ts(first_t10)) == sorted(["S15"] + top10[:9])
    # This book's own earlier pick stays within rank 18; a non-member is out at that decision only.
    assert _names(by_ts(second_t10)) == sorted(["S15"] + top10[1:9] + ["S09"])
    # The other book never read S15, and a name under its average is out at that decision only.
    assert _names(by_ts(t0_between)) == sorted([name for name in top10 if name != "S01"] + ["S10"])
    # At the next decision every held name is still inside the limit, so S00 finds no free slot.
    assert _names(by_ts(third_t10)) == _names(by_ts(second_t10))


def test_ties_break_by_symbol(monkeypatch):
    panel = _rising_panel()
    tie_vec = np.random.default_rng(0).integers(0, 4, 48).astype(float)
    config_obj = replace(LEG1_CONFIG, sector_cap_int=48, hold_rank_int=10)
    monkeypatch.setitem(legs_module.SCORE_FUNCTION_DICT, config_obj.score_kind_str, lambda _c, _p, row_vec: {int(r): tie_vec.copy() for r in row_vec})
    path_obj = build_target_path(config_obj, panel)
    book_ser = path_obj.book_by_decision_dict[max(path_obj.book_by_decision_dict)]
    expected_list = sorted(range(48), key=lambda k: (-tie_vec[k], k))[:10]
    assert _names(book_ser) == sorted(f"S{k:02d}" for k in expected_list)


def test_the_gate_is_read_at_t(monkeypatch):
    panel = _rising_panel()
    decision_df = build_decision_df(panel.date_index)
    row_int = int(decision_df.loc[decision_df["decision_row_int"] >= 300, "decision_row_int"].iloc[3])
    panel.gate_close_vec[row_int] = 1.0
    panel.close_mat[row_int + 1 - 2, 0] = 1.0  # S00 under its average on the session before T only: it stays eligible at T when the gate reopens
    _scripted(monkeypatch, LEG1_CONFIG, lambda _row: -np.arange(48.0))
    path_obj = build_target_path(LEG1_CONFIG, panel)
    assert path_obj.book_by_decision_dict[panel.date_index[row_int]].empty
    assert not path_obj.evidence_by_decision_dict[panel.date_index[row_int]]["gate_open_bool"]


@pytest.mark.parametrize("vxn_float,scale_float", [(15.0, 1.0), (25.0, 22.0 / 25.0), (200.0, .25)])
def test_the_vxn_scale_is_clipped_and_unfilled_slots_stay_in_cash(monkeypatch, vxn_float, scale_float):
    panel = _rising_panel()
    panel.vxn_vec[:] = vxn_float
    _scripted(monkeypatch, LEG1_CONFIG, lambda _row: -np.arange(48.0))
    panel.member_mat[:, 3:] = False
    path_obj = build_target_path(LEG1_CONFIG, panel)
    book_ser = path_obj.book_by_decision_dict[max(path_obj.book_by_decision_dict)]
    assert len(book_ser) == 3
    assert book_ser.to_numpy() == pytest.approx(np.full(3, scale_float / 10.0))


def test_leg3_weights_are_inverse_vol_at_t():
    panel = make_panel()
    path_obj = build_target_path(LEG3_CONFIG, panel)
    checked_int = 0
    for decision_ts, book_ser in path_obj.book_by_decision_dict.items():
        if book_ser.empty:
            continue
        row_int = panel.date_index.get_loc(decision_ts)
        ret_mat = daily_return_mat(panel.close_mat)[row_int - 62:row_int + 1]
        column_list = [list(panel.symbol_vec).index(symbol_str) for symbol_str in book_ser.index]
        vol_vec = ret_mat[:, column_list].std(axis=0, ddof=1) * np.sqrt(252.0)
        assert book_ser.to_numpy() == pytest.approx(inverse_vol_weight_vec(vol_vec, .985, 20))
        assert not np.allclose(book_ser.to_numpy(), book_ser.iloc[0])
        checked_int += 1
    assert checked_int >= 4


def test_leg3_filters_on_scripted_scores(monkeypatch):
    panel = _rising_panel()
    panel.raw_close_mat[:, 0] = 9.0
    panel.dollar_volume_mat[:, 1] = 1e6
    _scripted(monkeypatch, LEG3_CONFIG, lambda _row: -np.arange(48.0))
    path_obj = build_target_path(LEG3_CONFIG, panel)
    book_ser = path_obj.book_by_decision_dict[max(path_obj.book_by_decision_dict)]
    assert len(book_ser) and not {"S00", "S01"} & set(book_ser.index)


def test_a_shut_gate_keeps_the_other_books_share_until_its_own_decision():
    panel = make_panel()
    panel.gate_close_vec[-90:] = panel.gate_close_vec[-91] * np.linspace(.8, .6, 90)
    path_obj = build_target_path(LEG1_CONFIG, panel)
    decision_list = sorted(path_obj.target_by_decision_dict)
    shut_list = [ts for ts in decision_list if not path_obj.evidence_by_decision_dict[ts]["gate_open_bool"] and panel.date_index.get_loc(ts) > 300]
    first_ts = shut_list[0]
    other_ts = decision_list[decision_list.index(first_ts) - 1]
    other_book_ser = path_obj.book_by_decision_dict[other_ts]
    assert len(other_book_ser)
    pd.testing.assert_series_equal(path_obj.target_by_decision_dict[first_ts], (.5 * other_book_ser).sort_index())
    assert path_obj.target_by_decision_dict[shut_list[1]].empty


def test_the_path_starts_at_the_first_execution_on_or_after_the_start():
    panel = make_panel()
    config_obj = replace(LEG1_CONFIG, path_start_date_str="2023-06-01", backtest_start_date_str="2023-06-01")
    path_obj = build_target_path(config_obj, panel)
    # 2023-05-31 decides for the 2023-06-01 open, so it is the first; nothing earlier enters the buffers.
    assert min(path_obj.target_by_decision_dict) == pd.Timestamp("2023-05-31")
    earlier_config_obj = replace(config_obj, backtest_start_date_str="2023-01-03")
    assert min(build_target_path(earlier_config_obj, panel).target_by_decision_dict) < pd.Timestamp("2023-01-03") + pd.Timedelta(days=31)


def test_evidence_counts_are_read_from_the_panel(monkeypatch):
    panel = _rising_panel()

    def score_fn(_row):
        score_vec = np.full(48, np.nan)
        score_vec[:2] = 1.0
        score_vec[40] = 1.0  # a non-member's score does not count
        return score_vec
    _scripted(monkeypatch, LEG1_CONFIG, score_fn)
    panel.member_mat[:, 30:] = False
    path_obj = build_target_path(LEG1_CONFIG, panel)
    evidence_dict = path_obj.evidence_by_decision_dict[max(path_obj.evidence_by_decision_dict)]
    assert evidence_dict["member_count_int"] == 30 and evidence_dict["finite_score_member_count_int"] == 2


def test_the_frozen_rules_of_the_closed_design():
    assert (LEG1_CONFIG.indexname_str, LEG1_CONFIG.gate_symbol_str, LEG1_CONFIG.vxn_symbol_str, LEG1_CONFIG.max_positions_int, LEG1_CONFIG.sector_cap_int, LEG1_CONFIG.hold_rank_int, LEG1_CONFIG.gross_float, LEG1_CONFIG.stock_sma_int) == ("Nasdaq 100", "SPY", "$VXN", 10, 4, 18, 1.0, 100)
    assert (LEG2_CONFIG.market_symbol_str, LEG2_CONFIG.gate_symbol_str, LEG2_CONFIG.hold_rank_int) == ("$NDX", "SPY", 18)
    assert (LEG3_CONFIG.indexname_str, LEG3_CONFIG.gate_symbol_str, LEG3_CONFIG.market_symbol_str, LEG3_CONFIG.vxn_symbol_str, LEG3_CONFIG.max_positions_int, LEG3_CONFIG.sector_cap_int, LEG3_CONFIG.hold_rank_int, LEG3_CONFIG.gross_float, LEG3_CONFIG.min_raw_close_float, LEG3_CONFIG.min_dollar_volume_float, LEG3_CONFIG.inverse_vol_bool) == ("Russell 1000", "$RUI", "$RUI", None, 20, 4, 35, .985, 10.0, 1e7, True)
    assert (LEG1_CONFIG.gate_sma_int, LEG1_CONFIG.target_vxn_float, LEG1_CONFIG.min_vxn_scale_float, LEG1_CONFIG.roc_decision_count_int, LEG1_CONFIG.natr_window_int, LEG1_CONFIG.residual_window_int, LEG1_CONFIG.residual_look_int, LEG1_CONFIG.residual_skip_int, LEG1_CONFIG.peer_window_int, LEG1_CONFIG.peer_skip_int, LEG1_CONFIG.dollar_volume_window_int, LEG1_CONFIG.inverse_vol_window_int) == (200, 22.0, .25, 12, 20, 252, 126, 10, 126, 5, 20, 63)


# ---------------------------------------------------------------- the three scores, recomputed from the docstring
def _decision_rows(panel, offset_int):
    decision_df = build_decision_df(panel.date_index)
    return decision_df.loc[decision_df["book_offset_int"] == offset_int, "decision_row_int"].to_numpy()


def test_leg1_score_matches_the_formula():
    panel = make_panel()
    row_vec = _decision_rows(panel, 0)
    score_dict = legs_module.natr_score_dict(LEG1_CONFIG, panel, row_vec)
    position_int = len(row_vec) - 1
    row_int, base_int = int(row_vec[position_int]), int(row_vec[position_int - 12])
    for column_int in (0, 17, 40):
        close_vec = panel.close_mat[:, column_int]
        tr_vec = [max(panel.high_mat[r, column_int] - panel.low_mat[r, column_int], abs(panel.high_mat[r, column_int] - close_vec[r - 1]), abs(panel.low_mat[r, column_int] - close_vec[r - 1])) for r in range(row_int - 19, row_int + 1)]
        expected_float = (close_vec[row_int] / close_vec[base_int] - 1.0) / (np.mean(tr_vec) / close_vec[row_int])
        assert score_dict[row_int][column_int] == pytest.approx(expected_float)


def _ols_residual(y_vec, factor_list):
    x_mat = np.column_stack([np.ones(len(y_vec)), *factor_list])
    beta_vec = np.linalg.lstsq(x_mat, y_vec, rcond=None)[0]
    return y_vec - x_mat[:, 1:] @ beta_vec[1:]


def test_leg2_score_matches_the_formula():
    panel = make_panel()
    row_int = int(_decision_rows(panel, 0)[-1])
    score_vec = legs_module.sector_residual_score_dict(LEG2_CONFIG, panel, np.array([row_int]))[row_int]
    ret_mat = daily_return_mat(panel.close_mat)
    window_slice = slice(row_int - 251, row_int + 1)
    for column_int in (0, 17, 40):
        sector_vec = panel.sector_ret_df[panel.sector_label_vec[column_int]].to_numpy()[window_slice]
        u_vec = _ols_residual(ret_mat[window_slice, column_int], [panel.market_ret_vec[window_slice], sector_vec])
        momentum_vec = u_vec[126:242]  # T-125 .. T-10
        assert score_vec[column_int] == pytest.approx(momentum_vec.mean() / momentum_vec.std(ddof=1))


def test_leg3_score_matches_the_formula():
    panel = make_panel()
    panel.member_mat[:, 3] = False  # a non-member is nobody's peer
    row_int = int(_decision_rows(panel, 0)[-1])
    score_vec = legs_module.industry_residual_score_dict(LEG3_CONFIG, panel, np.array([row_int]))[row_int]
    ret_mat = daily_return_mat(panel.close_mat)
    window_slice = slice(row_int - 5 - 125, row_int - 5 + 1)
    for column_int in (0, 17, 40):
        peer_vec = np.flatnonzero((panel.group_label_vec == panel.group_label_vec[column_int]) & panel.member_mat[row_int])
        peer_vec = peer_vec[peer_vec != column_int]
        assert len(peer_vec) >= 5
        peer_ret_vec = ret_mat[window_slice][:, peer_vec].mean(axis=1)
        u_vec = _ols_residual(ret_mat[window_slice, column_int], [panel.market_ret_vec[window_slice], peer_ret_vec])
        volatility_float = np.sqrt(np.sum((u_vec - u_vec.mean()) ** 2) / (126 - 3))
        assert score_vec[column_int] == pytest.approx(u_vec.sum() / (np.sqrt(126.0) * volatility_float))


def test_leg3_falls_back_to_the_sector_when_the_group_is_thin():
    panel = make_panel()
    panel.group_label_vec = np.array([f"G{k}" for k in range(48)], dtype=object)
    panel.group_label_vec[[17, 18, 19, 20, 21]] = "THIN"  # four peers: one short of the rule
    row_int = int(_decision_rows(panel, 0)[-1])
    score_vec = legs_module.industry_residual_score_dict(LEG3_CONFIG, panel, np.array([row_int]))[row_int]
    ret_mat = daily_return_mat(panel.close_mat)
    window_slice = slice(row_int - 130, row_int - 4)
    column_int = 17
    sector_vec = panel.sector_ret_df[panel.sector_label_vec[column_int]].to_numpy()[window_slice]
    u_vec = _ols_residual(ret_mat[window_slice, column_int], [panel.market_ret_vec[window_slice], sector_vec])
    volatility_float = np.sqrt(np.sum((u_vec - u_vec.mean()) ** 2) / (126 - 3))
    assert score_vec[column_int] == pytest.approx(u_vec.sum() / (np.sqrt(126.0) * volatility_float))


def test_zero_return_and_dead_residual_names_get_no_score():
    panel = make_panel()
    row_int = int(_decision_rows(panel, 0)[-1])
    panel.close_mat[row_int - 80:row_int - 20, 17] = panel.close_mat[row_int - 80, 17]
    assert np.isnan(legs_module.sector_residual_score_dict(LEG2_CONFIG, panel, np.array([row_int]))[row_int][17])
    assert np.isnan(legs_module.industry_residual_score_dict(LEG3_CONFIG, panel, np.array([row_int]))[row_int][17])


# ---------------------------------------------------------------- the panel wires each field
def test_build_leg_panel_wires_each_field():
    date_idx = pd.bdate_range("2024-01-02", periods=6)
    frame_dict = {}
    for symbol_str in ("BBB", "AAA"):
        for field_str, base_float in (("Close", 100.0), ("High", 101.0), ("Low", 99.0), ("Unadjusted Close", 200.0), ("Turnover", 5e7), ("Volume", 1.0), ("Open", 100.0)):
            frame_dict[(symbol_str, field_str)] = pd.Series(base_float, index=date_idx)
    pricing_df = pd.DataFrame(frame_dict)
    universe_df = pd.DataFrame(1, index=date_idx, columns=["BBB", "AAA"])
    gate_ser = pd.Series(np.arange(1.0, 7.0), index=date_idx)
    vxn_ser = pd.Series([20.0, 21.0, np.nan, np.nan, 25.0, 26.0], index=date_idx).dropna()
    helper_dict = {"SPY": gate_ser, "$NDX": gate_ser * 2.0, "$VXN": vxn_ser}
    for k, symbol_str in enumerate(legs_module.SECTOR_INDEX_BY_LABEL_DICT.values()):
        helper_dict[symbol_str] = pd.Series(100.0 + np.arange(6.0) * (k + 1), index=date_idx)
    helper_dict["$SPXR"] = helper_dict["$SPXR"].where(date_idx >= date_idx[3])
    panel = legs_module.build_leg_panel(LEG2_CONFIG, pricing_df, universe_df, helper_dict, {"AAA": "45000000", "BBB": "60000000"}, None)
    assert list(panel.symbol_vec) == ["AAA", "BBB"] and list(panel.sector_label_vec) == ["45000000", "60000000"]
    assert (panel.raw_close_mat == 200.0).all() and (panel.dollar_volume_mat == 5e7).all()
    assert panel.gate_close_vec.tolist() == gate_ser.tolist()
    assert panel.vxn_vec.tolist() == [20.0, 21.0, 21.0, 21.0, 25.0, 26.0]
    assert panel.market_ret_vec[2] == pytest.approx(gate_ser.iloc[2] / gate_ser.iloc[1] - 1.0)
    financial_vec = helper_dict["$SPXF"].pct_change(fill_method=None)
    estate_ser = panel.sector_ret_df["60000000"]
    assert estate_ser.iloc[2] == pytest.approx(financial_vec.iloc[2])
    assert estate_ser.iloc[3] == pytest.approx(financial_vec.iloc[3])  # the first $SPXR bar has no return yet
    assert estate_ser.iloc[5] == pytest.approx(helper_dict["$SPXR"].iloc[5] / helper_dict["$SPXR"].iloc[4] - 1.0)


def test_dead_residual_gets_no_score():
    panel = make_panel()
    row_int = int(_decision_rows(panel, 0)[-1])
    # A stock equal to the market plus almost nothing: its residual is below the floor.
    ret_vec = panel.market_ret_vec + 1e-5 * np.random.default_rng(1).standard_normal(len(panel.market_ret_vec))
    panel.close_mat[:, 17] = 100.0 * np.cumprod(1.0 + ret_vec)
    assert np.isnan(legs_module.sector_residual_score_dict(LEG2_CONFIG, panel, np.array([row_int]))[row_int][17])
    assert np.isnan(legs_module.industry_residual_score_dict(LEG3_CONFIG, panel, np.array([row_int]))[row_int][17])


@pytest.mark.parametrize("config_obj", [LEG1_CONFIG, LEG3_CONFIG])
def test_membership_and_turnover_after_t_cannot_change_decisions_through_t(config_obj):
    panel = make_panel()
    full_path_obj = build_target_path(config_obj, panel)
    # Every later row of every name is shocked, one decision at a time, so a read of row T+1 shows.
    for decision_ts in [ts for ts in sorted(full_path_obj.book_by_decision_dict) if len(full_path_obj.book_by_decision_dict[ts])][-4:-1]:
        row_int = panel.date_index.get_loc(decision_ts)
        shocked = make_panel()
        shocked.member_mat[row_int + 1:] = False
        shocked.dollar_volume_mat[row_int + 1:] = 1.0
        shocked_path_obj = build_target_path(config_obj, shocked)
        pd.testing.assert_series_equal(shocked_path_obj.book_by_decision_dict[decision_ts], full_path_obj.book_by_decision_dict[decision_ts])
