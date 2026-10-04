"""alpha.scout.universes: membership alignment, causality of the liquidity features, and the costed DV2 replica."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.scout.panel import FIELD_TUPLE, Panel
from alpha.scout.specs import dv2
from alpha.scout.universes import (
    SupersetPanel,
    adv63_mat,
    adv_tercile_mat,
    costed_book,
    fill_slippage_mat,
    half_spread_mat,
    membership_matrix,
    pooled_half_spread_mat,
    universe_panel,
)

DATE_INDEX = pd.bdate_range("2020-01-01", periods=12)


# ---------------------------------------------------------------- membership
def test_membership_matrix_is_the_exact_flag_without_forward_fill():
    flag_a = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], index=DATE_INDEX)  # a gap, then a re-entry
    flag_b = pd.Series([1, 1], index=pd.DatetimeIndex(["2019-12-31", DATE_INDEX[3]]))  # a date outside the axis is ignored
    member_mat = membership_matrix({"A": flag_a, "B": flag_b, "Z": flag_a}, DATE_INDEX, ["A", "B", "C"])
    assert member_mat.dtype == np.int8
    assert member_mat[:, 0].tolist() == flag_a.tolist()
    assert member_mat[:, 1].tolist() == [0, 0, 0, 1] + [0] * 8  # only the flagged session, never filled forward
    assert member_mat[:, 2].sum() == 0  # no series -> never a member; "Z" is not on the axis


def test_membership_matrix_has_no_future_information():
    """Row T depends only on the flag dated T: truncating the flags after T leaves rows <= T unchanged."""
    rng_obj = np.random.default_rng(0)
    flag_dict = {s: pd.Series(rng_obj.integers(0, 2, len(DATE_INDEX)), index=DATE_INDEX) for s in "ABCD"}
    full_mat = membership_matrix(flag_dict, DATE_INDEX, list("ABCD"))
    for t_int in range(len(DATE_INDEX)):
        truncated_dict = {s: f.iloc[: t_int + 1] for s, f in flag_dict.items()}
        assert np.array_equal(membership_matrix(truncated_dict, DATE_INDEX, list("ABCD"))[: t_int + 1], full_mat[: t_int + 1])


def _superset(row_count_int: int = 30) -> SupersetPanel:
    date_index = pd.bdate_range("2022-11-01", periods=row_count_int)  # crosses the vault seal (2023-01-01)
    rng_obj = np.random.default_rng(1)
    field_dict = {f: rng_obj.random((row_count_int, 4)).astype(np.float32) + 1.0 for f in FIELD_TUPLE}
    member_mat = np.zeros((row_count_int, 4), dtype=np.int8)
    member_mat[5:, 0] = 1
    member_mat[10:15, 2] = 1
    return SupersetPanel(field_dict=field_dict, member_dict={"IDX": member_mat}, date_index=date_index,
                         symbol_list=["A", "B", "C", "D"], snapshot_id_str="test")


def test_universe_panel_keeps_ever_members_seals_and_masks_before_start():
    superset = _superset()
    panel = universe_panel(superset, "IDX", member_from_str=str(superset.date_index[12].date()))
    assert panel.symbol_list == ["A", "C"]
    assert panel.date_index.max() < pd.Timestamp("2023-01-01")  # sealed
    member_df = panel.member_df
    assert member_df.loc[: superset.date_index[11], :].to_numpy().sum() == 0  # nothing before the start
    assert member_df.loc[superset.date_index[12]:superset.date_index[14], "C"].tolist() == [1, 1, 1]
    assert np.array_equal(panel.field("Close")["C"].to_numpy(), superset.field_dict["Close"][: len(panel.date_index), 2])


# ---------------------------------------------------------------- causal liquidity features
def _ohlc(row_count_int: int = 120, column_count_int: int = 3, seed_int: int = 2):
    rng_obj = np.random.default_rng(seed_int)
    close_mat = 50.0 * np.exp(np.cumsum(rng_obj.normal(0, 0.02, (row_count_int, column_count_int)), axis=0))
    high_mat = close_mat * (1.0 + rng_obj.uniform(0.0, 0.02, close_mat.shape))
    low_mat = close_mat * (1.0 - rng_obj.uniform(0.0, 0.02, close_mat.shape))
    turnover_mat = rng_obj.uniform(1e6, 5e6, close_mat.shape)
    return high_mat, low_mat, close_mat, turnover_mat


def test_adv63_is_the_trailing_mean_and_an_invalid_day_voids_the_window():
    _, _, _, turnover_mat = _ohlc()
    turnover_mat[70, 1] = 0.0
    adv_mat = adv63_mat(turnover_mat)
    assert np.isnan(adv_mat[61, 0]) and adv_mat[62, 0] == pytest.approx(turnover_mat[:63, 0].mean())
    assert np.isnan(adv_mat[70:133, 1]).all() and np.isfinite(adv_mat[69, 1])


@pytest.mark.parametrize("feature_str", ["adv", "spread"])
def test_liquidity_features_are_prefix_invariant(feature_str):
    """The value at T is the same when every bar after T is removed or changed (no look-ahead)."""
    high_mat, low_mat, close_mat, turnover_mat = _ohlc()

    def compute(h, lo, c, v):
        return adv63_mat(v) if feature_str == "adv" else half_spread_mat(h, lo, c)

    full_mat = compute(high_mat, low_mat, close_mat, turnover_mat)
    for t_int in (40, 63, 64, 90, 119):
        prefix_mat = compute(high_mat[: t_int + 1], low_mat[: t_int + 1], close_mat[: t_int + 1], turnover_mat[: t_int + 1])
        np.testing.assert_array_equal(prefix_mat, full_mat[: t_int + 1])
        shocked = [m.copy() for m in (high_mat, low_mat, close_mat, turnover_mat)]
        for m in shocked:
            m[t_int + 1:] *= 3.0
        np.testing.assert_array_equal(compute(*shocked)[: t_int + 1], full_mat[: t_int + 1])


def test_half_spread_recovers_a_planted_spread_and_fills_lag_one_day():
    """Mid-price random walk; closes at bid or ask with probability 1/2; High/Low = the day's extreme mid +- half-spread.
    Abdi-Ranaldo should recover the half-spread on average."""
    rng_obj = np.random.default_rng(3)
    row_count_int, half_float = 4000, 0.004
    mid_vec = 100.0 * np.exp(np.cumsum(rng_obj.normal(0, 0.003, row_count_int)))
    close_vec = mid_vec * (1.0 + half_float * rng_obj.choice([-1.0, 1.0], row_count_int))
    high_vec = mid_vec * np.exp(np.abs(rng_obj.normal(0, 0.004, row_count_int))) * (1.0 + half_float)
    low_vec = mid_vec * np.exp(-np.abs(rng_obj.normal(0, 0.004, row_count_int))) * (1.0 - half_float)
    estimate_vec = half_spread_mat(high_vec[:, None], low_vec[:, None], close_vec[:, None])[:, 0]
    assert np.nanmedian(estimate_vec) == pytest.approx(half_float, rel=0.5)
    slip_mat = fill_slippage_mat(estimate_vec[:, None], floor_float=0.00025)
    assert slip_mat[0, 0] == 0.00025  # nothing known before the first row: the floor
    finite_vec = np.flatnonzero(np.isfinite(estimate_vec[:-1]))
    assert np.allclose(slip_mat[finite_vec + 1, 0], np.maximum(estimate_vec[finite_vec], 0.00025))  # row t uses row t-1


def _planted_spread_panel(row_count_int: int = 300, column_count_int: int = 200, seed_int: int = 5):
    """Liquid stocks (high ADV) quote a 2 bp half-spread, illiquid ones 40 bp; mid random walk with 2% daily volatility."""
    rng_obj = np.random.default_rng(seed_int)
    half_vec = np.where(np.arange(column_count_int) < column_count_int // 2, 0.0040, 0.0002)
    mid_mat = 50.0 * np.exp(np.cumsum(rng_obj.normal(0, 0.02, (row_count_int, column_count_int)), axis=0))
    close_mat = mid_mat * (1.0 + half_vec * rng_obj.choice([-1.0, 1.0], mid_mat.shape))
    high_mat = mid_mat * np.exp(np.abs(rng_obj.normal(0, 0.01, mid_mat.shape))) * (1.0 + half_vec)
    low_mat = mid_mat * np.exp(-np.abs(rng_obj.normal(0, 0.01, mid_mat.shape))) * (1.0 - half_vec)
    adv_mat = np.tile(np.where(half_vec > 0.001, 1e6, 1e9), (row_count_int, 1)) * rng_obj.uniform(0.9, 1.1, mid_mat.shape)
    return high_mat, low_mat, close_mat, adv_mat, half_vec


def test_pooled_half_spread_separates_liquid_from_illiquid_and_is_causal():
    high_mat, low_mat, close_mat, adv_mat, half_vec = _planted_spread_panel()
    eligible_mat = np.ones(close_mat.shape, dtype=bool)
    out_mat, _ = pooled_half_spread_mat(high_mat, low_mat, close_mat, adv_mat, eligible_mat, bucket_count_int=2, min_count_int=500)
    late_mat = out_mat[100:]
    assert np.nanmedian(late_mat[:, half_vec > 0.001]) == pytest.approx(0.0040, rel=0.35)
    assert np.nanmedian(late_mat[:, half_vec < 0.001]) < 0.0010  # the per-stock estimator cannot get this close
    for t_int in (80, 150, 299):
        prefix_mat, _ = pooled_half_spread_mat(high_mat[: t_int + 1], low_mat[: t_int + 1], close_mat[: t_int + 1], adv_mat[: t_int + 1],
                                               eligible_mat[: t_int + 1], bucket_count_int=2, min_count_int=500)
        np.testing.assert_array_equal(prefix_mat, out_mat[: t_int + 1])


def test_adv_terciles_rank_members_only_per_date():
    adv_mat = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6.0, 5.0, 4.0, 3.0, 2.0, np.nan]])
    member_mat = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1]], dtype=np.int8)
    tercile_mat = adv_tercile_mat(adv_mat, member_mat)
    assert tercile_mat[0].tolist() == [1, 1, 2, 2, 3, 3]
    assert tercile_mat[1].tolist() == [3, 3, 2, 0, 1, 0]  # a non-member and a NaN ADV are not ranked


# ---------------------------------------------------------------- the costed replica
def _synthetic_panel(row_count_int: int = 700, column_count_int: int = 40, seed_int: int = 4) -> Panel:
    rng_obj = np.random.default_rng(seed_int)
    date_index = pd.bdate_range("2001-01-01", periods=row_count_int)
    close_mat = 30.0 * np.exp(np.cumsum(rng_obj.normal(0.0006, 0.02, (row_count_int, column_count_int)), axis=0))
    open_mat = close_mat * np.exp(rng_obj.normal(0, 0.01, close_mat.shape))
    high_mat = np.maximum(open_mat, close_mat) * (1.0 + rng_obj.uniform(0, 0.02, close_mat.shape))
    low_mat = np.minimum(open_mat, close_mat) * (1.0 - rng_obj.uniform(0, 0.02, close_mat.shape))
    close_mat[650:, 3] = np.nan  # a delisting
    columns = [f"S{i:02d}" for i in range(column_count_int)]
    frame = lambda m: pd.DataFrame(m.astype(np.float32), index=date_index, columns=columns)
    field_dict = {"Open": frame(open_mat), "High": frame(high_mat), "Low": frame(low_mat), "Close": frame(close_mat),
                  "Volume": frame(np.ones_like(close_mat)), "Turnover": frame(np.full_like(close_mat, 1e6)),
                  "Unadjusted Close": frame(close_mat * 2.0), "Dividend": frame(np.zeros_like(close_mat))}
    member_mat = (rng_obj.random(close_mat.shape) < 0.9).astype(np.int8)
    member_mat[: 260] = 0  # the backtest start: no member before row 260
    return Panel("synthetic", field_dict, pd.DataFrame(member_mat, index=date_index, columns=columns), "synthetic", True)


def test_costed_book_without_costs_is_the_gross_replica():
    panel = _synthetic_panel()
    gross_vec = dv2.fast_daily_list_panel(panel, [{}], base_config=dv2.LIVE_CONFIG)[0][0]
    result = costed_book(panel, dv2.LIVE_CONFIG, 0.0, 0.0, 0.0, start_date_str=str(panel.date_index[260].date()))
    assert (result.fill_df["kind_int"] == 1).sum() > 50
    np.testing.assert_allclose(result.daily_ser.to_numpy(), gross_vec, rtol=0, atol=1e-12)


def test_costed_book_costs_and_share_units():
    panel = _synthetic_panel()
    start_str = str(panel.date_index[260].date())
    gross = costed_book(panel, dv2.LIVE_CONFIG, 0.0, 0.0, 0.0, start_date_str=start_str)
    slipped = costed_book(panel, dv2.LIVE_CONFIG, 0.001, 0.0, 0.0, start_date_str=start_str)
    adjusted = costed_book(panel, dv2.LIVE_CONFIG, 0.0, 0.005, 1.0, share_unit_str="adjusted", start_date_str=start_str)
    nominal = costed_book(panel, dv2.LIVE_CONFIG, 0.0, 0.005, 1.0, share_unit_str="nominal", start_date_str=start_str)
    growth = lambda r: float((1.0 + r.daily_ser).prod())
    assert growth(slipped) < growth(gross) and growth(adjusted) < growth(gross)
    # Unadjusted Close = 2 x Close here (a later 2:1 split): nominal fees count half the adjusted shares, so they cost less.
    assert growth(adjusted) < growth(nominal) < growth(gross)
    # The same decisions in every cost case (costs change size, never the signal).
    assert gross.fill_df[["date", "asset", "kind_int"]].equals(slipped.fill_df[["date", "asset", "kind_int"]])
    # A per-fill slippage matrix equal to the scalar gives the same book.
    matrix = costed_book(panel, dv2.LIVE_CONFIG, np.full(panel.member_df.shape, 0.001), 0.0, 0.0, start_date_str=start_str)
    np.testing.assert_allclose(matrix.daily_ser.to_numpy(), slipped.daily_ser.to_numpy(), atol=1e-15)
    assert (gross.fill_df["date"] >= panel.date_index[260]).all()


def test_tick_half_spread_is_half_a_cent_over_the_nominal_price():
    from alpha.scout.universes import tick_half_spread_mat

    out_mat = tick_half_spread_mat(np.array([[2.0, 50.0, np.nan, 0.0]]))
    assert out_mat[0, 0] == pytest.approx(0.0025) and out_mat[0, 1] == pytest.approx(0.0001)
    assert np.isnan(out_mat[0, 2]) and np.isnan(out_mat[0, 3])


def test_costed_book_fee_cap_and_ruin():
    panel = _synthetic_panel()
    start_str = str(panel.date_index[260].date())
    growth = lambda r: float((1.0 + r.daily_ser).prod())
    uncapped = costed_book(panel, dv2.LIVE_CONFIG, 0.0, 1.0, 1.0, start_date_str=start_str)
    capped = costed_book(panel, dv2.LIVE_CONFIG, 0.0, 1.0, 1.0, start_date_str=start_str, max_fee_fraction_float=0.01)
    assert growth(uncapped) < growth(capped) < growth(costed_book(panel, dv2.LIVE_CONFIG, 0.0, 0.0, 0.0, start_date_str=start_str))
    assert capped.ruin_date is None
    # A $5,000 minimum fee on a $100,000 book ruins it: floored at -100% that day, flat after.
    ruined = costed_book(panel, dv2.LIVE_CONFIG, 0.0, 0.005, 5_000.0, start_date_str=start_str)
    assert ruined.ruin_date is not None
    assert ruined.daily_ser.loc[ruined.ruin_date] >= -1.0
    assert (ruined.daily_ser.loc[ruined.ruin_date:].iloc[1:] == 0.0).all()
    assert (ruined.total_value_ser.loc[ruined.ruin_date:] >= 0.0).all()
