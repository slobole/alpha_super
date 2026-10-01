"""Scout P4 edge toolkit: panel and vault, S1 causality checks, S2 novelty, S3 edge study (synthetic data)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha.scout import features as F
from alpha.scout import panel as panel_module
from alpha.scout.features import Feature, trailing_return
from alpha.scout.ledger import Ledger
from alpha.scout.null import permuted_panel
from alpha.scout.panel import FIELD_TUPLE, Panel
from alpha.scout.stations.s1_causality import (
    membership_integrity,
    non_member_share,
    run_s1,
)
from alpha.scout.stations.s2_indicator import masters_battery, run_s2, sup_wald_break
from alpha.scout.stations.s3_edge import add_replication, forward_return_df, run_s3

DATE_INDEX = pd.bdate_range("2014-01-01", "2022-12-30")
SYMBOL_LIST = [f"S{i:02d}" for i in range(40)]


def _synthetic_panel(seed_int: int = 0, edge_float: float = 0.0, event_rate_float: float = 0.03, edge_from_symbol_int: int = 0):
    """Random-walk stocks; if edge_float is set, a stock whose 3-day return is in the bottom event_rate_float of the
    day gains edge_float per day over the next 5 sessions (a planted short-term reversal; negative: continuation).
    Only symbols from edge_from_symbol_int on carry it. Dollar turnover rises with the symbol number."""
    rng_obj = np.random.default_rng(seed_int)
    return_mat = rng_obj.normal(0.0003, 0.015, (len(DATE_INDEX), len(SYMBOL_LIST)))
    close_mat = 50 * np.cumprod(1 + return_mat, axis=0)
    if edge_float:
        trailing_mat = pd.DataFrame(close_mat).pct_change(3).to_numpy()
        cut_vec = np.nanquantile(trailing_mat, event_rate_float, axis=1)
        trigger_mat = trailing_mat < cut_vec[:, None]
        trigger_mat[:, :edge_from_symbol_int] = False
        for offset_int in range(1, 6):
            return_mat[offset_int:] += edge_float * trigger_mat[:-offset_int]
        close_mat = 50 * np.cumprod(1 + return_mat, axis=0)
    close_df = pd.DataFrame(close_mat, index=DATE_INDEX, columns=SYMBOL_LIST)
    open_df = close_df.shift(1).fillna(close_df.iloc[0])
    field_dict = {
        "Open": open_df, "High": np.maximum(open_df, close_df) * 1.005, "Low": np.minimum(open_df, close_df) * 0.995,
        "Close": close_df, "Volume": close_df * 0 + 1e6, "Turnover": close_df * 1e6 * np.arange(1, len(SYMBOL_LIST) + 1),
        "Unadjusted Close": close_df, "Dividend": close_df * 0.0,
    }
    member_df = pd.DataFrame(1, index=DATE_INDEX, columns=SYMBOL_LIST, dtype=np.int8)
    member_df.iloc[:, :5] = 0  # five symbols are never members
    return Panel("synthetic", field_dict, member_df, "synthetic", sealed_bool=True)


# ---------------------------------------------------------------- panel and vault
def test_vault_seal_opens_only_for_a_vault_opening_row_in_the_verified_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_module, "PANEL_ROOT_PATH", tmp_path)
    date_index = pd.bdate_range("2021-06-01", "2023-06-30")
    folder_path = tmp_path / "Test" / "abc"
    folder_path.mkdir(parents=True)
    for field_str in FIELD_TUPLE:
        pd.DataFrame({"A": np.arange(len(date_index), dtype=float)}, index=date_index).to_parquet(folder_path / f"{field_str.replace(' ', '_')}.parquet")
    pd.DataFrame({"A": np.ones(len(date_index), dtype=np.int8)}, index=date_index).to_parquet(folder_path / "member.parquet")
    (folder_path / "meta.json").write_text(json.dumps({"snapshot_id_str": "abc"}), encoding="utf-8")
    (tmp_path / "Test" / "latest.json").write_text(json.dumps({"snapshot_id_str": "abc"}), encoding="utf-8")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    note_row_id_int = ledger.append("note", {"family_id_str": "mine", "text_str": "not a vault opening"})["row_id_int"]
    opening_row_id_int = ledger.append("vault_opening", {"family_id_str": "mine"})["row_id_int"]

    sealed = panel_module.load_panel("Test")
    assert sealed.sealed_bool and sealed.date_index[-1] < pd.Timestamp(panel_module.VAULT_SEAL_STR)
    assert all(frame.index[-1] < pd.Timestamp(panel_module.VAULT_SEAL_STR) for frame in sealed.field_dict.values())
    for row_id_int, family_id_str in ((opening_row_id_int, "other"), (note_row_id_int, "mine"), (99, "mine")):
        assert panel_module.load_panel("Test", vault_opening_row_id_int=row_id_int, family_id_str=family_id_str, ledger=ledger).sealed_bool
    opened = panel_module.load_panel("Test", "abc", opening_row_id_int, "mine", ledger)
    assert not opened.sealed_bool and opened.date_index[-1] == date_index[-1]
    with pytest.raises(FileNotFoundError):
        panel_module.load_panel("Test", "other_snapshot")


# ---------------------------------------------------------------- S1
def test_s1_passes_clean_features_and_catches_seeded_leaks():
    panel = _synthetic_panel()
    leak = Feature("next_close", lambda p: p.field("Close").shift(-1) / p.field("Close"), 1, "scale_invariant", "leak")
    centred = Feature("centred", lambda p: p.field("Close").rolling(11, center=True).mean() / p.field("Close"), 11, "scale_invariant", "leak")
    full_sample = Feature("zscore_full", lambda p: (p.field("Close") - p.field("Close").mean()) / p.field("Close").std(), 1, "scale_invariant", "leak")
    mislabelled = Feature("above_60", lambda p: (p.field("Close") > 60).astype(float), 1, "scale_invariant", "level")
    declared_level = Feature("above_60_level", lambda p: (p.field("Close") > 60).astype(float), 1, "level", "level")
    result_dict = {r.feature_name_str: r for r in run_s1([trailing_return(3), leak, centred, full_sample, mislabelled, declared_level], panel, sample_size_int=20)}
    assert not result_dict["ret_3d"].hard_fail_bool
    for leak_name_str in ("next_close", "centred", "zscore_full"):
        assert not result_dict[leak_name_str].prefix_pass_bool and result_dict[leak_name_str].hard_fail_bool
    assert result_dict["above_60"].prefix_pass_bool and result_dict["above_60"].split_pass_bool is False
    assert result_dict["above_60"].hard_fail_bool
    assert result_dict["above_60_level"].split_pass_bool is None and not result_dict["above_60_level"].hard_fail_bool
    with pytest.raises(ValueError):
        Feature("typo", lambda p: p.field("Close"), 1, "scale-invariant", "a basis typo must not become 'level'")


def test_s1_passes_the_feature_library():
    # Catches a library feature that starts reading ahead, and a split simulation that forgets High/Low (NATR).
    panel = _synthetic_panel()
    feature_list = F.reference_feature_list() + [F.qpi(3, 2), F.dv2(126)]
    failed_list = [(r.feature_name_str, r.prefix_detail_str, r.split_detail_str) for r in run_s1(feature_list, panel, sample_size_int=25) if r.hard_fail_bool]
    assert failed_list == []


def test_membership_integrity_catches_a_tail_trimmed_mask():
    panel = _synthetic_panel()
    close_df, member_df = panel.field("Close").copy(), panel.member_df.copy()
    for column_int in range(5, 25):  # twenty members stop trading (acquired) while in the index
        close_df.iloc[200 + 50 * column_int:, column_int] = np.nan
    clean = replace(panel, field_dict={**panel.field_dict, "Close": close_df})
    assert membership_integrity(clean)["pass_bool"]
    for column_int in range(5, 25):
        member_df.iloc[200 + 50 * column_int - 5 : 200 + 50 * column_int, column_int] = 0  # hindsight: last 5 member days dropped
    trimmed_dict = membership_integrity(replace(clean, member_df=member_df))
    assert not trimmed_dict["pass_bool"] and trimmed_dict["trimmed_int"] == 20
    event_df = pd.DataFrame(False, index=DATE_INDEX, columns=SYMBOL_LIST)
    event_df.iloc[10, [0, 10]] = True  # S00 is never a member
    assert non_member_share(event_df, panel) == 0.5


# ---------------------------------------------------------------- S3
def test_forward_labels_enter_next_open_and_exit_at_the_horizon_close():
    panel = _synthetic_panel()
    forward_df = forward_return_df(panel, 5)
    t_int = 100
    expected_float = panel.field("Close").iloc[t_int + 5, 3] / panel.field("Open").iloc[t_int + 1, 3] - 1
    assert forward_df.iloc[t_int, 3] == pytest.approx(expected_float)
    assert forward_df.iloc[-5:].isna().all().all()  # no bars after the panel's end: the seal purges these labels


def _study(panel):
    return_3d_df = trailing_return(3).compute_fn(panel)
    regime_df = pd.DataFrame(True, index=DATE_INDEX, columns=SYMBOL_LIST)
    event_df = return_3d_df.lt(return_3d_df.quantile(0.03, axis=1), axis=0)
    return run_s3("synthetic", panel, regime_df, event_df, 5, indicator_df=return_3d_df)


def test_s3_finds_a_planted_edge_and_not_noise():
    edge_report = _study(_synthetic_panel(seed_int=1, edge_float=0.002))
    assert edge_report.headline_dict["nw_t_float"] > 4 and edge_report.verdict_str != "REJECTED (hard fail)"
    assert edge_report.headline_dict["non_member_event_share_float"] == pytest.approx(5 / 40, abs=0.05)  # reported, then dropped
    decile_list = edge_report.table_dict["deciles"]
    assert decile_list[0]["mean_float"] > decile_list[-1]["mean_float"]  # the bottom decile carries the edge
    noise_report_list = [_study(_synthetic_panel(seed_int=seed_int)) for seed_int in range(10, 16)]
    assert all(abs(report.headline_dict["nw_t_float"]) < 3 for report in noise_report_list)
    assert all(report.headline_dict["nw_lag_int"] == 4 for report in noise_report_list)
    assert edge_report.headline_dict["placebo_p_float"] <= 0.02
    assert np.mean([report.headline_dict["placebo_p_float"] <= 0.05 for report in noise_report_list]) <= 1 / 3


def test_replication_is_a_soft_check_on_the_sibling_sign():
    report = _study(_synthetic_panel(seed_int=1, edge_float=0.002))
    sibling = _study(_synthetic_panel(seed_int=5, edge_float=-0.002))
    sibling.name_str = "sibling"
    add_replication(report, [sibling])
    assert report.table_dict["replication"][0]["universe_str"] == "sibling"
    assert ("same sign in >= 1 sibling universe", False, "soft") in report.check_list
    assert report.verdict_str == "WATCHLIST (soft fail)"


def test_s3_rejects_only_on_evidence_against():
    # A planted continuation (the wrong sign for a reversal study) is evidence against: hard fail.
    against_report = _study(_synthetic_panel(seed_int=3, edge_float=-0.002))
    assert against_report.headline_dict["nw_t_float"] < -2 and against_report.verdict_str == "REJECTED (hard fail)"
    # On noise, a wrong sign under only one of the two estimators is not evidence against.
    for seed_int in range(20, 28):
        report = _study(_synthetic_panel(seed_int=seed_int))
        t_float, per_event_t_float = report.headline_dict["nw_t_float"], report.headline_dict["per_event_nw_t_float"]
        assert (report.verdict_str == "REJECTED (hard fail)") == ((t_float < 0 and per_event_t_float < 0) or t_float <= -2)


def test_s3_liquidity_terciles_follow_turnover():
    # The edge is planted only in the most traded third of the symbols.
    panel = _synthetic_panel(seed_int=4, edge_float=0.003, edge_from_symbol_int=27)
    return_3d_df = trailing_return(3).compute_fn(panel)
    regime_df = pd.DataFrame(True, index=DATE_INDEX, columns=SYMBOL_LIST)
    event_df = return_3d_df.lt(return_3d_df.quantile(0.03, axis=1), axis=0)
    report = run_s3("liquidity", panel, regime_df, event_df, 5, liquidity_rank_df=F.turnover_rank(63).compute_fn(panel))
    tercile_dict = {row["tercile_str"]: row for row in report.table_dict["liquidity_terciles"]}
    assert tercile_dict["top"]["t_float"] > 3 and abs(tercile_dict["bottom"]["t_float"]) < 3


def test_s3_lag_decay_and_eras_are_reported():
    report = _study(_synthetic_panel(seed_int=2, edge_float=0.002))
    lag_mean_vec = [row["mean_float"] for row in report.table_dict["lag_decay"]]
    assert lag_mean_vec[0] > lag_mean_vec[-1]  # the planted edge is gone after a 5-session delay
    assert [row["era_str"] for row in report.table_dict["eras"]][-1] == "2016-2022"


# ---------------------------------------------------------------- the per-asset null
def test_permuted_panel_keeps_structure_and_moves_whole_bars():
    panel = _synthetic_panel()
    close_df = panel.field("Close").copy()
    close_df.iloc[:300, 7] = np.nan  # a late listing
    member_df = panel.member_df.copy()
    member_df.iloc[:1000, 10] = 0  # joins the index later
    panel = replace(panel, field_dict={**panel.field_dict, "Close": close_df}, member_df=member_df)
    permuted = permuted_panel(panel, np.random.default_rng(0))
    new_close_df = permuted.field("Close")
    assert (new_close_df.isna() == close_df.isna()).all().all() and permuted.member_df.equals(panel.member_df)
    first_row_vec = close_df.notna().to_numpy().argmax(axis=0)
    np.testing.assert_allclose(new_close_df.to_numpy()[first_row_vec, range(len(SYMBOL_LIST))], close_df.to_numpy()[first_row_vec, range(len(SYMBOL_LIST))])
    for column_int in (3, 10):
        body_vec = np.log(close_df.iloc[:, column_int] / panel.field("Open").iloc[:, column_int]).dropna().to_numpy()[1:]
        new_body_vec = np.log(new_close_df.iloc[:, column_int] / permuted.field("Open").iloc[:, column_int]).dropna().to_numpy()[1:]
        np.testing.assert_allclose(np.sort(new_body_vec), np.sort(body_vec), atol=1e-12)  # whole bars moved, none invented
        assert not np.allclose(new_body_vec, body_vec)
    member_mask = (panel.member_df.iloc[1:, 10] == 1).to_numpy()
    real_member_body_vec = np.log(close_df.iloc[1:, 10] / panel.field("Open").iloc[1:, 10]).to_numpy()[member_mask]
    new_member_body_vec = np.log(new_close_df.iloc[1:, 10] / permuted.field("Open").iloc[1:, 10]).to_numpy()[member_mask]
    np.testing.assert_allclose(np.sort(new_member_body_vec), np.sort(real_member_body_vec), atol=1e-12)  # member bars stay member bars
    listed_df = new_close_df.notna()
    assert (permuted.field("High") >= np.maximum(permuted.field("Open"), new_close_df))[listed_df].fillna(True).all().all()
    assert (permuted.field("Low") <= np.minimum(permuted.field("Open"), new_close_df))[listed_df].fillna(True).all().all()
    assert permuted.field("Volume").equals(panel.field("Volume")) and permuted.field("Turnover").equals(panel.field("Turnover"))


# ---------------------------------------------------------------- S2
def test_masters_battery_flags_breaks_tails_and_finds_information():
    rng_obj = np.random.default_rng(0)
    assert sup_wald_break(np.concatenate([rng_obj.normal(0, 1, 150), rng_obj.normal(1.0, 1, 150)]))["warn_bool"]
    assert not sup_wald_break(rng_obj.normal(0, 1, 300))["warn_bool"]
    panel = _synthetic_panel(seed_int=1, edge_float=0.002)
    eligible_df = pd.DataFrame(True, index=DATE_INDEX, columns=SYMBOL_LIST)
    forward_df = forward_return_df(panel, 5)
    excess_df = forward_df.sub(forward_df.mean(axis=1), axis=0)
    informative = masters_battery(trailing_return(3).compute_fn(panel), eligible_df, excess_df)
    assert informative["mutual_information"]["bits_float"] > informative["mutual_information"]["shuffled_max_float"]
    heavy_df = trailing_return(3).compute_fn(panel) ** 3  # cubing piles most values into the middle bins
    assert any("entropy" in warn_str or "tails" in warn_str for warn_str in masters_battery(heavy_df, eligible_df)["warn_list"])
def test_s2_flags_an_indicator_in_disguise():
    panel = _synthetic_panel()
    disguised_df = trailing_return(3).compute_fn(panel) * 100 + 7  # a monotone transform of the 3-day return
    eligible_df = pd.DataFrame(True, index=DATE_INDEX, columns=SYMBOL_LIST)
    result = run_s2(disguised_df, eligible_df, lambda value_vec: value_vec < 0, panel, [trailing_return(3), trailing_return(20)], sample_size_int=20_000)
    assert any("ret_3d" in warn_str for warn_str in result["warn_list"])
    assert result["blocks"] and 0 <= result["threshold_share_range_float"] <= 1
