"""Scout P8: program-wide FDR report, S7 vault opening, S8 shadows, and the dividend-month spec (synthetic data)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha.scout import panel as panel_module
from alpha.scout.fdr_report import fdr_table, record_station_verdict, record_test_result, render_markdown
from alpha.scout.ledger import Ledger
from alpha.scout.panel import FIELD_TUPLE, Panel
from alpha.scout.registration import Registration, register
from alpha.scout.shadow import ShadowRefusal, load_expected, shadow_report, start_shadow
from alpha.scout.vault import VaultRefusal, contamination, open_vault, vault_verdict


def _registration(registration_id_str: str, family_id_str: str = "dividend_seeking_demand", retro_bool: bool = False,
                  published_str: str | None = "2013-09-01") -> Registration:
    return Registration(
        registration_id_str=registration_id_str, family_id_str=family_id_str, hypothesis_str="h", mechanism_str="m",
        expected_sign_and_location_str="e", hypothesis_class_str="X", universe_str="u", horizon_str="month",
        schedule_str="s", execution_str="x", param_grid_dict={"a": (1, 2)}, primary_metric_str="p", kill_criteria_str="k",
        source_str="src", source_published_date_str=published_str, retro_bool=retro_bool,
        prior_trials_int=10 if retro_bool else None, universe_choice_str="fixed",
    )


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    register(ledger, _registration("fresh_idea"))
    return ledger


# ---------------------------------------------------------------- FDR
def test_test_results_need_a_registration_and_a_valid_p(ledger):
    with pytest.raises(ValueError):
        record_test_result(ledger, "unknown", "t", "S3", 0.01)
    with pytest.raises(ValueError):
        record_test_result(ledger, "fresh_idea", "t", "S3", 1.5)
    with pytest.raises(ValueError):
        record_test_result(ledger, "fresh_idea", "t", "S9", 0.5)
    row = record_test_result(ledger, "fresh_idea", "t", "S3", 0.01, primary_bool=True)
    assert row["family_id_str"] == "dividend_seeking_demand"


def test_by_report_uses_the_latest_primary_test_per_registration(ledger):
    for idx_int in range(20):
        register(ledger, _registration(f"noise_{idx_int}"))
        record_test_result(ledger, f"noise_{idx_int}", "gate", "S5", 0.30 + 0.03 * idx_int, primary_bool=True)
    record_test_result(ledger, "fresh_idea", "gate", "S5", 0.0001, primary_bool=True)
    with pytest.raises(ValueError, match="already has a primary"):  # the first primary is final
        record_test_result(ledger, "fresh_idea", "gate", "S5", 0.20, primary_bool=True)
    record_test_result(ledger, "fresh_idea", "rerun", "S5", 0.20, primary_bool=False)
    record_test_result(ledger, "fresh_idea", "other", "S3", 0.9, primary_bool=False)
    table_dict = fdr_table(ledger)
    row_dict = {r["registration_id_str"]: r for r in table_dict["rows"]}
    assert table_dict["test_count_int"] == 21
    assert row_dict["fresh_idea"]["p_value_float"] == 0.0001 and row_dict["fresh_idea"]["discovery_bool"]
    assert row_dict["fresh_idea"]["test_count_int"] == 3
    assert not any(row_dict[f"noise_{i}"]["discovery_bool"] for i in range(20))
    harmonic_float = sum(1.0 / k for k in range(1, 22))
    assert row_dict["fresh_idea"]["by_q_float"] == pytest.approx(min(1.0, harmonic_float * 21 * 0.0001))
    assert "1 discoveries" in render_markdown(table_dict)


def test_untested_registrations_are_listed_but_not_counted(ledger):
    register(ledger, _registration("retro_study", family_id_str="calendar_and_flow", retro_bool=True))
    table_dict = fdr_table(ledger)
    assert table_dict["test_count_int"] == 0
    assert table_dict["families"]["calendar_and_flow"]["prior_trials_int"] == 10
    assert "Registrations without a recorded p (2)" in render_markdown(table_dict)


# ---------------------------------------------------------------- S7 vault
def test_only_a_candidate_opens_the_vault_and_only_once_per_family(ledger):
    with pytest.raises(VaultRefusal, match="graded None"):
        open_vault(ledger, "fresh_idea", {"a": 1})
    record_station_verdict(ledger, "fresh_idea", "grade", [("grade", "WATCHLIST", "")])
    with pytest.raises(VaultRefusal, match="WATCHLIST"):
        open_vault(ledger, "fresh_idea", {"a": 1})
    record_station_verdict(ledger, "fresh_idea", "grade", [("grade", "CANDIDATE (S3 pending)", "")])
    with pytest.raises(VaultRefusal, match="S3 pending"):
        open_vault(ledger, "fresh_idea", {"a": 1})
    with pytest.raises(ValueError, match="at least one"):
        record_station_verdict(ledger, "fresh_idea", "S4", [])
    record_station_verdict(ledger, "fresh_idea", "grade", [("grade", "CANDIDATE", "")])
    row = open_vault(ledger, "fresh_idea", {"a": 1})
    assert row["family_id_str"] == "dividend_seeking_demand" and not row["contaminated_bool"]
    register(ledger, _registration("sibling_idea"))
    record_station_verdict(ledger, "sibling_idea", "grade", [("grade", "CANDIDATE", "")])
    with pytest.raises(VaultRefusal, match="already opened"):
        open_vault(ledger, "sibling_idea", {"a": 2})
    assert sum(1 for _ in ledger.rows("vault_opening")) == 1


def test_contamination_reasons(ledger):
    assert not contamination(ledger, "fresh_idea")["contaminated_bool"]
    register(ledger, _registration("late_source", family_id_str="merger_arbitrage", published_str="2024-02-01"))
    reason_list = contamination(ledger, "late_source")["reason_list"]
    assert any("published 2024-02-01" in r for r in reason_list) and any("seen after the seal" in r for r in reason_list)
    register(ledger, _registration("old_study", retro_bool=True))
    assert any("retro registrations" in r for r in contamination(ledger, "fresh_idea")["reason_list"])


def test_a_shadow_contaminates_its_family_and_a_closed_candidate_must_open_the_vault_first(ledger, tmp_path):
    register(ledger, _registration("watched_idea"))
    record_station_verdict(ledger, "watched_idea", "grade", [("grade", "WATCHLIST", "")])
    start_shadow(ledger, "watched_idea", {}, _expected_ser(), "test", root_path=tmp_path)
    assert any("shadows" in r for r in contamination(ledger, "fresh_idea")["reason_list"])
    record_station_verdict(ledger, "fresh_idea", "grade", [("grade", "CANDIDATE", "")])
    with pytest.raises(ShadowRefusal, match="run S7 first"):
        start_shadow(ledger, "fresh_idea", {}, _expected_ser(), "test", root_path=tmp_path)
    opening = open_vault(ledger, "fresh_idea", {"a": 1})
    assert opening["contaminated_bool"]
    start_shadow(ledger, "fresh_idea", {}, _expected_ser(), "test", root_path=tmp_path)


def test_vault_verdict_is_tied_to_its_opening_once_per_family(ledger):
    from alpha.scout.vault import record_vault_verdict

    record_station_verdict(ledger, "fresh_idea", "grade", [("grade", "CANDIDATE", "")])
    opening_row_id_int = open_vault(ledger, "fresh_idea", {"a": 1})["row_id_int"]
    verdict_dict = {"verdict_str": "PASS", "start_str": "2023-01-03", "end_str": "2026-09-30", "vault_return_float": 0.1,
                    "tbill_return_float": 0.05, "vault_sharpe_float": 1.0, "bootstrap_floor_sharpe_float": 0.1}
    with pytest.raises(VaultRefusal, match="is not"):
        record_vault_verdict(ledger, "fresh_idea", opening_row_id_int + 99, verdict_dict)
    record_vault_verdict(ledger, "fresh_idea", opening_row_id_int, verdict_dict)
    with pytest.raises(VaultRefusal, match="already has"):
        record_vault_verdict(ledger, "fresh_idea", opening_row_id_int, verdict_dict)


def test_vault_verdict_pass_fail_and_contaminated():
    rng_obj = np.random.default_rng(0)
    in_index = pd.bdate_range("2010-01-01", "2022-12-30")
    vault_index = pd.bdate_range("2023-01-02", "2026-09-30")
    in_sample_ser = pd.Series(rng_obj.normal(0.0006, 0.01, len(in_index)), in_index)
    tbill_ser = pd.Series(0.0001, vault_index)
    good_ser = pd.Series(rng_obj.normal(0.0006, 0.01, len(vault_index)), vault_index)
    bad_ser = pd.Series(rng_obj.normal(-0.0010, 0.01, len(vault_index)), vault_index)
    assert vault_verdict(good_ser, in_sample_ser, tbill_ser, False)["verdict_str"] == "PASS"
    failed = vault_verdict(bad_ser, in_sample_ser, tbill_ser, False)
    assert failed["verdict_str"] == "FAIL" and not failed["beats_tbill_bool"]
    contaminated = vault_verdict(good_ser, in_sample_ser, tbill_ser, True)
    assert contaminated["verdict_str"] == "CONTAMINATED" and contaminated["verdict_if_clean_str"] == "PASS"
    # The in-sample yardstick never reads a vault bar.
    leaked = pd.concat([in_sample_ser, pd.Series(1.0, vault_index)])
    assert vault_verdict(good_ser, leaked, tbill_ser, False)["in_sample_sharpe_float"] == pytest.approx(
        vault_verdict(good_ser, in_sample_ser, tbill_ser, False)["in_sample_sharpe_float"])


# ---------------------------------------------------------------- S8 shadow
def _expected_ser() -> pd.Series:
    index = pd.bdate_range("2005-01-03", "2022-12-30")
    return pd.Series(np.random.default_rng(1).normal(0.0004, 0.008, len(index)), index)


def test_shadow_start_rules(ledger, tmp_path):
    registered_date = pd.Timestamp(next(ledger.rows("registration"))["utc_ts_str"][:10])
    with pytest.raises(ShadowRefusal, match="after both the registration and today"):
        start_shadow(ledger, "fresh_idea", {}, _expected_ser(), "test", start_date_str=str(registered_date.date()), root_path=tmp_path)
    late_ser = pd.Series(0.001, pd.bdate_range("2020-01-01", registered_date + pd.Timedelta(days=30)))
    with pytest.raises(ShadowRefusal, match="end before"):
        start_shadow(ledger, "fresh_idea", {}, late_ser, "test", root_path=tmp_path)
    row = start_shadow(ledger, "fresh_idea", {"a": 1}, _expected_ser(), "test", root_path=tmp_path)
    assert row["start_date_str"] > str(registered_date.date()) and row["min_trl_months_float"] > 0
    with pytest.raises(ShadowRefusal, match="already has a shadow"):
        start_shadow(ledger, "fresh_idea", {"a": 1}, _expected_ser() * 3, "test", root_path=tmp_path)
    assert load_expected(row).equals(_expected_ser().rename("expected_return_float"))  # a refused retry leaves the file alone
    path = tmp_path / "fresh_idea" / "expected.parquet"
    (_expected_ser() * 2).rename("expected_return_float").to_frame().to_parquet(path)
    with pytest.raises(RuntimeError, match="SHA-256"):
        load_expected(row)


def test_shadow_report_reads_only_the_forward_period(ledger, tmp_path):
    row = start_shadow(ledger, "fresh_idea", {"a": 1}, _expected_ser(), "test", root_path=tmp_path)
    start_ts = pd.Timestamp(row["start_date_str"])
    before_index = pd.bdate_range(start_ts - pd.Timedelta(days=400), start_ts - pd.Timedelta(days=1))
    no_forward = shadow_report(row, pd.Series(-0.05, before_index), start_ts.date())
    assert no_forward["status_str"] == "NO_DATA" and no_forward["shadow"]["forward_sessions_int"] == 0
    forward_index = pd.bdate_range(start_ts, periods=30)
    realized_ser = pd.concat([pd.Series(-0.05, before_index), pd.Series(np.random.default_rng(2).normal(0.0004, 0.008, 30), forward_index)])
    report_dict = shadow_report(row, realized_ser, forward_index[-1].date())
    assert report_dict["shadow"]["forward_sessions_int"] == 30 and report_dict["live"]["start_str"] == str(forward_index[0].date())
    assert report_dict["status_str"] in ("GREEN", "AMBER", "TOO_EARLY")


def test_shadow_start_row_opens_the_panel_only_for_its_family(tmp_path, monkeypatch):
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
    shadow_row_id_int = ledger.append("shadow_start", {"family_id_str": "mine", "registration_id_str": "x"})["row_id_int"]
    note_row_id_int = ledger.append("note", {"family_id_str": "mine"})["row_id_int"]
    assert panel_module.load_panel("Test", family_id_str="other", shadow_start_row_id_int=shadow_row_id_int, ledger=ledger).sealed_bool
    assert panel_module.load_panel("Test", family_id_str="mine", shadow_start_row_id_int=note_row_id_int, ledger=ledger).sealed_bool
    opened = panel_module.load_panel("Test", family_id_str="mine", shadow_start_row_id_int=shadow_row_id_int, ledger=ledger)
    assert not opened.sealed_bool and opened.date_index[-1] == date_index[-1]


# ---------------------------------------------------------------- dividend-month spec
def _dividend_panel(premium_float: float, seed_int: int = 0) -> Panel:
    """40 quarterly payers in four cohorts (ex-months 1/4/7/10, 2/5/8/11, ...), 20 non-payers; a payer gains
    premium_float a day during its ex-dividend month (a planted dividend month premium)."""
    date_index = pd.bdate_range("2008-01-01", "2022-12-30")
    symbol_list = [f"S{i:02d}" for i in range(60)]
    rng_obj = np.random.default_rng(seed_int)
    return_mat = rng_obj.normal(0.0003, 0.012, (len(date_index), len(symbol_list)))
    month_vec = date_index.month.to_numpy()
    dividend_mat = np.zeros_like(return_mat)
    for idx_int in range(40):
        cohort_int = idx_int % 3
        in_month_vec = (month_vec - 1) % 3 == cohort_int
        return_mat[in_month_vec, idx_int] += premium_float
        # one ex-date per payer month, the 10th session of the month (booked on the session before it)
        month_start_vec = np.flatnonzero(np.r_[True, month_vec[1:] != month_vec[:-1]])
        for start_int in month_start_vec:
            if (month_vec[start_int] - 1) % 3 == cohort_int and start_int + 10 < len(date_index):
                dividend_mat[start_int + 9, idx_int] = 0.2
    close_mat = 50 * np.cumprod(1 + return_mat, axis=0)
    close_df = pd.DataFrame(close_mat, index=date_index, columns=symbol_list)
    open_df = close_df.shift(1).fillna(close_df.iloc[0])
    field_dict = {"Open": open_df, "High": np.maximum(open_df, close_df), "Low": np.minimum(open_df, close_df), "Close": close_df,
                  "Volume": close_df * 0 + 1e6, "Turnover": close_df * 1e6, "Unadjusted Close": close_df,
                  "Dividend": pd.DataFrame(dividend_mat, index=date_index, columns=symbol_list)}
    member_df = pd.DataFrame(1, index=date_index, columns=symbol_list, dtype=np.int8)
    return Panel("synthetic", field_dict, member_df, "synthetic", sealed_bool=True)


def test_predicted_payers_use_only_past_ex_months():
    from alpha.scout.specs import dividend_month as dm
    from alpha.scout.stations.s1_causality import run_s1

    panel = _dividend_panel(0.0)
    row_vec = dm.decision_row_vec(panel.date_index)
    predicted = dm.predicted_mat(panel, row_vec, "m3")
    target_month_vec = (panel.date_index[row_vec].month.to_numpy() % 12) + 1
    late_rows = np.flatnonzero(row_vec > 300)
    for row_int in late_rows[:12]:
        expected_vec = np.array([(target_month_vec[row_int] - 1) % 3 == i % 3 for i in range(40)] + [False] * 20)
        assert np.array_equal(predicted[row_int], expected_vec)
    assert all(not r.hard_fail_bool for r in run_s1([dm.predicted_feature(rule) for rule in dm.LOOKBACK_RULE_TUPLE], panel, sample_size_int=20))


def test_fast_replica_finds_a_planted_premium_and_a_fixed_signal_null_does_not():
    from alpha.scout.null import permuted_panel
    from alpha.scout.specs import dividend_month as dm

    panel = _dividend_panel(0.0008)
    config_list = [{"lookback_rule_str": "m3", "weighting_str": "EW"}]
    daily_list, baseline_vec = dm.fast_daily_list_panel(panel, config_list)
    active_vec = (daily_list[0] - baseline_vec)[300:]
    assert active_vec.mean() / active_vec.std() * np.sqrt(252) > 1.0
    signal_dict = dm.mcpt_signal_dict(panel, config_list)
    null_list, null_baseline_vec = dm.fast_daily_list_panel(permuted_panel(panel, np.random.default_rng(3), {}), config_list, signal_dict=signal_dict)
    null_active_vec = (null_list[0] - null_baseline_vec)[300:]
    assert abs(null_active_vec.mean() / null_active_vec.std() * np.sqrt(252)) < 1.0


def test_total_return_panel_moves_the_dividend_to_its_ex_date():
    from alpha.scout.specs import dividend_month as dm

    panel = _dividend_panel(0.0)
    tr_close_df = dm.total_return_panel(panel).field("Close")
    raw_close_df = panel.field("Close")
    booked_ts = panel.field("Dividend")["S00"].gt(0).idxmax()
    ex_ts = panel.date_index[panel.date_index.get_loc(booked_ts) + 1]
    raw_ratio_float = raw_close_df.loc[ex_ts, "S00"] / raw_close_df.loc[booked_ts, "S00"]
    tr_ratio_float = tr_close_df.loc[ex_ts, "S00"] / tr_close_df.loc[booked_ts, "S00"]
    assert tr_ratio_float == pytest.approx(raw_ratio_float * (1 + 0.2 / raw_close_df.loc[booked_ts, "S00"]))
    booked_ratio_float = tr_close_df.loc[booked_ts, "S00"] / tr_close_df.shift(1).loc[booked_ts, "S00"]
    assert booked_ratio_float == pytest.approx(raw_close_df.loc[booked_ts, "S00"] / raw_close_df.shift(1).loc[booked_ts, "S00"])
