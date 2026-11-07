"""Scout A24: the append-only forward log of each shadow (new sessions appended, logged sessions never rewritten,
a rebuilt session that differs is recorded as a revision)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from alpha.scout.shadow import FORWARD_LOG_NAME_STR, REVISION_LOG_NAME_STR, log_forward


def _ser(value_dict: dict) -> pd.Series:
    return pd.Series(list(value_dict.values()), index=pd.to_datetime(list(value_dict)))


def test_new_sessions_are_appended_and_logged_ones_are_kept(tmp_path):
    first = log_forward("reg", _ser({"2026-11-02": 0.01, "2026-11-03": -0.02}), date(2026, 11, 3), root_path=tmp_path)
    assert first == {**first, "logged_sessions_int": 2, "appended_int": 2, "revised_today_int": 0, "revisions_total_int": 0}

    # same two days unchanged + one new day: only the new day is appended, with its own first-recorded date
    second = log_forward("reg", _ser({"2026-11-02": 0.01, "2026-11-03": -0.02, "2026-11-04": 0.005}), date(2026, 11, 4),
                         root_path=tmp_path)
    assert second["appended_int"] == 1 and second["logged_sessions_int"] == 3 and second["revisions_total_int"] == 0
    log_df = pd.read_csv(tmp_path / "reg" / FORWARD_LOG_NAME_STR, dtype={"date_str": str})
    assert log_df["date_str"].tolist() == ["2026-11-02", "2026-11-03", "2026-11-04"]
    assert log_df["first_recorded_str"].tolist() == ["2026-11-03", "2026-11-03", "2026-11-04"]


def test_a_rebuilt_session_that_differs_is_a_revision_and_the_log_keeps_the_first_value(tmp_path):
    log_forward("reg", _ser({"2026-11-02": 0.01, "2026-11-03": -0.02}), date(2026, 11, 3), root_path=tmp_path)
    # a data correction changes 11-03; a difference inside the tolerance on 11-02 is not a revision
    result = log_forward("reg", _ser({"2026-11-02": 0.01 + 1e-12, "2026-11-03": -0.015}), date(2026, 11, 4), root_path=tmp_path)
    assert result["revised_today_int"] == 1 and result["revisions_total_int"] == 1 and result["appended_int"] == 0
    revision_df = pd.read_csv(tmp_path / "reg" / REVISION_LOG_NAME_STR, dtype={"date_str": str})
    assert revision_df.iloc[0].to_dict() == {"date_str": "2026-11-03", "logged_return_float": -0.02,
                                             "rebuilt_return_float": -0.015, "seen_on_str": "2026-11-04"}
    log_df = pd.read_csv(tmp_path / "reg" / FORWARD_LOG_NAME_STR, dtype={"date_str": str})
    assert log_df.loc[log_df["date_str"] == "2026-11-03", "return_float"].item() == -0.02  # never rewritten

    # the same revised value seen again is not written again; a second change to the same session is
    again = log_forward("reg", _ser({"2026-11-03": -0.015}), date(2026, 11, 5), root_path=tmp_path)
    assert again["revised_today_int"] == 0 and again["revisions_total_int"] == 1
    third = log_forward("reg", _ser({"2026-11-03": -0.011}), date(2026, 11, 6), root_path=tmp_path)
    assert third["revised_today_int"] == 1 and third["revisions_total_int"] == 1  # still one revised session
    assert len(pd.read_csv(tmp_path / "reg" / REVISION_LOG_NAME_STR)) == 2
