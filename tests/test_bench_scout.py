"""Scout pages in Bench: every page renders from small synthetic files, and with all files missing."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alpha.bench import scout_pages
from alpha.bench.app import create_app

PAGE_LIST = ["/scout", "/scout/shadows", "/scout/ledger", "/scout/families"]
POD_ID_STR = "demo_shadow_20261001"


def _write_json(path_obj, value_obj) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(value_obj), encoding="utf-8")


@pytest.fixture()
def empty_client(tmp_path, monkeypatch):
    monkeypatch.setattr(scout_pages, "SCOUT_ROOT_PATH", tmp_path / "none" / "scout")
    monkeypatch.setattr(scout_pages, "LEDGER_PATH", tmp_path / "none" / "ledger.jsonl")
    monkeypatch.setattr(scout_pages, "CARDS_ROOT_PATH", tmp_path / "none" / "cards")
    monkeypatch.setattr(scout_pages, "REPO_ROOT_PATH", tmp_path / "none")
    return create_app().test_client()


@pytest.fixture()
def seeded_client(tmp_path, monkeypatch):
    scout_root = tmp_path / "results" / "scout"
    ledger_path = tmp_path / "research_ledger" / "scout_ledger.jsonl"
    monkeypatch.setattr(scout_pages, "REPO_ROOT_PATH", tmp_path)
    monkeypatch.setattr(scout_pages, "SCOUT_ROOT_PATH", scout_root)
    monkeypatch.setattr(scout_pages, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(scout_pages, "CARDS_ROOT_PATH", scout_root / "cards")

    _write_json(scout_root / "daily" / "2026-11-07" / "daily_status.json", {
        "as_of_str": "2026-11-07", "written_utc_str": "2026-11-07T10:12:52+00:00", "ok_bool": True,
        "parts": {
            "ledger": {"rows_int": 5, "ok_bool": True},
            "shadow": {"ok_bool": True, "status_by_registration_dict": {POD_ID_STR: "TOO_EARLY"}},
            "health": {"ok_bool": True, "status_by_pod_dict": {"TAA 3x": "GREEN", "NDX VXN": "RED"}},
            "fdr": {"ok_bool": True, "registrations_int": 2, "primary_tests_int": 2, "discoveries": ["reg_a_20261001"]},
            "kb": {"ok_bool": True, "status_str": "OK", "gap_count_int": 0, "overdue_id_list": []},
        }})
    _write_json(scout_root / "pod_health" / "2026-11-07" / "pod_health.json", [
        {"label_str": "TAA 3x", "status_str": "GREEN", "as_of_str": "2026-11-07",
         "live": {"end_str": "2026-10-30", "sessions_behind_int": 6, "stale_bool": True, "cumulative_return_float": 0.05,
                  "current_drawdown_float": 0.02, "session_count_int": 80}},
        {"label_str": "NDX VXN", "status_str": "RED", "as_of_str": "2026-11-07", "live": {"end_str": "2026-11-06"}}])
    _write_json(scout_root / "shadow" / "reports" / "2026-11-07" / "shadow.json", [
        {"pod_id_str": POD_ID_STR, "status_str": "TOO_EARLY",
         "shadow": {"start_date_str": "2026-10-17", "forward_sessions_int": 5, "forward_months_float": 0.2,
                    "min_trl_months_float": 21.5, "progress_float": 0.01, "expected_annual_sharpe_float": 1.2},
         "live": {"cumulative_return_float": 0.01}},
        {"pod_id_str": "no_log_shadow", "status_str": "NO_DATA", "shadow": {"start_date_str": "2026-11-07"}, "live": {}}])
    shadow_dir = scout_root / "shadow" / POD_ID_STR
    shadow_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    pd.DataFrame({"expected_return_float": rng.normal(0.0005, 0.01, 600)}).to_parquet(shadow_dir / "expected.parquet")
    (shadow_dir / "forward_log.csv").write_text(
        "date_str,return_float,first_recorded_str\n2026-10-19,0.01,2026-10-19\n2026-10-20,-0.004,2026-10-20\n", encoding="utf-8")
    (shadow_dir / "forward_revisions.csv").write_text("date_str,logged_return_float,rebuilt_return_float,seen_on_str\n", encoding="utf-8")
    _write_json(scout_root / "fdr" / "2026-11-07" / "fdr_report.json", {"rows": [
        {"registration_id_str": "reg_a_20261001", "family_id_str": "fam_one", "discovery_bool": True},
        {"registration_id_str": "reg_b_20261002", "family_id_str": "fam_two", "discovery_bool": False}]})
    (scout_root / "cards").mkdir(parents=True)
    (scout_root / "cards" / "Reg_A.html").write_text("<html></html>", encoding="utf-8")
    (scout_root / "cards" / "reg_b_20261002.html").write_text("<html></html>", encoding="utf-8")

    row_list = [
        {"row_type_str": "registration", "registration_id_str": "reg_a_20261001", "family_id_str": "fam_one",
         "prior_trials_int": 10, "retro_bool": True, "utc_ts_str": "2026-10-01T10:00:00+00:00"},
        {"row_type_str": "registration", "registration_id_str": "reg_b_20261002", "family_id_str": "fam_two",
         "prior_trials_int": 4, "retro_bool": False, "utc_ts_str": "2026-10-02T10:00:00+00:00"},
        {"row_type_str": "station_verdict", "station_str": "grade", "registration_id_str": "reg_a_20261001",
         "verdict_str": "WATCHLIST (shadow only)"},
        {"row_type_str": "station_verdict", "station_str": "grade", "registration_id_str": "reg_b_20261002",
         "verdict_str": "REJECTED"},
        {"row_type_str": "station_verdict", "station_str": "s5", "registration_id_str": "reg_b_20261002", "verdict_str": "PASS"},
        {"row_type_str": "shadow_start", "registration_id_str": POD_ID_STR,
         "expected_path_str": f"results\\scout\\shadow\\{POD_ID_STR}\\expected.parquet"},
    ]
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text("\n".join(json.dumps(r) for r in row_list) + "\n", encoding="utf-8")
    return create_app().test_client()


@pytest.mark.parametrize("path_str", PAGE_LIST)
def test_pages_render_with_all_files_missing(empty_client, path_str):
    response = empty_client.get(path_str)
    assert response.status_code == 200
    assert "Scout" in response.get_data(as_text=True)


def test_overview_shows_status_and_stale_pods(seeded_client):
    text = seeded_client.get("/scout").get_data(as_text=True)
    assert "5 rows, chain intact" in text
    assert "0 overdue, 0 pending" in text
    assert "1 discoveries from 2 tests" in text
    assert "TAA 3x" in text and "NDX VXN" in text
    assert "STALE, 6 sessions behind" in text
    assert "2026-10-30" in text


def test_shadows_page_draws_chart_and_missing_log_note(seeded_client):
    text = seeded_client.get("/scout/shadows").get_data(as_text=True)
    assert POD_ID_STR in text
    assert 'class="sc-fwd"' in text and 'class="sc-band"' in text
    assert "0.2 of 21.5" in text
    assert "The forward log starts 2026-11-07" in text


def test_ledger_page_lists_grade_filter_and_card_links(seeded_client):
    text = seeded_client.get("/scout/ledger").get_data(as_text=True)
    assert "reg_a_20261001" in text and "reg_b_20261002" in text
    assert 'data-grade="WATCHLIST"' in text and 'data-grade="REJECTED"' in text
    assert "/artifact/scout/cards/reg_b_20261002.html" in text
    assert "/artifact/scout/cards/Reg_A.html" in text  # card name plus the id date suffix


def test_families_page_counts_trials_grades_and_discoveries(seeded_client):
    text = seeded_client.get("/scout/families").get_data(as_text=True)
    assert "fam_one" in text and "fam_two" in text
    assert "Watchlist" in text and "Rejected" in text
    assert "Discoveries" in text


def test_nav_has_scout_link(seeded_client):
    assert 'href="/scout"' in seeded_client.get("/scout").get_data(as_text=True)
