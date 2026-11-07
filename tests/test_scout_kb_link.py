"""Scout A22/A23: the knowledge base -> ledger link (gap list, pre-Scout cut, grace period, daily part fails on an
overdue gap) and the A22 retro registrations (synthetic registry, temporary ledgers)."""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

from alpha.scout import daily as daily_module
from alpha.scout import shadow as shadow_module
from alpha.scout.fdr_report import GRADE_WORD_TUPLE
from alpha.scout.kb_link import cites_bool, kb_link_report, study_folder_str
from alpha.scout.ledger import Ledger
from alpha.scout.registration import Registration, register

SCRIPT_DIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "scout_kb_link_20261017"


def _registration(registration_id_str: str, source_str: str) -> Registration:
    return Registration(
        registration_id_str=registration_id_str, family_id_str="carry_premium", hypothesis_str="h", mechanism_str="m",
        expected_sign_and_location_str="e", hypothesis_class_str="X", universe_str="u", horizon_str="month",
        schedule_str="s", execution_str="x", param_grid_dict={"a": (1, 2)}, primary_metric_str="p", kill_criteria_str="k",
        source_str=source_str, universe_choice_str="fixed", retro_bool=True, prior_trials_int=3,
    )


def _study(study_id_str: str, concise_report_str: str | None, created_str: str | None) -> dict:
    return {"study_id": study_id_str, "concise_report": concise_report_str, "created_at": created_str}


@pytest.fixture
def registry_path(tmp_path) -> Path:
    study_list = [
        _study("cited_study", "pakal-research/reports/cited_study/REPORT.md", "2026-10-10T09:00:00+00:00"),
        # Windows separators and a nested folder, cited by its full sub-path
        _study("nested_lane", "C:\\w\\pakal-research\\reports\\parent_study\\lane_a\\REPORT.md", "2026-10-16T13:00:00+00:00"),
        _study("october_gap", "pakal-research/reports/october_gap/REPORT.md", "2026-10-01T00:00:01+00:00"),
        _study("old_study", "pakal-research/reports/old_study/REPORT.md", "2026-09-30T23:59:59+00:00"),
        _study("undated_gap", "pakal-research/reports/undated_gap/REPORT.md", None),
        _study("book_picture_20261014", "pakal-research/reports/book_picture_20261014/REPORT.md", "2026-10-14T00:00:00+00:00"),
        # a sub-folder citation also cites its parent folder
        _study("parent_study", "pakal-research/reports/parent_study/REPORT.md", "2026-10-16T12:00:00+00:00"),
        # a longer folder name does not cite its prefix
        _study("prefix", "pakal-research/reports/prefix/REPORT.md", "2026-10-11T12:00:00+00:00"),
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"studies": study_list}), encoding="utf-8")
    return path


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    register(ledger, _registration("a", "Paper; pakal-research/reports/cited_study (12 evaluations)."))
    register(ledger, _registration("b", "pakal-research/reports/parent_study/lane_a (3 evaluations)."))
    register(ledger, _registration("p", "pakal-research/reports/prefix_v2 (4 evaluations)."))
    return ledger


def test_folder_is_read_from_the_concise_report_path():
    assert study_folder_str("pakal-research/reports/x_study/REPORT.md") == "x_study"
    assert study_folder_str("C:\\a\\reports\\x\\lane\\REPORT_FULL.md") == "x/lane"
    assert study_folder_str("docs/research/NOTE.md") is None and study_folder_str(None) is None
    assert cites_bool("Paper; pakal-research/reports/x_study (3 evaluations).", "x_study")
    assert cites_bool("pakal-research/reports/x_study/lane (3).", "x_study") and cites_bool("see reports/x_study.", "x_study")
    assert not cites_bool("pakal-research/reports/x_study_v2 (3).", "x_study")


def test_gaps_are_october_studies_without_a_citing_registration(ledger, registry_path):
    report_dict = kb_link_report(ledger, registry_path)
    id_set = lambda key: {e["study_id_str"] for e in report_dict[key]}  # noqa: E731
    assert id_set("cited") == {"cited_study", "nested_lane", "parent_study"}
    assert id_set("gaps") == {"october_gap", "undated_gap", "prefix"}
    assert id_set("pre_scout") == {"old_study"}
    assert id_set("not_registrable") == {"book_picture_20261014"}
    assert report_dict["gap_count_int"] == 3 and report_dict["studies_int"] == 8

    register(ledger, _registration("c", "pakal-research/reports/october_gap (5 evaluations)."))
    assert "october_gap" not in {e["study_id_str"] for e in kb_link_report(ledger, registry_path)["gaps"]}


def _quiet_daily(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_module, "DEFAULT_FLEX_DB_PATH", tmp_path / "missing.sqlite3")
    monkeypatch.setattr(daily_module, "DEFAULT_FLEX_XML_GLOB", str(tmp_path / "none_*.xml"))
    monkeypatch.setattr(daily_module, "MAIN_CHECKOUT_ROOT_PATH", tmp_path)
    monkeypatch.setattr(shadow_module, "SHADOW_ROOT_PATH", tmp_path / "shadow")


def test_gaps_older_than_the_grace_period_are_overdue(ledger, registry_path):
    # cutoff 2026-10-09: october_gap (10-01) and the undated gap are overdue, prefix (10-11) is still pending
    report_dict = kb_link_report(ledger, registry_path, as_of_date=date(2026, 10, 12))
    assert {e["study_id_str"] for e in report_dict["overdue"]} == {"october_gap", "undated_gap"}
    assert {e["study_id_str"] for e in report_dict["pending"]} == {"prefix"}
    assert kb_link_report(ledger, registry_path, as_of_date=date(2026, 10, 15))["overdue_count_int"] == 3


def test_daily_kb_part_fails_on_an_overdue_gap_and_passes_on_pending_ones(ledger, registry_path, tmp_path, monkeypatch):
    _quiet_daily(monkeypatch, tmp_path)
    monkeypatch.setattr("alpha.scout.kb_link.KB_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(kb_link_report, "__defaults__", (registry_path,))
    as_of_date = date(2026, 10, 12)
    out_path, status_dict = daily_module.run_daily(as_of_date=as_of_date, ledger=ledger, output_root_path=tmp_path / "daily")
    kb_dict = status_dict["parts"]["kb"]
    assert not kb_dict["ok_bool"] and kb_dict["status_str"] == "OVERDUE" and kb_dict["gap_count_int"] == 3
    assert set(kb_dict["overdue_id_list"]) == {"october_gap", "undated_gap"}
    assert not status_dict["ok_bool"]  # an overdue study fails the run
    assert "october_gap" in (out_path / "daily_status.md").read_text(encoding="utf-8")

    register(ledger, _registration("c", "pakal-research/reports/october_gap (5 evaluations)."))
    register(ledger, _registration("d", "pakal-research/reports/undated_gap (2 evaluations)."))
    _, status_dict = daily_module.run_daily(as_of_date=as_of_date, ledger=ledger, output_root_path=tmp_path / "daily2")
    assert status_dict["parts"]["kb"]["status_str"] == "PENDING" and status_dict["ok_bool"]

    monkeypatch.setattr(kb_link_report, "__defaults__", (tmp_path / "no_such_registry.json",))
    _, status_dict = daily_module.run_daily(as_of_date=as_of_date, ledger=ledger, output_root_path=tmp_path / "daily3")
    assert status_dict["parts"]["kb"]["status_str"] == "UNREADABLE" and not status_dict["ok_bool"]


def _load_script(name_str: str):
    spec = importlib.util.spec_from_file_location(f"a22_{name_str}", SCRIPT_DIR_PATH / f"{name_str}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a22_registrations_validate_cite_their_study_and_are_graded():
    module = _load_script("register")
    reg_list = module.registration_list()
    id_list = [reg.registration_id_str for reg in reg_list]
    assert len(set(id_list)) == len(id_list)
    for reg in reg_list:
        reg.validate()
        assert reg.retro_bool and reg.prior_trials_int is not None and "pakal-research/reports/" in reg.source_str
    # every study of the A22 list is cited by at least one registration, and each study's evaluations count once
    for study_id_str, (folder_str, total_int) in module.STUDY_TOTAL_DICT.items():
        citing_list = [reg for reg in reg_list if f"reports/{folder_str} (" in reg.source_str]  # the row's own study, not one it mentions
        assert citing_list, study_id_str
        assert sum(reg.prior_trials_int for reg in citing_list) == total_int, study_id_str
    for rid, (word_str, reason_str, source_str) in module.GRADE_DICT.items():
        assert rid in id_list and word_str in GRADE_WORD_TUPLE and word_str != "CANDIDATE" and reason_str and source_str
    assert set(module.GRADE_DICT) == set(id_list)


A23_DIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "scout_kb_link_20261107"


def test_a23_registrations_validate_sum_to_each_study_and_are_ordered_parent_first():
    spec = importlib.util.spec_from_file_location("a23_register", A23_DIR_PATH / "register.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row_list, reg_list, grade_dict = module.row_list(), module.registration_list(), module.grade_dict()
    id_list = [reg.registration_id_str for reg in reg_list]
    assert len(set(id_list)) == len(id_list) == len(row_list) and set(grade_dict) == set(id_list)
    total_dict: dict[str, int] = {}
    for row, reg in zip(row_list, reg_list):
        reg.validate()
        assert reg.retro_bool and f"reports/{row['folder']} (" in reg.source_str
        assert cites_bool(reg.source_str, row["folder"])
        total_dict[row["study_id"]] = total_dict.get(row["study_id"], 0) + reg.prior_trials_int
        assert reg.prior_trials_int <= row["kb_total_int"]
    for row in row_list:  # each study's rows sum to its counted total, never above the knowledge-base count
        assert total_dict[row["study_id"]] == row["counted_total_int"] <= row["kb_total_int"], row["study_id"]
        # a count below the study's own is explained in the permanent row
        assert row["counted_total_int"] == row["kb_total_int"] or row["count_note"] in module.source_str(row), row["study_id"]
    placed_set = set()
    for reg in reg_list:  # a parent inside the batch comes first and sits in the same family
        if reg.parent_id_str in id_list:
            assert reg.parent_id_str in placed_set
            assert reg_list[id_list.index(reg.parent_id_str)].family_id_str == reg.family_id_str
        placed_set.add(reg.registration_id_str)
    for word_str, reason_str, source_str in grade_dict.values():
        assert word_str in GRADE_WORD_TUPLE and word_str != "CANDIDATE" and reason_str and source_str
