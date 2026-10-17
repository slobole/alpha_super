"""Scout A22: the knowledge base -> ledger link (gap list, pre-Scout cut, daily part never fails the run) and the
A22 retro registrations (synthetic registry, temporary ledgers)."""

from __future__ import annotations

import importlib.util
import json
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


def test_daily_kb_part_reports_gaps_and_never_fails_the_run(ledger, registry_path, tmp_path, monkeypatch):
    _quiet_daily(monkeypatch, tmp_path)
    monkeypatch.setattr("alpha.scout.kb_link.KB_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(kb_link_report, "__defaults__", (registry_path,))
    out_path, status_dict = daily_module.run_daily(ledger=ledger, output_root_path=tmp_path / "daily")
    kb_dict = status_dict["parts"]["kb"]
    assert kb_dict["ok_bool"] and kb_dict["status_str"] == "GAPS" and kb_dict["gap_count_int"] == 3
    assert set(kb_dict["gap_id_list"]) == {"october_gap", "undated_gap", "prefix"}
    assert status_dict["ok_bool"]  # gaps are reported, the run still passes
    assert "october_gap" in (out_path / "daily_status.md").read_text(encoding="utf-8")

    monkeypatch.setattr(kb_link_report, "__defaults__", (tmp_path / "no_such_registry.json",))
    _, status_dict = daily_module.run_daily(ledger=ledger, output_root_path=tmp_path / "daily2")
    assert status_dict["parts"]["kb"]["status_str"] == "UNREADABLE" and status_dict["ok_bool"]


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
