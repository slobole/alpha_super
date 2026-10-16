"""Scout A21: retro verdict rows, the grade vocabulary, the shadow monitor's error isolation, the re-audition shadow
provider's frozen-configuration guard, and the daily run (synthetic data, temporary ledgers)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha.scout import daily as daily_module
from alpha.scout import shadow as shadow_module
from alpha.scout.fdr_report import GRADE_WORD_TUPLE, record_station_verdict
from alpha.scout.ledger import Ledger
from alpha.scout.registration import Registration, register
from alpha.scout.shadow import run_shadow_monitor, start_shadow
from alpha.scout.specs import reaudition_shadow
from alpha.scout.vault import VaultRefusal, latest_grade_str, open_vault

SCRIPT_DIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "scout_ledger_completeness_20261016"


def _registration(registration_id_str: str) -> Registration:
    return Registration(
        registration_id_str=registration_id_str, family_id_str="carry_premium", hypothesis_str="h", mechanism_str="m",
        expected_sign_and_location_str="e", hypothesis_class_str="X", universe_str="u", horizon_str="month",
        schedule_str="s", execution_str="x", param_grid_dict={"a": (1, 2)}, primary_metric_str="p", kill_criteria_str="k",
        source_str="src", universe_choice_str="fixed", retro_bool=True, prior_trials_int=3,
    )


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    register(ledger, _registration("idea"))
    return ledger


def _load_script(name_str: str):
    spec = importlib.util.spec_from_file_location(f"a21_{name_str}", SCRIPT_DIR_PATH / f"{name_str}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grade_vocabulary_and_retro_fields(ledger):
    with pytest.raises(ValueError, match="must start with"):
        record_station_verdict(ledger, "idea", "grade", [("grade", "PROMISING", "")])
    row = record_station_verdict(ledger, "idea", "grade", [("grade", "REJECTED (outside Scout)", "why")], retro_bool=True, source_str="report.md")
    assert row["verdict_str"] == "REJECTED (outside Scout)" and row["retro_bool"] and row["source_str"] == "report.md"
    plain = record_station_verdict(ledger, "idea", "S4", [("cost", "PASS", "")])
    assert "retro_bool" not in plain  # rows written the old way keep their old shape
    assert ledger.verify() == 3


def test_a_qualified_grade_never_opens_the_vault(ledger):
    for grade_str in ("WATCHLIST (no card)", "NOT_TESTED (outside Scout)", "DIAGNOSTIC (no card)"):
        record_station_verdict(ledger, "idea", "grade", [("grade", grade_str, "")], retro_bool=True)
        assert latest_grade_str(ledger, "idea") == grade_str
        with pytest.raises(VaultRefusal):
            open_vault(ledger, "idea", {})


def test_backfill_tables_use_the_vocabulary_and_never_grade_candidate():
    module = _load_script("backfill_verdicts")
    for rid, (word_str, qualifier_str, reason_str, source_str) in module.NO_CARD_GRADE_DICT.items():
        assert word_str in GRADE_WORD_TUPLE and word_str != "CANDIDATE", rid
        assert qualifier_str in (module.NO_CARD_STR, module.OUTSIDE_STR) and reason_str and source_str, rid


def test_october_16_registrations_validate_and_count_each_study_once():
    module = _load_script("register")
    reg_list = module.registration_list()
    for reg in reg_list:
        reg.validate()
        assert reg.retro_bool and reg.prior_trials_int is not None
    assert len({reg.registration_id_str for reg in reg_list}) == 9
    by_id = {reg.registration_id_str: reg for reg in reg_list}
    # the commodity lane (259) is its own row; the equity row is the rest of the 1,080 evaluations
    assert by_id["commodity_dual_momentum_retro_20261016"].prior_trials_int + by_id["three_momentum_directions_equity_retro_20261016"].prior_trials_int == 1080


def _expected_ser() -> pd.Series:
    return pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 800), pd.bdate_range("2020-01-01", periods=800))


def test_shadow_monitor_reports_a_broken_provider_and_keeps_the_others(ledger, tmp_path, monkeypatch):
    register(ledger, _registration("other"))
    good = start_shadow(ledger, "idea", {"a": 1}, _expected_ser(), "test", root_path=tmp_path)
    start_shadow(ledger, "other", {"a": 1}, _expected_ser(), "test", root_path=tmp_path)
    module = type(sys)("fake_provider_module")
    module.good = lambda row: pd.Series(0.001, pd.bdate_range(good["start_date_str"], periods=5))

    def broken(row):
        raise RuntimeError("frozen configuration changed")

    module.broken = broken
    monkeypatch.setitem(sys.modules, "fake_provider_module", module)
    monkeypatch.setattr(shadow_module, "SHADOW_PROVIDER_DICT", {"idea": "fake_provider_module:good", "other": "fake_provider_module:broken"})
    out_path, report_list = run_shadow_monitor(ledger, output_dir_path=tmp_path / "out")
    status_dict = {r["pod_id_str"]: r["status_str"] for r in report_list}
    assert status_dict["other"] == "ERROR" and status_dict["idea"] != "ERROR"
    assert "frozen configuration changed" in (out_path / "shadow.md").read_text(encoding="utf-8")


class _FakeFamily:
    def __init__(self, live_config_dict):
        self.live_config_dict = live_config_dict

    def run_config(self, config_dict):
        class _Result:
            daily_return_ser = pd.Series(0.001, pd.bdate_range("2026-10-01", periods=20))
        return _Result()


def test_reaudition_provider_refuses_a_changed_live_configuration(monkeypatch):
    monkeypatch.setattr(reaudition_shadow, "build_family", lambda rid: _FakeFamily({"cut_tuple": (0.2, 0.6), "entry_int": 6}))
    row = {"registration_id_str": "eom_reaudition_20261002", "start_date_str": "2026-10-17",
           "frozen_config_dict": {"live_config_dict": {"cut_tuple": [0.2, 0.6], "entry_int": 6}}}
    forward_ser = reaudition_shadow.shadow_daily_return_ser(row)
    assert forward_ser.index[0] >= pd.Timestamp("2026-10-17")  # nothing before the start
    row["frozen_config_dict"]["live_config_dict"]["entry_int"] = 7
    with pytest.raises(RuntimeError, match="new registration"):
        reaudition_shadow.shadow_daily_return_ser(row)


def test_daily_run_isolates_parts_and_writes_one_status_file(ledger, tmp_path, monkeypatch):
    monkeypatch.setattr(daily_module, "DEFAULT_FLEX_DB_PATH", tmp_path / "missing.sqlite3")
    monkeypatch.setattr(daily_module, "DEFAULT_FLEX_XML_GLOB", str(tmp_path / "none_*.xml"))
    monkeypatch.setattr(daily_module, "MAIN_CHECKOUT_ROOT_PATH", tmp_path)
    monkeypatch.setattr(shadow_module, "SHADOW_ROOT_PATH", tmp_path / "shadow")

    def broken_report(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("alpha.scout.fdr_report.write_report", broken_report)
    out_path, status_dict = daily_module.run_daily(ledger=ledger, output_root_path=tmp_path / "daily")
    parts = status_dict["parts"]
    assert parts["ledger"]["ok_bool"] and parts["ledger"]["rows_int"] == 1
    assert parts["shadow"]["ok_bool"] and parts["health"]["status_str"] == "NO_DATA"
    assert not parts["fdr"]["ok_bool"] and "disk full" in parts["fdr"]["error_str"]
    assert not status_dict["ok_bool"]
    assert json.loads((out_path / "daily_status.json").read_text(encoding="utf-8"))["as_of_str"] == status_dict["as_of_str"]
    assert "FAILED" in (out_path / "daily_status.md").read_text(encoding="utf-8")
