"""Scout A27: the quant verdict (forward-proof source, error isolation, report files) and a smoke run on a saved
bundle when one is present."""

from __future__ import annotations

import json

import pytest

from alpha.scout import verdict as verdict_module
from alpha.scout.ledger import Ledger


def test_forward_dict_reads_the_shadow_log_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(verdict_module, "SHADOW_ROOT_PATH", tmp_path / "shadow")
    folder_path = tmp_path / "shadow" / "dv2_reaudition_20261002"
    folder_path.mkdir(parents=True)
    rows_str = "date_str,return_float,first_recorded_str\n" + "".join(f"2026-10-{d:02d},0.001,2026-11-07\n" for d in range(1, 22))
    (folder_path / "forward_log.csv").write_text(rows_str, encoding="utf-8")
    (tmp_path / "shadow" / "reports" / "2026-11-07").mkdir(parents=True)
    (tmp_path / "shadow" / "reports" / "2026-11-07" / "shadow.json").write_text(
        json.dumps([{"pod_id_str": "dv2_reaudition_20261002", "status_str": "TOO_EARLY"}]), encoding="utf-8")
    forward_dict = verdict_module._forward_dict("DV2", 35.0)
    assert forward_dict["months_float"] == pytest.approx(1.0) and forward_dict["status_str"] == "TOO_EARLY"
    assert verdict_module._forward_dict("UNKNOWN_POD", None)["source_str"] == "none"


def test_forward_dict_reads_live_pod_health(tmp_path, monkeypatch):
    monkeypatch.setattr(verdict_module, "POD_HEALTH_ROOT_PATH", tmp_path / "ph")
    (tmp_path / "ph" / "2026-11-07").mkdir(parents=True)
    (tmp_path / "ph" / "2026-11-07" / "pod_health.json").write_text(
        json.dumps([{"label_str": "NDX VXN", "status_str": "RED", "live": {"session_count_int": 84}}]), encoding="utf-8")
    forward_dict = verdict_module._forward_dict("NDX_VXN", 140.0)
    assert forward_dict["status_str"] == "RED" and forward_dict["months_float"] == pytest.approx(4.0)


def test_write_verdicts_isolates_a_broken_bundle(tmp_path, monkeypatch):
    def fake_build(pod_str, ledger):
        if pod_str == "BAD":
            raise RuntimeError("bundle unreadable")
        row_list = [{"key_str": "own_grid", "question_str": "q", "status_str": "green", "headline_str": "MCPT p 0.001", "detail_str": ""}]
        return {"pod_str": pod_str, "family_id_str": "f", "row_list": row_list, "worst_status_str": "green",
                "count_dict": {"green": 1, "amber": 0, "red": 0}}

    monkeypatch.setattr(verdict_module, "build_verdict", fake_build)
    verdict_list = verdict_module.write_verdicts(["GOOD", "BAD"], Ledger(tmp_path / "l.jsonl"), tmp_path / "out")
    assert [bool(v.get("error_str")) for v in verdict_list] == [False, True]
    index_list = json.loads((tmp_path / "out" / "index.json").read_text(encoding="utf-8"))
    assert {e["pod_str"] for e in index_list} == {"GOOD", "BAD"}
    assert "MCPT p 0.001" in (tmp_path / "out" / "verdict.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not (verdict_module.REAUDITION_ROOT_PATH / "DV2" / "bundle.pkl").exists(), reason="saved DV2 bundle not present")
def test_build_verdict_on_the_saved_dv2_bundle():
    verdict_dict = verdict_module.build_verdict("DV2", Ledger())
    assert [r["key_str"] for r in verdict_dict["row_list"]] == ["mechanism", "own_grid", "lucky_pick", "alive", "factors",
                                                               "book", "costs", "forward"]
    assert all(r["status_str"] in ("green", "amber", "red") and r["headline_str"] for r in verdict_dict["row_list"])
