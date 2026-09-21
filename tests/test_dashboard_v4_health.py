"""V4 exposes the real health cause without changing V3 thresholds or freshness."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.overview import build_overview_dict


@pytest.fixture(scope="module")
def healthy_workspace_tuple():
    workspace_dict, _, provider_obj = build_demo_workspace_tuple()
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:1]
    try:
        yield workspace_dict, provider_obj
    finally:
        provider_obj.close()


def _view_dict(healthy_workspace_tuple, *, age_int=0):
    workspace_dict, provider_obj = healthy_workspace_tuple
    return build_overview_dict(deepcopy(workspace_dict), None, provider_obj,
        as_of_ts=DEMO_NOW_TS + timedelta(seconds=age_int), include_finance_bool=False)


@pytest.mark.parametrize("used_int,state_str,value_str", [
    (749, "done", ""), (750, "late", "75% used"), (760, "late", "76% used"),
    (899, "late", "90% used"), (900, "fail", "90% used"), (970, "fail", "97% used"),
])
def test_exact_existing_disk_thresholds_have_a_visible_safe_cause(healthy_workspace_tuple, monkeypatch,
        used_int, state_str, value_str):
    call_list = []
    def disk_usage(path_str):
        call_list.append(path_str)
        return SimpleNamespace(total=1000, used=used_int, free=1000 - used_int)
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage", disk_usage)
    view_dict = _view_dict(healthy_workspace_tuple)
    assert call_list == ["."]  # Real rollup/probe boundary is still called.
    assert view_dict["system_dict"]["state_str"] == state_str
    assert view_dict["source_fresh_bool"] is True and view_dict["source_valid_ms_int"] == 120000
    assert view_dict["attention_list"] == []  # Host disk is not a fabricated Pod failure.
    if state_str == "done":
        assert view_dict["system_dict"]["detail_str"].startswith("Data ")
        assert view_dict["verdict_str"] == "No action needed."
    else:
        assert view_dict["system_dict"]["detail_str"] == "Disk " + value_str
        assert view_dict["verdict_detail_str"] == "Disk " + value_str + "."
        assert view_dict["verdict_str"] == ("System needs action." if state_str == "fail" else "Status needs review.")
        assert "Some saved evidence is unavailable" not in view_dict["verdict_detail_str"]
        assert "GB free at" not in view_dict["system_dict"]["detail_str"]


@pytest.mark.parametrize("age_int", [121, -1])
def test_disk_warning_cannot_renew_stale_or_future_operations(healthy_workspace_tuple, monkeypatch, age_int):
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage",
        lambda path_str: SimpleNamespace(total=100, used=97, free=3))
    view_dict = _view_dict(healthy_workspace_tuple, age_int=age_int)
    assert view_dict["system_dict"]["state_str"] == "unk"
    assert view_dict["system_dict"]["label_str"] == "System unknown"
    assert "Disk" not in view_dict["system_dict"]["detail_str"]
    assert view_dict["verdict_str"] == "Status unknown."
    assert view_dict["source_valid_ms_int"] == 0


def test_disk_probe_failure_is_unknown_with_sanitized_cause(healthy_workspace_tuple, monkeypatch):
    def unavailable(path_str):
        raise OSError("private path or account")
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage", unavailable)
    view_dict = _view_dict(healthy_workspace_tuple)
    assert view_dict["system_dict"] == {"state_str": "unk", "label_str": "System unknown", "detail_str": "Disk usage unavailable"}
    assert view_dict["verdict_detail_str"] == "Disk usage unavailable."
    assert "private" not in str(view_dict)


def test_disk_warning_keeps_higher_severity_scheduler_cause(healthy_workspace_tuple, monkeypatch):
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage",
        lambda path_str: SimpleNamespace(total=100, used=76, free=24))
    monkeypatch.setattr(healthy_workspace_tuple[1], "get_scheduler_status_dict",
        lambda *args, **kwargs: {"state_str": "error", "alive_bool": False})
    view_dict = _view_dict(healthy_workspace_tuple)
    assert view_dict["system_dict"]["state_str"] == "fail"
    assert view_dict["system_dict"]["detail_str"] == "Scheduler · Disk 76% used"
    assert view_dict["attention_list"][0]["title_str"] == "Scheduler error"
    assert view_dict["verdict_str"] == "1 pod needs action."


@pytest.mark.parametrize("cell_label_str", ["Norgate", "Pod state", "EOD Snapshot"])
@pytest.mark.parametrize("disk_unavailable_bool", [False, True])
def test_worse_saved_health_cause_is_not_hidden_by_disk(healthy_workspace_tuple, monkeypatch,
        cell_label_str, disk_unavailable_bool):
    workspace_dict, provider_obj = healthy_workspace_tuple
    workspace_dict = deepcopy(workspace_dict)
    for row_dict in workspace_dict["summary_dict"]["pod_row_dict_list"]:
        for item_dict in row_dict["data_freshness_dict"]["item_dict_list"]:
            if item_dict["label_str"] == cell_label_str:
                item_dict.update(severity_str="red", value_str="private source value",
                    detail_str="private source path")
    def disk_usage(path_str):
        if disk_unavailable_bool:
            raise OSError("private disk path")
        return SimpleNamespace(total=100, used=76, free=24)
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage", disk_usage)
    view_dict = build_overview_dict(workspace_dict, None, provider_obj,
        as_of_ts=DEMO_NOW_TS, include_finance_bool=False)
    disk_cause_str = "Disk usage unavailable" if disk_unavailable_bool else "Disk 76% used"
    expected_cause_str = cell_label_str + " needs action · " + disk_cause_str
    assert view_dict["system_dict"] == {
        "state_str": "fail", "label_str": "System needs action", "detail_str": expected_cause_str,
    }
    assert view_dict["verdict_str"] == "System needs action."
    assert view_dict["verdict_detail_str"] == expected_cause_str + "."
    assert "private" not in view_dict["system_dict"]["detail_str"]
    assert view_dict["source_fresh_bool"] is True and view_dict["source_valid_ms_int"] == 120000


def test_no_verified_pods_is_not_promoted_by_a_live_disk_probe(healthy_workspace_tuple, monkeypatch):
    workspace_dict, provider_obj = healthy_workspace_tuple
    workspace_dict = deepcopy(workspace_dict)
    workspace_dict["operations_account_list"] = []
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage",
        lambda path_str: SimpleNamespace(total=100, used=97, free=3))
    view_dict = build_overview_dict(workspace_dict, None, provider_obj, as_of_ts=DEMO_NOW_TS, include_finance_bool=False)
    assert view_dict["system_dict"]["state_str"] == "unk"
    assert view_dict["verdict_str"] == "No LIVE pods verified."
    assert "Disk" not in view_dict["system_dict"]["detail_str"]
