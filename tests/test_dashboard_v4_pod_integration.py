"""Production-format Pod GETs retain historical scope and never write state."""

from copy import deepcopy
from datetime import timedelta
import json

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.data import LiveDataProvider
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.pod_finance import build_pod_finance_dict
from test_dashboard_v4_evidence import NOW_TS, build_fixture_tuple, update_db
from test_dashboard_v4_pod_data import _decision_only_int, _reconcile, _upgrade_target_obj


def test_historical_release_provider_and_all_page_reads_preserve_database(tmp_path, monkeypatch):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj)
    update_db(target_obj, "UPDATE decision_plan SET snapshot_metadata_json_str=?", (json.dumps({
        "norgate_data_profile_str": "test", "norgate_snapshot_date_str": "2026-09-17"}),))
    target_obj = _upgrade_target_obj(target_obj)
    _decision_only_int(target_obj, release_id_str="release-v2", status_str="planned")
    now_ts = NOW_TS + timedelta(days=1)
    provider_obj = LiveDataProvider()
    monkeypatch.setattr(provider_obj, "get_target_for_pod", lambda pod_id_str: target_obj)
    db_path_obj = tmp_path / "pod.sqlite3"
    before_bytes, before_mtime_int = db_path_obj.read_bytes(), db_path_obj.stat().st_mtime_ns
    source_dict = provider_obj.get_pod_cycles_dict("pod", as_of_ts=now_ts, vplan_id_int=1)
    assert source_dict["cycle_evidence_dict"]["state_str"] == "complete"
    assert source_dict["pod_row_dict"]["release_id_str"] == "release"
    workspace_dict, snapshot_obj, demo_provider_obj = build_demo_workspace_tuple()
    finance_dict = build_pod_finance_dict(workspace_dict, snapshot_obj, demo_provider_obj,
        pod_id_str=workspace_dict["operations_account_list"][0]["pod_id"], as_of_ts=DEMO_NOW_TS)
    monkeypatch.setattr("alpha.live.dashboard_v4.app.build_pod_finance_dict", lambda *args, **kwargs: finance_dict)
    account_dict = deepcopy(workspace_dict["operations_account_list"][0])
    account_dict.update(pod_id="pod", account_route="U111")
    workspace_dict["operations_account_list"] = [account_dict]
    current_dict = provider_obj.get_pod_cycles_dict("pod", as_of_ts=now_ts)
    workspace_dict["summary_dict"].update(as_of_timestamp_str=now_ts.isoformat(),
        pod_row_dict_list=[current_dict["pod_row_dict"]])
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: now_ts)
    for tab_str in ("plan", "decision", "orders", "fills", "reconcile", "events", "files"):
        for suffix_str in ("", "/refresh"):
            response_obj = app_obj.test_client().get(f"/pods/pod{suffix_str}?cycle=vplan:1&tab={tab_str}")
            html_str = response_obj.get_data(as_text=True)
            assert response_obj.status_code == 200
            assert "Saved cycle" in html_str and "2 of 2 filled" in html_str
            assert "U111" not in html_str
    assert db_path_obj.read_bytes() == before_bytes
    assert db_path_obj.stat().st_mtime_ns == before_mtime_int


def test_default_keeps_completed_plan_with_missing_ack(monkeypatch):
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][1]
    row_dict.update(latest_vplan_status_str="completed", latest_vplan_is_for_latest_decision_bool=False,
        latest_decision_plan_id_int=3, latest_vplan_id_int=2, latest_vplan_decision_plan_id_int=2)
    call_list = []
    original_fn = provider_obj.get_pod_cycles_dict
    def capture_source(pod_id_str, **options_dict):
        call_list.append(options_dict)
        return original_fn(pod_id_str, **options_dict)
    monkeypatch.setattr(provider_obj, "get_pod_cycles_dict", capture_source)
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    response_obj = app_obj.test_client().get('/pods/' + row_dict["pod_id_str"])
    assert response_obj.status_code == 200
    assert call_list[0]["vplan_id_int"] == 2
    assert "decision_plan_id_int" not in call_list[0]
    assert "Review broker ACK" in response_obj.get_data(as_text=True)
