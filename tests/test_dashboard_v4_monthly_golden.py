"""Guard rail: monthly and demo pages render byte-identically while daily pods are added.

The hashes were generated at dee5499, before any daily-pod change. Every later
phase must keep them unchanged. A mismatch names the changed URL; compare that
page with its dee5499 rendering before considering any hash update.
"""

from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.data import LiveDataProvider
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.pod_finance import build_pod_finance_dict
from test_dashboard_v4_evidence import NOW_TS, build_fixture_tuple, update_db
from test_dashboard_v4_pod_data import _reconcile


PAGE_SUFFIX_TUPLE = ("", "?tab=orders", "?tab=fills", "?tab=reconcile", "?tab=events")
SHARED_URL_TUPLE = ("/", "/overview/refresh", "/overview/money?period=All",
                    "/system", "/activity/body?days=7", "/tools")
STORAGE_KEY_RE = re.compile(r"alpha\.ops\.v4\.activity\.live\.[0-9a-f]{64}")
GOLDEN_DEMO_DICT = {
    "/": "200:a5f6016ae823cc80b12daacf1998587deda0c7f28a83a112d98adf8041865d20",
    "/overview/refresh": "200:4e1bf0f64d3ca02866fcf6cf0ffd057b7d0d3fe5ae54d1b058867a1d1b266068",
    "/overview/money?period=All": "200:a569e5ded22c5fc239e31e7d7b311d86ecdcb07f71b51992e4bdb8474126fe56",
    "/system": "200:e3a588dc64a8a472c055f03a084e640d92b565bcd5dfaf965d543af7730246a9",
    "/activity/body?days=7": "200:eb2a1ba7ac081859b4de6f4a64340fdcbd4d1245a15b85130c8d7166b4b0107b",
    "/tools": "200:23fadbbcb7a4bb247ccbb7f3557a8bcd8a1c3285ef636cde0c58f915e97557df",
    "/pods/demo_1_0": "200:1917a29425245c61ce09a260b50d5e7ac39e46e37db2c4793fc6d43e5488c4e8",
    "/pods/demo_1_0?tab=orders": "200:78e39a90702c865c6f062300c95a0180f8978fa5c12abb602a285bd2e9185da0",
    "/pods/demo_1_0?tab=fills": "200:3571d2622296d8a21a8e3064a05e9877f3b209ee280273045795b628be55734d",
    "/pods/demo_1_0?tab=reconcile": "200:3055801c34f650008ac8100584592621653c2c16d225644cadf5530c5d41316a",
    "/pods/demo_1_0?tab=events": "200:7b379908a0278e565b9fe5fff23bb7026509c68db3e08b6c8009b2c114754a95",
    "/pods/demo_1_1": "200:62a33a97fedf656ea75471af9818f65c081d00b32254e04b3ec33c9695a2785a",
    "/pods/demo_1_1?tab=orders": "200:638f66e1ffcc7d3152a605839c0044aeaa01a621f416590180f4d7b7d1e6f399",
    "/pods/demo_1_1?tab=fills": "200:23725cd1fa769b70aa394ce89ac39de3ddf5746b571c11cd9cf4c28cf114b5a8",
    "/pods/demo_1_1?tab=reconcile": "200:b17b7b424d98fea2a33b9d0274f3a2d1abe03d31456fe817e91e8b821d3f2096",
    "/pods/demo_1_1?tab=events": "200:e1ebcd147a0e09b8ed1b49b48260ca267745393f812e19e80c99cb2dce003223",
    "/pods/demo_1_2": "200:2e5b7c182a2895a9b47d3e5b7496c0f6261557dcb77416a4ddf6908babf307bc",
    "/pods/demo_1_2?tab=orders": "200:bb2eca525cfe714ea5ecb7da11dfd368cb72330a9f898c6fdf84ce2cbf303282",
    "/pods/demo_1_2?tab=fills": "200:99c442c5b0a0de3d989fbd66470264d1d5686f9f6336409c10c0b2aeb3edc4f2",
    "/pods/demo_1_2?tab=reconcile": "200:88e5cdeec4f8bd9be8c91d289f63c4537e2caad3e1ad6a944b0b159c17dec72a",
    "/pods/demo_1_2?tab=events": "200:4f5e3ade7a7b38b346430e690b5cfba1ad4bb3899b7fc5875b8bfce32f1feaa7",
    "/pods/demo_1_3": "200:f05d4d55e2ff47601e07e03e8ed48deafbdd5460adf8b5df6894df367266e227",
    "/pods/demo_1_3?tab=orders": "200:6a1088b22099eabee5d0df16124b5bd0c305ce193f92f31a003655949f4307a6",
    "/pods/demo_1_3?tab=fills": "200:4333d77abe3d6fdadf9cf2285e685e4747f5ee4a810c03c69d95659072f67c5a",
    "/pods/demo_1_3?tab=reconcile": "200:f9aa97d76edd19b1552cb77585e9cbbe63a513ff8e2e874926eedbce54d50a25",
    "/pods/demo_1_3?tab=events": "200:13f3076184a3aaa245b700ef5547200b3ca22ea4899ffde1e142b36650df34c9"
}
GOLDEN_PRODUCTION_DICT = {
    "/": "200:70926b51e5d60e2996ed77fbab7ab2c923f1f07e2884439ae6361b7202a1174b",
    "/overview/refresh": "200:812ed30c75ae2c36c36ae4ed281a308c771e4d34f7ac14abdc1010d5759bb710",
    "/overview/money?period=All": "200:846197d7845456bdc19251a0059ef7eaebed36d7352723c7da1371cc325954f9",
    "/system": "200:1f5117716a182356104501c05b18c2a8565ce3f9061593aa3f8a657dd54c4c39",
    "/activity/body?days=7": "200:d4155e7376ef8563c07a60613a0edbb8ca4811d806413b19fe1bd3c08b903428",
    "/tools": "200:ba8ebb233324456b21616bf485e2bead5eb94f24223eac6da02a3e393b21c2a6",
    "/pods/pod": "200:5c4f7bc02f2369bff0bd3ef951c49f5d4eafaf1b2df75246e4fd541471cae194",
    "/pods/pod?tab=orders": "200:7e341a7b0e23cf41d2c338c0b1b4c8b3f203ad80c0ef3b1b6a21cad5a08370c2",
    "/pods/pod?tab=fills": "200:5f527bbc70f94b29ab5b26af7a782324fab8df0e53f62c5dc3e738535c871832",
    "/pods/pod?tab=reconcile": "200:6cc8522e7e0df6d45087a24a8118ea3f93b18cf252bae6e4d4292b18170354a6",
    "/pods/pod?tab=events": "200:b28b0ff37155ab9672d32400f5e04387584267dcd704c524049f4c4abc4e509f"
}


@pytest.fixture(autouse=True)
def healthy_host_disk(monkeypatch):
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage",
        lambda path_str: SimpleNamespace(total=100, used=50, free=50))


def _url_list(pod_id_list):
    return list(SHARED_URL_TUPLE) + [f"/pods/{pod_id_str}{suffix_str}"
        for pod_id_str in pod_id_list for suffix_str in PAGE_SUFFIX_TUPLE]


def _hash_dict(app_obj, url_list, *, replace_list=()):
    """Status plus SHA-256 of the rendered HTML, with local paths normalized out."""
    client_obj = app_obj.test_client()
    result_dict = {}
    for url_str in url_list:
        response_obj = client_obj.get(url_str)
        html_str = response_obj.get_data(as_text=True)
        for old_str, new_str in replace_list:
            html_str = html_str.replace(old_str, new_str)
        # The browser-local Activity key hashes the event-log location.
        html_str = STORAGE_KEY_RE.sub("alpha.ops.v4.activity.live.<key>", html_str)
        result_dict[url_str] = f"{response_obj.status_code}:" + hashlib.sha256(html_str.encode("utf-8")).hexdigest()
    return result_dict


def demo_hash_dict():
    """The demo wiring of demo.py with a fixed clock instead of a moving one."""
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    try:
        app_obj = create_app(provider_obj, demo_bool=True,
            workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj),
            operations_workspace_fn=lambda: deepcopy(workspace_dict), now_fn=lambda: DEMO_NOW_TS)
        pod_id_list = [account_dict["pod_id"] for account_dict in workspace_dict["operations_account_list"]]
        # The owned demo directory has a random name; it must never reach HTML.
        directory_str = str(provider_obj.pod_store_obj.directory_path_obj)
        return _hash_dict(app_obj, _url_list(pod_id_list), replace_list=((directory_str, "<demo>"),))
    finally:
        provider_obj.close()


def production_hash_dict(tmp_path, monkeypatch):
    """Production-format DB and LiveDataProvider, as in the Pod integration test."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj)
    update_db(target_obj, "UPDATE decision_plan SET snapshot_metadata_json_str=?", (json.dumps({
        "norgate_data_profile_str": "test", "norgate_snapshot_date_str": "2026-09-17"}),))
    now_ts = NOW_TS + timedelta(hours=1)
    provider_obj = LiveDataProvider(releases_root_path_str="", event_log_path_str=str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(provider_obj, "get_target_for_pod", lambda pod_id_str: target_obj if pod_id_str == "pod" else None)
    monkeypatch.setattr(provider_obj, "get_target_list", lambda: [target_obj], raising=False)
    workspace_dict, snapshot_obj, demo_provider_obj = build_demo_workspace_tuple()
    try:
        finance_dict = build_pod_finance_dict(workspace_dict, snapshot_obj, demo_provider_obj,
            pod_id_str=workspace_dict["operations_account_list"][0]["pod_id"], as_of_ts=DEMO_NOW_TS)
    finally:
        demo_provider_obj.close()
    monkeypatch.setattr("alpha.live.dashboard_v4.app.build_pod_finance_dict", lambda *args, **kwargs: finance_dict)
    account_dict = deepcopy(workspace_dict["operations_account_list"][0])
    account_dict.update(pod_id="pod", account_route="U111")
    workspace_dict["operations_account_list"] = [account_dict]
    workspace_dict["valuation_account_list"] = [account_dict]
    workspace_dict["client_dict"] = {**workspace_dict["client_dict"], "accounts": [account_dict]}
    current_dict = provider_obj.get_pod_cycles_dict("pod", as_of_ts=now_ts)
    assert current_dict["status_str"] == "ok"
    workspace_dict["summary_dict"].update(as_of_timestamp_str=now_ts.isoformat(),
        pod_row_dict_list=[current_dict["pod_row_dict"]])
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj),
        operations_workspace_fn=lambda: deepcopy(workspace_dict),
        performance_db_path_str=str(tmp_path / "performance.sqlite3"), now_fn=lambda: now_ts)
    path_obj = Path(tmp_path)
    replace_list = [(str(path_obj.resolve()), "<tmp>"), (str(path_obj), "<tmp>"), (path_obj.as_posix(), "<tmp>")]
    return _hash_dict(app_obj, _url_list(["pod"]), replace_list=replace_list)


def test_demo_pages_are_deterministic_and_match_golden():
    first_dict = demo_hash_dict()
    assert first_dict == demo_hash_dict()
    assert first_dict == GOLDEN_DEMO_DICT


def test_production_format_pages_are_deterministic_and_match_golden(tmp_path, monkeypatch):
    first_dict = production_hash_dict(tmp_path / "first", monkeypatch)
    with monkeypatch.context() as second_monkeypatch:
        second_dict = production_hash_dict(tmp_path / "second", second_monkeypatch)
    assert first_dict == second_dict
    assert first_dict == GOLDEN_PRODUCTION_DICT
