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
    "/": "200:16fd13f670478cf78ad8fca6e122d9ff1a37a982dda608d649e783dfe99a6185",
    "/overview/refresh": "200:cbcf72736537b0a5e0c4b8a63c917d153916f33b7e08bf6a84c7ca4ced4cd838",
    "/overview/money?period=All": "200:a569e5ded22c5fc239e31e7d7b311d86ecdcb07f71b51992e4bdb8474126fe56",
    "/system": "200:cf3bdda1baaa4db564fd68bdf746c9b656bd964cb42789ea2a5112c41176cd6d",
    "/activity/body?days=7": "200:eb2a1ba7ac081859b4de6f4a64340fdcbd4d1245a15b85130c8d7166b4b0107b",
    "/tools": "200:c0c77ebf0643e1c9d860edfa148e3f90cec55f97f5168af54cfb58adbfa54834",
    "/pods/demo_1_0": "200:0765ded31f5714b56d1b263bf933868b3540493ec81c642498bd84aebb092e61",
    "/pods/demo_1_0?tab=orders": "200:0fc155446f8537eb76cb31c3dc1846d69bfe0701a636f5b98f821157ad3b419e",
    "/pods/demo_1_0?tab=fills": "200:8f4dcaf5787200a951f25ec13ee9c82378c949ce5bd98ec38de33c9bfd8e1a41",
    "/pods/demo_1_0?tab=reconcile": "200:6576ec9c3af345ee3f225df9a4fa884696ddd6527cd7774f32d1f3714befec48",
    "/pods/demo_1_0?tab=events": "200:150bb2c0421edf0c8a32e5c9613bea24afb68fddee9d4603aeea6e3acfcdab7b",
    "/pods/demo_1_1": "200:0324a9a0552099b4b3528daa87f0001e5b70e10b5ae381fe7132d1dfdc7651a2",
    "/pods/demo_1_1?tab=orders": "200:019534866719e2c31c5d296eb6ed25d32944bea54703789b0aa086b752a97ea1",
    "/pods/demo_1_1?tab=fills": "200:cac7eb375937c09d347f875892df305fa442bf860d24eed7f5365ef6588f5343",
    "/pods/demo_1_1?tab=reconcile": "200:8a2b483b2174794d43482966c66a726f2cecef1f5b092623a21ecbf49e0b84e3",
    "/pods/demo_1_1?tab=events": "200:fb456b6141016fbfb48f6058ab9d44d553e5f2331a31cfee2129935b43fdb0bd",
    "/pods/demo_1_2": "200:a1a2eab5fe6181cbe3f24c8aea7f310ffb45c92db43a90e9da7f00c6d43a32c3",
    "/pods/demo_1_2?tab=orders": "200:f2b18b0f769ed40161fd634dc423d304bfa3ffbe5d1b84830df1d36454afc202",
    "/pods/demo_1_2?tab=fills": "200:e974858318385cd1e231d9ce1ef5da2a4271cefdedc79e642245a8f9f0a8d902",
    "/pods/demo_1_2?tab=reconcile": "200:81fa1cbc79aee899a1d80bf6b14abce313d5b67abcea3ef5ac500d2a2efbd0cd",
    "/pods/demo_1_2?tab=events": "200:dc3bc439a555e0cc449ddfbad0b44c406e42438265ea12c10da10bc626c3c35c",
    "/pods/demo_1_3": "200:3b9cf2c3726336237204b635abf05bfe85f46b9e20aa6f9df565429abaf5988d",
    "/pods/demo_1_3?tab=orders": "200:25f140c4d178565c42557520b1811eff93caa9a096cdaeb2d95cbfca5d4512e5",
    "/pods/demo_1_3?tab=fills": "200:040ae3267fcab87e7faeddf98b7358379cb7329ad72e5e323237f79ef2bfe423",
    "/pods/demo_1_3?tab=reconcile": "200:a77b80941ff1e7eb7f0131d038e19f88b0d0cfa9eb379eca1eb6971ddf6d8367",
    "/pods/demo_1_3?tab=events": "200:71ca3ac33e9cb578f9b88f649d97beadbc454136bf9b18b339d7848272cf0ddf"
}
GOLDEN_PRODUCTION_DICT = {
    "/": "200:a9fe8f4b32e63c10bc08318f3f046ced66e7b9755bfeb326d4fa75cf90370365",
    "/overview/refresh": "200:b55753b7ca8e037400e994db51748e1c6190154078b1055d2097452b8e5094e0",
    "/overview/money?period=All": "200:846197d7845456bdc19251a0059ef7eaebed36d7352723c7da1371cc325954f9",
    "/system": "200:4d641305b3e0d75ac46199ae814bd799f89aae1e6a6aef1ac9a926e0bfb935ed",
    "/activity/body?days=7": "200:d4155e7376ef8563c07a60613a0edbb8ca4811d806413b19fe1bd3c08b903428",
    "/tools": "200:7f2b4056ecce5f29ce61fd8490f9133c5571735ad87ca1859d4787c44278ba5b",
    "/pods/pod": "200:8533e91878189ce7e78f8f0903d8b7dd321a6000465f009c372b274faa87323d",
    "/pods/pod?tab=orders": "200:033843a9cd2aac8f2335e5860f92606cfe7b0a6843bec783c9d4568b8fde1cb5",
    "/pods/pod?tab=fills": "200:8f758a5a7f92d0f7b367a1f249564bf636bcd9ec636297769f9c468cc34e15db",
    "/pods/pod?tab=reconcile": "200:39e100c52e21d6f8cf0aab7dbbbfdeb522a2d64582de53941f4037b800e05e51",
    "/pods/pod?tab=events": "200:7c90ec465f2f3a367fbac729e3204617832d35dfcb62b08b0e6580410e219ceb"
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
