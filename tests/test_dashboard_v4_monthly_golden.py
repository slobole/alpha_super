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
    "/": "200:7478b3d4f1caae9327c6530a6342307ab39bae9313807f3c45326b68c1b31f34",
    "/overview/refresh": "200:4507bf6779bc91b6d01f4fe63faecde705e1356cfef34cd48487c6a32855c32e",
    "/overview/money?period=All": "200:a569e5ded22c5fc239e31e7d7b311d86ecdcb07f71b51992e4bdb8474126fe56",
    "/system": "200:0b9286584d36e279b63cc9bf041d8bf4f5bb54901591c396d00c2f90a4322f0b",
    "/activity/body?days=7": "200:d3da9835352a876927eb92ad6b7013093b93ceafc01e8dff2e46d9eeca9368ee",
    "/tools": "200:01e9f2141c5824934134620079c51014de233540161aac9e403e0ff2a48b261d",
    "/pods/demo_1_0": "200:89098a273c3ffbb677661fc91bb83c3c831703e91a62a5567db655fc0f193d1b",
    "/pods/demo_1_0?tab=orders": "200:585c63f0e28a2e1da25f1f104d9bbfde7956cf144d4a2f1a1b3b81dbc497c959",
    "/pods/demo_1_0?tab=fills": "200:209659e2c29bac0110794f33869654ba55a1750c8e12e72bf7bcbd4d3e55a660",
    "/pods/demo_1_0?tab=reconcile": "200:c7b38a25739c91daa816bfddfffca28c425403ce58e3266a7c3ff42bf60ef139",
    "/pods/demo_1_0?tab=events": "200:a2b164aec79bdc5f10af379e990f733686f1b7c17da96f07b5e20ba881829030",
    "/pods/demo_1_1": "200:cda7cff75a9fa6408192668896347812e47bfc8e8da4fa7c239428bbd158af35",
    "/pods/demo_1_1?tab=orders": "200:bbf6458b10e2278f79485f2ef85d83c2efabae60597b3d313524d65f29238739",
    "/pods/demo_1_1?tab=fills": "200:c2361ff8f657fd4cec5facf0eeb00f15cf9d74de45190a793687b80e6ceef683",
    "/pods/demo_1_1?tab=reconcile": "200:67cde0c6463a23dbd4dd43926f445195cd1cd6374b2339d42b20d3e7c734ef9b",
    "/pods/demo_1_1?tab=events": "200:97fcaae9afd11833bf7ee11815fc16e84928753259e8458fd0ff2685b01d0c95",
    "/pods/demo_1_2": "200:c83814bb201a4ec93934c79c5f174ed0bb093fc868a335624cda600e610ef171",
    "/pods/demo_1_2?tab=orders": "200:023b78bdc2041651ddb125b8c9a440869424e8838fac869d4425e7d2313a8dae",
    "/pods/demo_1_2?tab=fills": "200:fea486f74ad53e1f93c8182bd0de35a94c483f753f2f9be20d06966a005f942f",
    "/pods/demo_1_2?tab=reconcile": "200:e8ba894c64df150cdbfd762d6850c5f136adc2debf4b039ab7b9a174637b83ea",
    "/pods/demo_1_2?tab=events": "200:812e5725edfa539c6132900facce74d8a38ad65322acca17bb05508628e7da6c",
    "/pods/demo_1_3": "200:8d57f84dc04bfbcbb56edc68120fd1485b3bd5e0603331397f87d7667db0b508",
    "/pods/demo_1_3?tab=orders": "200:1ad8251477394069070977b1f05e1de707007035b5a3e7dc489be66976eb4a9d",
    "/pods/demo_1_3?tab=fills": "200:7650e1ea0297108bc15e526e4385cb4c557c76d4f90c15a29c988faa02f85816",
    "/pods/demo_1_3?tab=reconcile": "200:2b8bb06456a87a3dc2e9ec4279ef67b3be20e06e497a7a88139c6d35f9cc3269",
    "/pods/demo_1_3?tab=events": "200:79347cb52e03caac8a89314c2fc4ff72331001c319fe3546db67295bdbca2301"
}
GOLDEN_PRODUCTION_DICT = {
    "/": "200:9fe84c89a0d008d472d3d5f80103d135874b23ea56883ab727cb7119b15cbeea",
    "/overview/refresh": "200:34cadf0f307147d42f327e6a883c6d6bdfd62c58b328584929ce8064cf34bdd1",
    "/overview/money?period=All": "200:846197d7845456bdc19251a0059ef7eaebed36d7352723c7da1371cc325954f9",
    "/system": "200:afd5e21d55f3c938a1dbeac9629bc43457dce2b6de106ad4a09f27daf24c7259",
    "/activity/body?days=7": "200:d4155e7376ef8563c07a60613a0edbb8ca4811d806413b19fe1bd3c08b903428",
    "/tools": "200:b7d8132a06b0350dd112bd2fae9d3214360ec5ad4f04f3d21453529fd3b604c5",
    "/pods/pod": "200:ffd669964109aa1cacbbd778763b8bcc11b4f2adf0598adf21f4dd0ecf8d99c4",
    "/pods/pod?tab=orders": "200:bcfadc3b54443b1b8c3a6851029f2d467a974a0257f4582603b3424891f28a7a",
    "/pods/pod?tab=fills": "200:c9983cd648747d01ed73c0d89bb8fd52eef0d998acd1e51f0dfcae92a1b35363",
    "/pods/pod?tab=reconcile": "200:42f2262efd2c0ae76e0f1a99db4e062210e31b0f4487c0e73e465941b0234502",
    "/pods/pod?tab=events": "200:b9ef5d5f2192e9c7047e8e44216d6ecf8e9237b1e52f65ec2843ff914f6d34aa"
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
