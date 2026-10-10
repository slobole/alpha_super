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
    "/": "200:73c66b89c7db178af8576508d1dce0e2324a238627c0f837313e4ce732580ad6",
    "/overview/refresh": "200:57637ea7ca2372c50d7d6f5d61b846216878b174a6ffd1fdfb21009c0b04382f",
    "/overview/money?period=All": "200:a569e5ded22c5fc239e31e7d7b311d86ecdcb07f71b51992e4bdb8474126fe56",
    "/system": "200:e3a588dc64a8a472c055f03a084e640d92b565bcd5dfaf965d543af7730246a9",
    "/activity/body?days=7": "200:5e5e5a95bf3feaf2e3eaa7c1e30102960ecc5fc18b5ae371ae76b45a2d49793c",
    "/tools": "200:23fadbbcb7a4bb247ccbb7f3557a8bcd8a1c3285ef636cde0c58f915e97557df",
    "/pods/demo_1_0": "200:5640a8e84bbbfc773f563c1e9a2cb09172981a36698d27fb1360ee5bb368b6e9",
    "/pods/demo_1_0?tab=orders": "200:b1b6e4456e2b774455f6b8b31d8f81c1575f5bae171a549c2affa1d44ec4bd57",
    "/pods/demo_1_0?tab=fills": "200:bf1c8afd5867660ed3bc12d319a3a75f6b798ef93badcbaf17178cd643d538f0",
    "/pods/demo_1_0?tab=reconcile": "200:247bc0532752970e83ee13fc6ac0b512dadad7a4148ac7b5e8567ed75ff360cd",
    "/pods/demo_1_0?tab=events": "200:19e62e14b27c7a26a1f8defc17ec73e90ad914763c2850d2b879327698d96499",
    "/pods/demo_1_1": "200:bf349a008ed4586f7f6edfd2941853032ec1c3a02c51d798d7b1ccaa8e05fd3e",
    "/pods/demo_1_1?tab=orders": "200:8a029f557087516cb75608efdef98ed96c097a0141a80a2b2831a903f19b2e2d",
    "/pods/demo_1_1?tab=fills": "200:a793b77d2b72b7afbdfb52cc14aab7f5d549b4809945f4914cfa07484966f021",
    "/pods/demo_1_1?tab=reconcile": "200:2a303503be3283f7ee9d6929bb9aa503da8da1b1200114289ce11ef76280e37f",
    "/pods/demo_1_1?tab=events": "200:e0401edac7c5b2a32754d22d8e97e36d6ef4b92dd1952a8f2be3329755dacc87",
    "/pods/demo_1_2": "200:95670b0226dae7149612e1819a119a1c1266c8a276d9895da1708ff66cbf8616",
    "/pods/demo_1_2?tab=orders": "200:f2714a7c5bbcbc7ecc2e108d64676a2e11775c530bff1f11e9ee5a64ce6ad0f6",
    "/pods/demo_1_2?tab=fills": "200:71128a06bcd52d454ef42b374b330c1ccc815290123025a3e12036cfb868b546",
    "/pods/demo_1_2?tab=reconcile": "200:33f95f0f1d95fe5e4b5394ec82791f15aba1aec00a8580345ef99179b683cda5",
    "/pods/demo_1_2?tab=events": "200:2326bbb71b2589af8888f3d443448ad1b62bb6da40c0b883d2cb21cb4250a6d8",
    "/pods/demo_1_3": "200:aaa091d0153f018fa2dd91a7f3a4d5de0dc2998415284fc8dc387a02e0a791bf",
    "/pods/demo_1_3?tab=orders": "200:cc96a79ddd1f1448b698ea9197b4455b680f1be8d9ac68280c87ba0e31dc2d30",
    "/pods/demo_1_3?tab=fills": "200:a59989e475b1941b4dbcfb0d73f7314fa382517701ab511cb630b4bec1741045",
    "/pods/demo_1_3?tab=reconcile": "200:e8d4c9837d7702abed38e26f67d45d8b942f1aabc3a84030e617a4ff7a988625",
    "/pods/demo_1_3?tab=events": "200:73fc902da687a7312fe55f3766dd565823e1ab4519761d95ae0a2a73f717e911"
}
GOLDEN_PRODUCTION_DICT = {
    "/": "200:70926b51e5d60e2996ed77fbab7ab2c923f1f07e2884439ae6361b7202a1174b",
    "/overview/refresh": "200:812ed30c75ae2c36c36ae4ed281a308c771e4d34f7ac14abdc1010d5759bb710",
    "/overview/money?period=All": "200:846197d7845456bdc19251a0059ef7eaebed36d7352723c7da1371cc325954f9",
    "/system": "200:1f5117716a182356104501c05b18c2a8565ce3f9061593aa3f8a657dd54c4c39",
    "/activity/body?days=7": "200:d4155e7376ef8563c07a60613a0edbb8ca4811d806413b19fe1bd3c08b903428",
    "/tools": "200:ba8ebb233324456b21616bf485e2bead5eb94f24223eac6da02a3e393b21c2a6",
    "/pods/pod": "200:75aafb39ffb4ced87fc1da12fcccd4f4343df03c76af120087ff0241ba63e24e",
    "/pods/pod?tab=orders": "200:e2579452b3f82f5126190081931367acbbd4fb09e18f7ac44d48f95a8ef2e660",
    "/pods/pod?tab=fills": "200:920a3b1107e92ae33e9c58319560fd0dbbe69155a28665ba0752105938b45b0e",
    "/pods/pod?tab=reconcile": "200:7e6aaa11e02bf93020b3538b6ace2a6633528a3ee3b5d5538f33403936b64979",
    "/pods/pod?tab=events": "200:7df71a02d09a376dac183277163a2d3395bd59cc9e3e698878c0628c3ac10e57"
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
