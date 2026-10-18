"""Reviewed marks for backtest differences: saved per decision, hidden on Overview, protected POST."""

from datetime import datetime, timezone
import json
import re
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4 import backtest_check_data as btd
from alpha.live.dashboard_v4.backtest_review import BacktestReviewStore, REVIEW_LIMIT_INT

NOW = datetime(2026, 12, 10, 23, 40, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def healthy_host_disk(monkeypatch):
    from alpha.live.dashboard_v3 import health
    monkeypatch.setattr(health.shutil, "disk_usage", lambda path: SimpleNamespace(total=100, used=50, free=50))
    btd._CACHE_DICT.clear()


def test_marks_are_saved_next_to_the_report_once_per_decision(tmp_path):
    store_obj = BacktestReviewStore(SimpleNamespace(event_log_path_str=str(tmp_path / "events.jsonl")))
    store_obj.mark("pod_a|r1|7|2026-12-10", pod_id_str="pod_a", now_ts=NOW)
    store_obj.mark("pod_a|r1|7|2026-12-10", pod_id_str="pod_a", now_ts=NOW)
    payload_dict = json.loads((tmp_path / "backtest_reviewed.json").read_text(encoding="utf-8"))
    assert [entry_dict["key_str"] for entry_dict in payload_dict["reviewed_list"]] == ["pod_a|r1|7|2026-12-10"]
    assert store_obj.reviewed_key_set() == {"pod_a|r1|7|2026-12-10"}
    for index_int in range(REVIEW_LIMIT_INT + 5):
        store_obj.mark(f"pod_a|r1|{index_int}|x", pod_id_str="pod_a", now_ts=NOW)
    assert len(store_obj.reviewed_key_set()) == REVIEW_LIMIT_INT


def test_unreadable_file_means_nothing_reviewed(tmp_path):
    (tmp_path / "backtest_reviewed.json").write_text("{not json", encoding="utf-8")
    assert BacktestReviewStore(SimpleNamespace(event_log_path_str=str(tmp_path / "events.jsonl"))).reviewed_key_set() == set()


def test_a_new_decision_gets_a_new_key():
    def report_dict(decision_id_int):
        pod_dict = {"pod_id_str": "pod_a", "release_id_str": "r1", "decision_plan_id_int": decision_id_int, "session_str": "2026-12-10",
                    "decision_dict": {"status_str": "differs", "session_str": "2026-12-10",
                                      "row_list": [{"asset_str": "XLE", "live_str": "BUY", "backtest_str": "—", "same_bool": False}]}}
        return {"stale_bool": False, "generated_ts": NOW, "pod_list": [pod_dict]}
    first_row, second_row = (btd.attention_list(report_dict(id_int), {"pod_a": "A"})[0] for id_int in (7, 8))
    assert first_row["review_key_str"] == "pod_a|r1|7|2026-12-10"
    assert first_row["review_key_str"] != second_row["review_key_str"]


def _demo_client():
    from alpha.live.dashboard_v4.demo import create_demo_app
    return create_demo_app(demo_backtest_bool=True).test_client()


def _button_dict(html_str):
    match_obj = re.search(r'data-review-url="([^"]+)" data-review-key="([^"]+)" data-token="([^"]+)"', html_str)
    assert match_obj, "Reviewed button missing"
    return {"url_str": match_obj.group(1), "key_str": match_obj.group(2).replace("&#34;", '"'), "token_str": match_obj.group(3)}


def _post(client_obj, button_dict, *, token_str=None, key_str=None):
    return client_obj.post(button_dict["url_str"], json={"confirmed_bool": True, "key_str": key_str or button_dict["key_str"]},
        headers={"Origin": "http://localhost", "X-Alpha-Action-Token": token_str or button_dict["token_str"]})


def test_reviewed_hides_the_difference_on_overview():
    client_obj = _demo_client()
    html_str = client_obj.get("/").get_data(as_text=True)
    assert "Decision differs from the backtest." in html_str
    button_dict = _button_dict(html_str)
    response_obj = _post(client_obj, button_dict)
    assert response_obj.status_code == 200 and response_obj.get_json() == {"status_str": "ok"}
    after_str = client_obj.get("/").get_data(as_text=True)
    assert "Decision differs from the backtest." not in after_str
    assert "data-review-url" not in after_str


def test_reviewed_requires_the_token_and_a_current_difference():
    client_obj = _demo_client()
    button_dict = _button_dict(client_obj.get("/").get_data(as_text=True))
    assert _post(client_obj, button_dict, token_str="wrong").status_code == 403
    assert _post(client_obj, button_dict, key_str="demo_1_0|other|99|2026-01-01").status_code == 409
    assert client_obj.post(button_dict["url_str"], json={"confirmed_bool": True, "key_str": button_dict["key_str"]},
        headers={"Origin": "http://evil.example", "X-Alpha-Action-Token": button_dict["token_str"]}).status_code == 403
    assert "Decision differs from the backtest." in client_obj.get("/").get_data(as_text=True)
