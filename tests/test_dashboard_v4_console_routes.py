"""Console routes: bounded, read-only, isolated from the page refresh."""

from html.parser import HTMLParser
import threading

import pytest

from alpha.live.dashboard_v4 import app as app_module
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import create_demo_app
from alpha.live.logging_utils import render_operator_message_str


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.element_list = []

    def handle_starttag(self, tag_str, attribute_list):
        self.element_list.append((tag_str, dict(attribute_list)))


@pytest.fixture(scope="module")
def demo_client():
    return create_demo_app().test_client()


class _ConsoleOnlyProvider:
    """Any summary, workspace or SQLite read from a tail request is a failure."""

    def __init__(self, log_path_str):
        self.console_log_path_str = log_path_str

    def get_console_pod_list(self):
        return [{"pod_id_str": "pod_a", "name_str": "Pod A", "mode_str": "live"},
                {"pod_id_str": "pod_p", "name_str": "Pod P", "mode_str": "paper"}]

    def get_summary_dict(self):
        raise AssertionError("The console tail must not build the summary")

    def app_obj(self):
        raise AssertionError("The console tail must not open releases or state")


def _forbidden_fn(*args, **kwargs):
    raise AssertionError("The console tail must not read operations")


@pytest.fixture()
def console_tuple(tmp_path):
    log_path_obj = tmp_path / "live_operator.log"
    line_list = [render_operator_message_str("INFO", "cycle.wait", "2026-10-09T13:00:00+00:00", {"pod": "pod_a", "account": "U21192795"}),
        render_operator_message_str("ERROR", "cycle.fail", "2026-10-09T13:01:00+00:00", {"pod": "pod_ab", "reason": "other pod"}),
        render_operator_message_str("INFO", "norgate.sync.skipped", "2026-10-09T13:01:01+00:00", {"status": "direct"}),
        render_operator_message_str("WARN", "reconcile.wait", "2026-10-09T13:02:00+00:00", {"pod": "pod_p", "token": "abc123"})]
    log_path_obj.write_text("\n".join(line_list) + "\n", encoding="utf-8")
    app_obj = create_app(_ConsoleOnlyProvider(str(log_path_obj)), performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=_forbidden_fn, operations_workspace_fn=_forbidden_fn)
    return app_obj.test_client(), log_path_obj


def test_console_page_uses_status_only_refresh_and_its_own_script(demo_client):
    response_obj = demo_client.get("/console")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    page_obj = _PageParser()
    page_obj.feed(html_str)
    shell_dict, = [attribute_dict for _, attribute_dict in page_obj.element_list if attribute_dict.get("id") == "overview-shell"]
    assert shell_dict["hx-swap"] == "none" and shell_dict["hx-get"] == "/console/status"
    assert shell_dict["hx-trigger"] == "v4poll" and shell_dict["data-refresh-ms"] == "15000"
    panel_dict, = [attribute_dict for _, attribute_dict in page_obj.element_list if attribute_dict.get("id") == "console-panel"]
    assert panel_dict["data-console-tail-url"] == "/console/demo_1_0/tail"
    assert 'id="performance-status"' in html_str  # The status stamp renews the header only.
    assert "/static/console.js" in html_str and "/static/console.css" in html_str
    assert 'href="/console" aria-current="page"' in html_str
    assert "All pods" in html_str and "Download last 2 MB" in html_str


def test_console_page_validates_pod_and_arguments(demo_client):
    assert demo_client.get("/console?pod=demo_1_1").status_code == 200
    assert demo_client.get("/console?pod=all").status_code == 200
    assert demo_client.get("/console?pod=unknown_pod").status_code == 404
    assert demo_client.get("/console?pod=demo_1_1&pod=demo_1_2").status_code == 400
    assert demo_client.get("/console?other=1").status_code == 400
    assert demo_client.get("/console/status?x=1").status_code == 400
    status_str = demo_client.get("/console/status").get_data(as_text=True)
    assert status_str.count('hx-swap-oob="outerHTML"') >= 3


def test_demo_tail_returns_only_that_pods_redacted_lines(demo_client):
    response_obj = demo_client.get("/console/demo_1_1/tail")
    tail_dict = response_obj.get_json()
    assert response_obj.status_code == 200 and response_obj.headers["Cache-Control"] == "no-store"
    assert tail_dict["pod_id_str"] == "demo_1_1" and tail_dict["source_label_str"] == "Operator log"
    text_str = " ".join(line_dict["x"] for line_dict in tail_dict["line_list"])
    assert "pod=demo_1_1" in text_str and "pod=demo_1_0" not in text_str
    assert "DU1234562" not in text_str and "D···562" in text_str
    assert any(line_dict["l"] == "E" and line_dict["a"] == "post_execution_reconcile.fail" for line_dict in tail_dict["line_list"])
    again_dict = demo_client.get("/console/demo_1_1/tail?cursor=" + tail_dict["cursor_str"]).get_json()
    assert again_dict["line_list"] == [] and again_dict["reset_bool"] is False


def test_tail_reads_only_the_log_never_operations(console_tuple):
    client_obj, log_path_obj = console_tuple
    first_dict = client_obj.get("/console/pod_a/tail").get_json()
    assert [line_dict["a"] for line_dict in first_dict["line_list"]] == ["cycle.wait"]
    assert "U21192795" not in str(first_dict) and "U···795" in first_dict["line_list"][0]["x"]
    all_dict = client_obj.get("/console/all/tail").get_json()
    assert [line_dict["a"] for line_dict in all_dict["line_list"]] == ["cycle.wait", "cycle.fail", "reconcile.wait"]
    assert "abc123" not in str(all_dict)
    with log_path_obj.open("a", encoding="utf-8") as log_file_obj:
        log_file_obj.write(render_operator_message_str("ERROR", "cycle.fail", "2026-10-09T13:05:00+00:00", {"pod": "pod_a"}) + "\n")
    next_dict = client_obj.get("/console/pod_a/tail?cursor=" + first_dict["cursor_str"]).get_json()
    assert [line_dict["a"] for line_dict in next_dict["line_list"]] == ["cycle.fail"]
    peek_dict = client_obj.get("/console/pod_a/tail?peek=1&cursor=" + next_dict["cursor_str"]).get_json()
    assert peek_dict["pending_bytes_int"] == 0 and peek_dict["line_list"] == []


@pytest.mark.parametrize("path_str,code_int", [
    ("/console/unknown/tail", 404), ("/console/pod_a/tail?cursor=" + "x" * 301, 400),
    ("/console/pod_a/tail?peek=2", 400), ("/console/pod_a/tail?other=1", 400),
    ("/console/pod_a/tail?cursor=a&cursor=b", 400), ("/console/pod_a/download?x=1", 400),
    ("/console/unknown/download", 404),
])
def test_tail_and_download_reject_bad_requests(console_tuple, path_str, code_int):
    client_obj, _ = console_tuple
    assert client_obj.get(path_str).status_code == code_int


def test_garbage_cursor_resets_without_error(console_tuple):
    client_obj, _ = console_tuple
    tail_dict = client_obj.get("/console/pod_a/tail?cursor=not-a-cursor").get_json()
    assert tail_dict["reset_bool"] is True and len(tail_dict["line_list"]) == 1


def test_console_routes_are_read_only(console_tuple):
    client_obj, _ = console_tuple
    for path_str in ("/console", "/console/pod_a/tail", "/console/pod_a/download"):
        assert client_obj.post(path_str).status_code == 403


def test_download_is_redacted_filtered_text(console_tuple):
    client_obj, _ = console_tuple
    response_obj = client_obj.get("/console/pod_p/download")
    body_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200 and response_obj.mimetype == "text/plain"
    assert response_obj.headers["Content-Disposition"] == 'attachment; filename="console-pod_p.log"'
    assert body_str.splitlines()[0].startswith("# Operator log · pod=pod_p")
    assert "reconcile.wait" in body_str and "cycle.wait" not in body_str and "abc123" not in body_str


def test_tail_concurrency_is_capped_with_retry_after(console_tuple, monkeypatch):
    client_obj, _ = console_tuple
    release_obj, started_obj = threading.Event(), threading.Semaphore(0)

    def blocking_read(*args, **kwargs):
        started_obj.release()
        release_obj.wait(5)
        return {"line_list": [], "cursor_str": ""}

    monkeypatch.setattr(app_module, "read_console_tail_dict", blocking_read)
    thread_list = [threading.Thread(target=client_obj.get, args=("/console/pod_a/tail",)) for _ in range(4)]
    for thread_obj in thread_list:
        thread_obj.start()
    for _ in range(4):
        assert started_obj.acquire(timeout=5)
    busy_obj = client_obj.get("/console/pod_a/tail")
    release_obj.set()
    for thread_obj in thread_list:
        thread_obj.join(5)
    assert busy_obj.status_code == 429 and busy_obj.headers["Retry-After"] == "5"
