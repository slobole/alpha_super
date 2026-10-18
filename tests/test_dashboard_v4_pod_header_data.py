"""Pod page: calm header, Data tab, folded Backtest notes."""

from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4 import backtest_check_data as btd


@pytest.fixture(autouse=True)
def healthy_host_disk(monkeypatch):
    from alpha.live.dashboard_v3 import health
    monkeypatch.setattr(health.shutil, "disk_usage", lambda path: SimpleNamespace(total=100, used=50, free=50))
    btd._CACHE_DICT.clear()


@pytest.fixture(scope="module")
def client_obj():
    from alpha.live.dashboard_v3 import health
    original_fn = health.shutil.disk_usage
    health.shutil.disk_usage = lambda path: SimpleNamespace(total=100, used=50, free=50)
    try:
        from alpha.live.dashboard_v4.demo import create_demo_app
        yield create_demo_app(demo_backtest_bool=True).test_client()
    finally:
        health.shutil.disk_usage = original_fn


def test_waiting_pod_header_is_short_and_reads_in_words(client_obj):
    html_str = client_obj.get("/pods/demo_1_2").get_data(as_text=True)
    head_str = html_str[html_str.find("pod-head"):html_str.find("pod-actions")]
    assert "OK · waiting" in head_str
    assert 'class="verdict"' not in head_str  # the pill and the cycle panel already say it
    assert 'class="pod-next"' in head_str and "Decide Wed Sep 30</b> · after close" in head_str
    assert "Trade Thu Oct 1</b> · 09:30 open" in head_str and "in 23 days" in head_str
    assert 'title="demo_1_2 · ' in head_str  # the technical id moves to the name's tooltip
    assert "16:00:00" not in head_str and "Scheduled" not in head_str


def test_data_step_opens_a_data_tab_with_what_the_decision_used(client_obj):
    html_str = client_obj.get("/pods/demo_1_2").get_data(as_text=True)
    assert "Prices through Mon Aug 31" in html_str
    assert '<span class="pln">—</span>' not in html_str
    data_str = client_obj.get("/pods/demo_1_2?tab=data").get_data(as_text=True)
    assert 'data-selection-key="evidence:data"' in data_str
    for text_str in ("Prices through", "Mon Aug 31", "Data set", "Checked fresh for this cycle before the decision", "Decision made"):
        assert text_str in data_str


def test_backtest_notes_fold_away(client_obj):
    html_str = client_obj.get("/pods/demo_1_0?tab=backtest").get_data(as_text=True)
    assert '<details class="bt-how"><summary>How this is checked</summary>' in html_str


@pytest.mark.parametrize("reason_str,plain_str", [
    ("Incomplete same-days coverage: 0 live / 2 reference dates missing.", "No return curve vs the backtest yet: 2 backtest days missing."),
    ("Incomplete same-days coverage: 1 live / 1 reference dates missing. Invalid returns.",
     "No return curve vs the backtest yet: 1 backtest day and 1 live day missing. Some saved returns are invalid."),
    ("No matching return history.", "No matching return history."),
])
def test_coverage_gap_reads_in_plain_words(reason_str, plain_str):
    assert btd._coverage_plain_str(reason_str) == plain_str
