"""Rendered Performance controls and table geometry retain their data contract."""

import re

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple


@pytest.fixture
def display_client_obj():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    app_obj.config["TESTING"] = True
    try:
        yield app_obj.test_client()
    finally:
        provider_obj.close()


def test_dates_remain_iso_and_report_refresh_keeps_selected_scope(display_client_obj):
    response_obj = display_client_obj.get("/performance?level=pods&from=2026-07-01&to=2026-08-31")
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    from_str = re.search(r'<input[^>]+name="from"[^>]+>', html_str).group()
    to_str = re.search(r'<input[^>]+name="to"[^>]+>', html_str).group()
    assert 'type="text"' in from_str and 'type="text"' in to_str
    assert 'value="2026-07-01"' in from_str and 'value="2026-08-31"' in to_str
    assert 'placeholder="YYYY-MM-DD"' in from_str and 'placeholder="YYYY-MM-DD"' in to_str
    assert 'data-max-date="2026-09-08"' in from_str
    refresh_str = re.search(r'<a[^>]+title="Refresh report"[^>]*>Refresh</a>', html_str).group()
    assert "level=pods" in refresh_str and "from=2026-07-01" in refresh_str and "to=2026-08-31" in refresh_str


def test_pod_display_has_line_legend_separate_bars_and_single_year_heatmap(display_client_obj):
    response_obj = display_client_obj.get("/performance?level=pods")
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert html_str.index('class="performance-legend"') < html_str.index('class="v4-chart v4-chart-pods"')
    assert 'class="performance-line-key"' in html_str
    contribution_str = re.search(r'<table class="tbl dense performance-contributions">(.*?)</table>', html_str, re.S).group(1)
    first_row_str = re.search(r'<tbody>\s*<tr>(.*?)</tr>', contribution_str, re.S).group(1)
    assert len(re.findall(r'<td\b', first_row_str)) == 3
    assert 'class="contribution-plot-cell"' in first_row_str
    assert "--contribution-zero:" in first_row_str
    monthly_str = re.search(r'<table class="tbl performance-month-grid">(.*?)</table>', html_str, re.S).group(1)
    assert "heat-pos-" in monthly_str
    assert '>·</td>' in monthly_str
    assert "<br>2026" not in monthly_str
    assert "All 4 pods are up in this period." in html_str
