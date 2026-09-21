"""Market header boundaries follow ET sessions without changing execution gates."""

from datetime import datetime

import pytest

from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.dashboard_v3.schedule import build_market_status
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.market import build_market_view_dict


@pytest.mark.parametrize("local_str,label_str,detail_str", [
    ("2026-09-21T03:59:59", "Market closed", "Premarket at 04:00 ET"),
    ("2026-09-21T04:00:00", "Premarket", "opens in 05:30:00"),
    ("2026-09-21T09:29:59", "Premarket", "opens in 00:00:01"),
    ("2026-09-21T09:30:00", "Market open", "closes in 06:30:00"),
    ("2026-09-21T15:59:59", "Market open", "closes in 00:00:01"),
    ("2026-09-21T16:00:00", "Post-market", "ends in 04:00:00"),
    ("2026-09-21T19:59:59", "Post-market", "ends in 00:00:01"),
    ("2026-09-21T20:00:00", "Market closed", "Session completed"),
    ("2026-09-21T23:59:59", "Market closed", "Session completed"),
    ("2026-09-22T00:00:00", "Market closed", "Premarket at 04:00 ET"),
    ("2026-11-27T12:59:59", "Market open", "closes in 00:00:01"),
    ("2026-11-27T13:00:00", "Post-market", "ends in 04:00:00"),
    ("2026-11-27T16:59:59", "Post-market", "ends in 00:00:01"),
    ("2026-11-27T17:00:00", "Market closed", "Early close completed"),
    ("2026-12-24T13:00:00", "Post-market", "ends in 04:00:00"),
    ("2026-09-19T08:00:00", "Market closed", "Weekend"),
    ("2026-09-20T17:00:00", "Market closed", "Weekend"),
    ("2026-09-07T08:00:00", "Market closed", "Exchange holiday"),
    ("2026-09-07T17:00:00", "Market closed", "Exchange holiday"),
    ("2026-12-25T12:00:00", "Market closed", "Exchange holiday"),
])
def test_exact_session_boundaries(local_str, label_str, detail_str):
    now_ts = datetime.fromisoformat(local_str).replace(tzinfo=MARKET_TIMEZONE_OBJ)
    assert build_market_view_dict(now_ts=now_ts) == {
        "label_str": label_str, "detail_str": detail_str,
        "state_str": "skip" if label_str == "Market closed" else "now"}


@pytest.mark.parametrize("timestamp_str,label_str,detail_str", [
    ("2026-03-06T14:00:00+00:00", "Premarket", "opens in 00:30:00"),
    ("2026-03-09T13:00:00+00:00", "Premarket", "opens in 00:30:00"),
    ("2026-10-30T20:30:00+00:00", "Post-market", "ends in 03:30:00"),
    ("2026-11-02T21:30:00+00:00", "Post-market", "ends in 03:30:00"),
    ("2026-09-22T00:30:00+00:00", "Market closed", "Session completed"),
    ("2026-09-21T08:00:00", "Premarket", "opens in 05:30:00"),
])
def test_utc_dst_and_naive_convention(timestamp_str, label_str, detail_str):
    result_dict = build_market_view_dict(now_ts=datetime.fromisoformat(timestamp_str))
    assert result_dict["label_str"] == label_str
    assert result_dict["detail_str"] == detail_str


@pytest.mark.parametrize("hour_int,label_str", [(8, "Premarket"), (17, "Post-market")])
def test_common_status_header_and_v3_parity(hour_int, label_str):
    now_ts = datetime(2026, 9, 21, hour_int, tzinfo=MARKET_TIMEZONE_OBJ)
    workspace_dict = {"client_dict": {"accounts": [], "display_name": "Fixture"},
        "operations_account_list": [], "operations_error_str": None,
        "summary_dict": {"as_of_timestamp_str": now_ts.isoformat(), "pod_row_dict_list": []}}

    def forbidden_fn(*args_tuple, **kwargs_dict):
        pytest.fail("Market status attempted financial acquisition")

    app_obj = create_app(object(), workspace_snapshot_fn=forbidden_fn,
        operations_workspace_fn=lambda: workspace_dict, now_fn=lambda: now_ts)
    response_obj = app_obj.test_client().get("/performance/status")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert html_str.count(f'data-status-label>{label_str}</') == 2  # desktop and mobile
    assert 'hx-swap-oob="outerHTML"' in html_str
    assert "System unknown" in html_str  # Active market hours do not imply healthy Pods.
    assert build_market_status(now_dt=now_ts).status_label_str == "Market closed"
