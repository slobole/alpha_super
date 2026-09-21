"""US equity session labels for V4 only; never an execution-hours gate."""

from datetime import datetime, time, timezone

from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.dashboard_v3.schedule import build_market_status
from alpha.live.ops_report import parse_timestamp_ts


def build_market_view_dict(*, now_ts: datetime) -> dict:
    # Keep V3's UTC convention for naive timestamps and its XNYS holiday / core
    # session calendar. Extended hours follow NYSE Arca: 04:00 to core open,
    # core close to 20:00 (17:00 on early-close days). This is display only.
    # https://www.nyse.com/trade/hours-calendars
    if now_ts.tzinfo is None:
        now_ts = now_ts.replace(tzinfo=timezone.utc)
    market_obj = build_market_status(now_dt=now_ts)
    local_ts = now_ts.astimezone(MARKET_TIMEZONE_OBJ)
    label_str = market_obj.status_label_str
    detail_str = market_obj.reason_label_str
    transition_ts = parse_timestamp_ts(market_obj.next_transition_timestamp_str)
    countdown_str = ""
    if label_str == "Market open":
        countdown_str = "closes in "
    elif detail_str == "Pre-market":
        if local_ts.time() >= time(4):
            label_str, countdown_str = "Premarket", "opens in "
        else:
            detail_str = "Premarket at 04:00 ET"
    elif detail_str in {"Session completed", "Early close completed"}:
        end_hour_int = 17 if detail_str == "Early close completed" else 20
        transition_ts = local_ts.replace(hour=end_hour_int, minute=0, second=0, microsecond=0)
        if local_ts < transition_ts:
            label_str, countdown_str = "Post-market", "ends in "
    if countdown_str and transition_ts:
        seconds_int = max(0, int((transition_ts - now_ts).total_seconds()))
        hours_int, remaining_int = divmod(seconds_int, 3600)
        minutes_int, seconds_int = divmod(remaining_int, 60)
        detail_str = f"{countdown_str}{hours_int:02}:{minutes_int:02}:{seconds_int:02}"
    return {"state_str": "skip" if label_str == "Market closed" else "now",
            "label_str": label_str, "detail_str": detail_str}
