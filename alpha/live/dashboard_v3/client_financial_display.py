"""Presentation of saved financial evidence; no accounting or source mutations."""

from datetime import timedelta
from decimal import Decimal, InvalidOperation
import re
from zoneinfo import ZoneInfo

from alpha.live.client_reporting import CAPITAL_FIELD_TUPLE
from alpha.live.scheduler_utils import get_exchange_calendar_obj


def financial_dates_dict(client_dict, snapshot_obj, *, as_of_ts, valuation_account_list=None):
    """Select by complete finalized NAV coverage, never by a successful return."""
    today_obj = as_of_ts.astimezone(ZoneInfo("America/New_York")).date()
    yesterday_obj = today_obj - timedelta(days=1)
    calendar_obj = get_exchange_calendar_obj("XNYS")
    expected_str = calendar_obj.date_to_session(yesterday_obj.isoformat(), direction="previous").date().isoformat()
    account_list = client_dict["accounts"] if valuation_account_list is None else valuation_account_list
    route_set = {account_dict["account_route"] for account_dict in account_list}
    route_by_date_dict = {}
    for row_obj in snapshot_obj.row_tuple:
        if row_obj.account_route_str in route_set and row_obj.market_date_str < today_obj.isoformat():
            route_by_date_dict.setdefault(row_obj.market_date_str, set()).add(row_obj.account_route_str)
    complete_date_list = []
    # *** CRITICAL *** retrospective D+1 display cutoff only. Local identities
    # include every valued account; strategy coverage is not funding history.
    for date_str, observed_set in route_by_date_dict.items():
        required_set = route_set if valuation_account_list is not None else {
            account_dict["account_route"] for account_dict in account_list
            if account_dict["effective_from"] <= date_str <= (account_dict.get("effective_to") or "9999-12-31")}
        if required_set and required_set <= observed_set:
            complete_date_list.append(date_str)
        if required_set & observed_set:
            expected_str = max(expected_str, date_str)  # Include partial weekend activity.
    latest_str = max(complete_date_list, default=None) if not snapshot_obj.unavailable_reason_str else None
    return {"latest_complete_str": latest_str, "expected_str": expected_str,
        "delayed_bool": latest_str is None or latest_str < expected_str}


def summarized_issue_list(report_dict, notice_list):
    """Compact known repeated patterns; leave full raw diagnostics untouched."""
    group_dict, handled_set = {}, set()
    for strategy_dict in report_dict["strategy_list"]:
        prefix_str = f"{strategy_dict['display_name_str']} [{strategy_dict['account_route_str']}]"
        for issue_str in strategy_dict["issue_list"]:
            match_obj = re.fullmatch(r"(\d{4}-\d{2}-\d{2}): (.+)", issue_str)
            if match_obj:
                date_str, reason_str = match_obj.groups()
                group_dict.setdefault((prefix_str, reason_str), set()).add(date_str)
                handled_set.add(f"{strategy_dict['display_name_str']}: {issue_str}")
    remaining_list = []
    for issue_str in dict.fromkeys(notice_list + report_dict["issue_list"]):
        if issue_str in handled_set:
            continue
        match_obj = re.fullmatch(r"(.+): missing IBKR NAV for (\d{4}-\d{2}-\d{2})\.", issue_str)
        if match_obj:
            prefix_str, date_str = match_obj.groups()
            group_dict.setdefault((prefix_str, "Missing IBKR NAV"), set()).add(date_str)
        else:
            remaining_list.append(issue_str)
    summary_list = []
    for (prefix_str, reason_str), date_set in group_dict.items():
        date_range_str = min(date_set) if len(date_set) == 1 else f"{min(date_set)} → {max(date_set)}"
        summary_list.append(f"{prefix_str}: {reason_str} · {len(date_set)} {'day' if len(date_set) == 1 else 'days'} · {date_range_str}")
    return summary_list + remaining_list


def capital_day_key_set(snapshot_obj):
    """Positive evidence of movement, including offsetting fields/accounts.

    Absent/invalid fields cannot prove a movement or its absence. This marker
    never authorizes P&L, net-flow amounts, return availability or classification.
    """
    movement_set = set()
    for row_obj in snapshot_obj.row_tuple:
        for field_str in CAPITAL_FIELD_TUPLE:
            try:
                value_decimal = Decimal(row_obj.attribute_dict[field_str])
            except (KeyError, InvalidOperation):
                continue
            if value_decimal.is_finite() and value_decimal != 0:
                movement_set.add((row_obj.account_route_str, row_obj.market_date_str))
                break
    return movement_set
