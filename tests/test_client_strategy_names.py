"""Friendly labels never substitute for account/Pod routing identity."""

from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.client_views import local_strategy_name_str
from test_client_reporting import client_config_dict
from test_dashboard_operator_access import ForbiddenProvider, TEST_ACCESS_STR


@pytest.mark.parametrize("case_str", ["match", "wrong_account", "paper", "snapshot", "unconfigured", "retired", "future", "duplicate", "missing", "invalid"])
def test_friendly_names_require_unique_current_local_live_owner(case_str):
    config_dict = client_config_dict()
    config_dict["operations_source"] = "local"
    row_dict = {"pod_id_str": "strategy_a", "account_route_str": "U_TEST_A", "mode_str": "live", "strategy_import_str": "module:FallbackStrategy"}
    registry_dict = {"schema_version": 1, "clients": [config_dict]}
    if case_str == "wrong_account": row_dict["account_route_str"] = "U_TEST_B"
    if case_str == "paper": row_dict["mode_str"] = "paper"
    if case_str == "snapshot":
        config_dict.update(operations_source="snapshot", operations_snapshot_path="C:/synthetic/snapshot.json")
    if case_str == "unconfigured": config_dict.pop("operations_source")
    if case_str == "retired": config_dict["accounts"][0]["effective_to"] = "2026-09-04"
    if case_str == "future": config_dict["accounts"][0]["effective_from"] = "2026-09-07"
    if case_str == "duplicate":
        registry_dict["clients"].append(dict(deepcopy(config_dict), client_id="another"))
    if case_str == "invalid": registry_dict["schema_version"] = 999
    if case_str == "missing": registry_dict = None
    original_dict = deepcopy(row_dict)
    app_obj = create_app(ForbiddenProvider(), client_registry_dict=registry_dict,
        operator_access_token_str=TEST_ACCESS_STR, read_only_bool=True)
    with app_obj.test_request_context(), patch("alpha.live.dashboard_v3.client_views.datetime") as clock_mock:
        clock_mock.now.return_value = datetime(2026, 9, 6, 12, tzinfo=UTC)
        assert local_strategy_name_str(row_dict) == ("Strategy A" if case_str == "match" else "Fallback")
        assert local_strategy_name_str(row_dict) == ("Strategy A" if case_str == "match" else "Fallback")
    assert row_dict == original_dict


def test_same_pod_in_different_accounts_does_not_share_name():
    first_dict = client_config_dict()
    first_dict["operations_source"] = "local"
    second_dict = deepcopy(first_dict)
    second_dict.update(client_id="another", display_name="Another client")
    second_dict["accounts"][0].update(account_route="U_TEST_B", display_name="Other strategy")
    app_obj = create_app(ForbiddenProvider(), client_registry_dict={"schema_version": 1, "clients": [first_dict, second_dict]},
        operator_access_token_str=TEST_ACCESS_STR)
    with app_obj.test_request_context(), patch("alpha.live.dashboard_v3.client_views.datetime") as clock_mock:
        clock_mock.now.return_value = datetime(2026, 9, 6, 12, tzinfo=UTC)
        for route_str, expected_str in [("U_TEST_A", "Strategy A"), ("U_TEST_B", "Other strategy")]:
            assert local_strategy_name_str({"pod_id_str": "strategy_a", "account_route_str": route_str, "mode_str": "live"}) == expected_str
