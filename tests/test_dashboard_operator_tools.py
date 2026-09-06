"""Pure diagnostic projection, filtering and credential boundary tests."""

import pytest

from alpha.live.dashboard_v3.operator_tools import (
    build_diagnostic_payload_dict,
    redact_diagnostic_value,
    strategy_display_name_str,
)


def test_diagnostics_never_relabel_assembly_time_as_observation():
    payload_dict = build_diagnostic_payload_dict(
        {"as_of_timestamp_str": "2026-09-05T13:00:00Z"},
        {"pod_id_str": "pod_a", "latest_broker_snapshot_timestamp_str": "2026-08-01T13:00:00Z"},
        {}, [],
    )
    assert payload_dict["view_built_at_str"] == "2026-09-05T13:00:00Z"
    assert payload_dict["scope_dict"]["latest_broker_snapshot_timestamp_str"] == "2026-08-01T13:00:00Z"
    assert "no active broker or process probe" in payload_dict["limitations_str"]


def test_diagnostic_dates_use_same_et_timezone_as_display():
    event_dict_list = [
        {"timestamp_str": "2026-09-05T01:00:00Z", "level_str": "ERROR", "message_str": "late Friday"},
        {"timestamp_str": "2026-09-05T13:00:00Z", "level_str": "INFO", "message_str": "Saturday"},
        {"level_str": "ERROR", "message_str": "undated"},
    ]
    payload_dict = build_diagnostic_payload_dict(
        {}, {}, {}, event_dict_list, from_date_str="2026-09-04", to_date_str="2026-09-04", level_str="error",
    )
    assert [item_dict["message_str"] for item_dict in payload_dict["event_dict_list"]] == ["late Friday"]
    assert payload_dict["scanned_event_count_int"] == 3


@pytest.mark.parametrize("filter_dict", [
    {"level_str": "execute"}, {"from_date_str": "bad"},
    {"from_date_str": "2026-09-06", "to_date_str": "2026-09-04"},
    {"search_str": "x" * 161},
])
def test_diagnostics_reject_invalid_filters(filter_dict):
    with pytest.raises(ValueError):
        build_diagnostic_payload_dict({}, {}, {}, [], **filter_dict)


def test_redaction_preserves_evidence_but_not_credentials():
    result_dict = redact_diagnostic_value({
        "pod_id_str": "pod_a", "nested_dict": {"api_token_str": "private123"},
        "message_str": 'GET /url?token_str=abc123&date=20260904 Authorization: Bearer secret456 password="two words"',
    })
    assert result_dict["pod_id_str"] == "pod_a"
    assert result_dict["nested_dict"]["api_token_str"] == "[redacted]"
    for secret_str in ["private123", "abc123", "secret456", "two words"]:
        assert secret_str not in str(result_dict)
    assert "20260904" in result_dict["message_str"]


@pytest.mark.parametrize("key_str", ["api_key", "api_key_str", "APIKEY", "X-Api-Key", "api-key", "vendor_api_key_str"])
def test_nested_api_key_forms_are_redacted(key_str):
    result_dict = redact_diagnostic_value({"nested_list": [{key_str: "SYNTHETIC_PRIVATE_KEY", "reason_str": "saved evidence"}]})
    assert "SYNTHETIC_PRIVATE_KEY" not in str(result_dict)
    assert result_dict["nested_list"][0][key_str] == "[redacted]"
    assert result_dict["nested_list"][0]["reason_str"] == "saved evidence"


@pytest.mark.parametrize("message_str", [
    "Authorization: Basic SYNTHETIC_PRIVATE_CREDENTIAL reason=saved",
    "authorization=basic SYNTHETIC_PRIVATE_CREDENTIAL reason=saved",
    'Authorization: "Basic SYNTHETIC_PRIVATE_CREDENTIAL" reason=saved',
    "Basic SYNTHETIC_PRIVATE_CREDENTIAL reason=saved",
    "Bearer SYNTHETIC_PRIVATE_CREDENTIAL reason=saved",
    "GET /status?api_key=SYNTHETIC_PRIVATE_CREDENTIAL&reason=saved",
    "apikey_str=SYNTHETIC_PRIVATE_CREDENTIAL reason=saved",
    "X-Api-Key: SYNTHETIC_PRIVATE_CREDENTIAL reason=saved",
    "apiKey='SYNTHETIC_PRIVATE_CREDENTIAL' reason=saved",
])
def test_api_keys_and_basic_auth_in_text_do_not_survive_redaction(message_str):
    result_str = redact_diagnostic_value(message_str)
    assert "SYNTHETIC_PRIVATE_CREDENTIAL" not in result_str
    assert "[redacted]" in result_str and "reason=saved" in result_str
    assert redact_diagnostic_value(result_str) == result_str


def test_diagnostic_payload_applies_redaction_to_nested_status_and_events():
    result_dict = build_diagnostic_payload_dict({},
        {"required_action_dict": {"api_key": "SYNTHETIC_NESTED_SECRET"}}, {},
        [{"event_timestamp_str": "2026-09-01T12:00:00Z", "message_str": "Authorization: Basic SYNTHETIC_BASIC_SECRET"}])
    assert "SYNTHETIC_NESTED_SECRET" not in str(result_dict)
    assert "SYNTHETIC_BASIC_SECRET" not in str(result_dict)


def test_readable_strategy_name_prefers_explicit_identity():
    assert strategy_display_name_str({"display_name_str": "NDX Momentum", "pod_id_str": "opaque"}) == "NDX Momentum"
    assert strategy_display_name_str({"strategy_import_str": "strategies.demo:MomentumStrategy"}) == "Momentum"
