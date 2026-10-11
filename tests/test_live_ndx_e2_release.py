"""Independent E2 monthly releases retain frozen rules and dedicated accounts."""
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from alpha.live import reference_compare, scheduler_utils
from alpha.live.ndx_e2_adapter import NDX_E2_STRATEGY_IMPORT_TUPLE
from alpha.live.release_manifest import (
    SUPPORTED_STRATEGY_IMPORT_TUPLE,
    parse_release_manifest,
    validate_enabled_deployment_for_mode,
    validate_release_list,
    validate_release_manifest,
)
from data.norgate_snapshot_store import NDX_E2_PROFILE_STR
from scripts import doctor_norgate_client


TEMPLATE_PATH_TUPLE = (
    Path("docs/live/release_templates/pod_ndx_atr_normalized_vxn_scaled_sector_cap_monthly_open.yaml.example"),
    Path("docs/live/release_templates/pod_ndx_natr20_vxn_scaled_sector_cap_monthly_open.yaml.example"),
)
LEGACY_TEMPLATE_PATH = Path("docs/live/release_templates/pod_ndx_atr_normalized_vxn_scaled_monthly_open.yaml.example")


@pytest.fixture(params=TEMPLATE_PATH_TUPLE, ids=("atr", "natr20"))
def release_obj(request):
    return parse_release_manifest(str(request.param))


def test_templates_are_separate_disabled_components(release_obj):
    assert release_obj.strategy_import_str in NDX_E2_STRATEGY_IMPORT_TUPLE
    assert release_obj.strategy_import_str in SUPPORTED_STRATEGY_IMPORT_TUPLE
    assert release_obj.data_profile_str == NDX_E2_PROFILE_STR
    assert not release_obj.enabled_bool and not release_obj.auto_submit_enabled_bool
    assert release_obj.pod_budget_fraction_float == 1.0
    assert reference_compare.inspect_auto_reference_support_dict(release_obj)["supported_bool"]
    validate_enabled_deployment_for_mode([release_obj], "paper")
    with pytest.raises(ValueError, match="real IBKR account"):
        validate_enabled_deployment_for_mode([replace(release_obj, enabled_bool=True)], "paper")


@pytest.mark.parametrize("field_str,value_obj", [
    ("data_profile_str", "norgate_eod_ndx_pit_plus_vxn_helper"),
    ("session_calendar_id_str", "XTSE"),
    ("signal_clock_str", "eod_snapshot_ready"),
    ("execution_policy_str", "next_open_moo"),
])
def test_e2_rejects_wrong_profile_or_timing(release_obj, field_str, value_obj):
    with pytest.raises(ValueError, match="NDX E2 requires"):
        validate_release_manifest(replace(release_obj, **{field_str: value_obj}))


@pytest.mark.parametrize("parameter_str,value_obj", [
    ("max_positions_int", 9), ("sector_cap_int", 5), ("sector_cap_int", 4.5),
    ("lookback_month_int", 6), ("index_trend_window_int", 100),
    ("stock_trend_window_int", 200), ("regime_symbol_str", "QQQ"),
    ("vxn_symbol_str", "$VIX"), ("target_vxn_pct_float", 25.0),
    ("min_exposure_scale_float", 0.0), ("max_exposure_scale_float", 1.5),
    ("slippage_float", 0.0), ("commission_per_share_float", 0.0),
    ("commission_minimum_float", True), ("sector_map_path_str", "unbound.json"),
])
def test_e2_does_not_silently_override_frozen_defaults(release_obj, parameter_str, value_obj):
    with pytest.raises(ValueError, match="frozen strategy parameters"):
        validate_release_manifest(replace(release_obj,
            params_dict={**release_obj.params_dict, parameter_str: value_obj}))


@pytest.mark.parametrize("capital_float", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_bootstrap_capital_is_rejected(release_obj, capital_float):
    with pytest.raises(ValueError, match="capital_base_float"):
        validate_release_manifest(replace(release_obj,
            params_dict={**release_obj.params_dict, "capital_base_float": capital_float}))


@pytest.mark.parametrize("budget_float", [0.25, 1.0])
def test_standard_monthly_budget_fraction_remains_supported(release_obj, budget_float):
    validate_release_manifest(replace(release_obj, pod_budget_fraction_float=budget_float))
    validate_release_manifest(replace(release_obj, params_dict={"capital_base_float": 250000.0}))


@pytest.mark.parametrize("budget_float", [0.0, -0.5, 1.01, float("nan"), float("inf"), True])
def test_budget_fraction_cannot_overallocate_or_be_invalid(release_obj, budget_float):
    with pytest.raises(ValueError, match="budget fraction"):
        validate_release_manifest(replace(release_obj, pod_budget_fraction_float=budget_float))


def test_new_profile_is_reserved_and_legacy_monthly_profile_is_unchanged():
    legacy_obj = parse_release_manifest(str(LEGACY_TEMPLATE_PATH))
    validate_release_manifest(legacy_obj)
    assert legacy_obj.data_profile_str == "norgate_eod_ndx_pit_plus_vxn_helper"
    with pytest.raises(ValueError, match="reserved"):
        validate_release_manifest(replace(legacy_obj, data_profile_str=NDX_E2_PROFILE_STR))


@pytest.mark.parametrize("other_path_obj", [TEMPLATE_PATH_TUPLE[1], LEGACY_TEMPLATE_PATH])
@pytest.mark.parametrize("reverse_bool", [False, True])
def test_enabled_e2_rejects_shared_account_in_both_release_orders(other_path_obj, reverse_bool):
    e2_obj = replace(parse_release_manifest(str(TEMPLATE_PATH_TUPLE[0])), enabled_bool=True,
        account_route_str="DU123456", release_id_str="e2", pod_id_str="e2")
    other_obj = replace(parse_release_manifest(str(other_path_obj)), enabled_bool=True,
        account_route_str=" du123456 ", release_id_str="other", pod_id_str="other")
    release_list = [e2_obj, other_obj]
    with pytest.raises(ValueError, match="NDX E2 requires one dedicated account"):
        validate_release_list(list(reversed(release_list)) if reverse_bool else release_list)


def test_two_dedicated_component_accounts_are_valid_and_disabled_templates_do_not_claim_accounts():
    component_list = [replace(parse_release_manifest(str(path_obj)), enabled_bool=True,
        account_route_str=f"DU12345{component_int}")
        for component_int, path_obj in enumerate(TEMPLATE_PATH_TUPLE)]
    validate_release_list(component_list)
    validate_enabled_deployment_for_mode(component_list, "paper")
    disabled_obj = replace(component_list[1], enabled_bool=False,
        account_route_str=component_list[0].account_route_str)
    validate_release_list([component_list[0], disabled_obj])


@pytest.mark.parametrize("signal_date_str,execution_date_str", [
    ("2025-12-31", "2026-01-02"),
    ("2026-01-30", "2026-02-02"),
    ("2026-08-31", "2026-09-01"),
])
def test_e2_keeps_existing_monthly_exchange_timing(release_obj, signal_date_str, execution_date_str):
    signal_ts = datetime.fromisoformat(signal_date_str)
    target_ts = scheduler_utils.build_target_execution_timestamp_ts(signal_ts, release_obj)
    submit_ts = scheduler_utils.build_submission_timestamp_ts(signal_ts, release_obj)
    assert target_ts.date().isoformat() == execution_date_str
    assert (target_ts.hour, target_ts.minute) == (9, 30)
    assert submit_ts == target_ts - timedelta(seconds=scheduler_utils.DEFAULT_OPEN_SUBMISSION_LEAD_SECONDS_INT)
    assert scheduler_utils.DATA_PROFILE_HEARTBEAT_SYMBOL_MAP[NDX_E2_PROFILE_STR] == "$SPX"


@pytest.mark.parametrize("failed_bool", [False, True])
def test_client_doctor_smokes_the_new_profile_without_using_legacy_profile(tmp_path, monkeypatch, failed_bool):
    call_list = []
    def universe_fn(indexname_str, *, data_profile_str):
        call_list.append((indexname_str, data_profile_str))
        if failed_bool:
            raise ValueError("Hash-bound E2 classification snapshot is unavailable")
        return ["AAPL"], pd.DataFrame({"AAPL": [True]}, index=pd.to_datetime(["2026-09-30"]))
    def price_fn(symbol_str, adjustment_str, *, data_profile_str):
        call_list.append((symbol_str, data_profile_str))
        return pd.DataFrame({"Close": [22.0]}, index=pd.to_datetime(["2026-09-30"]))
    monkeypatch.setattr(doctor_norgate_client, "load_index_constituent_matrix_df", universe_fn)
    monkeypatch.setattr(doctor_norgate_client, "load_price_timeseries_df", price_fn)
    result_list = []
    doctor_norgate_client._check_profile_smoke(result_list, local_root_path_obj=tmp_path,
        profile_list=[NDX_E2_PROFILE_STR], printer_fn=lambda _: None)
    assert call_list == [("Nasdaq 100", NDX_E2_PROFILE_STR), ("$VXN", NDX_E2_PROFILE_STR)]
    assert [result_obj.status_str for result_obj in result_list] == (["FAIL", "PASS"] if failed_bool else ["PASS", "PASS"])
