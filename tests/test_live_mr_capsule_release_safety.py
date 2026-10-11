"""Capsule release rules (LIVE allowed, no margin attestation) and disabled example isolation; no broker access."""
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE
from alpha.live.release_manifest import parse_release_manifest, validate_release_manifest


TEMPLATE_DIR_PATH_OBJ = Path("docs/live/release_templates")
CAPSULE_TEMPLATE_PATH_TUPLE = tuple(sorted(TEMPLATE_DIR_PATH_OBJ.glob("pod_mr*.yaml.example")))


@pytest.mark.parametrize("template_path_obj", CAPSULE_TEMPLATE_PATH_TUPLE, ids=lambda path_obj: path_obj.stem)
@pytest.mark.parametrize("enabled_bool", [False, True])
def test_capsule_live_mode_parses_without_attestation(template_path_obj, enabled_bool, tmp_path):
    # LIVE is allowed (owner decisions 2026-10-10/11) under the same rules as paper.
    payload_dict = yaml.safe_load(template_path_obj.read_text(encoding="utf-8"))
    payload_dict["deployment"].update(mode="live", enabled_bool=enabled_bool)
    payload_dict["broker"]["account_route"] = "U_CAPSULE"
    manifest_path_obj = tmp_path / "capsule.yaml"
    manifest_path_obj.write_text(yaml.safe_dump(payload_dict), encoding="utf-8")
    release_obj = parse_release_manifest(str(manifest_path_obj))
    assert (release_obj.mode_str, release_obj.enabled_bool) == ("live", enabled_bool)


@pytest.mark.parametrize("template_path_obj", CAPSULE_TEMPLATE_PATH_TUPLE, ids=lambda path_obj: path_obj.stem)
@pytest.mark.parametrize("mode_str,account_route_str", [
    ("paper", "DU_CAPSULE"), ("incubation", "SIM_CAPSULE"), ("live", "U_CAPSULE")])
def test_capsule_routes_keep_budget_one(template_path_obj, mode_str, account_route_str):
    release_obj = parse_release_manifest(str(template_path_obj))
    validate_release_manifest(replace(
        release_obj, mode_str=mode_str, enabled_bool=True, account_route_str=account_route_str,
        params_dict={},
    ))
    with pytest.raises(ValueError, match="pod_budget_fraction_float"):
        validate_release_manifest(replace(release_obj, mode_str=mode_str, pod_budget_fraction_float=0.97))


@pytest.mark.parametrize("template_path_obj", CAPSULE_TEMPLATE_PATH_TUPLE, ids=lambda path_obj: path_obj.stem)
def test_capsule_live_ignores_old_margin_flag_and_keeps_frozen_params(template_path_obj):
    live_release_obj = replace(parse_release_manifest(str(template_path_obj)),
        mode_str="live", enabled_bool=True, account_route_str="U_CAPSULE")
    for params_dict in [{}, {"margin_account_confirmed_bool": False}, {"margin_account_confirmed_bool": True}]:
        validate_release_manifest(replace(live_release_obj, params_dict=params_dict))
    with pytest.raises(ValueError, match="frozen strategy parameters"):
        validate_release_manifest(replace(live_release_obj,
            params_dict={"margin_account_confirmed_bool": True, "allow_live_bool": True}))


def test_capsule_examples_have_distinct_ids_and_remain_disabled():
    release_list = [parse_release_manifest(str(template_path_obj)) for template_path_obj in CAPSULE_TEMPLATE_PATH_TUPLE]
    assert {release_obj.strategy_import_str for release_obj in release_list} == set(MR_CAPSULE_STRATEGY_IMPORT_TUPLE)
    client_id_set = {release_obj.broker_client_id_int for release_obj in release_list}
    assert len(client_id_set) == len(release_list) == 6
    assert not client_id_set.intersection({31, 32})
    # Disabled, but auto-submit once enabled: daily pods send every session.
    assert all(not release_obj.enabled_bool and release_obj.auto_submit_enabled_bool for release_obj in release_list)
    assert all(release_obj.mode_str == "paper" and release_obj.pod_budget_fraction_float == 1.0 for release_obj in release_list)


@pytest.mark.parametrize("template_name_str", [
    "pod_ndx_atr_normalized_monthly_open.yaml.example",
    "pod_ndx_atr_normalized_vxn_scaled_monthly_open.yaml.example",
    "pod_taa_btal_linearity_1n_fallback_qqq_vix_cash_monthly_open.yaml.example",
    "pod_taa_adaptive_macro_core5_daily_moo.yaml.example",
])
def test_non_capsule_release_validation_retains_existing_live_behavior(template_name_str):
    # CORE5 LIVE needs no attestation since 2026-10-11, like NDX/TAA.
    release_obj = parse_release_manifest(str(TEMPLATE_DIR_PATH_OBJ / template_name_str))
    validate_release_manifest(replace(release_obj, mode_str="live", enabled_bool=True, account_route_str="U_TEST"))
