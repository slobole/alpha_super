"""Capsule-only release lock and disabled example isolation; no broker access."""
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
def test_capsule_live_mode_refused_even_when_disabled_or_margin_confirmed(template_path_obj, enabled_bool, tmp_path):
    payload_dict = yaml.safe_load(template_path_obj.read_text(encoding="utf-8"))
    payload_dict["deployment"].update(mode="live", enabled_bool=enabled_bool)
    payload_dict["strategy"]["params"]["margin_account_confirmed_bool"] = True
    manifest_path_obj = tmp_path / "capsule.yaml"
    manifest_path_obj.write_text(yaml.safe_dump(payload_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="MR capsule LIVE trading is locked"):
        parse_release_manifest(str(manifest_path_obj))


@pytest.mark.parametrize("template_path_obj", CAPSULE_TEMPLATE_PATH_TUPLE, ids=lambda path_obj: path_obj.stem)
@pytest.mark.parametrize("mode_str", ["paper", "incubation"])
def test_capsule_nonlive_routes_keep_budget_one(template_path_obj, mode_str):
    release_obj = parse_release_manifest(str(template_path_obj))
    validate_release_manifest(replace(
        release_obj, mode_str=mode_str, enabled_bool=True,
        account_route_str="SIM_CAPSULE" if mode_str == "incubation" else "DU_CAPSULE",
        params_dict={"margin_account_confirmed_bool": True},
    ))
    with pytest.raises(ValueError, match="pod_budget_fraction_float"):
        validate_release_manifest(replace(release_obj, pod_budget_fraction_float=0.97))


def test_capsule_live_lock_has_no_yaml_override():
    release_obj = parse_release_manifest(str(CAPSULE_TEMPLATE_PATH_TUPLE[0]))
    with pytest.raises(ValueError, match="MR capsule LIVE trading is locked"):
        validate_release_manifest(replace(
            release_obj, mode_str="live", enabled_bool=True,
            params_dict={"margin_account_confirmed_bool": True, "allow_live_bool": True},
        ))


def test_capsule_examples_have_distinct_ids_and_remain_disabled():
    release_list = [parse_release_manifest(str(template_path_obj)) for template_path_obj in CAPSULE_TEMPLATE_PATH_TUPLE]
    assert {release_obj.strategy_import_str for release_obj in release_list} == set(MR_CAPSULE_STRATEGY_IMPORT_TUPLE)
    client_id_set = {release_obj.broker_client_id_int for release_obj in release_list}
    assert len(client_id_set) == len(release_list) == 6
    assert not client_id_set.intersection({31, 32})
    assert all(not release_obj.enabled_bool and not release_obj.auto_submit_enabled_bool for release_obj in release_list)
    assert all(release_obj.mode_str == "paper" and release_obj.pod_budget_fraction_float == 1.0 for release_obj in release_list)


@pytest.mark.parametrize("template_name_str", [
    "pod_ndx_atr_normalized_monthly_open.yaml.example",
    "pod_ndx_atr_normalized_vxn_scaled_monthly_open.yaml.example",
    "pod_taa_btal_linearity_1n_fallback_qqq_vix_cash_monthly_open.yaml.example",
    "pod_taa_adaptive_macro_core5_daily_moo.yaml.example",
])
def test_non_capsule_release_validation_retains_existing_live_behavior(template_name_str):
    release_obj = parse_release_manifest(str(TEMPLATE_DIR_PATH_OBJ / template_name_str))
    if "core5" in template_name_str:
        with pytest.raises(ValueError, match="CORE5"):
            validate_release_manifest(replace(release_obj, mode_str="live", enabled_bool=True, account_route_str="U_TEST"))
    else:
        validate_release_manifest(replace(release_obj, mode_str="live", enabled_bool=True, account_route_str="U_TEST"))
