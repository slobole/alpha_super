"""A full-account capsule cannot share an enabled account with another pod."""
from dataclasses import replace

import pytest

from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE
from alpha.live.release_manifest import parse_release_manifest, validate_release_list


OTHER_STRATEGY_STR = "strategies.dv2.strategy_mr_dv2:DVO2Strategy"


def _release_pair():
    first_obj = replace(
        parse_release_manifest(
            "docs/live/release_templates/pod_mr_dv2_vix_gated_bil_daily_moo.yaml.example"
        ),
        enabled_bool=True, account_route_str="DU123",
        params_dict={"margin_account_confirmed_bool": True},
    )
    return first_obj, replace(first_obj, release_id_str="second_release", pod_id_str="second_pod")


@pytest.mark.parametrize("strategy_str", MR_CAPSULE_STRATEGY_IMPORT_TUPLE)
@pytest.mark.parametrize("capsule_first_bool", [True, False])
def test_enabled_capsule_cannot_share_account_with_another_strategy(strategy_str, capsule_first_bool):
    first_obj, second_obj = _release_pair()
    first_obj = replace(first_obj, strategy_import_str=strategy_str)
    second_obj = replace(second_obj, strategy_import_str=OTHER_STRATEGY_STR)
    release_list = [first_obj, second_obj] if capsule_first_bool else [second_obj, first_obj]
    with pytest.raises(ValueError, match="one dedicated account per pod"):
        validate_release_list(release_list)


def test_two_capsule_pods_cannot_share_same_account_even_with_same_strategy():
    with pytest.raises(ValueError, match="one dedicated account per pod"):
        validate_release_list(list(_release_pair()))


def test_account_identity_is_case_and_whitespace_insensitive():
    first_obj, second_obj = _release_pair()
    with pytest.raises(ValueError, match="one dedicated account per pod"):
        validate_release_list([first_obj, replace(second_obj, account_route_str=" du123 ")])


@pytest.mark.parametrize("change_dict", [
    {"enabled_bool": False}, {"account_route_str": "DU456"}, {"mode_str": "incubation"},
])
def test_disabled_or_separate_mode_or_separate_account_does_not_conflict(change_dict):
    first_obj, second_obj = _release_pair()
    validate_release_list([first_obj, replace(second_obj, **change_dict)])


def test_disabled_templates_can_share_placeholder_account():
    first_obj, second_obj = _release_pair()
    validate_release_list([
        replace(release_obj, enabled_bool=False, account_route_str="DU_YOUR_PAPER_ACCOUNT")
        for release_obj in (first_obj, second_obj)
    ])


def test_existing_non_capsule_sharing_contract_is_unchanged_but_later_capsule_is_rejected():
    first_obj, second_obj = _release_pair()
    release_list = [replace(release_obj, strategy_import_str=OTHER_STRATEGY_STR)
                    for release_obj in (first_obj, second_obj)]
    validate_release_list(release_list)
    release_list.append(replace(first_obj, release_id_str="third_release", pod_id_str="third_pod"))
    with pytest.raises(ValueError, match="one dedicated account per pod"):
        validate_release_list(release_list)
