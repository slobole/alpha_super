from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from alpha.live import scheduler_utils
from alpha.live.models import LiveRelease
from alpha.live.order_clerk import validate_account_route_matches_mode


SUPPORTED_STRATEGY_IMPORT_TUPLE: tuple[str, ...] = (
    "strategies.dv2.strategy_mr_dv2:DVO2Strategy",
    "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy",
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy",
)
SUPPORTED_EXECUTION_POLICY_TUPLE: tuple[str, ...] = (
    "next_open_moo",
    "same_day_moc",
    "next_month_first_open",
)
SUPPORTED_MODE_TUPLE: tuple[str, ...] = ("paper", "live")
SUPPORTED_SIGNAL_CLOCK_TUPLE: tuple[str, ...] = scheduler_utils.SUPPORTED_SIGNAL_CLOCK_TUPLE
SUPPORTED_DATA_PROFILE_TUPLE: tuple[str, ...] = (
    "norgate_eod_sp500_pit",
    "norgate_eod_etf_plus_vix_helper",
    "norgate_eod_ndx_pit",
    "intraday_1m_plus_daily_pit",
)
SUPPORTED_SESSION_CALENDAR_ID_TUPLE: tuple[str, ...] = scheduler_utils.SUPPORTED_SESSION_CALENDAR_ID_TUPLE


def _get_optional_mapping(raw_payload_dict: dict[str, Any], field_name_str: str) -> dict[str, Any]:
    field_value_obj = raw_payload_dict.get(field_name_str, {})
    if field_value_obj is None:
        return {}
    if not isinstance(field_value_obj, dict):
        raise ValueError(f"Manifest field '{field_name_str}' must be a mapping.")
    return field_value_obj


def _resolve_manifest_value(
    raw_payload_dict: dict[str, Any],
    flat_field_name_str: str,
    section_field_name_str: str | None = None,
    nested_field_name_str: str | None = None,
    required_bool: bool = True,
    default_value_obj: Any | None = None,
) -> Any:
    if flat_field_name_str in raw_payload_dict:
        return raw_payload_dict[flat_field_name_str]

    if section_field_name_str is not None and nested_field_name_str is not None:
        section_payload_dict = _get_optional_mapping(raw_payload_dict, section_field_name_str)
        if nested_field_name_str in section_payload_dict:
            return section_payload_dict[nested_field_name_str]

    if required_bool:
        if section_field_name_str is None or nested_field_name_str is None:
            raise ValueError(f"Manifest is missing required field '{flat_field_name_str}'.")
        raise ValueError(
            "Manifest is missing required field "
            f"'{flat_field_name_str}' (or '{section_field_name_str}.{nested_field_name_str}')."
        )

    return default_value_obj


def parse_release_manifest(manifest_path_str: str) -> LiveRelease:
    manifest_path_obj = Path(manifest_path_str)
    raw_payload_dict = yaml.safe_load(manifest_path_obj.read_text(encoding="utf-8"))
    if not isinstance(raw_payload_dict, dict):
        raise ValueError(f"Manifest {manifest_path_obj} must decode to a mapping.")

    strategy_dict = _get_optional_mapping(raw_payload_dict, "strategy")
    bootstrap_dict = _get_optional_mapping(raw_payload_dict, "bootstrap")
    params_dict = raw_payload_dict.get("params", strategy_dict.get("params", {}))
    if params_dict is None:
        params_dict = {}
    if not isinstance(params_dict, dict):
        raise ValueError(f"Manifest {manifest_path_obj} field 'params' must be a mapping.")

    params_dict = dict(params_dict)
    initial_cash_float = bootstrap_dict.get("initial_cash_float", bootstrap_dict.get("capital_base_float"))
    if initial_cash_float is not None and "capital_base_float" not in params_dict:
        params_dict["capital_base_float"] = float(initial_cash_float)

    release_obj = LiveRelease(
        release_id_str=str(_resolve_manifest_value(raw_payload_dict, "release_id", "identity", "release_id")),
        user_id_str=str(_resolve_manifest_value(raw_payload_dict, "user_id", "identity", "user_id")),
        pod_id_str=str(_resolve_manifest_value(raw_payload_dict, "pod_id", "identity", "pod_id")),
        account_route_str=str(
            _resolve_manifest_value(raw_payload_dict, "account_route", "broker", "account_route")
        ),
        strategy_import_str=str(
            _resolve_manifest_value(raw_payload_dict, "strategy_import_str", "strategy", "strategy_import_str")
        ),
        mode_str=str(_resolve_manifest_value(raw_payload_dict, "mode", "deployment", "mode")),
        session_calendar_id_str=str(
            _resolve_manifest_value(
                raw_payload_dict,
                "session_calendar_id_str",
                "market",
                "session_calendar_id_str",
            )
        ),
        signal_clock_str=str(
            scheduler_utils.normalize_signal_clock_str(
                _resolve_manifest_value(raw_payload_dict, "signal_clock_str", "schedule", "signal_clock_str")
            )
        ),
        execution_policy_str=str(
            _resolve_manifest_value(raw_payload_dict, "execution_policy_str", "schedule", "execution_policy_str")
        ),
        data_profile_str=str(
            _resolve_manifest_value(raw_payload_dict, "data_profile_str", "strategy", "data_profile_str")
        ),
        params_dict=params_dict,
        risk_profile_str=str(
            _resolve_manifest_value(raw_payload_dict, "risk_profile_str", "risk", "risk_profile_str")
        ),
        enabled_bool=bool(
            _resolve_manifest_value(raw_payload_dict, "enabled_bool", "deployment", "enabled_bool")
        ),
        source_path_str=str(manifest_path_obj),
        pod_budget_fraction_float=float(
            _resolve_manifest_value(
                raw_payload_dict,
                "pod_budget_fraction_float",
                "execution",
                "pod_budget_fraction_float",
                required_bool=False,
                default_value_obj=0.03,
            )
        ),
        auto_submit_enabled_bool=bool(
            _resolve_manifest_value(
                raw_payload_dict,
                "auto_submit_enabled_bool",
                "execution",
                "auto_submit_enabled_bool",
                required_bool=False,
                default_value_obj=True,
            )
        ),
    )
    validate_release_manifest(release_obj)
    return release_obj


def validate_release_manifest(release_obj: LiveRelease) -> None:
    if release_obj.strategy_import_str not in SUPPORTED_STRATEGY_IMPORT_TUPLE:
        raise ValueError(
            "Unsupported strategy_import_str "
            f"'{release_obj.strategy_import_str}'. Expected one of {SUPPORTED_STRATEGY_IMPORT_TUPLE}."
        )
    if release_obj.mode_str not in SUPPORTED_MODE_TUPLE:
        raise ValueError(
            f"Unsupported mode_str '{release_obj.mode_str}'. Expected one of {SUPPORTED_MODE_TUPLE}."
        )
    if release_obj.session_calendar_id_str not in SUPPORTED_SESSION_CALENDAR_ID_TUPLE:
        raise ValueError(
            "Unsupported session_calendar_id_str "
            f"'{release_obj.session_calendar_id_str}'. Expected one of {SUPPORTED_SESSION_CALENDAR_ID_TUPLE}."
        )
    if release_obj.execution_policy_str not in SUPPORTED_EXECUTION_POLICY_TUPLE:
        raise ValueError(
            "Unsupported execution_policy_str "
            f"'{release_obj.execution_policy_str}'. Expected one of {SUPPORTED_EXECUTION_POLICY_TUPLE}."
        )
    if release_obj.signal_clock_str not in SUPPORTED_SIGNAL_CLOCK_TUPLE:
        raise ValueError(
            f"Unsupported signal_clock_str '{release_obj.signal_clock_str}'. "
            f"Expected one of {SUPPORTED_SIGNAL_CLOCK_TUPLE}."
        )
    if release_obj.data_profile_str not in SUPPORTED_DATA_PROFILE_TUPLE:
        raise ValueError(
            f"Unsupported data_profile_str '{release_obj.data_profile_str}'. "
            f"Expected one of {SUPPORTED_DATA_PROFILE_TUPLE}."
        )
    validate_account_route_matches_mode(
        mode_str=release_obj.mode_str,
        account_route_str=release_obj.account_route_str,
    )
    if not (0.0 < float(release_obj.pod_budget_fraction_float) <= 1.0):
        raise ValueError(
            "Manifest field 'pod_budget_fraction_float' must satisfy "
            "0 < pod_budget_fraction_float <= 1."
        )
    for field_name_str, field_value_obj in asdict(release_obj).items():
        if field_name_str == "params_dict":
            continue
        if isinstance(field_value_obj, str) and len(field_value_obj.strip()) == 0:
            raise ValueError(f"Manifest field '{field_name_str}' must not be empty.")


def load_release_list(releases_root_path_str: str) -> list[LiveRelease]:
    releases_root_path_obj = Path(releases_root_path_str)
    if not releases_root_path_obj.exists():
        return []

    release_list: list[LiveRelease] = []
    for manifest_path_obj in sorted(releases_root_path_obj.rglob("*.yaml")):
        release_list.append(parse_release_manifest(str(manifest_path_obj)))

    validate_release_list(release_list)
    return release_list


def validate_release_list(release_list: list[LiveRelease]) -> None:
    enabled_release_id_set: set[str] = set()
    enabled_pod_id_set: set[str] = set()

    for release_obj in release_list:
        if not release_obj.enabled_bool:
            continue
        if release_obj.release_id_str in enabled_release_id_set:
            raise ValueError(f"Duplicate enabled release_id_str '{release_obj.release_id_str}'.")
        if release_obj.pod_id_str in enabled_pod_id_set:
            raise ValueError(
                f"Duplicate enabled pod_id_str '{release_obj.pod_id_str}'. "
                "V1 allows only one enabled release per pod."
            )
        enabled_release_id_set.add(release_obj.release_id_str)
        enabled_pod_id_set.add(release_obj.pod_id_str)
