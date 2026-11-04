"""The momentum capsule's three legs on the existing full-target broker-truth path.

A decision replays the leg's whole path on the frozen snapshot through T and takes its last row. The path reads prices,
membership and labels, never the account, so no strategy memory is stored. The existing execution engine later sizes
whole shares from broker NAV and reference quotes before the next open; research Close_T share quantities are not an
execution-parity claim. Rules and evidence: docs/live/MOMENTUM_LEGS_WIRING_20261103.md.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from importlib import import_module
import math

import numpy as np

from alpha.live import scheduler_utils
from alpha.live.models import DecisionPlan, LiveRelease, PodState
from data import norgate_snapshot_store as snapshot_module
from data.momentum_legs_snapshot_contract import MOMENTUM_LEGS_NDX_PROFILE_STR, MOMENTUM_LEGS_R1000_PROFILE_STR


MOMENTUM_LEG1_STRATEGY_IMPORT_STR = "strategies.momentum.strategy_mo_capsule_leg1_natr_ndx:MomentumCapsuleLeg1Strategy"
MOMENTUM_LEG2_STRATEGY_IMPORT_STR = "strategies.momentum.strategy_mo_capsule_leg2_residual_ndx:MomentumCapsuleLeg2Strategy"
MOMENTUM_LEG3_STRATEGY_IMPORT_STR = "strategies.momentum.strategy_mo_capsule_leg3_residual_r1000:MomentumCapsuleLeg3Strategy"
MOMENTUM_LEGS_STRATEGY_IMPORT_TUPLE = (
    MOMENTUM_LEG1_STRATEGY_IMPORT_STR,
    MOMENTUM_LEG2_STRATEGY_IMPORT_STR,
    MOMENTUM_LEG3_STRATEGY_IMPORT_STR,
)
MOMENTUM_LEGS_PROFILE_BY_IMPORT_DICT = {
    MOMENTUM_LEG1_STRATEGY_IMPORT_STR: MOMENTUM_LEGS_NDX_PROFILE_STR,
    MOMENTUM_LEG2_STRATEGY_IMPORT_STR: MOMENTUM_LEGS_NDX_PROFILE_STR,
    MOMENTUM_LEG3_STRATEGY_IMPORT_STR: MOMENTUM_LEGS_R1000_PROFILE_STR,
}
MOMENTUM_LEGS_SIZING_CONTRACT_STR = "full_target_preopen_v1"
# With the gate open, a working panel scores most members: at least 96% at every saved decision of each leg.
MIN_FINITE_SCORE_MEMBER_FRACTION_FLOAT = 0.8
# A decision is a pure function of the exact manifest. A refusal on one is not recomputed every scheduler tick, because a
# Russell 1000 replay peaks at 7.8 GB. A new manifest or a new process decides again; a memory failure is never kept.
_REFUSED_DECISION_BY_KEY_DICT: dict[tuple[str, str], str] = {}


def validate_momentum_legs_release(release_obj: LiveRelease) -> None:
    """Keep every rule at the closed design's frozen value."""
    if release_obj.strategy_import_str not in MOMENTUM_LEGS_STRATEGY_IMPORT_TUPLE:
        raise ValueError("Unsupported momentum capsule leg identity.")
    required_field_dict = {
        "data_profile_str": MOMENTUM_LEGS_PROFILE_BY_IMPORT_DICT[release_obj.strategy_import_str],
        "session_calendar_id_str": "XNYS",
        "signal_clock_str": scheduler_utils.MONTH_SWAP_SIGNAL_CLOCK_STR,
        "execution_policy_str": "next_open_moo",
    }
    for field_str, expected_value_obj in required_field_dict.items():
        if getattr(release_obj, field_str) != expected_value_obj:
            raise ValueError(f"A momentum capsule leg requires {field_str}={expected_value_obj!r}.")
    if set(release_obj.params_dict) - {"capital_base_float"}:
        raise ValueError("Momentum capsule legs use frozen strategy parameters; overrides require parity qualification.")
    if "capital_base_float" in release_obj.params_dict:
        capital_obj = release_obj.params_dict["capital_base_float"]
        if isinstance(capital_obj, bool) or not math.isfinite(float(capital_obj)) or float(capital_obj) <= 0:
            raise ValueError("A momentum capsule leg capital_base_float must be finite and positive.")
    budget_float = float(release_obj.pod_budget_fraction_float)
    if isinstance(release_obj.pod_budget_fraction_float, bool) or not math.isfinite(budget_float) or not 0 < budget_float <= 1:
        raise ValueError("A momentum capsule leg requires a finite dedicated-account budget fraction in (0, 1].")


def _replayed_decision_dict(release_obj: LiveRelease, signal_date_ts, signal_date_str: str) -> dict:
    """The backtest's own loaders, panel and path on rows through T; every refusal of unavailable evidence is here."""
    from strategies.momentum import momentum_capsule_legs as legs_module

    strategy_module = import_module(release_obj.strategy_import_str.split(":")[0])
    # *** CRITICAL *** The same loaders, panel and path as the backtest, on rows through T only.
    config_obj = replace(strategy_module.DEFAULT_CONFIG, end_date_str=signal_date_str)
    input_dict = legs_module.load_leg_inputs(config_obj)
    pricing_data_df = input_dict.pop("pricing_data_df")
    if (pricing_data_df.empty or pricing_data_df.index.has_duplicates or not pricing_data_df.index.is_monotonic_increasing
            or pricing_data_df.index[-1] != signal_date_ts
            or input_dict["universe_df"].empty or input_dict["universe_df"].index[-1] != signal_date_ts):
        raise ValueError("A momentum capsule leg requires ordered prices and universe through the exact decision close.")
    panel_obj = legs_module.build_leg_panel(config_obj, pricing_data_df, **input_dict)
    target_path_obj = legs_module.build_target_path(config_obj, panel_obj)
    if signal_date_ts not in target_path_obj.target_by_decision_dict:
        raise ValueError(f"The leg's path has no decision at {signal_date_str}.")
    evidence_dict = target_path_obj.evidence_by_decision_dict[signal_date_ts]
    # *** CRITICAL *** Unavailable evidence must stop the decision; it must not read as a closed gate or an empty book.
    if (not math.isfinite(evidence_dict["gate_close_float"]) or evidence_dict["gate_close_float"] <= 0
            or not math.isfinite(evidence_dict["gate_level_float"]) or evidence_dict["gate_level_float"] <= 0):
        raise ValueError("A momentum capsule leg requires a finite gate close and average at T.")
    vxn_reference_date_str = None
    if config_obj.vxn_symbol_str is not None:
        if evidence_dict["vxn_close_float"] is None or not math.isfinite(evidence_dict["vxn_close_float"]) or evidence_dict["vxn_close_float"] <= 0:
            raise ValueError("A momentum capsule leg requires a finite positive VXN as of T.")
        vxn_ser = input_dict["helper_close_dict"][config_obj.vxn_symbol_str]
        vxn_reference_date_str = vxn_ser[np.isfinite(vxn_ser) & (vxn_ser > 0.0)].index[-1].date().isoformat()
    if evidence_dict["member_count_int"] <= 0:
        raise ValueError("A momentum capsule leg requires PIT members at T.")
    if panel_obj.market_ret_vec is not None and not np.isfinite(panel_obj.market_ret_vec[-config_obj.residual_window_int:]).all():
        raise ValueError("A momentum capsule leg requires complete market returns through T.")
    own_book_ser = target_path_obj.book_by_decision_dict[signal_date_ts]
    if evidence_dict["gate_open_bool"]:
        if evidence_dict["finite_score_member_count_int"] < MIN_FINITE_SCORE_MEMBER_FRACTION_FLOAT * evidence_dict["member_count_int"]:
            raise ValueError("A momentum capsule leg scored too few members at T; its evidence is unavailable.")
        # Every gate-open decision since 2000 filled its book in every leg. A thin book here is a data fault, not a signal.
        if len(own_book_ser) != config_obj.max_positions_int:
            raise ValueError(f"A momentum capsule leg picked {len(own_book_ser)} of {config_obj.max_positions_int} names with its gate open; its evidence is unavailable.")
    target_weight_ser = target_path_obj.target_by_decision_dict[signal_date_ts]
    # *** CRITICAL *** Checked before the near-zero filter: abs(NaN) > 1e-12 is False, so a NaN weight would read as a sale.
    if not all(math.isfinite(float(weight_float)) and float(weight_float) > 0 for weight_float in target_weight_ser.to_numpy()):
        raise ValueError("A momentum capsule leg target must hold finite positive weights.")
    target_weight_dict = {str(asset_str): float(weight_float) for asset_str, weight_float in target_weight_ser.items() if abs(float(weight_float)) > 1e-12}
    return {
        "target_weight_dict": target_weight_dict,
        "own_book_weight_dict": {str(k): float(v) for k, v in own_book_ser.items()},
        "book_offset_int": int(target_path_obj.book_offset_by_decision_dict[signal_date_ts]),
        "evidence_dict": evidence_dict,
        "vxn_reference_date_str": vxn_reference_date_str,
    }


def build_momentum_legs_decision_plan(
    release_obj: LiveRelease, as_of_ts: datetime, pod_state_obj: PodState | None,
) -> DecisionPlan:
    from alpha.live import strategy_host

    validate_momentum_legs_release(release_obj)
    if not snapshot_module.is_snapshot_mode_enabled_bool():
        raise ValueError("Momentum capsule leg decisions require the frozen snapshot profile.")
    if pod_state_obj is not None and any(
            getattr(pod_state_obj, field_str) != getattr(release_obj, field_str)
            for field_str in ("pod_id_str", "user_id_str", "account_route_str")):
        raise ValueError("Momentum capsule leg account state must match the release identity.")
    signal_date_ts = scheduler_utils.get_latest_completed_month_swap_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("A momentum capsule leg requires a completed swap session.")
    target_timestamp_ts = scheduler_utils.build_target_execution_timestamp_ts(signal_date_ts, release_obj)
    if scheduler_utils.is_execution_window_expired_bool(release_obj.execution_policy_str, target_timestamp_ts, as_of_ts):
        raise ValueError("A momentum capsule leg cannot build a stale swap cycle.")
    signal_date_str = signal_date_ts.date().isoformat()
    snapshot_metadata_dict = snapshot_module.build_data_source_metadata_dict(release_obj.data_profile_str)
    if (snapshot_metadata_dict.get("norgate_snapshot_date_str") != signal_date_str
            or snapshot_metadata_dict.get("norgate_data_profile_str") != release_obj.data_profile_str
            or not snapshot_metadata_dict.get("norgate_manifest_hash_str")):
        raise ValueError("A momentum capsule leg requires the exact completed swap-session snapshot and manifest identity.")
    manifest_obj = snapshot_module.load_valid_snapshot_manifest(release_obj.data_profile_str, snapshot_date_str=signal_date_str)
    if manifest_obj.manifest_hash_str != snapshot_metadata_dict["norgate_manifest_hash_str"]:
        raise ValueError("Momentum capsule leg snapshot identity changed before the decision.")
    data_contract_dict = manifest_obj.manifest_dict["data_contract"]
    refusal_key_tuple = (release_obj.strategy_import_str, manifest_obj.manifest_hash_str)
    if refusal_key_tuple in _REFUSED_DECISION_BY_KEY_DICT:
        raise ValueError(_REFUSED_DECISION_BY_KEY_DICT[refusal_key_tuple])
    try:
        decision_dict = _replayed_decision_dict(release_obj, signal_date_ts, signal_date_str)
    except MemoryError as error_obj:
        raise RuntimeError("A momentum capsule leg ran out of memory replaying its path.") from error_obj
    except ValueError as error_obj:
        _REFUSED_DECISION_BY_KEY_DICT[refusal_key_tuple] = str(error_obj)
        raise
    target_weight_dict, evidence_dict = decision_dict["target_weight_dict"], decision_dict["evidence_dict"]
    unobserved_set = set(data_contract_dict["unobserved_member_list"])
    snapshot_metadata_dict.update({
        "strategy_family_str": "momentum_capsule_leg",
        "strategy_import_str": release_obj.strategy_import_str,
        "momentum_legs_sizing_contract_str": MOMENTUM_LEGS_SIZING_CONTRACT_STR,
        "research_quantity_basis_str": "Close_T",
        "book_form_str": "one_account_average_of_two_books",
        "book_offset_int": decision_dict["book_offset_int"],
        "own_book_weight_dict": decision_dict["own_book_weight_dict"],
        "gate_open_bool": bool(evidence_dict["gate_open_bool"]),
        "gate_close_float": float(evidence_dict["gate_close_float"]),
        "gate_level_float": float(evidence_dict["gate_level_float"]),
        "member_count_int": int(evidence_dict["member_count_int"]),
        "finite_score_member_count_int": int(evidence_dict["finite_score_member_count_int"]),
        "vxn_close_float": evidence_dict["vxn_close_float"],
        "vxn_reference_date_str": decision_dict["vxn_reference_date_str"],
        "label_map_hash_str": data_contract_dict["label_map_hash_str"],
        "label_snapshot_date_str": data_contract_dict["label_snapshot_date_str"],
        "label_classification_basis_str": data_contract_dict["label_classification_basis_str"],
        "label_observed_timestamp_str": data_contract_dict["label_observed_timestamp_str"],
        "sector_by_target_symbol_dict": {symbol_str: data_contract_dict["sector_by_symbol_dict"][symbol_str] for symbol_str in target_weight_dict},
        "target_without_unpadded_close_list": sorted(unobserved_set & set(target_weight_dict)),
    })
    final_metadata_dict = snapshot_module.build_data_source_metadata_dict(release_obj.data_profile_str)
    if any(final_metadata_dict.get(field_str) != snapshot_metadata_dict.get(field_str)
            for field_str in ("norgate_manifest_hash_str", "norgate_snapshot_date_str", "norgate_data_profile_str")):
        raise ValueError("Momentum capsule leg snapshot identity changed during the decision.")
    position_map_dict = {} if pod_state_obj is None else pod_state_obj.position_amount_map
    return strategy_host._build_full_target_weight_decision_plan(
        release_obj=release_obj, signal_date_ts=signal_date_ts.to_pydatetime(),
        decision_base_position_map_dict={str(asset_str): float(amount_float)
            for asset_str, amount_float in position_map_dict.items() if abs(float(amount_float)) > 1e-9},
        full_target_weight_map_dict=target_weight_dict,
        cash_reserve_weight_float=max(0.0, 1.0 - sum(target_weight_dict.values())),
        strategy_state_dict={},
        snapshot_metadata_dict=snapshot_metadata_dict)
