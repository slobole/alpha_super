"""Beyond 60/40 trend tilt (T1) live route: the third guarded next-open pod, wired exactly like CORE5.

Fixtures only: no Norgate, no broker. The decision itself is built by hand to the adapter's contract (frozen
trunc(NAV_T * w / Close_T) shares, the four-asset weight book, the receipt keys) so that every shared live module
can be exercised; the adapter's own decision logic has its own test file.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha import strategy_registry
from alpha.live import doctor, execution_engine, incubation, runner, scheduler_service, scheduler_utils, strategy_host
from alpha.live import beyond6040_broker as broker_module
from alpha.live import reference_compare
from alpha.live.beyond6040_adapter import (
    BEYOND6040_ASSET_TUPLE,
    BEYOND6040_CONTRACT_STR,
    BEYOND6040_STRATEGY_IMPORT_STR,
    build_beyond6040_decision_from_prices,
    is_beyond6040_decision_bool,
    require_beyond6040_position_match,
    validate_beyond6040_release,
    validated_beyond6040_target_share_dict,
)
from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live.daily_broker import DailyExecutionSnapshot
from alpha.live.daily_reconcile import DAILY_STRATEGY_IMPORT_TUPLE, FROZEN_SHARE_CONTRACT_SET, is_daily_reconcile_release_bool
from alpha.live.dashboard_v4.activity import EVENT_DICT
from alpha.live.dispatch_state import is_guarded_release_bool
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerSnapshot, DecisionPlan, LivePriceSnapshot, PodState
from alpha.live.order_clerk import BrokerAdapter, IBKRGatewayBrokerAdapter, StubBrokerAdapter
from alpha.live.release_manifest import (
    SUPPORTED_DATA_PROFILE_TUPLE,
    SUPPORTED_STRATEGY_IMPORT_TUPLE,
    parse_release_manifest,
    validate_release_list,
    validate_release_manifest,
)
from alpha.live.runner import BrokerAdapterResolver
from alpha.live.state_store_v2 import LiveStateStore
from alpha.strategy_registry import MaturityTier
from data.beyond6040_snapshot_contract import BEYOND6040_PROFILE_STR
import test_live_core5_wiring as core5_wiring_fixture
import test_live_doctor as doctor_fixture
from test_live_daily_reconcile import CLOSE_TS, daily_case  # noqa: F401  (fixture)

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "docs" / "live" / "release_templates"
TEMPLATE_PATH = TEMPLATE_DIR / "pod_taa_beyond_6040_trend_tilt_daily_moo.yaml.example"
CORE5_TEMPLATE_PATH = TEMPLATE_DIR / "pod_taa_adaptive_macro_core5_daily_moo.yaml.example"
ROUTE_STR = "DU_B6040_TEST"
SIGNAL_DATE_STR = "2026-10-02"  # a Friday close; the next open is Monday 2026-10-05 09:30 ET
CLOSE_PRICE_DICT = {"VTI": 300.0, "TLT": 90.0, "GLD": 250.0, "BIL": 91.5}
HELD_DICT = {"VTI": 100.0, "TLT": 200.0, "GLD": 50.0, "BIL": 300.0}
CASH_FLOAT = 5_000.0
WEIGHT_DICT = {"VTI": 0.30, "TLT": 0.25, "GLD": 0.20, "BIL": 0.25}


# ---------------------------------------------------------------- fixtures built to the contract
def _release(**change_dict):
    return replace(parse_release_manifest(str(TEMPLATE_PATH)), **{"mode_str": "paper", "enabled_bool": True, "account_route_str": ROUTE_STR, **change_dict})


def _nav_float(held_dict=HELD_DICT, cash_float=CASH_FLOAT):
    return cash_float + sum(float(share_float) * CLOSE_PRICE_DICT[asset_str] for asset_str, share_float in held_dict.items())


def _frozen_target_dict(held_dict=HELD_DICT, cash_float=CASH_FLOAT):
    nav_float = _nav_float(held_dict, cash_float)
    return {asset_str: float(int(nav_float * WEIGHT_DICT[asset_str] / CLOSE_PRICE_DICT[asset_str])) for asset_str in BEYOND6040_ASSET_TUPLE}


def _timestamps(release_obj):
    signal_date_ts = pd.Timestamp(SIGNAL_DATE_STR)
    return (scheduler_utils.build_signal_timestamp_ts(signal_date_ts, release_obj),
        scheduler_utils.build_submission_timestamp_ts(signal_date_ts, release_obj),
        scheduler_utils.build_target_execution_timestamp_ts(signal_date_ts, release_obj))


def _pod_state(release_obj, held_dict=HELD_DICT, cash_float=CASH_FLOAT, strategy_state_dict=None):
    signal_ts, _, _ = _timestamps(release_obj)
    return PodState(pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
        position_amount_map=dict(held_dict), cash_float=cash_float, total_value_float=_nav_float(held_dict, cash_float),
        strategy_state_dict=strategy_state_dict or {"beyond6040_state_version_int": 1, "last_signal_date_str": "", "beyond6040_execution_receipt_dict": {}},
        updated_timestamp_ts=signal_ts + timedelta(minutes=12), snapshot_stage_str="eod", snapshot_source_str="broker")


def _decision(release_obj, held_dict=HELD_DICT, cash_float=CASH_FLOAT, warning_code_list=()):
    """A frozen Close_T decision with the keys the adapter contract names; shares are trunc(NAV_T * w / Close_T)."""
    signal_ts, submission_ts, target_ts = _timestamps(release_obj)
    target_dict = _frozen_target_dict(held_dict, cash_float)
    candidate_receipt_dict = {"last_applied_event_fill_date_str": "2026-10-05", "last_applied_target_weight_map_dict": dict(WEIGHT_DICT),
        "last_applied_target_share_map_dict": dict(target_dict)}
    metadata_dict = {
        "strategy_family_str": "beyond6040_trend_tilt", "sizing_contract_str": BEYOND6040_CONTRACT_STR,
        "norgate_data_source_mode_str": "snapshot", "norgate_data_profile_str": BEYOND6040_PROFILE_STR,
        "norgate_snapshot_date_str": SIGNAL_DATE_STR, "norgate_manifest_hash_str": "fixture_hash",
        "signal_date_str": SIGNAL_DATE_STR, "rebalance_bool": True, "no_order_bool": False,
        "fixed_target_share_map_dict": dict(target_dict), "sizing_close_price_map_dict": dict(CLOSE_PRICE_DICT),
        "sizing_close_nav_float": _nav_float(held_dict, cash_float), "sizing_close_cash_float": cash_float,
        "base_strategy_state_dict": {"beyond6040_execution_receipt_dict": {}},
        "beyond6040_candidate_execution_receipt_dict": candidate_receipt_dict,
        "beyond6040_warning_code_list": list(warning_code_list), "beyond6040_warning_bool": bool(warning_code_list),
        "beyond6040_catch_up_bool": False,
    }
    strategy_state_dict = {"beyond6040_state_version_int": 1, "last_signal_date_str": SIGNAL_DATE_STR,
        "last_target_weight_map_dict": dict(WEIGHT_DICT), "last_rebalance_date_str": SIGNAL_DATE_STR, "beyond6040_execution_receipt_dict": {}}
    return DecisionPlan(release_obj.release_id_str, release_obj.user_id_str, release_obj.pod_id_str, release_obj.account_route_str,
        signal_ts, submission_ts, target_ts, release_obj.execution_policy_str, dict(held_dict), metadata_dict, strategy_state_dict,
        decision_book_type_str="full_target_weight_book", full_target_weight_map_dict=dict(WEIGHT_DICT), cash_reserve_weight_float=0.0,
        preserve_untouched_positions_bool=False, rebalance_omitted_assets_to_zero_bool=True)


class RecordingBroker(StubBrokerAdapter):
    """The stub broker, recording which strict account read the runner asks for."""

    def __init__(self):
        super().__init__()
        self.read_list = []

    def get_beyond6040_account_snapshot(self, account_route_str, *, include_portfolio_valuation_bool=False):
        self.read_list.append("beyond6040")
        return super().get_beyond6040_account_snapshot(account_route_str, include_portfolio_valuation_bool=include_portfolio_valuation_bool)

    def get_core5_account_snapshot(self, account_route_str, *, include_portfolio_valuation_bool=False):
        self.read_list.append("core5")
        return super().get_core5_account_snapshot(account_route_str, include_portfolio_valuation_bool=include_portfolio_valuation_bool)

    def get_flow_vote_account_snapshot(self, account_route_str, *, include_portfolio_valuation_bool=False):
        self.read_list.append("flow_vote")
        return super().get_flow_vote_account_snapshot(account_route_str, include_portfolio_valuation_bool=include_portfolio_valuation_bool)

    def get_capsule_account_snapshot(self, account_route_str, *, include_portfolio_valuation_bool=False):
        self.read_list.append("capsule")
        return super().get_capsule_account_snapshot(account_route_str, include_portfolio_valuation_bool=include_portfolio_valuation_bool)

    def get_live_price_snapshot(self, *argument_tuple, **argument_dict):
        pytest.fail("Beyond 60/40 must not require or fall back to live quotes")


def _store_and_broker(tmp_path, release_obj, state_obj, snapshot_ts):
    store_obj = LiveStateStore(str(tmp_path / "beyond6040.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(state_obj)
    broker_obj = RecordingBroker()
    broker_obj.seed_account_snapshot(release_obj.account_route_str, state_obj.cash_float, state_obj.total_value_float,
        state_obj.position_amount_map, snapshot_ts, session_mode_str=release_obj.mode_str)

    def fresh_daily_fn(account_route_str):
        snapshot_obj = broker_obj.get_account_snapshot(account_route_str)
        return DailyExecutionSnapshot(snapshot_obj, [], snapshot_obj.snapshot_timestamp_ts, snapshot_obj.snapshot_timestamp_ts)

    broker_obj.get_daily_execution_snapshot = fresh_daily_fn
    return store_obj, broker_obj


def _stub_validator(monkeypatch):
    """Route the engine's call through a recorder that returns the frozen targets; the adapter's own checks have their own tests."""
    call_list = []

    def validator_fn(decision_plan_obj, release_obj, actual_position_dict):
        call_list.append((decision_plan_obj.pod_id_str, release_obj.strategy_import_str, dict(actual_position_dict)))
        return dict(decision_plan_obj.snapshot_metadata_dict["fixed_target_share_map_dict"])

    monkeypatch.setattr(execution_engine, "validated_beyond6040_target_share_dict", validator_fn)
    return call_list


def _planned_cycle(tmp_path, monkeypatch, *, real_validator_bool=False, broker_held_dict=None):
    """A saved Close_T decision and a seeded broker; no VPlan yet."""
    release_obj = _release()
    state_obj = _pod_state(release_obj)
    decision_obj = _decision(release_obj)
    store_obj, broker_obj = _store_and_broker(tmp_path, release_obj, state_obj, decision_obj.submission_timestamp_ts)
    if broker_held_dict is not None:
        broker_obj._snapshot_map[ROUTE_STR] = replace(broker_obj._snapshot_map[ROUTE_STR], position_amount_map=dict(broker_held_dict))
    call_list = [] if real_validator_bool else _stub_validator(monkeypatch)
    decision_obj = store_obj.insert_decision_plan(decision_obj)
    option_dict = {"log_path_str": str(tmp_path / "ops.log"), "trace_enabled_bool": False}
    return SimpleNamespace(release_obj=release_obj, state_obj=state_obj, store_obj=store_obj, broker_obj=broker_obj, decision_obj=decision_obj,
        vplan_obj=None, option_dict=option_dict, result_dict=None, validator_call_list=call_list)


def _prepared_cycle(tmp_path, monkeypatch, *, real_validator_bool=False, broker_held_dict=None):
    case_obj = _planned_cycle(tmp_path, monkeypatch, real_validator_bool=real_validator_bool, broker_held_dict=broker_held_dict)
    case_obj.result_dict = runner.build_vplans(case_obj.store_obj, case_obj.broker_obj, case_obj.decision_obj.submission_timestamp_ts,
        "paper", **case_obj.option_dict)
    case_obj.vplan_obj = case_obj.store_obj.get_latest_vplan_for_pod(case_obj.release_obj.pod_id_str)
    return case_obj


def _submit(case_obj, as_of_ts=None):
    return runner.submit_ready_vplans(case_obj.store_obj, case_obj.broker_obj, as_of_ts or case_obj.decision_obj.submission_timestamp_ts,
        "paper", False, vplan_id_int=case_obj.vplan_obj.vplan_id_int, **case_obj.option_dict)


# ---------------------------------------------------------------- release rules
def test_template_is_a_disabled_dedicated_next_open_pod():
    template_obj = parse_release_manifest(str(TEMPLATE_PATH))
    assert (template_obj.enabled_bool, template_obj.mode_str, template_obj.execution_policy_str, template_obj.signal_clock_str) == (
        False, "live", "next_open_moo", "eod_snapshot_ready")
    assert (template_obj.data_profile_str, template_obj.pod_budget_fraction_float, template_obj.auto_submit_enabled_bool) == (BEYOND6040_PROFILE_STR, 1.0, True)
    assert (template_obj.broker_client_id_int, template_obj.account_route_str, template_obj.risk_profile_str) == (57, "YOUR_BEYOND6040_ACCOUNT", "beyond6040_long_only")
    assert (template_obj.release_id_str, template_obj.pod_id_str) == ("your_user.beyond6040_t1.daily_moo.v1", "pod_beyond6040_t1_01")
    validate_release_manifest(template_obj)
    release_obj = _release()
    validate_release_manifest(release_obj)
    assert is_daily_reconcile_release_bool(release_obj) and is_guarded_release_bool(release_obj)


@pytest.mark.parametrize("change_dict", [
    {"execution_policy_str": "next_close_moc"}, {"execution_policy_str": "next_open_market"}, {"signal_clock_str": "month_end_snapshot_ready"},
    {"data_profile_str": "norgate_eod_core5"}, {"pod_budget_fraction_float": 0.5}, {"params_dict": {"tilt_strength_float": 0.25}},
])
def test_the_frozen_route_refuses_any_other_shape(change_dict):
    with pytest.raises(ValueError):
        validate_beyond6040_release(_release(**change_dict))
    with pytest.raises(ValueError):
        validate_release_manifest(_release(**change_dict))


def test_the_profile_is_reserved_for_the_trend_tilt():
    core5_obj = parse_release_manifest(str(CORE5_TEMPLATE_PATH))
    with pytest.raises(ValueError):  # CORE5 refuses the foreign profile by its own rule first
        validate_release_manifest(replace(core5_obj, data_profile_str=BEYOND6040_PROFILE_STR))
    with pytest.raises(ValueError, match="reserved"):
        validate_release_manifest(replace(core5_obj, strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy", data_profile_str=BEYOND6040_PROFILE_STR))
    # The trend tilt may not borrow another pod's profile either.
    with pytest.raises(ValueError):
        validate_release_manifest(_release(data_profile_str="norgate_eod_flow_vote"))


@pytest.mark.parametrize("beyond6040_first_bool", [True, False])
def test_two_enabled_pods_cannot_share_the_account(beyond6040_first_bool):
    pod_obj = _release()
    other_obj = replace(parse_release_manifest(str(TEMPLATE_DIR / "pod_dv2_daily_moo.yaml.example")),
        mode_str="paper", enabled_bool=True, account_route_str=ROUTE_STR, broker_client_id_int=58)
    with pytest.raises(ValueError, match="Beyond 60/40 requires one dedicated account"):
        validate_release_list([pod_obj, other_obj] if beyond6040_first_bool else [other_obj, pod_obj])
    with pytest.raises(ValueError, match="Beyond 60/40 requires one dedicated account"):
        validate_release_list([pod_obj, replace(pod_obj, release_id_str="other.v1", pod_id_str="pod_other", broker_client_id_int=58)])
    # Its own account, another pod's own account: fine.
    validate_release_list([pod_obj, replace(other_obj, account_route_str="DU_OTHER")])


def test_incubation_paper_and_live_are_allowed():
    validate_release_manifest(_release(mode_str="incubation", account_route_str="SIM_B6040"))
    validate_release_manifest(_release(mode_str="paper"))
    validate_release_manifest(_release(mode_str="live", account_route_str="U_B6040_TEST", broker_port_int=7496))


def test_disabled_live_template_refuses_decisions():
    disabled_obj = parse_release_manifest(str(TEMPLATE_PATH))
    assert disabled_obj.mode_str == "live" and not disabled_obj.enabled_bool
    validate_release_manifest(disabled_obj)  # reviewable
    state_obj = _pod_state(replace(disabled_obj, account_route_str="U_B6040_TEST"))
    as_of_ts = _timestamps(disabled_obj)[0] + timedelta(minutes=30)
    metadata_dict = {"norgate_data_source_mode_str": "snapshot", "norgate_data_profile_str": BEYOND6040_PROFILE_STR,
        "norgate_snapshot_date_str": SIGNAL_DATE_STR, "norgate_manifest_hash_str": "fixture_hash"}
    with pytest.raises(ValueError):
        build_beyond6040_decision_from_prices(disabled_obj, as_of_ts, state_obj, pd.DataFrame(), metadata_dict)


def test_registry_manifest_host_and_heartbeat_agree():
    assert strategy_registry.tier_for(BEYOND6040_STRATEGY_IMPORT_STR) is MaturityTier.WIRED
    assert BEYOND6040_STRATEGY_IMPORT_STR in SUPPORTED_STRATEGY_IMPORT_TUPLE
    assert BEYOND6040_PROFILE_STR in SUPPORTED_DATA_PROFILE_TUPLE
    assert BEYOND6040_STRATEGY_IMPORT_STR in strategy_host.FULL_TARGET_DECISION_STRATEGY_IMPORT_SET
    assert BEYOND6040_STRATEGY_IMPORT_STR not in strategy_host.INCREMENTAL_DECISION_STRATEGY_IMPORT_SET
    assert scheduler_utils.DATA_PROFILE_HEARTBEAT_SYMBOL_MAP[BEYOND6040_PROFILE_STR] == "$SPX"
    assert BEYOND6040_ASSET_TUPLE == ("VTI", "TLT", "GLD", "BIL")


def test_the_host_dispatches_this_pod_to_the_beyond6040_builder(monkeypatch):
    release_obj = _release()
    marker_obj = object()
    seen_list = []

    def builder_fn(builder_release_obj, as_of_ts, pod_state_obj):
        seen_list.append((builder_release_obj.pod_id_str, as_of_ts, pod_state_obj))
        return marker_obj

    monkeypatch.setattr(strategy_host, "build_beyond6040_decision_plan", builder_fn)
    monkeypatch.setattr(strategy_host, "build_core5_decision_plan", lambda *_a, **_k: pytest.fail("CORE5 builder used for the trend tilt"))
    as_of_ts = datetime(2026, 10, 2, 20, 30, tzinfo=UTC)
    assert strategy_host.build_decision_plan_for_release(release_obj=release_obj, as_of_ts=as_of_ts, pod_state_obj=None) is marker_obj
    assert seen_list == [(release_obj.pod_id_str, as_of_ts, None)]


# ---------------------------------------------------------------- the daily plumbing
def test_daily_reconcile_dispatch_state_and_contract_sets_know_the_pod():
    release_obj = _release()
    assert is_daily_reconcile_release_bool(release_obj) and is_guarded_release_bool(release_obj)
    assert BEYOND6040_STRATEGY_IMPORT_STR in DAILY_STRATEGY_IMPORT_TUPLE
    assert DAILY_STRATEGY_IMPORT_TUPLE[-1] == CORE5_STRATEGY_IMPORT_STR  # the V4 demo reads CORE5 from the end
    assert BEYOND6040_CONTRACT_STR in FROZEN_SHARE_CONTRACT_SET
    monthly_obj = replace(release_obj, strategy_import_str="strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash")
    assert not is_daily_reconcile_release_bool(monthly_obj) and not is_guarded_release_bool(monthly_obj)
    assert is_beyond6040_decision_bool(_decision(release_obj)) and not is_beyond6040_decision_bool(None)


@pytest.mark.parametrize("achieved_bool", [True, False])
def test_the_daily_finalizer_commits_the_beyond6040_receipt_only_when_the_book_is_held(daily_case, achieved_bool):  # noqa: F811
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    prior_dict, candidate_dict = {"last_applied_event_fill_date_str": "prior"}, {"last_applied_event_fill_date_str": "new"}
    target_dict = dict(plan_obj.target_share_map)
    metadata_dict = {**decision_obj.snapshot_metadata_dict, "fixed_target_share_map_dict": target_dict,
        "base_strategy_state_dict": {"beyond6040_execution_receipt_dict": prior_dict}, "beyond6040_candidate_execution_receipt_dict": candidate_dict}
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str=? WHERE decision_plan_id_int=?",
            (json.dumps(metadata_dict), decision_obj.decision_plan_id_int))
    actual_dict = target_dict if achieved_bool else {"MSFT": 3.0}
    exception_list = [] if achieved_bool else [{"asset_str": "MSFT", "quantity_float": 3.0, "side_str": "SELL", "reason_str": "unfilled"}]
    store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        BrokerSnapshot(release_obj.account_route_str, CLOSE_TS, cash_float=500.0, total_value_float=1e5, net_liq_float=1e5, position_amount_map=actual_dict),
        "completed" if achieved_bool else "completed_with_exceptions", exception_list, CLOSE_TS)
    committed_dict = store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict
    assert committed_dict["beyond6040_execution_receipt_dict"] == (candidate_dict if achieved_bool else prior_dict)
    assert "flow_vote_execution_receipt_dict" not in committed_dict


def test_v4_activity_texts_mirror_the_core5_ones():
    for suffix_str in ("decision_blocked", "eod_source_untrusted", "pre_submit_account_changed"):
        assert EVENT_DICT[f"beyond6040_{suffix_str}"] == EVENT_DICT[f"core5_{suffix_str}"]
        type_str, state_str, title_str, _tab_str = EVENT_DICT[f"beyond6040_{suffix_str}"]
        assert (type_str, state_str) == ("alerts", "fail") and title_str.endswith(".") and len(title_str.split("\n")) == 1


def test_the_automatic_reference_refuses_the_pod_with_its_own_reason(tmp_path):
    release_obj = _release()
    support_dict = reference_compare.inspect_auto_reference_support_dict(release_obj)
    assert (support_dict["supported_bool"], support_dict["reason_str"], support_dict["missing_parameter_list"]) == (False, "beyond6040_accounting_bridge_required", [])
    with pytest.raises(ValueError, match="accounting bridge"):
        reference_compare.run_auto_reference_strategy(release_obj=release_obj, deployment_start_date_str="2026-01-02",
            reference_end_date_str="2026-10-02", deployment_initial_cash_float=100_000.0, output_dir_path_obj=tmp_path)
    # CORE5 keeps its own reason.
    assert reference_compare.inspect_auto_reference_support_dict(parse_release_manifest(str(CORE5_TEMPLATE_PATH)))["reason_str"] == "core5_accounting_bridge_required"


def test_every_broker_adapter_exposes_the_strict_read(monkeypatch):
    with pytest.raises(NotImplementedError, match="Beyond 60/40"):
        BrokerAdapter.get_beyond6040_account_snapshot(None, ROUTE_STR)
    # IBKR: delegates to the strict broker read with the adapter's socket client.
    seen_list = []
    sentinel_obj, snapshot_obj = object(), object()

    def strict_read_fn(socket_client_obj, account_route_str, include_portfolio_valuation_bool=False):
        seen_list.append((socket_client_obj, account_route_str, include_portfolio_valuation_bool))
        return snapshot_obj

    monkeypatch.setattr(broker_module, "get_beyond6040_account_snapshot", strict_read_fn)
    ibkr_obj = object.__new__(IBKRGatewayBrokerAdapter)
    ibkr_obj.socket_client_obj = sentinel_obj
    assert ibkr_obj.get_beyond6040_account_snapshot(ROUTE_STR, include_portfolio_valuation_bool=True) is snapshot_obj
    assert seen_list == [(sentinel_obj, ROUTE_STR, True)]
    # Stub and virtual (incubation) brokers: the plain account read.
    stub_obj = StubBrokerAdapter()
    stub_obj.seed_account_snapshot(ROUTE_STR, CASH_FLOAT, _nav_float(), HELD_DICT, datetime(2026, 10, 2, 20, 5, tzinfo=UTC), session_mode_str="paper")
    assert stub_obj.get_beyond6040_account_snapshot(ROUTE_STR).position_amount_map == HELD_DICT
    virtual_obj = object.__new__(incubation.IncubationBrokerAdapter)
    virtual_obj.get_account_snapshot = lambda account_route_str: ("virtual", account_route_str)
    assert virtual_obj.get_beyond6040_account_snapshot("SIM_B6040", include_portfolio_valuation_bool=True) == ("virtual", "SIM_B6040")


# ---------------------------------------------------------------- the runner, with fakes
def test_the_vplan_is_built_from_the_beyond6040_read_and_the_frozen_close_t(tmp_path, monkeypatch):
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    assert case_obj.result_dict["created_vplan_count_int"] == 1
    assert case_obj.broker_obj.read_list == ["beyond6040"]
    assert case_obj.validator_call_list == [(case_obj.release_obj.pod_id_str, BEYOND6040_STRATEGY_IMPORT_STR, HELD_DICT)]
    vplan_obj = case_obj.vplan_obj
    target_dict = case_obj.decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    assert vplan_obj.live_price_source_str == "beyond6040.frozen_close_t"
    assert vplan_obj.live_reference_price_map == CLOSE_PRICE_DICT
    assert set(vplan_obj.live_reference_source_map_dict.values()) == {"beyond6040.frozen_close_t"}
    assert vplan_obj.live_reference_snapshot_timestamp_ts == case_obj.decision_obj.signal_timestamp_ts
    assert vplan_obj.target_share_map == target_dict
    assert vplan_obj.order_delta_map == {asset_str: target_dict[asset_str] - HELD_DICT[asset_str] for asset_str in BEYOND6040_ASSET_TUPLE}
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    assert request_list and all(request_obj.broker_order_type_str == "MOO" and request_obj.unit_str == "shares" and request_obj.target_bool is False
        for request_obj in request_list)
    assert {request_obj.asset_str: request_obj.amount_float for request_obj in request_list} == {
        asset_str: delta_float for asset_str, delta_float in vplan_obj.order_delta_map.items() if delta_float != 0.0}


def test_the_execution_engine_asks_the_beyond6040_validator_only_for_this_pod(monkeypatch):
    release_obj = _release()
    decision_obj = _decision(release_obj)
    snapshot_obj = BrokerSnapshot(ROUTE_STR, decision_obj.submission_timestamp_ts, CASH_FLOAT, _nav_float(), position_amount_map=dict(HELD_DICT), net_liq_float=_nav_float())
    quote_obj = LivePriceSnapshot(account_route_str=ROUTE_STR, snapshot_timestamp_ts=decision_obj.signal_timestamp_ts, price_source_str="beyond6040.frozen_close_t",
        asset_reference_price_map=dict(CLOSE_PRICE_DICT))
    call_list = _stub_validator(monkeypatch)
    monkeypatch.setattr(execution_engine, "validated_core5_target_share_dict", lambda *_a: pytest.fail("CORE5 validator used for the trend tilt"))
    monkeypatch.setattr(execution_engine, "validated_flow_vote_target_share_dict", lambda *_a: pytest.fail("Flow Vote validator used for the trend tilt"))
    vplan_obj = execution_engine.build_vplan(release_obj=release_obj, decision_plan_obj=decision_obj, broker_snapshot_obj=snapshot_obj, live_price_snapshot_obj=quote_obj)
    assert call_list == [(release_obj.pod_id_str, BEYOND6040_STRATEGY_IMPORT_STR, HELD_DICT)]
    assert vplan_obj.target_share_map == decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    # A monthly TAA book keeps floor(weight * budget / quote): the frozen path is a no-op for it.
    monthly_obj = replace(release_obj, strategy_import_str="strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        execution_policy_str="next_month_first_open", data_profile_str="norgate_eod_etf_plus_vix_helper")
    monthly_plan_obj = replace(decision_obj, snapshot_metadata_dict={}, execution_policy_str="next_month_first_open")
    monthly_vplan_obj = execution_engine.build_vplan(release_obj=monthly_obj, decision_plan_obj=monthly_plan_obj, broker_snapshot_obj=snapshot_obj, live_price_snapshot_obj=quote_obj)
    assert len(call_list) == 1
    assert monthly_vplan_obj.target_share_map["VTI"] == float(int(WEIGHT_DICT["VTI"] * _nav_float() / CLOSE_PRICE_DICT["VTI"]))


def test_the_real_validator_accepts_the_consistent_frozen_book_and_refuses_a_tampered_one():
    release_obj = _release()
    decision_obj = _decision(release_obj)
    target_dict = decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    assert validated_beyond6040_target_share_dict(decision_obj, release_obj, dict(HELD_DICT)) == target_dict
    require_beyond6040_position_match(HELD_DICT, dict(HELD_DICT))
    tampered_dict = {**decision_obj.snapshot_metadata_dict, "fixed_target_share_map_dict": {**target_dict, "VTI": target_dict["VTI"] + 1.0}}
    with pytest.raises(ValueError):
        validated_beyond6040_target_share_dict(replace(decision_obj, snapshot_metadata_dict=tampered_dict), release_obj, dict(HELD_DICT))
    with pytest.raises(ValueError):
        validated_beyond6040_target_share_dict(decision_obj, release_obj, {**HELD_DICT, "VTI": HELD_DICT["VTI"] + 1.0})
    with pytest.raises(ValueError):
        require_beyond6040_position_match(HELD_DICT, {**HELD_DICT, "TLT": -1.0})  # long only
    with pytest.raises(ValueError):
        validated_beyond6040_target_share_dict(decision_obj, replace(release_obj, pod_id_str="pod_other"), dict(HELD_DICT))


def test_a_frozen_position_mismatch_blocks_the_decision_before_any_plan(tmp_path, monkeypatch):
    case_obj = _prepared_cycle(tmp_path, monkeypatch, broker_held_dict={**HELD_DICT, "GLD": HELD_DICT["GLD"] + 5.0})
    assert case_obj.result_dict["created_vplan_count_int"] == 0
    assert case_obj.result_dict["reason_count_map_dict"].get("beyond6040_frozen_position_mismatch") == 1
    assert case_obj.vplan_obj is None
    assert case_obj.store_obj.get_latest_decision_plan_for_pod(case_obj.release_obj.pod_id_str).status_str == "blocked"
    assert case_obj.validator_call_list == []


@pytest.mark.parametrize("drift_str", ["holdings", "orders", "account", "disabled"])
def test_the_final_account_read_blocks_the_send(tmp_path, monkeypatch, drift_str):
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    as_of_ts = case_obj.decision_obj.submission_timestamp_ts
    if drift_str == "disabled":
        case_obj.store_obj.upsert_release(replace(case_obj.release_obj, enabled_bool=False))
    else:
        original_fn = case_obj.broker_obj.get_beyond6040_account_snapshot

        def changed_fn(account_route_str, *argument_tuple, **argument_dict):
            snapshot_obj = original_fn(account_route_str, *argument_tuple, **argument_dict)
            if drift_str == "holdings":
                return replace(snapshot_obj, position_amount_map={**snapshot_obj.position_amount_map, "TLT": 201.0})
            if drift_str == "orders":
                return replace(snapshot_obj, open_order_id_list=["foreign-order"])
            return replace(snapshot_obj, account_route_str="DU_OTHER")

        monkeypatch.setattr(case_obj.broker_obj, "get_beyond6040_account_snapshot", changed_fn)
    assert _submit(case_obj, as_of_ts)["submitted_vplan_count_int"] == 0
    assert not case_obj.broker_obj.submitted_order_request_list
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str != "submitting"
    assert "beyond6040_pre_submit_account_changed" in Path(case_obj.option_dict["log_path_str"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("late_str", ["cutoff", "after_open"])
def test_at_or_after_the_cutoff_nothing_is_read_claimed_or_sent_and_every_leg_is_recorded_unsent(tmp_path, monkeypatch, late_str):
    """Open minus two minutes is the same deadline as CORE5's: the generic guard fires before the account read, and a late
    opening batch is parked for the daily close instead of becoming a later trade."""
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    target_ts = case_obj.decision_obj.target_execution_timestamp_ts
    as_of_ts = target_ts - timedelta(minutes=2) if late_str == "cutoff" else target_ts + timedelta(minutes=5)
    monkeypatch.setattr(case_obj.broker_obj, "get_beyond6040_account_snapshot", lambda *_a, **_k: pytest.fail("cutoff reached the account read"))
    monkeypatch.setattr(case_obj.store_obj, "claim_vplan_for_submission", lambda *_a: pytest.fail("cutoff claimed dispatch"))
    result_dict = _submit(case_obj, as_of_ts)
    assert result_dict["submitted_vplan_count_int"] == 0 and result_dict["reason_count_map_dict"].get("opening_dispatch_incomplete") == 1
    assert case_obj.broker_obj.submitted_order_request_list == []
    saved_decision_obj = case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int)
    failure_dict = saved_decision_obj.snapshot_metadata_dict["opening_dispatch_result_dict"]
    assert set(failure_dict["never_dispatched_request_key_list"]) == {
        request_obj.order_request_key_str for request_obj in build_broker_order_request_list_from_vplan(case_obj.vplan_obj)}
    assert saved_decision_obj.status_str == "submitted"  # the daily close settles it from holdings; no resend


def test_a_send_one_second_before_the_cutoff_still_goes_and_reads_the_account_first(tmp_path, monkeypatch):
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    phase_list = []
    original_read_fn, original_claim_fn = case_obj.broker_obj.get_beyond6040_account_snapshot, case_obj.store_obj.claim_vplan_for_submission

    def read_fn(account_route_str, *argument_tuple, **argument_dict):
        phase_list.append("account_read")
        assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "ready"
        return original_read_fn(account_route_str, *argument_tuple, **argument_dict)

    def claim_fn(vplan_id_int):
        phase_list.append("claim")
        return original_claim_fn(vplan_id_int)

    monkeypatch.setattr(case_obj.broker_obj, "get_beyond6040_account_snapshot", read_fn)
    monkeypatch.setattr(case_obj.store_obj, "claim_vplan_for_submission", claim_fn)
    as_of_ts = case_obj.decision_obj.target_execution_timestamp_ts - timedelta(minutes=2, seconds=1)
    assert _submit(case_obj, as_of_ts)["submitted_vplan_count_int"] == 1
    assert phase_list == ["account_read", "claim"]


def test_a_full_cycle_sends_moo_legs_settles_at_the_close_and_commits_the_receipt(tmp_path, monkeypatch):
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    release_obj, store_obj, broker_obj, decision_obj, vplan_obj = (case_obj.release_obj, case_obj.store_obj, case_obj.broker_obj,
        case_obj.decision_obj, case_obj.vplan_obj)
    target_dict = decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    prior_state_dict = dict(store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 1
    open_ts = decision_obj.target_execution_timestamp_ts
    request_list = broker_obj.submitted_order_request_list
    assert request_list and all(request_obj.broker_order_type_str == "MOO" and request_obj.unit_str == "shares"
        and request_obj.submission_deadline_timestamp_str == (open_ts - timedelta(minutes=2)).isoformat() for request_obj in request_list)
    assert {request_obj.asset_str: request_obj.amount_float for request_obj in request_list} == {
        asset_str: target_dict[asset_str] - HELD_DICT[asset_str] for asset_str in BEYOND6040_ASSET_TUPLE if target_dict[asset_str] != HELD_DICT[asset_str]}
    assert all(request_obj.amount_float >= 0.0 or HELD_DICT[request_obj.asset_str] + request_obj.amount_float >= 0.0 for request_obj in request_list)  # long only
    close_ts = scheduler_utils.get_session_close_timestamp_ts(pd.Timestamp("2026-10-05"), "XNYS")
    for as_of_ts, completed_int in ((open_ts + timedelta(minutes=5), 0), (close_ts, 1)):
        broker_obj._snapshot_map[ROUTE_STR] = replace(broker_obj.get_account_snapshot(ROUTE_STR), snapshot_timestamp_ts=as_of_ts)
        assert runner.post_execution_reconcile(store_obj, broker_obj, as_of_ts, "paper", **case_obj.option_dict)["completed_vplan_count_int"] == completed_int
        if completed_int == 0:
            assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == prior_state_dict  # nothing settles before the close
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "completed"
    assert broker_obj.get_account_snapshot(ROUTE_STR).position_amount_map == target_dict
    committed_dict = store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict
    assert committed_dict["beyond6040_execution_receipt_dict"] == decision_obj.snapshot_metadata_dict["beyond6040_candidate_execution_receipt_dict"]
    assert committed_dict["last_signal_date_str"] == SIGNAL_DATE_STR


def test_a_quote_or_a_changed_nav_cannot_resize_the_frozen_targets(tmp_path, monkeypatch):
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    original_target_dict = dict(case_obj.vplan_obj.target_share_map)
    original_snapshot_obj = case_obj.broker_obj.get_account_snapshot(ROUTE_STR)
    case_obj.broker_obj._snapshot_map[ROUTE_STR] = replace(original_snapshot_obj, cash_float=original_snapshot_obj.cash_float + 1_000_000.0,
        net_liq_float=original_snapshot_obj.net_liq_float * 7, total_value_float=original_snapshot_obj.total_value_float * 7)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 1
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).target_share_map == original_target_dict
    assert {request_obj.asset_str: request_obj.amount_float for request_obj in case_obj.broker_obj.submitted_order_request_list} == {
        asset_str: delta_float for asset_str, delta_float in case_obj.vplan_obj.order_delta_map.items() if delta_float != 0.0}


def test_the_state_store_refuses_a_plan_or_a_claim_for_an_inactive_decision(tmp_path, monkeypatch):
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    case_obj.store_obj.mark_decision_plan_status(case_obj.decision_obj.decision_plan_id_int, "expired")
    assert case_obj.store_obj.claim_vplan_for_submission(case_obj.vplan_obj.vplan_id_int) is False
    with pytest.raises(ValueError, match="active planned decision"):
        case_obj.store_obj.insert_vplan(replace(case_obj.vplan_obj, vplan_id_int=None))


@pytest.mark.parametrize("open_order_bool", [False, True])
def test_the_eod_capture_uses_the_strict_read_and_refuses_an_account_with_open_orders(tmp_path, open_order_bool):
    release_obj = _release()
    store_obj = LiveStateStore(str(tmp_path / "beyond6040.sqlite3"))
    store_obj.upsert_release(release_obj)
    broker_obj = RecordingBroker()
    as_of_ts = scheduler_utils.get_session_close_timestamp_ts(pd.Timestamp(SIGNAL_DATE_STR), "XNYS") + timedelta(minutes=20)  # a past close
    broker_obj.seed_account_snapshot(ROUTE_STR, CASH_FLOAT, _nav_float(), HELD_DICT, as_of_ts, session_mode_str="paper")
    if open_order_bool:
        broker_obj._snapshot_map[ROUTE_STR] = replace(broker_obj._snapshot_map[ROUTE_STR], open_order_id_list=["9"])
    log_path_str = str(tmp_path / "eod.log")
    result_dict = runner.eod_snapshot(store_obj, broker_obj, as_of_ts, "paper", log_path_str=log_path_str, trace_enabled_bool=False)
    assert broker_obj.read_list == ["beyond6040"]
    state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    if open_order_bool:
        assert result_dict["eod_snapshot_count_int"] == 0
        assert state_obj is None or state_obj.snapshot_stage_str != "eod"
        assert "beyond6040_eod_source_untrusted" in Path(log_path_str).read_text(encoding="utf-8")
    else:
        assert result_dict["eod_snapshot_count_int"] == 1
        assert (state_obj.snapshot_stage_str, state_obj.snapshot_source_str) == ("eod", "broker")
        assert state_obj.position_amount_map == HELD_DICT and state_obj.cash_float == CASH_FLOAT


def _releases_root(tmp_path):
    releases_root_path_obj = tmp_path / "releases"
    releases_root_path_obj.mkdir()
    (releases_root_path_obj / "beyond6040.yaml").write_text(TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("mode: live", "mode: paper").replace("enabled_bool: false", "enabled_bool: true")
        .replace("YOUR_BEYOND6040_ACCOUNT", ROUTE_STR), encoding="utf-8")
    return releases_root_path_obj


def test_a_decision_error_is_blocked_not_raised(tmp_path, monkeypatch):
    # Direct mode: the adapter refuses to decide without the qualified snapshot, and the runner records that as a block.
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    monkeypatch.setattr(scheduler_utils, "evaluate_build_gate_dict", lambda *argument_tuple, **argument_dict: {"due_bool": True})
    as_of_ts = scheduler_utils.get_session_close_timestamp_ts(pd.Timestamp(SIGNAL_DATE_STR), "XNYS") + timedelta(minutes=30)
    result_dict = runner.build_decision_plans(LiveStateStore(str(tmp_path / "beyond6040.sqlite3")), as_of_ts, str(_releases_root(tmp_path)), "paper",
        auto_sync_norgate_snapshots_bool=False, log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert result_dict["created_decision_plan_count_int"] == 0
    assert result_dict["reason_count_map_dict"].get("beyond6040_decision_blocked") == 1


def test_decision_warnings_are_logged_under_the_beyond6040_key(tmp_path, monkeypatch):
    release_obj = _release()
    warned_plan_obj = _decision(release_obj, warning_code_list=["beyond6040_strategy_state_cache_gap"])
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    monkeypatch.setattr(scheduler_utils, "evaluate_build_gate_dict", lambda *argument_tuple, **argument_dict: {"due_bool": True})
    monkeypatch.setattr(strategy_host, "build_decision_plan_for_release", lambda **_kwargs: warned_plan_obj)
    store_obj = LiveStateStore(str(tmp_path / "beyond6040.sqlite3"))
    log_path_str = str(tmp_path / "ops.log")
    result_dict = runner.build_decision_plans(store_obj, warned_plan_obj.signal_timestamp_ts + timedelta(minutes=30), str(_releases_root(tmp_path)), "paper",
        auto_sync_norgate_snapshots_bool=False, log_path_str=log_path_str, trace_enabled_bool=False)
    assert result_dict["created_decision_plan_count_int"] == 1
    log_str = Path(log_path_str).read_text(encoding="utf-8")
    assert "decision_plan_warning" in log_str and "beyond6040_strategy_state_cache_gap" in log_str
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).snapshot_metadata_dict["beyond6040_warning_code_list"] == ["beyond6040_strategy_state_cache_gap"]


# ---------------------------------------------------------------- the doctor
def test_the_doctor_previews_the_frozen_close_t_source_without_quotes_or_orders(tmp_path, monkeypatch):
    release_obj = _release()
    state_obj = _pod_state(release_obj)
    decision_obj = _decision(release_obj)
    release_path = core5_wiring_fixture._write_release_yaml(tmp_path, release_obj)
    store_obj, broker_obj = _store_and_broker(tmp_path, release_obj, state_obj, decision_obj.submission_timestamp_ts)
    _stub_validator(monkeypatch)
    monkeypatch.setattr(doctor.scheduler_utils, "evaluate_build_gate_dict", lambda *_args: {"due_bool": True, "reason_code_str": "snapshot_ready"})
    monkeypatch.setattr(doctor.strategy_host, "build_decision_plan_for_release", lambda **_kwargs: decision_obj)
    monkeypatch.delenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", raising=False)
    monkeypatch.delenv("NORGATE_SNAPSHOT_ROOT", raising=False)
    captured_vplan_list = []
    real_builder_fn = doctor.build_vplan

    def capture_builder(**kwargs_dict):
        vplan_obj = real_builder_fn(**kwargs_dict)
        captured_vplan_list.append(vplan_obj)
        return vplan_obj

    monkeypatch.setattr(doctor, "build_vplan", capture_builder)
    db_path = tmp_path / "beyond6040.sqlite3"
    before_bytes = db_path.read_bytes()
    report_dict = doctor.compute_doctor_verdict(releases_root_path_str=str(release_path), env_mode_str="paper",
        as_of_ts=decision_obj.submission_timestamp_ts, pod_id_str=release_obj.pod_id_str,
        db_path_str=str(tmp_path / "not_existing.sqlite3"), pod_state_loader_fn=lambda *_args: state_obj,
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=broker_obj),
        norgate_snapshot_sync_checker_fn=lambda **_kwargs: {"status_str": "ready", "reason_code_str": "local_snapshot_ready"},
        trace_enabled_bool=False, **doctor_fixture._doctor_config_kwargs(tmp_path, release_path))
    assert report_dict["overall_verdict_str"] == "PASS"
    assert db_path.read_bytes() == before_bytes
    assert not broker_obj.submitted_order_request_list
    preview_obj, = captured_vplan_list
    assert preview_obj.live_price_source_str == "beyond6040.frozen_close_t"
    assert set(preview_obj.live_reference_source_map_dict.values()) == {"beyond6040.frozen_close_t"}
    assert all(row_obj.live_reference_source_str == "beyond6040.frozen_close_t" for row_obj in preview_obj.vplan_row_list)
    assert preview_obj.live_reference_price_map == CLOSE_PRICE_DICT
    assert preview_obj.target_share_map == decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    assert "core5" not in broker_obj.read_list and "flow_vote" not in broker_obj.read_list


# ---------------------------------------------------------------- recovery and restart (modeled on the CORE5 suites)
def _freeze_dispatch_clock(monkeypatch, clock_ts):
    """record_dispatch_failure also checks the wall clock against the deadline; pin it to the fixture's session."""
    class DispatchClock:
        @classmethod
        def now(cls, timezone_obj):
            return clock_ts.astimezone(timezone_obj)

    monkeypatch.setattr("alpha.live.dispatch_state.datetime", DispatchClock)


def _restarted_store(case_obj):
    return LiveStateStore(case_obj.store_obj.db_path_str)


def _route_scheduler_to_this_release(monkeypatch, release_obj):
    monkeypatch.setattr(scheduler_service, "_load_release_list_and_sync", lambda *_argument_tuple, **_argument_dict: [release_obj])
    monkeypatch.setattr(scheduler_utils, "evaluate_build_gate_dict", lambda *_argument_tuple, **_argument_dict: {"due_bool": True})


def _set_broker_clock(broker_obj, as_of_ts):
    broker_obj._snapshot_map[ROUTE_STR] = replace(broker_obj.get_account_snapshot(ROUTE_STR), snapshot_timestamp_ts=as_of_ts)


def _session_close_ts(case_obj):
    return scheduler_utils.get_session_close_timestamp_ts(case_obj.decision_obj.target_execution_timestamp_ts.date(), "XNYS")


def test_a_crash_after_the_claim_is_reconciled_from_holdings_and_never_resent(tmp_path, monkeypatch):
    """(a) The process claims the batch, then dies before any ACK. Every later pass routes the cycle to the daily close."""
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    assert case_obj.store_obj.claim_vplan_for_submission(case_obj.vplan_obj.vplan_id_int)  # the crash happens here
    store_obj = _restarted_store(case_obj)
    assert store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "submitting"
    after_open_ts = case_obj.decision_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    for as_of_ts in (case_obj.decision_obj.submission_timestamp_ts, after_open_ts):
        result_dict = runner.submit_ready_vplans(store_obj, case_obj.broker_obj, as_of_ts, "paper", False,
            vplan_id_int=case_obj.vplan_obj.vplan_id_int, **case_obj.option_dict)
        assert result_dict["submitted_vplan_count_int"] == 0
    # Auto-submit passes skip a claimed batch too.
    assert runner.submit_ready_vplans(store_obj, case_obj.broker_obj, case_obj.decision_obj.submission_timestamp_ts, "paper", False,
        **case_obj.option_dict)["submitted_vplan_count_int"] == 0
    assert case_obj.broker_obj.submitted_order_request_list == []
    _route_scheduler_to_this_release(monkeypatch, case_obj.release_obj)
    schedule_obj = scheduler_service.get_scheduler_decision(store_obj, after_open_ts, "unused", "paper")
    assert (schedule_obj.next_phase_str, schedule_obj.reason_code_str) == ("post_execution_reconcile", "ready_to_reconcile")
    # The daily close settles from actual holdings. The opening batch is never re-sent; the shared daily rule may only sell the
    # BIL excess over its frozen target, once, as one claimed MKT order during the session (the same rule CORE5 runs under).
    _set_broker_clock(case_obj.broker_obj, after_open_ts)
    assert runner.post_execution_reconcile(store_obj, case_obj.broker_obj, after_open_ts, "paper", **case_obj.option_dict)["completed_vplan_count_int"] == 0
    target_dict = case_obj.decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    completion_list = case_obj.broker_obj.submitted_order_request_list
    assert [(request_obj.asset_str, request_obj.broker_order_type_str, request_obj.amount_float) for request_obj in completion_list] == [
        ("BIL", "MKT", target_dict["BIL"] - HELD_DICT["BIL"])]
    assert completion_list[0].order_request_key_str.endswith(":daily-completion:BIL")
    assert not any(request_obj.broker_order_type_str == "MOO" for request_obj in completion_list)
    _set_broker_clock(case_obj.broker_obj, after_open_ts + timedelta(minutes=5))
    runner.post_execution_reconcile(store_obj, case_obj.broker_obj, after_open_ts + timedelta(minutes=5), "paper", **case_obj.option_dict)
    assert len(case_obj.broker_obj.submitted_order_request_list) == 1  # one attempt per decision and asset
    close_ts = _session_close_ts(case_obj) + timedelta(minutes=1)
    _set_broker_clock(case_obj.broker_obj, close_ts)
    assert runner.post_execution_reconcile(store_obj, case_obj.broker_obj, close_ts, "paper", **case_obj.option_dict)["completed_vplan_count_int"] == 1
    assert store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int).status_str == "completed_with_exceptions"
    committed_state_obj = store_obj.get_pod_state(case_obj.release_obj.pod_id_str)
    assert committed_state_obj.position_amount_map == {**HELD_DICT, "BIL": target_dict["BIL"]}
    assert committed_state_obj.strategy_state_dict["beyond6040_execution_receipt_dict"] == {}  # the book was not reached: caught up next
    assert len(case_obj.broker_obj.submitted_order_request_list) == 1
    # A second close pass after another restart changes nothing.
    assert runner.post_execution_reconcile(LiveStateStore(store_obj.db_path_str), case_obj.broker_obj, close_ts, "paper",
        **case_obj.option_dict)["completed_vplan_count_int"] == 0


def test_a_restart_after_the_claim_keeps_the_pending_cycle(tmp_path, monkeypatch):
    """(b) Expiry and later submit passes must neither expire nor resend a claimed batch; the scheduler sends it to reconcile."""
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    assert case_obj.store_obj.claim_vplan_for_submission(case_obj.vplan_obj.vplan_id_int)
    store_obj = _restarted_store(case_obj)
    after_open_ts = case_obj.decision_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    assert store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int).status_str == "vplan_ready"
    expiry_dict = runner.expire_stale_decision_plans(store_obj, after_open_ts, "unused", "paper", **case_obj.option_dict)
    assert expiry_dict["expired_decision_plan_count_int"] == 0
    assert runner.submit_ready_vplans(store_obj, case_obj.broker_obj, after_open_ts, "paper", False,
        vplan_id_int=case_obj.vplan_obj.vplan_id_int, **case_obj.option_dict)["submitted_vplan_count_int"] == 0
    assert case_obj.broker_obj.submitted_order_request_list == []
    assert store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int).status_str == "vplan_ready"
    assert store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "submitting"
    assert store_obj.get_pod_state(case_obj.release_obj.pod_id_str).strategy_state_dict == case_obj.state_obj.strategy_state_dict
    _route_scheduler_to_this_release(monkeypatch, case_obj.release_obj)
    schedule_obj = scheduler_service.get_scheduler_decision(store_obj, after_open_ts, "unused", "paper")
    assert (schedule_obj.next_phase_str, schedule_obj.reason_code_str) == ("post_execution_reconcile", "ready_to_reconcile")


def test_an_unresolved_prior_cycle_is_counted_under_the_beyond6040_reason(tmp_path, monkeypatch):
    """(d) Past its window, a decision whose batch is no longer just ready cannot be expired: it is counted, not changed."""
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    assert case_obj.store_obj.claim_vplan_for_submission(case_obj.vplan_obj.vplan_id_int)
    after_open_ts = case_obj.decision_obj.target_execution_timestamp_ts + timedelta(minutes=1)
    expiry_dict = runner.expire_stale_decision_plans(case_obj.store_obj, after_open_ts, "unused", "paper", **case_obj.option_dict)
    assert expiry_dict["expired_decision_plan_count_int"] == 0
    assert expiry_dict["reason_count_map_dict"] == {"beyond6040_prior_cycle_unresolved": 1}
    assert case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int).status_str == "vplan_ready"
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "submitting"
    # Before the window closes, nothing is counted at all.
    before_open_ts = case_obj.decision_obj.target_execution_timestamp_ts - timedelta(seconds=1)
    assert runner.expire_stale_decision_plans(case_obj.store_obj, before_open_ts, "unused", "paper",
        **case_obj.option_dict)["reason_count_map_dict"] == {}


def test_an_expired_window_expires_the_unsubmitted_daily_cycle(tmp_path, monkeypatch):
    """(c) A ready, unclaimed batch past the open: decision and VPlan both expire, logged as submission_window_expired."""
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    target_ts = case_obj.decision_obj.target_execution_timestamp_ts
    assert runner.expire_stale_decision_plans(case_obj.store_obj, target_ts - timedelta(seconds=1), "unused", "paper",
        **case_obj.option_dict)["expired_decision_plan_count_int"] == 0
    expiry_dict = runner.expire_stale_decision_plans(case_obj.store_obj, target_ts, "unused", "paper", **case_obj.option_dict)
    assert expiry_dict["expired_decision_plan_count_int"] == 1
    assert expiry_dict["reason_count_map_dict"] == {"submission_window_expired": 1}
    assert case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int).status_str == "expired"
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "expired"
    log_str = Path(case_obj.option_dict["log_path_str"]).read_text(encoding="utf-8")
    assert "decision_plan_expired" in log_str and "submission_window_expired" in log_str
    assert runner.submit_ready_vplans(case_obj.store_obj, case_obj.broker_obj, target_ts + timedelta(minutes=1), "paper", False,
        **case_obj.option_dict)["submitted_vplan_count_int"] == 0
    assert case_obj.broker_obj.submitted_order_request_list == []
    assert case_obj.store_obj.get_pod_state(case_obj.release_obj.pod_id_str).strategy_state_dict == case_obj.state_obj.strategy_state_dict


def test_a_plan_built_after_the_cutoff_is_parked_without_an_account_read(tmp_path, monkeypatch):
    """(c) The build side of a missed window: at open minus two minutes the decision is blocked, never planned late."""
    case_obj = _planned_cycle(tmp_path, monkeypatch)
    cutoff_ts = case_obj.decision_obj.target_execution_timestamp_ts - timedelta(minutes=2)
    result_dict = runner.build_vplans(case_obj.store_obj, case_obj.broker_obj, cutoff_ts, "paper", **case_obj.option_dict)
    assert result_dict["created_vplan_count_int"] == 0
    assert result_dict["reason_count_map_dict"].get("opening_build_parked") == 1
    assert case_obj.broker_obj.read_list == []
    assert case_obj.store_obj.get_latest_vplan_for_pod(case_obj.release_obj.pod_id_str) is None
    assert case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int).status_str == "blocked"


@pytest.mark.parametrize("error_obj", [TimeoutError("synthetic account read timeout"), ConnectionError("socket disconnected")])
def test_a_transient_account_error_at_build_is_retried_not_committed(tmp_path, monkeypatch, error_obj):
    """(e) build_vplans: a transient read error leaves the decision planned with no plan; the next pass builds it."""
    case_obj = _planned_cycle(tmp_path, monkeypatch)
    original_fn = case_obj.broker_obj.get_beyond6040_account_snapshot

    def failing_fn(*_argument_tuple, **_argument_dict):
        raise error_obj

    monkeypatch.setattr(case_obj.broker_obj, "get_beyond6040_account_snapshot", failing_fn)
    submission_ts = case_obj.decision_obj.submission_timestamp_ts
    first_dict = runner.build_vplans(case_obj.store_obj, case_obj.broker_obj, submission_ts, "paper", **case_obj.option_dict)
    assert first_dict["created_vplan_count_int"] == 0
    assert first_dict["reason_count_map_dict"].get("build_retry_pending") == 1
    assert case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int).status_str == "planned"
    assert case_obj.store_obj.get_latest_vplan_for_pod(case_obj.release_obj.pod_id_str) is None
    assert case_obj.store_obj.get_pod_state(case_obj.release_obj.pod_id_str).strategy_state_dict == case_obj.state_obj.strategy_state_dict
    monkeypatch.setattr(case_obj.broker_obj, "get_beyond6040_account_snapshot", original_fn)
    second_dict = runner.build_vplans(case_obj.store_obj, case_obj.broker_obj, submission_ts + timedelta(seconds=30), "paper", **case_obj.option_dict)
    assert second_dict["created_vplan_count_int"] == 1
    assert case_obj.store_obj.get_latest_vplan_for_pod(case_obj.release_obj.pod_id_str).target_share_map == _frozen_target_dict()


def test_a_non_transient_account_error_at_build_parks_the_cycle(tmp_path, monkeypatch):
    case_obj = _planned_cycle(tmp_path, monkeypatch)
    monkeypatch.setattr(case_obj.broker_obj, "get_beyond6040_account_snapshot",
        lambda *_argument_tuple, **_argument_dict: (_ for _ in ()).throw(ValueError("Beyond 60/40 found an unsupported account position: QQQ.")))
    result_dict = runner.build_vplans(case_obj.store_obj, case_obj.broker_obj, case_obj.decision_obj.submission_timestamp_ts, "paper",
        **case_obj.option_dict)
    assert result_dict["created_vplan_count_int"] == 0 and result_dict["reason_count_map_dict"].get("opening_build_parked") == 1
    assert case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int).status_str == "blocked"


def test_a_transient_account_error_before_submit_is_retried_not_committed(tmp_path, monkeypatch):
    """(e) submit: a transient final account read keeps the batch ready and unclaimed; the next pass sends every leg once."""
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    submission_ts = case_obj.decision_obj.submission_timestamp_ts
    _freeze_dispatch_clock(monkeypatch, submission_ts)
    original_read_fn, original_claim_fn = case_obj.broker_obj.get_beyond6040_account_snapshot, case_obj.store_obj.claim_vplan_for_submission

    def failing_fn(*_argument_tuple, **_argument_dict):
        raise TimeoutError("synthetic transient before placeOrder")

    monkeypatch.setattr(case_obj.broker_obj, "get_beyond6040_account_snapshot", failing_fn)
    monkeypatch.setattr(case_obj.store_obj, "claim_vplan_for_submission", lambda *_argument_tuple: pytest.fail("a failed preflight claimed the batch"))
    first_dict = _submit(case_obj)
    assert first_dict["submitted_vplan_count_int"] == 0 and first_dict["reason_count_map_dict"].get("dispatch_retry_pending") == 1
    saved_decision_obj = case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int)
    assert saved_decision_obj.snapshot_metadata_dict["opening_dispatch_result_dict"]["reason_code_str"] == "dispatch_retry_pending"
    assert saved_decision_obj.status_str == "vplan_ready"
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "ready"
    assert case_obj.broker_obj.submitted_order_request_list == []
    assert case_obj.store_obj.get_pod_state(case_obj.release_obj.pod_id_str).strategy_state_dict == case_obj.state_obj.strategy_state_dict
    monkeypatch.setattr(case_obj.broker_obj, "get_beyond6040_account_snapshot", original_read_fn)
    monkeypatch.setattr(case_obj.store_obj, "claim_vplan_for_submission", original_claim_fn)
    second_dict = _submit(case_obj, submission_ts + timedelta(seconds=30))
    assert second_dict["submitted_vplan_count_int"] == 1
    assert ([replace(request_obj, submission_deadline_timestamp_str=None) for request_obj in case_obj.broker_obj.submitted_order_request_list]
        == build_broker_order_request_list_from_vplan(case_obj.vplan_obj))


def test_a_transient_error_is_not_retried_once_the_wall_clock_passes_the_cutoff(tmp_path, monkeypatch):
    case_obj = _prepared_cycle(tmp_path, monkeypatch)
    _freeze_dispatch_clock(monkeypatch, case_obj.decision_obj.target_execution_timestamp_ts - timedelta(minutes=2))
    monkeypatch.setattr(case_obj.broker_obj, "get_beyond6040_account_snapshot",
        lambda *_argument_tuple, **_argument_dict: (_ for _ in ()).throw(TimeoutError("late transient")))
    result_dict = _submit(case_obj)
    assert result_dict["submitted_vplan_count_int"] == 0 and result_dict["reason_count_map_dict"].get("opening_dispatch_incomplete") == 1
    saved_decision_obj = case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int)
    assert saved_decision_obj.snapshot_metadata_dict["opening_dispatch_result_dict"]["reason_code_str"] == "opening_dispatch_incomplete"
    assert saved_decision_obj.status_str == "submitted"  # parked for the daily close; no resend
    assert case_obj.broker_obj.submitted_order_request_list == []
