"""The live adapter with liquidity sizing: weights, parking, fail-closed boundaries, and no change to the 1/N identities. No Norgate, no broker."""

import sys
from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from test_live_mr_capsule_host import _state

import strategies.mr_capsule as capsule_package
from alpha.live import mr_capsule_adapter as adapter_module, runner as runner_module, strategy_host
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan, validate_mr_capsule_execution_contract
from alpha.live.models import BrokerSnapshot, DecisionPlan, LivePriceSnapshot
from alpha.live.release_manifest import parse_release_manifest, validate_release_manifest
from alpha.live.state_store_v2 import LiveStateStore
from strategies.hpi import stateful_long as hpi_module
from strategies.mr_capsule import dv2_vix_gated, dv2_vix_gated_liq, hpi_vote_vix_gated, hpi_vote_vix_gated_liq, vix_stress_gate


SIGNAL_DATE_TS = pd.Timestamp("2024-01-12")
AS_OF_TS = datetime(2024, 1, 12, 18, tzinfo=ZoneInfo("America/New_York"))
POD_LIST = ["dv2_vix_gated", "hpi_vote_vix_gated"]


def _release(pod_str):
    return parse_release_manifest(f"docs/live/release_templates/pod_mr_{pod_str}_liq_bil_daily_moo.yaml.example")


def _inputs(monkeypatch, release_obj, row_int=300, turnover_dict=None):
    """AAA and BBB are candidates every day; AAA ranks first. sqrt(4e6) = 2,000 and sqrt(1e6) = 1,000, so the typical candidate is 1,500."""
    pricing_index = pd.bdate_range(end=SIGNAL_DATE_TS, periods=row_int)
    turnover_dict = {"AAA": 4e6, "BBB": 1e6} if turnover_dict is None else turnover_dict
    pricing_data_df = pd.DataFrame({
        (symbol_str, field_str): value_float
        for symbol_str in ("AAA", "BBB", "BIL", "SPMO", "$SPX", "$SPXTR")
        for field_str, value_float in {
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 10000.0, "Dividend": 0.0, "ibs_value_ser": 0.0, "rsi2_value_ser": 0.0,
        }.items()
    }, index=pricing_index)
    for symbol_str, turnover_float in turnover_dict.items():
        pricing_data_df[(symbol_str, "Turnover")] = turnover_float
    universe_df = pd.DataFrame(1, index=pricing_index, columns=["AAA", "BBB"])
    vix_close_ser = pd.Series(10.0, index=pd.bdate_range("1990-01-02", SIGNAL_DATE_TS))
    vix_close_ser.loc[SIGNAL_DATE_TS] = 50.0
    metadata_dict = {"norgate_snapshot_date_str": "2024-01-12", "norgate_data_profile_str": release_obj.data_profile_str, "norgate_manifest_hash_str": "synthetic-manifest"}
    monkeypatch.setattr(adapter_module, "build_data_source_metadata_dict", lambda *_: metadata_dict)
    monkeypatch.setattr(strategy_host, "build_data_source_metadata_dict", lambda *_: metadata_dict)
    monkeypatch.setattr(dv2_vix_gated, "load_pricing_data", lambda *_: (pricing_data_df, universe_df))
    monkeypatch.setattr(hpi_module, "load_exact_hpi_inputs", lambda **_: (["AAA", "BBB"], universe_df, pricing_data_df))
    monkeypatch.setattr(hpi_vote_vix_gated, "load_exact_hpi_inputs", lambda **_: (["AAA", "BBB"], universe_df, pricing_data_df))
    monkeypatch.setattr(hpi_vote_vix_gated, "append_parking_prices", lambda frame_df, *_: frame_df)
    monkeypatch.setattr(vix_stress_gate, "load_vix_close_ser", lambda *_: vix_close_ser)

    def synthetic_signals(strategy_obj, frame_df):
        strategy_obj._prepare_capsule_state()
        return frame_df

    def candidates(strategy_obj, signal_df, stock_list):
        return pd.DataFrame({symbol_str: symbol_str in ("AAA", "BBB") for symbol_str in stock_list}, index=signal_df.index)

    monkeypatch.setattr(dv2_vix_gated.DV2VixGatedStrategy, "compute_signals", synthetic_signals)
    monkeypatch.setattr(hpi_vote_vix_gated.HPIVoteVixGatedStrategy, "compute_signals", synthetic_signals)
    monkeypatch.setattr(dv2_vix_gated_liq.DV2VixGatedLiquidityStrategy, "_liquidity_candidate_df", candidates)
    monkeypatch.setattr(hpi_vote_vix_gated_liq.HPIVoteVixGatedLiquidityStrategy, "_liquidity_candidate_df", candidates)
    monkeypatch.setattr(dv2_vix_gated.DV2VixGatedStrategy, "get_opportunities", lambda *_: ["AAA", "BBB"])
    monkeypatch.setattr(hpi_vote_vix_gated.HPIVoteVixGatedStrategy, "get_opportunity_list", lambda *_: ["AAA", "BBB"])
    monkeypatch.setattr(strategy_host, "_validate_hpi_live_feature_readiness", lambda **_: None)
    return pricing_data_df


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_entry_weights_follow_liquidity_and_parking_funds_the_true_dollars(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    _inputs(monkeypatch, release_obj)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))
    assert decision_obj.entry_target_weight_map_dict == pytest.approx({"AAA": 2000.0 / 1500.0 / 10.0, "BBB": 1000.0 / 1500.0 / 10.0})
    assert decision_obj.entry_priority_list == ["AAA", "BBB"]
    # (100,000 - 20,000 of entries - the 1% buffer) / 100; the same dollars as two 1/N entries here.
    assert decision_obj.target_share_map_dict == {"BIL": 790.0}
    assert decision_obj.snapshot_metadata_dict["liquidity_reference_float"] == pytest.approx(1500.0)
    assert decision_obj.snapshot_metadata_dict["liquidity_coverage_float"] == 1.0
    assert decision_obj.snapshot_metadata_dict["liquidity_entry_wanted_weight_dict"] == pytest.approx(decision_obj.entry_target_weight_map_dict)
    assert decision_obj.snapshot_metadata_dict["liquidity_skipped_entry_list"] == []
    assert decision_obj.snapshot_metadata_dict["sizing_contract_str"] == adapter_module.MR_CAPSULE_CONTRACT_STR
    assert decision_obj.strategy_state_dict["mr_capsule_strategy_import_str"] == release_obj.strategy_import_str
    assert decision_obj == strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_uneven_entries_change_the_parking_target(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    _inputs(monkeypatch, release_obj, turnover_dict={"AAA": 16e6, "BBB": 1e6})
    # Scores 4,000 and 1,000, typical 2,500: AAA 1.6, BBB 0.5 (clipped from 0.4). Entries 16,000 + 5,000.
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))
    assert decision_obj.entry_target_weight_map_dict == pytest.approx({"AAA": 0.16, "BBB": 0.05})
    assert decision_obj.target_share_map_dict == {"BIL": 780.0}


def test_the_plan_saves_the_coverage_the_check_measured(monkeypatch):
    release_obj = _release("dv2_vix_gated")
    _inputs(monkeypatch, release_obj)
    monkeypatch.setattr(dv2_vix_gated_liq.DV2VixGatedLiquidityStrategy, "validate_liquidity_sizing_ready", lambda *_: 0.9)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))
    assert decision_obj.snapshot_metadata_dict["liquidity_coverage_float"] == 0.9


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_a_short_history_blocks_the_decision(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    _inputs(monkeypatch, release_obj, row_int=60)
    with pytest.raises(ValueError, match="no usable reference"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_a_snapshot_without_turnover_blocks_the_decision(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    _inputs(monkeypatch, release_obj, turnover_dict={})
    with pytest.raises(ValueError, match="requires Turnover"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))


@pytest.mark.parametrize("pod_str", POD_LIST)
@pytest.mark.parametrize("fault_str", ["thin_window", "zero_tail", "infinite_rows", "missing_last_row", "column_missing"])
def test_partial_turnover_loss_blocks_instead_of_trading_another_size(monkeypatch, pod_str, fault_str):
    # A carried reference and m = 1 would otherwise trade these days at other sizes without a word.
    release_obj = _release(pod_str)
    pricing_data_df = _inputs(monkeypatch, release_obj, turnover_dict={"AAA": 4e6} if fault_str == "column_missing" else None)
    if fault_str == "thin_window":
        pricing_data_df.loc[pricing_data_df.index[-31:-1], [("AAA", "Turnover"), ("BBB", "Turnover")]] = np.nan
    elif fault_str == "zero_tail":  # the last row stays good: entries would all go out at half size
        pricing_data_df.loc[pricing_data_df.index[-63:-1], [("AAA", "Turnover"), ("BBB", "Turnover")]] = 0.0
    elif fault_str == "infinite_rows":  # the rolling mean drops these: entries would all go out at m = 1
        pricing_data_df.loc[pricing_data_df.index[-31:-1], [("AAA", "Turnover"), ("BBB", "Turnover")]] = np.inf
    elif fault_str == "missing_last_row":
        pricing_data_df.loc[pricing_data_df.index[-1], ("BBB", "Turnover")] = np.nan
    with pytest.raises(ValueError, match="needs usable Turnover"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))


@pytest.mark.parametrize("pod_str,module_obj,builder_str,class_obj,family_str", [
    ("dv2_vix_gated", dv2_vix_gated_liq, "build_dv2_capsule_liquidity_strategy", dv2_vix_gated_liq.DV2VixGatedLiquidityStrategy, "mr_capsule_dv2"),
    ("hpi_vote_vix_gated", hpi_vote_vix_gated_liq, "build_hpi_capsule_liquidity_strategy", hpi_vote_vix_gated_liq.HPIVoteVixGatedLiquidityStrategy, "mr_capsule_hpi_vote"),
])
def test_the_adapter_builds_the_pod_class_of_the_release(monkeypatch, pod_str, module_obj, builder_str, class_obj, family_str):
    release_obj = _release(pod_str)
    _inputs(monkeypatch, release_obj)
    original_fn, built_list = getattr(module_obj, builder_str), []

    def spy_fn(**kwarg_dict):
        built_list.append((original_fn(**kwarg_dict), kwarg_dict))
        return built_list[-1][0]

    monkeypatch.setattr(module_obj, builder_str, spy_fn)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))
    assert [type(strategy_obj) for strategy_obj, _ in built_list] == [class_obj]
    assert built_list[0][1]["parking_enabled_bool"] and not built_list[0][1]["spmo_parking_enabled_bool"]
    assert decision_obj.snapshot_metadata_dict["strategy_family_str"] == family_str


BROKEN_SOURCE_LIST = [
    pytest.param("import a_module_that_does_not_exist", id="import_error"),
    pytest.param("def broken(:", id="syntax_error"),
    pytest.param("raise NameError('broken')", id="name_error"),
    pytest.param("raise TypeError('broken')", id="type_error"),
]


def _break_the_sizing_module(monkeypatch, tmp_path, source_str):
    (tmp_path / "dv2_vix_gated_liq.py").write_text(source_str, encoding="utf-8")
    monkeypatch.setattr(capsule_package, "__path__", [str(tmp_path), *capsule_package.__path__])
    monkeypatch.delattr(capsule_package, "dv2_vix_gated_liq")
    monkeypatch.delitem(sys.modules, "strategies.mr_capsule.dv2_vix_gated_liq")


@pytest.mark.parametrize("source_str", BROKEN_SOURCE_LIST)
def test_a_broken_sizing_module_blocks_before_any_data_load(monkeypatch, tmp_path, source_str):
    release_obj = _release("dv2_vix_gated")
    _inputs(monkeypatch, release_obj)
    monkeypatch.setattr(dv2_vix_gated, "load_pricing_data", lambda *_: pytest.fail("the panel must not load first"))
    _break_the_sizing_module(monkeypatch, tmp_path, source_str)
    with pytest.raises(RuntimeError, match="cannot load"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))


@pytest.mark.parametrize("fault_str,message_str", [("no_turnover", "requires Turnover"), ("broken_module", "cannot load")])
def test_a_sizing_failure_blocks_only_its_own_pod_in_the_shared_tick(monkeypatch, tmp_path, fault_str, message_str):
    sized_obj = replace(_release("dv2_vix_gated"), enabled_bool=True, account_route_str="DU_SIZED")
    ndx_obj = replace(parse_release_manifest("docs/live/release_templates/pod_ndx_atr_normalized_monthly_open.yaml.example"),
                      enabled_bool=True, account_route_str="DU_NDX", release_id_str="ndx.v1", pod_id_str="pod_ndx_01")
    _inputs(monkeypatch, sized_obj, turnover_dict={})
    if fault_str == "broken_module":
        _break_the_sizing_module(monkeypatch, tmp_path, "raise TypeError('broken')")
    original_fn, visited_list = strategy_host.build_decision_plan_for_release, []

    def build_fn(*, release_obj, as_of_ts, pod_state_obj):
        visited_list.append(release_obj.pod_id_str)
        if release_obj.pod_id_str == sized_obj.pod_id_str:
            return original_fn(release_obj=release_obj, as_of_ts=as_of_ts, pod_state_obj=_state(sized_obj))
        return DecisionPlan(release_obj.release_id_str, release_obj.user_id_str, release_obj.pod_id_str, release_obj.account_route_str, as_of_ts, as_of_ts + timedelta(days=4),
                            as_of_ts + timedelta(days=4), release_obj.execution_policy_str, {}, {}, {}, decision_book_type_str="full_target_weight_book")

    monkeypatch.setattr(runner_module.strategy_host, "build_decision_plan_for_release", build_fn)
    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync", lambda *_a, **_k: [sized_obj, ndx_obj])
    monkeypatch.setattr(runner_module.scheduler_utils, "select_due_release_list", lambda release_list, _as_of_ts: release_list)
    state_store_obj = LiveStateStore(str(tmp_path / "tick.sqlite3"))
    log_path_obj = tmp_path / "events.jsonl"
    result_dict = runner_module.build_decision_plans(state_store_obj, AS_OF_TS, str(tmp_path), auto_sync_norgate_snapshots_bool=False, trace_enabled_bool=False, log_path_str=str(log_path_obj))
    assert visited_list == [sized_obj.pod_id_str, ndx_obj.pod_id_str]
    assert result_dict["reason_count_map_dict"]["mr_capsule_decision_blocked"] == 1 and result_dict["created_decision_plan_count_int"] == 1
    assert state_store_obj.get_latest_decision_plan_for_pod(sized_obj.pod_id_str) is None
    assert state_store_obj.get_latest_decision_plan_for_pod(ndx_obj.pod_id_str) is not None
    log_str = log_path_obj.read_text(encoding="utf-8")
    assert "mr_capsule_decision_blocked" in log_str and message_str in log_str


@pytest.mark.parametrize("pod_str", POD_LIST)
@pytest.mark.parametrize("position_map_dict", [{"AAA": 10.0}, {"BIL": 10.0}, {}], ids=["stock", "bil", "flat"])
def test_a_pod_cannot_move_between_the_forms_under_one_id(monkeypatch, pod_str, position_map_dict):
    sized_obj = _release(pod_str)
    plain_obj = parse_release_manifest(f"docs/live/release_templates/pod_mr_{pod_str}_bil_daily_moo.yaml.example")
    _inputs(monkeypatch, sized_obj)
    for release_obj, prior_obj in ((sized_obj, plain_obj), (plain_obj, sized_obj)):
        state_obj = _state(release_obj, position_map_dict, 99_000.0)
        state_obj = replace(state_obj, strategy_state_dict={"mr_capsule_strategy_import_str": prior_obj.strategy_import_str})
        with pytest.raises(adapter_module.CapsuleHoldingMismatchError, match="cannot adopt"):
            strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, state_obj)


@pytest.mark.parametrize("pod_str", POD_LIST)
@pytest.mark.parametrize("position_map_dict", [{"AAA": 10.0}, {"BIL": 10.0}], ids=["stock", "bil"])
def test_a_new_pod_cannot_adopt_holdings(monkeypatch, pod_str, position_map_dict):
    release_obj = _release(pod_str)
    _inputs(monkeypatch, release_obj)
    state_obj = replace(_state(release_obj, position_map_dict, 99_000.0), strategy_state_dict={})
    with pytest.raises(adapter_module.CapsuleHoldingMismatchError, match="cannot adopt"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, state_obj)


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_a_new_pod_on_a_flat_account_starts_the_sized_form(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    _inputs(monkeypatch, release_obj)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, replace(_state(release_obj), strategy_state_dict={}))
    assert decision_obj.entry_target_weight_map_dict
    assert decision_obj.strategy_state_dict["mr_capsule_strategy_import_str"] == release_obj.strategy_import_str


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_spmo_cannot_sit_in_a_sized_pod(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    _inputs(monkeypatch, release_obj)
    with pytest.raises(adapter_module.CapsuleHoldingMismatchError, match="parking holdings"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj, {"SPMO": 10.0}, 99_000.0))


def _order_dict(release_obj, decision_obj, state_obj, pricing_data_df):
    """The decision through the real execution contract and VPlan, up to the broker requests."""
    now_ts = datetime(2024, 1, 16, 9, 20, tzinfo=ZoneInfo("America/New_York"))  # the session after the decision: Monday the 15th is a holiday
    snapshot_obj = BrokerSnapshot(account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=now_ts, cash_float=state_obj.cash_float, total_value_float=state_obj.total_value_float,
                                  net_liq_float=state_obj.total_value_float, position_amount_map=dict(state_obj.position_amount_map), open_order_id_list=[])
    asset_set = set(decision_obj.entry_target_weight_map_dict) | set(decision_obj.exit_asset_set) | set(decision_obj.target_share_map_dict)
    quote_dict = {asset_str: float(pricing_data_df.loc[SIGNAL_DATE_TS, (asset_str, "Close")]) for asset_str in asset_set}
    price_obj = LivePriceSnapshot(account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=now_ts, price_source_str="synthetic", asset_reference_price_map=quote_dict)
    decision_obj = replace(decision_obj, decision_plan_id_int=1)
    assert validate_mr_capsule_execution_contract(release_obj, decision_obj, snapshot_obj, price_obj) == pytest.approx(decision_obj.snapshot_metadata_dict["decision_nav_float"])
    vplan_obj = build_vplan(release_obj, decision_obj, snapshot_obj, price_obj)
    return {order_obj.asset_str: order_obj.amount_float for order_obj in build_broker_order_request_list_from_vplan(vplan_obj)}


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_a_sized_decision_reaches_the_broker_requests_at_its_own_weights(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    pricing_data_df = _inputs(monkeypatch, release_obj, turnover_dict={"AAA": 16e6, "BBB": 1e6})
    state_obj = _state(release_obj)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, state_obj)
    # NAV 100,000 at 100: AAA 16%, BBB 5%, and BIL at its frozen whole-share target.
    assert _order_dict(release_obj, decision_obj, state_obj, pricing_data_df) == {"AAA": 160.0, "BBB": 50.0, "BIL": 780.0}


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_an_entry_that_rounds_to_zero_shares_sends_no_order_and_keeps_the_rest(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    pricing_data_df = _inputs(monkeypatch, release_obj, turnover_dict={"AAA": 16e6, "BBB": 1e6})
    pricing_data_df[("BBB", "Close")] = 6_000.0  # a 5% entry of 100,000 buys no share at this price
    state_obj = _state(release_obj)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, state_obj)
    assert decision_obj.entry_target_weight_map_dict["BBB"] == pytest.approx(0.05)
    # BIL stays sized as if those dollars were spent, as the module's caveat says.
    assert _order_dict(release_obj, decision_obj, state_obj, pricing_data_df) == {"AAA": 160.0, "BIL": 780.0}


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_a_capped_entry_keeps_its_size_before_the_cap_beside_the_plan(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    # Scores 1,000 and 4,000, typical 2,500: BBB wants 16%. AAA stays held at 90,000, which leaves 9,000 under the 99% cap.
    _inputs(monkeypatch, release_obj, turnover_dict={"AAA": 1e6, "BBB": 16e6})
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj, {"AAA": 900.0}, 10_000.0))
    assert decision_obj.entry_target_weight_map_dict == pytest.approx({"BBB": 0.09})
    assert decision_obj.snapshot_metadata_dict["liquidity_entry_wanted_weight_dict"] == pytest.approx({"BBB": 0.16})
    assert decision_obj.snapshot_metadata_dict["liquidity_skipped_entry_list"] == []


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_a_skipped_entry_is_named_beside_the_plan(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    # 98,700 held leaves 300 under the cap, under the 500 that is 0.5% of NAV.
    _inputs(monkeypatch, release_obj, turnover_dict={"AAA": 1e6, "BBB": 16e6})
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj, {"AAA": 987.0}, 1_300.0))
    assert decision_obj.entry_target_weight_map_dict == {}
    assert decision_obj.snapshot_metadata_dict["liquidity_entry_wanted_weight_dict"] == {}
    assert decision_obj.snapshot_metadata_dict["liquidity_skipped_entry_list"] == ["BBB"]


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_the_release_keeps_the_capsule_contract(pod_str):
    release_obj = _release(pod_str)
    validate_release_manifest(release_obj)
    assert not release_obj.enabled_bool and release_obj.mode_str == "paper" and release_obj.auto_submit_enabled_bool
    assert release_obj.data_profile_str == adapter_module.MR_CAPSULE_PROFILE_TUPLE[int("hpi_vote" in pod_str)]
    with pytest.raises(ValueError, match="frozen strategy parameters"):
        validate_release_manifest(replace(release_obj, params_dict={"liquidity_size_max_float": 3.0}))


def test_the_one_over_n_identities_keep_their_order_and_names():
    assert adapter_module.MR_CAPSULE_STRATEGY_IMPORT_TUPLE[:6] == tuple(
        f"strategies.mr_capsule.strategy_mr_{pod_str}_{mode_str}" for pod_str in POD_LIST for mode_str in ("cash", "bil", "spmo"))
    assert adapter_module.MR_CAPSULE_STRATEGY_IMPORT_TUPLE[6:] == adapter_module.MR_CAPSULE_LIQUIDITY_STRATEGY_IMPORT_TUPLE


@pytest.mark.parametrize("pod_str", POD_LIST)
def test_a_one_over_n_decision_carries_no_sizing_fields(monkeypatch, pod_str):
    from test_live_mr_capsule_host import _inputs as plain_inputs, AS_OF_TS as plain_as_of_ts

    release_obj = parse_release_manifest(f"docs/live/release_templates/pod_mr_{pod_str}_bil_daily_moo.yaml.example")
    plain_inputs(monkeypatch, release_obj)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, plain_as_of_ts, _state(release_obj))
    assert decision_obj.entry_target_weight_map_dict == {"AAA": .1}
    assert not [key_str for key_str in decision_obj.snapshot_metadata_dict if key_str.startswith("liquidity_")]
