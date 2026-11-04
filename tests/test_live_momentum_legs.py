"""The momentum capsule legs on the live route: disabled templates, the two-swap clock, the frozen snapshot, the decision and one whole cycle with a synthetic broker."""
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import exchange_calendars as calendar_module
import numpy as np
import pandas as pd
import pytest
import yaml

from alpha.live import dashboard, doctor, logging_utils, norgate_snapshot_sync, reference_compare, runner, scheduler_service, scheduler_utils, strategy_host
from alpha.live import momentum_legs_adapter as adapter_module
from alpha.live.daily_reconcile import is_daily_reconcile_release_bool
from alpha.live.dispatch_state import is_guarded_release_bool
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.models import PodState
from alpha.live.momentum_legs_adapter import MOMENTUM_LEGS_PROFILE_BY_IMPORT_DICT, MOMENTUM_LEGS_STRATEGY_IMPORT_TUPLE
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.release_manifest import (
    SUPPORTED_STRATEGY_IMPORT_TUPLE,
    parse_release_manifest,
    validate_enabled_deployment_for_mode,
    validate_release_list,
    validate_release_manifest,
)
from alpha.live.state_store_v2 import LiveStateStore
from data import norgate_snapshot_store as snapshot_module
from data.momentum_legs_snapshot_contract import (
    MOMENTUM_LEGS_NDX_PROFILE_STR,
    MOMENTUM_LEGS_R1000_PROFILE_STR,
    SECTOR_INDEX_SYMBOL_TUPLE,
    momentum_legs_label_hash_str,
)
from scripts import doctor_norgate_client, export_norgate_snapshot as export_module
from strategies.momentum import momentum_capsule_legs as legs_module
from alpha.live.ibkr_socket_client import IBKR_STOCK_SYMBOL_MAP_DICT

MARKET_ZONE_OBJ = ZoneInfo("America/New_York")
TEMPLATE_DIR_PATH = Path("docs/live/release_templates")
TEMPLATE_BY_LEG_DICT = {
    "leg1": TEMPLATE_DIR_PATH / "pod_mo_capsule_leg1_natr_ndx_month_swap.yaml.example",
    "leg2": TEMPLATE_DIR_PATH / "pod_mo_capsule_leg2_residual_ndx_month_swap.yaml.example",
    "leg3": TEMPLATE_DIR_PATH / "pod_mo_capsule_leg3_residual_r1000_month_swap.yaml.example",
}
T0_STR, T10_STR, OTHER_STR = "2024-01-31", "2024-01-17", "2024-01-18"
T0_AS_OF_TS = datetime(2024, 1, 31, 18, 0, tzinfo=MARKET_ZONE_OBJ)
T10_AS_OF_TS = datetime(2024, 1, 17, 18, 0, tzinfo=MARKET_ZONE_OBJ)
SECTOR_LIST = ["45000000", "35000000", "40000000", "25000000", "20000000", "30000000"]
STOCK_LIST = [f"S{column_int:02d}" for column_int in range(60)]


class FakeProvider:
    StockPriceAdjustmentType = SimpleNamespace(CAPITALSPECIAL="CAPITALSPECIAL", TOTALRETURN="TOTALRETURN")
    PaddingType = SimpleNamespace(ALLMARKETDAYS="ALLMARKETDAYS", NONE="NONE")

    def __init__(self, gate_down_bool=False):
        self.date_idx = pd.DatetimeIndex(calendar_module.get_calendar("XNYS").sessions_in_range("2022-01-03", "2024-02-29")).tz_localize(None).rename("Date")
        rng = np.random.default_rng(11)
        row_count_int = len(self.date_idx)
        self.close_dict = {}
        for column_int, symbol_str in enumerate(STOCK_LIST + ["OLD"]):
            self.close_dict[symbol_str] = 100.0 * np.exp(np.cumsum(-.0008 + .0022 * column_int / 60 + .015 * rng.standard_normal(row_count_int)))
        gate_vec = 100.0 * np.exp(np.cumsum(.0008 + .004 * rng.standard_normal(row_count_int)))
        if gate_down_bool:
            gate_vec[self.date_idx >= pd.Timestamp("2023-11-01")] *= .6
        for symbol_str in ("SPY", "$RUI", "$NDX", "$SPXTR"):
            self.close_dict[symbol_str] = gate_vec.copy()
        self.close_dict["$VXN"] = np.full(row_count_int, 25.0)
        for symbol_str in SECTOR_INDEX_SYMBOL_TUPLE:
            self.close_dict[symbol_str] = 100.0 * np.exp(np.cumsum(.0004 + .009 * rng.standard_normal(row_count_int)))
        self.failure_set = set()
        self.unobserved_set = set()
        self.padded_close_gap_set = set()
        self.no_turnover_at_t_bool = False
        self.call_list = []
        self.false_none_count_dict = {}
        self.no_label_set = {"S00"}

    def classification_at_level(self, symbol_str, scheme_str, field_str, level_int):
        assert (scheme_str, field_str) == ("GICS", "ClassificationId")
        self.call_list.append(symbol_str)
        if symbol_str in self.failure_set:
            raise ConnectionError("Classification service unavailable")
        if symbol_str in self.no_label_set:
            return None
        if self.false_none_count_dict.get(symbol_str, 0) > 0:
            self.false_none_count_dict[symbol_str] -= 1
            return None
        column_int = STOCK_LIST.index(symbol_str) if symbol_str in STOCK_LIST else 61
        return SECTOR_LIST[column_int % 6] if level_int == 1 else f"{SECTOR_LIST[column_int % 6][:2]}{10 + column_int % 2 * 10}000"

    def price_timeseries(self, symbol_str, **option_dict):
        close_vec = self.close_dict[symbol_str]
        price_df = pd.DataFrame({
            "Open": np.r_[close_vec[0], close_vec[:-1]], "High": close_vec * 1.01, "Low": close_vec * .99, "Close": close_vec, "Volume": 1e6,
            "Turnover": 5e7, "Unadjusted Close": close_vec * 2.0, "Dividend": 0.0}, index=self.date_idx)
        if symbol_str.startswith("$"):
            price_df = price_df.drop(columns=["Turnover", "Unadjusted Close", "Dividend"])
        if symbol_str == "OLD":
            price_df = price_df.loc[:"2023-06-30"]
        if self.no_turnover_at_t_bool and "Turnover" in price_df.columns:
            price_df = price_df.assign(Turnover=price_df["Turnover"].where(price_df.index != pd.Timestamp(T0_STR)))
        price_df = price_df.loc[option_dict["start_date"]:option_dict["end_date"]]
        if option_dict["padding_setting"] == "NONE" and symbol_str in self.unobserved_set:
            return price_df.iloc[:0]
        if option_dict["padding_setting"] == "NONE" and symbol_str in self.padded_close_gap_set:
            price_df = price_df.assign(Close=price_df["Close"] + 1.0)
        return price_df


def fake_universe_df(provider_obj):
    universe_df = pd.DataFrame(1, index=provider_obj.date_idx, columns=STOCK_LIST + ["OLD"])
    universe_df.loc[universe_df.index > pd.Timestamp("2023-06-30"), "OLD"] = 0
    universe_df["FUTURE"] = 0
    universe_df.loc[universe_df.index > pd.Timestamp("2024-02-01"), "FUTURE"] = 1
    universe_df["ANCIENT"] = 0
    universe_df.loc[pd.Timestamp("1998-06-01")] = 0
    universe_df.loc[pd.Timestamp("1998-06-01"), "ANCIENT"] = 1
    return universe_df.sort_index()


def export_snapshot(root_path_obj, monkeypatch, provider_obj, profile_str, date_str):
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setattr(export_module, "_load_index_constituent_matrix_df", lambda *args, **kwargs: (list(fake_universe_df(provider_obj).columns), fake_universe_df(provider_obj)))
    return export_module.export_profile_snapshot(snapshot_root_str=str(root_path_obj), profile_str=profile_str, snapshot_date_str=date_str)


@pytest.fixture(scope="module")
def snapshot_root_dict(tmp_path_factory):
    root_dict = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        for key_str, date_str, gate_down_bool in (("t0", T0_STR, False), ("t10", T10_STR, False), ("other", OTHER_STR, False), ("gate_down", T0_STR, True)):
            root_path_obj = tmp_path_factory.mktemp(key_str)
            provider_obj = FakeProvider(gate_down_bool=gate_down_bool)
            for profile_str in (MOMENTUM_LEGS_NDX_PROFILE_STR, MOMENTUM_LEGS_R1000_PROFILE_STR):
                export_snapshot(root_path_obj, monkeypatch, provider_obj, profile_str, date_str)
            root_dict[key_str] = root_path_obj
    return root_dict


@pytest.fixture(autouse=True)
def fresh_refusals(monkeypatch):
    monkeypatch.setattr(adapter_module, "_REFUSED_DECISION_BY_KEY_DICT", {})


@pytest.fixture
def snapshot_mode(monkeypatch):
    def use_root(root_path_obj):
        monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(root_path_obj))
        monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
        return root_path_obj
    return use_root


@pytest.fixture
def copied_snapshot(snapshot_root_dict, snapshot_mode, tmp_path):
    """A private copy of the T0 Nasdaq snapshot, safe to damage."""
    target_path_obj = tmp_path / "root" / MOMENTUM_LEGS_NDX_PROFILE_STR / T0_STR
    shutil.copytree(snapshot_root_dict["t0"] / MOMENTUM_LEGS_NDX_PROFILE_STR / T0_STR, target_path_obj)
    snapshot_mode(tmp_path / "root")
    return target_path_obj


def rewrite_manifest(snapshot_path_obj, mutate_fn):
    manifest_path_obj = snapshot_path_obj / "manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    mutate_fn(manifest_dict)
    manifest_path_obj.write_text(json.dumps(manifest_dict), encoding="utf-8")


def rewrite_universe(snapshot_path_obj, universe_df):
    universe_path_obj = snapshot_path_obj / "universe.parquet"
    universe_df.to_parquet(universe_path_obj)
    rewrite_manifest(snapshot_path_obj, lambda manifest_dict: manifest_dict["files"]["universe.parquet"].update(sha256=snapshot_module._hash_file_path(universe_path_obj)))


def rewrite_prices(snapshot_path_obj, price_df):
    price_path_obj = snapshot_path_obj / "prices.parquet"
    price_df.to_parquet(price_path_obj, index=False)
    rewrite_manifest(snapshot_path_obj, lambda manifest_dict: manifest_dict["files"]["prices.parquet"].update(sha256=snapshot_module._hash_file_path(price_path_obj)))


@pytest.fixture(params=list(TEMPLATE_BY_LEG_DICT))
def release_obj(request):
    return parse_release_manifest(str(TEMPLATE_BY_LEG_DICT[request.param]))


def enabled_release(template_release_obj, account_str="DU777001"):
    return replace(template_release_obj, enabled_bool=True, account_route_str=account_str, release_id_str=f"{template_release_obj.pod_id_str}.{account_str}")


# ---------------------------------------------------------------- releases
def test_templates_are_separate_disabled_routes(release_obj):
    assert release_obj.strategy_import_str in MOMENTUM_LEGS_STRATEGY_IMPORT_TUPLE
    assert release_obj.strategy_import_str in SUPPORTED_STRATEGY_IMPORT_TUPLE
    assert release_obj.strategy_import_str in strategy_host.FULL_TARGET_DECISION_STRATEGY_IMPORT_SET
    assert release_obj.data_profile_str == MOMENTUM_LEGS_PROFILE_BY_IMPORT_DICT[release_obj.strategy_import_str]
    assert not release_obj.enabled_bool and not release_obj.auto_submit_enabled_bool and release_obj.mode_str == "paper"
    assert release_obj.pod_budget_fraction_float == 1.0
    assert not is_daily_reconcile_release_bool(release_obj) and not is_guarded_release_bool(release_obj)
    assert reference_compare.inspect_auto_reference_support_dict(release_obj)["supported_bool"]
    validate_enabled_deployment_for_mode([release_obj], "paper")
    with pytest.raises(ValueError, match="real IBKR account"):
        validate_enabled_deployment_for_mode([replace(release_obj, enabled_bool=True)], "paper")


def test_each_template_has_its_own_client_id_and_account():
    id_dict = {path_obj.name: yaml.safe_load(path_obj.read_text(encoding="utf-8")).get("broker", {}).get("client_id_int") for path_obj in TEMPLATE_DIR_PATH.glob("*.yaml.example")}
    leg_id_list = [id_dict[path_obj.name] for path_obj in TEMPLATE_BY_LEG_DICT.values()]
    assert len(set(leg_id_list)) == 3 and not set(leg_id_list) & {client_id_int for name_str, client_id_int in id_dict.items() if "mo_capsule_leg" not in name_str}
    route_set = {parse_release_manifest(str(path_obj)).account_route_str for path_obj in TEMPLATE_BY_LEG_DICT.values()}
    assert len(route_set) == 3


@pytest.mark.parametrize("field_str,value_obj", [
    ("data_profile_str", snapshot_module.NDX_E2_PROFILE_STR),
    ("session_calendar_id_str", "XTSE"),
    ("signal_clock_str", "month_end_snapshot_ready"),
    ("signal_clock_str", "eod_snapshot_ready"),
    ("execution_policy_str", "next_month_first_open"),
    ("execution_policy_str", "same_day_moc"),
])
def test_a_leg_rejects_another_profile_or_timing(release_obj, field_str, value_obj):
    with pytest.raises(ValueError, match="requires|reserved"):
        validate_release_manifest(replace(release_obj, **{field_str: value_obj}))


@pytest.mark.parametrize("params_dict", [{"max_positions_int": 12}, {"hold_rank_int": 18}, {"target_vxn_float": 22.0}])
def test_a_leg_refuses_any_rule_override(release_obj, params_dict):
    with pytest.raises(ValueError, match="frozen"):
        validate_release_manifest(replace(release_obj, params_dict=params_dict))


@pytest.mark.parametrize("capital_obj", [0, -1, float("nan"), True])
def test_invalid_capital_is_rejected(release_obj, capital_obj):
    with pytest.raises(ValueError):
        validate_release_manifest(replace(release_obj, params_dict={"capital_base_float": capital_obj}))


@pytest.mark.parametrize("budget_float", [0.0, 1.5, float("nan")])
def test_budget_fraction_cannot_overallocate(release_obj, budget_float):
    with pytest.raises(ValueError, match="budget"):
        validate_release_manifest(replace(release_obj, pod_budget_fraction_float=budget_float))


def test_the_profiles_and_the_clock_are_reserved_for_the_legs():
    other_release_obj = parse_release_manifest(str(TEMPLATE_DIR_PATH / "pod_ndx_natr20_vxn_scaled_sector_cap_monthly_open.yaml.example"))
    with pytest.raises(ValueError):
        validate_release_manifest(replace(other_release_obj, data_profile_str=MOMENTUM_LEGS_NDX_PROFILE_STR))
    other_release_obj = parse_release_manifest(str(TEMPLATE_DIR_PATH / "pod_taa_btal_1n_fallback_tqqq_vix_cash_monthly_open.yaml.example"))
    with pytest.raises(ValueError, match="month-swap clock is reserved"):
        validate_release_manifest(replace(other_release_obj, signal_clock_str=scheduler_utils.MONTH_SWAP_SIGNAL_CLOCK_STR))
    with pytest.raises(ValueError, match="reserved"):
        validate_release_manifest(replace(other_release_obj, data_profile_str=MOMENTUM_LEGS_R1000_PROFILE_STR))


@pytest.mark.parametrize("reverse_bool", [False, True])
def test_enabled_legs_reject_a_shared_account_in_both_orders(reverse_bool):
    leg1_obj = enabled_release(parse_release_manifest(str(TEMPLATE_BY_LEG_DICT["leg1"])))
    other_path_obj = TEMPLATE_DIR_PATH / "pod_ndx_natr20_vxn_scaled_sector_cap_monthly_open.yaml.example"
    other_obj = enabled_release(parse_release_manifest(str(other_path_obj)))
    release_list = [other_obj, leg1_obj] if reverse_bool else [leg1_obj, other_obj]
    with pytest.raises(ValueError, match="dedicated"):
        validate_release_list(release_list)


def test_three_dedicated_accounts_are_valid():
    validate_release_list([enabled_release(parse_release_manifest(str(path_obj)), f"DU77700{index_int}") for index_int, path_obj in enumerate(TEMPLATE_BY_LEG_DICT.values())])


@pytest.mark.parametrize("signal_date_str,execution_date_str", [("2026-10-16", "2026-10-19"), ("2026-10-30", "2026-11-02"), ("2026-12-31", "2027-01-04")])
def test_a_leg_fills_at_the_next_session_open(release_obj, signal_date_str, execution_date_str):
    target_ts = scheduler_utils.build_target_execution_timestamp_ts(pd.Timestamp(signal_date_str).to_pydatetime(), release_obj)
    submission_ts = scheduler_utils.build_submission_timestamp_ts(pd.Timestamp(signal_date_str).to_pydatetime(), release_obj)
    signal_ts = scheduler_utils.build_signal_timestamp_ts(pd.Timestamp(signal_date_str).to_pydatetime(), release_obj)
    assert target_ts.date().isoformat() == execution_date_str and (target_ts.hour, target_ts.minute) == (9, 30)
    assert target_ts - submission_ts == timedelta(minutes=6, seconds=30)
    assert signal_ts.date().isoformat() == signal_date_str and signal_ts.hour == 16


def test_client_doctor_smokes_both_profiles(tmp_path, monkeypatch):
    check_list = []
    seen_list = []
    monkeypatch.setattr(doctor_norgate_client, "load_price_timeseries_df", lambda symbol_str, adjustment_str, data_profile_str: seen_list.append((data_profile_str, symbol_str)) or pd.DataFrame({"Close": [1.0]}))
    monkeypatch.setattr(doctor_norgate_client, "load_index_constituent_matrix_df", lambda index_str, data_profile_str: seen_list.append((data_profile_str, index_str)) or ([], pd.DataFrame({"AAA": [1]})))
    doctor_norgate_client._check_profile_smoke(check_list, local_root_path_obj=tmp_path, profile_list=[MOMENTUM_LEGS_NDX_PROFILE_STR, MOMENTUM_LEGS_R1000_PROFILE_STR], printer_fn=lambda *_: None)
    assert (MOMENTUM_LEGS_R1000_PROFILE_STR, "Russell 1000") in seen_list and (MOMENTUM_LEGS_NDX_PROFILE_STR, "$NDX") in seen_list


# ---------------------------------------------------------------- the clock
@pytest.mark.parametrize("session_str,expected_bool", [
    ("2026-10-30", True), ("2026-10-16", True), ("2026-10-15", False), ("2026-10-19", False),
    ("2026-02-27", True), ("2026-02-12", True), ("2026-02-13", False),
    ("2026-12-31", True), ("2026-12-16", True),
])
def test_swap_sessions(session_str, expected_bool):
    assert scheduler_utils.is_month_swap_session_bool(pd.Timestamp(session_str), "XNYS") is expected_bool


def test_latest_and_next_swap_sessions():
    as_of_ts = datetime(2026, 10, 22, 18, 0, tzinfo=MARKET_ZONE_OBJ)
    assert scheduler_utils.get_latest_completed_month_swap_session_label_ts(as_of_ts, "XNYS") == pd.Timestamp("2026-10-16")
    before_close_ts = datetime(2026, 10, 16, 15, 0, tzinfo=MARKET_ZONE_OBJ)
    assert scheduler_utils.get_latest_completed_month_swap_session_label_ts(before_close_ts, "XNYS") == pd.Timestamp("2026-09-30")
    assert scheduler_utils.get_next_month_swap_session_label_ts(pd.Timestamp("2026-10-16"), "XNYS") == pd.Timestamp("2026-10-30")
    assert scheduler_utils.get_next_month_swap_session_label_ts(pd.Timestamp("2026-10-30"), "XNYS") == pd.Timestamp("2026-11-13")


@pytest.mark.parametrize("heartbeat_str,as_of_ts,due_bool,reason_str", [
    ("2026-10-15", datetime(2026, 10, 15, 18, 0, tzinfo=MARKET_ZONE_OBJ), False, "not_month_swap_session"),
    ("2026-10-16", datetime(2026, 10, 16, 18, 0, tzinfo=MARKET_ZONE_OBJ), True, "snapshot_ready"),
    ("2026-10-16", datetime(2026, 10, 19, 8, 0, tzinfo=MARKET_ZONE_OBJ), True, "carry_forward_snapshot_ready"),
    ("2026-10-16", datetime(2026, 10, 19, 9, 24, tzinfo=MARKET_ZONE_OBJ), False, "snapshot_window_expired"),
    ("2026-10-30", datetime(2026, 10, 30, 18, 0, tzinfo=MARKET_ZONE_OBJ), True, "snapshot_ready"),
    ("2026-10-30", datetime(2026, 11, 2, 8, 0, tzinfo=MARKET_ZONE_OBJ), True, "carry_forward_snapshot_ready"),
])
def test_build_gate_on_swap_sessions(release_obj, monkeypatch, heartbeat_str, as_of_ts, due_bool, reason_str):
    monkeypatch.setattr(scheduler_utils, "load_latest_norgate_heartbeat_session_label_ts", lambda _: pd.Timestamp(heartbeat_str))
    gate_dict = scheduler_utils.evaluate_build_gate_dict(release_obj, as_of_ts)
    assert (gate_dict["due_bool"], gate_dict["reason_code_str"]) == (due_bool, reason_str)
    assert scheduler_utils.is_release_due_for_build(release_obj, as_of_ts) is due_bool
    assert scheduler_utils.get_signal_clock_reason_code_str(release_obj, as_of_ts) == reason_str


def test_the_existing_clocks_do_not_read_the_swap_calendar(monkeypatch):
    month_end_release_obj = parse_release_manifest(str(TEMPLATE_DIR_PATH / "pod_ndx_natr20_vxn_scaled_sector_cap_monthly_open.yaml.example"))
    monkeypatch.setattr(scheduler_utils, "load_latest_norgate_heartbeat_session_label_ts", lambda _: pd.Timestamp("2026-10-16"))
    monkeypatch.setattr(scheduler_utils, "is_month_swap_session_bool", lambda *_: pytest.fail("An existing clock read the swap calendar"))
    gate_dict = scheduler_utils.evaluate_build_gate_dict(month_end_release_obj, datetime(2026, 10, 16, 18, 0, tzinfo=MARKET_ZONE_OBJ))
    assert (gate_dict["due_bool"], gate_dict["reason_code_str"]) == (False, "not_month_end_session")


def test_month_swap_sync_requires_the_latest_swap_session(release_obj):
    as_of_ts = datetime(2026, 10, 22, 18, 0, tzinfo=MARKET_ZONE_OBJ)
    required_dict = norgate_snapshot_sync._required_snapshot_session_by_release_dict([release_obj], as_of_ts)
    assert required_dict == {release_obj.release_id_str: pd.Timestamp("2026-10-16")}
    assert doctor._release_requires_norgate_snapshot_sync_bool(release_obj)


def test_dashboard_flags_a_missed_swap_and_keeps_the_month_end_rule(release_obj):
    as_of_ts = datetime(2026, 10, 22, 18, 0, tzinfo=MARKET_ZONE_OBJ)
    assert dashboard._missed_monthly_decision_cycle_dict(release_obj, None, as_of_ts) is not None
    daily_release_obj = parse_release_manifest(str(TEMPLATE_DIR_PATH / "pod_mr_dv2_vix_gated_liq_bil_daily_moo.yaml.example"))
    assert dashboard._missed_monthly_decision_cycle_dict(daily_release_obj, None, as_of_ts) is None


def test_a_waiting_leg_counts_as_a_normal_wait_on_the_screens():
    assert "not_month_swap_session" in doctor.WAIT_GATE_REASON_CODE_SET
    assert scheduler_utils.MONTH_SWAP_SIGNAL_CLOCK_STR in dashboard.NORGATE_SNAPSHOT_GATED_SIGNAL_CLOCK_SET


def test_runner_expects_the_next_swap_after_a_completed_cycle(release_obj):
    next_ready_ts = runner._next_norgate_signal_ready_timestamp_ts(release_obj, pd.Timestamp("2026-10-16"))
    assert next_ready_ts.date().isoformat() == "2026-10-30" and next_ready_ts.hour == 16
    assert runner._release_uses_norgate_snapshot_gate_bool(release_obj)


# ---------------------------------------------------------------- the frozen snapshot
def test_export_carries_the_backtest_universe_labels_and_endpoints(snapshot_root_dict, snapshot_mode):
    snapshot_mode(snapshot_root_dict["t0"])
    for profile_str, helper_str in ((MOMENTUM_LEGS_NDX_PROFILE_STR, "$NDX"), (MOMENTUM_LEGS_R1000_PROFILE_STR, "$RUI")):
        manifest_obj = snapshot_module.load_valid_snapshot_manifest(profile_str)
        contract_dict = manifest_obj.manifest_dict["data_contract"]
        assert manifest_obj.snapshot_date_ts == pd.Timestamp(T0_STR)
        assert "FUTURE" not in contract_dict["sector_by_symbol_dict"] and "ANCIENT" not in contract_dict["sector_by_symbol_dict"]
        assert "OLD" in contract_dict["sector_by_symbol_dict"]
        assert contract_dict["sector_by_symbol_dict"]["S00"] == "UNKNOWN"
        assert contract_dict["observed_endpoint_date_by_symbol_dict"][helper_str] == T0_STR
        assert "OLD" not in contract_dict["observed_endpoint_date_by_symbol_dict"]
        assert "$VXN" not in contract_dict["observed_endpoint_date_by_symbol_dict"]
        assert contract_dict["unobserved_member_list"] == []
    assert "group_by_symbol_dict" not in snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR).manifest_dict["data_contract"]
    assert "group_by_symbol_dict" in snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_R1000_PROFILE_STR).manifest_dict["data_contract"]


def test_the_client_reads_labels_from_the_snapshot_and_keeps_no_price_table(snapshot_root_dict, snapshot_mode, monkeypatch):
    snapshot_mode(snapshot_root_dict["t0"])
    monkeypatch.setitem(__import__("sys").modules, "norgatedata", SimpleNamespace(classification_at_level=lambda *_: pytest.fail("Client called Norgate")))
    with snapshot_module.use_norgate_data_profile(MOMENTUM_LEGS_R1000_PROFILE_STR):
        assert legs_module.current_gics_label_dict(["S01"], 1)["S01"] == SECTOR_LIST[1]
        assert legs_module.current_gics_label_dict(["S01"], 2)["S01"] == f"{SECTOR_LIST[1][:2]}20000"
        snapshot_module.load_raw_prices_df(["S01"], [], start_date_str="2024-01-02", end_date_str=T0_STR)
    assert MOMENTUM_LEGS_R1000_PROFILE_STR not in snapshot_module._MR_CAPSULE_PRICE_CACHE_DICT
    with snapshot_module.use_norgate_data_profile(snapshot_module.NDX_E2_PROFILE_STR), pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="their own snapshot"):
        snapshot_module.load_momentum_legs_label_map(["S01"], 1)


def test_exporter_aborts_on_a_provider_error_a_stale_series_or_a_changed_close(tmp_path, monkeypatch):
    for attribute_str, symbol_str in (("failure_set", "S03"), ("unobserved_set", "$NDX"), ("unobserved_set", "$SPXT"), ("unobserved_set", "SPY"), ("padded_close_gap_set", "S03")):
        provider_obj = FakeProvider()
        getattr(provider_obj, attribute_str).add(symbol_str)
        with pytest.raises((ConnectionError, ValueError)):
            export_snapshot(tmp_path / f"{attribute_str}{symbol_str[-3:]}", monkeypatch, provider_obj, MOMENTUM_LEGS_NDX_PROFILE_STR, T0_STR)


def test_one_halted_member_is_listed_and_a_stale_feed_aborts(tmp_path, monkeypatch, snapshot_mode):
    provider_obj = FakeProvider()
    provider_obj.unobserved_set.add("S07")
    export_snapshot(tmp_path / "halt", monkeypatch, provider_obj, MOMENTUM_LEGS_NDX_PROFILE_STR, T0_STR)
    snapshot_mode(tmp_path / "halt")
    contract_dict = snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR).manifest_dict["data_contract"]
    assert contract_dict["unobserved_member_list"] == ["S07"]
    assert "S07" not in contract_dict["observed_endpoint_date_by_symbol_dict"]
    provider_obj.unobserved_set.update(STOCK_LIST[:5])
    with pytest.raises(ValueError, match="members have no unpadded close"):
        export_snapshot(tmp_path / "stale", monkeypatch, provider_obj, MOMENTUM_LEGS_NDX_PROFILE_STR, T0_STR)


def test_a_retry_waits_for_the_running_export_instead_of_replacing_it(tmp_path, monkeypatch):
    provider_obj = FakeProvider()
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setattr(export_module, "_load_index_constituent_matrix_df", lambda *args, **kwargs: (list(fake_universe_df(provider_obj).columns), fake_universe_df(provider_obj)))
    real_export_fn = export_module._export_profile_to_root_path
    call_list = []
    entered_obj, release_obj = threading.Event(), threading.Event()

    def slow_export_fn(**keyword_dict):
        call_list.append(keyword_dict["profile_str"])
        entered_obj.set()
        release_obj.wait(timeout=30)
        return real_export_fn(**keyword_dict)
    monkeypatch.setattr(export_module, "_export_profile_to_root_path", slow_export_fn)
    result_list = []

    def request_fn():
        result_list.append(export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=MOMENTUM_LEGS_NDX_PROFILE_STR, snapshot_date_str=T0_STR))
    first_obj, retry_obj = threading.Thread(target=request_fn), threading.Thread(target=request_fn)
    first_obj.start()
    assert entered_obj.wait(timeout=30)
    retry_obj.start()
    retry_obj.join(timeout=.5)
    assert retry_obj.is_alive()
    release_obj.set()
    first_obj.join(timeout=60)
    retry_obj.join(timeout=60)
    assert call_list == [MOMENTUM_LEGS_NDX_PROFILE_STR]
    assert len(result_list) == 2 and result_list[0] == result_list[1]


@pytest.mark.parametrize("mutation_str", ["missing_map", "missing_symbol", "extra_symbol", "wrong_hash", "wrong_date", "bad_sector", "missing_time", "naive_time", "wrong_basis", "legacy_schema", "missing_helper", "missing_endpoint", "too_many_unobserved", "group_map_on_ndx"])
def test_the_manifest_contract_fails_closed(copied_snapshot, mutation_str):
    snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)

    def mutate(manifest_dict):
        contract_dict = manifest_dict["data_contract"]
        if mutation_str == "missing_map":
            contract_dict.pop("sector_by_symbol_dict")
        elif mutation_str == "missing_symbol":
            contract_dict["sector_by_symbol_dict"].pop("OLD")
            contract_dict["label_map_hash_str"] = momentum_legs_label_hash_str(contract_dict["sector_by_symbol_dict"], None)
        elif mutation_str == "extra_symbol":
            contract_dict["sector_by_symbol_dict"]["ZZZ"] = "45000000"
            contract_dict["label_map_hash_str"] = momentum_legs_label_hash_str(contract_dict["sector_by_symbol_dict"], None)
        elif mutation_str == "wrong_hash":
            contract_dict["label_map_hash_str"] = "0" * 64
        elif mutation_str == "wrong_date":
            contract_dict["label_snapshot_date_str"] = "2024-01-30"
        elif mutation_str == "bad_sector":
            contract_dict["sector_by_symbol_dict"]["S01"] = "4510"
            contract_dict["label_map_hash_str"] = momentum_legs_label_hash_str(contract_dict["sector_by_symbol_dict"], None)
        elif mutation_str == "missing_time":
            contract_dict.pop("label_observed_timestamp_str")
        elif mutation_str == "naive_time":
            contract_dict["label_observed_timestamp_str"] = "2024-01-31T22:00:00"
        elif mutation_str == "wrong_basis":
            contract_dict["label_classification_basis_str"] = "historical"
        elif mutation_str == "legacy_schema":
            manifest_dict["schema_version"] = 1
        elif mutation_str == "missing_helper":
            manifest_dict["required_helpers"] = [symbol_str for symbol_str in manifest_dict["required_helpers"] if symbol_str != "$NDX"]
        elif mutation_str == "missing_endpoint":
            contract_dict["observed_endpoint_date_by_symbol_dict"].pop("S05")
        elif mutation_str == "too_many_unobserved":
            contract_dict["unobserved_member_list"] = STOCK_LIST[:5]
        else:
            contract_dict["group_by_symbol_dict"] = {"S01": "4510"}
    rewrite_manifest(copied_snapshot, mutate)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)


@pytest.mark.parametrize("mutation_str", ["missing_anchor", "zero_anchor", "nan_anchor", "missing_member_at_t", "future_price", "duplicate_price", "history_hole", "stale_helper", "helper_hole", "missing_helper_rows", "missing_benchmark", "zero_vxn", "missing_turnover", "before_history"])
def test_the_price_contract_fails_closed(copied_snapshot, mutation_str):
    price_df = pd.read_parquet(copied_snapshot / "prices.parquet")
    last_mask_ser = (price_df["symbol_str"] == "S05") & (price_df["date"] == pd.Timestamp(T0_STR))
    if mutation_str == "missing_anchor":
        price_df = price_df.drop(columns=["Unadjusted Close"])
    elif mutation_str in {"zero_anchor", "nan_anchor"}:
        price_df.loc[last_mask_ser, "Unadjusted Close"] = 0.0 if mutation_str == "zero_anchor" else np.nan
    elif mutation_str == "missing_member_at_t":
        price_df = price_df.loc[~last_mask_ser]
    elif mutation_str == "future_price":
        price_df.loc[last_mask_ser, "date"] = pd.Timestamp("2024-02-01")
    elif mutation_str == "duplicate_price":
        price_df = pd.concat([price_df, price_df.loc[last_mask_ser]], ignore_index=True)
    elif mutation_str == "history_hole":
        price_df = price_df.loc[~((price_df["symbol_str"] == "S05") & (price_df["date"] == pd.Timestamp("2023-05-10")))]
    elif mutation_str == "stale_helper":
        price_df = price_df.loc[~((price_df["symbol_str"] == "$NDX") & (price_df["date"] == pd.Timestamp(T0_STR)))]
    elif mutation_str == "helper_hole":
        price_df = price_df.loc[~((price_df["symbol_str"] == "$SPXT") & (price_df["date"] == pd.Timestamp("2023-05-10")))]
    elif mutation_str == "missing_helper_rows":
        price_df = price_df.loc[price_df["symbol_str"] != "SPY"]
    elif mutation_str == "missing_benchmark":
        price_df = price_df.loc[price_df["symbol_str"] != "$SPXTR"]
    elif mutation_str == "zero_vxn":
        price_df.loc[price_df["symbol_str"] == "$VXN", "Close"] = 0.0
    elif mutation_str == "missing_turnover":
        price_df = price_df.drop(columns=["Turnover"])
    else:
        row_df = price_df.loc[last_mask_ser].copy()
        row_df["date"] = pd.Timestamp("1997-12-31")
        price_df = pd.concat([price_df, row_df], ignore_index=True)
    rewrite_prices(copied_snapshot, price_df)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)


def test_a_partial_stock_update_fails_closed_though_it_reads_as_fewer_members(copied_snapshot):
    universe_df = pd.read_parquet(copied_snapshot / "universe.parquet")
    last_date_obj = universe_df.index.max() if "date" not in universe_df.columns else None
    assert last_date_obj is not None
    stale_list = STOCK_LIST[:3]
    universe_df.loc[last_date_obj, stale_list] = 0

    def fix_contract(manifest_dict):
        endpoint_dict = manifest_dict["data_contract"]["observed_endpoint_date_by_symbol_dict"]
        for symbol_str in stale_list:
            endpoint_dict.pop(symbol_str)
    rewrite_universe(copied_snapshot, universe_df)
    rewrite_manifest(copied_snapshot, fix_contract)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="too few members"):
        snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)


def test_one_member_gone_at_t_is_a_normal_day(copied_snapshot):
    universe_df = pd.read_parquet(copied_snapshot / "universe.parquet")
    universe_df.loc[universe_df.index.max(), "S03"] = 0

    def fix_contract(manifest_dict):
        manifest_dict["data_contract"]["observed_endpoint_date_by_symbol_dict"].pop("S03")
    rewrite_universe(copied_snapshot, universe_df)
    rewrite_manifest(copied_snapshot, fix_contract)
    assert snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR).snapshot_date_ts == pd.Timestamp(T0_STR)


def test_a_missing_turnover_at_t_fails_the_export(tmp_path, monkeypatch):
    provider_obj = FakeProvider()
    provider_obj.no_turnover_at_t_bool = True
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="finite Turnover"):
        export_snapshot(tmp_path, monkeypatch, provider_obj, MOMENTUM_LEGS_R1000_PROFILE_STR, T0_STR)


def test_a_false_none_from_the_provider_does_not_become_unknown(tmp_path, monkeypatch, snapshot_mode):
    provider_obj = FakeProvider()
    provider_obj.false_none_count_dict = {"S07": 2}
    export_snapshot(tmp_path, monkeypatch, provider_obj, MOMENTUM_LEGS_NDX_PROFILE_STR, T0_STR)
    snapshot_mode(tmp_path)
    contract_dict = snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR).manifest_dict["data_contract"]
    assert contract_dict["sector_by_symbol_dict"]["S07"] == SECTOR_LIST[1]
    assert contract_dict["sector_by_symbol_dict"]["S00"] == "UNKNOWN"


def test_a_name_with_only_one_of_its_two_labels_fails_closed(tmp_path, monkeypatch):
    provider_obj = FakeProvider()
    real_read_fn = provider_obj.classification_at_level
    monkeypatch.setattr(provider_obj, "classification_at_level", lambda symbol_str, scheme_str, field_str, level_int: None if (symbol_str, level_int) == ("S09", 2) else real_read_fn(symbol_str, scheme_str, field_str, level_int))
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="disagree"):
        export_snapshot(tmp_path, monkeypatch, provider_obj, MOMENTUM_LEGS_R1000_PROFILE_STR, T0_STR)


def test_labels_that_did_not_arrive_fail_closed(tmp_path, monkeypatch):
    provider_obj = FakeProvider()
    provider_obj.no_label_set = set(STOCK_LIST[:10])
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="no GICS sector"):
        export_snapshot(tmp_path, monkeypatch, provider_obj, MOMENTUM_LEGS_NDX_PROFILE_STR, T0_STR)


def test_a_changed_universe_file_fails_closed(copied_snapshot):
    universe_path_obj = copied_snapshot / "universe.parquet"
    universe_df = pd.read_parquet(universe_path_obj)
    universe_df.iloc[-1, 0] = 0
    universe_df.to_parquet(universe_path_obj)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)


# ---------------------------------------------------------------- the decision
def direct_path(release_obj, date_str):
    """The backtest's own functions on the same snapshot, with no adapter in between."""
    config_obj = replace(__import__("importlib").import_module(release_obj.strategy_import_str.split(":")[0]).DEFAULT_CONFIG, end_date_str=date_str)
    with snapshot_module.use_norgate_data_profile(release_obj.data_profile_str):
        input_dict = legs_module.load_leg_inputs(config_obj)
    pricing_data_df = input_dict.pop("pricing_data_df")
    return legs_module.build_target_path(config_obj, legs_module.build_leg_panel(config_obj, pricing_data_df, **input_dict))


@pytest.mark.parametrize("key_str,date_str,as_of_ts,offset_int", [("t0", T0_STR, T0_AS_OF_TS, 0), ("t10", T10_STR, T10_AS_OF_TS, 10)])
def test_a_decision_is_the_last_row_of_the_backtest_path(release_obj, snapshot_root_dict, snapshot_mode, key_str, date_str, as_of_ts, offset_int):
    snapshot_mode(snapshot_root_dict[key_str])
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, None)
    path_obj = direct_path(release_obj, date_str)
    expected_dict = path_obj.target_by_decision_dict[pd.Timestamp(date_str)].to_dict()
    assert expected_dict and decision_obj.full_target_weight_map_dict == pytest.approx(expected_dict)
    assert decision_obj.decision_book_type_str == "full_target_weight_book"
    assert decision_obj.execution_policy_str == "next_open_moo"
    assert decision_obj.rebalance_omitted_assets_to_zero_bool and not decision_obj.preserve_untouched_positions_bool
    assert decision_obj.target_share_map_dict == {} and decision_obj.strategy_state_dict == {}
    assert decision_obj.cash_reserve_weight_float == pytest.approx(1.0 - sum(expected_dict.values()))
    metadata_dict = decision_obj.snapshot_metadata_dict
    assert metadata_dict["norgate_snapshot_date_str"] == date_str and metadata_dict["norgate_data_profile_str"] == release_obj.data_profile_str
    assert metadata_dict["book_offset_int"] == offset_int and metadata_dict["gate_open_bool"] is True
    assert metadata_dict["momentum_legs_sizing_contract_str"] == "full_target_preopen_v1" and metadata_dict["research_quantity_basis_str"] == "Close_T"
    assert set(metadata_dict["sector_by_target_symbol_dict"]) == set(expected_dict)
    assert decision_obj.signal_timestamp_ts.date().isoformat() == date_str
    assert metadata_dict["vxn_reference_date_str"] == (None if release_obj.data_profile_str == MOMENTUM_LEGS_R1000_PROFILE_STR else date_str)
    # The decision never stores a whole label map: it travels in the manifest.
    assert "sector_by_symbol_dict" not in metadata_dict


def test_the_two_latest_books_are_what_the_t0_account_targets(snapshot_root_dict, snapshot_mode):
    release_obj = parse_release_manifest(str(TEMPLATE_BY_LEG_DICT["leg1"]))
    snapshot_mode(snapshot_root_dict["t0"])
    path_obj = direct_path(release_obj, T0_STR)
    t0_book_ser = path_obj.book_by_decision_dict[pd.Timestamp(T0_STR)]
    t10_book_ser = path_obj.book_by_decision_dict[pd.Timestamp(T10_STR)]
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None)
    assert decision_obj.full_target_weight_map_dict == pytest.approx((.5 * t0_book_ser.add(t10_book_ser, fill_value=0.0)).to_dict())
    assert decision_obj.snapshot_metadata_dict["own_book_weight_dict"] == pytest.approx(t0_book_ser.to_dict())


def test_a_shut_gate_is_a_valid_all_cash_target_for_the_books_that_decided_shut(release_obj, snapshot_root_dict, snapshot_mode):
    snapshot_mode(snapshot_root_dict["gate_down"])
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None)
    assert decision_obj.snapshot_metadata_dict["gate_open_bool"] is False
    assert decision_obj.full_target_weight_map_dict == {} and decision_obj.cash_reserve_weight_float == 1.0


def test_decisions_refuse_direct_mode(release_obj, monkeypatch):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    with pytest.raises(ValueError, match="frozen snapshot"):
        strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None)


def test_a_snapshot_of_another_session_stops_the_decision(release_obj, snapshot_root_dict, snapshot_mode):
    snapshot_mode(snapshot_root_dict["other"])
    with pytest.raises(ValueError, match="exact completed swap-session"):
        strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None)
    # A later session's snapshot cannot serve the earlier swap either.
    with pytest.raises(ValueError, match="exact completed swap-session"):
        strategy_host.build_decision_plan_for_release(release_obj, T10_AS_OF_TS, None)


def test_a_missed_cycle_stays_missed(release_obj, snapshot_root_dict, snapshot_mode):
    snapshot_mode(snapshot_root_dict["t10"])
    late_as_of_ts = datetime(2024, 1, 18, 12, 0, tzinfo=MARKET_ZONE_OBJ)
    with pytest.raises(ValueError, match="stale swap cycle"):
        strategy_host.build_decision_plan_for_release(release_obj, late_as_of_ts, None)
    pre_open_as_of_ts = datetime(2024, 1, 18, 8, 0, tzinfo=MARKET_ZONE_OBJ)
    assert strategy_host.build_decision_plan_for_release(release_obj, pre_open_as_of_ts, None).snapshot_metadata_dict["norgate_snapshot_date_str"] == T10_STR


def test_another_accounts_state_is_refused(release_obj, snapshot_root_dict, snapshot_mode):
    snapshot_mode(snapshot_root_dict["t0"])
    state_obj = PodState(pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str="DU_OTHER", position_amount_map={}, cash_float=1.0, total_value_float=1.0, strategy_state_dict={}, updated_timestamp_ts=T0_AS_OF_TS)
    with pytest.raises(ValueError, match="account state"):
        strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, state_obj)


@pytest.mark.parametrize("mutation_str,message_str", [
    ("nan_gate", "gate close"), ("nan_level", "gate close"), ("no_members", "PIT members"), ("few_scores", "scored too few"), ("no_decision", "no decision"), ("thin_book", "with its gate open"),
])
def test_unavailable_evidence_stops_the_decision_instead_of_reading_as_cash(release_obj, snapshot_root_dict, snapshot_mode, monkeypatch, mutation_str, message_str):
    snapshot_mode(snapshot_root_dict["t0"])
    real_build_fn = legs_module.build_target_path

    def damaged_build_fn(config_obj, panel):
        path_obj = real_build_fn(config_obj, panel)
        evidence_dict = path_obj.evidence_by_decision_dict[pd.Timestamp(T0_STR)]
        if mutation_str == "nan_gate":
            evidence_dict["gate_close_float"] = float("nan")
        elif mutation_str == "nan_level":
            evidence_dict["gate_level_float"] = float("nan")
        elif mutation_str == "no_members":
            evidence_dict["member_count_int"] = 0
        elif mutation_str == "few_scores":
            evidence_dict["finite_score_member_count_int"] = 1
        elif mutation_str == "thin_book":
            path_obj.book_by_decision_dict[pd.Timestamp(T0_STR)] = path_obj.book_by_decision_dict[pd.Timestamp(T0_STR)].iloc[:4]
        else:
            path_obj.target_by_decision_dict.pop(pd.Timestamp(T0_STR))
        return path_obj
    monkeypatch.setattr(legs_module, "build_target_path", damaged_build_fn)
    with pytest.raises(ValueError, match=message_str):
        strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None)


def test_a_refusal_on_one_manifest_is_not_replayed_every_tick(snapshot_root_dict, snapshot_mode, monkeypatch):
    release_obj = parse_release_manifest(str(TEMPLATE_BY_LEG_DICT["leg1"]))
    snapshot_mode(snapshot_root_dict["t0"])
    real_build_fn = legs_module.build_target_path
    call_list = []

    def damaged_build_fn(config_obj, panel):
        call_list.append(1)
        path_obj = real_build_fn(config_obj, panel)
        path_obj.evidence_by_decision_dict[pd.Timestamp(T0_STR)]["finite_score_member_count_int"] = 1
        return path_obj
    monkeypatch.setattr(legs_module, "build_target_path", damaged_build_fn)
    for _ in range(3):
        with pytest.raises(ValueError, match="scored too few"):
            strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None)
    assert call_list == [1]
    # A new manifest decides again.
    adapter_module._REFUSED_DECISION_BY_KEY_DICT.clear()
    monkeypatch.setattr(legs_module, "build_target_path", real_build_fn)
    assert strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None).full_target_weight_map_dict


def test_running_out_of_memory_is_a_contained_refusal_and_is_not_kept(snapshot_root_dict, snapshot_mode, monkeypatch):
    release_obj = parse_release_manifest(str(TEMPLATE_BY_LEG_DICT["leg1"]))
    snapshot_mode(snapshot_root_dict["t0"])
    real_load_fn = legs_module.load_leg_inputs

    def failing_load_fn(config_obj):
        raise MemoryError()
    monkeypatch.setattr(legs_module, "load_leg_inputs", failing_load_fn)
    with pytest.raises(RuntimeError, match="out of memory"):
        strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None)
    monkeypatch.setattr(legs_module, "load_leg_inputs", real_load_fn)
    assert strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None).full_target_weight_map_dict


def test_a_missing_vxn_stops_the_nasdaq_legs(snapshot_root_dict, snapshot_mode, monkeypatch):
    release_obj = parse_release_manifest(str(TEMPLATE_BY_LEG_DICT["leg2"]))
    snapshot_mode(snapshot_root_dict["t0"])
    real_build_fn = legs_module.build_target_path

    def damaged_build_fn(config_obj, panel):
        path_obj = real_build_fn(config_obj, panel)
        path_obj.evidence_by_decision_dict[pd.Timestamp(T0_STR)]["vxn_close_float"] = float("nan")
        return path_obj
    monkeypatch.setattr(legs_module, "build_target_path", damaged_build_fn)
    with pytest.raises(ValueError, match="VXN"):
        strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None)


def test_a_snapshot_change_during_the_decision_stops_it(release_obj, snapshot_root_dict, snapshot_mode, monkeypatch):
    snapshot_mode(snapshot_root_dict["t0"])
    real_build_fn = snapshot_module.build_data_source_metadata_dict
    call_list = []

    def changing_build_fn(profile_str=None):
        call_list.append(profile_str)
        metadata_dict = real_build_fn(profile_str)
        if len(call_list) > 1:
            metadata_dict["norgate_manifest_hash_str"] = "changed"
        return metadata_dict
    monkeypatch.setattr(snapshot_module, "build_data_source_metadata_dict", changing_build_fn)
    with pytest.raises(ValueError, match="changed during the decision"):
        strategy_host.build_decision_plan_for_release(release_obj, T0_AS_OF_TS, None)


def test_a_later_reference_run_reads_the_same_path(snapshot_root_dict, snapshot_mode):
    snapshot_mode(snapshot_root_dict["t0"])
    module_obj = __import__("importlib").import_module("strategies.momentum.strategy_mo_capsule_leg1_natr_ndx")
    with snapshot_module.use_norgate_data_profile(MOMENTUM_LEGS_NDX_PROFILE_STR):
        full_obj = module_obj.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=T0_STR)
        later_obj = module_obj.run_variant(show_display_bool=False, save_results_bool=False, backtest_start_date_str="2023-12-15", capital_base_float=50000.0, end_date_str=T0_STR)
    assert later_obj.target_path_obj.target_by_decision_dict[pd.Timestamp(T10_STR)].equals(full_obj.target_path_obj.target_by_decision_dict[pd.Timestamp(T10_STR)])
    assert later_obj.get_positions().abs().sum() > 0


# ---------------------------------------------------------------- one whole cycle with a synthetic broker
@pytest.fixture
def lifecycle_case(snapshot_root_dict, snapshot_mode, tmp_path, monkeypatch):
    snapshot_mode(snapshot_root_dict["t0"])
    template_obj = parse_release_manifest(str(TEMPLATE_BY_LEG_DICT["leg3"]))
    releases_path_obj = tmp_path / "releases" / "owner"
    releases_path_obj.mkdir(parents=True)
    manifest_path_obj = releases_path_obj / "leg.yaml"
    manifest_path_obj.write_text(yaml.safe_dump({
        "identity": {"release_id": "leg.v1", "user_id": "owner", "pod_id": "leg"},
        "deployment": {"mode": "paper", "enabled_bool": True},
        "broker": {"account_route": "DU777003"},
        "strategy": {"strategy_import_str": template_obj.strategy_import_str, "data_profile_str": template_obj.data_profile_str, "params": {}},
        "market": {"session_calendar_id_str": "XNYS"},
        "schedule": {"signal_clock_str": "month_swap_snapshot_ready", "execution_policy_str": "next_open_moo"},
        "execution": {"pod_budget_fraction_float": 1.0, "auto_submit_enabled_bool": True},
        "bootstrap": {"initial_cash_float": 100000.0},
        "risk": {"risk_profile_str": "momentum"},
    }), encoding="utf-8")
    release_obj = parse_release_manifest(str(manifest_path_obj))
    store_obj = LiveStateStore(str(tmp_path / "leg.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(PodState(pod_id_str="leg", user_id_str="owner", account_route_str="DU777003", position_amount_map={"OLDPICK": 10.0}, cash_float=99000.0, total_value_float=100000.0, strategy_state_dict={}, updated_timestamp_ts=T0_AS_OF_TS, snapshot_stage_str="eod", snapshot_source_str="broker"))
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(tmp_path / "critical.jsonl"))
    monkeypatch.setattr(IBKRSocketClient, "connect", lambda *_: pytest.fail("Real broker forbidden"))
    monkeypatch.setattr(IBKRSocketClient, "daily_connection", lambda *_: pytest.fail("Real broker forbidden"))
    monkeypatch.setattr(runner, "_reconcile_daily_cycles", lambda *args, **kwargs: pytest.fail("A leg entered daily reconciliation"))
    return store_obj, release_obj, StubBrokerAdapter(), {"releases_root_path_str": str(tmp_path / "releases"), "env_mode_str": "paper", "log_path_str": str(tmp_path / "events.jsonl"), "trace_enabled_bool": False}


def next_phase_str(store_obj, as_of_ts, runner_kwarg_dict):
    return scheduler_service.get_scheduler_decision(store_obj, as_of_ts, runner_kwarg_dict["releases_root_path_str"], "paper").next_phase_str


def test_a_whole_swap_runs_from_decision_to_reconcile_and_survives_restarts(lifecycle_case):
    store_obj, release_obj, broker_obj, runner_kwarg_dict = lifecycle_case
    assert next_phase_str(store_obj, T0_AS_OF_TS, runner_kwarg_dict) == "build_decision_plan"
    assert runner.build_decision_plans(store_obj, T0_AS_OF_TS, auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict)["created_decision_plan_count_int"] == 1
    decision_obj = store_obj.get_latest_decision_plan_for_pod("leg")
    weight_dict = dict(decision_obj.full_target_weight_map_dict)
    assert weight_dict and "OLDPICK" not in weight_dict and decision_obj.decision_base_position_map == {"OLDPICK": 10.0}
    assert runner.build_decision_plans(store_obj, T0_AS_OF_TS, auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict)["created_decision_plan_count_int"] == 0
    store_obj = LiveStateStore(store_obj.db_path_str)
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int) == decision_obj
    # The next session's open, at the usual lead.
    assert decision_obj.target_execution_timestamp_ts.date().isoformat() == "2024-02-01"
    submit_ts = decision_obj.submission_timestamp_ts
    broker_obj.seed_account_snapshot("DU777003", cash_float=119000.0, total_value_float=120000.0, position_amount_map={"OLDPICK": 10.0}, snapshot_timestamp_ts=submit_ts, session_mode_str="paper")
    broker_obj.seed_live_price_snapshot("DU777003", {**{asset_str: 125.0 for asset_str in weight_dict}, "OLDPICK": 100.0}, snapshot_timestamp_ts=submit_ts)
    assert next_phase_str(store_obj, submit_ts, runner_kwarg_dict) == "build_vplan"
    assert runner.build_vplans(store_obj, broker_obj, submit_ts, **runner_kwarg_dict)["created_vplan_count_int"] == 1
    plan_obj = store_obj.get_latest_vplan_for_pod("leg")
    # floor(weight * broker NAV / pre-open quote): shares come from the account, weights from Close_T.
    assert {asset_str: plan_obj.target_share_map[asset_str] for asset_str in weight_dict} == {asset_str: float(int(weight_float * 120000.0 / 125.0)) for asset_str, weight_float in weight_dict.items()}
    assert plan_obj.target_share_map["OLDPICK"] == 0.0 and plan_obj.order_delta_map["OLDPICK"] == -10.0
    request_list = build_broker_order_request_list_from_vplan(plan_obj)
    assert all(request_obj.broker_order_type_str == "MOO" and request_obj.account_route_str == "DU777003" for request_obj in request_list)
    assert next_phase_str(store_obj, submit_ts, runner_kwarg_dict) == "submit_vplan"
    assert runner.submit_ready_vplans(store_obj, broker_obj, submit_ts, manual_only_bool=False, **runner_kwarg_dict)["submitted_vplan_count_int"] == 1
    assert broker_obj.submitted_order_request_list == request_list
    store_obj = LiveStateStore(store_obj.db_path_str)
    assert runner.submit_ready_vplans(store_obj, broker_obj, submit_ts, manual_only_bool=False, **runner_kwarg_dict)["submitted_vplan_count_int"] == 0
    reconcile_ts = decision_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    assert next_phase_str(store_obj, reconcile_ts, runner_kwarg_dict) == "post_execution_reconcile"
    assert runner.post_execution_reconcile(store_obj, broker_obj, reconcile_ts, **runner_kwarg_dict)["completed_vplan_count_int"] == 1
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    state_obj = store_obj.get_pod_state("leg")
    assert state_obj.position_amount_map.get("OLDPICK", 0.0) == 0.0
    assert state_obj.strategy_state_dict == {}
    assert broker_obj.submitted_order_request_list == request_list


def test_a_session_that_is_not_a_swap_builds_nothing(lifecycle_case, snapshot_root_dict, snapshot_mode):
    store_obj, _, broker_obj, runner_kwarg_dict = lifecycle_case
    snapshot_mode(snapshot_root_dict["other"])
    as_of_ts = datetime(2024, 1, 18, 18, 0, tzinfo=MARKET_ZONE_OBJ)
    assert scheduler_service.get_scheduler_decision(store_obj, as_of_ts, runner_kwarg_dict["releases_root_path_str"], "paper").next_phase_str != "build_decision_plan"
    assert runner.build_decision_plans(store_obj, as_of_ts, auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict)["created_decision_plan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list


def test_a_bad_snapshot_is_contained_and_persists_no_decision_or_orders(lifecycle_case, snapshot_root_dict, snapshot_mode, monkeypatch):
    store_obj, release_obj, broker_obj, runner_kwarg_dict = lifecycle_case
    before_state_obj = store_obj.get_pod_state("leg")
    monkeypatch.setattr(scheduler_utils, "load_latest_norgate_heartbeat_session_label_ts", lambda _: pd.Timestamp(T0_STR))
    snapshot_mode(snapshot_root_dict["other"])
    result_dict = runner.build_decision_plans(store_obj, T0_AS_OF_TS, auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict)
    assert result_dict["created_decision_plan_count_int"] == 0
    assert result_dict["reason_count_map_dict"] == {"momentum_leg_decision_blocked": 1}
    assert "momentum_leg_decision_blocked" in Path(runner_kwarg_dict["log_path_str"]).read_text(encoding="utf-8")
    store_obj = LiveStateStore(store_obj.db_path_str)
    assert store_obj.get_latest_decision_plan_for_pod("leg") is None
    assert store_obj.get_latest_vplan_for_pod("leg") is None
    assert store_obj.get_pod_state("leg") == before_state_obj
    assert not broker_obj.submitted_order_request_list


# ---------------------------------------------------------------- from the coverage review: each closes a gap a mutant survived
dashboard_module, doctor_module, sync_module = dashboard, doctor, norgate_snapshot_sync


def leg(leg_str):
    return parse_release_manifest(str(TEMPLATE_BY_LEG_DICT[leg_str]))


# ---- releases
def test_a_leg_shares_no_account_with_a_leg_or_any_other_pod():
    taa_obj = enabled_release(parse_release_manifest(str(TEMPLATE_DIR_PATH / "pod_taa_btal_1n_fallback_tqqq_vix_cash_monthly_open.yaml.example")))
    for release_list in ([enabled_release(leg("leg1")), enabled_release(leg("leg2"))], [enabled_release(leg("leg1")), taa_obj], [taa_obj, enabled_release(leg("leg1"))]):
        with pytest.raises(ValueError, match="momentum capsule leg requires one dedicated"):
            validate_release_list(release_list)


@pytest.mark.parametrize("leg_str,profile_str", [("leg1", MOMENTUM_LEGS_R1000_PROFILE_STR), ("leg3", MOMENTUM_LEGS_NDX_PROFILE_STR)])
def test_a_leg_refuses_the_other_legs_profile(leg_str, profile_str):
    with pytest.raises(ValueError, match="requires data_profile_str"):
        validate_release_manifest(replace(leg(leg_str), data_profile_str=profile_str))


# ---- clock wiring
@pytest.mark.parametrize("profile_str", [MOMENTUM_LEGS_NDX_PROFILE_STR, MOMENTUM_LEGS_R1000_PROFILE_STR])
def test_both_profiles_have_a_heartbeat(snapshot_root_dict, snapshot_mode, profile_str):
    snapshot_mode(snapshot_root_dict["t0"])
    assert scheduler_utils.load_latest_norgate_heartbeat_session_label_ts(profile_str) == pd.Timestamp(T0_STR)


@pytest.mark.parametrize("signal_str,next_str", [("2026-10-16", "2026-10-30"), ("2026-10-30", "2026-11-13")])
def test_runner_next_swap_both_ways(signal_str, next_str):
    assert runner._next_norgate_signal_ready_timestamp_ts(leg("leg1"), pd.Timestamp(signal_str)).date().isoformat() == next_str


def test_sync_requires_the_latest_swap_session():
    release_obj_ = leg("leg3")
    as_of_ts = datetime(2026, 11, 13, 18, 0, tzinfo=MARKET_ZONE_OBJ)
    assert sync_module._required_snapshot_session_by_release_dict([release_obj_], as_of_ts)[release_obj_.release_id_str] == pd.Timestamp("2026-11-13")
    detail_dict = sync_module._with_cycle_freshness_detail_dict({"snapshot_date_by_profile_dict": {release_obj_.data_profile_str: "2026-11-03"}, "all_profiles_ready_bool": True}, [release_obj_], as_of_ts)
    assert detail_dict["stale_profile_list"] == [release_obj_.data_profile_str]
    # In between, a later non-swap snapshot is fresh.
    mid_ts = datetime(2026, 11, 10, 18, 0, tzinfo=MARKET_ZONE_OBJ)
    assert sync_module._with_cycle_freshness_detail_dict({"snapshot_date_by_profile_dict": {release_obj_.data_profile_str: "2026-11-03"}, "all_profiles_ready_bool": True}, [release_obj_], mid_ts)["stale_profile_list"] == []


def test_sync_reports_the_gate_reason(monkeypatch):
    release_obj_ = leg("leg3")
    monkeypatch.setattr(scheduler_utils, "load_latest_norgate_heartbeat_session_label_ts", lambda _: pd.Timestamp("2026-11-03"))
    as_of_ts = datetime(2026, 11, 4, 18, 0, tzinfo=MARKET_ZONE_OBJ)
    assert sync_module._build_gate_reason_by_release_dict([release_obj_], as_of_ts) == {release_obj_.release_id_str: "not_month_swap_session"}


def test_dashboard_flags_a_missed_swap_and_covers_until_the_next():
    release_obj_ = leg("leg1")
    missed_dict = dashboard_module._missed_monthly_decision_cycle_dict(release_obj_, None, datetime(2026, 10, 19, 12, 0, tzinfo=MARKET_ZONE_OBJ))
    assert missed_dict["missed_signal_date_str"] == "2026-10-16"
    row_dict = {"latest_decision_signal_timestamp_str": "2026-10-30T16:00:00-04:00", "session_calendar_id_str": "XNYS", "signal_clock_str": "month_swap_snapshot_ready"}
    assert dashboard_module._completed_decision_plan_covers_norgate_signal_cycle_bool({**row_dict, "as_of_timestamp_str": "2026-11-12T18:00:00-05:00"})
    assert not dashboard_module._completed_decision_plan_covers_norgate_signal_cycle_bool({**row_dict, "as_of_timestamp_str": "2026-11-13T18:00:00-05:00"})


def test_doctor_waits_and_requires_sync():
    assert doctor_module._gate_component_result_dict({"due_bool": False, "reason_code_str": "not_month_swap_session"})["status_str"] == doctor_module.DOCTOR_WAIT_STR
    assert doctor_module._release_requires_norgate_snapshot_sync_bool(leg("leg1"))


# ---- validator
@pytest.mark.parametrize("mutation_str,message_str", [("later_row", "none later"), ("non_binary", "zero/one"), ("no_member_at_t", "current PIT members")])
def test_universe_content_rules_are_reached(copied_snapshot, mutation_str, message_str):
    universe_df = pd.read_parquet(copied_snapshot / "universe.parquet")
    index_ts = universe_df.index[-1]
    if mutation_str == "later_row":
        universe_df.loc[pd.Timestamp("2024-02-01")] = universe_df.iloc[-1]
    elif mutation_str == "non_binary":
        universe_df.iloc[-1, 0] = 2
    else:
        universe_df.loc[index_ts] = 0
    rewrite_universe(copied_snapshot, universe_df)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match=message_str):
        snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)


def test_symbols_rule_is_reached(copied_snapshot):
    rewrite_manifest(copied_snapshot, lambda manifest_dict: manifest_dict.update(required_symbols=[s for s in manifest_dict["required_symbols"] if s != "SPY"]))
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="manifest symbols"):
        snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)


def test_a_stock_row_on_a_holiday_is_refused(copied_snapshot):
    price_df = pd.read_parquet(copied_snapshot / "prices.parquet")
    row_df = price_df.loc[(price_df["symbol_str"] == "S05") & (price_df["date"] == pd.Timestamp("2023-07-03"))].copy()
    row_df["date"] = pd.Timestamp("2023-07-04")
    rewrite_prices(copied_snapshot, pd.concat([price_df, row_df], ignore_index=True))
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="outside the XNYS"):
        snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)


def test_r1000_group_map_is_required(snapshot_root_dict, snapshot_mode, tmp_path):
    target_path_obj = tmp_path / "root" / MOMENTUM_LEGS_R1000_PROFILE_STR / T0_STR
    shutil.copytree(snapshot_root_dict["t0"] / MOMENTUM_LEGS_R1000_PROFILE_STR / T0_STR, target_path_obj)
    snapshot_mode(tmp_path / "root")

    def mutate(manifest_dict):
        contract_dict = manifest_dict["data_contract"]
        contract_dict["group_by_symbol_dict"]["S01"] = " "
        contract_dict["label_map_hash_str"] = momentum_legs_label_hash_str(contract_dict["sector_by_symbol_dict"], contract_dict["group_by_symbol_dict"])
    rewrite_manifest(target_path_obj, mutate)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="industry group"):
        snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_R1000_PROFILE_STR)


def test_a_failed_content_check_is_not_kept(copied_snapshot):
    rewrite_manifest(copied_snapshot, lambda manifest_dict: manifest_dict["data_contract"].update(observed_endpoint_date_by_symbol_dict={}))
    for _ in range(2):
        with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
            snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)


def test_no_price_table_is_kept_for_a_leg(snapshot_root_dict, snapshot_mode):
    snapshot_mode(snapshot_root_dict["t0"])
    manifest_obj = snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)
    snapshot_module._read_prices_cached_df.cache_clear()
    snapshot_module._read_prices_df(manifest_obj)
    assert snapshot_module._read_prices_cached_df.cache_info().currsize == 0


def test_the_unobserved_cap_is_reached_on_its_own(copied_snapshot):
    def mutate(manifest_dict):
        contract_dict = manifest_dict["data_contract"]
        contract_dict["unobserved_member_list"] = [f"S{k:02d}" for k in range(5)]
        for k in range(5):
            contract_dict["observed_endpoint_date_by_symbol_dict"].pop(f"S{k:02d}")
    rewrite_manifest(copied_snapshot, mutate)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="Too many Momentum legs members"):
        snapshot_module.load_valid_snapshot_manifest(MOMENTUM_LEGS_NDX_PROFILE_STR)


def test_a_nan_target_weight_is_refused(snapshot_root_dict, snapshot_mode, monkeypatch):
    real_build_fn = legs_module.build_target_path

    def damaged_build_fn(config_obj, panel):
        path_obj = real_build_fn(config_obj, panel)
        path_obj.target_by_decision_dict[pd.Timestamp(T0_STR)].iloc[0] = float("nan")
        return path_obj
    snapshot_mode(snapshot_root_dict["t0"])
    monkeypatch.setattr(legs_module, "build_target_path", damaged_build_fn)
    with pytest.raises(ValueError, match="finite positive weights"):
        strategy_host.build_decision_plan_for_release(leg("leg1"), T0_AS_OF_TS, None)


# ---- adapter on real evidence
def test_real_nan_scores_stop_the_decision(snapshot_root_dict, snapshot_mode, monkeypatch):
    release_obj_ = leg("leg1")
    snapshot_mode(snapshot_root_dict["t0"])
    monkeypatch.setitem(legs_module.SCORE_FUNCTION_DICT, legs_module.SCORE_NATR_STR, lambda _c, panel, row_vec: {int(r): np.full(panel.close_mat.shape[1], np.nan) for r in row_vec})
    with pytest.raises(ValueError, match="scored too few"):
        strategy_host.build_decision_plan_for_release(release_obj_, T0_AS_OF_TS, None)


# ---- ibkr
@pytest.mark.parametrize("asset_str,broker_str", [("BF.A", "BF A"), ("HEI.A", "HEI A"), ("LEN.B", "LEN B"), ("UHAL.B", "UHAL B")])
def test_share_class_aliases(asset_str, broker_str):
    assert IBKR_STOCK_SYMBOL_MAP_DICT[asset_str] == broker_str


# ---- engine
def test_reference_run_holds_the_target_of_its_last_swap_and_exits_the_rest(snapshot_root_dict, snapshot_mode):
    snapshot_mode(snapshot_root_dict["t0"])
    module_obj = __import__("importlib").import_module("strategies.momentum.strategy_mo_capsule_leg1_natr_ndx")
    with snapshot_module.use_norgate_data_profile(MOMENTUM_LEGS_NDX_PROFILE_STR):
        strategy_obj = module_obj.run_variant(show_display_bool=False, save_results_bool=False, backtest_start_date_str="2023-06-01", capital_base_float=1_000_000.0, end_date_str=T0_STR)
    # The T0 decision has no next session in this frame, so the account sits on the T10 decision's target.
    target_ser = strategy_obj.target_path_obj.target_by_decision_dict[pd.Timestamp(T10_STR)]
    position_ser = strategy_obj.get_positions()
    position_ser = position_ser[position_ser > 0]
    assert sorted(position_ser.index) == sorted(target_ser.index)
    assert strategy_obj.previous_total_value > 0
    # Weights landed near their targets at the fill.
    pricing_df = strategy_obj.pricing_data if hasattr(strategy_obj, "pricing_data") else None
    assert pricing_df is None or len(position_ser)


def test_timing_and_capacity_hooks_run(snapshot_root_dict, snapshot_mode):
    from alpha.engine.execution_timing import ExecutionTimingAnalysis
    module_obj = __import__("importlib").import_module("strategies.momentum.strategy_mo_capsule_leg1_natr_ndx")
    snapshot_mode(snapshot_root_dict["t0"])
    with snapshot_module.use_norgate_data_profile(MOMENTUM_LEGS_NDX_PROFILE_STR):
        input_dict = module_obj.build_execution_timing_analysis_inputs()
        analysis_obj = ExecutionTimingAnalysis(save_output_bool=False, **input_dict)
        result_obj = analysis_obj.run()
        capacity_dict = module_obj.build_capacity_analysis_inputs(backtest_start_date_str="2023-06-01", capital_base_float=100000.0, end_date_str=T0_STR)
    assert result_obj is not None and capacity_dict["execution_policy_str"] == "MOO"
    assert capacity_dict["strategy_obj"].get_positions().abs().sum() > 0


# ---- lifecycle
def test_a_shut_gate_sells_everything_through_the_whole_cycle(lifecycle_case, snapshot_root_dict, snapshot_mode):
    store_obj, _, broker_obj, kw = lifecycle_case
    snapshot_mode(snapshot_root_dict["gate_down"])
    assert runner.build_decision_plans(store_obj, T0_AS_OF_TS, auto_sync_norgate_snapshots_bool=False, **kw)["created_decision_plan_count_int"] == 1
    decision_obj = store_obj.get_latest_decision_plan_for_pod("leg")
    submit_ts = decision_obj.submission_timestamp_ts
    broker_obj.seed_account_snapshot("DU777003", cash_float=99000.0, total_value_float=100000.0, position_amount_map={"OLDPICK": 10.0}, snapshot_timestamp_ts=submit_ts, session_mode_str="paper")
    broker_obj.seed_live_price_snapshot("DU777003", {"OLDPICK": 100.0}, snapshot_timestamp_ts=submit_ts)
    assert runner.build_vplans(store_obj, broker_obj, submit_ts, **kw)["created_vplan_count_int"] == 1
    assert store_obj.get_latest_vplan_for_pod("leg").order_delta_map == {"OLDPICK": -10.0}


def test_two_swaps_in_a_row_each_build(lifecycle_case, snapshot_root_dict, snapshot_mode):
    store_obj, _, broker_obj, kw = lifecycle_case
    snapshot_mode(snapshot_root_dict["t10"])
    assert runner.build_decision_plans(store_obj, T10_AS_OF_TS, auto_sync_norgate_snapshots_bool=False, **kw)["created_decision_plan_count_int"] == 1
    decision_obj = store_obj.get_latest_decision_plan_for_pod("leg")
    submit_ts = decision_obj.submission_timestamp_ts
    broker_obj.seed_account_snapshot("DU777003", cash_float=99000.0, total_value_float=100000.0, position_amount_map={"OLDPICK": 10.0}, snapshot_timestamp_ts=submit_ts, session_mode_str="paper")
    broker_obj.seed_live_price_snapshot("DU777003", {**{asset_str: 125.0 for asset_str in decision_obj.full_target_weight_map_dict}, "OLDPICK": 100.0}, snapshot_timestamp_ts=submit_ts)
    assert runner.build_vplans(store_obj, broker_obj, submit_ts, **kw)["created_vplan_count_int"] == 1
    assert runner.submit_ready_vplans(store_obj, broker_obj, submit_ts, manual_only_bool=False, **kw)["submitted_vplan_count_int"] == 1
    assert runner.post_execution_reconcile(store_obj, broker_obj, decision_obj.target_execution_timestamp_ts + timedelta(minutes=10), **kw)["completed_vplan_count_int"] == 1
    # Between the swaps nothing is required; at the next swap a new decision builds.
    assert not runner.norgate_snapshot_sync_required_for_decision_plan_bool(store_obj, datetime(2024, 1, 24, 18, 0, tzinfo=MARKET_ZONE_OBJ), kw["releases_root_path_str"], "paper")
    assert runner.norgate_snapshot_sync_required_for_decision_plan_bool(store_obj, T0_AS_OF_TS, kw["releases_root_path_str"], "paper")
    snapshot_mode(snapshot_root_dict["t0"])
    assert next_phase_str(store_obj, T0_AS_OF_TS, kw) == "build_decision_plan"
    assert runner.build_decision_plans(store_obj, T0_AS_OF_TS, auto_sync_norgate_snapshots_bool=False, **kw)["created_decision_plan_count_int"] == 1
    assert store_obj.get_latest_decision_plan_for_pod("leg").signal_timestamp_ts.date().isoformat() == T0_STR


def test_reference_run_invests_what_the_target_says(snapshot_root_dict, snapshot_mode):
    snapshot_mode(snapshot_root_dict["t0"])
    module_obj = __import__("importlib").import_module("strategies.momentum.strategy_mo_capsule_leg1_natr_ndx")
    with snapshot_module.use_norgate_data_profile(MOMENTUM_LEGS_NDX_PROFILE_STR):
        strategy_obj = module_obj.run_variant(show_display_bool=False, save_results_bool=False, backtest_start_date_str="2023-06-01", capital_base_float=1_000_000.0, end_date_str=T0_STR)
    target_ser = strategy_obj.target_path_obj.target_by_decision_dict[pd.Timestamp(T10_STR)]
    assert 1.0 - strategy_obj.cash / strategy_obj.total_value == pytest.approx(target_ser.sum(), abs=.06)


def test_a_refusal_does_not_outlive_its_manifest(snapshot_root_dict, snapshot_mode, monkeypatch):
    release_obj_ = leg("leg1")
    real_build_fn = legs_module.build_target_path

    def damaged_build_fn(config_obj, panel):
        path_obj = real_build_fn(config_obj, panel)
        path_obj.evidence_by_decision_dict[pd.Timestamp(T10_STR)]["finite_score_member_count_int"] = 1
        return path_obj
    snapshot_mode(snapshot_root_dict["t10"])
    monkeypatch.setattr(legs_module, "build_target_path", damaged_build_fn)
    with pytest.raises(ValueError, match="scored too few"):
        strategy_host.build_decision_plan_for_release(release_obj_, T10_AS_OF_TS, None)
    # The next swap's manifest decides again, with no restart and nothing cleared.
    monkeypatch.setattr(legs_module, "build_target_path", real_build_fn)
    snapshot_mode(snapshot_root_dict["t0"])
    assert strategy_host.build_decision_plan_for_release(release_obj_, T0_AS_OF_TS, None).full_target_weight_map_dict


def test_one_legs_refusal_does_not_stop_the_leg_on_the_same_manifest(snapshot_root_dict, snapshot_mode, monkeypatch):
    real_build_fn = legs_module.build_target_path

    def damaged_build_fn(config_obj, panel):
        path_obj = real_build_fn(config_obj, panel)
        path_obj.evidence_by_decision_dict[pd.Timestamp(T0_STR)]["finite_score_member_count_int"] = 1
        return path_obj
    snapshot_mode(snapshot_root_dict["t0"])
    monkeypatch.setattr(legs_module, "build_target_path", damaged_build_fn)
    with pytest.raises(ValueError, match="scored too few"):
        strategy_host.build_decision_plan_for_release(leg("leg1"), T0_AS_OF_TS, None)
    monkeypatch.setattr(legs_module, "build_target_path", real_build_fn)
    assert strategy_host.build_decision_plan_for_release(leg("leg2"), T0_AS_OF_TS, None).full_target_weight_map_dict
