"""Capsule account truth: incomplete reads fail closed; valid order intent is unchanged."""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from alpha.live import runner
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.order_clerk import IBKRGatewayBrokerAdapter
from alpha.live.state_store_v2 import LiveStateStore
from test_live_daily_broker import Event, FakeIB, _trade, socket_obj  # noqa: F401
from test_live_mr_capsule_recovery import capsule_case as execution_case, SUBMIT_TIMESTAMP_TS  # noqa: F401
from test_live_mr_capsule_scheduler import capsule_case, _broker, _state, _time  # noqa: F401
from test_live_mr_capsule_target_shares import _capsule_inputs


@pytest.fixture(autouse=True)
def no_broker_network(monkeypatch):
    monkeypatch.setattr(IBKRSocketClient, "connect", lambda *_: pytest.fail("Real broker forbidden"))
    monkeypatch.setattr(IBKRSocketClient, "daily_connection", lambda *_: pytest.fail("Real broker forbidden"))


@pytest.fixture
def capsule_socket_tuple(socket_obj, monkeypatch):
    state_obj = socket_obj.state_obj
    state_obj.position_list = [SimpleNamespace(account="DU1", position=amount_float,
        contract=SimpleNamespace(symbol=asset_str, secType="STK", currency="USD"))
        for asset_str, amount_float in (("KEEP", 50.0), ("BIL", 500.0))]
    state_obj.account_value_list[0].value = "5000"
    state_obj.account_value_list[1].value = "120000"
    legacy_ib_obj = FakeIB(state_obj, 31)
    legacy_ib_obj.wrapper._futures = {}
    legacy_ib_obj.accountSummary = lambda account: list(state_obj.account_value_list)
    legacy_ib_obj.positions = lambda account: ([] if state_obj.failure_str == "positions"
        else [position_obj for position_obj in state_obj.position_list if position_obj.account == account])
    legacy_ib_obj.reqOpenOrders = lambda: []
    client_obj = IBKRSocketClient()
    monkeypatch.setattr(client_obj, "connect", lambda: nullcontext(legacy_ib_obj))
    monkeypatch.setattr(client_obj, "daily_connection", socket_obj.daily_connection)
    return client_obj, state_obj, legacy_ib_obj


@pytest.mark.parametrize("failure_str", ["positions", "account_summary", "all_orders"])
def test_capsule_incomplete_requests_fail_instead_of_returning_cached_account(capsule_socket_tuple, failure_str):
    client_obj, state_obj, legacy_ib_obj = capsule_socket_tuple
    state_obj.failure_str = failure_str
    legacy_ib_obj.wrapper._futures["positions"] = object()
    with pytest.raises((TimeoutError, ConnectionError)):
        client_obj.get_capsule_account_snapshot("DU1")


@pytest.mark.parametrize("mutation_str", ["missing_cash", "missing_nav", "ambiguous_cash", "wrong_currency",
    "model_cash", "nonfinite_cash", "disconnected", "broker_error", "duplicate_position", "wrong_contract"])
def test_capsule_rejects_incomplete_or_ambiguous_account_evidence(capsule_socket_tuple, mutation_str):
    client_obj, state_obj, _ = capsule_socket_tuple
    if mutation_str == "missing_cash":
        state_obj.account_value_list.pop(0)
    elif mutation_str == "missing_nav":
        state_obj.account_value_list.pop(1)
    elif mutation_str == "ambiguous_cash":
        state_obj.account_value_list.append(SimpleNamespace(account="DU1", tag="TotalCashValue", currency="USD", value="7"))
    elif mutation_str == "wrong_currency":
        state_obj.account_value_list[0].currency = "EUR"
    elif mutation_str == "model_cash":
        state_obj.account_value_list[0].modelCode = "MODEL"
    elif mutation_str == "nonfinite_cash":
        state_obj.account_value_list[0].value = "nan"
    elif mutation_str == "disconnected":
        state_obj.disconnect_bool = True
    elif mutation_str == "broker_error":
        state_obj.error_code_int = 1100
    elif mutation_str == "duplicate_position":
        state_obj.position_list.append(state_obj.position_list[0])
    else:
        state_obj.position_list[0].contract.currency = "EUR"
    with pytest.raises((ValueError, ConnectionError)):
        client_obj.get_capsule_account_snapshot("DU1")


def test_capsule_cannot_use_an_account_outside_the_connected_session(capsule_socket_tuple):
    client_obj, _, _ = capsule_socket_tuple
    with pytest.raises(ConnectionError):
        client_obj.get_capsule_account_snapshot("OTHER")


def test_capsule_startup_timeout_never_falls_back_to_the_legacy_cache(capsule_socket_tuple, monkeypatch):
    client_obj, _, _ = capsule_socket_tuple
    @contextmanager
    def failed_connection():
        raise ConnectionError("positions request timed out")
        yield  # pragma: no cover
    monkeypatch.setattr(client_obj, "daily_connection", failed_connection)
    with pytest.raises(ConnectionError, match="positions request timed out"):
        client_obj.get_capsule_account_snapshot("DU1")


def test_capsule_orders_from_other_clients_are_visible_and_legacy_reader_is_unchanged(capsule_socket_tuple):
    client_obj, state_obj, _ = capsule_socket_tuple
    state_obj.position_list[0].contract.symbol = "BRK B"
    state_obj.trade_list = [_trade(client_int=77), _trade("foreign", 88, "DU2")]
    legacy_snapshot_obj = client_obj.get_account_snapshot("DU1")
    snapshot_obj = client_obj.get_capsule_account_snapshot("DU1")
    assert legacy_snapshot_obj.open_order_id_list == []
    assert snapshot_obj.open_order_id_list
    assert snapshot_obj.position_amount_map == {"BRK.B": 50.0, "BIL": 500.0}
    assert snapshot_obj.cash_float == 5000 and snapshot_obj.net_liq_float == 120000
    assert state_obj.call_list[-4:] == [(31, "all_orders"), (31, "positions"), (31, "account_summary"), (31, "all_orders")]


@pytest.mark.parametrize("pod_str", ["dv2_vix_gated", "hpi_vote_vix_gated"])
@pytest.mark.parametrize("parking_str", ["cash", "bil", "spmo"])
@pytest.mark.parametrize("cash_float", [-500.0, 0.0, 5000.0])
def test_complete_account_preserves_vplan_and_every_order_request(capsule_socket_tuple, pod_str, parking_str, cash_float):
    client_obj, state_obj, _ = capsule_socket_tuple
    state_obj.account_value_list[0].value = str(cash_float)
    release_obj, decision_obj, _, quote_obj = _capsule_inputs()
    strategy_str = f"strategies.mr_capsule.strategy_mr_{pod_str}_{parking_str}"
    release_obj = replace(release_obj, account_route_str="DU1", strategy_import_str=strategy_str)
    target_dict = {"BIL": 400.0}
    if parking_str == "cash":
        state_obj.position_list = state_obj.position_list[:1]
        target_dict = {}
    elif parking_str == "spmo":
        target_dict["SPMO"] = 25.0
        quote_obj = replace(quote_obj, asset_reference_price_map={**quote_obj.asset_reference_price_map, "SPMO": 100.0})
    base_dict = {position_obj.contract.symbol: position_obj.position for position_obj in state_obj.position_list}
    decision_obj = replace(decision_obj, account_route_str="DU1", decision_base_position_map=base_dict,
        target_share_map_dict=target_dict, snapshot_metadata_dict={**decision_obj.snapshot_metadata_dict,
            "strategy_import_str": strategy_str, "mr_capsule_parking_mode_str": parking_str})
    quote_obj = replace(quote_obj, account_route_str="DU1")
    old_snapshot_obj = client_obj.get_account_snapshot("DU1")
    new_snapshot_obj = client_obj.get_capsule_account_snapshot("DU1")
    new_snapshot_obj = replace(new_snapshot_obj, snapshot_timestamp_ts=old_snapshot_obj.snapshot_timestamp_ts)
    old_plan_obj = build_vplan(release_obj, decision_obj, old_snapshot_obj, quote_obj)
    new_plan_obj = build_vplan(release_obj, decision_obj, new_snapshot_obj, quote_obj)
    assert asdict(new_plan_obj) == asdict(old_plan_obj)
    assert build_broker_order_request_list_from_vplan(new_plan_obj) == build_broker_order_request_list_from_vplan(old_plan_obj)


@pytest.mark.parametrize("mode_str", ["paper", "live"])
@pytest.mark.parametrize("failed_bool", [False, True])
def test_capsule_eod_uses_strict_reader_and_never_saves_failed_evidence(capsule_case, monkeypatch, mode_str, failed_bool):
    store_obj, release_obj, tmp_path, _ = capsule_case
    release_obj = replace(release_obj, mode_str=mode_str)
    broker_obj = _broker(release_obj)
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    call_list = []
    def strict_snapshot(account_route_str, *, include_portfolio_valuation_bool=False):
        call_list.append((account_route_str, include_portfolio_valuation_bool))
        if failed_bool:
            raise ConnectionError("Incomplete account download")
        return snapshot_obj
    store_obj.upsert_release(release_obj)
    monkeypatch.setattr(broker_obj, "get_session_mode_str", lambda _: mode_str)
    monkeypatch.setattr(broker_obj, "get_capsule_account_snapshot", strict_snapshot)
    monkeypatch.setattr(broker_obj, "get_account_snapshot", lambda *_: pytest.fail("Legacy account reader used"))
    monkeypatch.setattr(broker_obj, "get_eod_account_snapshot", lambda *_: pytest.fail("Legacy EOD reader used"))
    argument_dict = dict(state_store_obj=store_obj, broker_adapter_obj=broker_obj, as_of_ts=_time(),
        env_mode_str=mode_str, releases_root_path_str=None,
        log_path_str=str(tmp_path / "events.jsonl"), trace_enabled_bool=False)
    if failed_bool:
        with pytest.raises(ConnectionError, match="Incomplete account download"):
            runner.eod_snapshot(**argument_dict)
        assert store_obj.get_pod_state(release_obj.pod_id_str) is None
    else:
        assert runner.eod_snapshot(**argument_dict)["eod_snapshot_count_int"] == 1
        saved_obj = store_obj.get_pod_state(release_obj.pod_id_str)
        assert saved_obj.cash_float == snapshot_obj.cash_float
        assert saved_obj.position_amount_map == snapshot_obj.position_amount_map
    assert call_list == [(release_obj.account_route_str, mode_str == "live")]


@pytest.mark.parametrize("failure_str", [None, "timeout", "error"])
@pytest.mark.parametrize("account_list", [["DU1"], ["DU1", "DU2"]])
def test_capsule_eod_marks_refresh_on_the_same_connection_and_never_replace_account_truth(
        capsule_socket_tuple, monkeypatch, failure_str, account_list):
    client_obj, state_obj, _ = capsule_socket_tuple
    connection_list = []
    @contextmanager
    def valuation_connection():
        ib_obj = FakeIB(state_obj, 31)
        ib_obj.wrapper._futures = {}
        ib_obj.managedAccounts = lambda: account_list
        ib_obj.updatePortfolioEvent = Event()
        ib_obj.accountValueEvent = Event()
        ib_obj.portfolio = lambda **_: pytest.fail("No startup portfolio cache on daily connection")
        ib_obj.accountValues = lambda **_: pytest.fail("No startup account-value cache on daily connection")
        ib_obj.reqAccountUpdatesAsync = lambda account_str: account_str
        def run_update(request_obj, *, timeout):
            assert request_obj == "DU1" and timeout > 0
            assert not ib_obj.summary_active_set
            if failure_str == "timeout":
                raise TimeoutError("Optional display unavailable")
            if failure_str == "error":
                raise RuntimeError("Optional display unavailable")
            for tag_str, value_str in (("TotalCashValue", "9876"), ("NetLiquidation", "130000")):
                ib_obj.accountValueEvent.emit(SimpleNamespace(account="DU1", tag=tag_str,
                    currency="USD", value=value_str, modelCode=""))
            for asset_int, position_obj in enumerate(state_obj.position_list, start=1):
                contract_obj = SimpleNamespace(**vars(position_obj.contract), conId=asset_int, multiplier="")
                ib_obj.updatePortfolioEvent.emit(SimpleNamespace(account="DU1", contract=contract_obj,
                    position=position_obj.position, marketPrice=100.0, marketValue=position_obj.position * 100.0))
        ib_obj.run = run_update
        connection_list.append(ib_obj)
        try:
            yield ib_obj
        finally:
            assert not ib_obj.summary_active_set
            assert not ib_obj.updatePortfolioEvent.handler_list
            assert not ib_obj.accountValueEvent.handler_list
    monkeypatch.setattr(client_obj, "daily_connection", valuation_connection)
    adapter_obj = IBKRGatewayBrokerAdapter.__new__(IBKRGatewayBrokerAdapter)
    adapter_obj.socket_client_obj = client_obj
    snapshot_obj = adapter_obj.get_capsule_account_snapshot("DU1", include_portfolio_valuation_bool=True)
    assert len(connection_list) == 1
    assert snapshot_obj.cash_float == 5000.0 and snapshot_obj.net_liq_float == 120000.0
    assert snapshot_obj.position_amount_map == {"KEEP": 50.0, "BIL": 500.0}
    assert snapshot_obj.portfolio_valuation_dict["available_bool"] is (failure_str is None)
    if failure_str is None:
        assert snapshot_obj.portfolio_valuation_dict["cash_float"] == 9876.0
        assert {row_dict["symbol_str"] for row_dict in snapshot_obj.portfolio_valuation_dict["position_list"]} == {"KEEP", "BIL"}


def test_capsule_fresh_usd_totals_ignore_other_currency_account_and_model_rows(capsule_socket_tuple):
    client_obj, state_obj, _ = capsule_socket_tuple
    state_obj.account_value_list.extend([
        SimpleNamespace(account="DU1", tag="TotalCashValue", currency="EUR", value="99999"),
        SimpleNamespace(account="DU2", tag="TotalCashValue", currency="USD", value="99999"),
        SimpleNamespace(account="DU1", tag="TotalCashValue", currency="USD", value="99999", modelCode="MODEL"),
    ])
    snapshot_obj = client_obj.get_capsule_account_snapshot("DU1")
    assert snapshot_obj.cash_float == 5000.0
    assert snapshot_obj.available_funds_float is snapshot_obj.excess_liquidity_float is None
    assert snapshot_obj.portfolio_valuation_dict is None


def test_capsule_changing_open_orders_are_a_retryable_read_failure(capsule_socket_tuple):
    client_obj, state_obj, _ = capsule_socket_tuple
    from alpha.live.guarded_dispatch import is_transient_broker_error_bool
    def change_orders(ib_obj):
        if len(state_obj.call_list) == 4:
            state_obj.trade_list = [_trade()]
    state_obj.order_hook_fn = change_orders
    with pytest.raises(ConnectionError, match="open orders changed") as error_info:
        client_obj.get_capsule_account_snapshot("DU1")
    assert is_transient_broker_error_bool(error_info.value)


def _trusted_account_rows_dict(store_obj):
    with store_obj._connect() as connection_obj:
        return {table_str: [tuple(row_obj) for row_obj in connection_obj.execute(
            f"SELECT * FROM {table_str} ORDER BY rowid")]
            for table_str in ("pod_state", "pod_state_history", "broker_snapshot_cache")}


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError, ValueError])
def test_failed_capsule_account_read_cannot_size_or_create_a_vplan(execution_case, monkeypatch, error_type):
    source_store_obj, broker_obj, release_obj, source_plan_obj, runner_kwarg_dict, tmp_path = execution_case
    store_obj = LiveStateStore(str(tmp_path / "failed-build.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(source_store_obj.get_pod_state(release_obj.pod_id_str))
    store_obj.upsert_broker_snapshot_cache(source_store_obj.get_latest_broker_snapshot_for_account(release_obj.account_route_str))
    decision_obj = store_obj.insert_decision_plan(replace(
        source_store_obj.get_decision_plan_by_id(source_plan_obj.decision_plan_id_int),
        decision_plan_id_int=None, status_str="planned"))
    before_dict = _trusted_account_rows_dict(store_obj)
    read_list = []
    def failed_read(account_route_str):
        read_list.append(account_route_str)
        raise error_type("Incomplete capsule account read")
    monkeypatch.setattr(broker_obj, "get_capsule_account_snapshot", failed_read)
    monkeypatch.setattr(broker_obj, "get_live_price_snapshot", lambda *args, **kwargs: pytest.fail("Quote after failed account read"))
    monkeypatch.setattr(runner, "build_vplan", lambda *args, **kwargs: pytest.fail("Sizing after failed account read"))
    result_dict = runner.build_vplans(store_obj, broker_obj, SUBMIT_TIMESTAMP_TS, "paper", **runner_kwarg_dict)
    retry_bool = error_type is not ValueError
    assert result_dict["created_vplan_count_int"] == 0
    assert result_dict["reason_count_map_dict"] == {"build_retry_pending" if retry_bool else "opening_build_parked": 1}
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == ("planned" if retry_bool else "blocked")
    assert store_obj.get_latest_vplan_for_decision(decision_obj.decision_plan_id_int) is None
    assert _trusted_account_rows_dict(store_obj) == before_dict
    assert read_list == [release_obj.account_route_str]
    assert not broker_obj.submitted_order_request_list


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError, ValueError])
def test_failed_capsule_presubmit_account_read_cannot_claim_or_send(execution_case, monkeypatch, error_type):
    store_obj, broker_obj, release_obj, plan_obj, runner_kwarg_dict, _ = execution_case
    class DispatchClock:
        @classmethod
        def now(cls, timezone_obj):
            return SUBMIT_TIMESTAMP_TS.astimezone(timezone_obj)
    monkeypatch.setattr("alpha.live.dispatch_state.datetime", DispatchClock)
    request_list = build_broker_order_request_list_from_vplan(plan_obj)
    before_dict = _trusted_account_rows_dict(store_obj)
    read_list = []
    def failed_read(account_route_str):
        read_list.append(account_route_str)
        raise error_type("Incomplete capsule account read")
    monkeypatch.setattr(broker_obj, "get_capsule_account_snapshot", failed_read)
    monkeypatch.setattr(store_obj, "claim_vplan_for_submission", lambda *_: pytest.fail("Claim after failed account read"))
    monkeypatch.setattr(broker_obj, "submit_order_request_list", lambda **_: pytest.fail("Send after failed account read"))
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, SUBMIT_TIMESTAMP_TS, "paper", False, **runner_kwarg_dict)
    retry_bool = error_type is not ValueError
    saved_plan_obj = store_obj.get_vplan_by_id(plan_obj.vplan_id_int)
    assert result_dict["submitted_vplan_count_int"] == 0
    assert saved_plan_obj.status_str == ("ready" if retry_bool else "blocked")
    assert store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int).status_str == ("vplan_ready" if retry_bool else "blocked")
    assert build_broker_order_request_list_from_vplan(saved_plan_obj) == request_list
    assert store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int) == []
    assert store_obj.get_broker_ack_row_dict_list_for_vplan(plan_obj.vplan_id_int) == []
    assert _trusted_account_rows_dict(store_obj) == before_dict
    assert read_list == [release_obj.account_route_str]


@pytest.mark.parametrize("mode_str", ["paper", "live"])
def test_failed_capsule_eod_read_preserves_prior_trusted_state_and_history(capsule_case, monkeypatch, mode_str):
    store_obj, release_obj, tmp_path, _ = capsule_case
    release_obj = replace(release_obj, mode_str=mode_str)
    store_obj.upsert_release(release_obj)
    prior_ts = _time("2024-01-30")
    prior_state_obj = _state(release_obj, prior_ts, position_amount_map={"BIL": 80.0},
        cash_float=1234.0, total_value_float=9234.0, strategy_state_dict={"prior_session_int": 7})
    store_obj.upsert_pod_state(prior_state_obj)
    broker_obj = _broker(release_obj)
    store_obj.upsert_broker_snapshot_cache(replace(broker_obj.get_account_snapshot(release_obj.account_route_str),
        snapshot_timestamp_ts=prior_ts, position_amount_map=dict(prior_state_obj.position_amount_map),
        cash_float=1234.0, total_value_float=9234.0, net_liq_float=9234.0))
    before_dict = _trusted_account_rows_dict(store_obj)
    assert all(before_dict.values())
    read_list = []
    def failed_read(account_route_str, *, include_portfolio_valuation_bool=False):
        read_list.append((account_route_str, include_portfolio_valuation_bool))
        raise ConnectionError("Incomplete capsule EOD account read")
    monkeypatch.setattr(broker_obj, "get_session_mode_str", lambda _: mode_str)
    monkeypatch.setattr(broker_obj, "get_capsule_account_snapshot", failed_read)
    monkeypatch.setattr(broker_obj, "get_account_snapshot", lambda *_: pytest.fail("Legacy account reader used"))
    monkeypatch.setattr(broker_obj, "get_eod_account_snapshot", lambda *_: pytest.fail("Legacy EOD reader used"))
    with pytest.raises(ConnectionError, match="Incomplete capsule EOD account read"):
        runner.eod_snapshot(store_obj, broker_obj, _time(), env_mode_str=mode_str,
            log_path_str=str(tmp_path / "failed-eod.jsonl"), trace_enabled_bool=False)
    assert _trusted_account_rows_dict(store_obj) == before_dict
    assert read_list == [(release_obj.account_route_str, mode_str == "live")]
