"""Read-only IBKR probe: wrapper, sample orders and step reporting; fake broker only."""
from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
import yaml

from alpha.live.core5_broker import get_core5_funding_evidence
from alpha.live.daily_broker import DailyExecutionSnapshot
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.models import BrokerSnapshot
import scripts.probe_daily_pod_ibkr as probe_module


TEMPLATE_DIR_PATH_OBJ = Path("docs/live/release_templates")
CORE5_TEMPLATE_STR = "pod_taa_adaptive_macro_core5_daily_moo.yaml.example"
CAPSULE_TEMPLATE_STR = "pod_mr_dv2_vix_gated_bil_daily_moo.yaml.example"


class _ReachedBroker(Exception):
    pass


class FakeAdapter:
    def __init__(self, fail_dict=None, **kwargs):
        self.kwargs_dict = kwargs
        self.fail_dict = fail_dict or {}
        self.call_list = []
        self.request_list = []

    def _call(self, name_str, value_obj):
        self.call_list.append(name_str)
        if name_str in self.fail_dict:
            raise self.fail_dict[name_str]
        return value_obj

    def _snapshot(self):
        return BrokerSnapshot("U_PROBE", datetime.now(UTC), 1000.0, 100000.0, {"DBC": 10.0, "BIL": 50.0})

    def get_visible_account_route_set(self):
        return self._call("get_visible_account_route_set", {"U_PROBE"})

    def get_core5_account_snapshot(self, account_route_str, *, include_portfolio_valuation_bool=False):
        return self._call("get_core5_account_snapshot", self._snapshot())

    def get_eod_account_snapshot(self, account_route_str):
        return self._call("get_eod_account_snapshot", self._snapshot())

    def get_capsule_account_snapshot(self, account_route_str):
        return self._call("get_capsule_account_snapshot", self._snapshot())

    def get_daily_execution_snapshot(self, account_route_str):
        now_ts = datetime.now(UTC)
        return self._call("get_daily_execution_snapshot", DailyExecutionSnapshot(self._snapshot(), [], now_ts, now_ts))

    def get_core5_funding_evidence(self, account_route_str, request_list, position_dict):
        self.request_list = list(request_list)
        return self._call("get_core5_funding_evidence", {"trading_type_str": "STKMRGN", "available_funds_float": 50000.0,
            "excess_liquidity_float": 60000.0, "dbc_shortable_shares_float": 1000000.0,
            "preview_list": [{"asset_str": request_obj.asset_str, "status_str": "PreSubmitted"} for request_obj in request_list]})

    def get_capsule_funding_evidence(self, account_route_str, request_list):
        self.request_list = list(request_list)
        return self._call("get_capsule_funding_evidence", {"trading_type_str": "STKMRGN", "buying_power_float": 200000.0,
            "preview_list": [{"asset_str": request_obj.asset_str, "status_str": "PreSubmitted"} for request_obj in request_list]})

    def submit_order_request_list(self, **_kwargs):
        raise AssertionError("The probe must never reach an order call.")


def _release_path(tmp_path, template_str):
    payload_dict = yaml.safe_load((TEMPLATE_DIR_PATH_OBJ / template_str).read_text(encoding="utf-8"))
    # As deployed: a LIVE release on a real (U) account.
    payload_dict["deployment"]["mode"] = "live"
    payload_dict["broker"]["account_route"] = "U_PROBE"
    release_path_obj = tmp_path / "pod.yaml"
    release_path_obj.write_text(yaml.safe_dump(payload_dict), encoding="utf-8")
    return release_path_obj


def _run(tmp_path, template_str, capsys, *extra_list, **fake_kwargs):
    fake_holder_dict = {}

    def factory_fn(**kwargs):
        fake_holder_dict["fake"] = FakeAdapter(**fake_kwargs, **kwargs)
        return fake_holder_dict["fake"]

    exit_int = probe_module.main(["--release", str(_release_path(tmp_path, template_str)), *extra_list],
        adapter_factory_fn=factory_fn)
    return exit_int, fake_holder_dict.get("fake"), capsys.readouterr()


def test_core5_probe_runs_every_read_and_samples_a_dbc_short(tmp_path, capsys):
    exit_int, fake_obj, output_obj = _run(tmp_path, CORE5_TEMPLATE_STR, capsys)
    assert exit_int == 0
    assert fake_obj.kwargs_dict["client_id_int"] == 90 and fake_obj.kwargs_dict["port_int"] == 7497
    assert fake_obj.call_list == ["get_visible_account_route_set", "get_core5_account_snapshot",
        "get_daily_execution_snapshot", "get_core5_funding_evidence"]
    amount_dict = {request_obj.asset_str: request_obj.amount_float for request_obj in fake_obj.request_list}
    # DBC held long 10: the sample sells 11 so the account ends one share short.
    assert amount_dict == {"SPY": 1.0, "IEF": 1.0, "GLD": 1.0, "DBC": -11.0, "UUP": 1.0, "BIL": 1.0}
    assert output_obj.out.count("PASS  ") == 4 and "RESULT: all checks passed." in output_obj.out
    assert "no order is transmitted" in output_obj.out


def test_capsule_probe_uses_the_capsule_reads(tmp_path, capsys):
    exit_int, fake_obj, output_obj = _run(tmp_path, CAPSULE_TEMPLATE_STR, capsys, "--client-id", "91")
    assert exit_int == 0 and fake_obj.kwargs_dict["client_id_int"] == 91
    assert fake_obj.call_list == ["get_visible_account_route_set", "get_eod_account_snapshot",
        "get_capsule_account_snapshot", "get_daily_execution_snapshot", "get_capsule_funding_evidence"]
    assert [(request_obj.asset_str, request_obj.amount_float) for request_obj in fake_obj.request_list] == [
        ("AAPL", 1.0), ("BIL", 1.0)]


def test_a_failed_check_is_reported_and_the_other_steps_still_run(tmp_path, capsys):
    error_obj = ValueError("CORE5 requires a verified standard margin account (IBKR TradingType-S).")
    exit_int, fake_obj, output_obj = _run(tmp_path, CORE5_TEMPLATE_STR, capsys,
        fail_dict={"get_core5_funding_evidence": error_obj})
    assert exit_int == 1
    assert "FAIL  Pre-submit margin, funding and DBC borrow check (what-if only): ValueError: CORE5 requires" in output_obj.out
    assert "PASS  Daily close read" in output_obj.out and "RESULT: at least one check failed" in output_obj.out


def test_a_failed_account_read_skips_the_core5_margin_check(tmp_path, capsys):
    exit_int, fake_obj, output_obj = _run(tmp_path, CORE5_TEMPLATE_STR, capsys,
        fail_dict={"get_core5_account_snapshot": ValueError("CORE5 requires unambiguous USD TotalCashValue.")})
    assert exit_int == 1 and "get_core5_funding_evidence" not in fake_obj.call_list
    assert "skipped: needs the account read above to pass first" in output_obj.out


def test_json_output_is_complete(tmp_path, capsys):
    exit_int, _, output_obj = _run(tmp_path, CAPSULE_TEMPLATE_STR, capsys, "--json")
    result_dict = json.loads(output_obj.out)
    assert exit_int == 0 and result_dict["passed_bool"] is True and len(result_dict["step_list"]) == 5


def test_read_only_wrapper_blocks_every_order_call():
    broker_obj = probe_module.ReadOnlyBroker(FakeAdapter())
    for name_str in ("submit_order_request_list", "cancel_daily_owned_orders", "submit_manual_order"):
        with pytest.raises(AttributeError, match="read-only"):
            getattr(broker_obj, name_str)


def test_refuses_monthly_releases_and_the_pods_own_client_id(tmp_path, capsys):
    exit_int, fake_obj, _ = _run(tmp_path, "pod_ndx_atr_normalized_monthly_open.yaml.example", capsys)
    assert exit_int == 2 and fake_obj is None
    exit_int, fake_obj, _ = _run(tmp_path, CORE5_TEMPLATE_STR, capsys, "--client-id", "47")
    assert exit_int == 2 and fake_obj is None


def test_sample_orders_pass_the_production_validation(tmp_path):
    """The real checks validate requests before connecting: reaching connect proves they are accepted."""
    release_obj = probe_module.parse_release_manifest(str(_release_path(tmp_path, CORE5_TEMPLATE_STR)))

    class BrokerStopsAtConnect:
        timeout_seconds_float = 4.0

        @contextmanager
        def connect(self):
            raise _ReachedBroker()
            yield

    for position_dict in ({}, {"DBC": 10.0}, {"DBC": -5.0, "SPY": 3.0}):
        with pytest.raises(_ReachedBroker):
            get_core5_funding_evidence(BrokerStopsAtConnect(), "U_PROBE",
                probe_module.core5_sample_request_list(release_obj, position_dict), position_dict)
    capsule_release_obj = probe_module.parse_release_manifest(str(_release_path(tmp_path, CAPSULE_TEMPLATE_STR)))
    socket_client_obj = IBKRSocketClient()
    socket_client_obj.connect = BrokerStopsAtConnect().connect
    with pytest.raises(_ReachedBroker):
        socket_client_obj.get_capsule_funding_evidence("U_PROBE", probe_module.capsule_sample_request_list(capsule_release_obj))
