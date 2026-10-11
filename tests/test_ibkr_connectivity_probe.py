"""IBKR connectivity probe: unchanged default output, plus the read-only daily pod checks; fake broker only."""
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from types import ModuleType

import pytest
import yaml

from alpha.live.daily_broker import DailyExecutionSnapshot
from alpha.live.models import BrokerSnapshot
import scripts.live_debug.ibkr_connectivity_probe as probe_module


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
TEMPLATE_DIR_PATH_OBJ = REPO_ROOT_PATH / "docs" / "live" / "release_templates"
CORE5_TEMPLATE_STR = "pod_taa_adaptive_macro_core5_daily_moo.yaml.example"
CAPSULE_TEMPLATE_STR = "pod_mr_dv2_vix_gated_bil_daily_moo.yaml.example"
# The last commit before the daily pod checks were added to this probe.
PRE_MERGE_COMMIT_STR = "6cdba8f"


def _report_dict():
    return {"probe_timestamp_str": "2026-10-12T13:20:00+00:00", "connected_bool": True, "host_str": "127.0.0.1",
        "port_int": 7496, "client_id_int": 90, "timeout_seconds_float": 4.0, "requested_account_route_str": "U_PROBE",
        "requested_account_mode_str": "live", "visible_account_route_list": ["U_PROBE"], "open_order_count_int": 0,
        "position_count_int": 1, "gross_position_share_float": 10.0,
        "formula_str": "gross_position_share_float = sum_i abs(position_share_i)",
        "account_value_map_dict": {tag_str: 1.0 for tag_str in probe_module.SELECTED_ACCOUNT_TAG_TUPLE},
        "position_amount_map_dict": {"DBC": 10.0}, "open_order_row_dict_list": []}


class FakeAdapter:
    def __init__(self, fail_dict=None, **kwargs):
        self.kwargs_dict = kwargs
        self.fail_dict = fail_dict or {}
        self.call_list = []

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

    def submit_order_request_list(self, **_kwargs):
        raise AssertionError("The probe must never reach an order call.")


def _release_path(tmp_path, template_str):
    payload_dict = yaml.safe_load((TEMPLATE_DIR_PATH_OBJ / template_str).read_text(encoding="utf-8"))
    # As deployed: a LIVE release on a real (U) account.
    payload_dict["deployment"]["mode"] = "live"
    payload_dict["broker"]["account_route"] = "U_PROBE"
    release_path_obj = tmp_path / (template_str.split(".")[0] + ".yaml")
    release_path_obj.write_text(yaml.safe_dump(payload_dict), encoding="utf-8")
    return release_path_obj


def _run_daily(tmp_path, template_str, capsys, monkeypatch, *extra_list, **fake_kwargs):
    fake_holder_dict = {}
    monkeypatch.setattr(probe_module, "_collect_probe_report_dict", lambda _config_dict: _report_dict())

    def factory_fn(**kwargs):
        fake_holder_dict["fake"] = FakeAdapter(**fake_kwargs, **kwargs)
        return fake_holder_dict["fake"]

    exit_int = probe_module.main(["--release-manifest-path", str(_release_path(tmp_path, template_str)),
        "--client-id", "90", "--daily-pod-checks", *extra_list], adapter_factory_fn=factory_fn)
    return exit_int, fake_holder_dict.get("fake"), capsys.readouterr()


def _pre_merge_module():
    source_str = subprocess.check_output(["git", "-c", f"safe.directory={REPO_ROOT_PATH.as_posix()}",
        "-c", "core.fsmonitor=false", "show", f"{PRE_MERGE_COMMIT_STR}:scripts/live_debug/ibkr_connectivity_probe.py"],
        cwd=REPO_ROOT_PATH, text=True, encoding="utf-8")
    module_obj = ModuleType("pre_merge_ibkr_connectivity_probe")
    module_obj.__file__ = str(REPO_ROOT_PATH / "scripts" / "live_debug" / "ibkr_connectivity_probe.py")
    exec(compile(source_str, module_obj.__file__, "exec"), module_obj.__dict__)
    return module_obj


@pytest.mark.parametrize("argv_list", [["--account-route", "U_PROBE"], ["--account-route", "U_PROBE", "--json"]])
@pytest.mark.parametrize("fail_bool", [False, True])
def test_default_output_is_identical_to_the_pre_merge_probe(monkeypatch, capsys, argv_list, fail_bool):
    observation_list = []
    for module_obj in (_pre_merge_module(), probe_module):
        def collect_fn(_config_dict):
            if fail_bool:
                raise RuntimeError("IBKR API connected, but the requested account_route_str was not visible.")
            return _report_dict()
        monkeypatch.setattr(module_obj, "_collect_probe_report_dict", collect_fn)
        exit_int = module_obj.main(argv_list)
        observation_list.append((exit_int, capsys.readouterr().out))
    assert observation_list[1] == observation_list[0]


def test_core5_daily_checks_run_the_serve_reads_only(tmp_path, capsys, monkeypatch):
    exit_int, fake_obj, output_obj = _run_daily(tmp_path, CORE5_TEMPLATE_STR, capsys, monkeypatch)
    assert exit_int == 0
    assert fake_obj.kwargs_dict["client_id_int"] == 90 and fake_obj.kwargs_dict["port_int"] == 7497
    assert fake_obj.call_list == ["get_visible_account_route_set", "get_core5_account_snapshot",
        "get_daily_execution_snapshot"]
    assert "IBKR connectivity probe" in output_obj.out
    assert "no order is transmitted" in output_obj.out and output_obj.out.count("  PASS  ") == 3
    assert "what-if" not in output_obj.out
    assert "RESULT: all daily pod checks passed." in output_obj.out


def test_capsule_daily_checks_use_the_capsule_reads(tmp_path, capsys, monkeypatch):
    exit_int, fake_obj, _ = _run_daily(tmp_path, CAPSULE_TEMPLATE_STR, capsys, monkeypatch)
    assert exit_int == 0
    assert fake_obj.call_list == ["get_visible_account_route_set", "get_eod_account_snapshot",
        "get_capsule_account_snapshot", "get_daily_execution_snapshot"]


def test_a_failed_daily_check_is_reported_and_the_other_steps_still_run(tmp_path, capsys, monkeypatch):
    error_obj = ValueError("CORE5 requires unambiguous USD TotalCashValue.")
    exit_int, fake_obj, output_obj = _run_daily(tmp_path, CORE5_TEMPLATE_STR, capsys, monkeypatch,
        fail_dict={"get_core5_account_snapshot": error_obj})
    assert exit_int == 1 and fake_obj.call_list[-1] == "get_daily_execution_snapshot"
    assert "FAIL  Account read (16:10 and before sending): ValueError: CORE5 requires" in output_obj.out
    assert "PASS  Daily close read" in output_obj.out and "at least one daily pod check failed" in output_obj.out


def test_json_output_adds_the_daily_checks(tmp_path, capsys, monkeypatch):
    exit_int, _, output_obj = _run_daily(tmp_path, CAPSULE_TEMPLATE_STR, capsys, monkeypatch, "--json")
    result_dict = json.loads(output_obj.out)
    assert exit_int == 0 and result_dict["daily_pod_checks_passed_bool"] is True
    assert len(result_dict["daily_pod_check_list"]) == 4 and result_dict["connected_bool"] is True


@pytest.mark.parametrize("case_str", ["no_manifest", "monthly_release", "pod_client_id", "no_client_id"])
def test_daily_checks_refuse_before_connecting(tmp_path, capsys, monkeypatch, case_str):
    monkeypatch.setattr(probe_module, "_collect_probe_report_dict",
        lambda _config_dict: pytest.fail("Connected before the daily check preconditions passed"))
    argv_dict = {
        "no_manifest": ["--account-route", "U_PROBE", "--client-id", "90"],
        "monthly_release": ["--release-manifest-path",
            str(_release_path(tmp_path, "pod_ndx_atr_normalized_monthly_open.yaml.example")), "--client-id", "90"],
        "pod_client_id": ["--release-manifest-path", str(_release_path(tmp_path, CORE5_TEMPLATE_STR)), "--client-id", "47"],
        "no_client_id": ["--release-manifest-path", str(_release_path(tmp_path, CORE5_TEMPLATE_STR))],
    }
    exit_int = probe_module.main([*argv_dict[case_str], "--daily-pod-checks"],
        adapter_factory_fn=lambda **_kwargs: pytest.fail("Built a broker adapter"))
    assert exit_int == 2 and "daily pod checks:" in capsys.readouterr().err


def test_read_only_wrapper_blocks_every_order_call():
    broker_obj = probe_module.ReadOnlyBroker(FakeAdapter())
    for name_str in ("submit_order_request_list", "cancel_daily_owned_orders", "submit_manual_order"):
        with pytest.raises(AttributeError, match="read-only"):
            getattr(broker_obj, name_str)
