"""Audit reproduction of current Doctor-versus-runner CORE5 quote policy."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

from alpha.live import doctor, runner
from alpha.live.runner import BrokerAdapterResolver
from alpha.live.state_store_v2 import LiveStateStore
import test_live_core5_adapter as fixture_module
import test_live_core5_wiring as wiring_module


def test_core5_runner_builds_without_quotes_while_doctor_blocks(tmp_path, monkeypatch):
    release_obj = fixture_module.release_obj.__wrapped__()
    pricing_df = fixture_module.price_df.__wrapped__()
    fixture_module._controlled_signals(monkeypatch, pricing_df)
    state_obj = fixture_module._state(release_obj, "2026-09-11", cash_float=1_000_000)
    decision_obj = fixture_module._build(release_obj, pricing_df, "2026-09-11", state_obj)
    releases_path = wiring_module._write_release_yaml(tmp_path, release_obj)
    config_path = tmp_path / "config.env"
    config_path.write_text(f"NORGATE_RELEASES_ROOT={releases_path}\n", encoding="utf-8")
    store_obj, broker_obj = fixture_module._store_and_broker(tmp_path, release_obj, state_obj)
    def no_quote_fn(*argument_tuple, **keyword_dict):
        raise RuntimeError("simulated live quote unavailable")
    monkeypatch.setattr(broker_obj, "get_live_price_snapshot", no_quote_fn)
    monkeypatch.setattr(doctor.scheduler_utils, "evaluate_build_gate_dict", lambda *_args: {"due_bool": True, "reason_code_str": "snapshot_ready"})
    monkeypatch.setattr(doctor.strategy_host, "build_decision_plan_for_release", lambda **_kwargs: decision_obj)
    monkeypatch.delenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", raising=False)
    monkeypatch.delenv("NORGATE_SNAPSHOT_ROOT", raising=False)
    report_dict = doctor.compute_doctor_verdict(releases_root_path_str=str(releases_path), env_mode_str="paper",
        as_of_ts=fixture_module._time("2026-09-11"), pod_id_str=release_obj.pod_id_str,
        db_path_str=str(tmp_path / "not_existing.sqlite3"),
        config_env_path_str=str(config_path), loaded_config_env_dict={"NORGATE_RELEASES_ROOT": str(releases_path)},
        pod_state_loader_fn=lambda *_args: state_obj,
        norgate_snapshot_sync_checker_fn=lambda **_kwargs: {"status_str": "ready", "reason_code_str": "local_snapshot_ready"},
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=broker_obj), trace_enabled_bool=False)
    assert report_dict["overall_verdict_str"] == "BLOCK"
    assert report_dict["component_result_dict_list"][-1]["reason_code_str"] == "live_price_snapshot_error"
    store_obj.insert_decision_plan(decision_obj)
    runner_dict = runner.build_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper",
        log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert runner_dict["created_vplan_count_int"] == 1
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).live_price_source_str == "core5.frozen_close_t"
    assert not broker_obj.submitted_order_request_list
    output_path = Path(__file__).resolve().parents[3] / "results/review/wired_signoff_20261015/ops_doctor_quote_probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"status_str": "reproduced", "doctor_status_str": "BLOCK",
        "doctor_reason_str": "live_price_snapshot_error", "runner_created_vplans_int": 1,
        "runner_reference_str": "core5.frozen_close_t", "broker_io_bool": False,
        "finding_str": "Doctor requires a livequote for CORE5 while the actual runner uses its approved frozen Close_T reference."}, indent=2), encoding="utf-8")
