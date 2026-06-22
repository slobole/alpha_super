from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import scripts.live_debug.collect_vps_debug_bundle as bundle_module


def test_redact_text_str_hides_secret_env_values_and_sensitive_urls() -> None:
    raw_text_str = "\n".join(
        [
            "NORGATE_API_TOKEN=abc123",
            "ALPHA_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/a/b",
            "ALPHA_INSPECTOR_HEARTBEAT_URL=https://hc-ping.com/uuid",
            "NORGATE_API_HOST=100.64.1.2",
            '{"api_token": "secret", "safe": "value"}',
        ]
    )

    redacted_text_str = bundle_module.redact_text_str(raw_text_str)

    assert "abc123" not in redacted_text_str
    assert "discord.com/api/webhooks" not in redacted_text_str
    assert "hc-ping.com" not in redacted_text_str
    assert '"api_token": "<REDACTED>"' in redacted_text_str
    assert "NORGATE_API_HOST=100.64.1.2" in redacted_text_str


def test_build_command_spec_skips_opt_in_live_checks_without_explicit_flags() -> None:
    parsed_args_obj = argparse.Namespace(
        mode_str="live",
        pod_id_str="pod_live_01",
        release_manifest_path_str="alpha/live/releases/user/pod.yaml",
        releases_root_path_str=None,
        db_path_str=None,
        as_of_timestamp_str=None,
        include_runner_details_bool=False,
        include_doctor_bool=False,
        doctor_broker_client_id_int=None,
        include_norgate_doctor_bool=False,
        ibkr_probe_client_id_int=None,
        norgate_report_path_obj=Path("norgate_report.json"),
    )

    command_spec_list = bundle_module.build_command_spec_list(parsed_args_obj)
    command_map_dict = {
        name_str: (argv_list, should_run_bool)
        for name_str, argv_list, should_run_bool in command_spec_list
    }

    assert command_map_dict["doctor"][1] is False
    assert command_map_dict["runner_detail_commands"][1] is False
    assert "--include-doctor" in " ".join(command_map_dict["doctor"][0])
    assert "--include-runner-details" in " ".join(command_map_dict["runner_detail_commands"][0])
    assert command_map_dict["norgate_client_doctor"][1] is False
    assert "may sync snapshot files" in " ".join(command_map_dict["norgate_client_doctor"][0])
    assert command_map_dict["ibkr_connectivity_probe"][1] is False
    assert "--ibkr-probe-client-id" in " ".join(command_map_dict["ibkr_connectivity_probe"][0])


def test_default_command_spec_does_not_include_mutating_runner_actions() -> None:
    parsed_args_obj = argparse.Namespace(
        mode_str="live",
        pod_id_str="pod_live_01",
        release_manifest_path_str=None,
        releases_root_path_str=None,
        db_path_str=None,
        as_of_timestamp_str=None,
        include_runner_details_bool=False,
        include_doctor_bool=False,
        doctor_broker_client_id_int=None,
        include_norgate_doctor_bool=False,
        ibkr_probe_client_id_int=None,
        norgate_report_path_obj=Path("norgate_report.json"),
    )

    command_spec_list = bundle_module.build_command_spec_list(parsed_args_obj)
    joined_command_str = "\n".join(" ".join(argv_list) for _name, argv_list, _run in command_spec_list)

    forbidden_token_list = [
        " tick",
        " submit_vplan",
        " post_execution_reconcile",
        " eod_snapshot",
        " run_once",
        " serve",
    ]
    for forbidden_token_str in forbidden_token_list:
        assert forbidden_token_str not in joined_command_str


def test_runner_detail_commands_are_explicit_opt_in() -> None:
    parsed_args_obj = argparse.Namespace(
        mode_str="live",
        pod_id_str="pod_live_01",
        release_manifest_path_str=None,
        releases_root_path_str=None,
        db_path_str=None,
        as_of_timestamp_str=None,
        include_runner_details_bool=True,
        include_doctor_bool=True,
        doctor_broker_client_id_int=91,
        include_norgate_doctor_bool=False,
        ibkr_probe_client_id_int=None,
        norgate_report_path_obj=Path("norgate_report.json"),
    )

    command_spec_list = bundle_module.build_command_spec_list(parsed_args_obj)
    command_map_dict = {
        name_str: (argv_list, should_run_bool)
        for name_str, argv_list, should_run_bool in command_spec_list
    }

    assert command_map_dict["doctor"][1] is True
    assert command_map_dict["status"][1] is True
    assert command_map_dict["scheduler_next_due"][1] is True
    assert command_map_dict["show_decision_plan"][1] is True
    assert command_map_dict["show_vplan"][1] is True
    assert command_map_dict["execution_report"][1] is True


def test_doctor_requires_explicit_alternate_client_id() -> None:
    parsed_args_obj = argparse.Namespace(
        mode_str="live",
        pod_id_str="pod_live_01",
        release_manifest_path_str=None,
        releases_root_path_str=None,
        db_path_str=None,
        as_of_timestamp_str=None,
        include_runner_details_bool=False,
        include_doctor_bool=True,
        doctor_broker_client_id_int=None,
        include_norgate_doctor_bool=False,
        ibkr_probe_client_id_int=None,
        norgate_report_path_obj=Path("norgate_report.json"),
    )

    command_spec_list = bundle_module.build_command_spec_list(parsed_args_obj)
    command_map_dict = {
        name_str: (argv_list, should_run_bool)
        for name_str, argv_list, should_run_bool in command_spec_list
    }

    doctor_argv_list, doctor_run_bool = command_map_dict["doctor"]
    assert doctor_run_bool is False
    assert "--doctor-broker-client-id" in " ".join(doctor_argv_list)


def test_python_argv_list_uses_current_interpreter() -> None:
    argv_list = bundle_module.python_argv_list("-m", "alpha.live.runner", "status")

    assert argv_list[:3] == [sys.executable, "-m", "alpha.live.runner"]


def test_build_command_spec_runs_ibkr_probe_only_with_explicit_client_id() -> None:
    parsed_args_obj = argparse.Namespace(
        mode_str="live",
        pod_id_str="pod_live_01",
        release_manifest_path_str="alpha/live/releases/user/pod.yaml",
        releases_root_path_str=None,
        db_path_str=None,
        as_of_timestamp_str=None,
        include_runner_details_bool=False,
        include_doctor_bool=False,
        doctor_broker_client_id_int=None,
        include_norgate_doctor_bool=True,
        ibkr_probe_client_id_int=91,
        norgate_report_path_obj=Path("norgate_report.json"),
    )

    command_spec_list = bundle_module.build_command_spec_list(parsed_args_obj)
    command_map_dict = {
        name_str: (argv_list, should_run_bool)
        for name_str, argv_list, should_run_bool in command_spec_list
    }

    assert command_map_dict["norgate_client_doctor"][1] is True
    probe_argv_list, probe_run_bool = command_map_dict["ibkr_connectivity_probe"]
    assert probe_run_bool is True
    assert "--client-id" in probe_argv_list
    assert "91" in probe_argv_list


def test_run_command_result_captures_nonzero_without_raising(tmp_path: Path) -> None:
    result_obj = bundle_module.run_command_result(
        name_str="failing_command",
        argv_list=["python", "-c", "import sys; print('visible'); sys.exit(7)"],
        output_dir_path=tmp_path,
        timeout_seconds_float=10.0,
    )

    assert result_obj.return_code_int == 7
    assert (tmp_path / "failing_command.stdout.txt").read_text(encoding="utf-8").strip() == "visible"
    meta_dict = json.loads((tmp_path / "failing_command.meta.json").read_text(encoding="utf-8"))
    assert meta_dict["return_code_int"] == 7
    assert meta_dict["timed_out_bool"] is False


def test_collect_bundle_writes_manifest_summary_and_zip(monkeypatch, tmp_path: Path) -> None:
    repo_root_path = tmp_path / "repo"
    repo_root_path.mkdir()
    (repo_root_path / "config.env").write_text(
        "NORGATE_API_TOKEN=secret\nNORGATE_API_HOST=100.64.1.2\n",
        encoding="utf-8",
    )
    manifest_path_obj = repo_root_path / "pod.yaml"
    manifest_path_obj.write_text(
        "broker:\n  account_route: U1\n  api_token: manifest_secret\n",
        encoding="utf-8",
    )
    log_dir_path = repo_root_path / "alpha" / "live" / "logs"
    log_dir_path.mkdir(parents=True)
    (log_dir_path / "live_critical_events.jsonl").write_text(
        '{"webhook_url": "https://discord.com/api/webhooks/a/b"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(bundle_module, "REPO_ROOT_PATH", repo_root_path)
    monkeypatch.setattr(
        bundle_module,
        "default_config_env_path_obj",
        lambda: repo_root_path / "config.env",
    )
    monkeypatch.setattr(bundle_module, "load_config_env_file", lambda override_existing_bool=False: {})
    monkeypatch.setattr(
        bundle_module,
        "build_command_spec_list",
        lambda parsed_args_obj: [
            ("safe_skip", ["SKIP", "test skip"], False),
            ("safe_run", ["python", "-c", "print('ok')"], True),
            ("safe_fail", ["python", "-c", "import sys; print('bad'); sys.exit(3)"], True),
        ],
    )

    parsed_args_obj = argparse.Namespace(
        mode_str="live",
        pod_id_str="pod_live_01",
        release_manifest_path_str=str(manifest_path_obj),
        releases_root_path_str=None,
        db_path_str=None,
        as_of_timestamp_str=None,
        output_root_path_str=str(tmp_path / "bundles"),
        tail_line_count_int=10,
        timeout_seconds_float=10.0,
        include_runner_details_bool=False,
        include_doctor_bool=False,
        doctor_broker_client_id_int=None,
        include_norgate_doctor_bool=False,
        ibkr_probe_client_id_int=None,
        no_zip_bool=False,
    )

    result_dict = bundle_module.collect_bundle_dict(parsed_args_obj)

    bundle_dir_path = Path(str(result_dict["bundle_dir_path_str"]))
    zip_path = Path(str(result_dict["zip_path_str"]))
    assert bundle_dir_path.exists()
    assert zip_path.exists()
    manifest_dict = json.loads((bundle_dir_path / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest_dict["schema_version_str"] == "vps_debug_bundle.v1"
    assert (bundle_dir_path / "config_env_redacted.txt").read_text(encoding="utf-8").startswith(
        "NORGATE_API_TOKEN=<REDACTED>"
    )
    assert "manifest_secret" not in (bundle_dir_path / "release_manifest_redacted.yaml").read_text(
        encoding="utf-8"
    )
    assert "discord.com/api/webhooks" not in (
        bundle_dir_path / "logs" / "live_critical_events.jsonl.tail.txt"
    ).read_text(encoding="utf-8")
    system_info_dict = json.loads((bundle_dir_path / "system_info.json").read_text(encoding="utf-8"))
    assert any(
        "Does not submit" in contract_str
        for contract_str in system_info_dict["safety_contract_list"]
    )
    assert "trusted operator/Codex review" in system_info_dict["sharing_notice_str"]
    assert (bundle_dir_path / "commands" / "safe_run.stdout.txt").read_text(encoding="utf-8").strip() == "ok"
    assert (bundle_dir_path / "commands" / "safe_fail.stdout.txt").read_text(encoding="utf-8").strip() == "bad"
    bundle_summary_dict = json.loads((bundle_dir_path / "bundle_summary.json").read_text(encoding="utf-8"))
    assert "safe_skip" in result_dict["skipped_command_name_list"]
    assert "safe_fail" in result_dict["failed_command_name_list"]
    assert "safe_fail" in bundle_summary_dict["failed_command_name_list"]
    import zipfile

    with zipfile.ZipFile(zip_path) as zip_file_obj:
        zip_name_set = set(zip_file_obj.namelist())
    assert "bundle_manifest.json" in zip_name_set
    assert "system_info.json" in zip_name_set
    assert "commands/safe_fail.meta.json" in zip_name_set
