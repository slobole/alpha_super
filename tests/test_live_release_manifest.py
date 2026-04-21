from __future__ import annotations

from pathlib import Path

import pytest

from alpha.live.release_manifest import load_release_list, parse_release_manifest


def test_parse_release_manifest_reads_example_release():
    manifest_path_str = str(
        Path("alpha/live/releases/excelence_trade_paper_001/pod_dv2_01_example.yaml").resolve()
    )

    release_obj = parse_release_manifest(manifest_path_str)

    assert release_obj.release_id_str.endswith(".pod_dv2.daily_moo.v1")
    assert release_obj.user_id_str in release_obj.release_id_str
    assert release_obj.pod_id_str == "pod_dv2_01_example"
    assert release_obj.execution_policy_str == "next_open_moo"
    assert release_obj.enabled_bool is False
    assert release_obj.session_calendar_id_str == "XNYS"
    assert release_obj.params_dict["capital_base_float"] == 100000.0
    assert release_obj.pod_budget_fraction_float == 0.03
    assert release_obj.auto_submit_enabled_bool is True
    assert release_obj.broker_host_str == "127.0.0.1"
    assert release_obj.broker_port_int == 7497
    assert release_obj.broker_client_id_int == 31
    assert release_obj.broker_timeout_seconds_float == 4.0


def test_parse_release_manifest_reads_qpi_example_release():
    manifest_path_str = str(
        Path("alpha/live/releases/excelence_trade_paper_001/pod_qpi_01.yaml").resolve()
    )

    release_obj = parse_release_manifest(manifest_path_str)

    assert release_obj.release_id_str.endswith(".pod_qpi.daily_moo.v1")
    assert release_obj.user_id_str in release_obj.release_id_str
    assert release_obj.pod_id_str == "pod_qpi_01"
    assert release_obj.strategy_import_str == "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy"
    assert release_obj.enabled_bool is False


def test_parse_release_manifest_reads_example_edi_live_release_with_broker_fields():
    manifest_path_str = str(
        Path("alpha/live/releases/example_edi/pod_dv2_01.yaml").resolve()
    )

    release_obj = parse_release_manifest(manifest_path_str)

    assert release_obj.user_id_str == "example_edi"
    assert release_obj.mode_str == "live"
    assert release_obj.account_route_str == "U_EXAMPLE_EDI_DV2"
    assert release_obj.broker_host_str == "127.0.0.1"
    assert release_obj.broker_port_int == 7496
    assert release_obj.broker_client_id_int == 31
    assert release_obj.broker_timeout_seconds_float == 4.0
    assert release_obj.auto_submit_enabled_bool is False


def test_parse_release_manifest_rejects_bad_execution_policy(tmp_path: Path):
    manifest_path_obj = tmp_path / "bad.yaml"
    manifest_path_obj.write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: bad",
                "  user_id: user_001",
                "  pod_id: pod_bad",
                "broker:",
                "  account_route: DU1",
                "strategy:",
                "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
                "  data_profile_str: norgate_eod_sp500_pit",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_snapshot_ready",
                "  execution_policy_str: bad_policy",
                "risk:",
                "  risk_profile_str: standard",
                "deployment:",
                "  mode: paper",
                "  enabled_bool: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported execution_policy_str"):
        parse_release_manifest(str(manifest_path_obj))


def test_load_release_list_rejects_duplicate_enabled_pods(tmp_path: Path):
    manifest_a_path_obj = tmp_path / "a.yaml"
    manifest_b_path_obj = tmp_path / "b.yaml"
    base_line_list = [
        "identity:",
        "  user_id: user_001",
        "  pod_id: pod_dup",
        "broker:",
        "  account_route: DU1",
        "strategy:",
        "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        "  data_profile_str: norgate_eod_sp500_pit",
        "  params: {}",
        "market:",
        "  session_calendar_id_str: XNYS",
        "schedule:",
        "  signal_clock_str: eod_snapshot_ready",
        "  execution_policy_str: next_open_moo",
        "risk:",
        "  risk_profile_str: standard",
        "deployment:",
        "  mode: paper",
        "  enabled_bool: true",
    ]
    manifest_a_path_obj.write_text(
        "\n".join(["identity:", "  release_id: release_a", *base_line_list[1:]]),
        encoding="utf-8",
    )
    manifest_b_path_obj.write_text(
        "\n".join(["identity:", "  release_id: release_b", *base_line_list[1:]]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate enabled pod_id_str"):
        load_release_list(str(tmp_path))


def test_parse_release_manifest_accepts_grouped_bootstrap_shape(tmp_path: Path):
    manifest_path_obj = tmp_path / "grouped.yaml"
    manifest_path_obj.write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: grouped_release",
                "  user_id: user_001",
                "  pod_id: pod_grouped",
                "broker:",
                "  account_route: DU100",
                "strategy:",
                "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
                "  data_profile_str: norgate_eod_sp500_pit",
                "  params:",
                "    max_positions_int: 7",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_snapshot_ready",
                "  execution_policy_str: next_open_moo",
                "bootstrap:",
                "  initial_cash_float: 250000.0",
                "risk:",
                "  risk_profile_str: standard_equity_mr",
                "deployment:",
                "  mode: paper",
                "  enabled_bool: true",
            ]
        ),
        encoding="utf-8",
    )

    release_obj = parse_release_manifest(str(manifest_path_obj))

    assert release_obj.params_dict["max_positions_int"] == 7
    assert release_obj.params_dict["capital_base_float"] == 250000.0


def test_parse_release_manifest_rejects_paper_mode_with_live_style_account(tmp_path: Path):
    manifest_path_obj = tmp_path / "bad_paper_account.yaml"
    manifest_path_obj.write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: grouped_release",
                "  user_id: user_001",
                "  pod_id: pod_grouped",
                "broker:",
                "  account_route: U100",
                "strategy:",
                "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
                "  data_profile_str: norgate_eod_sp500_pit",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_snapshot_ready",
                "  execution_policy_str: next_open_moo",
                "risk:",
                "  risk_profile_str: standard_equity_mr",
                "deployment:",
                "  mode: paper",
                "  enabled_bool: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paper-style IBKR account_route_str"):
        parse_release_manifest(str(manifest_path_obj))


def test_parse_release_manifest_normalizes_legacy_signal_clock_alias(tmp_path: Path):
    manifest_path_obj = tmp_path / "legacy_clock.yaml"
    manifest_path_obj.write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: grouped_release",
                "  user_id: user_001",
                "  pod_id: pod_grouped",
                "broker:",
                "  account_route: DU100",
                "strategy:",
                "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
                "  data_profile_str: norgate_eod_sp500_pit",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_close_plus_10m",
                "  execution_policy_str: next_open_moo",
                "risk:",
                "  risk_profile_str: standard_equity_mr",
                "deployment:",
                "  mode: paper",
                "  enabled_bool: true",
            ]
        ),
        encoding="utf-8",
    )

    release_obj = parse_release_manifest(str(manifest_path_obj))

    assert release_obj.signal_clock_str == "eod_snapshot_ready"


def test_parse_release_manifest_reads_execution_section(tmp_path: Path):
    manifest_path_obj = tmp_path / "execution.yaml"
    manifest_path_obj.write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: exec_release",
                "  user_id: user_001",
                "  pod_id: pod_exec",
                "broker:",
                "  account_route: DU100",
                "strategy:",
                "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
                "  data_profile_str: norgate_eod_sp500_pit",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_snapshot_ready",
                "  execution_policy_str: next_open_moo",
                "execution:",
                "  pod_budget_fraction_float: 0.05",
                "  auto_submit_enabled_bool: true",
                "risk:",
                "  risk_profile_str: standard_equity_mr",
                "deployment:",
                "  mode: paper",
                "  enabled_bool: true",
            ]
        ),
        encoding="utf-8",
    )

    release_obj = parse_release_manifest(str(manifest_path_obj))

    assert release_obj.pod_budget_fraction_float == 0.05
    assert release_obj.auto_submit_enabled_bool is True


def test_parse_release_manifest_reads_broker_section(tmp_path: Path):
    manifest_path_obj = tmp_path / "broker.yaml"
    manifest_path_obj.write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: broker_release",
                "  user_id: user_001",
                "  pod_id: pod_broker",
                "broker:",
                "  account_route: U100",
                "  host_str: 127.0.0.1",
                "  port_int: 7496",
                "  client_id_int: 41",
                "  timeout_seconds_float: 6.5",
                "strategy:",
                "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
                "  data_profile_str: norgate_eod_sp500_pit",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_snapshot_ready",
                "  execution_policy_str: next_open_moo",
                "risk:",
                "  risk_profile_str: standard_equity_mr",
                "deployment:",
                "  mode: live",
                "  enabled_bool: false",
            ]
        ),
        encoding="utf-8",
    )

    release_obj = parse_release_manifest(str(manifest_path_obj))

    assert release_obj.broker_host_str == "127.0.0.1"
    assert release_obj.broker_port_int == 7496
    assert release_obj.broker_client_id_int == 41
    assert release_obj.broker_timeout_seconds_float == 6.5
