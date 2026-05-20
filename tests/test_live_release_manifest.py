from __future__ import annotations

from pathlib import Path

import pytest

from alpha.live.release_manifest import (
    SUPPORTED_STRATEGY_IMPORT_TUPLE,
    load_release_list,
    parse_release_manifest,
    validate_enabled_deployment_for_mode,
)


RELEASE_TEMPLATE_PATH_TUPLE: tuple[Path, ...] = (
    Path("docs/live/release_templates/pod_dv2_daily_moo.yaml.example"),
    Path("docs/live/release_templates/pod_qpi_daily_moo.yaml.example"),
    Path(
        "docs/live/release_templates/"
        "pod_taa_btal_fallback_tqqq_vix_cash_monthly_open.yaml.example"
    ),
    Path(
        "docs/live/release_templates/"
        "pod_taa_btal_1n_fallback_tqqq_vix_cash_monthly_open.yaml.example"
    ),
    Path(
        "docs/live/release_templates/"
        "pod_taa_btal_linearity_1n_fallback_qqq_vix_cash_monthly_open.yaml.example"
    ),
    Path("docs/live/release_templates/pod_ndx_atr_normalized_monthly_open.yaml.example"),
    Path(
        "docs/live/release_templates/"
        "pod_ndx_atr_normalized_vxn_scaled_monthly_open.yaml.example"
    ),
)


def _write_guardrail_manifest(
    root_path_obj: Path,
    *,
    file_name_str: str,
    release_id_str: str,
    user_id_str: str,
    pod_id_str: str,
    mode_str: str,
    enabled_bool: bool,
    account_route_str: str,
) -> None:
    (root_path_obj / file_name_str).write_text(
        "\n".join(
            [
                "identity:",
                f"  release_id: {release_id_str}",
                f"  user_id: {user_id_str}",
                f"  pod_id: {pod_id_str}",
                "broker:",
                f"  account_route: {account_route_str}",
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
                f"  mode: {mode_str}",
                f"  enabled_bool: {'true' if enabled_bool else 'false'}",
            ]
        ),
        encoding="utf-8",
    )


def test_release_templates_cover_every_wired_strategy():
    template_strategy_import_set = {
        parse_release_manifest(str(template_path_obj.resolve())).strategy_import_str
        for template_path_obj in RELEASE_TEMPLATE_PATH_TUPLE
    }

    assert template_strategy_import_set == set(SUPPORTED_STRATEGY_IMPORT_TUPLE)


@pytest.mark.parametrize("template_path_obj", RELEASE_TEMPLATE_PATH_TUPLE)
def test_release_template_parses_and_starts_disabled(template_path_obj: Path):
    release_obj = parse_release_manifest(str(template_path_obj.resolve()))

    assert release_obj.enabled_bool is False
    assert release_obj.auto_submit_enabled_bool is False
    assert release_obj.user_id_str == "your_user"
    assert release_obj.release_id_str.startswith("your_user.")
    assert release_obj.broker_host_str == "127.0.0.1"
    assert release_obj.broker_port_int == 7497
    assert release_obj.broker_client_id_int == 31
    assert release_obj.broker_timeout_seconds_float == 4.0
    assert release_obj.session_calendar_id_str == "XNYS"
    assert release_obj.pod_budget_fraction_float == 0.03
    assert release_obj.params_dict["capital_base_float"] == 100000.0


def test_release_template_qpi_exposes_strategy_knobs():
    release_obj = parse_release_manifest(
        str(Path("docs/live/release_templates/pod_qpi_daily_moo.yaml.example").resolve())
    )

    assert release_obj.strategy_import_str == "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy"
    assert release_obj.data_profile_str == "norgate_eod_sp500_pit"
    assert release_obj.signal_clock_str == "eod_snapshot_ready"
    assert release_obj.execution_policy_str == "next_open_moo"
    assert release_obj.params_dict["qpi_threshold_float"] == 30.0
    assert release_obj.params_dict["max_entry_ibs_float"] == 0.1
    assert release_obj.params_dict["exit_rsi2_threshold_float"] == 90.0


def test_release_template_vxn_scaled_ndx_exposes_vxn_knobs():
    release_obj = parse_release_manifest(
        str(
            Path(
                "docs/live/release_templates/"
                "pod_ndx_atr_normalized_vxn_scaled_monthly_open.yaml.example"
            ).resolve()
        )
    )

    assert release_obj.strategy_import_str == (
        "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:"
        "VxnScaledAtrNormalizedNdxStrategy"
    )
    assert release_obj.data_profile_str == "norgate_eod_ndx_pit_plus_vxn_helper"
    assert release_obj.signal_clock_str == "month_end_snapshot_ready"
    assert release_obj.execution_policy_str == "next_month_first_open"
    assert release_obj.params_dict["vxn_symbol_str"] == "$VXN"
    assert release_obj.params_dict["target_vxn_pct_float"] == 22.0
    assert release_obj.params_dict["min_exposure_scale_float"] == 0.25
    assert release_obj.params_dict["max_exposure_scale_float"] == 1.0


def test_parse_release_manifest_accepts_next_open_market(tmp_path: Path):
    manifest_path_obj = tmp_path / "next_open_market.yaml"
    manifest_path_obj.write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_market.daily.v1",
                "  user_id: user_001",
                "  pod_id: pod_market",
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
                "  execution_policy_str: next_open_market",
                "risk:",
                "  risk_profile_str: standard",
                "deployment:",
                "  mode: paper",
                "  enabled_bool: true",
            ]
        ),
        encoding="utf-8",
    )

    release_obj = parse_release_manifest(str(manifest_path_obj))

    assert release_obj.execution_policy_str == "next_open_market"


def test_parse_release_manifest_incubation_client_id_override_wins(tmp_path: Path):
    manifest_path_obj = tmp_path / "incubation_override.yaml"
    manifest_path_obj.write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_inc.incubation.v1",
                "  user_id: user_001",
                "  pod_id: pod_inc",
                "broker:",
                "  account_route: SIM_pod_inc",
                "  client_id_int: 92",
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
                "  mode: incubation",
                "  enabled_bool: true",
            ]
        ),
        encoding="utf-8",
    )

    release_obj = parse_release_manifest(str(manifest_path_obj))

    assert release_obj.broker_client_id_int == 92


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


def test_validate_enabled_deployment_accepts_one_client_with_multiple_pods(tmp_path: Path):
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_a.yaml",
        release_id_str="user_001.pod_a.paper",
        user_id_str="user_001",
        pod_id_str="pod_a",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU1",
    )
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_b.yaml",
        release_id_str="user_001.pod_b.paper",
        user_id_str="user_001",
        pod_id_str="pod_b",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU2",
    )

    release_list = load_release_list(str(tmp_path))

    validate_enabled_deployment_for_mode(release_list, "paper")


def test_validate_enabled_deployment_rejects_mixed_enabled_users_in_selected_mode(tmp_path: Path):
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_a.yaml",
        release_id_str="user_001.pod_a.paper",
        user_id_str="user_001",
        pod_id_str="pod_a",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU1",
    )
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_b.yaml",
        release_id_str="user_002.pod_b.paper",
        user_id_str="user_002",
        pod_id_str="pod_b",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU2",
    )

    release_list = load_release_list(str(tmp_path))

    with pytest.raises(ValueError, match="one client per VPS/deployment"):
        validate_enabled_deployment_for_mode(release_list, "paper")


def test_validate_enabled_deployment_ignores_disabled_mixed_client_release(tmp_path: Path):
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_a.yaml",
        release_id_str="user_001.pod_a.paper",
        user_id_str="user_001",
        pod_id_str="pod_a",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU1",
    )
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_b.yaml",
        release_id_str="user_002.pod_b.paper",
        user_id_str="user_002",
        pod_id_str="pod_b",
        mode_str="paper",
        enabled_bool=False,
        account_route_str="DU2",
    )

    release_list = load_release_list(str(tmp_path))

    validate_enabled_deployment_for_mode(release_list, "paper")


def test_validate_enabled_deployment_ignores_other_mode_enabled_release(tmp_path: Path):
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_paper.yaml",
        release_id_str="paper_user.pod_a.paper",
        user_id_str="paper_user",
        pod_id_str="pod_paper",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU1",
    )
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_live.yaml",
        release_id_str="live_user.pod_b.live",
        user_id_str="live_user",
        pod_id_str="pod_live",
        mode_str="live",
        enabled_bool=True,
        account_route_str="U1",
    )

    release_list = load_release_list(str(tmp_path))

    validate_enabled_deployment_for_mode(release_list, "paper")


def test_validate_enabled_deployment_rejects_enabled_placeholder_account_route(tmp_path: Path):
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_placeholder.yaml",
        release_id_str="user_001.pod_placeholder.paper",
        user_id_str="user_001",
        pod_id_str="pod_placeholder",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU_YOUR_PAPER_ACCOUNT",
    )

    release_list = load_release_list(str(tmp_path))

    with pytest.raises(ValueError, match="real IBKR account/subaccount"):
        validate_enabled_deployment_for_mode(release_list, "paper")


def test_validate_enabled_deployment_ignores_disabled_placeholder_account_route(tmp_path: Path):
    _write_guardrail_manifest(
        tmp_path,
        file_name_str="pod_placeholder.yaml",
        release_id_str="user_001.pod_placeholder.paper",
        user_id_str="user_001",
        pod_id_str="pod_placeholder",
        mode_str="paper",
        enabled_bool=False,
        account_route_str="DU_YOUR_PAPER_ACCOUNT",
    )

    release_list = load_release_list(str(tmp_path))

    validate_enabled_deployment_for_mode(release_list, "paper")


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
