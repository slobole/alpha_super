from scripts.review.triage import _dedupe_path_list, classify_path_list, main


def test_docs_only_is_tier_0():
    result = classify_path_list(["docs/notes/README.md"])

    assert result.tier_int == 0
    assert result.live_impact_checklist_required_bool is False
    assert result.required_agent_tuple == ()


def test_strategy_file_is_tier_1():
    result = classify_path_list(["strategies/dv2/strategy_mr_dv2.py"])

    assert result.tier_int == 1
    assert result.required_agent_tuple == ("quant-pitfalls",)


def test_engine_file_is_tier_2():
    result = classify_path_list(["alpha/engine/backtest.py"])

    assert result.tier_int == 2
    assert result.required_agent_tuple == (
        "quant-pitfalls",
        "parity",
        "coverage",
    )


def test_live_runner_reconcile_and_release_yaml_are_tier_3():
    result = classify_path_list(
        [
            "alpha/live/runner.py",
            "alpha/live/reconcile.py",
            "alpha/live/releases/caspersky_account/pod.yaml",
        ]
    )

    assert result.tier_int == 3
    assert result.live_impact_checklist_required_bool is True
    assert result.required_agent_tuple == (
        "parity",
        "failure-modes",
        "coverage",
    )


def test_live_runbook_is_tier_3_operator_surface():
    result = classify_path_list(["docs/live/LIVE_RUNBOOK.md"])

    assert result.tier_int == 3
    assert result.live_impact_checklist_required_bool is True


def test_root_incubation_flow_doc_is_tier_3_operator_surface():
    result = classify_path_list(["INCUBATION_FLOW.md"])

    assert result.tier_int == 3
    assert result.live_impact_checklist_required_bool is True


def test_mixed_files_choose_highest_tier_and_keep_quant_agent_when_needed():
    result = classify_path_list(
        [
            "docs/notes/README.md",
            "strategies/dv2/strategy_mr_dv2.py",
            "alpha/live/runner.py",
        ]
    )

    assert result.tier_int == 3
    assert result.required_agent_tuple == (
        "quant-pitfalls",
        "parity",
        "failure-modes",
        "coverage",
    )


def test_run_daily_runtime_controls_test_is_tier_2_engine_surface():
    result = classify_path_list(["tests/test_run_daily_runtime_controls.py"])

    assert result.tier_int == 2
    assert result.required_agent_tuple == (
        "quant-pitfalls",
        "parity",
        "coverage",
    )


def test_cli_name_only_prints_tier(capsys):
    exit_code_int = main(["--name-only", "alpha/live/runner.py"])

    captured = capsys.readouterr()
    assert exit_code_int == 0
    assert "tier_int=3" in captured.out
    assert "live_impact_checklist_required_bool=True" in captured.out


def test_path_dedupe_preserves_first_seen_order():
    result_list = _dedupe_path_list(
        ["alpha/live/runner.py", "AGENTS.md", "alpha/live/runner.py"]
    )

    assert result_list == ["alpha/live/runner.py", "AGENTS.md"]
