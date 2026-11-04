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


import subprocess

import pytest

from scripts.review.triage import read_git_diff_name_only, normalize_path_str


@pytest.mark.parametrize("path_str,tier_int", [
    ("scripts/setup_live_tasks.ps1", 3),
    ("scripts/live_debug/ibkr_connectivity_probe.py", 3),
    ("scripts/export_norgate_snapshot.py", 3),
    ("tests/test_dashboard_v4_orders.py", 3),
    ("tests/dashboard_v4_orders.test.cjs", 3),
    ("tests/test_client_operations.py", 3),
    ("tests/test_norgate_snapshot_parity.py", 3),
    ("alpha/strategy_registry.py", 2),
    ("tests/test_portfolio_manager.py", 2),
    ("tests/conftest.py", 2),
    (".gitattributes", 2),
    ("pyproject.toml", 2),
    ("uv.lock", 2),
    ("requirements.txt", 2),
    ("requirements-dev.txt", 2),
    ("docs/notes/example.txt", 0),
    ("portfolios/fund_growth.yaml", 1),
    ("alpha/scout/ledger.py", 1),
    ("scripts/research/example/replay_live_decisions.py", 1),
    ("tests/test_bench.py", 1),
    ("tests/test_research_proof_failures.py", 1),
])
def test_risk_classification_covers_operational_and_shared_surfaces(path_str, tier_int):
    classification_obj = classify_path_list([path_str]).path_classification_tuple[0]
    assert classification_obj.tier_int == tier_int
    assert not classification_obj.unclassified_bool


def test_unknown_path_requires_manual_classification(capsys):
    assert main(["--name-only", "new_package/order_router.py"]) == 2
    output_str = capsys.readouterr().out
    assert "UNCLASSIFIED" in output_str
    assert "manual_classification_required_bool=True" in output_str
    assert "tier_int=3" in output_str


def test_path_normalization_does_not_erase_filename_whitespace():
    assert normalize_path_str(" ./file name.py ") == " ./file name.py "
    assert normalize_path_str("./tests/test_live_runner.py") == "tests/test_live_runner.py"


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git_fn(*arg_tuple):
        return subprocess.run(["git", "-c", "core.fsmonitor=false", *arg_tuple],
                              check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()

    git_fn("init")
    git_fn("config", "user.name", "Fixture")
    git_fn("config", "user.email", "fixture@example.invalid")
    git_fn("config", "core.autocrlf", "false")
    hook_path = tmp_path / "empty-hooks"
    hook_path.mkdir()
    git_fn("config", "core.hooksPath", str(hook_path))
    git_fn("config", "commit.gpgsign", "false")
    for name_str in ("staged.txt", "unstaged.txt", "deleted.txt", "cancelled.txt", "alpha/live/runner.py"):
        path_obj = tmp_path / name_str
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text("original\n", encoding="utf-8")
    git_fn("add", ".")
    git_fn("commit", "-m", "fixture base")
    return tmp_path, git_fn


def test_git_paths_include_staged_unstaged_untracked_deleted_and_rename_source(git_repo):
    repo_path, git_fn = git_repo
    (repo_path / "staged.txt").write_text("staged edit\n", encoding="utf-8")
    git_fn("add", "staged.txt")
    (repo_path / "unstaged.txt").write_text("unstaged edit\n", encoding="utf-8")
    git_fn("rm", "deleted.txt")
    (repo_path / "docs").mkdir()
    git_fn("mv", "alpha/live/runner.py", "docs/retired.txt")
    unicode_name_str = "new file \u05d0.py"
    (repo_path / unicode_name_str).write_text("fixture", encoding="utf-8")
    path_list = read_git_diff_name_only()
    assert set(path_list) == {"staged.txt", "unstaged.txt", "deleted.txt",
                              "alpha/live/runner.py", "docs/retired.txt", unicode_name_str}
    assert len(path_list) == len(set(path_list))
    assert classify_path_list(path_list).tier_int == 3


def test_staged_and_unstaged_cancellation_still_needs_review(git_repo):
    repo_path, git_fn = git_repo
    path_obj = repo_path / "cancelled.txt"
    path_obj.write_text("index change\n", encoding="utf-8")
    git_fn("add", "cancelled.txt")
    path_obj.write_text("original\n", encoding="utf-8")
    assert git_fn("diff", "HEAD", "--name-only") == ""
    assert read_git_diff_name_only() == ["cancelled.txt"]


def test_base_includes_committed_index_and_worktree_changes(git_repo):
    repo_path, git_fn = git_repo
    base_str = git_fn("rev-parse", "HEAD")
    (repo_path / "staged.txt").write_text("committed change\n", encoding="utf-8")
    git_fn("add", "staged.txt")
    git_fn("commit", "-m", "fixture later")
    (repo_path / "unstaged.txt").write_text("current edit\n", encoding="utf-8")
    (repo_path / "new.txt").write_text("untracked\n", encoding="utf-8")
    assert set(read_git_diff_name_only(base_str)) == {"staged.txt", "unstaged.txt", "new.txt"}


def test_arbitrary_text_config_is_not_classified_as_documentation():
    classification_obj = classify_path_list(["config/runtime.txt"]).path_classification_tuple[0]
    assert classification_obj.unclassified_bool
    assert classification_obj.tier_int == 3
