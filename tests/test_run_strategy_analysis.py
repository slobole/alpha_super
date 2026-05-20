import sys
import importlib
from types import ModuleType, SimpleNamespace

import pytest

import scripts.research.run_strategy_analysis as analysis_runner


def _install_fake_strategy_module(monkeypatch, module_name_str: str, hook_name_tuple: tuple[str, ...]):
    fake_module_obj = ModuleType(module_name_str)
    for hook_name_str in hook_name_tuple:
        setattr(fake_module_obj, hook_name_str, lambda: None)
    monkeypatch.setitem(sys.modules, module_name_str, fake_module_obj)
    monkeypatch.setattr(
        analysis_runner,
        "_resolve_strategy_module_import_str",
        lambda strategy_ref_str: module_name_str,
    )
    return fake_module_obj


def test_strategy_analysis_runs_vanilla_and_friction_then_skips_missing_timing(monkeypatch, capsys):
    module_name_str = "test_fake_analysis_strategy"
    _install_fake_strategy_module(
        monkeypatch,
        module_name_str,
        ("run_variant", "run_friction_analysis"),
    )
    command_list = []

    def run_stub(command, cwd, check):
        command_list.append(tuple(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(analysis_runner.subprocess, "run", run_stub)

    return_code_int, result_list = analysis_runner.run_strategy_analysis(
        strategy_ref_str="fake.py",
        save_results_bool=False,
    )

    assert return_code_int == 0
    assert [result_obj.status_str for result_obj in result_list] == ["PASS", "PASS", "SKIP"]
    assert [command_tuple[2] for command_tuple in command_list] == [
        module_name_str,
        module_name_str,
    ]
    assert "strategies\\run_strategy.py" in command_list[0][1]
    assert "strategies\\run_friction_analysis.py" in command_list[1][1]
    assert "missing strategy hook: build_execution_timing_analysis_inputs(...)" in capsys.readouterr().out


def test_strategy_analysis_accepts_wired_module_class_import(monkeypatch):
    module_name_str = "test_fake_wired_analysis_strategy"
    _install_fake_strategy_module(
        monkeypatch,
        module_name_str,
        ("run_variant",),
    )
    captured_strategy_ref_list = []

    def resolve_stub(strategy_ref_str: str) -> str:
        captured_strategy_ref_list.append(strategy_ref_str)
        return module_name_str

    monkeypatch.setattr(analysis_runner, "_resolve_strategy_module_import_str", resolve_stub)

    module_import_str, strategy_module_obj = analysis_runner._load_strategy_module(
        f"{module_name_str}:FakeStrategy"
    )

    assert module_import_str == module_name_str
    assert strategy_module_obj is sys.modules[module_name_str]
    assert captured_strategy_ref_list == [module_name_str]


def test_strategy_analysis_runs_timing_when_hook_exists(monkeypatch):
    module_name_str = "test_fake_timing_strategy"
    _install_fake_strategy_module(
        monkeypatch,
        module_name_str,
        (
            "run_variant",
            "run_friction_analysis",
            "build_execution_timing_analysis_inputs",
        ),
    )
    command_list = []

    def run_stub(command, cwd, check):
        command_list.append(tuple(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(analysis_runner.subprocess, "run", run_stub)

    return_code_int, result_list = analysis_runner.run_strategy_analysis(
        strategy_ref_str="fake.py",
        analysis_tuple=("timing",),
        output_dir_str="custom_results",
        show_signal_progress_bool=True,
    )

    assert return_code_int == 0
    assert [result_obj.status_str for result_obj in result_list] == ["PASS"]
    assert len(command_list) == 1
    assert "scripts\\research\\execution_timing_analyzer.py" in command_list[0][1]
    assert command_list[0][2] == module_name_str
    assert "--output-dir" in command_list[0]
    assert "custom_results" in command_list[0]
    assert "--show-signal-progress" in command_list[0]


def test_strategy_analysis_stops_after_failure_without_keep_going(monkeypatch):
    module_name_str = "test_fake_failing_analysis_strategy"
    _install_fake_strategy_module(
        monkeypatch,
        module_name_str,
        ("run_variant", "run_friction_analysis"),
    )
    command_list = []

    def run_stub(command, cwd, check):
        command_list.append(tuple(command))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(analysis_runner.subprocess, "run", run_stub)

    return_code_int, result_list = analysis_runner.run_strategy_analysis(
        strategy_ref_str="fake.py",
        analysis_tuple=("vanilla", "friction"),
    )

    assert return_code_int == 1
    assert [result_obj.status_str for result_obj in result_list] == ["FAIL"]
    assert len(command_list) == 1


def test_strategy_analysis_returns_zero_when_everything_is_skipped(monkeypatch):
    _install_fake_strategy_module(monkeypatch, "test_fake_no_analysis_strategy", ())

    return_code_int, result_list = analysis_runner.run_strategy_analysis(
        strategy_ref_str="fake.py",
        analysis_tuple=("timing",),
    )

    assert return_code_int == 0
    assert [result_obj.status_str for result_obj in result_list] == ["SKIP"]


def test_unique_analysis_tuple_keeps_requested_order_without_duplicates():
    assert analysis_runner._unique_analysis_tuple(["timing", "vanilla", "timing"]) == (
        "timing",
        "vanilla",
    )
    assert analysis_runner._unique_analysis_tuple(None) == analysis_runner.SUPPORTED_ANALYSIS_TUPLE


def test_wired_strategy_modules_expose_all_analysis_hooks():
    from alpha.live.release_manifest import SUPPORTED_STRATEGY_IMPORT_TUPLE

    hook_tuple = (
        "run_variant",
        "run_friction_analysis",
        "build_execution_timing_analysis_inputs",
    )
    missing_hook_dict = {}

    for strategy_import_str in SUPPORTED_STRATEGY_IMPORT_TUPLE:
        module_import_str = strategy_import_str.split(":", maxsplit=1)[0]
        strategy_module_obj = importlib.import_module(module_import_str)
        missing_hook_list = [
            hook_name_str
            for hook_name_str in hook_tuple
            if not callable(getattr(strategy_module_obj, hook_name_str, None))
        ]
        if len(missing_hook_list) > 0:
            missing_hook_dict[module_import_str] = missing_hook_list

    assert missing_hook_dict == {}
