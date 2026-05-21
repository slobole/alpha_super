import sys
from types import ModuleType

import pytest

import strategies.run_strategy as strategy_runner


def test_parse_run_kwarg_tuple_parses_json_values_and_string_paths():
    kwarg_dict = strategy_runner._parse_run_kwarg_tuple(
        (
            "market_cap_csv_path_str=C:\\data\\pit_market_cap.csv",
            "capital_base_float=250000",
            "save_debug_bool=true",
        )
    )

    assert kwarg_dict["market_cap_csv_path_str"] == "C:\\data\\pit_market_cap.csv"
    assert kwarg_dict["capital_base_float"] == 250000
    assert kwarg_dict["save_debug_bool"] is True


def test_run_strategy_module_passes_supported_strategy_kwargs(monkeypatch):
    module_name_str = "test_fake_run_strategy_module"
    captured_kwarg_dict = {}

    fake_module_obj = ModuleType(module_name_str)

    def run_variant(**kwargs):
        captured_kwarg_dict.update(kwargs)
        return None

    fake_module_obj.run_variant = run_variant
    monkeypatch.setitem(sys.modules, module_name_str, fake_module_obj)
    monkeypatch.setattr(
        strategy_runner,
        "_resolve_strategy_module_import_str",
        lambda strategy_name_str: module_name_str,
    )

    strategy_runner._run_strategy_module(
        strategy_name_str="fake.py",
        show_display_bool=False,
        save_results_bool=True,
        output_dir_str="results",
        dry_run_bool=False,
        strategy_kwarg_dict={"market_cap_csv_path_str": "C:\\data\\pit_market_cap.csv"},
    )

    assert captured_kwarg_dict["show_display_bool"] is False
    assert captured_kwarg_dict["save_results_bool"] is True
    assert captured_kwarg_dict["output_dir_str"] == "results"
    assert captured_kwarg_dict["market_cap_csv_path_str"] == "C:\\data\\pit_market_cap.csv"


def test_run_strategy_module_rejects_unsupported_strategy_kwargs(monkeypatch):
    module_name_str = "test_fake_rejecting_run_strategy_module"
    fake_module_obj = ModuleType(module_name_str)

    def run_variant(show_display_bool=False):
        return None

    fake_module_obj.run_variant = run_variant
    monkeypatch.setitem(sys.modules, module_name_str, fake_module_obj)
    monkeypatch.setattr(
        strategy_runner,
        "_resolve_strategy_module_import_str",
        lambda strategy_name_str: module_name_str,
    )

    with pytest.raises(ValueError, match="does not support strategy kwargs"):
        strategy_runner._run_strategy_module(
            strategy_name_str="fake.py",
            show_display_bool=False,
            save_results_bool=True,
            output_dir_str="results",
            dry_run_bool=False,
            strategy_kwarg_dict={"market_cap_csv_path_str": "C:\\data\\pit_market_cap.csv"},
        )
