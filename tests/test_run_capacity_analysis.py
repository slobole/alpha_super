import importlib
import sys
from types import ModuleType

import pandas as pd
import pytest

import strategies.run_capacity_analysis as runner_module
from alpha.engine.strategy import Strategy


class RunnerToyStrategy(Strategy):
    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


def _runner_input_dict(capital_base_float: float) -> dict[str, object]:
    date_idx = pd.date_range("2020-01-02", periods=800, freq="B")
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Close"): [10.0] * len(date_idx),
            ("AAA", "Volume"): [100_000.0] * len(date_idx),
            ("$SPX", "Close"): [100.0 + index_int * 0.01 for index_int in range(len(date_idx))],
        },
        index=date_idx,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    strategy_obj = RunnerToyStrategy(
        name="runner_toy",
        benchmarks=["$SPX"],
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
        performance_benchmark_symbol_str="$SPX",
        performance_benchmark_adjustment_str="TOTALRETURN",
    )
    strategy_obj._transactions = pd.DataFrame(
        [
            {
                "bar": date_idx[-1],
                "asset": "AAA",
                "amount": 10.0,
                "price": 10.0,
                "total_value": 100.0,
                "commission": 1.0,
            }
        ]
    )
    strategy_obj.results = pd.DataFrame(
        {"total_value": [capital_base_float * (1.0002 ** index_int) for index_int in range(len(date_idx))]},
        index=date_idx,
    )
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_LARGE_MIXED",
    }


def test_runner_loops_custom_aum_grid_and_writes_one_study(tmp_path, monkeypatch, capsys):
    module_name_str = "test_fake_capacity_strategy"
    fake_module_obj = ModuleType(module_name_str)
    captured_aum_list: list[float] = []

    def build_capacity_analysis_inputs(capital_base_float=100_000.0, **_kwargs):
        captured_aum_list.append(float(capital_base_float))
        return _runner_input_dict(float(capital_base_float))

    fake_module_obj.build_capacity_analysis_inputs = build_capacity_analysis_inputs
    sys.modules[module_name_str] = fake_module_obj
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda _strategy_name_str: module_name_str,
    )
    try:
        study_result_obj = runner_module._run_capacity_analysis_module(
            strategy_name_str=module_name_str,
            aum_float_tuple=(250_000.0, 50_000.0),
            output_dir_str=str(tmp_path),
        )
    finally:
        sys.modules.pop(module_name_str, None)

    assert captured_aum_list == [50_000.0, 250_000.0]
    assert study_result_obj is not None
    assert study_result_obj.capacity_curve_df["capital_base_float"].tolist() == [
        50_000.0,
        250_000.0,
    ]
    assert study_result_obj.output_dir_path is not None
    assert "CapacityAnalysis summary" in capsys.readouterr().out


def test_runner_fails_when_builder_is_missing(monkeypatch):
    module_name_str = "test_fake_missing_capacity_strategy"
    sys.modules[module_name_str] = ModuleType(module_name_str)
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda _strategy_name_str: module_name_str,
    )
    try:
        with pytest.raises(AttributeError, match="build_capacity_analysis_inputs"):
            runner_module._run_capacity_analysis_module(
                strategy_name_str=module_name_str,
                aum_float_tuple=(100_000.0,),
            )
    finally:
        sys.modules.pop(module_name_str, None)


def test_runner_fails_when_moo_builder_omits_impact_profile(monkeypatch):
    module_name_str = "test_fake_missing_moo_profile_strategy"
    fake_module_obj = ModuleType(module_name_str)

    def build_capacity_analysis_inputs(capital_base_float=100_000.0):
        input_dict = _runner_input_dict(float(capital_base_float))
        input_dict.pop("impact_profile_str")
        return input_dict

    fake_module_obj.build_capacity_analysis_inputs = build_capacity_analysis_inputs
    sys.modules[module_name_str] = fake_module_obj
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda _strategy_name_str: module_name_str,
    )
    try:
        with pytest.raises(ValueError, match="requires impact_profile_str"):
            runner_module._run_capacity_analysis_module(
                module_name_str,
                (100_000.0,),
                save_results_bool=False,
            )
    finally:
        sys.modules.pop(module_name_str, None)


def test_runner_fails_when_moo_builder_returns_invalid_impact_profile(monkeypatch):
    module_name_str = "test_fake_invalid_moo_profile_strategy"
    fake_module_obj = ModuleType(module_name_str)

    def build_capacity_analysis_inputs(capital_base_float=100_000.0):
        input_dict = _runner_input_dict(float(capital_base_float))
        input_dict["impact_profile_str"] = "INVALID_PROFILE"
        return input_dict

    fake_module_obj.build_capacity_analysis_inputs = build_capacity_analysis_inputs
    sys.modules[module_name_str] = fake_module_obj
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda _strategy_name_str: module_name_str,
    )
    try:
        with pytest.raises(ValueError, match="Supported profiles"):
            runner_module._run_capacity_analysis_module(
                module_name_str,
                (100_000.0,),
                save_results_bool=False,
            )
    finally:
        sys.modules.pop(module_name_str, None)


def test_runner_requires_builder_to_accept_and_honor_aum(monkeypatch):
    module_name_str = "test_fake_bad_capacity_strategy"
    fake_module_obj = ModuleType(module_name_str)

    def missing_aum_builder():
        return _runner_input_dict(100_000.0)

    fake_module_obj.build_capacity_analysis_inputs = missing_aum_builder
    sys.modules[module_name_str] = fake_module_obj
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda _strategy_name_str: module_name_str,
    )
    try:
        with pytest.raises(TypeError, match="must accept capital_base_float"):
            runner_module._run_capacity_analysis_module(module_name_str, (250_000.0,))

        def ignored_aum_builder(capital_base_float=100_000.0):
            return _runner_input_dict(100_000.0)

        fake_module_obj.build_capacity_analysis_inputs = ignored_aum_builder
        with pytest.raises(ValueError, match="expected 250000"):
            runner_module._run_capacity_analysis_module(module_name_str, (250_000.0,))
    finally:
        sys.modules.pop(module_name_str, None)


def test_runner_supports_real_moc_report_branch(tmp_path, monkeypatch):
    module_name_str = "test_fake_moc_capacity_strategy"
    fake_module_obj = ModuleType(module_name_str)

    def build_capacity_analysis_inputs(capital_base_float=100_000.0):
        input_dict = _runner_input_dict(float(capital_base_float))
        input_dict["execution_policy_str"] = "MOC"
        return input_dict

    fake_module_obj.build_capacity_analysis_inputs = build_capacity_analysis_inputs
    sys.modules[module_name_str] = fake_module_obj
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda _strategy_name_str: module_name_str,
    )
    try:
        study_result_obj = runner_module._run_capacity_analysis_module(
            module_name_str,
            (100_000.0,),
            output_dir_str=str(tmp_path),
        )
    finally:
        sys.modules.pop(module_name_str, None)

    assert study_result_obj is not None
    report_html_str = (study_result_obj.output_dir_path / "report.html").read_text(encoding="utf-8")
    assert "Worked MOC example" in report_html_str


def test_runner_dry_run_resolves_existing_file(capsys):
    result_obj = runner_module._run_capacity_analysis_module(
        strategy_name_str="dv2/strategy_mr_dv2.py",
        dry_run_bool=True,
    )
    assert result_obj is None
    assert "strategies.dv2.strategy_mr_dv2" in capsys.readouterr().out


def test_deployment_wired_strategy_modules_expose_capacity_builders():
    pytest.importorskip("norgatedata")
    from alpha.live.release_manifest import SUPPORTED_STRATEGY_IMPORT_TUPLE

    missing_builder_list = []
    for strategy_import_str in SUPPORTED_STRATEGY_IMPORT_TUPLE:
        module_import_str = strategy_import_str.split(":", maxsplit=1)[0]
        strategy_module_obj = importlib.import_module(module_import_str)
        if not callable(getattr(strategy_module_obj, "build_capacity_analysis_inputs", None)):
            missing_builder_list.append(module_import_str)
    assert missing_builder_list == []
