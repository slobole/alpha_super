import importlib
import inspect
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

    def build_capacity_analysis_inputs(
        capital_base_float=100_000.0,
        backtest_start_date_str=None,
        end_date_str=None,
    ):
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
    assert set(study_result_obj.capacity_curve_df["window_str"]) == {
        "full_history",
        "recent_5y",
    }
    assert study_result_obj.capacity_curve_df.groupby("window_str")[
        "capital_base_float"
    ].apply(list).to_dict() == {
        "full_history": [50_000.0, 250_000.0],
        "recent_5y": [50_000.0, 250_000.0],
    }
    assert study_result_obj.output_dir_path is not None
    output_text_str = capsys.readouterr().out
    assert "CapacityAnalysis summary" in output_text_str
    assert "Outer capacity: >= $250,000" in output_text_str


def test_runner_uses_exact_trailing_five_year_start_and_common_endpoint(monkeypatch):
    module_name_str = "test_fake_long_capacity_strategy"
    fake_module_obj = ModuleType(module_name_str)
    captured_date_tuple_list: list[tuple[str | None, str | None]] = []

    def build_capacity_analysis_inputs(
        capital_base_float=100_000.0,
        backtest_start_date_str=None,
        end_date_str=None,
    ):
        captured_date_tuple_list.append((backtest_start_date_str, end_date_str))
        start_date_str = backtest_start_date_str or "2015-06-30"
        effective_end_date_str = end_date_str or "2025-06-30"
        input_dict = _runner_input_dict(float(capital_base_float))
        full_date_idx = pd.date_range(start_date_str, effective_end_date_str, freq="B")
        pricing_start_ts = pd.Timestamp(start_date_str) - pd.offsets.BDay(30)
        pricing_date_idx = pd.date_range(pricing_start_ts, effective_end_date_str, freq="B")
        input_dict["pricing_data_df"] = input_dict["pricing_data_df"].iloc[:0].reindex(
            pricing_date_idx
        )
        input_dict["pricing_data_df"][("AAA", "Close")] = 10.0
        input_dict["pricing_data_df"][("AAA", "Volume")] = 100_000.0
        input_dict["pricing_data_df"][("$SPX", "Close")] = [
            100.0 + index_int * 0.01 for index_int in range(len(pricing_date_idx))
        ]
        strategy_obj = input_dict["strategy_obj"]
        strategy_obj.results = pd.DataFrame(
            {
                "total_value": [
                    capital_base_float * (1.0002**index_int)
                    for index_int in range(len(full_date_idx))
                ]
            },
            index=full_date_idx,
        )
        strategy_obj._transactions = pd.DataFrame(
            [
                {
                    "bar": full_date_idx[-1],
                    "asset": "AAA",
                    "amount": 10.0,
                    "price": 10.0,
                    "total_value": 100.0,
                    "commission": 1.0,
                }
            ]
        )
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
            save_results_bool=False,
        )
    finally:
        sys.modules.pop(module_name_str, None)

    assert captured_date_tuple_list == [
        (None, None),
        ("2020-06-30", "2025-06-30"),
    ]
    assert study_result_obj is not None
    assert study_result_obj.summary_dict["window_date_dict"] == {
        "full_history": {
            "actual_start_date_str": "2015-06-30",
            "actual_end_date_str": "2025-06-30",
        },
        "recent_5y": {
            "actual_start_date_str": "2020-06-30",
            "actual_end_date_str": "2025-06-30",
        },
    }


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


def test_runner_requires_dual_window_date_parameters(monkeypatch):
    module_name_str = "test_fake_missing_capacity_dates_strategy"
    fake_module_obj = ModuleType(module_name_str)

    def build_capacity_analysis_inputs(capital_base_float=100_000.0):
        return _runner_input_dict(float(capital_base_float))

    fake_module_obj.build_capacity_analysis_inputs = build_capacity_analysis_inputs
    sys.modules[module_name_str] = fake_module_obj
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda _strategy_name_str: module_name_str,
    )
    try:
        with pytest.raises(TypeError, match="dual-window reruns"):
            runner_module._run_capacity_analysis_module(
                module_name_str,
                (100_000.0,),
                save_results_bool=False,
            )
    finally:
        sys.modules.pop(module_name_str, None)


def test_runner_fails_when_builder_ignores_recent_start(monkeypatch):
    module_name_str = "test_fake_ignored_recent_start_strategy"
    fake_module_obj = ModuleType(module_name_str)

    def build_capacity_analysis_inputs(
        capital_base_float=100_000.0,
        backtest_start_date_str=None,
        end_date_str=None,
    ):
        input_dict = _runner_input_dict(float(capital_base_float))
        date_idx = pd.date_range("2015-06-30", "2025-06-30", freq="B")
        pricing_date_idx = pd.date_range("2015-05-01", "2025-06-30", freq="B")
        pricing_data_df = input_dict["pricing_data_df"].iloc[:0].reindex(pricing_date_idx)
        pricing_data_df[("AAA", "Close")] = 10.0
        pricing_data_df[("AAA", "Volume")] = 100_000.0
        pricing_data_df[("$SPX", "Close")] = [
            100.0 + index_int * 0.01 for index_int in range(len(pricing_date_idx))
        ]
        input_dict["pricing_data_df"] = pricing_data_df
        strategy_obj = input_dict["strategy_obj"]
        strategy_obj.results = pd.DataFrame(
            {
                "total_value": [
                    capital_base_float * (1.0002**index_int)
                    for index_int in range(len(date_idx))
                ]
            },
            index=date_idx,
        )
        strategy_obj._transactions["bar"] = date_idx[-1]
        return input_dict

    fake_module_obj.build_capacity_analysis_inputs = build_capacity_analysis_inputs
    sys.modules[module_name_str] = fake_module_obj
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda _strategy_name_str: module_name_str,
    )
    try:
        with pytest.raises(ValueError, match="did not honor"):
            runner_module._run_capacity_analysis_module(
                module_name_str,
                (100_000.0,),
                save_results_bool=False,
            )
    finally:
        sys.modules.pop(module_name_str, None)


def test_runner_fails_when_moo_builder_omits_impact_profile(monkeypatch):
    module_name_str = "test_fake_missing_moo_profile_strategy"
    fake_module_obj = ModuleType(module_name_str)

    def build_capacity_analysis_inputs(
        capital_base_float=100_000.0,
        backtest_start_date_str=None,
        end_date_str=None,
    ):
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

    def build_capacity_analysis_inputs(
        capital_base_float=100_000.0,
        backtest_start_date_str=None,
        end_date_str=None,
    ):
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

        def ignored_aum_builder(
            capital_base_float=100_000.0,
            backtest_start_date_str=None,
            end_date_str=None,
        ):
            return _runner_input_dict(100_000.0)

        fake_module_obj.build_capacity_analysis_inputs = ignored_aum_builder
        with pytest.raises(ValueError, match="expected 250000"):
            runner_module._run_capacity_analysis_module(module_name_str, (250_000.0,))
    finally:
        sys.modules.pop(module_name_str, None)


def test_runner_supports_real_moc_report_branch(tmp_path, monkeypatch):
    module_name_str = "test_fake_moc_capacity_strategy"
    fake_module_obj = ModuleType(module_name_str)

    def build_capacity_analysis_inputs(
        capital_base_float=100_000.0,
        backtest_start_date_str=None,
        end_date_str=None,
    ):
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


def test_all_capacity_builders_accept_v2_1_window_contract():
    pytest.importorskip("norgatedata")
    from alpha.bench import catalog

    invalid_builder_list = []
    required_parameter_set = {
        "capital_base_float",
        "backtest_start_date_str",
        "end_date_str",
    }
    for strategy_entry_obj in catalog.list_strategies():
        if not strategy_entry_obj.has_capacity_analysis_bool:
            continue
        strategy_module_obj = importlib.import_module(strategy_entry_obj.module_import_str)
        build_inputs_fn = strategy_module_obj.build_capacity_analysis_inputs
        missing_parameter_set = required_parameter_set.difference(
            inspect.signature(build_inputs_fn).parameters
        )
        if missing_parameter_set:
            invalid_builder_list.append(
                (strategy_entry_obj.module_import_str, sorted(missing_parameter_set))
            )

    assert invalid_builder_list == []
