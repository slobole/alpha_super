import pandas as pd
import pytest

from alpha.bench import catalog
from alpha.engine.crisis import CrisisPeriodConfig, SUPPORTED_CRISIS_STRATEGY_SPEC_MAP
from alpha.engine.execution_timing import ExecutionTimingAnalyzer
from alpha.engine.stress_test import StressTestAnalyzer
from scripts.research import run_strategy_analysis as analysis_runner
from strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock import (
    DEFAULT_CONFIG as BASE_DEFAULT_CONFIG,
    _write_assumptions_md,
)
from strategies.mean_reversion import strategy_mr_us_sector_etf_ibs_downshock_vox_iyr as variant_module
from strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr import (
    DEFAULT_CONFIG,
    STRATEGY_NAME_STR,
    VOX_IYR_SYMBOL_TUPLE,
    UsSectorEtfIbsDownshockVoxIyrStrategy,
    _write_vox_iyr_notes_md,
    run_variant,
)


def make_pricing_data_df(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    column_map_dict: dict[tuple[str, str], pd.Series] = {}
    for symbol_str in VOX_IYR_SYMBOL_TUPLE:
        column_map_dict[(symbol_str, "Open")] = pd.Series(
            100.0,
            index=date_index,
        )
        column_map_dict[(symbol_str, "High")] = pd.Series(
            101.0,
            index=date_index,
        )
        column_map_dict[(symbol_str, "Low")] = pd.Series(
            99.0,
            index=date_index,
        )
        column_map_dict[(symbol_str, "Close")] = pd.Series(
            100.0,
            index=date_index,
        )
    column_map_dict[("$SPX", "Close")] = pd.Series(
        range(2_000, 2_000 + len(date_index)),
        index=date_index,
        dtype=float,
    )
    pricing_data_df = pd.DataFrame(column_map_dict, index=date_index)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def make_trade_pricing_data_df() -> pd.DataFrame:
    date_index = pd.bdate_range("2024-01-02", periods=40)
    pricing_data_df = make_pricing_data_df(date_index)
    for symbol_str in VOX_IYR_SYMBOL_TUPLE:
        pricing_data_df[(symbol_str, "Dividend")] = 0.0
    pricing_data_df = pricing_data_df.sort_index(axis=1)
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        symbol_str: "CAPITALSPECIAL"
        for symbol_str in VOX_IYR_SYMBOL_TUPLE
    }
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"]["$SPX"] = (
        "TOTALRETURN"
    )
    shock_ts = date_index[25]
    post_shock_ts = date_index[26]
    exit_signal_ts = date_index[27]
    exit_fill_ts = date_index[28]

    pricing_data_df.loc[shock_ts, ("XLB", "Open")] = 100.0
    pricing_data_df.loc[shock_ts, ("XLB", "High")] = 100.0
    pricing_data_df.loc[shock_ts, ("XLB", "Low")] = 90.0
    pricing_data_df.loc[shock_ts, ("XLB", "Close")] = 90.0

    pricing_data_df.loc[post_shock_ts, ("XLB", "Open")] = 91.0
    pricing_data_df.loc[post_shock_ts, ("XLB", "High")] = 92.0
    pricing_data_df.loc[post_shock_ts, ("XLB", "Low")] = 90.0
    pricing_data_df.loc[post_shock_ts, ("XLB", "Close")] = 91.0
    pricing_data_df.loc[post_shock_ts, ("XLB", "Dividend")] = 1.0

    pricing_data_df.loc[exit_signal_ts, ("XLB", "Open")] = 91.0
    pricing_data_df.loc[exit_signal_ts, ("XLB", "High")] = 101.0
    pricing_data_df.loc[exit_signal_ts, ("XLB", "Low")] = 90.0
    pricing_data_df.loc[exit_signal_ts, ("XLB", "Close")] = 100.9
    pricing_data_df.loc[exit_fill_ts, ("XLB", "Open")] = 101.0
    return pricing_data_df


def test_default_config_uses_exact_requested_basket_and_unchanged_costs():
    strategy_obj = UsSectorEtfIbsDownshockVoxIyrStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[DEFAULT_CONFIG.benchmark_symbol_str],
    )

    assert DEFAULT_CONFIG.symbol_tuple == VOX_IYR_SYMBOL_TUPLE
    assert strategy_obj.config_obj is DEFAULT_CONFIG
    assert len(VOX_IYR_SYMBOL_TUPLE) == 11
    assert "VOX" in VOX_IYR_SYMBOL_TUPLE
    assert "IYR" in VOX_IYR_SYMBOL_TUPLE
    assert "XLC" not in VOX_IYR_SYMBOL_TUPLE
    assert "XLRE" not in VOX_IYR_SYMBOL_TUPLE
    assert DEFAULT_CONFIG.max_positions_int == BASE_DEFAULT_CONFIG.max_positions_int
    assert DEFAULT_CONFIG.sizing_universe_count_int == 11
    assert strategy_obj.target_weight_float == pytest.approx(1.5 / 11.0)
    assert strategy_obj._slippage == pytest.approx(0.00025)
    assert strategy_obj._commission_per_share == pytest.approx(0.005)
    assert strategy_obj._commission_minimum == pytest.approx(1.0)


def test_run_variant_preserves_causal_startup_and_proxy_basket():
    date_index = pd.bdate_range("2024-01-02", periods=24)
    pricing_data_df = make_pricing_data_df(date_index)

    strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        pricing_data_df=pricing_data_df,
        audit_override_bool=False,
    )

    assert isinstance(strategy_obj, UsSectorEtfIbsDownshockVoxIyrStrategy)
    assert strategy_obj.name == STRATEGY_NAME_STR
    assert strategy_obj.symbol_tuple == VOX_IYR_SYMBOL_TUPLE
    assert strategy_obj.results.index[0] == date_index[22]
    assert strategy_obj.get_transactions().empty


def test_timing_default_cell_matches_vanilla_next_open_contract(monkeypatch):
    pricing_data_df = make_trade_pricing_data_df()
    vanilla_strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        pricing_data_df=pricing_data_df,
        audit_override_bool=False,
    )
    monkeypatch.setattr(
        variant_module,
        "get_us_sector_etf_ibs_downshock_data",
        lambda _config_obj: pricing_data_df,
    )
    timing_input_dict = variant_module.build_execution_timing_analysis_inputs()
    timing_result_obj = ExecutionTimingAnalyzer(
        strategy_factory_fn=timing_input_dict["strategy_factory_fn"],
        pricing_data_df=timing_input_dict["pricing_data_df"],
        calendar_idx=timing_input_dict["calendar_idx"],
        entry_timing_str_tuple=("next_open",),
        exit_timing_str_tuple=("next_open",),
        save_output_bool=False,
        audit_override_bool=False,
        order_generation_mode_str=timing_input_dict["order_generation_mode_str"],
        risk_model_str=timing_input_dict["risk_model_str"],
        default_entry_timing_str=timing_input_dict["default_entry_timing_str"],
        default_exit_timing_str=timing_input_dict["default_exit_timing_str"],
    ).run()
    timing_strategy_obj = timing_result_obj.strategy_map[("next_open", "next_open")]

    assert not vanilla_strategy_obj.get_transactions().empty
    pd.testing.assert_series_equal(
        vanilla_strategy_obj.results["total_value"],
        timing_strategy_obj.results["total_value"],
        check_names=False,
        check_freq=False,
        rtol=0.0,
        atol=1e-8,
    )
    pd.testing.assert_frame_equal(
        vanilla_strategy_obj.get_transactions()
        .drop(columns=["order_id"])
        .reset_index(drop=True),
        timing_strategy_obj.get_transactions()
        .drop(columns=["order_id"])
        .reset_index(drop=True),
    )


def test_stress_registry_uses_vanilla_strategy_and_calendar(monkeypatch):
    pricing_data_df = make_trade_pricing_data_df()
    monkeypatch.setattr(
        variant_module,
        "get_us_sector_etf_ibs_downshock_data",
        lambda _config_obj: pricing_data_df,
    )
    strategy_spec_obj = SUPPORTED_CRISIS_STRATEGY_SPEC_MAP[STRATEGY_NAME_STR]
    context_dict = strategy_spec_obj.load_context_fn()
    strategy_obj = strategy_spec_obj.build_strategy_fn(context_dict)

    assert type(strategy_obj) is UsSectorEtfIbsDownshockVoxIyrStrategy
    assert context_dict["calendar_idx"].equals(
        variant_module.resolve_us_sector_etf_execution_calendar_idx(
            pricing_data_df=pricing_data_df,
            config_obj=DEFAULT_CONFIG,
        )
    )
    assert strategy_obj._capital_base == DEFAULT_CONFIG.capital_base_float
    assert strategy_obj._slippage == pytest.approx(0.00025)


def test_registered_stress_analyzer_runs_and_discloses_pre_inception(monkeypatch):
    pricing_data_df = make_trade_pricing_data_df()
    monkeypatch.setattr(
        variant_module,
        "get_us_sector_etf_ibs_downshock_data",
        lambda _config_obj: pricing_data_df,
    )
    supported_start_ts = pd.Timestamp(pricing_data_df.index[28])
    supported_end_ts = pd.Timestamp(pricing_data_df.index[30])

    stress_result_obj = StressTestAnalyzer(
        strategy_key_str=STRATEGY_NAME_STR,
        crisis_period_list=[
            CrisisPeriodConfig(
                crisis_name_str="pre_inception_crisis",
                start_date_str="2020-01-02",
                end_date_str="2020-01-31",
            ),
            CrisisPeriodConfig(
                crisis_name_str="supported_crisis",
                start_date_str=supported_start_ts.strftime("%Y-%m-%d"),
                end_date_str=supported_end_ts.strftime("%Y-%m-%d"),
            ),
        ],
        launch_offset_tuple=(2,),
        save_output_bool=False,
    ).run()

    assert len(stress_result_obj.stress_metric_df) == 1
    assert len(stress_result_obj.stress_path_df) == 6
    assert len(stress_result_obj.stress_strategy_map) == 1
    assert list(stress_result_obj.stress_strategy_map) == [
        "supported_crisis__launch_offset_2"
    ]
    assert type(next(iter(stress_result_obj.stress_strategy_map.values()))) is (
        UsSectorEtfIbsDownshockVoxIyrStrategy
    )
    assert stress_result_obj.stress_metric_df.loc[0, "crisis_name_str"] == "supported_crisis"
    assert stress_result_obj.skipped_window_list[0]["crisis_name_str"] == (
        "pre_inception_crisis"
    )
    assert "before supported history" in stress_result_obj.skipped_window_list[0][
        "reason_str"
    ]


def test_all_five_bench_analyzers_resolve_without_skip():
    module_import_str = variant_module.__name__
    strategy_entry_obj = catalog.get_strategy_by_module(module_import_str)

    assert strategy_entry_obj is not None
    assert strategy_entry_obj.has_capacity_analysis_bool is True
    assert strategy_entry_obj.has_timing_analysis_bool is True
    for analysis_str in ("vanilla", "capacity", "timing", "risk", "stress"):
        assert analysis_runner._missing_hook_detail_str(
            variant_module,
            analysis_str,
        ) is None


def test_saved_notes_disclose_proxy_semantics_without_splicing(tmp_path):
    strategy_obj = UsSectorEtfIbsDownshockVoxIyrStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[DEFAULT_CONFIG.benchmark_symbol_str],
        config_obj=DEFAULT_CONFIG,
    )

    _write_assumptions_md(tmp_path, strategy_obj)
    _write_vox_iyr_notes_md(tmp_path)

    assumptions_md_str = (
        tmp_path / "us_sector_etf_ibs_downshock_assumptions.md"
    ).read_text(encoding="utf-8")
    notes_md_str = (
        tmp_path / "us_sector_etf_ibs_downshock_vox_iyr_notes.md"
    ).read_text(encoding="utf-8")

    assert "VOX" in assumptions_md_str
    assert "IYR" in assumptions_md_str
    assert "XLC" not in assumptions_md_str
    assert "XLRE" not in assumptions_md_str
    assert "not a definitionally exact pre-2018 XLC history" in notes_md_str
    assert "No price series are spliced" in notes_md_str
