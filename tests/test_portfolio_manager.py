import json
import textwrap
from pathlib import Path

import pytest
import pandas as pd
import yaml

import alpha.engine.portfolio_manager as portfolio_manager
from alpha.engine.report import _build_portfolio_html


LINEARITY_QQQ_VIX_CASH_IMPORT_STR = (
    "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash"
)
BTAL_1N_TQQQ_VIX_CASH_IMPORT_STR = (
    "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash"
)
VXN_SCALED_NDX_IMPORT_STR = (
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy"
)
SECTOR_DISPERSION_KIE_IHI_XLC_IMPORT_STR = (
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc"
)
SECTOR_DISPERSION_KIE_IHI_XLC_ASSET_SMA200_IMPORT_STR = (
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200"
)
SECTOR_DISPERSION_KIE_IHI_ASSET_SMA200_IMPORT_STR = (
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200"
)


def test_supported_strategy_imports_include_vxn_scaled_ndx_momentum():
    assert VXN_SCALED_NDX_IMPORT_STR in portfolio_manager.SUPPORTED_STRATEGY_IMPORT_TUPLE


def test_supported_strategy_imports_include_sector_dispersion_mean_reversion_variants():
    assert SECTOR_DISPERSION_KIE_IHI_XLC_IMPORT_STR in portfolio_manager.SUPPORTED_STRATEGY_IMPORT_TUPLE
    assert (
        SECTOR_DISPERSION_KIE_IHI_XLC_ASSET_SMA200_IMPORT_STR
        in portfolio_manager.SUPPORTED_STRATEGY_IMPORT_TUPLE
    )
    assert (
        SECTOR_DISPERSION_KIE_IHI_ASSET_SMA200_IMPORT_STR
        in portfolio_manager.SUPPORTED_STRATEGY_IMPORT_TUPLE
    )


def write_dummy_strategy_module(
    tmp_path: Path,
    module_name_str: str,
    strategy_name_str: str,
    return_float: float,
    sleep_seconds_float: float = 0.0,
    raise_message_str: str | None = None,
    start_offset_business_day_int: int = 0,
) -> str:
    module_path = tmp_path / f"{module_name_str}.py"
    raise_line_str = (
        f"                raise RuntimeError({raise_message_str!r})"
        if raise_message_str is not None
        else ""
    )
    module_path.write_text(
        textwrap.dedent(
            f"""
            import time

            import pandas as pd

            from alpha.engine.strategy import Strategy


            class DummyPMStrategy(Strategy):
                def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
                    return pricing_data

                def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
                    return None


            def run_variant(
                show_display_bool: bool = True,
                save_results_bool: bool = True,
                output_dir_str: str = "results",
                backtest_start_date_str: str = "2004-01-01",
                capital_base_float: float = 100_000.0,
                end_date_str: str | None = None,
            ):
{raise_line_str}
                time.sleep({sleep_seconds_float!r})
                start_ts = (
                    pd.Timestamp(backtest_start_date_str)
                    + pd.offsets.BDay({start_offset_business_day_int!r})
                )
                date_index = pd.bdate_range(start=start_ts, periods=3)
                daily_return_ser = pd.Series(
                    [0.0, {return_float!r}, 0.0],
                    index=date_index,
                    dtype=float,
                )
                total_value_ser = float(capital_base_float) * (1.0 + daily_return_ser).cumprod()

                strategy = DummyPMStrategy(
                    name={strategy_name_str!r},
                    benchmarks=[],
                    capital_base=float(capital_base_float),
                    slippage=0.0,
                    commission_per_share=0.0,
                    commission_minimum=0.0,
                )
                strategy.results = pd.DataFrame(
                    {{
                        "daily_returns": daily_return_ser,
                        "total_value": total_value_ser,
                        "portfolio_value": total_value_ser,
                    }},
                    index=date_index,
                )
                strategy.summary = pd.DataFrame()
                strategy.summary_trades = pd.DataFrame()
                return strategy
            """
        ),
        encoding="utf-8",
    )
    return f"{module_name_str}:DummyPMStrategy"


def make_fixed_config_dict(strategy_import_list: list[str]) -> dict:
    weight_float = 1.0 / float(len(strategy_import_list))
    return {
        "name_str": "FreshBook",
        "capital_base_float": 1000.0,
        "backtest_start_date_str": "2024-01-02",
        "end_date_str": None,
        "allocation_policy_str": "fixed",
        "max_workers_int": 1,
        "rebalance": None,
        "save_pod_artifacts_bool": True,
        "pods": [
            {
                "pod_id_str": f"pod_{idx_int}",
                "strategy_import_str": strategy_import_str,
                "weight_float": weight_float,
            }
            for idx_int, strategy_import_str in enumerate(strategy_import_list, start=1)
        ],
    }


def test_fixed_config_allocates_capital_and_rejects_pkl(monkeypatch, tmp_path):
    strategy_import_str = write_dummy_strategy_module(
        tmp_path,
        "dummy_pm_fixed",
        "StrategyFixed",
        0.10,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (strategy_import_str,),
    )

    config_dict = make_fixed_config_dict([strategy_import_str])
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)

    assert config_obj.pod_config_list[0].weight_float == pytest.approx(1.0)
    assert config_obj.pod_config_list[0].allocated_capital_float == pytest.approx(1000.0)

    config_dict["pods"][0]["pkl"] = "old_result.pkl"
    with pytest.raises(ValueError, match="unsupported field"):
        portfolio_manager.build_portfolio_manager_config(config_dict)


def test_equal_policy_assigns_one_over_n_and_rejects_explicit_weight(monkeypatch, tmp_path):
    left_import_str = write_dummy_strategy_module(tmp_path, "dummy_pm_equal_a", "StrategyA", 0.01)
    right_import_str = write_dummy_strategy_module(tmp_path, "dummy_pm_equal_b", "StrategyB", 0.02)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (left_import_str, right_import_str),
    )

    config_dict = {
        "name_str": "EqualBook",
        "capital_base_float": 900.0,
        "backtest_start_date_str": "2024-01-02",
        "allocation_policy_str": "equal",
        "pods": [
            {"pod_id_str": "pod_a", "strategy_import_str": left_import_str},
            {"pod_id_str": "pod_b", "strategy_import_str": right_import_str},
        ],
    }
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)

    assert [pod.weight_float for pod in config_obj.pod_config_list] == pytest.approx([0.5, 0.5])
    assert [pod.allocated_capital_float for pod in config_obj.pod_config_list] == pytest.approx([450.0, 450.0])

    config_dict["pods"][0]["weight_float"] = 0.5
    with pytest.raises(ValueError, match="does not allow pod weight_float"):
        portfolio_manager.build_portfolio_manager_config(config_dict)


def test_rebalance_equal_policy_config_is_accepted(monkeypatch):
    monkeypatch.setattr(portfolio_manager, "SUPPORTED_STRATEGY_IMPORT_TUPLE", ("dummy:a", "dummy:b"))
    config_dict = make_fixed_config_dict(["dummy:a", "dummy:b"])
    config_dict["rebalance"] = {
        "frequency_str": "monthly",
        "policy_str": "equal",
    }

    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)

    assert config_obj.rebalance is not None
    assert config_obj.rebalance.frequency_str == "monthly"
    assert config_obj.rebalance.policy_str == "equal"
    assert config_obj.rebalance.lookback_day_int is None


def test_rebalance_risk_parity_alias_normalizes_to_inverse_volatility(monkeypatch):
    monkeypatch.setattr(portfolio_manager, "SUPPORTED_STRATEGY_IMPORT_TUPLE", ("dummy:a", "dummy:b"))
    config_dict = make_fixed_config_dict(["dummy:a", "dummy:b"])
    config_dict["rebalance"] = {
        "frequency_str": "quarterly",
        "policy_str": "risk_parity",
        "lookback_day_int": 20,
    }

    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)

    assert config_obj.rebalance is not None
    assert config_obj.rebalance.frequency_str == "quarterly"
    assert config_obj.rebalance.policy_str == "inverse_volatility"
    assert config_obj.rebalance.lookback_day_int == 20


def test_allocated_capital_floor_rejects_underfunded_stock_pod(monkeypatch):
    strategy_import_str = "dummy_stock:a"
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (strategy_import_str,),
    )
    monkeypatch.setattr(
        portfolio_manager,
        "POD_MINIMUM_ALLOCATED_CAPITAL_FLOAT_DICT",
        {strategy_import_str: 25_000.0},
    )

    config_dict = make_fixed_config_dict([strategy_import_str])
    config_dict["capital_base_float"] = 10_000.0

    with pytest.raises(ValueError, match="below the practical minimum"):
        portfolio_manager.build_portfolio_manager_config(config_dict)


def test_allocated_capital_floor_allows_funded_stock_pod(monkeypatch):
    strategy_import_str = "dummy_stock:a"
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (strategy_import_str,),
    )
    monkeypatch.setattr(
        portfolio_manager,
        "POD_MINIMUM_ALLOCATED_CAPITAL_FLOAT_DICT",
        {strategy_import_str: 25_000.0},
    )

    config_dict = make_fixed_config_dict([strategy_import_str])
    config_dict["capital_base_float"] = 25_000.0
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)

    assert config_obj.pod_config_list[0].allocated_capital_float == pytest.approx(25_000.0)


@pytest.mark.parametrize(
    "config_update_dict, expected_message_str",
    [
        ({"rebalance": {"frequency_str": "weekly", "policy_str": "equal"}}, "rebalance.frequency_str"),
        ({"pods": [{"pod_id_str": "pod_a", "strategy_import_str": "unsupported", "weight_float": 1.0}]}, "Unsupported strategy_import_str"),
        (
            {
                "pods": [
                    {"pod_id_str": "pod_a", "strategy_import_str": "dummy:a", "weight_float": 0.5},
                    {"pod_id_str": "pod_a", "strategy_import_str": "dummy:a", "weight_float": 0.5},
                ]
            },
            "Duplicate pod_id_str",
        ),
        (
            {"pods": [{"pod_id_str": "pod_a", "strategy_import_str": "dummy:a", "weight_float": 1.0, "params": {}}]},
            "Per-pod params",
        ),
    ],
)
def test_config_validation_failures(monkeypatch, config_update_dict, expected_message_str):
    monkeypatch.setattr(portfolio_manager, "SUPPORTED_STRATEGY_IMPORT_TUPLE", ("dummy:a",))
    config_dict = make_fixed_config_dict(["dummy:a"])
    config_dict.update(config_update_dict)

    with pytest.raises(ValueError, match=expected_message_str):
        portfolio_manager.build_portfolio_manager_config(config_dict)


def test_current_aggressive_config_accepts_promoted_taa_pod():
    config_obj = portfolio_manager.load_portfolio_manager_config(
        Path("portfolios/current_multipod_all_aggresive.yaml")
    )

    strategy_import_by_pod_id_dict = {
        pod_config.pod_id_str: pod_config.strategy_import_str
        for pod_config in config_obj.pod_config_list
    }

    assert strategy_import_by_pod_id_dict["pod_taa"] in {
        LINEARITY_QQQ_VIX_CASH_IMPORT_STR,
        BTAL_1N_TQQQ_VIX_CASH_IMPORT_STR,
    }
    assert config_obj.regression_benchmark_symbol_str == '$SPX'


def test_config_accepts_optional_regression_benchmark_symbol():
    config_dict = make_fixed_config_dict([VXN_SCALED_NDX_IMPORT_STR])
    config_dict['regression_benchmark_symbol_str'] = '$NDX'

    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)

    assert config_obj.regression_benchmark_symbol_str == '$NDX'


def test_config_defaults_pm_performance_benchmark_to_spx():
    config_obj = portfolio_manager.build_portfolio_manager_config(
        make_fixed_config_dict([VXN_SCALED_NDX_IMPORT_STR])
    )

    assert config_obj.regression_benchmark_symbol_str == '$SPX'


@pytest.mark.parametrize(
    'config_path_str',
    [
        'portfolios/current_book_fresh.yaml',
        'portfolios/current_multipod_all.yaml',
        'portfolios/current_multipod_all_aggresive.yaml',
        'portfolios/current_multipod_all_aggressive_yearly_inverse_volatility.yaml',
        'portfolios/current_multipod_all_aggressive_yearly_rebalanced.yaml',
        'portfolios/current_multipod_monthly.yaml',
        'portfolios/current_multipod_yearly_inverse_volatility.yaml',
        'portfolios/current_multipod_yearly_rebalanced.yaml',
        'portfolios/multipod_low_risk.yaml',
        'portfolios/multipod_monthly.yaml',
        'portfolios/current_book.yaml',
        'portfolios/multipod.yaml',
    ],
)
def test_tracked_current_pm_configs_do_not_require_benchmark_declaration(
    config_path_str: str,
):
    config_dict = yaml.safe_load(Path(config_path_str).read_text(encoding='utf-8'))

    assert 'regression_benchmark_symbol_str' not in config_dict


def test_manager_loads_total_return_pm_regression_benchmark(monkeypatch, tmp_path):
    strategy_import_str = write_dummy_strategy_module(
        tmp_path,
        'dummy_pm_regression_benchmark',
        'StrategyRegression',
        0.01,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        'SUPPORTED_STRATEGY_IMPORT_TUPLE',
        (strategy_import_str,),
    )
    config_dict = make_fixed_config_dict([strategy_import_str])
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)
    load_call_dict: dict[str, object] = {'call_count_int': 0}

    def fake_load_price_timeseries(symbol_str: str, **kwargs) -> pd.DataFrame:
        load_call_dict['call_count_int'] = int(load_call_dict['call_count_int']) + 1
        load_call_dict['symbol_str'] = symbol_str
        load_call_dict.update(kwargs)
        date_index = pd.bdate_range('2024-01-02', periods=3)
        return pd.DataFrame({'Close': [100.0, 101.0, 102.0]}, index=date_index)

    monkeypatch.setattr(portfolio_manager, 'load_price_timeseries', fake_load_price_timeseries)

    result_obj = portfolio_manager.PortfolioManager(config_obj).run(
        save_results_bool=False,
        show_display_bool=False,
    )

    assert load_call_dict['symbol_str'] == '$SPX'
    assert load_call_dict['call_count_int'] == 1
    assert load_call_dict['adjustment_str'] == portfolio_manager.TOTALRETURN_ADJUSTMENT_STR
    assert result_obj.portfolio.regression_benchmark_label_str == '$SPX · TOTALRETURN'
    portfolio_metadata_dict = result_obj.portfolio.benchmark_regression_metadata_by_column_dict[
        result_obj.portfolio.name
    ]
    assert portfolio_metadata_dict['reason_str'] == 'insufficient_observations'


def test_manager_continues_with_explicit_na_when_pm_benchmark_load_fails(
    monkeypatch,
    tmp_path,
):
    strategy_import_str = write_dummy_strategy_module(
        tmp_path,
        'dummy_pm_missing_regression_benchmark',
        'StrategyMissingRegression',
        0.01,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        'SUPPORTED_STRATEGY_IMPORT_TUPLE',
        (strategy_import_str,),
    )
    config_obj = portfolio_manager.build_portfolio_manager_config(
        make_fixed_config_dict([strategy_import_str])
    )

    def failing_load_price_timeseries(*args, **kwargs):
        raise RuntimeError('benchmark unavailable')

    monkeypatch.setattr(
        portfolio_manager,
        'load_price_timeseries',
        failing_load_price_timeseries,
    )

    with pytest.warns(RuntimeWarning, match='Portfolio benchmark regression is unavailable'):
        result_obj = portfolio_manager.PortfolioManager(config_obj).run(
            save_results_bool=False,
            show_display_bool=False,
        )

    assert result_obj.portfolio.regression_benchmark_label_str == '$SPX · TOTALRETURN'
    assert result_obj.portfolio.benchmark_monthly_returns is None
    report_html_str = _build_portfolio_html(
        result_obj.portfolio,
        chart_b64='portfolio-chart-b64',
    )
    assert 'Benchmark Portfolio Monthly Returns — $SPX · TOTALRETURN' in report_html_str
    assert 'N/A — PM performance benchmark data is unavailable' in report_html_str


def test_monthly_multipod_config_accepts_sector_dispersion_raw_pod():
    config_obj = portfolio_manager.load_portfolio_manager_config(
        Path("portfolios/multipod_monthly.yaml")
    )
    strategy_import_by_pod_id_dict = {
        pod_config.pod_id_str: pod_config.strategy_import_str
        for pod_config in config_obj.pod_config_list
    }

    assert (
        strategy_import_by_pod_id_dict["pod_sector_dispersion_kie_ihi_xlc"]
        == SECTOR_DISPERSION_KIE_IHI_XLC_IMPORT_STR
    )


def test_low_risk_no_xlc_config_uses_full_history_sma200_sector_pod():
    config_obj = portfolio_manager.load_portfolio_manager_config(
        Path("portfolios/multipod_low_risk_noXLC.yaml")
    )
    strategy_import_by_pod_id_dict = {
        pod_config.pod_id_str: pod_config.strategy_import_str
        for pod_config in config_obj.pod_config_list
    }

    assert config_obj.name_str == "multipod_low_risk_no_xlc"
    assert config_obj.backtest_start_date_str == "2012-10-01"
    assert (
        strategy_import_by_pod_id_dict[
            "pod_mr_sector_dispersion_ibs_kie_ihi_asset_sma200"
        ]
        == SECTOR_DISPERSION_KIE_IHI_ASSET_SMA200_IMPORT_STR
    )


def test_linearity_qqq_vix_cash_taa_exposes_manager_run_contract():
    strategy_module = __import__(
        LINEARITY_QQQ_VIX_CASH_IMPORT_STR,
        fromlist=["run_variant"],
    )

    portfolio_manager._validate_run_variant_signature(
        strategy_module.run_variant,
        LINEARITY_QQQ_VIX_CASH_IMPORT_STR,
    )


def test_btal_1n_tqqq_vix_cash_taa_validates_as_supported_manager_pod():
    config_dict = make_fixed_config_dict([BTAL_1N_TQQQ_VIX_CASH_IMPORT_STR])
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)

    assert config_obj.pod_config_list[0].strategy_import_str == BTAL_1N_TQQQ_VIX_CASH_IMPORT_STR


def test_btal_1n_tqqq_vix_cash_taa_exposes_manager_run_contract():
    strategy_module = __import__(
        BTAL_1N_TQQQ_VIX_CASH_IMPORT_STR,
        fromlist=["run_variant"],
    )

    portfolio_manager._validate_run_variant_signature(
        strategy_module.run_variant,
        BTAL_1N_TQQQ_VIX_CASH_IMPORT_STR,
    )


@pytest.mark.parametrize(
    "strategy_import_str",
    [
        SECTOR_DISPERSION_KIE_IHI_XLC_IMPORT_STR,
        SECTOR_DISPERSION_KIE_IHI_XLC_ASSET_SMA200_IMPORT_STR,
        SECTOR_DISPERSION_KIE_IHI_ASSET_SMA200_IMPORT_STR,
    ],
)
def test_sector_dispersion_variants_validate_as_supported_manager_pods(strategy_import_str):
    config_dict = make_fixed_config_dict([strategy_import_str])
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)

    assert config_obj.pod_config_list[0].strategy_import_str == strategy_import_str


@pytest.mark.parametrize(
    "strategy_import_str",
    [
        SECTOR_DISPERSION_KIE_IHI_XLC_IMPORT_STR,
        SECTOR_DISPERSION_KIE_IHI_XLC_ASSET_SMA200_IMPORT_STR,
        SECTOR_DISPERSION_KIE_IHI_ASSET_SMA200_IMPORT_STR,
    ],
)
def test_sector_dispersion_variants_expose_manager_run_contract(strategy_import_str):
    strategy_module = __import__(
        strategy_import_str,
        fromlist=["run_variant"],
    )

    portfolio_manager._validate_run_variant_signature(
        strategy_module.run_variant,
        strategy_import_str,
    )


def test_serial_run_uses_allocated_capital_and_builds_portfolio(monkeypatch, tmp_path):
    left_import_str = write_dummy_strategy_module(tmp_path, "dummy_pm_serial_a", "StrategyA", 0.10)
    right_import_str = write_dummy_strategy_module(tmp_path, "dummy_pm_serial_b", "StrategyB", -0.05)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (left_import_str, right_import_str),
    )

    config_dict = make_fixed_config_dict([left_import_str, right_import_str])
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)
    manager_obj = portfolio_manager.PortfolioManager(config_obj, source_config_path_str="fresh.yaml")

    result_obj = manager_obj.run(save_results_bool=False, show_display_bool=False)

    assert [strategy.name for strategy in result_obj.portfolio.strategies] == ["StrategyA", "StrategyB"]
    assert [strategy._capital_base for strategy in result_obj.portfolio.strategies] == pytest.approx([500.0, 500.0])
    assert float(result_obj.portfolio.results.iloc[0]["total_value"]) == pytest.approx(1000.0)
    assert result_obj.portfolio.results.iloc[-1]["total_value"] == pytest.approx(1025.0)
    assert result_obj.portfolio.pod_info_list[0]["source_type_str"] == "fresh_run"
    assert "source_pkl" not in result_obj.portfolio.pod_info_list[0]


def test_serial_run_passes_rebalance_policy_to_portfolio(monkeypatch, tmp_path):
    left_import_str = write_dummy_strategy_module(tmp_path, "dummy_pm_rebalance_a", "StrategyA", 0.10)
    right_import_str = write_dummy_strategy_module(tmp_path, "dummy_pm_rebalance_b", "StrategyB", -0.05)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (left_import_str, right_import_str),
    )

    config_dict = make_fixed_config_dict([left_import_str, right_import_str])
    config_dict["rebalance"] = {
        "frequency_str": "monthly",
        "policy_str": "equal",
    }
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)
    manager_obj = portfolio_manager.PortfolioManager(config_obj)

    result_obj = manager_obj.run(save_results_bool=False, show_display_bool=False)

    assert result_obj.portfolio._rebalance == "monthly"
    assert result_obj.portfolio._rebalance_policy == "equal"


def test_parallel_run_preserves_config_order(monkeypatch, tmp_path):
    slow_import_str = write_dummy_strategy_module(
        tmp_path,
        "dummy_pm_parallel_slow",
        "StrategySlow",
        0.01,
        sleep_seconds_float=0.20,
    )
    fast_import_str = write_dummy_strategy_module(
        tmp_path,
        "dummy_pm_parallel_fast",
        "StrategyFast",
        0.02,
        sleep_seconds_float=0.0,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (slow_import_str, fast_import_str),
    )

    config_dict = make_fixed_config_dict([slow_import_str, fast_import_str])
    config_dict["max_workers_int"] = 2
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)
    manager_obj = portfolio_manager.PortfolioManager(config_obj)

    result_obj = manager_obj.run(save_results_bool=False, show_display_bool=False)

    assert [strategy.name for strategy in result_obj.portfolio.strategies] == ["StrategySlow", "StrategyFast"]


def test_run_max_workers_override_can_force_serial(monkeypatch, tmp_path):
    left_import_str = write_dummy_strategy_module(tmp_path, "dummy_pm_override_a", "StrategyA", 0.01)
    right_import_str = write_dummy_strategy_module(tmp_path, "dummy_pm_override_b", "StrategyB", 0.02)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (left_import_str, right_import_str),
    )

    config_dict = make_fixed_config_dict([left_import_str, right_import_str])
    config_dict["max_workers_int"] = 2
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)
    manager_obj = portfolio_manager.PortfolioManager(config_obj)

    result_obj = manager_obj.run(
        save_results_bool=False,
        show_display_bool=False,
        max_workers_int=1,
    )

    assert [strategy.name for strategy in result_obj.portfolio.strategies] == ["StrategyA", "StrategyB"]


def test_failed_pod_error_includes_pod_context(monkeypatch, tmp_path):
    strategy_import_str = write_dummy_strategy_module(
        tmp_path,
        "dummy_pm_failure",
        "StrategyFailure",
        0.01,
        raise_message_str="boom",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (strategy_import_str,),
    )

    config_dict = make_fixed_config_dict([strategy_import_str])
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)
    manager_obj = portfolio_manager.PortfolioManager(config_obj)

    with pytest.raises(RuntimeError, match="pod_id_str='pod_1'.*strategy_import_str"):
        manager_obj.run(save_results_bool=False, show_display_bool=False, max_workers_int=1)


def test_artifact_metadata_records_pod_and_portfolio_outputs(monkeypatch, tmp_path):
    strategy_import_str = write_dummy_strategy_module(
        tmp_path,
        "dummy_pm_artifacts",
        "StrategyArtifacts",
        0.01,
        start_offset_business_day_int=5,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        portfolio_manager,
        "SUPPORTED_STRATEGY_IMPORT_TUPLE",
        (strategy_import_str,),
    )

    def fake_save_strategy_results(strategy_obj, output_dir: str = "results", output_path=None):
        artifact_dir_path = Path(output_path if output_path is not None else output_dir) / strategy_obj.name / "pod_artifact"
        artifact_dir_path.mkdir(parents=True, exist_ok=True)
        return artifact_dir_path

    def fake_save_portfolio_results(portfolio_obj, output_dir: str = "results", output_path=None):
        artifact_dir_path = Path(output_path if output_path is not None else output_dir) / portfolio_obj.name / "portfolio_artifact"
        artifact_dir_path.mkdir(parents=True, exist_ok=True)
        return artifact_dir_path

    monkeypatch.setattr(portfolio_manager, "save_strategy_results", fake_save_strategy_results)
    monkeypatch.setattr(portfolio_manager, "save_portfolio_results", fake_save_portfolio_results)

    config_dict = make_fixed_config_dict([strategy_import_str])
    config_obj = portfolio_manager.build_portfolio_manager_config(config_dict)
    manager_obj = portfolio_manager.PortfolioManager(
        config_obj,
        source_config_path_str=str(tmp_path / "fresh.yaml"),
    )

    result_obj = manager_obj.run(
        output_dir_str=str(tmp_path / "results"),
        save_results_bool=True,
        show_display_bool=False,
    )

    assert result_obj.manager_metadata_path is not None
    assert result_obj.manager_metadata_path.name == "manager_metadata.json"
    relative_manager_path = result_obj.manager_run_dir_path.relative_to(tmp_path / "results")
    assert relative_manager_path.parts[:4] == (
        "research",
        "portfolio",
        "FreshBook",
        "vanilla_backtest",
    )
    metadata_dict = json.loads(result_obj.manager_metadata_path.read_text(encoding="utf-8"))
    assert metadata_dict["artifact_type"] == "portfolio_manager"
    assert metadata_dict["validation_status_str"] == "passed"
    assert metadata_dict["requested_backtest_start_date_str"] == "2024-01-02"
    assert metadata_dict["effective_portfolio_start_date_str"] == "2024-01-09"
    assert metadata_dict["portfolio_output_dir_path"].endswith("portfolio_artifact")
    assert metadata_dict["pods"][0]["pod_artifact_dir_path"].endswith("pod_artifact")
    assert metadata_dict["pods"][0]["requested_backtest_start_date_str"] == "2024-01-02"
    assert metadata_dict["pods"][0]["effective_backtest_start_date_str"] == "2024-01-09"
    assert result_obj.portfolio.pod_info_list[0]["pod_artifact_dir"].endswith("pod_artifact")
    assert result_obj.portfolio.pod_info_list[0]["backtest_start_date_str"] == "2024-01-09"
    assert result_obj.portfolio.pod_info_list[0]["requested_backtest_start_date_str"] == "2024-01-02"
