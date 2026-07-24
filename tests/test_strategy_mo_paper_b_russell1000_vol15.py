import numpy as np
import pandas as pd
import pytest

from alpha.bench import catalog
from strategies.momentum import strategy_mo_paper_b_russell1000_vol15 as variant


def _runner_fixture_tuple():
    decision_date_idx = pd.date_range("2022-12-30", periods=13, freq="BME")
    execution_date_idx = pd.DatetimeIndex(
        [decision_date_ts + pd.offsets.BDay(1) for decision_date_ts in decision_date_idx]
    )
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": decision_date_idx},
        index=execution_date_idx,
    )
    rebalance_schedule_df.index.name = "execution_date_ts"

    selection_df = pd.DataFrame(
        {
            "execution_date_ts": execution_date_idx,
            "decision_date_ts": decision_date_idx,
            "symbol_str": ["A"] * len(execution_date_idx),
            "base_target_weight_float": [0.02] * len(execution_date_idx),
        }
    )
    pricing_data_df = pd.DataFrame(index=execution_date_idx)
    universe_df = pd.DataFrame({"A": 1}, index=decision_date_idx)
    return pricing_data_df, universe_df, rebalance_schedule_df, selection_df


def test_default_config_is_exact_russell1000_vol15_variant():
    config_obj = variant.DEFAULT_CONFIG

    assert config_obj.variant_key_str == "paper_b_russell1000_top50_bottom50_vol15"
    assert config_obj.indexname_str == "Russell 1000"
    assert config_obj.benchmark_list == ("$RUI",)
    assert config_obj.max_long_positions_int == 50
    assert config_obj.max_short_positions_int == 50
    assert config_obj.target_annualized_volatility_float == 0.15
    assert config_obj.volatility_lookback_month_int == 12
    assert config_obj.maximum_exposure_multiplier_float == 1.0
    assert variant.AUDIT_ENABLED_BOOL is False
    assert variant.PaperBRussell1000Vol15Strategy.enable_signal_audit is False


def test_vol15_exposure_uses_12_completed_unscaled_base_returns():
    base_monthly_return_ser = pd.Series(
        [0.10, -0.10] * 6,
        index=pd.period_range("2023-01", periods=12, freq="M"),
        dtype=float,
    )
    decision_date_idx = pd.date_range("2023-01-31", periods=12, freq="BME")
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": decision_date_idx},
        index=pd.DatetimeIndex(
            [decision_date_ts + pd.offsets.BDay(1) for decision_date_ts in decision_date_idx]
        ),
    )

    exposure_schedule_df = variant.paper_b_base.build_exposure_schedule_df(
        base_monthly_return_ser=base_monthly_return_ser,
        rebalance_schedule_df=rebalance_schedule_df,
        config=variant.DEFAULT_CONFIG,
    )

    expected_volatility_float = float(base_monthly_return_ser.std(ddof=1) * np.sqrt(12.0))
    expected_exposure_float = min(1.0, 0.15 / expected_volatility_float)
    assert (exposure_schedule_df.iloc[:11]["exposure_multiplier_float"] == 0.0).all()
    assert exposure_schedule_df.iloc[-1]["completed_base_return_count_int"] == 12
    assert exposure_schedule_df.iloc[-1]["exposure_multiplier_float"] == pytest.approx(
        expected_exposure_float
    )


def test_run_variant_hard_disables_audit_in_both_vanilla_passes(monkeypatch):
    (
        pricing_data_df,
        universe_df,
        rebalance_schedule_df,
        selection_df,
    ) = _runner_fixture_tuple()
    observed_config_list = []
    audit_override_list = []

    def fake_get_paper_b_russell1000_data(**kwargs):
        observed_config_list.append(kwargs["config"])
        return pricing_data_df, universe_df, rebalance_schedule_df, selection_df

    def fake_run_daily(strategy_obj, _pricing_data_df, **kwargs):
        audit_override_list.append(kwargs["audit_override_bool"])
        if strategy_obj.name.endswith("_unscaled_base"):
            strategy_obj.results = pd.DataFrame(
                {"daily_returns": [0.01, -0.01] * 6},
                index=pd.date_range("2023-01-31", periods=12, freq="BME"),
            )
        return strategy_obj

    monkeypatch.setattr(
        variant,
        "get_paper_b_russell1000_data",
        fake_get_paper_b_russell1000_data,
    )
    monkeypatch.setattr(variant, "run_daily", fake_run_daily)

    strategy_obj = variant.run_variant(
        show_display_bool=False,
        save_results_bool=False,
    )

    assert observed_config_list == [variant.DEFAULT_CONFIG]
    assert audit_override_list == [False, False]
    assert strategy_obj.config.indexname_str == "Russell 1000"
    assert strategy_obj.config.target_annualized_volatility_float == 0.15
    assert strategy_obj.reported_start_date_ts == rebalance_schedule_df.index[-1]


def test_bench_discovers_russell1000_variant_as_research_only_cross_sectional():
    entry_obj = next(
        entry_obj
        for entry_obj in catalog.list_strategies()
        if entry_obj.stem_str == "strategy_mo_paper_b_russell1000_vol15"
    )

    assert entry_obj.has_run_variant_bool is True
    assert entry_obj.is_wired_bool is False
    assert entry_obj.category_str == "momentum"
    assert entry_obj.subcategory_str == "cross_sectional"
