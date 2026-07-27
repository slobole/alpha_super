import pandas as pd
import pytest

from strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock import (
    DEFAULT_CONFIG as BASE_DEFAULT_CONFIG,
    _write_assumptions_md,
)
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
