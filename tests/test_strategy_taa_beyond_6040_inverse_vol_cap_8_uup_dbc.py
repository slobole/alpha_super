import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.bench import catalog
from strategies.taa_beyond_6040.strategy_taa_beyond_6040 import compute_gross_exposure_float
from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_inverse_vol_cap_8_uup_dbc as variant_module


MODULE_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_beyond_6040_inverse_vol_cap_8_uup_dbc"


def make_pricing_data_df(num_days_int: int = 220) -> pd.DataFrame:
    date_index = pd.date_range("2023-01-02", periods=num_days_int, freq="B")
    bar_idx_vec = np.arange(num_days_int, dtype=float)

    return_vec_by_symbol_dict = {
        "VTI": 0.0005 + 0.0080 * np.sin(bar_idx_vec / 4.0),
        "GLD": 0.0002 + 0.0040 * np.sin(bar_idx_vec / 6.0 + 0.5),
        "TLT": 0.0001 + 0.0030 * np.cos(bar_idx_vec / 7.0),
        "UUP": 0.0001 + 0.0025 * np.sin(bar_idx_vec / 9.0 + 0.2),
        "DBC": 0.0003 + 0.0100 * np.sin(bar_idx_vec / 5.0 + 1.0),
        "$SPX": 0.0004 + 0.0070 * np.sin(bar_idx_vec / 5.0 + 0.25),
    }

    base_price_by_symbol_dict = {
        "VTI": 100.0,
        "GLD": 120.0,
        "TLT": 110.0,
        "UUP": 25.0,
        "DBC": 20.0,
        "$SPX": 4000.0,
    }

    pricing_data_dict: dict[tuple[str, str], np.ndarray] = {}
    for symbol_str, return_vec in return_vec_by_symbol_dict.items():
        close_vec = base_price_by_symbol_dict[symbol_str] * np.cumprod(1.0 + return_vec)
        open_vec = close_vec * 0.999
        high_vec = np.maximum(open_vec, close_vec) * 1.001
        low_vec = np.minimum(open_vec, close_vec) * 0.999
        pricing_data_dict[(symbol_str, "Open")] = open_vec
        pricing_data_dict[(symbol_str, "High")] = high_vec
        pricing_data_dict[(symbol_str, "Low")] = low_vec
        pricing_data_dict[(symbol_str, "Close")] = close_vec

    pricing_data_df = pd.DataFrame(pricing_data_dict, index=date_index, dtype=float)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def test_default_config_adds_uup_and_dbc_with_8pct_target_85pct_trigger():
    assert variant_module.DEFAULT_CONFIG.asset_list == ("VTI", "GLD", "TLT", "UUP", "DBC")
    assert variant_module.DEFAULT_CONFIG.target_portfolio_vol_float == 0.08
    assert variant_module.DEFAULT_CONFIG.trigger_portfolio_vol_float == 0.085


def test_exposure_formula_still_matches_requested_trigger_behavior():
    high_vol_return_ser = pd.Series([0.02, -0.02] * 32, dtype=float)
    high_trailing_return_ser = high_vol_return_ser.iloc[-variant_module.DEFAULT_CONFIG.portfolio_vol_lookback_int:]
    high_portfolio_vol_float = float(high_trailing_return_ser.std(ddof=1) * np.sqrt(252.0))

    high_exposure_float = compute_gross_exposure_float(
        realized_return_ser=high_vol_return_ser,
        portfolio_vol_lookback_int=variant_module.DEFAULT_CONFIG.portfolio_vol_lookback_int,
        target_portfolio_vol_float=variant_module.DEFAULT_CONFIG.target_portfolio_vol_float,
        trigger_portfolio_vol_float=variant_module.DEFAULT_CONFIG.trigger_portfolio_vol_float,
    )

    assert np.isclose(high_exposure_float, 0.08 / high_portfolio_vol_float, atol=1e-12)
    assert high_exposure_float < 1.0

    mid_vol_return_ser = pd.Series([0.0052, -0.0052] * 32, dtype=float)
    mid_trailing_return_ser = mid_vol_return_ser.iloc[-variant_module.DEFAULT_CONFIG.portfolio_vol_lookback_int:]
    mid_portfolio_vol_float = float(mid_trailing_return_ser.std(ddof=1) * np.sqrt(252.0))
    assert 0.08 < mid_portfolio_vol_float < 0.085

    mid_exposure_float = compute_gross_exposure_float(
        realized_return_ser=mid_vol_return_ser,
        portfolio_vol_lookback_int=variant_module.DEFAULT_CONFIG.portfolio_vol_lookback_int,
        target_portfolio_vol_float=variant_module.DEFAULT_CONFIG.target_portfolio_vol_float,
        trigger_portfolio_vol_float=variant_module.DEFAULT_CONFIG.trigger_portfolio_vol_float,
    )

    assert mid_exposure_float == 1.0


def test_run_variant_honors_bench_run_contract_with_added_assets():
    captured_config_list = []
    pricing_data_df = make_pricing_data_df()

    def loader_fn(config):
        captured_config_list.append(config)
        return pricing_data_df

    with patch.object(variant_module, "get_beyond_6040_data", side_effect=loader_fn):
        strategy_obj = variant_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str="2023-05-01",
            capital_base_float=12_345.0,
            end_date_str="2023-09-01",
        )

    strategy_entry_obj = catalog.get_strategy_by_module(MODULE_IMPORT_STR)
    assert strategy_entry_obj is not None
    assert strategy_entry_obj.has_run_variant_bool is True
    assert captured_config_list[0].asset_list == ("VTI", "GLD", "TLT", "UUP", "DBC")
    assert captured_config_list[0].end_date_str == "2023-09-01"
    assert strategy_obj.name == variant_module.STRATEGY_NAME_STR
    assert strategy_obj.asset_list == ["VTI", "GLD", "TLT", "UUP", "DBC"]
    assert strategy_obj._capital_base == 12_345.0
    assert {"VTI", "GLD", "TLT", "UUP", "DBC", "Cash"}.issubset(strategy_obj.daily_target_weights.columns)
    assert len(strategy_obj.results) > 0
    assert strategy_obj.results.index.min() >= pd.Timestamp("2023-05-01")
