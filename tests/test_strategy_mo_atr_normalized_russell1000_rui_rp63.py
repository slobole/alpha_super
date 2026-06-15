from alpha.bench import catalog
from strategies.momentum import strategy_mo_atr_normalized_russell1000_rui_rp63 as strategy_module
from strategies.momentum.strategy_mo_atr_normalized_index_vix_scaled import (
    SELECTION_SCORE_MODE_ATR20_STR,
)


MODULE_IMPORT_STR = "strategies.momentum.strategy_mo_atr_normalized_russell1000_rui_rp63"


def test_default_config_pins_russell1000_rui_atr20_rp63_inputs():
    assert strategy_module.DEFAULT_CONFIG.indexname_str == "Russell 1000"
    assert strategy_module.DEFAULT_CONFIG.regime_symbol_str == "$RUI"
    assert strategy_module.DEFAULT_CONFIG.selection_score_mode_str == SELECTION_SCORE_MODE_ATR20_STR
    assert strategy_module.DEFAULT_CONFIG.inverse_vol_window_int == 63


def test_bench_catalog_discovers_russell1000_rui_rp63_strategy():
    strategy_entry_obj = catalog.get_strategy_by_module(MODULE_IMPORT_STR)

    assert strategy_entry_obj is not None
    assert strategy_entry_obj.has_run_variant_bool is True
    assert strategy_entry_obj.is_wired_bool is False
    assert strategy_entry_obj.category_str == "momentum"
    assert strategy_entry_obj.display_name_str == "Mo Atr Normalized Russell1000 Rui Rp63"
    assert "Russell 1000" in strategy_entry_obj.summary_str


def test_run_variant_delegates_to_base_with_rp63_enabled(monkeypatch):
    captured_kwarg_dict = {}
    sentinel_strategy_obj = object()

    def fake_run_base_variant(**run_kwarg_dict):
        captured_kwarg_dict.update(run_kwarg_dict)
        return sentinel_strategy_obj

    monkeypatch.setattr(strategy_module, "run_base_variant", fake_run_base_variant)

    strategy_obj = strategy_module.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        output_dir_str="tmp-results",
        backtest_start_date_str="2020-01-01",
        capital_base_float=123_456.0,
        end_date_str="2024-12-31",
    )

    assert strategy_obj is sentinel_strategy_obj
    assert captured_kwarg_dict["config"] == strategy_module.DEFAULT_CONFIG
    assert captured_kwarg_dict["risk_parity_63_bool"] is True
    assert captured_kwarg_dict["show_display_bool"] is False
    assert captured_kwarg_dict["save_results_bool"] is False
    assert captured_kwarg_dict["output_dir_str"] == "tmp-results"
    assert captured_kwarg_dict["backtest_start_date_str"] == "2020-01-01"
    assert captured_kwarg_dict["capital_base_float"] == 123_456.0
    assert captured_kwarg_dict["end_date_str"] == "2024-12-31"
