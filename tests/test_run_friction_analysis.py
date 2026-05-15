import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import strategies.run_friction_analysis as runner_module


def test_run_friction_analysis_module_calls_strategy_hook(tmp_path, capsys, monkeypatch):
    module_name_str = "test_fake_friction_strategy"
    fake_module = ModuleType(module_name_str)
    captured_kwarg_dict = {}

    def run_friction_analysis(
        save_results_bool=True,
        output_dir_str="results",
        show_display_bool=False,
        backtest_start_date_str="2004-01-01",
        capital_base_float=100_000.0,
        end_date_str=None,
    ):
        captured_kwarg_dict.update(
            {
                "save_results_bool": save_results_bool,
                "output_dir_str": output_dir_str,
                "show_display_bool": show_display_bool,
                "backtest_start_date_str": backtest_start_date_str,
                "capital_base_float": capital_base_float,
                "end_date_str": end_date_str,
            }
        )
        return SimpleNamespace(
            output_dir_path=tmp_path / "research" / "strategy" / "fake" / "friction_analysis",
            summary_dict={
                "strategy_name_str": "fake",
                "friction_verdict_str": "Watch",
                "cost_verdict_str": "Conservative",
                "auction_verdict_str": "Stressed",
                "verdict_explanation_str": (
                    "Cost model is conservative by 1.50 bps versus the ADV-rule "
                    "estimate, but auction stress needs inspection; combined verdict is Watch."
                ),
                "assessed_order_count_int": 3,
                "assessed_notional_float": 1250.0,
                "p95_participation_float": 0.031,
                "max_participation_float": 0.08,
                "red_order_count_int": 1,
                "red_order_share_float": 1 / 3,
                "capacity_stressed_order_count_int": 2,
                "capacity_stressed_order_share_float": 2 / 3,
                "red_notional_share_float": 0.2,
                "capacity_pass_order_rate_float": 0.667,
                "adv_p95_participation_float": 0.00062,
                "adv_max_participation_float": 0.0016,
                "adv_red_order_count_int": 0,
                "adv_rule_slippage_dollar_float": 1.25,
                "adv_rule_slippage_blended_bps_float": 1.0,
                "default_slippage_dollar_float": 3.125,
                "default_slippage_blended_bps_float": 2.5,
                "adv_rule_minus_default_dollar_float": -1.875,
                "adv_rule_minus_default_bps_float": -1.5,
                "auction_impact_dollar_float": 10.0,
                "auction_impact_blended_bps_float": 8.0,
                "auction_impact_minus_default_dollar_float": 6.875,
                "auction_impact_minus_default_bps_float": 5.5,
                "auction_impact_p95_bps_float": 12.0,
                "auction_impact_max_bps_float": 20.0,
                "auction_impact_interpretation_str": (
                    "This suggests current 2.5 bps is materially optimistic "
                    "versus the auction-impact proxy."
                ),
                "current_annual_return_float": 0.1425,
                "auction_impact_estimated_annual_return_float": 0.1310,
                "auction_impact_estimated_annual_return_delta_float": -0.0115,
                "current_sharpe_float": 1.23,
                "auction_impact_estimated_sharpe_float": 1.16,
                "auction_impact_estimated_sharpe_delta_float": -0.07,
                "top_red_asset_str": "AAA",
                "top_red_asset_red_notional_share_float": 0.80,
                "scale_to_p95_auction_red_threshold_float": 0.50,
                "scale_to_max_auction_proxy_float": 12.50,
                "scale_to_p95_adv_5pct_float": 80.65,
            },
        )

    fake_module.run_friction_analysis = run_friction_analysis
    sys.modules[module_name_str] = fake_module
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda strategy_name_str: module_name_str,
    )
    try:
        friction_result_obj = runner_module._run_friction_analysis_module(
            strategy_name_str=module_name_str,
            save_results_bool=True,
            output_dir_str=str(tmp_path),
            show_display_bool=False,
            dry_run_bool=False,
            backtest_start_date_str="2020-01-01",
            capital_base_float=50_000.0,
            end_date_str="2024-12-31",
        )
    finally:
        sys.modules.pop(module_name_str, None)

    assert friction_result_obj.summary_dict["strategy_name_str"] == "fake"
    assert captured_kwarg_dict == {
        "save_results_bool": True,
        "output_dir_str": str(tmp_path),
        "show_display_bool": False,
        "backtest_start_date_str": "2020-01-01",
        "capital_base_float": 50_000.0,
        "end_date_str": "2024-12-31",
    }

    output_str = capsys.readouterr().out
    assert "Ran FrictionAnalysis module: test_fake_friction_strategy" in output_str
    assert "Friction Verdict: Watch" in output_str
    assert "Cost Verdict: Conservative" in output_str
    assert "Auction Verdict: Stressed" in output_str
    assert "Auction P95 participation: 3.10%" in output_str
    assert "ADV-rule delta vs default: -$2 (-1.50 bps)" in output_str
    assert "Auction-impact estimate: $10 (8.00 bps)" in output_str
    assert "Auction-impact delta vs default: +$7 (+5.50 bps)" in output_str
    assert "Auction-impact P95 / max: 12.00 bps / 20.00 bps" in output_str
    assert "current 2.5 bps is materially optimistic" in output_str
    assert "Current ann return / Sharpe: 14.25% / 1.23" in output_str
    assert "Auction-impact est ann return / Sharpe: 13.10% / 1.16" in output_str
    assert "Estimated ann return / Sharpe change: -1.15 pp / -0.07" in output_str
    assert "Top red asset: AAA" in output_str
    assert "Scale to p95 auction Red: 0.50x" in output_str
    assert "Scale to max = 100% auction proxy" not in output_str


def test_run_friction_analysis_module_fails_when_hook_is_missing(monkeypatch):
    module_name_str = "test_fake_missing_friction_strategy"
    sys.modules[module_name_str] = ModuleType(module_name_str)
    monkeypatch.setattr(
        runner_module,
        "_resolve_strategy_module_import_str",
        lambda strategy_name_str: module_name_str,
    )
    try:
        with pytest.raises(AttributeError, match="run_friction_analysis"):
            runner_module._run_friction_analysis_module(
                strategy_name_str=module_name_str,
                save_results_bool=True,
                output_dir_str="results",
                show_display_bool=False,
                dry_run_bool=False,
                backtest_start_date_str=None,
                capital_base_float=None,
                end_date_str=None,
            )
    finally:
        sys.modules.pop(module_name_str, None)


def test_run_friction_analysis_module_dry_run_resolves_existing_file(capsys):
    result_obj = runner_module._run_friction_analysis_module(
        strategy_name_str="dv2/strategy_mr_dv2.py",
        save_results_bool=False,
        output_dir_str="results",
        show_display_bool=False,
        dry_run_bool=True,
        backtest_start_date_str=None,
        capital_base_float=None,
        end_date_str=None,
    )

    assert result_obj is None
    output_str = capsys.readouterr().out
    assert "Resolved strategy module: strategies.dv2.strategy_mr_dv2" in output_str


def test_deployment_wired_strategy_modules_expose_friction_hooks():
    pytest.importorskip("norgatedata")

    from alpha.live.release_manifest import SUPPORTED_STRATEGY_IMPORT_TUPLE

    missing_hook_list = []
    for strategy_import_str in SUPPORTED_STRATEGY_IMPORT_TUPLE:
        module_import_str = strategy_import_str.split(":", maxsplit=1)[0]
        strategy_module = importlib.import_module(module_import_str)
        run_friction_analysis_fn = getattr(strategy_module, "run_friction_analysis", None)
        if not callable(run_friction_analysis_fn):
            missing_hook_list.append(module_import_str)

    assert missing_hook_list == []
