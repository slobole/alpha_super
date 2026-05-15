"""
Run a strategy module's FrictionAnalysis hook by name.

Usage:
    uv run python strategies/run_friction_analysis.py strategy_mr_dv2.py
    uv run python strategies/run_friction_analysis.py dv2/strategy_mr_dv2.py
    uv run python strategies/run_friction_analysis.py strategies.dv2.strategy_mr_dv2

The selected module must expose:

    run_friction_analysis(save_results_bool, output_dir_str)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]

if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from strategies.run_strategy import _resolve_strategy_module_import_str


def _run_friction_analysis_module(
    strategy_name_str: str,
    save_results_bool: bool,
    output_dir_str: str,
    show_display_bool: bool,
    dry_run_bool: bool,
    backtest_start_date_str: str | None,
    capital_base_float: float | None,
    end_date_str: str | None,
):
    module_import_str = _resolve_strategy_module_import_str(strategy_name_str)

    if dry_run_bool:
        print(f"Resolved strategy module: {module_import_str}")
        return None

    strategy_module = importlib.import_module(module_import_str)
    run_friction_analysis_fn = getattr(strategy_module, "run_friction_analysis", None)
    if run_friction_analysis_fn is None:
        raise AttributeError(
            f"Module '{module_import_str}' does not expose run_friction_analysis(...). "
            "Add a strategy-level FrictionAnalysis hook before using this runner."
        )

    signature_obj = inspect.signature(run_friction_analysis_fn)
    run_kwarg_dict = {}
    if "save_results_bool" in signature_obj.parameters:
        run_kwarg_dict["save_results_bool"] = save_results_bool
    if "output_dir_str" in signature_obj.parameters:
        run_kwarg_dict["output_dir_str"] = output_dir_str
    if "show_display_bool" in signature_obj.parameters:
        run_kwarg_dict["show_display_bool"] = show_display_bool
    if (
        backtest_start_date_str is not None
        and "backtest_start_date_str" in signature_obj.parameters
    ):
        run_kwarg_dict["backtest_start_date_str"] = backtest_start_date_str
    if capital_base_float is not None and "capital_base_float" in signature_obj.parameters:
        run_kwarg_dict["capital_base_float"] = capital_base_float
    if end_date_str is not None and "end_date_str" in signature_obj.parameters:
        run_kwarg_dict["end_date_str"] = end_date_str

    friction_result_obj = run_friction_analysis_fn(**run_kwarg_dict)
    print(f"Ran FrictionAnalysis module: {module_import_str}")
    _print_friction_analysis_summary(friction_result_obj)
    return friction_result_obj


def _print_friction_analysis_summary(friction_result_obj) -> None:
    if friction_result_obj is None:
        return

    output_dir_path = getattr(friction_result_obj, "output_dir_path", None)
    if output_dir_path is not None:
        print(f"Report folder: {Path(output_dir_path).resolve()}")

    summary_dict = getattr(friction_result_obj, "summary_dict", None)
    if not summary_dict:
        return

    print("\nFriction summary:")
    print(f"  Strategy: {summary_dict.get('strategy_name_str', 'N/A')}")
    print(f"  Friction Verdict: {summary_dict.get('friction_verdict_str', 'N/A')}")
    print(f"  Cost Verdict: {summary_dict.get('cost_verdict_str', 'N/A')}")
    print(f"  Auction Verdict: {summary_dict.get('auction_verdict_str', 'N/A')}")
    print(f"  Why: {summary_dict.get('verdict_explanation_str', 'N/A')}")
    print(f"  Assessed orders: {_format_int_str(summary_dict.get('assessed_order_count_int'))}")
    print(f"  Assessed notional: {_format_dollar_str(summary_dict.get('assessed_notional_float'))}")
    print(f"  Auction P95 participation: {_format_pct_str(summary_dict.get('p95_participation_float'))}")
    print(f"  Auction max participation: {_format_pct_str(summary_dict.get('max_participation_float'))}")
    print(
        "  Auction red orders: "
        f"{_format_int_str(summary_dict.get('red_order_count_int'))} "
        f"({_format_pct_str(summary_dict.get('red_order_share_float'))})"
    )
    print(
        "  Auction stressed orders: "
        f"{_format_int_str(summary_dict.get('capacity_stressed_order_count_int'))} "
        f"({_format_pct_str(summary_dict.get('capacity_stressed_order_share_float'))})"
    )
    print(f"  Red notional share: {_format_pct_str(summary_dict.get('red_notional_share_float'))}")
    print(f"  Capacity pass rate: {_format_pct_str(summary_dict.get('capacity_pass_order_rate_float'))}")
    print(f"  ADV P95 participation: {_format_pct_str(summary_dict.get('adv_p95_participation_float'))}")
    print(f"  ADV max participation: {_format_pct_str(summary_dict.get('adv_max_participation_float'))}")
    print(f"  ADV red orders: {_format_int_str(summary_dict.get('adv_red_order_count_int'))}")
    print(
        "  ADV-rule slippage cost: "
        f"{_format_dollar_str(summary_dict.get('adv_rule_slippage_dollar_float'))} "
        f"({_format_bps_str(summary_dict.get('adv_rule_slippage_blended_bps_float'))})"
    )
    print(
        "  Current default slippage cost: "
        f"{_format_dollar_str(summary_dict.get('default_slippage_dollar_float'))} "
        f"({_format_bps_str(summary_dict.get('default_slippage_blended_bps_float'))})"
    )
    print(
        "  ADV-rule delta vs default: "
        f"{_format_dollar_signed_str(summary_dict.get('adv_rule_minus_default_dollar_float'))} "
        f"({_format_bps_signed_str(summary_dict.get('adv_rule_minus_default_bps_float'))})"
    )
    print(
        "  Auction-impact estimate: "
        f"{_format_dollar_str(summary_dict.get('auction_impact_dollar_float'))} "
        f"({_format_bps_str(summary_dict.get('auction_impact_blended_bps_float'))})"
    )
    print(
        "  Auction-impact delta vs default: "
        f"{_format_dollar_signed_str(summary_dict.get('auction_impact_minus_default_dollar_float'))} "
        f"({_format_bps_signed_str(summary_dict.get('auction_impact_minus_default_bps_float'))})"
    )
    print(
        "  Auction-impact P95 / max: "
        f"{_format_bps_str(summary_dict.get('auction_impact_p95_bps_float'))} / "
        f"{_format_bps_str(summary_dict.get('auction_impact_max_bps_float'))}"
    )
    print(
        "  Auction-impact readout: "
        f"{summary_dict.get('auction_impact_interpretation_str', 'N/A')}"
    )
    print(
        "  Current ann return / Sharpe: "
        f"{_format_pct_str(summary_dict.get('current_annual_return_float'))} / "
        f"{_format_float_str(summary_dict.get('current_sharpe_float'))}"
    )
    print(
        "  Auction-impact est ann return / Sharpe: "
        f"{_format_pct_str(summary_dict.get('auction_impact_estimated_annual_return_float'))} / "
        f"{_format_float_str(summary_dict.get('auction_impact_estimated_sharpe_float'))}"
    )
    print(
        "  Estimated ann return / Sharpe change: "
        f"{_format_pp_signed_str(summary_dict.get('auction_impact_estimated_annual_return_delta_float'))} / "
        f"{_format_float_signed_str(summary_dict.get('auction_impact_estimated_sharpe_delta_float'))}"
    )
    print(f"  Top red asset: {summary_dict.get('top_red_asset_str') or 'N/A'}")
    print(
        "  Top red asset share: "
        f"{_format_pct_str(summary_dict.get('top_red_asset_red_notional_share_float'))}"
    )
    print(
        "  Scale to p95 auction Red: "
        f"{_format_scale_str(summary_dict.get('scale_to_p95_auction_red_threshold_float'))}"
    )
    print(
        "  Scale to p95 ADV = 5%: "
        f"{_format_scale_str(summary_dict.get('scale_to_p95_adv_5pct_float'))}"
    )


def _format_int_str(value_obj) -> str:
    try:
        return f"{int(value_obj):,}"
    except (TypeError, ValueError):
        return "N/A"


def _format_dollar_str(value_obj) -> str:
    try:
        return f"${float(value_obj):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_dollar_signed_str(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return "N/A"
    sign_str = "+" if value_float >= 0.0 else "-"
    return f"{sign_str}${abs(value_float):,.0f}"


def _format_pct_str(value_obj) -> str:
    try:
        return f"{float(value_obj) * 100.0:,.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_bps_str(value_obj) -> str:
    try:
        return f"{float(value_obj):,.2f} bps"
    except (TypeError, ValueError):
        return "N/A"


def _format_bps_signed_str(value_obj) -> str:
    try:
        return f"{float(value_obj):+,.2f} bps"
    except (TypeError, ValueError):
        return "N/A"


def _format_scale_str(value_obj) -> str:
    try:
        return f"{float(value_obj):,.2f}x"
    except (TypeError, ValueError):
        return "N/A"


def _format_float_str(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return "N/A"
    if value_float != value_float:
        return "N/A"
    return f"{value_float:,.2f}"


def _format_float_signed_str(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return "N/A"
    if value_float != value_float:
        return "N/A"
    return f"{value_float:+,.2f}"


def _format_pp_signed_str(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return "N/A"
    if value_float != value_float:
        return "N/A"
    return f"{value_float * 100.0:+,.2f} pp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy_name_str",
        help="Strategy module name, full import path, or .py path.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root directory for saved FrictionAnalysis artifacts.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing report artifacts.",
    )
    parser.add_argument(
        "--show-display",
        action="store_true",
        help="Let the strategy print its own verbose diagnostic displays.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve the strategy module.",
    )
    parser.add_argument(
        "--backtest-start-date",
        dest="backtest_start_date_str",
        default=None,
        help="Optional backtest start date passed through when the hook supports it.",
    )
    parser.add_argument(
        "--capital-base",
        dest="capital_base_float",
        type=float,
        default=None,
        help="Optional capital base passed through when the hook supports it.",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date_str",
        default=None,
        help="Optional data end date passed through when the hook supports it.",
    )
    arg_namespace = parser.parse_args()

    _run_friction_analysis_module(
        strategy_name_str=arg_namespace.strategy_name_str,
        save_results_bool=not arg_namespace.no_save,
        output_dir_str=arg_namespace.output_dir,
        show_display_bool=arg_namespace.show_display,
        dry_run_bool=arg_namespace.dry_run,
        backtest_start_date_str=arg_namespace.backtest_start_date_str,
        capital_base_float=arg_namespace.capital_base_float,
        end_date_str=arg_namespace.end_date_str,
    )


if __name__ == "__main__":
    main()
