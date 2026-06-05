from __future__ import annotations

import html
import importlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from alpha.engine.strategy import Strategy
from alpha.live.models import LiveRelease
from data.norgate_loader import use_norgate_data_profile


REFERENCE_RUN_VARIANT_PARAMETER_NAME_LIST: list[str] = [
    "backtest_start_date_str",
    "capital_base_float",
    "end_date_str",
]


def build_reference_output_dir_path(
    output_dir_str: str,
    env_mode_str: str,
    pod_id_str: str,
    as_of_ts: datetime,
) -> Path:
    timestamp_label_str = as_of_ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(output_dir_str)
        / "live_reference_compare"
        / str(env_mode_str)
        / str(pod_id_str)
        / timestamp_label_str
    )


def strategy_module_import_str_from_release(release_obj: LiveRelease) -> str:
    return str(release_obj.strategy_import_str).split(":", maxsplit=1)[0]


def inspect_auto_reference_support_dict(release_obj: LiveRelease) -> dict[str, object]:
    module_import_str = strategy_module_import_str_from_release(release_obj)
    strategy_module_obj = importlib.import_module(module_import_str)
    run_variant_fn = getattr(strategy_module_obj, "run_variant", None)
    if run_variant_fn is None:
        return {
            "supported_bool": False,
            "module_import_str": module_import_str,
            "missing_parameter_list": list(REFERENCE_RUN_VARIANT_PARAMETER_NAME_LIST),
            "reason_str": "missing_run_variant",
        }

    signature_obj = inspect.signature(run_variant_fn)
    missing_parameter_list = [
        parameter_name_str
        for parameter_name_str in REFERENCE_RUN_VARIANT_PARAMETER_NAME_LIST
        if parameter_name_str not in signature_obj.parameters
    ]
    return {
        "supported_bool": len(missing_parameter_list) == 0,
        "module_import_str": module_import_str,
        "missing_parameter_list": missing_parameter_list,
        "reason_str": "ok" if len(missing_parameter_list) == 0 else "missing_reference_parameters",
    }


def run_auto_reference_strategy(
    *,
    release_obj: LiveRelease,
    deployment_start_date_str: str,
    reference_end_date_str: str,
    deployment_initial_cash_float: float,
    output_dir_path_obj: Path,
) -> Strategy:
    module_import_str = strategy_module_import_str_from_release(release_obj)
    strategy_module_obj = importlib.import_module(module_import_str)
    run_variant_fn = getattr(strategy_module_obj, "run_variant", None)
    if run_variant_fn is None:
        raise AttributeError(
            f"Strategy module '{module_import_str}' does not expose run_variant(...), "
            "so automatic deployment reference compare cannot run it."
        )

    support_dict = inspect_auto_reference_support_dict(release_obj)
    missing_parameter_list = list(support_dict["missing_parameter_list"])
    if len(missing_parameter_list) > 0:
        raise AttributeError(
            f"Strategy module '{module_import_str}' run_variant(...) is missing "
            f"deployment-reference arguments: {missing_parameter_list}."
        )

    signature_obj = inspect.signature(run_variant_fn)
    run_kwarg_dict: dict[str, Any] = {
        "backtest_start_date_str": deployment_start_date_str,
        "capital_base_float": float(deployment_initial_cash_float),
        "end_date_str": reference_end_date_str,
    }
    if "show_display_bool" in signature_obj.parameters:
        run_kwarg_dict["show_display_bool"] = False
    if "save_results_bool" in signature_obj.parameters:
        run_kwarg_dict["save_results_bool"] = False
    if "output_dir_str" in signature_obj.parameters:
        run_kwarg_dict["output_dir_str"] = str(output_dir_path_obj)

    with use_norgate_data_profile(release_obj.data_profile_str):
        strategy_obj = run_variant_fn(**run_kwarg_dict)
    if not isinstance(strategy_obj, Strategy):
        raise TypeError(
            f"Strategy module '{module_import_str}' run_variant(...) returned "
            f"{type(strategy_obj).__name__}, expected Strategy."
        )
    return strategy_obj


def load_reference_maps_from_pickle(reference_strategy_pickle_path_str: str) -> dict[str, Any]:
    reference_strategy_obj = Strategy.read_pickle(reference_strategy_pickle_path_str)
    return build_reference_maps_dict(reference_strategy_obj)


def _date_key_str(timestamp_obj: object) -> str:
    return pd.Timestamp(timestamp_obj).date().isoformat()


def build_reference_maps_dict(reference_strategy_obj: Strategy) -> dict[str, Any]:
    transaction_map_dict: dict[tuple[str, str], dict[str, float | int | None]] = {}
    transaction_df = reference_strategy_obj.get_transactions()
    if transaction_df is not None and len(transaction_df) > 0:
        transaction_work_df = transaction_df.copy()
        transaction_work_df["date_str"] = transaction_work_df["bar"].map(_date_key_str)
        transaction_work_df["asset_str"] = transaction_work_df["asset"].astype(str)
        for (date_str, asset_str), group_df in transaction_work_df.groupby(["date_str", "asset_str"]):
            quantity_float = float(group_df["amount"].astype(float).sum())
            absolute_quantity_float = float(group_df["amount"].astype(float).abs().sum())
            weighted_notional_float = float(
                (group_df["amount"].astype(float).abs() * group_df["price"].astype(float)).sum()
            )
            avg_price_float = None
            if absolute_quantity_float > 0.0:
                avg_price_float = weighted_notional_float / absolute_quantity_float
            transaction_map_dict[(str(date_str), str(asset_str))] = {
                "quantity_float": quantity_float,
                "absolute_quantity_float": absolute_quantity_float,
                "weighted_notional_float": weighted_notional_float,
                "avg_price_float": avg_price_float,
                "trade_count_int": int(len(group_df)),
            }

    position_history_df = _build_reference_position_history_df(transaction_df)
    equity_by_date_dict, cash_by_date_dict = _build_reference_result_maps(reference_strategy_obj)
    realized_weight_history_df = _build_reference_realized_weight_history_df(reference_strategy_obj)

    return {
        "reference_strategy_obj": reference_strategy_obj,
        "transaction_map_dict": transaction_map_dict,
        "equity_by_date_dict": equity_by_date_dict,
        "cash_by_date_dict": cash_by_date_dict,
        "position_history_df": position_history_df,
        "realized_weight_history_df": realized_weight_history_df,
    }


def _build_reference_position_history_df(transaction_df: pd.DataFrame | None) -> pd.DataFrame:
    if transaction_df is None or len(transaction_df) == 0:
        return pd.DataFrame()

    current_position_map_dict: dict[str, float] = {}
    position_by_date_dict: dict[str, dict[str, float]] = {}
    transaction_work_df = transaction_df.copy().sort_values("bar")
    for _, transaction_ser in transaction_work_df.iterrows():
        date_str = _date_key_str(transaction_ser["bar"])
        asset_str = str(transaction_ser["asset"])
        current_position_map_dict[asset_str] = (
            float(current_position_map_dict.get(asset_str, 0.0))
            + float(transaction_ser["amount"])
        )
        if abs(float(current_position_map_dict[asset_str])) <= 1e-9:
            current_position_map_dict.pop(asset_str, None)
        position_by_date_dict[date_str] = dict(current_position_map_dict)

    if len(position_by_date_dict) == 0:
        return pd.DataFrame()
    position_history_df = pd.DataFrame.from_dict(position_by_date_dict, orient="index").sort_index()
    return position_history_df.fillna(0.0)


def _build_reference_result_maps(reference_strategy_obj: Strategy) -> tuple[dict[str, float], dict[str, float]]:
    equity_by_date_dict: dict[str, float] = {}
    cash_by_date_dict: dict[str, float] = {}
    results_df = getattr(reference_strategy_obj, "results", None)
    if results_df is None or len(results_df) == 0:
        return equity_by_date_dict, cash_by_date_dict

    if "total_value" in results_df.columns:
        for index_obj, value_obj in results_df["total_value"].dropna().items():
            equity_by_date_dict[_date_key_str(index_obj)] = float(value_obj)
    if "cash" in results_df.columns:
        for index_obj, value_obj in results_df["cash"].dropna().items():
            cash_by_date_dict[_date_key_str(index_obj)] = float(value_obj)
    return equity_by_date_dict, cash_by_date_dict


def _build_reference_realized_weight_history_df(reference_strategy_obj: Strategy) -> pd.DataFrame:
    realized_weight_df = getattr(reference_strategy_obj, "realized_weight_df", None)
    if realized_weight_df is None or len(realized_weight_df) == 0:
        return pd.DataFrame()
    realized_weight_history_df = realized_weight_df.copy()
    realized_weight_history_df.index = [_date_key_str(index_obj) for index_obj in realized_weight_history_df.index]
    return realized_weight_history_df.sort_index().fillna(0.0)


def _row_map_on_or_before_date_dict(history_df: pd.DataFrame, target_date_str: str) -> dict[str, float]:
    if history_df is None or len(history_df) == 0:
        return {}
    eligible_index = [str(index_obj) for index_obj in history_df.index if str(index_obj) <= target_date_str]
    if len(eligible_index) == 0:
        return {}
    row_ser = history_df.loc[eligible_index[-1]]
    return {
        str(field_name_str): float(value_float)
        for field_name_str, value_float in row_ser.dropna().items()
        if abs(float(value_float)) > 1e-12
    }


def value_on_or_before_date_float(
    value_by_date_dict: dict[str, float],
    target_date_str: str,
) -> float | None:
    eligible_date_list = sorted(date_str for date_str in value_by_date_dict if date_str <= target_date_str)
    if len(eligible_date_list) == 0:
        return None
    return float(value_by_date_dict[eligible_date_list[-1]])


def get_reference_position_map_dict(
    reference_maps_dict: dict[str, Any],
    target_date_str: str,
) -> dict[str, float]:
    return _row_map_on_or_before_date_dict(
        reference_maps_dict.get("position_history_df", pd.DataFrame()),
        target_date_str,
    )


def get_reference_realized_weight_map_dict(
    reference_maps_dict: dict[str, Any],
    target_date_str: str,
) -> dict[str, float]:
    return _row_map_on_or_before_date_dict(
        reference_maps_dict.get("realized_weight_history_df", pd.DataFrame()),
        target_date_str,
    )


def build_trade_fill_diff_row_dict_list(
    *,
    target_session_date_str: str,
    vplan_id_int: int,
    decision_plan_id_int: int,
    execution_row_dict_list: list[dict[str, object]],
    transaction_map_dict: dict[tuple[str, str], dict[str, float | int | None]],
    tolerance_float: float = 1e-9,
) -> list[dict[str, object]]:
    execution_row_by_asset_dict = {}
    for execution_row_dict in execution_row_dict_list:
        live_filled_share_float = float(execution_row_dict.get("filled_share_float") or 0.0)
        live_planned_share_float = float(execution_row_dict.get("planned_order_delta_share_float") or 0.0)
        if (
            abs(live_filled_share_float) > tolerance_float
            or abs(live_planned_share_float) > tolerance_float
        ):
            execution_row_by_asset_dict[str(execution_row_dict["asset_str"])] = execution_row_dict
    backtest_asset_set = {
        asset_str
        for date_str, asset_str in transaction_map_dict
        if str(date_str) == str(target_session_date_str)
    }
    row_dict_list: list[dict[str, object]] = []
    for asset_str in sorted(set(execution_row_by_asset_dict) | backtest_asset_set):
        execution_row_dict = execution_row_by_asset_dict.get(asset_str)
        transaction_dict = transaction_map_dict.get((target_session_date_str, asset_str), {})

        live_filled_share_float = (
            None
            if execution_row_dict is None
            else float(execution_row_dict.get("filled_share_float") or 0.0)
        )
        live_avg_fill_price_float = (
            None
            if execution_row_dict is None or execution_row_dict.get("avg_fill_price_float") is None
            else float(execution_row_dict["avg_fill_price_float"])
        )
        backtest_trade_share_float = (
            None
            if "quantity_float" not in transaction_dict
            else float(transaction_dict["quantity_float"] or 0.0)
        )
        backtest_avg_fill_price_float = (
            None
            if transaction_dict.get("avg_price_float") is None
            else float(transaction_dict["avg_price_float"])
        )
        live_planned_share_float = (
            None
            if execution_row_dict is None
            else float(execution_row_dict.get("planned_order_delta_share_float") or 0.0)
        )

        share_diff_float = (live_filled_share_float or 0.0) - (backtest_trade_share_float or 0.0)
        side_float = _side_float(backtest_trade_share_float)
        if side_float is None:
            side_float = _side_float(live_filled_share_float)
        if side_float is None:
            side_float = _side_float(live_planned_share_float)

        price_diff_float = None
        price_diff_bps_float = None
        price_diff_notional_float = None
        if live_avg_fill_price_float is not None and backtest_avg_fill_price_float is not None:
            price_diff_float = live_avg_fill_price_float - backtest_avg_fill_price_float
            if side_float is not None and backtest_avg_fill_price_float > 0.0:
                price_diff_bps_float = 10000.0 * side_float * (
                    (live_avg_fill_price_float / backtest_avg_fill_price_float) - 1.0
                )
            if side_float is not None:
                price_diff_notional_float = (
                    abs(live_filled_share_float or 0.0)
                    * side_float
                    * price_diff_float
                )

        notional_price_float = backtest_avg_fill_price_float
        if notional_price_float is None:
            notional_price_float = live_avg_fill_price_float
        notional_diff_float = None
        if notional_price_float is not None:
            notional_diff_float = share_diff_float * notional_price_float

        has_live_fill_bool = live_filled_share_float is not None and abs(live_filled_share_float) > tolerance_float
        has_live_plan_bool = live_planned_share_float is not None and abs(live_planned_share_float) > tolerance_float
        has_backtest_trade_bool = (
            backtest_trade_share_float is not None and abs(backtest_trade_share_float) > tolerance_float
        )
        if has_live_fill_bool and has_backtest_trade_bool:
            note_str = "matched"
        elif has_live_fill_bool:
            note_str = "live fill without matching backtest trade"
        elif has_live_plan_bool and has_backtest_trade_bool:
            note_str = "planned live order without live fill; backtest traded"
        elif has_live_plan_bool:
            note_str = "planned live order without live fill or matching backtest trade"
        elif has_backtest_trade_bool:
            note_str = "backtest trade without matching live fill"
        else:
            note_str = "no fill"

        row_dict_list.append(
            {
                "execution_date_str": target_session_date_str,
                "asset_str": asset_str,
                "side_str": _side_label_str(side_float),
                "live_planned_share_float": live_planned_share_float,
                "live_filled_share_float": live_filled_share_float,
                "backtest_trade_share_float": backtest_trade_share_float,
                "share_diff_float": share_diff_float,
                "live_avg_fill_price_float": live_avg_fill_price_float,
                "backtest_avg_fill_price_float": backtest_avg_fill_price_float,
                "price_diff_float": price_diff_float,
                "price_diff_bps_float": None if price_diff_bps_float is None else round(price_diff_bps_float, 10),
                "notional_diff_float": notional_diff_float,
                "price_diff_notional_float": price_diff_notional_float,
                "live_fill_count_int": (
                    0
                    if execution_row_dict is None
                    else int(execution_row_dict.get("fill_count_int") or 0)
                ),
                "backtest_trade_count_int": int(transaction_dict.get("trade_count_int") or 0),
                "vplan_id_int": int(vplan_id_int),
                "decision_plan_id_int": int(decision_plan_id_int),
                "note_str": note_str,
            }
        )
    return row_dict_list


def summarize_trade_fill_diff_dict(trade_fill_diff_df: pd.DataFrame) -> dict[str, object]:
    if trade_fill_diff_df is None or len(trade_fill_diff_df) == 0:
        return {
            "trade_row_count_int": 0,
            "assets_traded_count_int": 0,
            "total_abs_share_diff_float": 0.0,
            "total_abs_notional_diff_float": 0.0,
            "total_price_diff_notional_float": 0.0,
        }
    share_diff_ser = trade_fill_diff_df.get("share_diff_float", pd.Series(dtype=float)).fillna(0.0).astype(float)
    notional_diff_ser = trade_fill_diff_df.get("notional_diff_float", pd.Series(dtype=float)).fillna(0.0).astype(float)
    price_diff_notional_ser = (
        trade_fill_diff_df.get("price_diff_notional_float", pd.Series(dtype=float)).fillna(0.0).astype(float)
    )
    return {
        "trade_row_count_int": int(len(trade_fill_diff_df)),
        "assets_traded_count_int": int(trade_fill_diff_df["asset_str"].nunique()),
        "total_abs_share_diff_float": float(share_diff_ser.abs().sum()),
        "total_abs_notional_diff_float": float(notional_diff_ser.abs().sum()),
        "total_price_diff_notional_float": float(price_diff_notional_ser.sum()),
    }


def _side_float(quantity_float: float | None) -> float | None:
    if quantity_float is None:
        return None
    if float(quantity_float) > 0.0:
        return 1.0
    if float(quantity_float) < 0.0:
        return -1.0
    return None


def _side_label_str(side_float: float | None) -> str:
    if side_float is None:
        return "unknown"
    return "buy" if side_float > 0.0 else "sell"


def write_reference_compare_artifacts(
    *,
    output_dir_path_obj: Path,
    release_obj: LiveRelease,
    compare_report_dict: dict[str, Any],
    reference_maps_dict: dict[str, Any],
    live_history_row_dict_list: list[dict[str, object]],
    reference_strategy_pickle_path_str: str | None,
) -> dict[str, str]:
    output_dir_path_obj.mkdir(parents=True, exist_ok=True)

    reference_equity_df = _build_reference_equity_df(reference_maps_dict)
    live_equity_df = _build_live_equity_df(live_history_row_dict_list, compare_report_dict)
    tracking_error_df = _build_tracking_error_df(reference_equity_df, live_equity_df)
    fill_compare_df = pd.DataFrame(compare_report_dict.get("compare_row_dict_list", []))
    trade_fill_diff_df = pd.DataFrame(compare_report_dict.get("trade_fill_diff_row_dict_list", []))
    position_compare_df = _build_position_compare_df(fill_compare_df)
    trade_fill_summary_dict = summarize_trade_fill_diff_dict(trade_fill_diff_df)

    reference_equity_csv_path_obj = output_dir_path_obj / "reference_equity.csv"
    live_equity_csv_path_obj = output_dir_path_obj / "live_equity.csv"
    tracking_error_csv_path_obj = output_dir_path_obj / "tracking_error.csv"
    fill_compare_csv_path_obj = output_dir_path_obj / "fill_compare.csv"
    trade_fill_diff_csv_path_obj = output_dir_path_obj / "trade_fill_diff.csv"
    position_compare_csv_path_obj = output_dir_path_obj / "position_compare.csv"
    summary_json_path_obj = output_dir_path_obj / "summary.json"
    equity_png_path_obj = output_dir_path_obj / "equity_compare.png"
    tracking_png_path_obj = output_dir_path_obj / "tracking_error.png"
    html_path_obj = output_dir_path_obj / "index.html"

    reference_equity_df.to_csv(reference_equity_csv_path_obj, index=False)
    live_equity_df.to_csv(live_equity_csv_path_obj, index=False)
    tracking_error_df.to_csv(tracking_error_csv_path_obj, index=False)
    fill_compare_df.to_csv(fill_compare_csv_path_obj, index=False)
    trade_fill_diff_df.to_csv(trade_fill_diff_csv_path_obj, index=False)
    position_compare_df.to_csv(position_compare_csv_path_obj, index=False)

    _write_equity_chart_png(reference_equity_df, live_equity_df, equity_png_path_obj)
    _write_tracking_error_chart_png(tracking_error_df, tracking_png_path_obj)

    summary_dict = {
        "release_id_str": release_obj.release_id_str,
        "pod_id_str": release_obj.pod_id_str,
        "mode_str": release_obj.mode_str,
        "target_session_date_str": compare_report_dict.get("target_session_date_str"),
        "deployment_start_date_str": compare_report_dict.get("deployment_start_date_str"),
        "deployment_initial_cash_float": compare_report_dict.get("deployment_initial_cash_float"),
        "actual_equity_float": compare_report_dict.get("actual_equity_float"),
        "actual_equity_source_str": compare_report_dict.get("actual_equity_source_str"),
        "actual_equity_basis_str": compare_report_dict.get("actual_equity_basis_str"),
        "actual_equity_timestamp_str": compare_report_dict.get("actual_equity_timestamp_str"),
        "backtest_equity_float": compare_report_dict.get("backtest_equity_float"),
        "equity_tracking_error_float": compare_report_dict.get("equity_tracking_error_float"),
        "actual_cash_float": compare_report_dict.get("actual_cash_float"),
        "backtest_cash_float": compare_report_dict.get("backtest_cash_float"),
        "cash_diff_float": compare_report_dict.get("cash_diff_float"),
        **trade_fill_summary_dict,
        "reference_strategy_pickle_path_str": reference_strategy_pickle_path_str,
    }
    summary_json_path_obj.write_text(json.dumps(summary_dict, indent=2, sort_keys=True), encoding="utf-8")

    _write_html_report(
        html_path_obj=html_path_obj,
        release_obj=release_obj,
        compare_report_dict=compare_report_dict,
        summary_dict=summary_dict,
        equity_png_path_obj=equity_png_path_obj,
        tracking_png_path_obj=tracking_png_path_obj,
        trade_fill_diff_df=trade_fill_diff_df,
        fill_compare_df=fill_compare_df,
        position_compare_df=position_compare_df,
    )

    return {
        "output_dir_path_str": str(output_dir_path_obj),
        "html_path_str": str(html_path_obj),
        "summary_json_path_str": str(summary_json_path_obj),
        "reference_equity_csv_path_str": str(reference_equity_csv_path_obj),
        "live_equity_csv_path_str": str(live_equity_csv_path_obj),
        "tracking_error_csv_path_str": str(tracking_error_csv_path_obj),
        "fill_compare_csv_path_str": str(fill_compare_csv_path_obj),
        "trade_fill_diff_csv_path_str": str(trade_fill_diff_csv_path_obj),
        "position_compare_csv_path_str": str(position_compare_csv_path_obj),
        "equity_png_path_str": str(equity_png_path_obj),
        "tracking_error_png_path_str": str(tracking_png_path_obj),
    }


def _build_reference_equity_df(reference_maps_dict: dict[str, Any]) -> pd.DataFrame:
    equity_by_date_dict = dict(reference_maps_dict.get("equity_by_date_dict", {}))
    return pd.DataFrame(
        [
            {"date_str": date_str, "reference_equity_float": float(equity_float)}
            for date_str, equity_float in sorted(equity_by_date_dict.items())
        ]
    )


def _build_live_equity_df(
    live_history_row_dict_list: list[dict[str, object]],
    compare_report_dict: dict[str, Any],
) -> pd.DataFrame:
    live_row_dict_list: list[dict[str, object]] = []
    for history_row_dict in live_history_row_dict_list:
        updated_timestamp_str = str(history_row_dict["updated_timestamp_str"])
        live_row_dict_list.append(
            {
                "date_str": _date_key_str(updated_timestamp_str),
                "updated_timestamp_str": updated_timestamp_str,
                "live_equity_float": float(history_row_dict["total_value_float"]),
                "live_cash_float": float(history_row_dict["cash_float"]),
                "snapshot_stage_str": str(history_row_dict.get("snapshot_stage_str", "unknown")),
                "snapshot_source_str": str(history_row_dict.get("snapshot_source_str", "pod_state")),
            }
        )

    if len(live_row_dict_list) == 0 and compare_report_dict.get("actual_equity_float") is not None:
        live_row_dict_list.append(
            {
                "date_str": str(compare_report_dict["target_session_date_str"]),
                "updated_timestamp_str": str(compare_report_dict["target_session_date_str"]),
                "live_equity_float": float(compare_report_dict["actual_equity_float"]),
                "live_cash_float": compare_report_dict.get("actual_cash_float"),
                "snapshot_stage_str": str(compare_report_dict.get("actual_equity_source_str") or "fallback"),
                "snapshot_source_str": str(compare_report_dict.get("actual_equity_basis_str") or "fallback"),
            }
        )

    if len(live_row_dict_list) == 0:
        return pd.DataFrame(
            columns=[
                "date_str",
                "updated_timestamp_str",
                "live_equity_float",
                "live_cash_float",
                "snapshot_stage_str",
                "snapshot_source_str",
            ]
        )

    live_equity_df = pd.DataFrame(live_row_dict_list).sort_values(["date_str", "updated_timestamp_str"])
    return live_equity_df.groupby("date_str", as_index=False).tail(1).reset_index(drop=True)


def _build_tracking_error_df(reference_equity_df: pd.DataFrame, live_equity_df: pd.DataFrame) -> pd.DataFrame:
    if len(reference_equity_df) == 0 or len(live_equity_df) == 0:
        return pd.DataFrame(
            columns=["date_str", "reference_equity_float", "live_equity_float", "tracking_error_float"]
        )
    tracking_error_df = live_equity_df.merge(reference_equity_df, on="date_str", how="left")
    tracking_error_df["tracking_error_float"] = (
        tracking_error_df["live_equity_float"] / tracking_error_df["reference_equity_float"] - 1.0
    )
    return tracking_error_df


def _build_position_compare_df(fill_compare_df: pd.DataFrame) -> pd.DataFrame:
    if len(fill_compare_df) == 0:
        return pd.DataFrame()
    column_name_list = [
        column_name_str
        for column_name_str in [
            "asset_str",
            "target_position_float",
            "actual_position_float",
            "reference_position_float",
            "position_diff_float",
            "reference_position_diff_float",
        ]
        if column_name_str in fill_compare_df.columns
    ]
    return fill_compare_df[column_name_list].copy()


def _write_equity_chart_png(
    reference_equity_df: pd.DataFrame,
    live_equity_df: pd.DataFrame,
    output_png_path_obj: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_obj, axis_obj = plt.subplots(figsize=(10, 4))
    if len(reference_equity_df) > 0:
        axis_obj.plot(
            pd.to_datetime(reference_equity_df["date_str"]),
            reference_equity_df["reference_equity_float"],
            label="Reference backtest",
            linewidth=1.8,
        )
    if len(live_equity_df) > 0:
        axis_obj.scatter(
            pd.to_datetime(live_equity_df["date_str"]),
            live_equity_df["live_equity_float"],
            label="Live/Paper/Incubation",
            s=34,
            zorder=3,
        )
    axis_obj.set_title("Equity: Deployment Reference vs Actual")
    axis_obj.set_xlabel("Date")
    axis_obj.set_ylabel("Equity")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend()
    fig_obj.tight_layout()
    fig_obj.savefig(output_png_path_obj, dpi=140)
    plt.close(fig_obj)


def _write_tracking_error_chart_png(
    tracking_error_df: pd.DataFrame,
    output_png_path_obj: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_obj, axis_obj = plt.subplots(figsize=(10, 3.5))
    if len(tracking_error_df) > 0:
        axis_obj.axhline(0.0, color="black", linewidth=0.9)
        axis_obj.plot(
            pd.to_datetime(tracking_error_df["date_str"]),
            tracking_error_df["tracking_error_float"],
            marker="o",
            linewidth=1.4,
        )
    axis_obj.set_title("Tracking Error")
    axis_obj.set_xlabel("Date")
    axis_obj.set_ylabel("live/reference - 1")
    axis_obj.grid(True, alpha=0.25)
    fig_obj.tight_layout()
    fig_obj.savefig(output_png_path_obj, dpi=140)
    plt.close(fig_obj)


def _write_html_report(
    *,
    html_path_obj: Path,
    release_obj: LiveRelease,
    compare_report_dict: dict[str, Any],
    summary_dict: dict[str, Any],
    equity_png_path_obj: Path,
    tracking_png_path_obj: Path,
    trade_fill_diff_df: pd.DataFrame,
    fill_compare_df: pd.DataFrame,
    position_compare_df: pd.DataFrame,
) -> None:
    window_card_html_str = _card_grid_html_str(
        [
            ("Target session", summary_dict.get("target_session_date_str")),
            ("Deployment start", summary_dict.get("deployment_start_date_str")),
            ("Starting capital", summary_dict.get("deployment_initial_cash_float")),
            ("Trade rows", summary_dict.get("trade_row_count_int")),
        ]
    )
    trade_card_html_str = _card_grid_html_str(
        [
            ("Assets traded", summary_dict.get("assets_traded_count_int")),
            ("Abs share diff", summary_dict.get("total_abs_share_diff_float")),
            ("Abs notional diff", summary_dict.get("total_abs_notional_diff_float")),
            ("Price diff notional", summary_dict.get("total_price_diff_notional_float")),
        ]
    )
    equity_card_html_str = _card_grid_html_str(
        [
            ("Actual equity", summary_dict.get("actual_equity_float")),
            ("Reference equity", summary_dict.get("backtest_equity_float")),
            ("Equity tracking error", summary_dict.get("equity_tracking_error_float")),
            ("Actual equity source", summary_dict.get("actual_equity_source_str")),
            ("Actual equity basis", summary_dict.get("actual_equity_basis_str")),
            ("Cash diff", summary_dict.get("cash_diff_float")),
        ]
    )
    trade_fill_table_html_str = _df_to_labeled_html_table_str(
        trade_fill_diff_df,
        {
            "execution_date_str": "Date",
            "asset_str": "Asset",
            "side_str": "Side",
            "live_planned_share_float": "Live planned shares",
            "live_filled_share_float": "Live shares",
            "backtest_trade_share_float": "Backtest shares",
            "share_diff_float": "Share diff",
            "live_avg_fill_price_float": "Live avg fill",
            "backtest_avg_fill_price_float": "Backtest fill",
            "price_diff_float": "Price diff",
            "price_diff_bps_float": "Price diff bps",
            "notional_diff_float": "Notional diff",
            "price_diff_notional_float": "Price diff $",
            "live_fill_count_int": "Live fill rows",
            "backtest_trade_count_int": "Backtest rows",
            "note_str": "Note",
        },
    )
    execution_quality_table_html_str = _df_to_labeled_html_table_str(
        fill_compare_df,
        {
            "asset_str": "Asset",
            "planned_order_delta_share_float": "Planned shares",
            "filled_share_float": "Filled shares",
            "quantity_diff_float": "Live quantity diff",
            "avg_fill_price_float": "Avg fill",
            "official_open_price_float": "Official open",
            "official_open_slippage_bps_float": "Open slippage bps",
            "official_open_slippage_notional_float": "Open slippage $",
            "vplan_reference_price_float": "VPlan reference",
            "vplan_reference_slippage_bps_float": "Reference slippage bps",
            "vplan_reference_slippage_notional_float": "Reference slippage $",
            "backtest_quantity_diff_float": "Backtest quantity diff",
            "backtest_fill_price_diff_float": "Backtest fill price diff",
        },
    )
    position_table_html_str = _df_to_labeled_html_table_str(
        position_compare_df,
        {
            "asset_str": "Asset",
            "target_position_float": "Target position",
            "actual_position_float": "Actual position",
            "reference_position_float": "Reference position",
            "position_diff_float": "Live target diff",
            "reference_position_diff_float": "Reference position diff",
        },
    )
    raw_fill_compare_df = fill_compare_df.drop(columns=["reference_price_float"], errors="ignore")
    raw_fill_table_html_str = _df_to_html_table_str(raw_fill_compare_df)
    raw_position_table_html_str = _df_to_html_table_str(position_compare_df)

    html_text_str = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(release_obj.pod_id_str)} Live vs Backtest</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #101828; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin: 16px 0 24px; }}
    .card {{ border: 1px solid #d0d5dd; border-radius: 6px; padding: 10px; background: #fff; }}
    .card span {{ display: block; color: #667085; font-size: 12px; }}
    .card strong {{ display: block; margin-top: 4px; font-size: 14px; }}
    section {{ margin: 22px 0 30px; }}
    details {{ margin-top: 14px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    img {{ max-width: 100%; border: 1px solid #d0d5dd; border-radius: 6px; margin: 8px 0 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d0d5dd; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f2f4f7; }}
  </style>
</head>
<body>
  <h1>{html.escape(release_obj.pod_id_str)} Live vs Backtest</h1>
  <p>Release: {html.escape(release_obj.release_id_str)}</p>
  <section>
    <h2>Comparison Window</h2>
    {window_card_html_str}
  </section>
  <section>
    <h2>Trade Fill Difference</h2>
    {trade_card_html_str}
    {trade_fill_table_html_str}
  </section>
  <section>
    <h2>Equity Snapshot</h2>
    {equity_card_html_str}
    <img src="{html.escape(equity_png_path_obj.name)}" alt="Equity comparison chart">
    <img src="{html.escape(tracking_png_path_obj.name)}" alt="Tracking error chart">
  </section>
  <section>
    <h2>Execution Detail</h2>
    {execution_quality_table_html_str}
  </section>
  <section>
    <h2>Position Parity</h2>
    {position_table_html_str}
  </section>
  <section>
    <h2>Raw Detail Tables</h2>
    <details>
      <summary>Fill compare rows</summary>
      {raw_fill_table_html_str}
    </details>
    <details>
      <summary>Position compare rows</summary>
      {raw_position_table_html_str}
    </details>
  </section>
</body>
</html>
"""
    html_path_obj.write_text(html_text_str, encoding="utf-8")


def _card_grid_html_str(item_list: list[tuple[str, object]]) -> str:
    card_html_list = [
        (
            f"<div class=\"card\"><span>{html.escape(str(label_str))}</span>"
            f"<strong>{html.escape(str('unavailable' if value_obj is None else value_obj))}</strong></div>"
        )
        for label_str, value_obj in item_list
    ]
    return f"<div class=\"cards\">{''.join(card_html_list)}</div>"


def _df_to_labeled_html_table_str(
    data_df: pd.DataFrame,
    column_label_map_dict: dict[str, str],
) -> str:
    if data_df is None or len(data_df) == 0:
        return "<p>No rows.</p>"
    column_name_list = [
        column_name_str
        for column_name_str in column_label_map_dict
        if column_name_str in data_df.columns
    ]
    if len(column_name_list) == 0:
        return "<p>No rows.</p>"
    display_df = data_df[column_name_list].copy()
    display_df = display_df.rename(columns=column_label_map_dict)
    return display_df.to_html(index=False, escape=True)


def _df_to_html_table_str(data_df: pd.DataFrame) -> str:
    if data_df is None or len(data_df) == 0:
        return "<p>No rows.</p>"
    display_df = data_df.copy()
    return display_df.to_html(index=False, escape=True)
