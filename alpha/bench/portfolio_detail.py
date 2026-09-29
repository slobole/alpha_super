"""Read saved portfolio charts and format engine attribution for BENCH."""

import math

from alpha.bench import portfolio_compare
from alpha.engine.portfolio_attribution import portfolio_attribution_dict


def _chart_dict(value_series, label_str: str) -> dict:
    """SVG coordinates from every saved observation; no return resampling."""
    value_list = [float(value_float) for value_float in value_series]
    if len(value_list) < 2 or not all(math.isfinite(value_float) for value_float in value_list):
        return {}
    low_float, high_float = min(value_list), max(value_list)
    span_float = high_float - low_float or 1.0
    point_str = " ".join(
        f"{60 + 900 * index_int / (len(value_list) - 1):.2f},{20 + 190 * (high_float - value_float) / span_float:.2f}"
        for index_int, value_float in enumerate(value_list)
    )
    return {"label_str": label_str, "point_str": point_str, "low_float": low_float, "high_float": high_float,
            "start_str": str(value_series.index[0].date()), "end_str": str(value_series.index[-1].date())}


def detail_dict(overview_obj) -> dict:
    result_dict = {"chart_list": [], "contribution_list": [], "attribution_error_str": "", "episode_str": "",
                   "observation_count_int": 0, "saved_config_dict": None, "run_history_list": []}
    for run_obj in overview_obj.run_entry_list:
        result_dict["run_history_list"].append({"run_obj": run_obj, "snapshot_bool": isinstance(run_obj.metadata_dict.get("source_config_dict"), dict)})
    run_obj = overview_obj.latest_metric_run
    if run_obj is None:
        result_dict["attribution_error_str"] = "Build this portfolio to measure its results."
        return result_dict
    result_dict["saved_config_dict"] = run_obj.metadata_dict.get("source_config_dict")
    portfolio_obj = portfolio_compare._load_portfolio(run_obj)
    if portfolio_obj is None:
        result_dict["attribution_error_str"] = "The saved portfolio pickle is unavailable. Reports remain accessible below."
        return result_dict
    for column_str, label_str, scale_float in (("total_value", "Portfolio value", 1.0), ("drawdown", "Drawdown (%)", 100.0)):
        if column_str in portfolio_obj.results:
            chart_dict = _chart_dict(portfolio_obj.results[column_str] * scale_float, label_str)
            if chart_dict:
                result_dict["chart_list"].append(chart_dict)
    try:
        attribution_dict = portfolio_attribution_dict(portfolio_obj)
        contribution_df = attribution_dict["contribution_df"]
        result_dict["contribution_list"] = [{"name_str": str(name_str), **row_series.to_dict()} for name_str, row_series in contribution_df.iterrows()]
        result_dict["observation_count_int"] = attribution_dict["observation_count_int"]
        if attribution_dict["peak_obj"] is not None:
            result_dict["episode_str"] = f"{attribution_dict['peak_obj'].date()} → {attribution_dict['trough_obj'].date()} ({attribution_dict['episode_count_int']} return observations)"
    except (ValueError, KeyError, AttributeError, TypeError) as exception_obj:
        result_dict["attribution_error_str"] = str(exception_obj)
    return result_dict
