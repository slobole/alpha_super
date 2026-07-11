from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path, save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    DEFAULT_CONFIG,
    AtrNormalizedNdxStrategy,
    get_atr_normalized_ndx_data,
)


UNIVERSE_CONFIG_LIST: list[dict[str, str]] = [
    {
        "request_label_str": "Nasdaq 100 Biotechnology",
        "norgate_indexname_str": "Nasdaq Biotechnology",
    },
    {
        "request_label_str": "Nasdaq Composite",
        "norgate_indexname_str": "Nasdaq Composite",
    },
    {
        "request_label_str": "NYSE Composite",
        "norgate_indexname_str": "NYSE Composite",
    },
    {
        "request_label_str": "Russell Micro Cap",
        "norgate_indexname_str": "Russell Micro Cap",
    },
    {
        "request_label_str": "Russell Mid Cap",
        "norgate_indexname_str": "Russell Mid Cap",
    },
    {
        "request_label_str": "Russell Small Cap Completeness",
        "norgate_indexname_str": "Russell Small Cap Completeness",
    },
]


def _slug_str(raw_value_str: str) -> str:
    keep_char_list: list[str] = []
    for char_str in raw_value_str.lower():
        if char_str.isalnum():
            keep_char_list.append(char_str)
        else:
            keep_char_list.append("_")
    return "_".join(filter(None, "".join(keep_char_list).split("_")))


def _summary_value_float(summary_df: pd.DataFrame, metric_name_str: str) -> float | None:
    if metric_name_str not in summary_df.index:
        return None
    value_obj = summary_df.loc[metric_name_str]
    if isinstance(value_obj, pd.Series):
        value_obj = value_obj.iloc[0]
    if pd.isna(value_obj):
        return None
    return float(value_obj)


def _summary_value_str(summary_df: pd.DataFrame, metric_name_str: str) -> str | None:
    if metric_name_str not in summary_df.index:
        return None
    value_obj = summary_df.loc[metric_name_str]
    if isinstance(value_obj, pd.Series):
        value_obj = value_obj.iloc[0]
    if pd.isna(value_obj):
        return None
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.date().isoformat()
    return str(value_obj)


def _write_sweep_index_html(sweep_output_path: Path, summary_df: pd.DataFrame) -> None:
    row_html_list: list[str] = []
    resolved_sweep_output_path = sweep_output_path.resolve()
    for _, row_ser in summary_df.iterrows():
        report_path_obj = row_ser.get("report_path_str")
        error_str = row_ser.get("error_str", "")
        if pd.isna(report_path_obj):
            report_cell_str = str(error_str)
        else:
            report_rel_path_str = str(Path(report_path_obj).resolve().relative_to(resolved_sweep_output_path))
            report_cell_str = f"<a href=\"{report_rel_path_str}\">report.html</a>"
        row_html_list.append(
            "<tr>"
            f"<td>{row_ser['request_label_str']}</td>"
            f"<td>{row_ser['norgate_indexname_str']}</td>"
            f"<td>{report_cell_str}</td>"
            f"<td>{row_ser.get('start_date_str', '')}</td>"
            f"<td>{row_ser.get('end_date_str', '')}</td>"
            f"<td>{row_ser.get('ann_return_pct_float', '')}</td>"
            f"<td>{row_ser.get('sharpe_float', '')}</td>"
            f"<td>{row_ser.get('max_drawdown_pct_float', '')}</td>"
            f"<td>{row_ser.get('trade_count_int', '')}</td>"
            "</tr>"
        )

    index_html_str = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ATR-normalized universe sweep</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d5d8dc; padding: 8px; text-align: left; }}
th {{ background: #f4f6f7; }}
td:nth-child(6), td:nth-child(7), td:nth-child(8), td:nth-child(9) {{ text-align: right; }}
</style>
</head>
<body>
<h1>ATR-normalized universe sweep</h1>
<p>Vanilla monthly ATR-normalized momentum logic from strategy_mo_atr_normalized_ndx.py.</p>
<table>
<thead>
<tr>
<th>Requested basket</th>
<th>Norgate index</th>
<th>Report</th>
<th>Start</th>
<th>End</th>
<th>Ann. return %</th>
<th>Sharpe</th>
<th>Max DD %</th>
<th>Trades</th>
</tr>
</thead>
<tbody>
{''.join(row_html_list)}
</tbody>
</table>
</body>
</html>
"""
    (sweep_output_path / "index.html").write_text(index_html_str, encoding="utf-8")


def run_universe_backtest(
    request_label_str: str,
    norgate_indexname_str: str,
    sweep_output_path: Path,
    output_dir_str: str,
    show_progress_bool: bool,
) -> dict[str, object]:
    universe_slug_str = _slug_str(norgate_indexname_str)
    strategy_name_str = f"strategy_mo_atr_normalized_{universe_slug_str}"
    universe_output_path = sweep_output_path / universe_slug_str

    config_obj = replace(DEFAULT_CONFIG, indexname_str=norgate_indexname_str)
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_data(config_obj)

    strategy_obj = AtrNormalizedNdxStrategy(
        name=strategy_name_str,
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** This sweep preserves the base strategy timing:
    # decisions use the completed month-end close, and orders execute on the
    # next tradable open. Only the PIT universe name changes between rows.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=None,
    )

    saved_output_path = save_results(
        strategy_obj,
        output_dir=output_dir_str,
        output_path=universe_output_path,
    )

    summary_df = strategy_obj.summary
    trade_count_float = _summary_value_float(strategy_obj.summary_trades, "# Trades")
    return {
        "request_label_str": request_label_str,
        "norgate_indexname_str": norgate_indexname_str,
        "strategy_name_str": strategy_name_str,
        "start_date_str": _summary_value_str(summary_df, "Start"),
        "end_date_str": _summary_value_str(summary_df, "End"),
        "ann_return_pct_float": _summary_value_float(summary_df, "Return (Ann.) [%]"),
        "volatility_ann_pct_float": _summary_value_float(summary_df, "Volatility (Ann.) [%]"),
        "sharpe_float": _summary_value_float(summary_df, "Sharpe Ratio"),
        "max_drawdown_pct_float": _summary_value_float(summary_df, "Max. Drawdown [%]"),
        "trade_count_int": None if trade_count_float is None else int(trade_count_float),
        "report_path_str": str((saved_output_path / "report.html").resolve()),
        "summary_path_str": str((saved_output_path / "summary.json").resolve()),
        "transaction_path_str": str((saved_output_path / "transactions.csv").resolve()),
    }


def _load_existing_result_dict_list(sweep_output_path: Path) -> list[dict[str, object]]:
    summary_json_path = sweep_output_path / "summary.json"
    if not summary_json_path.exists():
        return []
    return json.loads(summary_json_path.read_text(encoding="utf-8"))


def run_sweep(
    output_dir_str: str,
    show_progress_bool: bool,
    sweep_output_dir_str: str | None = None,
) -> Path:
    if sweep_output_dir_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        sweep_output_path = build_research_output_path(
            output_dir=output_dir_str,
            entity_type_str="strategy",
            entity_id_str="strategy_mo_atr_normalized_universe_sweep",
            analysis_type_str="vanilla_backtest",
            timestamp_str=timestamp_str,
        ).resolve()
    else:
        sweep_output_path = Path(sweep_output_dir_str).resolve()
    sweep_output_path.mkdir(parents=True, exist_ok=True)

    result_dict_list = _load_existing_result_dict_list(sweep_output_path)
    completed_label_set = {
        str(result_dict["request_label_str"])
        for result_dict in result_dict_list
        if "request_label_str" in result_dict
    }
    for universe_config_dict in UNIVERSE_CONFIG_LIST:
        request_label_str = universe_config_dict["request_label_str"]
        norgate_indexname_str = universe_config_dict["norgate_indexname_str"]
        if request_label_str in completed_label_set:
            print(f"SKIP {request_label_str}: already present in {sweep_output_path}", flush=True)
            continue
        print(f"RUN {request_label_str} -> {norgate_indexname_str}", flush=True)
        try:
            result_dict = run_universe_backtest(
                request_label_str=request_label_str,
                norgate_indexname_str=norgate_indexname_str,
                sweep_output_path=sweep_output_path,
                output_dir_str=output_dir_str,
                show_progress_bool=show_progress_bool,
            )
        except Exception as exc:
            result_dict = {
                "request_label_str": request_label_str,
                "norgate_indexname_str": norgate_indexname_str,
                "error_str": f"{type(exc).__name__}: {exc}",
            }
            print(f"ERROR {request_label_str}: {result_dict['error_str']}", flush=True)
        result_dict_list.append(result_dict)
        completed_label_set.add(request_label_str)
        summary_df = pd.DataFrame(result_dict_list)
        summary_df.to_csv(sweep_output_path / "summary.csv", index=False)
        (sweep_output_path / "summary.json").write_text(
            json.dumps(result_dict_list, indent=2),
            encoding="utf-8",
        )
        if "report_path_str" in result_dict:
            _write_sweep_index_html(sweep_output_path, summary_df)

    if len(result_dict_list) > 0:
        _write_sweep_index_html(sweep_output_path, pd.DataFrame(result_dict_list))
    return sweep_output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--sweep-output-dir", default=None)
    parser.add_argument("--show-progress", action="store_true")
    arg_namespace = parser.parse_args()

    sweep_output_path = run_sweep(
        output_dir_str=arg_namespace.output_dir,
        show_progress_bool=arg_namespace.show_progress,
        sweep_output_dir_str=arg_namespace.sweep_output_dir,
    )
    print(f"Sweep saved to: {sweep_output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
