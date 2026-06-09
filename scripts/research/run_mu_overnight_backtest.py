"""
Research-only MU close-to-next-open overnight backtest.

The user-defined strategy is:

    buy MU at day T close
    sell MU at day T+1 open

Core formulas
-------------
For each completed overnight trade t:

    price_return_t = Open_{t+1} / Close_t - 1

    gross_return_t = (Open_{t+1} + Dividend_{t+1}) / Close_t - 1

With a per-side bps cost c:

    net_return_t =
        (Open_{t+1} * (1 - c) + Dividend_{t+1})
        / (Close_t * (1 + c)) - 1

where c = cost_bps_per_side / 10,000.

This is intentionally separate from the main Strategy engine because the
engine's normal contract is prior-bar decision to next-open execution. This
research path needs same-day close entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.report import build_research_output_path
from data.norgate_loader import CAPITALSPECIAL_ADJUSTMENT_STR, load_price_timeseries


DEFAULT_SYMBOL_STR = "MU"
DEFAULT_REFERENCE_CALENDAR_SYMBOL_STR = "$SPX"
DEFAULT_START_DATE_STR = "1998-01-01"
DEFAULT_CAPITAL_BASE_FLOAT = 100_000.0
DEFAULT_COST_BPS_PER_SIDE_TUPLE = (0.0, 2.5, 5.0, 10.0)


@dataclass(frozen=True)
class MuOvernightBacktestConfig:
    symbol_str: str = DEFAULT_SYMBOL_STR
    reference_calendar_symbol_str: str = DEFAULT_REFERENCE_CALENDAR_SYMBOL_STR
    start_date_str: str = DEFAULT_START_DATE_STR
    end_date_str: str | None = None
    capital_base_float: float = DEFAULT_CAPITAL_BASE_FLOAT
    primary_cost_bps_per_side_float: float = 0.0
    cost_bps_per_side_tuple: tuple[float, ...] = DEFAULT_COST_BPS_PER_SIDE_TUPLE

    def __post_init__(self) -> None:
        if not str(self.symbol_str).strip():
            raise ValueError("symbol_str must be non-empty.")
        if not str(self.reference_calendar_symbol_str).strip():
            raise ValueError("reference_calendar_symbol_str must be non-empty.")
        if self.capital_base_float <= 0.0 or not np.isfinite(self.capital_base_float):
            raise ValueError("capital_base_float must be positive and finite.")
        _validate_cost_bps_per_side_float(float(self.primary_cost_bps_per_side_float))
        if len(self.cost_bps_per_side_tuple) == 0:
            raise ValueError("cost_bps_per_side_tuple must not be empty.")
        for cost_bps_per_side_float in self.cost_bps_per_side_tuple:
            _validate_cost_bps_per_side_float(float(cost_bps_per_side_float))


@dataclass(frozen=True)
class MuOvernightBacktestResult:
    trade_detail_df: pd.DataFrame
    primary_summary_df: pd.DataFrame
    cost_sensitivity_df: pd.DataFrame
    output_dir_path: Path | None = None


def _validate_cost_bps_per_side_float(cost_bps_per_side_float: float) -> None:
    if not np.isfinite(cost_bps_per_side_float) or cost_bps_per_side_float < 0.0:
        raise ValueError("cost_bps_per_side_float must be non-negative and finite.")
    if cost_bps_per_side_float >= 10_000.0:
        raise ValueError("cost_bps_per_side_float must be below 10000 bps.")


def load_mu_price_df(config: MuOvernightBacktestConfig) -> pd.DataFrame:
    return load_price_timeseries(
        config.symbol_str,
        adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )


def load_reference_calendar_idx(config: MuOvernightBacktestConfig) -> pd.DatetimeIndex:
    calendar_price_df = load_price_timeseries(
        config.reference_calendar_symbol_str,
        adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    return pd.DatetimeIndex(calendar_price_df.index)


def _field_price_ser(
    ohlcv_price_df: pd.DataFrame,
    symbol_str: str,
    field_name_str: str,
) -> pd.Series:
    if isinstance(ohlcv_price_df.columns, pd.MultiIndex):
        key_tuple = (symbol_str, field_name_str)
        if key_tuple not in ohlcv_price_df.columns:
            raise RuntimeError(f"Missing required price column: {key_tuple}")
        return ohlcv_price_df.loc[:, key_tuple].astype(float)

    if field_name_str not in ohlcv_price_df.columns:
        raise RuntimeError(f"Missing required price column: {field_name_str}")
    return ohlcv_price_df.loc[:, field_name_str].astype(float)


def _optional_dividend_cash_ser(
    ohlcv_price_df: pd.DataFrame,
    symbol_str: str,
) -> pd.Series:
    if isinstance(ohlcv_price_df.columns, pd.MultiIndex):
        key_tuple = (symbol_str, "Dividend")
        if key_tuple in ohlcv_price_df.columns:
            return ohlcv_price_df.loc[:, key_tuple].astype(float).fillna(0.0)
        return pd.Series(0.0, index=ohlcv_price_df.index, name="dividend_cash_float")

    if "Dividend" in ohlcv_price_df.columns:
        return ohlcv_price_df.loc[:, "Dividend"].astype(float).fillna(0.0)
    return pd.Series(0.0, index=ohlcv_price_df.index, name="dividend_cash_float")


def _validate_expected_trading_dates(
    actual_date_idx: pd.DatetimeIndex,
    expected_trading_date_idx: pd.DatetimeIndex | None,
) -> None:
    if expected_trading_date_idx is None:
        return

    start_date = actual_date_idx[0].date()
    end_date = actual_date_idx[-1].date()
    expected_date_set = {
        pd.Timestamp(timestamp_obj).date()
        for timestamp_obj in expected_trading_date_idx
        if start_date <= pd.Timestamp(timestamp_obj).date() <= end_date
    }
    actual_date_set = {pd.Timestamp(timestamp_obj).date() for timestamp_obj in actual_date_idx}
    missing_date_list = sorted(expected_date_set - actual_date_set)
    extra_date_list = sorted(actual_date_set - expected_date_set)
    if len(missing_date_list) > 0:
        preview_str = ", ".join(date_obj.isoformat() for date_obj in missing_date_list[:5])
        raise RuntimeError(
            "ohlcv_price_df is missing expected trading session(s): "
            f"{preview_str}"
        )
    if len(extra_date_list) > 0:
        preview_str = ", ".join(date_obj.isoformat() for date_obj in extra_date_list[:5])
        raise RuntimeError(
            "ohlcv_price_df contains date(s) outside the expected trading calendar: "
            f"{preview_str}"
        )


def build_overnight_trade_detail_df(
    ohlcv_price_df: pd.DataFrame,
    *,
    symbol_str: str = DEFAULT_SYMBOL_STR,
    cost_bps_per_side_float: float = 0.0,
    expected_trading_date_idx: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    _validate_cost_bps_per_side_float(cost_bps_per_side_float)
    if not isinstance(ohlcv_price_df.index, pd.DatetimeIndex):
        raise RuntimeError("ohlcv_price_df must use a DatetimeIndex.")
    if not ohlcv_price_df.index.is_monotonic_increasing:
        raise RuntimeError("ohlcv_price_df index must be sorted in increasing date order.")
    if not ohlcv_price_df.index.is_unique:
        raise RuntimeError("ohlcv_price_df index must not contain duplicate dates.")
    _validate_expected_trading_dates(
        actual_date_idx=pd.DatetimeIndex(ohlcv_price_df.index),
        expected_trading_date_idx=expected_trading_date_idx,
    )

    open_price_ser = _field_price_ser(
        ohlcv_price_df=ohlcv_price_df,
        symbol_str=symbol_str,
        field_name_str="Open",
    )
    close_price_ser = _field_price_ser(
        ohlcv_price_df=ohlcv_price_df,
        symbol_str=symbol_str,
        field_name_str="Close",
    )
    dividend_cash_ser = _optional_dividend_cash_ser(
        ohlcv_price_df=ohlcv_price_df,
        symbol_str=symbol_str,
    )
    open_close_price_df = pd.DataFrame(
        {
            "open_price_float": open_price_ser,
            "close_price_float": close_price_ser,
            "dividend_cash_float": dividend_cash_ser,
        }
    )
    if open_close_price_df[["open_price_float", "close_price_float"]].isna().any().any():
        raise RuntimeError("Open and Close prices must not contain missing values.")
    if len(open_close_price_df) < 2:
        raise RuntimeError("At least two complete Open/Close bars are required.")
    if (open_close_price_df[["open_price_float", "close_price_float"]] <= 0.0).any().any():
        raise RuntimeError("Open and Close prices must be strictly positive.")
    if (open_close_price_df["dividend_cash_float"] < 0.0).any():
        raise RuntimeError("Dividend cash values must be non-negative.")

    entry_date_idx = pd.DatetimeIndex(open_close_price_df.index[:-1])
    exit_date_idx = pd.DatetimeIndex(open_close_price_df.index[1:])
    entry_close_price_vec = open_close_price_df["close_price_float"].iloc[:-1].to_numpy(
        dtype=float
    )
    exit_open_price_vec = open_close_price_df["open_price_float"].iloc[1:].to_numpy(
        dtype=float
    )
    exit_dividend_cash_vec = open_close_price_df["dividend_cash_float"].iloc[1:].to_numpy(
        dtype=float
    )
    cost_rate_float = float(cost_bps_per_side_float) * 0.0001

    # *** CRITICAL*** execution-timing boundary:
    # this research path pairs Close_t with Open_{t+1} plus any Dividend_{t+1}
    # cash to measure the realized overnight hold. Open_{t+1} is the exit
    # price after the close entry; it must not be used to decide whether to
    # enter at Close_t.
    price_only_overnight_return_vec = exit_open_price_vec / entry_close_price_vec - 1.0
    dividend_return_vec = exit_dividend_cash_vec / entry_close_price_vec
    gross_overnight_return_vec = (
        (exit_open_price_vec + exit_dividend_cash_vec) / entry_close_price_vec
        - 1.0
    )
    net_overnight_return_vec = (
        (exit_open_price_vec * (1.0 - cost_rate_float) + exit_dividend_cash_vec)
        / (entry_close_price_vec * (1.0 + cost_rate_float))
        - 1.0
    )

    trade_detail_df = pd.DataFrame(
        {
            "exit_date": exit_date_idx,
            "entry_close_price_float": entry_close_price_vec,
            "exit_open_price_float": exit_open_price_vec,
            "exit_dividend_cash_float": exit_dividend_cash_vec,
            "price_only_overnight_return_float": price_only_overnight_return_vec,
            "dividend_return_float": dividend_return_vec,
            "gross_overnight_return_float": gross_overnight_return_vec,
            "net_overnight_return_float": net_overnight_return_vec,
            "cost_drag_return_float": gross_overnight_return_vec
            - net_overnight_return_vec,
            "cost_bps_per_side_float": float(cost_bps_per_side_float),
        },
        index=entry_date_idx,
    )
    trade_detail_df.index.name = "entry_date"
    trade_detail_df["holding_period_days_int"] = (
        pd.DatetimeIndex(trade_detail_df["exit_date"]) - pd.DatetimeIndex(trade_detail_df.index)
    ).days.astype(int)
    return trade_detail_df


def build_equity_ser(
    trade_detail_df: pd.DataFrame,
    *,
    return_column_str: str,
    capital_base_float: float,
) -> pd.Series:
    daily_return_ser = trade_detail_df[return_column_str].astype(float)
    compounded_value_vec = np.cumprod(1.0 + daily_return_ser.to_numpy(dtype=float))
    equity_value_vec = np.concatenate(
        ([float(capital_base_float)], float(capital_base_float) * compounded_value_vec)
    )
    equity_date_idx = pd.DatetimeIndex(
        [trade_detail_df.index[0], *pd.DatetimeIndex(trade_detail_df["exit_date"]).tolist()]
    )
    return pd.Series(equity_value_vec, index=equity_date_idx, name="equity_float")


def summarize_trade_detail_dict(
    trade_detail_df: pd.DataFrame,
    *,
    return_column_str: str,
    capital_base_float: float,
) -> dict[str, Any]:
    daily_return_ser = trade_detail_df[return_column_str].astype(float)
    equity_ser = build_equity_ser(
        trade_detail_df=trade_detail_df,
        return_column_str=return_column_str,
        capital_base_float=capital_base_float,
    )
    elapsed_year_float = max(
        (pd.Timestamp(equity_ser.index[-1]) - pd.Timestamp(equity_ser.index[0])).days
        / 365.25,
        1.0 / 365.25,
    )
    total_return_float = float(equity_ser.iloc[-1] / float(capital_base_float) - 1.0)
    cagr_float = float((1.0 + total_return_float) ** (1.0 / elapsed_year_float) - 1.0)
    annual_volatility_float = float(daily_return_ser.std(ddof=1) * np.sqrt(252.0))
    if annual_volatility_float == 0.0 or not np.isfinite(annual_volatility_float):
        sharpe_float = np.nan
    else:
        sharpe_float = float(daily_return_ser.mean() / daily_return_ser.std(ddof=1) * np.sqrt(252.0))

    # *** CRITICAL*** drawdown uses only the realized equity path up to each
    # exit date: drawdown_t = equity_t / max(equity_0, ..., equity_t) - 1.
    drawdown_ser = equity_ser / equity_ser.cummax() - 1.0
    max_drawdown_float = float(drawdown_ser.min())
    mar_float = (
        float(cagr_float / abs(max_drawdown_float))
        if max_drawdown_float < 0.0 and np.isfinite(cagr_float)
        else np.nan
    )

    return {
        "start_entry_date_str": pd.Timestamp(trade_detail_df.index[0]).date().isoformat(),
        "end_exit_date_str": pd.Timestamp(trade_detail_df["exit_date"].iloc[-1]).date().isoformat(),
        "trade_count_int": int(len(trade_detail_df)),
        "final_equity_float": float(equity_ser.iloc[-1]),
        "total_return_float": total_return_float,
        "cagr_float": cagr_float,
        "annual_volatility_float": annual_volatility_float,
        "sharpe_float": sharpe_float,
        "max_drawdown_float": max_drawdown_float,
        "mar_float": mar_float,
        "win_rate_float": float((daily_return_ser > 0.0).mean()),
        "average_trade_return_float": float(daily_return_ser.mean()),
        "median_trade_return_float": float(daily_return_ser.median()),
        "best_trade_return_float": float(daily_return_ser.max()),
        "worst_trade_return_float": float(daily_return_ser.min()),
        "round_trips_per_year_float": 252.0,
        "notional_turnover_x_per_year_float": 504.0,
    }


def build_cost_sensitivity_df(
    ohlcv_price_df: pd.DataFrame,
    *,
    config: MuOvernightBacktestConfig,
    expected_trading_date_idx: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    row_dict_list: list[dict[str, Any]] = []
    for cost_bps_per_side_float in config.cost_bps_per_side_tuple:
        trade_detail_df = build_overnight_trade_detail_df(
            ohlcv_price_df=ohlcv_price_df,
            symbol_str=config.symbol_str,
            cost_bps_per_side_float=float(cost_bps_per_side_float),
            expected_trading_date_idx=expected_trading_date_idx,
        )
        metric_dict = summarize_trade_detail_dict(
            trade_detail_df=trade_detail_df,
            return_column_str="net_overnight_return_float",
            capital_base_float=config.capital_base_float,
        )
        row_dict_list.append(
            {
                "symbol_str": config.symbol_str,
                "cost_bps_per_side_float": float(cost_bps_per_side_float),
                **metric_dict,
            }
        )
    return pd.DataFrame(row_dict_list)


def compute_buy_hold_total_return_float(trade_detail_df: pd.DataFrame) -> float:
    first_entry_close_price_float = float(trade_detail_df["entry_close_price_float"].iloc[0])
    final_exit_open_price_float = float(trade_detail_df["exit_open_price_float"].iloc[-1])
    cumulative_dividend_cash_float = float(trade_detail_df["exit_dividend_cash_float"].sum())
    return (
        (final_exit_open_price_float + cumulative_dividend_cash_float)
        / first_entry_close_price_float
        - 1.0
    )


def _json_default(value_obj: Any) -> Any:
    if isinstance(value_obj, Path):
        return str(value_obj)
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.isoformat()
    if isinstance(value_obj, np.integer):
        return int(value_obj)
    if isinstance(value_obj, np.floating):
        if np.isnan(float(value_obj)):
            return None
        return float(value_obj)
    if isinstance(value_obj, np.ndarray):
        return value_obj.tolist()
    return value_obj


def _write_report_markdown(
    output_dir_path: Path,
    *,
    config: MuOvernightBacktestConfig,
    primary_summary_df: pd.DataFrame,
    cost_sensitivity_df: pd.DataFrame,
    buy_hold_total_return_float: float,
) -> None:
    primary_summary_ser = primary_summary_df.iloc[0]
    report_cost_sensitivity_df = cost_sensitivity_df[
        [
            "cost_bps_per_side_float",
            "final_equity_float",
            "cagr_float",
            "sharpe_float",
            "max_drawdown_float",
            "win_rate_float",
        ]
    ]
    report_line_list = [
        "# MU Overnight Close-To-Open Research Backtest",
        "",
        "TL;DR: This is a research-only test of buying MU at each close and selling at the next open.",
        "",
        "## Timing",
        "",
        "```text",
        "Close(T) entry -> Open(T+1) exit",
        "gross_return_t = (Open_{t+1} + Dividend_{t+1}) / Close_t - 1",
        "```",
        "",
        "## Assumptions",
        "",
        f"- Symbol: `{config.symbol_str}`",
        f"- Reference calendar symbol: `{config.reference_calendar_symbol_str}`",
        "- Price adjustment: `CAPITALSPECIAL` fills plus ordinary cash dividends from `Dividend`.",
        f"- Start date request: `{config.start_date_str}`",
        f"- End date request: `{config.end_date_str}`",
        f"- Capital base: `{config.capital_base_float:,.2f}`",
        "- Position: 100% of equity overnight on every completed trade.",
        "- Intraday exposure: flat from the open exit until the close entry.",
        "- Live caveat: this assumes close-auction entry and does not use the repo's normal next-open engine.",
        "",
        "## Primary Row",
        "",
        f"- Cost bps per side: `{float(primary_summary_ser['cost_bps_per_side_float']):.4f}`",
        f"- Trade count: `{int(primary_summary_ser['trade_count_int'])}`",
        f"- Final equity: `{float(primary_summary_ser['final_equity_float']):,.2f}`",
        f"- CAGR: `{float(primary_summary_ser['cagr_float']):.4%}`",
        f"- Sharpe: `{float(primary_summary_ser['sharpe_float']):.4f}`",
        f"- Max drawdown: `{float(primary_summary_ser['max_drawdown_float']):.4%}`",
        f"- Win rate: `{float(primary_summary_ser['win_rate_float']):.4%}`",
        "- Same-window MU buy-and-hold cash-dividend return, no dividend reinvestment: "
        f"`{buy_hold_total_return_float:.4%}`",
        "",
        "## Cost Sensitivity",
        "",
        _markdown_table_str(report_cost_sensitivity_df),
        "",
    ]
    (output_dir_path / "README.md").write_text("\n".join(report_line_list), encoding="utf-8")


def _markdown_table_str(table_df: pd.DataFrame) -> str:
    header_list = [str(column_obj) for column_obj in table_df.columns]
    line_list = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join(["---"] * len(header_list)) + " |",
    ]
    for _row_idx, row_ser in table_df.iterrows():
        cell_str_list = []
        for value_obj in row_ser:
            if isinstance(value_obj, (float, np.floating)):
                cell_str_list.append(f"{float(value_obj):.6g}")
            else:
                cell_str_list.append(str(value_obj))
        line_list.append("| " + " | ".join(cell_str_list) + " |")
    return "\n".join(line_list)


def _save_artifacts(
    *,
    output_dir_path: Path,
    config: MuOvernightBacktestConfig,
    trade_detail_df: pd.DataFrame,
    primary_summary_df: pd.DataFrame,
    cost_sensitivity_df: pd.DataFrame,
    buy_hold_total_return_float: float,
) -> None:
    output_dir_path.mkdir(parents=True, exist_ok=False)
    trade_detail_df.to_csv(output_dir_path / "trades.csv", float_format="%.10f")
    primary_summary_df.to_csv(output_dir_path / "summary.csv", index=False, float_format="%.10f")
    cost_sensitivity_df.to_csv(
        output_dir_path / "cost_sensitivity.csv",
        index=False,
        float_format="%.10f",
    )
    run_info_dict = {
        "entity_type": "strategy",
        "entity_id": f"{config.symbol_str.lower()}_overnight_close_to_open",
        "analysis_type": "research_backtest",
            "parameters": {
                "symbol_str": config.symbol_str,
                "reference_calendar_symbol_str": config.reference_calendar_symbol_str,
                "start_date_str": config.start_date_str,
            "end_date_str": config.end_date_str,
            "capital_base_float": config.capital_base_float,
            "primary_cost_bps_per_side_float": config.primary_cost_bps_per_side_float,
            "cost_bps_per_side_tuple": config.cost_bps_per_side_tuple,
            "price_adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
            "buy_hold_total_return_float": buy_hold_total_return_float,
        },
    }
    (output_dir_path / "run_info.json").write_text(
        json.dumps(run_info_dict, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _write_report_markdown(
        output_dir_path=output_dir_path,
        config=config,
        primary_summary_df=primary_summary_df,
        cost_sensitivity_df=cost_sensitivity_df,
        buy_hold_total_return_float=buy_hold_total_return_float,
    )


def run_mu_overnight_backtest(
    *,
    config: MuOvernightBacktestConfig = MuOvernightBacktestConfig(),
    ohlcv_price_df: pd.DataFrame | None = None,
    expected_trading_date_idx: pd.DatetimeIndex | None = None,
    output_dir_str: str = "results",
    save_results_bool: bool = True,
) -> MuOvernightBacktestResult:
    if ohlcv_price_df is None:
        ohlcv_price_df = load_mu_price_df(config)
        if expected_trading_date_idx is None:
            expected_trading_date_idx = load_reference_calendar_idx(config)

    trade_detail_df = build_overnight_trade_detail_df(
        ohlcv_price_df=ohlcv_price_df,
        symbol_str=config.symbol_str,
        cost_bps_per_side_float=config.primary_cost_bps_per_side_float,
        expected_trading_date_idx=expected_trading_date_idx,
    )
    primary_metric_dict = summarize_trade_detail_dict(
        trade_detail_df=trade_detail_df,
        return_column_str="net_overnight_return_float",
        capital_base_float=config.capital_base_float,
    )
    buy_hold_total_return_float = compute_buy_hold_total_return_float(trade_detail_df)
    primary_summary_df = pd.DataFrame(
        [
            {
                "symbol_str": config.symbol_str,
                "cost_bps_per_side_float": float(config.primary_cost_bps_per_side_float),
                "buy_hold_total_return_float": buy_hold_total_return_float,
                **primary_metric_dict,
            }
        ]
    )
    cost_sensitivity_df = build_cost_sensitivity_df(
        ohlcv_price_df=ohlcv_price_df,
        config=config,
        expected_trading_date_idx=expected_trading_date_idx,
    )

    output_dir_path: Path | None = None
    if save_results_bool:
        output_dir_path = build_research_output_path(
            output_dir=output_dir_str,
            entity_type_str="strategy",
            entity_id_str=f"{config.symbol_str.lower()}_overnight_close_to_open",
            analysis_type_str="research_backtest",
        )
        _save_artifacts(
            output_dir_path=output_dir_path,
            config=config,
            trade_detail_df=trade_detail_df,
            primary_summary_df=primary_summary_df,
            cost_sensitivity_df=cost_sensitivity_df,
            buy_hold_total_return_float=buy_hold_total_return_float,
        )

    return MuOvernightBacktestResult(
        trade_detail_df=trade_detail_df,
        primary_summary_df=primary_summary_df,
        cost_sensitivity_df=cost_sensitivity_df,
        output_dir_path=output_dir_path,
    )


def _cost_bps_per_side_tuple(raw_cost_list: Sequence[float] | None) -> tuple[float, ...]:
    if raw_cost_list is None or len(raw_cost_list) == 0:
        return DEFAULT_COST_BPS_PER_SIDE_TUPLE
    return tuple(float(cost_bps_per_side_float) for cost_bps_per_side_float in raw_cost_list)


def main() -> int:
    parser_obj = argparse.ArgumentParser(
        description="Run the MU close-to-next-open overnight research backtest.",
    )
    parser_obj.add_argument("--symbol", default=DEFAULT_SYMBOL_STR)
    parser_obj.add_argument("--reference-calendar-symbol", default=DEFAULT_REFERENCE_CALENDAR_SYMBOL_STR)
    parser_obj.add_argument("--start-date", default=DEFAULT_START_DATE_STR)
    parser_obj.add_argument("--end-date", default=None)
    parser_obj.add_argument("--capital-base", type=float, default=DEFAULT_CAPITAL_BASE_FLOAT)
    parser_obj.add_argument(
        "--primary-cost-bps-per-side",
        type=float,
        default=0.0,
        help="Per-side bps cost for the primary summary row. Default is gross before costs.",
    )
    parser_obj.add_argument(
        "--cost-sweep-bps-per-side",
        action="append",
        type=float,
        default=None,
        help="Per-side bps cost sensitivity row. Repeat for multiple rows.",
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--no-save", action="store_true")
    args_obj = parser_obj.parse_args()

    config = MuOvernightBacktestConfig(
        symbol_str=str(args_obj.symbol).upper(),
        reference_calendar_symbol_str=str(args_obj.reference_calendar_symbol).upper(),
        start_date_str=args_obj.start_date,
        end_date_str=args_obj.end_date,
        capital_base_float=float(args_obj.capital_base),
        primary_cost_bps_per_side_float=float(args_obj.primary_cost_bps_per_side),
        cost_bps_per_side_tuple=_cost_bps_per_side_tuple(args_obj.cost_sweep_bps_per_side),
    )
    result_obj = run_mu_overnight_backtest(
        config=config,
        output_dir_str=args_obj.output_dir,
        save_results_bool=not bool(args_obj.no_save),
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("Primary summary")
    print(result_obj.primary_summary_df.to_string(index=False))
    print("")
    print("Cost sensitivity")
    print(result_obj.cost_sensitivity_df.to_string(index=False))
    if result_obj.output_dir_path is not None:
        print(f"\nSaved research artifacts to: {result_obj.output_dir_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
