"""Compare a Scout spec's output with the real engine's output (design section 7.1).

Inputs are plain frames, so the comparator does not care how either side was produced:

    daily return series        (DatetimeIndex -> float)
    decision weight frames     (decision date x asset -> target weight after execution rounding)

Two tiers:

- **Parity tier (the gate).** Scout copies the engine's arithmetic, so it must match it exactly: the largest
  daily return difference and the largest daily weight difference must be <= 1e-9 on every common date, the set
  of trade dates must be identical, and the comparison must cover the engine's whole run (same first date, no
  missing engine date inside the span, at most 5 sessions missing at the end). The P3 review showed the looser
  tolerance tier below lets real look-ahead bugs through (membership read one session early: 4 bps, 99.7% cells).
- **Tolerance tier (informational, for truth mode and cross-vintage runs):**

| Check | Threshold |
|---|---|
| decision cells | |Δw| <= 0.5 pp on >= 99.5% of (decision, asset) cells |
| daily return correlation | >= 0.999 |
| annualised return difference | <= 5 bps |
| max drawdown difference | <= 0.25 pp |

Every mismatching cell is listed in the report; the gate never hides a difference behind the aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EXACT_TOLERANCE_FLOAT = 1e-9
TRAILING_GAP_SESSION_INT = 5
WEIGHT_TOLERANCE_FLOAT = 0.005
CELL_MATCH_SHARE_FLOAT = 0.995
DAILY_CORRELATION_FLOAT = 0.999
ANNUAL_RETURN_DIFFERENCE_FLOAT = 0.0005
MAX_DRAWDOWN_DIFFERENCE_FLOAT = 0.0025
TRADING_DAYS_PER_YEAR_INT = 252


def _annualized_return_float(return_ser: pd.Series) -> float:
    growth_float = float(np.prod(1.0 + return_ser.to_numpy()))
    return growth_float ** (TRADING_DAYS_PER_YEAR_INT / max(len(return_ser), 1)) - 1.0


def _max_drawdown_float(return_ser: pd.Series) -> float:
    equity_vec = np.cumprod(1.0 + return_ser.to_numpy())
    return float(np.min(equity_vec / np.maximum.accumulate(equity_vec) - 1.0))


@dataclass
class GateReport:
    passed_bool: bool
    check_dict: dict
    mismatch_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    note_list: list[str] = field(default_factory=list)

    def summary_str(self) -> str:
        line_list = [f"Identity gate: {'PASS' if self.passed_bool else 'FAIL'}"]
        for name_str, check in self.check_dict.items():
            line_list.append(f"  {'ok  ' if check['pass_bool'] else 'FAIL'} {name_str}: {check['value']} (limit {check['limit']})")
        if not self.mismatch_df.empty:
            line_list.append(f"  {len(self.mismatch_df)} mismatching cells in total (first 10 shown):")
            line_list.append(self.mismatch_df.head(10).to_string())
        line_list.extend(f"  note: {note_str}" for note_str in self.note_list)
        return "\n".join(line_list)


def compare_exact(
    engine_return_ser: pd.Series,
    scout_return_ser: pd.Series,
    engine_daily_weight_df: pd.DataFrame,
    scout_daily_weight_df: pd.DataFrame,
    engine_trade_date_index: pd.DatetimeIndex,
    scout_trade_date_index: pd.DatetimeIndex,
) -> GateReport:
    """Parity tier: exact match on every date, full coverage, identical trade dates."""
    engine_index, scout_index = engine_return_ser.index, scout_return_ser.index
    check_dict: dict = {}
    first_match_bool = len(scout_index) > 0 and scout_index[0] == engine_index[0]
    check_dict["same first date"] = {
        "value": f"engine {engine_index[0].date()} / scout {scout_index[0].date() if len(scout_index) else 'none'}",
        "limit": "equal", "pass_bool": first_match_bool,
    }
    inside_engine_index = engine_index[engine_index <= scout_index[-1]] if len(scout_index) else engine_index
    missing_inside_int = len(inside_engine_index.difference(scout_index))
    trailing_gap_int = int((engine_index > (scout_index[-1] if len(scout_index) else engine_index[-1])).sum())
    check_dict["coverage"] = {
        "value": f"{missing_inside_int} engine dates missing inside the span, {trailing_gap_int} at the end",
        "limit": f"0 inside, <= {TRAILING_GAP_SESSION_INT} at the end",
        "pass_bool": missing_inside_int == 0 and trailing_gap_int <= TRAILING_GAP_SESSION_INT,
    }
    common_index = engine_index.intersection(scout_index)
    return_gap_float = float((engine_return_ser.loc[common_index] - scout_return_ser.loc[common_index]).abs().max())
    check_dict["largest daily return difference"] = {
        "value": f"{return_gap_float:.2e}", "limit": f"<= {EXACT_TOLERANCE_FLOAT:.0e}",
        "pass_bool": bool(np.isfinite(return_gap_float) and return_gap_float <= EXACT_TOLERANCE_FLOAT),
    }
    weight_index = engine_daily_weight_df.index.intersection(scout_daily_weight_df.index).intersection(common_index)
    asset_index = engine_daily_weight_df.columns.union(scout_daily_weight_df.columns)
    weight_gap_df = (
        engine_daily_weight_df.reindex(index=weight_index, columns=asset_index).fillna(0.0)
        - scout_daily_weight_df.reindex(index=weight_index, columns=asset_index).fillna(0.0)
    ).abs()
    weight_gap_float = float(weight_gap_df.to_numpy().max()) if weight_gap_df.size else float("nan")
    check_dict["largest daily weight difference"] = {
        "value": f"{weight_gap_float:.2e} over {len(weight_index)} days", "limit": f"<= {EXACT_TOLERANCE_FLOAT:.0e}",
        "pass_bool": bool(np.isfinite(weight_gap_float) and weight_gap_float <= EXACT_TOLERANCE_FLOAT),
    }
    engine_trade_set = set(engine_trade_date_index[engine_trade_date_index <= common_index[-1]])
    scout_trade_set = set(scout_trade_date_index[scout_trade_date_index <= common_index[-1]])
    trade_date_difference_list = sorted(engine_trade_set.symmetric_difference(scout_trade_set))
    check_dict["same trade dates"] = {
        "value": f"{len(trade_date_difference_list)} dates differ", "limit": "0",
        "pass_bool": not trade_date_difference_list,
    }
    stacked_ser = (weight_gap_df > EXACT_TOLERANCE_FLOAT).stack()
    mismatch_index = stacked_ser[stacked_ser].index
    mismatch_df = pd.DataFrame(
        {"weight_difference_float": [weight_gap_df.at[date, asset] for date, asset in mismatch_index]}, index=mismatch_index
    )
    note_list = [f"Trade dates that differ: {[d.date() for d in trade_date_difference_list[:10]]}"] if trade_date_difference_list else []
    return GateReport(
        passed_bool=all(check["pass_bool"] for check in check_dict.values()),
        check_dict=check_dict, mismatch_df=mismatch_df, note_list=note_list,
    )


def compare(
    engine_return_ser: pd.Series,
    scout_return_ser: pd.Series,
    engine_weight_df: pd.DataFrame | None = None,
    scout_weight_df: pd.DataFrame | None = None,
) -> GateReport:
    """Tolerance tier (informational): every check on the common date range of the two return series."""
    note_list: list[str] = []
    common_index = engine_return_ser.index.intersection(scout_return_ser.index)
    if len(common_index) < TRADING_DAYS_PER_YEAR_INT:
        raise ValueError("Fewer than one year of common dates between engine and Scout returns.")
    for name_str, return_ser in (("engine", engine_return_ser), ("scout", scout_return_ser)):
        dropped_int = len(return_ser.index.difference(common_index))
        if dropped_int:
            note_list.append(f"{dropped_int} {name_str} dates outside the common range were not compared.")
    engine_ser = engine_return_ser.loc[common_index].astype(float)
    scout_ser = scout_return_ser.loc[common_index].astype(float)
    if engine_ser.isna().any() or scout_ser.isna().any():
        raise ValueError("Return series contain NaN on common dates.")

    correlation_float = float(np.corrcoef(engine_ser, scout_ser)[0, 1])
    annual_difference_float = abs(_annualized_return_float(engine_ser) - _annualized_return_float(scout_ser))
    drawdown_difference_float = abs(_max_drawdown_float(engine_ser) - _max_drawdown_float(scout_ser))
    check_dict = {
        "daily return correlation": {
            "value": round(correlation_float, 6), "limit": f">= {DAILY_CORRELATION_FLOAT}",
            "pass_bool": correlation_float >= DAILY_CORRELATION_FLOAT,
        },
        "annualised return difference": {
            "value": f"{annual_difference_float * 1e4:.2f} bps", "limit": f"<= {ANNUAL_RETURN_DIFFERENCE_FLOAT * 1e4:.0f} bps",
            "pass_bool": annual_difference_float <= ANNUAL_RETURN_DIFFERENCE_FLOAT,
        },
        "max drawdown difference": {
            "value": f"{drawdown_difference_float * 100:.3f} pp", "limit": f"<= {MAX_DRAWDOWN_DIFFERENCE_FLOAT * 100:.2f} pp",
            "pass_bool": drawdown_difference_float <= MAX_DRAWDOWN_DIFFERENCE_FLOAT,
        },
    }

    mismatch_df = pd.DataFrame()
    if engine_weight_df is not None and scout_weight_df is not None:
        date_index = engine_weight_df.index.union(scout_weight_df.index)
        asset_index = engine_weight_df.columns.union(scout_weight_df.columns)
        engine_cell_df = engine_weight_df.reindex(index=date_index, columns=asset_index).fillna(0.0)
        scout_cell_df = scout_weight_df.reindex(index=date_index, columns=asset_index).fillna(0.0)
        difference_df = (engine_cell_df - scout_cell_df).abs()
        # Only cells where at least one side holds or targets the asset count; all-zero cells would inflate the share.
        active_mask_df = (engine_cell_df.abs() > 0) | (scout_cell_df.abs() > 0)
        active_count_int = int(active_mask_df.to_numpy().sum())
        mismatch_mask_df = active_mask_df & (difference_df > WEIGHT_TOLERANCE_FLOAT)
        match_share_float = 1.0 - mismatch_mask_df.to_numpy().sum() / max(active_count_int, 1)
        check_dict["decision cells within 0.5 pp"] = {
            "value": f"{match_share_float * 100:.2f}% of {active_count_int}", "limit": f">= {CELL_MATCH_SHARE_FLOAT * 100:.1f}%",
            "pass_bool": match_share_float >= CELL_MATCH_SHARE_FLOAT,
        }
        stacked_ser = mismatch_mask_df.stack()
        mismatch_index = stacked_ser[stacked_ser].index
        mismatch_df = pd.DataFrame(
            {
                "engine_weight_float": [engine_cell_df.at[date, asset] for date, asset in mismatch_index],
                "scout_weight_float": [scout_cell_df.at[date, asset] for date, asset in mismatch_index],
            },
            index=mismatch_index,
        )
    else:
        note_list.append("No decision weights supplied: only the return checks ran.")

    return GateReport(
        passed_bool=all(check["pass_bool"] for check in check_dict.values()),
        check_dict=check_dict,
        mismatch_df=mismatch_df,
        note_list=note_list,
    )
