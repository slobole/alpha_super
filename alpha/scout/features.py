"""Feature library with declared contracts (design section 5).

Every time-series operation in Scout lives in a registered feature. A feature is a function of the panel that
returns a (date x symbol) frame, and it declares its contract:

- `lookback_int`: sessions of history it needs (the S1 prefix test starts after that);
- `basis_str`: "scale_invariant" (ratios, ranks, returns: unchanged when a future split rescales past prices) or
  "level" (uses adjusted price levels, e.g. a dollar threshold; S1 flags it, because a future corporate action
  changes its history);
- `description_str`.

*** CRITICAL*** A feature value at date t may use only rows dated <= t. S1 checks this automatically (prefix
invariance) for every feature a study uses; a feature that fails is never used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.scout.panel import Panel

BASIS_TUPLE = ("scale_invariant", "level")


@dataclass(frozen=True)
class Feature:
    name_str: str
    compute_fn: Callable[[Panel], pd.DataFrame]
    lookback_int: int
    basis_str: str
    description_str: str

    def __post_init__(self) -> None:
        if self.basis_str not in BASIS_TUPLE:
            raise ValueError(f"basis_str must be one of {BASIS_TUPLE}, got {self.basis_str!r}.")


def _per_symbol(panel: Panel, fn, *field_str_tuple) -> pd.DataFrame:
    column_dict = {}
    for symbol_str in panel.symbol_list:
        series_tuple = [panel.field(field_str)[symbol_str] for field_str in field_str_tuple]
        valid_mask = series_tuple[0].notna()
        if not valid_mask.any():
            column_dict[symbol_str] = pd.Series(np.nan, index=panel.date_index)
            continue
        # Indicators run on each symbol's own history: NaN rows (before listing, after delisting) are skipped; padded flat
        # bars inside a stock's life (0.2% of rows) stay in, then the result is realigned to the panel.
        result_ser = fn(*[series[valid_mask] for series in series_tuple])
        column_dict[symbol_str] = result_ser.reindex(panel.date_index)
    return pd.DataFrame(column_dict, index=panel.date_index)


def trailing_return(window_int: int) -> Feature:
    def compute(panel: Panel) -> pd.DataFrame:
        close_df = panel.field("Close")
        # *** CRITICAL*** backward-looking: Close(t) / Close(t − window) − 1.
        return close_df / close_df.shift(window_int) - 1.0

    return Feature(f"ret_{window_int}d", compute, window_int, "scale_invariant", f"{window_int}-session trailing return")


def close_over_sma(window_int: int) -> Feature:
    def compute(panel: Panel) -> pd.DataFrame:
        close_df = panel.field("Close")
        return close_df / close_df.rolling(window_int, min_periods=window_int).mean() - 1.0

    return Feature(f"close_over_sma{window_int}", compute, window_int, "scale_invariant", f"Close / SMA{window_int} − 1 (trailing)")


def qpi(window_int: int = 3, lookback_years_int: int = 5) -> Feature:
    from alpha.engine.qp_indicator_fast import qp_indicator_fast

    def compute(panel: Panel) -> pd.DataFrame:
        return _per_symbol(panel, lambda close_ser: qp_indicator_fast(close_ser, window_int, lookback_years_int), "Close")

    return Feature(
        f"qpi_{window_int}_{lookback_years_int}y", compute, window_int + 252 * lookback_years_int, "scale_invariant",
        "QPI: percent rank of the trailing return scaled by the down/up probability (alpha.engine.qp_indicator_fast)",
    )


def dv2(length_int: int = 126) -> Feature:
    from alpha.engine.dv2_indicator_fast import dv2_indicator_fast

    def compute(panel: Panel) -> pd.DataFrame:
        return _per_symbol(panel, lambda c, h, l: dv2_indicator_fast(c, h, l, length_int), "Close", "High", "Low")

    return Feature(f"dv2_{length_int}", compute, length_int + 2, "scale_invariant", "DV2 (alpha.engine.dv2_indicator_fast)")


def natr(window_int: int = 14) -> Feature:
    def compute(panel: Panel) -> pd.DataFrame:
        import talib

        return _per_symbol(
            panel, lambda c, h, l: pd.Series(talib.NATR(h.to_numpy(dtype=float), l.to_numpy(dtype=float), c.to_numpy(dtype=float), timeperiod=window_int), index=c.index),
            "Close", "High", "Low",
        )

    return Feature(f"natr_{window_int}", compute, 3 * window_int, "scale_invariant", "Normalised ATR (TA-Lib NATR, Wilder smoothing)")


def realized_volatility(window_int: int = 20) -> Feature:
    def compute(panel: Panel) -> pd.DataFrame:
        close_df = panel.field("Close")
        return (close_df / close_df.shift(1) - 1.0).rolling(window_int, min_periods=window_int).std() * np.sqrt(252.0)

    return Feature(f"vol_{window_int}d", compute, window_int + 1, "scale_invariant", f"{window_int}-session realised volatility")


def turnover_rank(window_int: int = 63) -> Feature:
    def compute(panel: Panel) -> pd.DataFrame:
        # Native Norgate Turnover (dollar volume, QUANT_PHILOSOPHY.md), trailing median, ranked across members that day.
        median_df = panel.field("Turnover").rolling(window_int, min_periods=window_int // 2).median()
        return median_df.where(panel.member_df == 1).rank(axis=1, pct=True)

    return Feature(f"turnover_rank_{window_int}d", compute, window_int, "scale_invariant", "Cross-sectional rank of trailing median dollar turnover")


def momentum_12_1() -> Feature:
    def compute(panel: Panel) -> pd.DataFrame:
        close_df = panel.field("Close")
        return close_df.shift(21) / close_df.shift(252) - 1.0

    return Feature("mom_12_1", compute, 252, "scale_invariant", "12-1 month momentum")


def reference_feature_list() -> list[Feature]:
    """Known features a new indicator is compared with in S2 (novelty check)."""
    return [
        trailing_return(1), trailing_return(3), trailing_return(5), momentum_12_1(),
        realized_volatility(20), natr(14), turnover_rank(63), close_over_sma(200),
    ]
