"""
VXX long-short term-structure strategy using a stitched old/new VXX series.

TL;DR: Trade a synthetic `VXX_COMBINED` sleeve that is:

    long  when VIX_t - VIX3M_t > diff
    short when VIX_t - VIX3M_t <= diff

but only when the regime flips, so the position is entered or reversed at the
next open and then left alone until the next regime flip.

Core formulas
-------------
At close t:

    term_spread_t
        = VIX_t - VIX3M_t

    backwardation_t
        = 1[term_spread_t > diff]

    regime_flip_t
        = 1[backwardation_t != backwardation_{t-1}]

    signed_regime_weight_t
        = w_alloc,   if backwardation_t = 1
        = -w_alloc,  otherwise

Only flip dates submit orders:

    entry_target_weight_t
        = signed_regime_weight_t * regime_flip_t

At the next open t + 1, if regime_flip_t = 1:

    q_{t+1}^{target_pct}
        = entry_target_weight_t

where:

    w_alloc = 1 / 3

The tradeable sleeve is stitched by date to mirror the source spec:

    price_t^{trade}
        = price_t^{VXX-201901}, if t < 2019-01-01
        = price_t^{VXX},        if t >= 2019-01-01

Research note
-------------
This stitch is a research convenience to reproduce the requested historical
logic across the matured pre-2019 VXX ETN and the later VXX series. It is not
claiming these are literally the same security.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


SIGNAL_NAMESPACE_STR = "signal_state"
STITCHED_TRADE_SYMBOL_STR = "VXX_COMBINED"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class VxxLongShortConfig:
    old_trade_symbol_str: str = "VXX-201901"
    new_trade_symbol_str: str = "VXX"
    trade_symbol_str: str = STITCHED_TRADE_SYMBOL_STR
    vix_symbol_str: str = "$VIX"
    vix3m_symbol_str: str = "$VIX3M"
    benchmark_symbol_str: str = "SPY"
    splice_date_str: str = "2019-01-01"
    diff_float: float = 0.0
    target_weight_float: float = 1.0 / 3.0
    start_date_str: str = "2019-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        symbol_set = {
            self.old_trade_symbol_str,
            self.new_trade_symbol_str,
            self.trade_symbol_str,
            self.vix_symbol_str,
            self.vix3m_symbol_str,
            self.benchmark_symbol_str,
        }
        if len(symbol_set) != 6:
            raise ValueError("All symbols must be distinct.")
        if not np.isfinite(self.diff_float):
            raise ValueError("diff_float must be finite.")
        if not np.isfinite(self.target_weight_float) or self.target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive and finite.")
        if self.target_weight_float > 1.0:
            raise ValueError("target_weight_float must be <= 1.0.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = VxxLongShortConfig()


def load_optional_symbol_price_df(
    symbol_str: str,
    start_date_str: str,
    end_date_str: str | None,
) -> pd.DataFrame:
    try:
        return load_raw_prices(
            symbols=[symbol_str],
            benchmarks=[],
            start_date=start_date_str,
            end_date=end_date_str,
        )
    except Exception:
        return pd.DataFrame()


def build_stitched_trade_price_df(
    pricing_data_df: pd.DataFrame,
    old_trade_symbol_str: str,
    new_trade_symbol_str: str,
    stitched_trade_symbol_str: str,
    splice_date_str: str,
) -> pd.DataFrame:
    """
    Build a synthetic tradeable price block that uses the old VXX series before
    the splice date and the new VXX series on and after the splice date.
    """
    if not isinstance(pricing_data_df.columns, pd.MultiIndex):
        raise ValueError("pricing_data_df must use MultiIndex columns.")

    splice_bar_ts = pd.Timestamp(splice_date_str)
    old_trade_column_mask = pricing_data_df.columns.get_level_values(0) == old_trade_symbol_str
    new_trade_column_mask = pricing_data_df.columns.get_level_values(0) == new_trade_symbol_str

    old_field_list = pricing_data_df.columns[old_trade_column_mask].get_level_values(1).tolist()
    new_field_list = pricing_data_df.columns[new_trade_column_mask].get_level_values(1).tolist()
    field_list = sorted(set(old_field_list) | set(new_field_list))

    if len(field_list) == 0:
        raise RuntimeError(
            f"No tradeable fields found for either {old_trade_symbol_str} or {new_trade_symbol_str}."
        )

    trade_index = pricing_data_df.index
    use_old_bool_ser = pd.Series(trade_index < splice_bar_ts, index=trade_index, dtype=bool)
    use_new_bool_ser = ~use_old_bool_ser

    stitched_field_ser_map: dict[tuple[str, str], pd.Series] = {}
    for field_str in field_list:
        old_field_ser = pricing_data_df.get(
            (old_trade_symbol_str, field_str),
            pd.Series(np.nan, index=trade_index, dtype=float),
        )
        new_field_ser = pricing_data_df.get(
            (new_trade_symbol_str, field_str),
            pd.Series(np.nan, index=trade_index, dtype=float),
        )

        stitched_field_ser = pd.Series(np.nan, index=trade_index, dtype=float, name=field_str)
        stitched_field_ser.loc[use_old_bool_ser] = old_field_ser.loc[use_old_bool_ser].astype(float)
        stitched_field_ser.loc[use_new_bool_ser] = new_field_ser.loc[use_new_bool_ser].astype(float)
        stitched_field_ser_map[(stitched_trade_symbol_str, field_str)] = stitched_field_ser

    stitched_trade_price_df = pd.DataFrame(stitched_field_ser_map, index=trade_index)
    stitched_trade_price_df.columns = pd.MultiIndex.from_tuples(stitched_trade_price_df.columns)
    return stitched_trade_price_df.sort_index(axis=1)


def get_vxx_long_short_data(
    config: VxxLongShortConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    required_data_df = load_raw_prices(
        symbols=[config.vix_symbol_str, config.vix3m_symbol_str],
        benchmarks=[config.benchmark_symbol_str],
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )

    optional_old_trade_df = load_optional_symbol_price_df(
        symbol_str=config.old_trade_symbol_str,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    optional_new_trade_df = load_optional_symbol_price_df(
        symbol_str=config.new_trade_symbol_str,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )

    if optional_old_trade_df.empty and optional_new_trade_df.empty:
        raise RuntimeError(
            f"No VXX trade data found for either {config.old_trade_symbol_str} or {config.new_trade_symbol_str}."
        )

    pricing_block_list = [required_data_df] + [
        df_part for df_part in (optional_old_trade_df, optional_new_trade_df) if not df_part.empty
    ]
    pricing_data_df = pd.concat(pricing_block_list, axis=1).sort_index()

    stitched_trade_price_df = build_stitched_trade_price_df(
        pricing_data_df=pricing_data_df,
        old_trade_symbol_str=config.old_trade_symbol_str,
        new_trade_symbol_str=config.new_trade_symbol_str,
        stitched_trade_symbol_str=config.trade_symbol_str,
        splice_date_str=config.splice_date_str,
    )
    return pd.concat([pricing_data_df, stitched_trade_price_df], axis=1).sort_index(axis=1)


def get_first_tradeable_bar_ts(
    pricing_data_df: pd.DataFrame,
    trade_symbol_str: str,
) -> pd.Timestamp:
    open_price_ser = pricing_data_df[(trade_symbol_str, "Open")].astype(float)
    valid_open_ser = open_price_ser.dropna()
    if len(valid_open_ser) == 0:
        raise RuntimeError(f"No valid open prices found for {trade_symbol_str}.")
    return pd.Timestamp(valid_open_ser.index[0])


def compute_vxx_long_short_signal_df(
    vix_close_ser: pd.Series,
    vix3m_close_ser: pd.Series,
    diff_float: float,
    target_weight_float: float,
) -> pd.DataFrame:
    """
    Compute the close-indexed VXX long/short regime-flip state.

        term_spread_t = VIX_t - VIX3M_t
        backwardation_t = 1[term_spread_t > diff]
        regime_flip_t = 1[backwardation_t != backwardation_{t-1}]
    """
    if not np.isfinite(diff_float):
        raise ValueError("diff_float must be finite.")
    if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
        raise ValueError("target_weight_float must be positive and finite.")

    vix_close_ser = pd.Series(vix_close_ser, copy=True).astype(float)
    vix3m_close_ser = pd.Series(vix3m_close_ser, copy=True).astype(float)

    signal_input_df = pd.concat(
        [
            vix_close_ser.rename("vix_close_ser"),
            vix3m_close_ser.rename("vix3m_close_ser"),
        ],
        axis=1,
    ).sort_index()

    term_spread_ser = signal_input_df["vix_close_ser"] - signal_input_df["vix3m_close_ser"]
    backwardation_bool_ser = term_spread_ser > float(diff_float)

    # *** CRITICAL*** This shift aligns backwardation_{t-1} against the current
    # close-bar regime so regime_flip_t uses only information available by the
    # close of bar t. The engine then executes any flip at open t + 1.
    prior_backwardation_obj_ser = backwardation_bool_ser.shift(1)
    prior_backwardation_bool_ser = pd.Series(False, index=signal_input_df.index, dtype=bool)
    valid_prior_bool_ser = prior_backwardation_obj_ser.notna()
    prior_backwardation_bool_ser.loc[valid_prior_bool_ser] = (
        prior_backwardation_obj_ser.loc[valid_prior_bool_ser].astype(bool)
    )

    regime_flip_bool_ser = pd.Series(False, index=signal_input_df.index, dtype=bool)
    regime_flip_bool_ser.loc[valid_prior_bool_ser] = (
        backwardation_bool_ser.loc[valid_prior_bool_ser]
        != prior_backwardation_bool_ser.loc[valid_prior_bool_ser]
    )

    signed_regime_weight_ser = pd.Series(
        np.where(backwardation_bool_ser, float(target_weight_float), -float(target_weight_float)),
        index=signal_input_df.index,
        dtype=float,
        name="signed_regime_weight_ser",
    )
    entry_target_weight_ser = pd.Series(
        np.where(regime_flip_bool_ser, signed_regime_weight_ser, 0.0),
        index=signal_input_df.index,
        dtype=float,
        name="entry_target_weight_ser",
    )

    signal_feature_df = pd.DataFrame(
        {
            "vix_close_ser": signal_input_df["vix_close_ser"],
            "vix3m_close_ser": signal_input_df["vix3m_close_ser"],
            "term_spread_ser": term_spread_ser,
            "backwardation_bool_ser": backwardation_bool_ser.astype(bool),
            "prior_backwardation_bool_ser": prior_backwardation_bool_ser,
            "regime_flip_bool_ser": regime_flip_bool_ser,
            "signed_regime_weight_ser": signed_regime_weight_ser,
            "entry_target_weight_ser": entry_target_weight_ser,
        },
        index=signal_input_df.index,
    )
    return signal_feature_df


class VxxLongShortStrategy(Strategy):
    """
    Single-asset long/short VXX sleeve driven by the VIX term structure.

        term_spread_t = VIX_t - VIX3M_t
        long if term_spread_t > diff
        short otherwise

    Orders are submitted only when the regime flips, matching the source
    strategy's event-driven semantics rather than a daily rebalance.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        trade_symbol_str: str,
        vix_symbol_str: str,
        vix3m_symbol_str: str,
        diff_float: float,
        target_weight_float: float,
        capital_base: float = 100_000.0,
        slippage: float = 0.001,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        if not trade_symbol_str or not vix_symbol_str or not vix3m_symbol_str:
            raise ValueError("Symbols must not be empty.")
        if not np.isfinite(diff_float):
            raise ValueError("diff_float must be finite.")
        if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive and finite.")
        if target_weight_float > 1.0:
            raise ValueError("target_weight_float must be <= 1.0.")

        self.trade_symbol_str = str(trade_symbol_str)
        self.vix_symbol_str = str(vix_symbol_str)
        self.vix3m_symbol_str = str(vix3m_symbol_str)
        self.diff_float = float(diff_float)
        self.target_weight_float = float(target_weight_float)
        self.trade_id_int = 0
        self.active_trade_id_int = default_trade_id_int()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        required_key_list = [
            (self.vix_symbol_str, "Close"),
            (self.vix3m_symbol_str, "Close"),
        ]
        missing_key_list = [key_tup for key_tup in required_key_list if key_tup not in pricing_data_df.columns]
        if len(missing_key_list) > 0:
            raise RuntimeError(f"Missing required signal columns: {missing_key_list}")

        signal_data_df = pricing_data_df.copy()
        signal_feature_df = compute_vxx_long_short_signal_df(
            vix_close_ser=signal_data_df[(self.vix_symbol_str, "Close")],
            vix3m_close_ser=signal_data_df[(self.vix3m_symbol_str, "Close")],
            diff_float=self.diff_float,
            target_weight_float=self.target_weight_float,
        )

        signal_feature_df.columns = pd.MultiIndex.from_tuples(
            [(SIGNAL_NAMESPACE_STR, field_str) for field_str in signal_feature_df.columns]
        )
        return pd.concat([signal_data_df, signal_feature_df], axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        regime_flip_key = (SIGNAL_NAMESPACE_STR, "regime_flip_bool_ser")
        entry_target_weight_key = (SIGNAL_NAMESPACE_STR, "entry_target_weight_ser")
        if regime_flip_key not in close_row_ser.index or entry_target_weight_key not in close_row_ser.index:
            return

        regime_flip_bool = bool(close_row_ser.loc[regime_flip_key])
        if not regime_flip_bool:
            return

        target_weight_float = float(close_row_ser.loc[entry_target_weight_key])
        if not np.isfinite(target_weight_float) or np.isclose(target_weight_float, 0.0, atol=1e-12):
            return

        open_price_float = float(open_price_ser.get(self.trade_symbol_str, np.nan))
        current_position_float = float(self.get_position(self.trade_symbol_str))
        if not np.isfinite(open_price_float) or open_price_float <= 0.0:
            if np.isclose(current_position_float, 0.0, atol=1e-12):
                return
            raise RuntimeError(f"Invalid open price for {self.trade_symbol_str} on {self.current_bar}.")

        sign_flip_bool = (
            not np.isclose(current_position_float, 0.0, atol=1e-12)
            and current_position_float * target_weight_float < 0.0
        )

        if sign_flip_bool:
            exit_trade_id_int = None if self.active_trade_id_int == default_trade_id_int() else self.active_trade_id_int
            self.order_target_value(
                self.trade_symbol_str,
                0.0,
                trade_id=exit_trade_id_int,
            )
            self.active_trade_id_int = default_trade_id_int()

        if (
            np.isclose(current_position_float, 0.0, atol=1e-12)
            or sign_flip_bool
            or self.active_trade_id_int == default_trade_id_int()
        ):
            self.trade_id_int += 1
            self.active_trade_id_int = self.trade_id_int

        # *** CRITICAL*** The order is submitted only on regime flips detected
        # from the previous close. The engine converts this target percent into
        # a next-open fill, preserving the causal timing contract.
        self.order_target_percent(
            self.trade_symbol_str,
            target_weight_float,
            trade_id=self.active_trade_id_int,
        )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = get_vxx_long_short_data(config=config)
    first_tradeable_bar_ts = get_first_tradeable_bar_ts(
        pricing_data_df=pricing_data_df,
        trade_symbol_str=config.trade_symbol_str,
    )
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= first_tradeable_bar_ts]

    strategy = VxxLongShortStrategy(
        name="strategy_mr_vxx_long_short",
        benchmarks=[config.benchmark_symbol_str],
        trade_symbol_str=config.trade_symbol_str,
        vix_symbol_str=config.vix_symbol_str,
        vix3m_symbol_str=config.vix3m_symbol_str,
        diff_float=config.diff_float,
        target_weight_float=config.target_weight_float,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )

    run_daily(
        strategy,
        pricing_data_df,
        calendar=calendar_idx,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
