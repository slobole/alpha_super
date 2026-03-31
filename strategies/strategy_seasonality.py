"""
Seasonality strategy.

Core formulas
-------------
Season state:

    season_state_t = 1[month(current_bar_t) in M]

Entry and exit timing:

    target_weight_map_t = W(month(current_bar_t))
    target_weight_map_{t-1} = W(month(previous_bar_t))

    entry_{a,t} = 1[w_{a,t-1} = 0 and w_{a,t} != 0]
    exit_{a,t} = 1[w_{a,t-1} != 0 and w_{a,t} = 0]
    resize_{a,t} = 1[w_{a,t-1} != w_{a,t}]
    hold_t = 1[target_weight_map_t = target_weight_map_{t-1}]

Signed target sizing inside an active month:

    q_{a,t}^{intent} ~= floor(V_{t-1} * w_{a,t} / O_{a,t})

For the current default map:

    month_target_weight_map = {
        1: {"GLD": 1.0},
        2: {"DBA": 1.0},
        3: {"XLU": 1.0},
        4: {"XLU": 1.0},
        7: {"XLU": 1.0},
        8: {"IEF": 1.0},
    }

    so:

    first_january_entry_t = 1[month(previous_bar_t) = 12 and month(current_bar_t) = 1]
    first_february_rotation_t = 1[month(previous_bar_t) = 1 and month(current_bar_t) = 2]
    first_march_rotation_t = 1[month(previous_bar_t) = 2 and month(current_bar_t) = 3]
    march_hold_t = 1[month(previous_bar_t) = 3 and month(current_bar_t) = 4]
    first_may_exit_t = 1[month(previous_bar_t) = 4 and month(current_bar_t) = 5]
    first_july_entry_t = 1[month(previous_bar_t) = 6 and month(current_bar_t) = 7]
    first_august_rotation_t = 1[month(previous_bar_t) = 7 and month(current_bar_t) = 8]
    first_september_exit_t = 1[month(previous_bar_t) = 8 and month(current_bar_t) = 9]

Implementation uses the engine's generic target helpers:

    entry_{a,t} -> order_target_percent(a, w_{a,t})
    resize_{a,t} -> order_target_percent(a, w_{a,t})
    exit_{a,t} -> order_target_value(a, 0.0)

Outside mapped months the strategy is fully in cash.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


DEFAULT_MONTH_TARGET_WEIGHT_MAP = {
    1: {"GLD": 1.0},
    2: {"DBA": 1.0},
    3: {"XLU": 1.0},
    4: {"XLU": 1.0},
    7: {"XLU": 1.0},
    8: {"IEF": 1.0},
}
DEFAULT_BENCHMARK_LIST = ("$SPX",)


def default_trade_id_int() -> int:
    return -1


def normalize_month_target_weight_map(
    month_target_weight_map: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    normalized_map: dict[int, dict[str, float]] = {}

    for month_int, target_weight_map_obj in month_target_weight_map.items():
        if month_int < 1 or month_int > 12:
            raise ValueError(f"Month key must be between 1 and 12. Found {month_int}.")

        target_weight_map = {
            str(asset_str): float(target_weight_float)
            for asset_str, target_weight_float in target_weight_map_obj.items()
        }
        if len(target_weight_map) == 0:
            raise ValueError(f"Month {month_int} must map to at least one asset weight.")
        if any(abs(target_weight_float) <= 0.0 for target_weight_float in target_weight_map.values()):
            raise ValueError(f"Month {month_int} contains a zero target weight, which is not allowed.")

        normalized_map[month_int] = target_weight_map

    return normalized_map


def get_prices(
    symbol_list: Sequence[str],
    benchmark_list: Sequence[str],
    start_date_str: str = "2004-01-01",
    end_date_str: str | None = None,
) -> pd.DataFrame:
    return load_raw_prices(list(symbol_list), list(benchmark_list), start_date_str, end_date_str)


class SeasonalityStrategy(Strategy):
    """
    Generic month-to-signed-target-weight seasonal pod.

    The strategy obeys the engine timing contract:

        decision at t uses information through previous_bar
        execution happens at the current bar open

    For the current default map:

        enter 100% GLD at the first tradable open of January
        rotate from 100% GLD into 100% DBA at the first tradable open of February
        rotate from 100% DBA into 100% XLU at the first tradable open of March
        keep holding 100% XLU through April
        exit XLU at the first tradable open of May
        re-enter 100% XLU at the first tradable open of July
        rotate from 100% XLU into 100% IEF at the first tradable open of August
        exit IEF at the first tradable open of September
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        month_target_weight_map: Mapping[int, Mapping[str, float]] | None = None,
        capital_base: float = 100_000,
        slippage: float = 0.0001,
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
        self.month_target_weight_map = normalize_month_target_weight_map(
            DEFAULT_MONTH_TARGET_WEIGHT_MAP if month_target_weight_map is None else month_target_weight_map
        )
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or self.previous_bar is None:
            return

        current_month_int = int(self.current_bar.month)
        previous_month_int = int(self.previous_bar.month)
        current_target_weight_map = self.month_target_weight_map.get(current_month_int, {})
        previous_target_weight_map = self.month_target_weight_map.get(previous_month_int, {})

        # *** CRITICAL*** The month-to-month season switch must use the
        # previous_bar/current_bar month boundary so fills occur at the first
        # tradable open of the new month rather than the prior close.
        season_change_bool = previous_target_weight_map != current_target_weight_map
        if not season_change_bool:
            return

        asset_union_list = sorted(set(previous_target_weight_map) | set(current_target_weight_map))

        # Submit liquidations first so same-open month rotations do not rely on
        # temporary leverage from the engine's order-processing sequence.
        for asset_str in asset_union_list:
            previous_target_weight_float = float(previous_target_weight_map.get(asset_str, 0.0))
            current_target_weight_float = float(current_target_weight_map.get(asset_str, 0.0))

            if abs(previous_target_weight_float - current_target_weight_float) <= 1e-12:
                continue

            current_position_float = float(self.get_position(asset_str))
            if abs(current_target_weight_float) <= 1e-12:
                if abs(current_position_float) <= 1e-12:
                    continue
                self.order_target_value(
                    asset_str,
                    0.0,
                    trade_id=self.current_trade_map[asset_str],
                )
        for asset_str in asset_union_list:
            previous_target_weight_float = float(previous_target_weight_map.get(asset_str, 0.0))
            current_target_weight_float = float(current_target_weight_map.get(asset_str, 0.0))

            if abs(previous_target_weight_float - current_target_weight_float) <= 1e-12:
                continue
            if abs(current_target_weight_float) <= 1e-12:
                continue

            current_position_float = float(self.get_position(asset_str))

            if (
                abs(previous_target_weight_float) <= 1e-12
                or current_position_float * current_target_weight_float <= 0.0
                or self.current_trade_map[asset_str] == default_trade_id_int()
            ):
                self.trade_id_int += 1
                self.current_trade_map[asset_str] = self.trade_id_int

            self.order_target_percent(
                asset_str,
                current_target_weight_float,
                trade_id=self.current_trade_map[asset_str],
            )


if __name__ == "__main__":
    symbol_list = sorted(
        {
            asset_str
            for target_weight_map in DEFAULT_MONTH_TARGET_WEIGHT_MAP.values()
            for asset_str in target_weight_map
        }
    )
    benchmark_list = list(DEFAULT_BENCHMARK_LIST)
    pricing_data_df = get_prices(
        symbol_list=symbol_list,
        benchmark_list=benchmark_list,
        start_date_str="2008-01-01",
        end_date_str=None,
    )

    strategy = SeasonalityStrategy(
        name="strategy_seasonality",
        benchmarks=benchmark_list,
        month_target_weight_map=DEFAULT_MONTH_TARGET_WEIGHT_MAP,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )

    calendar_idx = pricing_data_df.index
    run_daily(strategy, pricing_data_df, calendar_idx)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
