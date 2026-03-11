"""
Defense First VIX Tranche with BTAL and SPY fallback.

Strategy Rules:
- Defensive assets: GLD, TLT, UUP, DBC, BTAL
- Fallback risk-on asset: SPY
- Fallback risk-off state: CASH
- Monthly rebalancing at the first trading day of each month
- Defensive signal: average of 1m, 3m, 6m, 12m returns
- Absolute momentum filter: keep only assets beating T-bills
- Rank weights: [5, 4, 3, 2, 1] / 15 to top momentum survivors
- VIX tranche logic applies only to leftover fallback weight
- Benchmark: $SPX

Data Requirements:
- Norgate Data subscription
- DTB3 historical rates CSV
- Internet access for FRED VIXCLS data
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import norgatedata
import numpy as np
import pandas as pd
from IPython.display import display
from pandas_datareader import data as web
from tqdm.auto import tqdm

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy


@dataclass(frozen=True)
class DefenseFirstVixTrancheBtalConfig:
    defensive_asset_list: tuple[str, ...] = ("GLD", "UUP", "TLT", "DBC", "BTAL")
    signal_asset_list: tuple[str, ...] = ("SPY", "GLD", "UUP", "TLT", "DBC", "BTAL")
    fallback_asset: str = "SPY"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    rank_weight_vec: tuple[float, ...] = (5.0 / 15.0, 4.0 / 15.0, 3.0 / 15.0, 2.0 / 15.0, 1.0 / 15.0)
    vol_lookback_day_vec: tuple[int, ...] = (10, 15, 20)
    start_date_str: str = "2011-01-01"
    end_date_str: str | None = None
    dtb3_csv_path_str: str = r"C:\Users\User\Documents\workspace\1_data\DTB3.csv"

    def __post_init__(self):
        if len(self.rank_weight_vec) != len(self.defensive_asset_list):
            raise ValueError("rank_weight_vec length must match defensive_asset_list length.")
        if not np.isclose(sum(self.rank_weight_vec), 1.0, atol=1e-12):
            raise ValueError("rank_weight_vec must sum to 1.0.")

    @property
    def tradeable_asset_list(self) -> tuple[str, ...]:
        return self.defensive_asset_list + (self.fallback_asset,)


BTAL_VIX_SPY_CONFIG = DefenseFirstVixTrancheBtalConfig(fallback_asset="SPY")


def default_trade_id_int() -> int:
    return -1


class DefenseFirstVixTrancheBtalStrategy(Strategy):
    """
    Defense First TAA strategy with BTAL and multi-lookback VIX tranche scaling.

    Defensive allocations are assigned first. Only the leftover remainder weight
    is scaled into the fallback asset or left in cash.
    """

    def __init__(
        self,
        name: str,
        benchmarks: list | tuple,
        monthly_weight_df: pd.DataFrame,
        tradeable_asset_list: list[str] | tuple[str, ...],
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
        self.monthly_weight_df = monthly_weight_df.copy()
        self.tradeable_asset_list = list(tradeable_asset_list)
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.last_rebalance_month_tuple: tuple[int, int] | None = None

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        if close is None:
            return

        current_month_tuple = (self.current_bar.year, self.current_bar.month)
        if current_month_tuple == self.last_rebalance_month_tuple:
            return

        valid_rebalance_date_idx = self.monthly_weight_df.index[self.monthly_weight_df.index <= self.current_bar]
        if len(valid_rebalance_date_idx) == 0:
            return

        target_weight_ser = self.monthly_weight_df.loc[valid_rebalance_date_idx[-1]].fillna(0.0)
        self.last_rebalance_month_tuple = current_month_tuple

        for asset_str in self.tradeable_asset_list:
            target_weight_float = float(target_weight_ser.get(asset_str, 0.0))
            current_position_float = float(self.get_position(asset_str))
            if target_weight_float == 0.0 and current_position_float != 0.0:
                self.order_target_value(asset_str, 0.0, trade_id=self.current_trade_map[asset_str])

        for asset_str in self.tradeable_asset_list:
            target_weight_float = float(target_weight_ser.get(asset_str, 0.0))
            if target_weight_float <= 0.0:
                continue

            current_position_float = float(self.get_position(asset_str))
            if current_position_float == 0.0:
                self.trade_id_int += 1
                self.current_trade_map[asset_str] = self.trade_id_int

            self.order_target_percent(asset_str, target_weight_float, trade_id=self.current_trade_map[asset_str])



def load_tbills_ser(dtb3_csv_path_str: str) -> pd.Series:
    tbills_df = pd.read_csv(dtb3_csv_path_str)
    date_col = next(col for col in tbills_df.columns if "date" in col.lower() or col == "DATE")
    value_col = next(col for col in tbills_df.columns if col != date_col)

    tbills_ser = tbills_df.set_index(date_col)[value_col]
    tbills_ser.index = pd.to_datetime(tbills_ser.index)
    tbills_ser = pd.to_numeric(tbills_ser, errors="coerce")
    tbills_ser = (1.0 + tbills_ser / 100.0) ** (1.0 / 12.0) - 1.0
    tbills_ser.name = "DTB3"
    return tbills_ser.sort_index()



def load_vix_close_ser(start_date_str: str, end_date_str: str | None) -> pd.Series:
    vix_df = web.DataReader("VIXCLS", "fred", start=start_date_str, end=end_date_str)
    vix_close_ser = pd.to_numeric(vix_df["VIXCLS"], errors="coerce")
    vix_close_ser.name = "VIXCLS"
    return vix_close_ser.sort_index()



def get_prices_and_weights(
    config: DefenseFirstVixTrancheBtalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    pricing_frame_list: list[pd.DataFrame] = []
    close_map: dict[str, pd.Series] = {}
    symbol_list = list(dict.fromkeys(list(config.signal_asset_list) + [config.fallback_asset] + list(config.benchmark_list)))

    for symbol_str in tqdm(symbol_list, desc="loading price data"):
        if symbol_str in config.benchmark_list:
            adjustment_type = norgatedata.StockPriceAdjustmentType.TOTALRETURN
        else:
            adjustment_type = norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL

        price_df = norgatedata.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=adjustment_type,
            padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
            start_date=config.start_date_str,
            end_date=config.end_date_str,
            timeseriesformat="pandas-dataframe",
        )
        if len(price_df) == 0:
            continue

        if symbol_str in config.signal_asset_list:
            close_map[symbol_str] = price_df["Close"].copy()

        if symbol_str != "SPY":
            price_df.columns = pd.MultiIndex.from_tuples([(symbol_str, field_str) for field_str in price_df.columns])
            pricing_frame_list.append(price_df)

    if len(pricing_frame_list) == 0:
        raise RuntimeError("No pricing data was loaded.")

    pricing_data_df = pd.concat(pricing_frame_list, axis=1).sort_index()
    close_df = pd.DataFrame(close_map).sort_index()
    tbills_ser = load_tbills_ser(config.dtb3_csv_path_str)
    vix_close_ser = load_vix_close_ser(config.start_date_str, config.end_date_str)

    score_ser_list: list[pd.Series] = []
    for asset_str in config.signal_asset_list:
        close_ser = close_df[asset_str]
        momentum_1m_ser = close_ser.pct_change(21, fill_method=None)
        momentum_3m_ser = close_ser.pct_change(21 * 3, fill_method=None)
        momentum_6m_ser = close_ser.pct_change(21 * 6, fill_method=None)
        momentum_12m_ser = close_ser.pct_change(21 * 12, fill_method=None)
        score_ser = (momentum_1m_ser + momentum_3m_ser + momentum_6m_ser + momentum_12m_ser) / 4.0
        score_ser.name = asset_str
        score_ser_list.append(score_ser)

    score_df = pd.concat(score_ser_list, axis=1).sort_index()
    trading_calendar_idx = score_df.index
    spy_return_ser = close_df["SPY"].pct_change(fill_method=None)
    vix_close_aligned_ser = vix_close_ser.reindex(trading_calendar_idx).ffill()
    vix_close_aligned_ser.name = "VIXCLS"

    realized_volatility_ser_list: list[pd.Series] = []
    safe_mask_ser_list: list[pd.Series] = []
    for lookback_day_int in config.vol_lookback_day_vec:
        realized_volatility_ser = spy_return_ser.rolling(lookback_day_int).std() * np.sqrt(252.0) * 100.0
        realized_volatility_ser.name = f"rv_{lookback_day_int}d"
        realized_volatility_ser_list.append(realized_volatility_ser)

        # *** CRITICAL*** Compare lagged RV to lagged VIX so the month-end signal
        # uses only information available before the next rebalance.
        safe_mask_ser = realized_volatility_ser.shift(1) < vix_close_aligned_ser.shift(1)
        safe_mask_ser.name = f"safe_{lookback_day_int}d"
        safe_mask_ser_list.append(safe_mask_ser)

    realized_volatility_df = pd.concat(realized_volatility_ser_list, axis=1)
    safe_mask_df = pd.concat(safe_mask_ser_list, axis=1)
    safe_count_ser = safe_mask_df.sum(axis=1).rename("safe_count")

    signal_df = pd.concat([score_df, tbills_ser, safe_count_ser], axis=1).sort_index().dropna()
    month_end_signal_df = signal_df.resample("ME").last()

    monthly_weight_df = pd.DataFrame(0.0, index=trading_calendar_idx, columns=list(config.tradeable_asset_list), dtype=float)
    fallback_scale_map: dict[int, float] = {0: 0.0, 1: 1.0 / 3.0, 2: 2.0 / 3.0, 3: 1.0}

    for decision_date, signal_row_ser in tqdm(month_end_signal_df.iterrows(), total=len(month_end_signal_df), desc="computing monthly weights"):
        defensive_score_ser = signal_row_ser[list(config.defensive_asset_list)].astype(float)
        eligible_asset_ser = defensive_score_ser[defensive_score_ser > float(signal_row_ser["DTB3"])].sort_values(ascending=False)

        assigned_count_int = len(eligible_asset_ser)
        target_weight_ser = pd.Series(0.0, index=list(config.tradeable_asset_list), dtype=float)
        if assigned_count_int > 0:
            rank_weight_vec = np.array(config.rank_weight_vec[:assigned_count_int], dtype=float)
            target_weight_ser.loc[eligible_asset_ser.index] = rank_weight_vec

        unassigned_weight_float = 1.0 - float(target_weight_ser.sum())
        safe_count_int = int(signal_row_ser["safe_count"])
        fallback_weight_float = unassigned_weight_float * fallback_scale_map[safe_count_int]

        if fallback_weight_float > 0.0:
            target_weight_ser.loc[config.fallback_asset] = fallback_weight_float

        next_month_end_dt = decision_date + pd.offsets.MonthEnd(1)
        next_month_mask = (monthly_weight_df.index > decision_date) & (monthly_weight_df.index <= next_month_end_dt)
        monthly_weight_df.loc[next_month_mask, target_weight_ser.index] = target_weight_ser.values

    monthly_weight_df = monthly_weight_df.fillna(0.0)
    active_mask = monthly_weight_df.sum(axis=1) > 0.0
    if active_mask.any():
        monthly_weight_df = monthly_weight_df.loc[monthly_weight_df.index[active_mask][0] :]

    return pricing_data_df, monthly_weight_df, score_df, vix_close_aligned_ser, safe_count_ser, realized_volatility_df


if __name__ == "__main__":
    config = BTAL_VIX_SPY_CONFIG

    pricing_data_df, monthly_weight_df, score_df, vix_close_ser, safe_count_ser, realized_volatility_df = get_prices_and_weights(config)

    calendar_idx = pricing_data_df.index
    weight_start_dt = monthly_weight_df.index[0]

    data_start_dt_list: list[pd.Timestamp] = []
    for asset_str in config.tradeable_asset_list:
        close_col = (asset_str, "Close")
        if close_col in pricing_data_df.columns:
            valid_close_ser = pricing_data_df[close_col].dropna()
            if len(valid_close_ser) > 0:
                data_start_dt_list.append(valid_close_ser.index[0])

    calendar_start_dt = max(weight_start_dt, max(data_start_dt_list)) if len(data_start_dt_list) > 0 else weight_start_dt
    calendar_idx = calendar_idx[calendar_idx >= calendar_start_dt]

    strategy = DefenseFirstVixTrancheBtalStrategy(
        name="DefenseFirstTAAVixTranche_Btal_SPY",
        benchmarks=config.benchmark_list,
        monthly_weight_df=monthly_weight_df,
        tradeable_asset_list=config.tradeable_asset_list,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.daily_target_weights = monthly_weight_df.reindex(pricing_data_df.index).ffill().dropna()
    strategy.show_taa_weights_report = True

    run_daily(strategy, pricing_data_df, calendar_idx)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(f"Fallback asset: {config.fallback_asset}")
    print(f"Defensive asset list: {config.defensive_asset_list}")
    print(f"Rank weight vector: {config.rank_weight_vec}")
    print(f"Vol lookback day vector: {config.vol_lookback_day_vec}")

    print("First score rows:")
    display(score_df.dropna().head())

    print("First realized volatility rows:")
    display(realized_volatility_df.dropna().head())

    print("First VIX rows:")
    display(vix_close_ser.dropna().head())

    print("First safe-count rows:")
    display(safe_count_ser.dropna().head())

    print("First monthly weights:")
    display(monthly_weight_df.head())

    display(strategy._transactions.tail(10))
    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
