"""
Monthly ATR-adjusted short momentum rotation for point-in-time Nasdaq 100 members.

Core formulas
-------------
For stock i on month-end decision date t:

    monthly_roc_{i,t}^{(L)}
        = Close_ME_{i,t} / Close_ME_{i,t-L} - 1

    ATR20_{i,t}
        = mean(TR_{i,t-19:t})

    risk_adj_score_{i,t}
        = monthly_roc_{i,t}^{(L)} / ATR20_{i,t}

Short-side regime and selection:

    regime_pass_t^{short}
        = 1[SPY_t < SMA200_t]

    eligible_{i,t}^{short}
        = 1[PIT_NDX_{i,t} = 1 and Close_{i,t} < SMA100_{i,t}]

    selected_t^{short}
        = bottom max_positions eligible symbols by risk_adj_score_{i,t}

    target_weight_{i,t}^{short}
        = -1 / max_positions    if i in selected_t^{short}
        = 0                     otherwise

Execution mapping:

    q^{target}_{i,t}
        = floor(V_{t-1} * target_weight_{i,t}^{short} / Close_{i,t-1})

    fill_price_{i,t}
        = Open_{i,t}

Short realism note
------------------
This strategy uses the Vanilla backtest execution contract, but the repo still
does not model borrow availability or borrow fees. Treat the short-side results
as optimistic until those gaps are modeled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtester import VanillaBacktester
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_radge_ndx import (
    ATR_WINDOW_INT,
    RadgeMomentumNdxConfig,
    RadgeMomentumNdxStrategy,
    audit_pit_universe_df,
    compute_radge_signal_tables,
    get_monthly_decision_close_df,
    get_radge_momentum_ndx_data,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)

__all__ = [
    "ATR_WINDOW_INT",
    "AtrNormalizedNdxShortConfig",
    "AtrNormalizedNdxShortStrategy",
    "DEFAULT_CONFIG",
    "audit_pit_universe_df",
    "compute_radge_signal_tables",
    "get_atr_normalized_ndx_short_data",
    "get_monthly_decision_close_df",
    "map_month_end_decision_dates_to_rebalance_schedule_df",
]


@dataclass(frozen=True)
class AtrNormalizedNdxShortConfig(RadgeMomentumNdxConfig):
    indexname_str: str = "Nasdaq 100"


DEFAULT_CONFIG = AtrNormalizedNdxShortConfig()


class AtrNormalizedNdxShortStrategy(RadgeMomentumNdxStrategy):
    """Canonical NDX preset for the ATR-adjusted monthly short rotation."""

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before monthly rebalances.")
        if self.previous_bar not in self.universe_df.index:
            raise RuntimeError(f"universe_df is missing decision date {self.previous_bar}.")

        candidate_feature_df = close_row_ser.unstack()
        if self.regime_symbol_str not in candidate_feature_df.index:
            raise RuntimeError(f"Missing regime feature row for {self.regime_symbol_str}.")

        regime_pass_value = candidate_feature_df.loc[self.regime_symbol_str].get("regime_pass_bool", np.nan)
        if pd.isna(regime_pass_value) or bool(regime_pass_value):
            return pd.Series(dtype=float)

        required_field_list = ["stock_trend_pass_bool", "risk_adj_score_ser"]
        if any(field_str not in candidate_feature_df.columns for field_str in required_field_list):
            return pd.Series(dtype=float)

        universe_member_ser = self.universe_df.loc[self.previous_bar]
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        candidate_feature_df = candidate_feature_df.assign(
            risk_adj_score_float=pd.to_numeric(candidate_feature_df["risk_adj_score_ser"], errors="coerce"),
            stock_trend_fail_bool=candidate_feature_df["stock_trend_pass_bool"].eq(False),
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_risk_adj_mask_vec = np.isfinite(
            candidate_feature_df["risk_adj_score_float"].to_numpy(dtype=float)
        )
        short_trend_mask_vec = candidate_feature_df["stock_trend_fail_bool"].to_numpy(dtype=bool)
        candidate_feature_df = candidate_feature_df.loc[
            finite_risk_adj_mask_vec & short_trend_mask_vec
        ]
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        candidate_feature_df = candidate_feature_df.sort_values(
            by=["risk_adj_score_float", "symbol_str"],
            ascending=[True, True],
            kind="mergesort",
        )
        selected_feature_df = candidate_feature_df.iloc[: self.max_positions_int].copy()

        target_weight_float = -1.0 / float(self.max_positions_int)
        target_weight_ser = pd.Series(
            target_weight_float,
            index=selected_feature_df.index,
            dtype=float,
        )
        return target_weight_ser

    def get_target_share_int_map(
        self,
        target_weight_ser: pd.Series,
        close_row_ser: pd.Series,
    ) -> dict[str, int]:
        target_share_int_map: dict[str, int] = {}
        if len(target_weight_ser) == 0:
            return target_share_int_map

        budget_value_float = float(self.previous_total_value)
        for symbol_str, target_weight_float in target_weight_ser.items():
            close_price_float = float(close_row_ser[(symbol_str, "Close")])
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(f"Invalid prior close for target asset {symbol_str} on {self.previous_bar}.")

            # *** CRITICAL*** Short target shares must be fixed from the
            # previous_bar close so the rebalance does not adapt to the
            # realized current-bar open.
            target_share_int = int(budget_value_float * float(target_weight_float) / close_price_float)
            if target_share_int < 0:
                target_share_int_map[str(symbol_str)] = int(target_share_int)

        return target_share_int_map

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** The scheduled month-end decision close must equal
        # previous_bar exactly, otherwise signals and next-open execution drift.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        target_weight_ser = self.get_target_weight_ser(close_row_ser=close_row_ser)
        target_share_int_map = self.get_target_share_int_map(
            target_weight_ser=target_weight_ser,
            close_row_ser=close_row_ser,
        )
        target_symbol_set = set(target_share_int_map)

        current_position_ser = self.get_positions()
        active_position_ser = current_position_ser[current_position_ser != 0]
        for symbol_str in active_position_ser.index:
            if symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        for symbol_str, target_share_int in target_share_int_map.items():
            current_share_int = int(current_position_ser.get(symbol_str, 0.0))
            if current_share_int == target_share_int:
                continue

            if current_share_int >= 0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            target_weight_float = float(target_weight_ser.loc[symbol_str])
            self.order_target_percent(
                symbol_str,
                target_weight_float,
                trade_id=self.current_trade_map[symbol_str],
            )


def get_atr_normalized_ndx_short_data(
    config: AtrNormalizedNdxShortConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return get_radge_momentum_ndx_data(config=config)


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_short_data(config)

    strategy = AtrNormalizedNdxShortStrategy(
        name="strategy_mo_atr_normalized_ndx_short",
        benchmarks=[config.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config.regime_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
        lookback_month_int=config.lookback_month_int,
        index_trend_window_int=config.index_trend_window_int,
        stock_trend_window_int=config.stock_trend_window_int,
        max_positions_int=config.max_positions_int,
    )
    strategy.universe_df = universe_df

    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config.backtest_start_date_str)]
    VanillaBacktester().run(
        strategy=strategy,
        pricing_data_df=pricing_data_df,
        calendar_idx=calendar_idx,
        show_progress_bool=False,
        show_signal_progress_bool=False,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
