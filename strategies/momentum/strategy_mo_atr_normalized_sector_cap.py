"""
Monthly ATR-adjusted momentum rotation with a hard per-sector position cap.

Comparison arm to the correlation-penalized variant: instead of penalizing
candidates by trailing return correlation, this variant walks the base ranking
top-down and skips any candidate whose GICS level-1 sector already holds
sector_cap positions in the basket.

Core rule on month-end decision date t (after the base regime, universe, and
stock-trend gates):

    ranked candidates by score_{i,t} = monthly_roc / ATR20   (unchanged)

    accept candidate i iff count(sector(i) in selected) < sector_cap
    stop when max_positions accepted or candidates exhausted

    target_weight_{i,t} = 1 / max_positions per accepted name

Sector labels
-------------
Labels come from Norgate's CURRENT GICS classification (level 1, IDs), fetched
once at data-prep time. Norgate does not store point-in-time classifications,
so a symbol's historical sector is approximated by today's label.

*** REALISM GAP *** Current-label GICS is mildly anticipatory: reclassified
symbols (notably the 2018 Communication Services restructure) carry today's
sector back through history. Documented in ASSUMPTIONS_AND_GAPS.md; the
correlation-penalty variant is the label-free control for this arm.

Symbols with no available classification map to the sentinel sector
"UNKNOWN", which is capped like any real sector so unclassified names cannot
crowd the basket.

Execution mapping is unchanged: decision at the actual last tradable close of
the month, execution at the next tradable open.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    AtrNormalizedNdxStrategy,
)


UNKNOWN_SECTOR_STR = "UNKNOWN"

__all__ = [
    "SectorCapAtrNormalizedStrategy",
    "UNKNOWN_SECTOR_STR",
    "build_current_gics_sector_map",
    "select_sector_capped_symbol_list",
]


def build_current_gics_sector_map(symbol_list: Sequence[str]) -> dict[str, str]:
    """
    Fetch the current GICS level-1 sector ID for each symbol from Norgate.

    Returns {symbol: sector_id_str}; symbols without a classification map to
    UNKNOWN_SECTOR_STR. Requires a live Norgate connection (NDU).
    """
    import norgatedata

    sector_by_symbol_map: dict[str, str] = {}
    unknown_symbol_list: list[str] = []
    for symbol_str in symbol_list:
        sector_id_str = None
        try:
            sector_id_str = norgatedata.classification_at_level(
                symbol_str, "GICS", "ClassificationId", 1
            )
        except Exception:
            sector_id_str = None
        if sector_id_str is None or str(sector_id_str).strip() == "":
            sector_by_symbol_map[str(symbol_str)] = UNKNOWN_SECTOR_STR
            unknown_symbol_list.append(str(symbol_str))
        else:
            sector_by_symbol_map[str(symbol_str)] = str(sector_id_str)

    if len(unknown_symbol_list) > 0:
        # Fail loud enough to be seen, soft enough not to kill research runs:
        # unclassified names are capped via the UNKNOWN bucket.
        print(
            f"build_current_gics_sector_map: {len(unknown_symbol_list)} of "
            f"{len(symbol_list)} symbols have no GICS sector and were mapped to "
            f"{UNKNOWN_SECTOR_STR}. Sample: {unknown_symbol_list[:10]}"
        )
    return sector_by_symbol_map


def select_sector_capped_symbol_list(
    ranked_symbol_list: Sequence[str],
    sector_by_symbol_map: Mapping[str, str],
    max_positions_int: int,
    sector_cap_int: int,
) -> list[str]:
    """
    Walk the ranking top-down, skipping candidates whose sector is full.
    """
    if max_positions_int <= 0:
        raise ValueError("max_positions_int must be positive.")
    if sector_cap_int <= 0:
        raise ValueError("sector_cap_int must be positive.")

    selected_symbol_list: list[str] = []
    position_count_by_sector_map: dict[str, int] = {}
    for symbol_str in ranked_symbol_list:
        if len(selected_symbol_list) >= max_positions_int:
            break
        sector_str = str(sector_by_symbol_map.get(str(symbol_str), UNKNOWN_SECTOR_STR))
        if position_count_by_sector_map.get(sector_str, 0) >= sector_cap_int:
            continue
        selected_symbol_list.append(str(symbol_str))
        position_count_by_sector_map[sector_str] = (
            position_count_by_sector_map.get(sector_str, 0) + 1
        )
    return selected_symbol_list


class SectorCapAtrNormalizedStrategy(AtrNormalizedNdxStrategy):
    """
    ATR-normalized momentum with a hard GICS sector cap on selection.

    Sizing is unchanged from the base:

        q^{intent}_{i,t} = floor(V_{t-1} * (1 / max_positions) / Close_{i,t-1})
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        sector_by_symbol_map: Mapping[str, str],
        sector_cap_int: int,
        regime_symbol_str: str = "SPY",
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        lookback_month_int: int = 12,
        index_trend_window_int: int = 200,
        stock_trend_window_int: int = 100,
        max_positions_int: int = 10,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            rebalance_schedule_df=rebalance_schedule_df,
            regime_symbol_str=regime_symbol_str,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
            lookback_month_int=lookback_month_int,
            index_trend_window_int=index_trend_window_int,
            stock_trend_window_int=stock_trend_window_int,
            max_positions_int=max_positions_int,
        )
        if sector_cap_int <= 0:
            raise ValueError("sector_cap_int must be positive.")
        if len(sector_by_symbol_map) == 0:
            raise ValueError("sector_by_symbol_map must not be empty.")

        self.sector_by_symbol_map = dict(sector_by_symbol_map)
        self.sector_cap_int = int(sector_cap_int)
        self.selection_audit_row_list: list[dict[str, object]] = []

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        ranked_candidate_feature_df = self.get_ranked_candidate_feature_df(close_row_ser=close_row_ser)
        if len(ranked_candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        ranked_symbol_list = ranked_candidate_feature_df.index.astype(str).tolist()
        selected_symbol_list = select_sector_capped_symbol_list(
            ranked_symbol_list=ranked_symbol_list,
            sector_by_symbol_map=self.sector_by_symbol_map,
            max_positions_int=self.max_positions_int,
            sector_cap_int=self.sector_cap_int,
        )

        selected_sector_ser = pd.Series(
            [
                self.sector_by_symbol_map.get(symbol_str, UNKNOWN_SECTOR_STR)
                for symbol_str in selected_symbol_list
            ]
        )
        sector_count_ser = selected_sector_ser.value_counts()
        self.selection_audit_row_list.append(
            {
                "decision_date_ts": pd.Timestamp(self.previous_bar),
                "candidate_count_int": int(len(ranked_symbol_list)),
                "selected_symbol_list": list(selected_symbol_list),
                "max_sector_count_int": int(sector_count_ser.max()) if len(sector_count_ser) > 0 else 0,
                "sector_count_map": sector_count_ser.to_dict(),
            }
        )

        target_weight_float = 1.0 / float(self.max_positions_int)
        return pd.Series(target_weight_float, index=selected_symbol_list, dtype=float)

    def get_selection_audit_df(self) -> pd.DataFrame:
        if len(self.selection_audit_row_list) == 0:
            return pd.DataFrame(
                columns=[
                    "decision_date_ts",
                    "candidate_count_int",
                    "selected_symbol_list",
                    "max_sector_count_int",
                    "sector_count_map",
                ]
            )
        return pd.DataFrame(self.selection_audit_row_list).set_index("decision_date_ts")
