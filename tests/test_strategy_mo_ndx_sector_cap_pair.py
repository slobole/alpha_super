"""The sector-capped NDX VXN pair (Scout A15 design of record): selection walk, VXN scaling, wiring."""

from __future__ import annotations

import pandas as pd
import pytest

from alpha import strategy_registry
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import VxnScaledAtrNormalizedNdxStrategy
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap import (
    DEFAULT_CONFIG as ATR_CONFIG,
    SectorCapSelectionMixin,
    SectorCapVxnScaledAtrNormalizedNdxStrategy,
)
from strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled import Natr20VxnScaledNdxStrategy
from strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap import (
    DEFAULT_CONFIG as NATR_CONFIG,
    SectorCapNatr20VxnScaledNdxStrategy,
)


class _RankedStub:
    def __init__(self, ranked_symbol_list, max_positions_int, vxn_close_float):
        self._ranked_df = pd.DataFrame({"score": range(len(ranked_symbol_list), 0, -1)}, index=ranked_symbol_list)
        self.max_positions_int = max_positions_int
        self.previous_bar = pd.Timestamp("2026-08-31")
        self.vxn_scale_signal_df = pd.DataFrame(
            {"vxn_close": [vxn_close_float], "vxn_exposure_scale_float": [min(1.0, max(0.25, 22.0 / vxn_close_float))]},
            index=[pd.Timestamp("2026-08-28")],
        )

    def get_ranked_candidate_feature_df(self, close_row_ser):
        return self._ranked_df


class _Capped(SectorCapSelectionMixin, _RankedStub):
    pass


def test_cap_skips_full_sector_and_fills_from_the_next_ranked_names():
    ranked_list = ["A1", "A2", "A3", "B1", "A4", "B2", "C1"]
    sector_map = {s: s[0] for s in ranked_list}
    strategy_obj = _Capped(ranked_list, max_positions_int=4, vxn_close_float=11.0)
    strategy_obj.configure_sector_cap(sector_map, sector_cap_int=2)
    weight_ser = strategy_obj.get_target_weight_ser(pd.Series(dtype=float))
    assert list(weight_ser.index) == ["A1", "A2", "B1", "B2"]
    assert weight_ser.tolist() == pytest.approx([0.25] * 4)  # VXN 11 -> scale clipped at 1.0
    assert strategy_obj.get_selection_audit_df()["max_sector_count_int"].iloc[0] == 2


def test_vxn_scale_multiplies_the_capped_weights_and_unknown_is_a_capped_bucket():
    ranked_list = ["X1", "X2", "X3", "Y1"]
    strategy_obj = _Capped(ranked_list, max_positions_int=4, vxn_close_float=44.0)
    strategy_obj.configure_sector_cap({"Y1": "Y"}, sector_cap_int=2)  # X* unclassified -> UNKNOWN
    weight_ser = strategy_obj.get_target_weight_ser(pd.Series(dtype=float))
    assert list(weight_ser.index) == ["X1", "X2", "Y1"]
    assert weight_ser.tolist() == pytest.approx([0.125] * 3)  # 0.25 x (22 / 44)


def test_pair_inherits_the_two_engine_rules_and_is_pm_ready_not_wired():
    assert issubclass(SectorCapVxnScaledAtrNormalizedNdxStrategy, VxnScaledAtrNormalizedNdxStrategy)
    assert issubclass(SectorCapNatr20VxnScaledNdxStrategy, Natr20VxnScaledNdxStrategy)
    assert SectorCapVxnScaledAtrNormalizedNdxStrategy.get_target_weight_ser is SectorCapSelectionMixin.get_target_weight_ser
    assert SectorCapNatr20VxnScaledNdxStrategy.get_target_weight_ser is SectorCapSelectionMixin.get_target_weight_ser
    assert ATR_CONFIG.sector_cap_int == 4 and NATR_CONFIG.sector_cap_int == 4
    assert ATR_CONFIG.max_positions_int == 10 and NATR_CONFIG.max_positions_int == 10
    for ref_str in (
        "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap:SectorCapVxnScaledAtrNormalizedNdxStrategy",
        "strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap:SectorCapNatr20VxnScaledNdxStrategy",
    ):
        assert strategy_registry.tier_for(ref_str) == strategy_registry.MaturityTier.PM_READY
        assert ref_str not in strategy_registry.wired_import_tuple()
