import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.momentum.strategy_mo_atr_normalized_sector_cap import (
    SectorCapAtrNormalizedStrategy,
    UNKNOWN_SECTOR_STR,
    select_sector_capped_symbol_list,
)


class SelectSectorCappedSymbolListTests(unittest.TestCase):
    def test_cap_skips_full_sector_and_backfills_from_ranking(self):
        # AAA/BBB/CCC are all tech; cap 2 pushes DDD (health) into the basket.
        ranked_symbol_list = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        sector_map = {"AAA": "45", "BBB": "45", "CCC": "45", "DDD": "35", "EEE": "30"}
        selected = select_sector_capped_symbol_list(
            ranked_symbol_list=ranked_symbol_list,
            sector_by_symbol_map=sector_map,
            max_positions_int=3,
            sector_cap_int=2,
        )
        self.assertEqual(selected, ["AAA", "BBB", "DDD"])

    def test_no_cap_binding_reproduces_top_n(self):
        ranked_symbol_list = ["AAA", "BBB", "CCC"]
        sector_map = {"AAA": "45", "BBB": "35", "CCC": "30"}
        selected = select_sector_capped_symbol_list(
            ranked_symbol_list=ranked_symbol_list,
            sector_by_symbol_map=sector_map,
            max_positions_int=2,
            sector_cap_int=3,
        )
        self.assertEqual(selected, ["AAA", "BBB"])

    def test_missing_symbol_maps_to_unknown_and_unknown_is_capped(self):
        # XXX and YYY are not in the map: both land in UNKNOWN; cap 1 keeps
        # only the first, so unclassified names cannot crowd the basket.
        ranked_symbol_list = ["XXX", "YYY", "AAA"]
        sector_map = {"AAA": "45"}
        selected = select_sector_capped_symbol_list(
            ranked_symbol_list=ranked_symbol_list,
            sector_by_symbol_map=sector_map,
            max_positions_int=3,
            sector_cap_int=1,
        )
        self.assertEqual(selected, ["XXX", "AAA"])

    def test_fewer_eligible_than_slots_returns_partial_basket(self):
        ranked_symbol_list = ["AAA", "BBB"]
        sector_map = {"AAA": "45", "BBB": "45"}
        selected = select_sector_capped_symbol_list(
            ranked_symbol_list=ranked_symbol_list,
            sector_by_symbol_map=sector_map,
            max_positions_int=5,
            sector_cap_int=1,
        )
        self.assertEqual(selected, ["AAA"])

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            select_sector_capped_symbol_list(
                ranked_symbol_list=["AAA"],
                sector_by_symbol_map={"AAA": "45"},
                max_positions_int=0,
                sector_cap_int=1,
            )
        with self.assertRaises(ValueError):
            select_sector_capped_symbol_list(
                ranked_symbol_list=["AAA"],
                sector_by_symbol_map={"AAA": "45"},
                max_positions_int=1,
                sector_cap_int=0,
            )


class SectorCapStrategySelectionTests(unittest.TestCase):
    def make_rebalance_schedule_df(self) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-28")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def make_strategy_obj(self, sector_cap_int: int) -> SectorCapAtrNormalizedStrategy:
        strategy_obj = SectorCapAtrNormalizedStrategy(
            name="SectorCapTest",
            benchmarks=["$SPX"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            sector_by_symbol_map={"AAA": "45", "BBB": "45", "CCC": "35"},
            sector_cap_int=sector_cap_int,
            regime_symbol_str="SPY",
            max_positions_int=2,
        )
        strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
        strategy_obj.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1]},
            index=[strategy_obj.previous_bar],
        )
        return strategy_obj

    def make_close_row(self) -> pd.Series:
        return self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 2.0,
                ("AAA", "stock_trend_pass_bool"): True,
                ("BBB", "risk_adj_score_ser"): 1.9,
                ("BBB", "stock_trend_pass_bool"): True,
                ("CCC", "risk_adj_score_ser"): 1.2,
                ("CCC", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): True,
            }
        )

    def test_cap_one_forces_sector_diversity(self):
        strategy_obj = self.make_strategy_obj(sector_cap_int=1)
        target_weight_ser = strategy_obj.get_target_weight_ser(self.make_close_row())
        self.assertEqual(sorted(target_weight_ser.index.tolist()), ["AAA", "CCC"])
        self.assertTrue(np.allclose(target_weight_ser.to_numpy(), 0.5))

    def test_loose_cap_matches_base_top_n(self):
        strategy_obj = self.make_strategy_obj(sector_cap_int=2)
        target_weight_ser = strategy_obj.get_target_weight_ser(self.make_close_row())
        self.assertEqual(sorted(target_weight_ser.index.tolist()), ["AAA", "BBB"])

    def test_regime_fail_returns_empty(self):
        strategy_obj = self.make_strategy_obj(sector_cap_int=1)
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 2.0,
                ("AAA", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): False,
            }
        )
        self.assertEqual(len(strategy_obj.get_target_weight_ser(close_row_ser)), 0)

    def test_audit_records_sector_counts(self):
        strategy_obj = self.make_strategy_obj(sector_cap_int=1)
        strategy_obj.get_target_weight_ser(self.make_close_row())
        audit_df = strategy_obj.get_selection_audit_df()
        self.assertEqual(len(audit_df), 1)
        self.assertEqual(audit_df.iloc[0]["max_sector_count_int"], 1)
        self.assertEqual(audit_df.iloc[0]["sector_count_map"], {"45": 1, "35": 1})


if __name__ == "__main__":
    unittest.main()
