"""How often two legs would send opposite orders in one stock to the same opening auction, read from the engine runs' transactions. IBKR cancels the later order of such a pair when the accounts count as related (docs/plans/IBKR_CROSS_CANCEL_20261102.md).

Usage: opposite_orders.py <results root>. Reads the latest vanilla run of each leg, swap days only: a forced exit of a name that stopped trading is not an order at the auction. A read, no new design.
"""
from __future__ import annotations

import pickle
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
MODULE_DICT = {"1": "strategy_mo_capsule_leg1_natr_ndx", "2": "strategy_mo_capsule_leg2_residual_ndx", "3": "strategy_mo_capsule_leg3_residual_r1000"}
RECENT_START_TS = pd.Timestamp("2021-01-01")


def latest_leg_df(root_path: Path, leg_str: str) -> pd.DataFrame:
    run_path = sorted((root_path / MODULE_DICT[leg_str] / "vanilla_backtest").iterdir())[-1]
    frame_df = pd.read_csv(run_path / "transactions.csv")
    frame_df["date_ts"] = pd.to_datetime(frame_df["bar"]).dt.normalize()
    with open(run_path / f"{MODULE_DICT[leg_str]}.pkl", "rb") as handle:
        strategy_obj = pickle.load(handle)
    equity_ser = strategy_obj.results["total_value"].astype(float)
    equity_ser.index = pd.to_datetime(equity_ser.index).normalize()
    swap_set = set(pd.DatetimeIndex(strategy_obj.rebalance_schedule_df.index))
    frame_df = frame_df[frame_df["date_ts"].isin(swap_set)].copy()
    frame_df["money_float"] = frame_df["total_value"].astype(float).abs()
    frame_df["nav_share_float"] = frame_df["money_float"] / frame_df["date_ts"].map(equity_ser.shift(1))
    return frame_df.groupby(["date_ts", "asset"]).agg(amount_float=("amount", "sum"), money_float=("money_float", "sum"), nav_share_float=("nav_share_float", "sum"))


def main(root_str: str) -> None:
    leg_dict = {leg_str: latest_leg_df(Path(root_str), leg_str) for leg_str in MODULE_DICT}
    all_day_list = sorted(set().union(*[set(frame_df.index.get_level_values(0)) for frame_df in leg_dict.values()]))
    opposite_dict: dict[tuple[str, str], pd.DataFrame] = {}
    row_list = []
    for first_str, second_str in combinations(leg_dict, 2):
        joined_df = leg_dict[first_str].join(leg_dict[second_str], how="inner", lsuffix="_a", rsuffix="_b")
        opposite_df = joined_df[joined_df["amount_float_a"] * joined_df["amount_float_b"] < 0]
        opposite_dict[(first_str, second_str)] = opposite_df
        common_day_set = set(leg_dict[first_str].index.get_level_values(0)) & set(leg_dict[second_str].index.get_level_values(0))
        for label_str, start_ts in (("all years", pd.Timestamp("2000-01-01")), ("since 2021", RECENT_START_TS)):
            window_df = opposite_df[opposite_df.index.get_level_values(0) >= start_ts]
            day_int = sum(day_ts >= start_ts for day_ts in common_day_set)
            hit_int = window_df.index.get_level_values(0).nunique()
            row_list.append({"pair": f"Leg {first_str} / Leg {second_str}", "window": label_str, "swap days": day_int, "with an opposite pair": round(hit_int / max(day_int, 1), 3), "names on such a day": round(len(window_df) / max(hit_int, 1), 2),
                "smaller order, median % of NAV": round(float(window_df[["nav_share_float_a", "nav_share_float_b"]].min(axis=1).median() * 100), 2),
                "larger order, median % of NAV": round(float(window_df[["nav_share_float_a", "nav_share_float_b"]].max(axis=1).median() * 100), 2)})
    print(pd.DataFrame(row_list).to_string(index=False), flush=True)
    for label_str, start_ts in (("all years", pd.Timestamp("2000-01-01")), ("since 2021", RECENT_START_TS)):
        window_day_list = [day_ts for day_ts in all_day_list if day_ts >= start_ts]
        any_df = pd.concat([frame_df[frame_df.index.get_level_values(0) >= start_ts] for frame_df in opposite_dict.values()])
        hit_int = any_df.index.get_level_values(0).nunique()
        name_int = any_df.index.drop_duplicates().size
        print(f"Any two legs, {label_str}: {hit_int} of {len(window_day_list)} swap days = {hit_int / len(window_day_list):.1%}; {name_int / max(hit_int, 1):.1f} names on such a day", flush=True)
        for leg_str, frame_df in leg_dict.items():
            window_df = frame_df[frame_df.index.get_level_values(0) >= start_ts]
            pair_index = pd.Index([])
            for pair_tuple, opposite_df in opposite_dict.items():
                if leg_str in pair_tuple:
                    pair_index = pair_index.union(opposite_df.index)
            share_float = window_df.loc[window_df.index.intersection(pair_index), "money_float"].sum() / window_df["money_float"].sum()
            print(f"  Leg {leg_str}: {share_float:.1%} of its traded money is in such a pair", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
