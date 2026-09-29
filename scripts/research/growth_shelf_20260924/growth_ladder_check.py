"""Does the ladder_4 family keep its quality edge once trading costs rise? (owner's bench screenshot, 2026-09-24)

The ladder_4 books carry a third in single-stock mean reversion (DV2 + HPI, ~200 trade days a year), so their
return per unit of drawdown is the most exposed to trading costs. Same cost stress as growth_study.py: +5 bps per
side on every traded dollar of every sleeve. Window 2012-10-02 -> 2026-08-19; drawdown incl. 2008 from the long
window (2x no-BTAL TAA stand-ins before 2012-10-02 only). Calmar = CAGR / |worst drawdown|.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import growth_dossier as gd  # noqa: E402
import growth_study as gs  # noqa: E402
from growth_dossier import common  # noqa: E402
from growth_frontier import STAND_IN_DICT, book_dict  # noqa: E402

sys.path.insert(0, str(gd.REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import evaluation  # noqa: E402

BOOK_LIST = ["ladder_4 (drift)", "ladder_4 (annual)", "ladder_4_1n (drift)", "ladder_3 (drift)",
             "G2 TAA 3x 1/N + MOSAIC", "G3 TAA 3x + NDX (live pair)", "G4 TAA 3x 1/N + NDX + MOSAIC"]


def stats(ser: pd.Series) -> tuple[float, float, float]:
    nav_ser = (1.0 + ser).cumprod()
    years_float = (ser.index[-1] - ser.index[0]).days / 365.25
    return (float(nav_ser.iloc[-1] ** (1.0 / years_float) - 1.0), float(ser.mean() / ser.std() * np.sqrt(252.0)),
            float((nav_ser / nav_ser.cummax() - 1.0).min()))


def main() -> int:
    all_book_dict = book_dict()
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in STAND_IN_DICT.items():
        long_df.loc[long_df.index < gd.CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < gd.CUT_TS, stand_in_str]
    path_by_alias_dict = common.load_sleeve_path_dict()
    alias_set = {a for n in BOOK_LIST for a in all_book_dict[n][0]}
    stressed_df = sleeve_df.copy()
    for alias_str in sorted(alias_set):
        transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        drag_ser = evaluation.extra_slippage_cost_ser(transaction_df, path_by_alias_dict[alias_str]["total_value_float"].loc[:gd.END_TS],
                                                      gs.EXTRA_SLIPPAGE_PER_SIDE_FLOAT)
        live_mask = sleeve_df[alias_str].notna()
        stressed_df.loc[live_mask, alias_str] = sleeve_df.loc[live_mask, alias_str] - drag_ser.reindex(sleeve_df.index).fillna(0.0)[live_mask]
    row_list = []
    for name_str in BOOK_LIST:
        weight_dict, policy_str = all_book_dict[name_str]
        long_dd_float = stats(common.book_return_ser(long_df.loc[gd.LONG_TS:, list(weight_dict)], weight_dict, policy_str)[0])[2]
        for label_str, frame_df in (("base", sleeve_df), ("cost +5bps/side", stressed_df)):
            cagr_float, sharpe_float, dd_float = stats(common.book_return_ser(frame_df.loc[gd.CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0])
            worst_dd_float = min(dd_float, long_dd_float)
            row_list.append({"book": name_str, "case": label_str, "cagr": cagr_float, "sharpe": sharpe_float, "maxdd": dd_float,
                             "maxdd_incl_2008": worst_dd_float, "calmar_incl_2008": cagr_float / abs(worst_dd_float)})
    result_df = pd.DataFrame(row_list)
    result_df.to_csv(gd.OUT_DIR_PATH / "ladder_cost_check.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 220)
    print(f"extra cost per side: {gs.EXTRA_SLIPPAGE_PER_SIDE_FLOAT}")
    print(result_df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
