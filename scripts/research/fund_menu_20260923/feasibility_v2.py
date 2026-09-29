"""Pre-freeze risk map for the two-bucket capital template (risk only, no returns).

Template: book = s * ReturnBucket + (1 - s) * StabilizerBucket, capital weights.
- Stabilizer bucket capital: CORE5 50% (owner anchor), Tactical FI 25%, EOM flow 25%.
- Return bucket capital: four engines at equal capital; inside an engine the
  capital is split equally across its members, except that a secondary member
  is used only when every member of that engine would hold >= 5% of the book.
The script reports the ex-ante volatility (weekly covariance, exact window) of
the book for s in 0..1 and each engine's ex-ante risk share, for the main line
and the low-touch line. Nothing here reads a return path.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import construction  # noqa: E402
from build_books import weekly_covariance_df, window_start_ts  # noqa: E402

LINES = {
    "main": {
        "return_engines": {
            "MR_S": ["hpi_vote", "dv2"],
            "MR_X": ["sector_vox_iyr", "disp_kie_ihi_sma"],
            "MOM": ["ndx_vxn", "mosaic"],
            "TAA": ["taa_btal_tqqq", "infl_compass"],
        },
        "stabilizers": {"core5": 0.5, "tactical_fi": 0.25, "eom_flow": 0.25},
    },
    "low_touch": {
        "return_engines": {
            "MOM": ["ndx_vxn", "mosaic"],
            "TAA": ["taa_btal_tqqq", "infl_compass"],
        },
        "stabilizers": {"core5": 2.0 / 3.0, "tactical_fi": 1.0 / 3.0},
    },
}
SECONDARY_MIN_WEIGHT_FLOAT = 0.05


def template_weight_ser(line_dict: dict, share_float: float) -> pd.Series:
    weight_dict: dict[str, float] = {}
    engine_count_int = len(line_dict["return_engines"])
    for member_list in line_dict["return_engines"].values():
        engine_capital_float = share_float / engine_count_int
        if len(member_list) > 1 and engine_capital_float / len(member_list) >= SECONDARY_MIN_WEIGHT_FLOAT:
            for alias_str in member_list:
                weight_dict[alias_str] = weight_dict.get(alias_str, 0.0) + engine_capital_float / len(member_list)
        elif engine_capital_float > 0:
            weight_dict[member_list[0]] = weight_dict.get(member_list[0], 0.0) + engine_capital_float
    for alias_str, stabilizer_share_float in line_dict["stabilizers"].items():
        if share_float < 1.0:
            weight_dict[alias_str] = weight_dict.get(alias_str, 0.0) + (1.0 - share_float) * stabilizer_share_float
    return pd.Series(weight_dict)


def main() -> int:
    sleeve_return_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)
    sleeve_return_df = sleeve_return_df.loc[:"2026-08-19"]
    row_list = []
    for line_str, line_dict in LINES.items():
        alias_list = sorted({a for m in line_dict["return_engines"].values() for a in m} | set(line_dict["stabilizers"]))
        start_ts = window_start_ts(sleeve_return_df, alias_list)
        covariance_df = weekly_covariance_df(sleeve_return_df.loc[start_ts:, alias_list])
        for share_float in np.round(np.arange(0.0, 1.0001, 0.05), 2):
            weight_ser = template_weight_ser(line_dict, share_float)
            sigma_mat = covariance_df.loc[weight_ser.index, weight_ser.index].to_numpy()
            share_arr = construction.risk_share_arr(sigma_mat, weight_ser.to_numpy())
            engine_of_dict = {a: e for e, m in line_dict["return_engines"].items() for a in m}
            engine_share_ser = pd.Series(share_arr, index=weight_ser.index).groupby(lambda a: engine_of_dict.get(a, "STAB")).sum()
            row_list.append({
                "line_str": line_str, "return_share_float": share_float,
                "ex_ante_volatility_float": construction.book_volatility_float(covariance_df, weight_ser),
                **{f"risk_{k}": v for k, v in engine_share_ser.items()},
                "weights_str": ", ".join(f"{a} {w:.0%}" for a, w in weight_ser.sort_values(ascending=False).items()),
            })
    feasibility_df = pd.DataFrame(row_list)
    feasibility_df.to_csv(common.STUDY_DIR_PATH / "construction" / "feasibility_v2.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 150)
    print(feasibility_df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
