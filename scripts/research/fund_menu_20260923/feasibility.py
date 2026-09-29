"""Pre-freeze risk map: which book volatilities each product line can reach.

For each line in frozen_spec_DRAFT.yaml it sweeps the dial (equity-engine risk
share) from 0 to 1 and records the ex-ante volatility and capital split of the
risk-budget book on the exact window's weekly covariance. Only covariance is
used - no returns, no book performance - so the numbers can inform the
volatility targets without steering them toward a result.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import construction  # noqa: E402
from build_books import weekly_covariance_df, window_start_ts  # noqa: E402


def main() -> int:
    spec_dict = yaml.safe_load((Path(__file__).resolve().parent / "frozen_spec_DRAFT.yaml").read_text(encoding="utf-8"))
    sleeve_return_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)
    sleeve_return_df = sleeve_return_df.loc[: spec_dict["end_date_str"]]
    output_dir_path = common.STUDY_DIR_PATH / "construction"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    row_list = []
    for line_str, engine_list in spec_dict["lines_for_feasibility"].items():
        alias_list = sorted({a for e in engine_list for a in spec_dict["engines"][e]["members"]})
        start_ts = window_start_ts(sleeve_return_df, alias_list)
        covariance_df = weekly_covariance_df(sleeve_return_df.loc[start_ts:, alias_list])
        group_share_dict = {}
        for group_str, share_dict in spec_dict["group_engine_share"].items():
            kept_dict = {e: s for e, s in share_dict.items() if e in engine_list}
            if kept_dict:
                total_float = sum(kept_dict.values())
                group_share_dict[group_str] = {e: s / total_float for e, s in kept_dict.items()}
        member_share_dict = {e: dict(spec_dict["engines"][e]["members"]) for e in engine_list}
        for dial_float in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
            use_group_dict = {g: d for g, d in group_share_dict.items() if not (dial_float >= 1.0 and g == "diversifier") and not (dial_float <= 0.0 and g == "equity")}
            budget_ser = construction.budget_ser_for_dial(min(max(dial_float, 0.0), 1.0), use_group_dict, member_share_dict)
            weight_ser = construction.risk_budget_weight_ser(covariance_df, budget_ser)
            equity_alias_list = [a for e in engine_list if spec_dict["engines"][e]["group_str"] == "equity" for a in spec_dict["engines"][e]["members"]]
            row_list.append(
                {
                    "line_str": line_str,
                    "window_start_str": start_ts.date().isoformat(),
                    "dial_float": dial_float,
                    "ex_ante_volatility_float": construction.book_volatility_float(covariance_df, weight_ser),
                    "equity_engine_capital_float": float(weight_ser.reindex(equity_alias_list).fillna(0.0).sum()),
                    "weights_str": ", ".join(f"{a} {w:.0%}" for a, w in weight_ser.sort_values(ascending=False).items()),
                }
            )
        vol_ser = pd.Series(np.sqrt(np.diag(covariance_df)), index=covariance_df.index)
        print(f"\n{line_str}: window from {start_ts.date()}; sleeve weekly-annualised vols: " + ", ".join(f"{a} {v:.1%}" for a, v in vol_ser.items()))
    feasibility_df = pd.DataFrame(row_list)
    feasibility_df.to_csv(output_dir_path / "feasibility.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 140)
    print(feasibility_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
