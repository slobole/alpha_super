"""How much does the 2x stand-in understate 2008 for the 3x TAA books? (owner question, 2026-09-25)

The BTAL TAA sleeves start 2012-10-02 (TQQQ from 2010, BTAL from 2011). Before that the growth tables use
taa_1n_qld: the same Defense First 1/N rules, defensive slots GLD / UUP / TLT / DBC (no BTAL), fallback QLD (2x).
This check scales the fallback leg to 3x: on each day the stand-in holds QLD with weight w (prior close),
    r_3x = r_2x + 0.5 * w * r_QLD - 0.5 * w * T-bill (the extra leverage is financed),
i.e. what the same trades would have returned with a 3x Nasdaq ETF. BTAL cannot be rebuilt; its slot stays
spread over the four other defensive assets, as in the stand-in.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import growth_dossier as gd  # noqa: E402
from growth_dossier import common, load_price_timeseries  # noqa: E402
from growth_menu import BOOK_DICT  # noqa: E402
from dv2_liquidity_eval import variant_return_ser  # noqa: E402

sys.path.insert(0, str(gd.REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import evaluation  # noqa: E402

GFC_TUPLE = ("2008-05-19", "2009-03-09")


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    sleeve_df["dv2_liq_floor"] = variant_return_ser("dv2_liq_floor", sleeve_df.index)
    path_df = common.load_sleeve_path_dict()["taa_1n_qld"]
    transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / "taa_1n_qld__transactions.csv.gz", parse_dates=["date"])
    qld_df = load_price_timeseries("QLD", start_date_str="2006-01-01", end_date_str="2012-12-31")
    qld_df.index = pd.to_datetime(qld_df.index).normalize()
    qld_close_ser = qld_df["Close"].astype(float)
    shares_ser = transaction_df[transaction_df["asset_str"] == "QLD"].groupby("date")["amount_float"].sum()
    shares_ser = shares_ser.reindex(path_df.index).fillna(0.0).cumsum()
    # *** CRITICAL*** prior-close weight: the exposure held going into day t.
    weight_ser = (shares_ser * qld_close_ser.reindex(path_df.index) / path_df["total_value_float"]).shift(1).fillna(0.0)
    qld_return_ser = qld_close_ser.pct_change(fill_method=None).reindex(sleeve_df.index).fillna(0.0)
    tbill_daily_ser = evaluation.lagged_tbill_annual_rate_ser(sleeve_df.index) / 252.0
    three_x_ser = sleeve_df["taa_1n_qld"] + 0.5 * weight_ser.reindex(sleeve_df.index).fillna(0.0) * (qld_return_ser - tbill_daily_ser)
    print(f"stand-in QLD weight 2008-05..2009-03: mean {weight_ser.loc[GFC_TUPLE[0]:GFC_TUPLE[1]].mean():.1%}, "
          f"max {weight_ser.loc[GFC_TUPLE[0]:GFC_TUPLE[1]].max():.1%}")

    row_list = []
    for label_str, stand_in_ser in (("2x stand-in (tables)", sleeve_df["taa_1n_qld"]), ("3x-scaled stand-in", three_x_ser)):
        long_df = sleeve_df.copy()
        for alias_str in ("taa_btal_tqqq", "taa_btal_1n_tqqq"):
            long_df.loc[long_df.index < gd.CUT_TS, alias_str] = stand_in_ser[long_df.index < gd.CUT_TS]
        for book_str, (weight_dict, policy_str) in BOOK_DICT.items():
            if not any(a.startswith("taa_") for a in weight_dict):
                continue
            long_ser = common.book_return_ser(long_df.loc[gd.LONG_TS:, list(weight_dict)], weight_dict, policy_str)[0]
            pre_ser = long_ser.loc[: gd.CUT_TS - pd.Timedelta(days=1)]
            nav_ser = (1.0 + pre_ser).cumprod()
            row_list.append({"book": book_str, "stand_in": label_str, "gfc_2008": common.window_return_float(long_ser, *GFC_TUPLE),
                             "maxdd_2008_2012": float((nav_ser / nav_ser.cummax() - 1.0).min()),
                             "maxdd_2008_2026": float(((1 + long_ser).cumprod() / (1 + long_ser).cumprod().cummax() - 1.0).min())})
        taa_ser = stand_in_ser.loc[gd.LONG_TS: gd.CUT_TS - pd.Timedelta(days=1)]
        taa_nav = (1.0 + taa_ser).cumprod()
        row_list.append({"book": "TAA sleeve alone", "stand_in": label_str, "gfc_2008": common.window_return_float(stand_in_ser, *GFC_TUPLE),
                         "maxdd_2008_2012": float((taa_nav / taa_nav.cummax() - 1.0).min()), "maxdd_2008_2026": np.nan})
    result_df = pd.DataFrame(row_list)
    result_df.to_csv(gd.OUT_DIR_PATH / "proxy_2008_3x_check.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print(result_df.pivot_table(index="book", columns="stand_in", values=["gfc_2008", "maxdd_2008_2012", "maxdd_2008_2026"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
