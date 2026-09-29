"""Where do the books stand after a realistic backtest haircut? (owner question, 2026-09-24)

Descriptive, nothing tuned.
  Haircut: live excess return = (1 - h) x backtest excess return over T-bills, applied as a constant daily
  drag on the backtest path (same volatility, same shocks, less drift), h in {0, 30, 50, 70%}.
  References on the same window (2012-10-02 -> 2026-08-19, or from inception when later), real and
  un-haircut: S&P 500 TR, 60/40, QQQ, and investable hedge-fund / managed-futures proxies from Norgate:
  QAI (hedge-fund multi-strategy tracker), HDG (hedge-fund index replication), MNA (merger arbitrage),
  WTMF / FMF / DBMF / KMLM (managed futures).
  Shekel view: products held unhedged by an Israeli investor, USDILS from Norgate.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

END_TS, START_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02")
PRODUCT_LIST = ["DEF", "LT_DEF", "LT_BAL", "LT_GRO", "GRO", "AGG"]
LADDER_DICT = {"ladder_4_growth": {"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_tqqq": 0.34}}
HAIRCUT_LIST = [0.0, 0.3, 0.5, 0.7]
PROXY_LIST = ["QAI", "HDG", "MNA", "WTMF", "FMF", "DBMF", "KMLM"]


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    book_dict = {p: (dict(zip(g["alias_str"], g["weight_float"])), "annual") for p, g in weight_df.groupby("product_id_str") if p in PRODUCT_LIST}
    book_dict.update({k: (v, "none") for k, v in LADDER_DICT.items()})
    tbill_ser = bench_df["TBILL"]

    def base_ts_for(ser: pd.Series) -> pd.Timestamp:
        return sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]

    def stats(ser: pd.Series) -> dict:
        ser = ser.dropna()
        m = common.metric_dict(ser, bench_df["SPXTR"], tbill_ser, base_ts_for(ser))
        return {"start": ser.index[0].date(), "cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe_excess": m["sharpe_excess_float"],
                "maxdd": m["max_drawdown_float"], "worst_year": m["worst_year_float"]}

    row_list = []
    book_ser_dict = {}
    for name_str, (weight_dict, policy_str) in [(p, book_dict[p]) for p in PRODUCT_LIST] + [(k, book_dict[k]) for k in LADDER_DICT]:
        book_ser = common.book_return_ser(sleeve_df.loc[START_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        book_ser_dict[name_str] = book_ser
        mean_excess_float = float((book_ser - tbill_ser.reindex(book_ser.index)).mean())
        for haircut_float in HAIRCUT_LIST:
            # *** CRITICAL*** a constant drag: same path and volatility, (1 - h) of the backtest excess drift.
            row_list.append({"series": name_str, "haircut": haircut_float, **stats(book_ser - haircut_float * mean_excess_float)})
    reference_dict = {"S&P 500 TR": bench_df["SPXTR"], "60/40": bench_df["SIXTY_FORTY"], "QQQ": bench_df["QQQ"]}
    for symbol_str in PROXY_LIST:
        close_ser = common.load_total_return_close_ser(symbol_str, "2009-01-01", END_TS.strftime("%Y-%m-%d")).reindex(sleeve_df.index)
        reference_dict[symbol_str] = close_ser.pct_change(fill_method=None)
    for name_str, ret_ser in reference_dict.items():
        ret_ser = ret_ser.loc[START_TS:].dropna().iloc[1:]
        row_list.append({"series": name_str, "haircut": None, **stats(ret_ser)})
    result_df = pd.DataFrame(row_list)

    usdils_ser = common.load_total_return_close_ser("USDILS", "2012-01-01", END_TS.strftime("%Y-%m-%d")).reindex(sleeve_df.index).ffill()
    fx_ser = usdils_ser.pct_change(fill_method=None).loc[START_TS:]
    fx_row_list = [{"series": "USDILS", "start_level": float(usdils_ser.loc[:START_TS].iloc[-1]), "end_level": float(usdils_ser.iloc[-1])}]
    for name_str in ["DEF", "LT_GRO", "AGG"]:
        usd_ser = book_ser_dict[name_str]
        ils_ser = (1.0 + usd_ser) * (1.0 + fx_ser.reindex(usd_ser.index).fillna(0.0)) - 1.0
        for label_str, ser in (("usd", usd_ser), ("ils_unhedged", ils_ser)):
            nav_ser = common.nav_from_return_ser(ser)
            fx_row_list.append({"series": f"{name_str}:{label_str}", "cagr": float(nav_ser.iloc[-1] ** (252.0 / len(ser)) - 1.0),
                                "maxdd": float((nav_ser / nav_ser.cummax() - 1.0).min()),
                                "last_2y": float(nav_ser.iloc[-1] / nav_ser.loc[:END_TS - pd.DateOffset(years=2)].iloc[-1] - 1.0)})
    fx_df = pd.DataFrame(fx_row_list)

    output_dir_path = common.STUDY_DIR_PATH / "books"
    result_df.to_csv(output_dir_path / "where_do_i_stand.csv", index=False, float_format="%.6g")
    fx_df.to_csv(output_dir_path / "where_do_i_stand_fx.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(result_df.round(3).to_string(index=False))
    print()
    print(fx_df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
