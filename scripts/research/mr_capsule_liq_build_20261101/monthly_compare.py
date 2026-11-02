"""Today's capsule against the sized capsule, month by month: alone and in the Growth products. Descriptive.

Usage: monthly_compare.py research | fresh.
research: the study's pods to 2026-10-14, fresh TAA and CORE5 of growth_closed_capsule_20261101 after the Lab's end.
fresh: every pod from fresh_runs.py through the panel's last session.
Up to the Lab's end TAA 1N and CORE5 are the Lab's columns (before 2012-10 TAA is a synthetic proxy); fund cash on every pod; annual reset. Writes monthly_<source>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
sys.path.insert(0, str(MAIN / "scripts" / "research" / "growth_hunt_20261030" / "shared"))
import hunt  # noqa: E402

OUT = REPO / "results" / "research" / "mr_capsule_liq_build_20261101"
STUDY_PODS = MAIN / "results" / "scout" / "growth_mr_products_20261101" / "pods"
CLOSED = MAIN / "results" / "scout" / "growth_closed_capsule_20261101" / "sleeves"
BOOKS = {"Growth": (0.40, 0.30), "Growth Plus": (0.50, 0.20), "Aggressive": (0.60, 0.10)}


def pod_return(path_df: pd.DataFrame) -> pd.Series:
    nav, cash = path_df["total_value_float"].astype(float), path_df["cash_float"].astype(float)
    return nav.pct_change() + hunt.cash_add(cash, nav)


def read(folder: Path, alias: str) -> pd.Series:
    path_df = pd.read_csv(folder / f"{alias}__path.csv.gz", index_col=0, parse_dates=True)
    path_df.index = pd.to_datetime(path_df.index).normalize()
    return pod_return(path_df)


def month_returns(r: pd.Series) -> pd.Series:
    out = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    out.index = [f"{y}-{m:02d}" for y, m in out.index]
    return out


def stats(r: pd.Series, rf: pd.Series) -> dict:
    x = r - rf.reindex(r.index)
    nav = (1.0 + r).cumprod()
    return {"cagr": float(nav.iloc[-1] ** (252 / len(r)) - 1), "xs": float(x.mean() / x.std() * 252 ** 0.5), "vol": float(r.std() * 252 ** 0.5), "dd": float((nav / nav.cummax().clip(lower=1.0) - 1).min())}


def build(source: str) -> tuple[dict[str, pd.Series], pd.Series, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """(the daily return series by name, T-bills, the pod returns, the Lab's end, the last session)."""
    frame, _, meta = hunt._frames()
    lab_start, lab_end = pd.Timestamp(meta["start_str"]), pd.Timestamp(meta["end_str"])
    tbill_col = meta["tbill_column_str"]
    fresh = OUT / "fresh"
    mr_folder = STUDY_PODS if source == "research" else fresh
    etf_folder = CLOSED if source == "research" else fresh
    pods = {alias: read(mr_folder, alias) for alias in ("dv2_g_n", "hpi_g_n", "dv2_g_l", "hpi_g_l")}
    taa, core = read(etf_folder, "taa3x_1n"), read(etf_folder, "core5")
    last = min(min(s.dropna().index[-1] for s in pods.values()), taa.dropna().index[-1], core.dropna().index[-1])
    after = taa.loc[lab_end:last].index[1:]
    index = frame.loc[lab_start:lab_end].index.append(after)
    ext = pd.DataFrame(index=index)
    for alias, s in pods.items():
        ext[alias] = s.reindex(index)
    ext["taa"] = pd.concat([frame.loc[lab_start:lab_end, "taa3x_1n"], taa.reindex(after)])
    ext["core"] = pd.concat([frame.loc[lab_start:lab_end, "core5"], core.reindex(after)])
    dtb3 = pd.read_parquet(hunt.SHARED_PATH / "dtb3.parquet")["dtb3"]
    rate = dtb3.reindex(dtb3.index.union(after)).ffill().reindex(after)
    days = pd.Series(index, index=index).diff().dt.days.reindex(after)
    rf = pd.concat([frame.loc[lab_start:lab_end, tbill_col], rate * days / 360.0])
    if ext.isna().any().any():
        raise SystemExit(f"missing returns {ext.isna().sum().to_dict()}")

    series = {"Capsule today": hunt.book_annual(ext, {"dv2_g_n": 0.5, "hpi_g_n": 0.5}, index[0], last), "Capsule sized": hunt.book_annual(ext, {"dv2_g_l": 0.5, "hpi_g_l": 0.5}, index[0], last)}
    for pod, label in (("dv2", "DV2"), ("hpi", "HPI")):
        series[f"{label} today"], series[f"{label} sized"] = ext[f"{pod}_g_n"], ext[f"{pod}_g_l"]
    for name, (t, c) in BOOKS.items():
        for key, label in (("n", "today"), ("l", "sized")):
            series[f"{name} {label}"] = hunt.book_annual(ext, {"taa": t, "core": c, f"dv2_g_{key}": 0.15, f"hpi_g_{key}": 0.15}, index[0], last)
    return series, rf, ext, lab_end, last


def main(source: str) -> None:
    series, rf, ext, lab_end, last = build(source)
    index = ext.index
    res = {"source": source, "first": str(index[0].date()), "last": str(last.date()), "lab_end": str(lab_end.date()), "stats": {k: stats(v, rf) for k, v in series.items()}, "monthly": {k: month_returns(v).to_dict() for k, v in series.items()},
           "yearly": {k: {int(y): float(v_) for y, v_ in ((1.0 + v).groupby(v.index.year).prod() - 1.0).items()} for k, v in series.items()}, "tbill_monthly": month_returns(rf.loc[index[0]:last]).to_dict(), "stats_2023": {k: stats(v.loc["2023":], rf) for k, v in series.items()},
           "stats_12m": {k: stats(v.loc[str((last - pd.DateOffset(years=1)).date()):], rf) for k, v in series.items()}}
    res["taa_monthly"], res["core_monthly"] = month_returns(ext["taa"]).to_dict(), month_returns(ext["core"]).to_dict()
    (OUT / f"monthly_{source}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 60)
    print("window", res["first"], res["last"])
    print(pd.DataFrame(res["stats"]).T.round(3).to_string())
    print(pd.DataFrame(res["yearly"]).loc[2020:, ["Capsule today", "Capsule sized", "Growth Plus today", "Growth Plus sized"]].round(4).to_string())
    print(pd.DataFrame(res["monthly"]).iloc[-14:][["DV2 today", "DV2 sized", "HPI today", "HPI sized", "Capsule today", "Capsule sized", "Growth today", "Growth sized", "Growth Plus today", "Growth Plus sized", "Aggressive today", "Aggressive sized"]].round(4).to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(sys.argv[1])
