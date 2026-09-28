"""EOM extras from the saved baseline: warm-up (missing-bucket) months, sub-windows, T-bill comparison.

Usage: uv run python tb_eom_extra.py   (needs baseline_eom outputs)
"""

from __future__ import annotations

import pandas as pd

import tb_common as tc


def main() -> None:
    eq = pd.read_parquet(tc.OUT / "equity_eom.parquet")["total_value"].astype(float)
    mt = pd.read_csv(tc.OUT / "eom_month_table.csv", parse_dates=["measure_date", "final_fill_date", "exit_fill_date"])
    dtb3 = pd.read_csv(tc.REPO.parent / "1_data" / "DTB3.csv", parse_dates=["observation_date"], na_values=".")
    dtb3 = dtb3.set_index("observation_date")["DTB3"].astype(float).ffill() / 100.0
    rate = dtb3.reindex(eq.index, method="ffill").shift(1).fillna(0.0)
    tbill = (1 + rate / 252.0).cumprod()
    out = {"windows": {}}
    for label, start in (("full", None), ("first_finite_bucket_2004-08", "2004-08-02"), ("2014+", "2014-01-02"),
                         ("2019+", "2019-01-02"), ("last3y", tc.LAST3Y_START_STR)):
        out["windows"][label] = {"strategy": tc.metrics(eq, start=start), "tbill": tc.metrics(tbill, start=start)}
    warm = mt[mt["bucket_ief_measure_causal_int"].isna() & (mt["final_fill_date"] >= "2003-01-02")]
    out["missing_bucket_months_traded"] = int(len(warm))
    out["missing_bucket_window"] = [str(warm["final_fill_date"].min().date()), str(warm["exit_fill_date"].max().date())]
    seg = eq.loc[:warm["exit_fill_date"].max()]
    out["missing_bucket_segment_return"] = float(seg.iloc[-1] / seg.iloc[0] - 1)
    buckets = mt["bucket_ief_measure_causal_int"].value_counts(dropna=False).to_dict()
    out["bucket_counts"] = {str(k): int(v) for k, v in buckets.items()}
    tc.write_json("eom_extra.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    main()
