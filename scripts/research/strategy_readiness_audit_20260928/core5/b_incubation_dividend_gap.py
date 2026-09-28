"""LP bound: incubation credits no distributions and no borrow (code read). Size of that gap from the baseline ledger,
as % of mean NAV per year, full run and last 3 years."""
import pickle, json
import pandas as pd
import core5_common as c
with (c.CACHE / "baseline_strategy.pkl").open("rb") as fh:
    base = pickle.load(fh)
dl = base["div"].copy(); dl["ex_date"] = pd.to_datetime(dl["ex_date"])
tv = base["results"]["total_value"].astype(float)
bf = base["borrow"].copy(); bf["d"] = pd.to_datetime(bf["accrual_start_date_ts"])
out = {}
for name, start in (("full", tv.index[0]), ("last3y", pd.Timestamp("2023-09-25"))):
    t = tv[tv.index >= start]; yrs = (t.index[-1] - t.index[0]).days / 365.25
    d = dl[dl["ex_date"] >= start]; b = bf[bf["d"] >= start]
    out[name] = {"gross_div_pct_nav_per_year": 100 * d["gross_dividend_cash_float"].sum() / t.mean() / yrs,
                 "net_div_pct_nav_per_year": 100 * d["net_dividend_cash_float"].sum() / t.mean() / yrs,
                 "borrow_pct_nav_per_year": 100 * b["borrow_fee_float"].sum() / t.mean() / yrs}
print(json.dumps(out, indent=1)); c.dump(out, "b_incubation_dividend_gap.json")
