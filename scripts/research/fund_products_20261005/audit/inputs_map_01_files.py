"""inputs_map audit, step 1: open the raw files (no Norgate). Read-only on MAIN.

Prints the shape / columns / first-last rows of: shelf-rebuild sleeve path + transactions, proxy path + transactions,
MR capsule engine NAV + transactions (cash / bil / parked), capsule PM-book pods, E2 PM-book pods, the portfolio
refresh sleeve csv, DTB3.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
SR = MAIN / "results/research/portfolio/shelf_rebuild_20260929"
CAP = MAIN / "results/research/mr_capsule_build_20261004"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def show(path: Path, **kw) -> pd.DataFrame:
    d = pd.read_csv(path, **kw)
    print(f"\n=== {path.relative_to(MAIN)}  shape={d.shape}")
    print("columns:", list(d.columns))
    print(d.head(3).to_string())
    print(d.tail(2).to_string())
    return d


for f in ("sources/taa3x__path.csv.gz", "sources/taa3x__transactions.csv.gz", "proxy_runs/splice_scaled/taa3x__path.csv.gz",
          "proxy_runs/splice_scaled/taa3x__transactions.csv.gz", "sources/dv2__path.csv.gz", "sources/hpi_vote__path.csv.gz",
          "sources/ndx_vxn__path.csv.gz"):
    show(SR / f)
for f in ("dv2_bil_nav.csv", "dv2_bil_transactions.csv", "hpi_bil_nav.csv", "hpi_bil_transactions.csv", "dv2_cash_nav.csv",
          "hpi_cash_nav.csv", "dv2_parked_nav.csv"):
    show(CAP / f)
show(MAIN / "results/research/portfolio/portfolio_refresh_20260927/sleeve_series_incl_2008.csv.gz")
show(MAIN / "results/research/portfolio/mr_capsule_bil/vanilla_backtest/2026-10-04_115751/pods/pod_mr_dv2_gated_bil/transactions.csv")
show(MAIN / "results/research/portfolio/ndx_e2_sector_cap_5050/vanilla_backtest/2026-10-04_095907/pods/pod_ndx_atr_vxn_sector_cap/transactions.csv")
d = pd.read_csv(MAIN.parent / "1_data" / "DTB3.csv")
print("\nDTB3", d.shape, d.head(2).to_string(), d.tail(3).to_string())
