"""Tactical FI: how much of the return is modeled cash interest, and what an implementable cash leg earns.

Since 2017 the frozen rule is in cash on almost every decision, so the cash-rate assumption is first order.
Variants (engine, frozen decisions, 100K unless stated):
- HEAD: positive cash earns DGS3MO (observation <= T-2), ACT/365, no broker spread
- BIL:  positive cash earns the BIL total-return close-to-close return (a T-bill ETF held instead of cash;
        includes BIL's expense ratio; 0% withholding like the module)
- BIL with 25% withholding approximated by 75% of BIL's distribution yield (upper bound on the tax drag)
Also: share of HEAD's total P&L that is cash interest, and the Sharpe in excess of the causal cash rate.

Output: OUT/tfi_cash_proxy.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import tb_common as tb
import strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd as tfi
from data.norgate_loader import load_price_timeseries
from tfi_checks import m_all, run_on_data


def main() -> None:
    out = {}
    data = tfi.get_tactical_yield_data(tfi.DEFAULT_CONFIG)
    execution_price_df, y, sig, w, cash_ret, snaps = data
    s = run_on_data(data)
    out["head"] = m_all(s)
    nav = tb.nav_ser(s)
    pnl_total = float(nav.iloc[-1] - nav.iloc[0])
    out["cash_interest_total_usd"] = float(s.cash_interest_total_float)
    out["pnl_total_usd"] = pnl_total
    out["dividends_gross_usd"] = float(s.dividend_cash_gross_total_float)
    out["research_metric_basis"] = s.research_metric_basis_dict
    ledger = pd.DataFrame(s.cash_interest_ledger_row_dict_list).set_index("date")
    for start in ("2012-10-02", "2017-01-01", "2023-08-19"):
        sub = ledger.loc[start:]
        nav_sub = nav.loc[start:]
        out[f"cash_interest_share_of_pnl_since_{start}"] = float(
            sub["cash_interest_float"].sum() / (nav_sub.iloc[-1] - nav_sub.iloc[0])
        )

    bil = load_price_timeseries("BIL", adjustment_str="TOTALRETURN", start_date_str="2002-01-01", end_date_str="2026-08-19")
    bil_ret = bil["Close"].astype(float).pct_change()
    idx = cash_ret.index
    bil_ret = bil_ret.reindex(idx)
    out["bil_first_date"] = str(bil.index[0].date())
    # Before BIL exists (2007-05-30) keep the DGS3MO accrual.
    proxy = cash_ret.copy()
    have = bil_ret.notna()
    proxy.loc[have] = bil_ret.loc[have]
    s_bil = run_on_data((execution_price_df, y, sig, w, proxy, snaps))
    out["cash_as_BIL_TR_from_2007_06"] = m_all(s_bil)
    ann_diff = {}
    for start in ("2008-01-01", "2012-10-02", "2017-01-01", "2023-08-19"):
        yrs = (pd.Timestamp("2026-08-19") - pd.Timestamp(start)).days / 365.25
        a = (1 + cash_ret.loc[start:"2026-08-19"]).prod() ** (1 / yrs) - 1
        b = (1 + proxy.loc[start:"2026-08-19"]).prod() ** (1 / yrs) - 1
        ann_diff[start] = {"dgs3mo_accrual_ann_pct": float(a * 100), "bil_tr_ann_pct": float(b * 100)}
    out["cash_leg_annualized"] = ann_diff
    tb.write_json("tfi_cash_proxy.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    main()
