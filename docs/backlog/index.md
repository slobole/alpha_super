---
title: Backlog
description: Unprioritized ideas that may be explored later.
document_type: reference
authority: guide
risk_scope: research
status: backlog
---

# Backlog

Ideas worth keeping, with no current priority, schedule, or commitment.

## Israeli-market momentum strategy

**Status:** Unprioritized

**Prerequisite:** The platform must first support the Tel Aviv Stock Exchange,
including reliable data, trading calendars, instruments, corporate actions,
costs, and broker execution.

**Idea:** Research a simple momentum strategy for Israeli-listed securities.

!!! note "Boundary"
    This is a future research idea only. No strategy rules, allocation, PAPER,
    or LIVE implementation have been approved.

## Opening-auction execution cost and impact

**Status:** Unprioritized

**Why:** The 2026-09-28 readiness audit graded the KIE/IHI dispersion pods and pre-2013 VOX/IYR NOT READY on backtest
correctness. The reason is that the house square-root MOO model (`alpha/engine/capacity_analysis.py`) says the
modelled 2.5 bp slippage understates opening-auction cost in thin ETFs. That model's ETF profile is marked
low confidence. Earlier house work concluded that USD 0.005/share fees plus 2.5 bp slippage is a normal *average*
cost. The same model sets the TAA capacity figures: about +0.9 pp/yr at USD 1M and about +2 pp/yr at USD 5M, mostly
BTAL.

**Idea:**
1. Measure the real cost of MOO fills: compare IBKR execution reports of the live pods with the Norgate Open and
   the official opening print.
2. Calibrate the impact model per symbol class (large-cap stock, liquid ETF, thin ETF), and per auction venue (NYSE
   Arca, Nasdaq, Cboe BZX).
3. Re-grade the affected pods and the capacity figures with the calibrated model.

!!! note "Boundary"
    Research only. No change to engine costs or strategy verdicts until the measurement exists.

