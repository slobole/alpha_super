"""Build the Strategy map's data: every engine at PM_READY or above, charts first, from one return frame.

History comes from the frozen fund frame (results/scout/growth_hunt_20261030/shared/frame_main.parquet: one window, one
cost and cash convention) and later sessions from refresh.py (results/bench/engine_map/_work/daily). A module with no column
in that frame takes its history from its newest saved research run that covers the page's start, with the same cash add. The
page holds daily returns and computes everything else in the browser, so the period buttons can change every chart. No
tracked file changes and no backtest runs here. Output: results/bench/engine_map/data.json, served by Bench at /map/data.json,
and build_summary.json beside it. Checks that depend on the repository warn instead of stopping the build: build_summary.json
carries the warnings.

This is a script: running or importing it is the build. Use python -m alpha.bench.engine_map build.
"""
from __future__ import annotations

import base64
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from alpha.bench.engine_map import (
    BUILD_SUMMARY_PATH, DAILY_DIR_PATH as DAILY, DATA_PATH, MAP_BASE_PATH, OUTPUT_DIR_PATH as ROOT, REPO_ROOT_PATH as REPO, WORK_DIR_PATH as WORK,
)
from alpha.bench.engine_map.refresh import RUNS, cash_add, dtb3_rate, house_returns, path_of, saved_run

FROZEN = pd.read_parquet(REPO / "results/scout/growth_hunt_20261030/shared/frame_main.parquet")
PLUS5 = pd.read_parquet(REPO / "results/scout/growth_hunt_20261030/shared/frame_plus5.parquet")
ETF_RAW = pd.read_parquet(WORK / "factor_etf_daily.parquet")
FF = pd.read_parquet(WORK / "factor_french_monthly.parquet")
FF.index = pd.PeriodIndex(FF.index, freq="M")
N = json.loads((WORK / "numbers_factors.json").read_text(encoding="utf-8"))
R = json.loads((WORK / "records.json").read_text(encoding="utf-8"))
T = json.loads((WORK / "taxonomy.json").read_text(encoding="utf-8"))
ROWS = {r["key"]: r for r in N["rows_page_ready"]}
MEASURED = {m["key"]: m for group in T["groups"] for idea in group["ideas"] for m in (idea.get("measured") or [])}
REGISTRY = (REPO / "alpha/strategy_registry.py").read_text(encoding="utf-8")
LEDGER = [json.loads(line) for line in (REPO / "research_ledger/scout_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
FDR_FILES = sorted((REPO / "results/scout/fdr").glob("*/fdr_report.json"))
FDR = json.loads(FDR_FILES[-1].read_text(encoding="utf-8")) if FDR_FILES else {}
FDR_ROWS = {row["registration_id_str"]: row for row in FDR.get("rows", [])}
try:
    TRADES = json.loads((WORK / "trading_facts.json").read_text(encoding="utf-8"))
except (OSError, ValueError):  # written by hand by trading_facts.py; without it the drawer reads not measured
    TRADES = {}
WARNINGS: list[str] = []


def warn(text: str) -> None:
    WARNINGS.append(text)
    print("warning:", text)


def expect(holds: bool, text: str) -> None:
    if not holds:
        warn(text)


LAB = ("2008-03-04", "2026-08-19")
SHADED = [("2008", "2008-09-01", "2009-03-09"), ("2020", "2020-02-19", "2020-03-23"), ("2022", "2022-01-03", "2022-12-30")]
FAMILIES = [("rotation", "Rotation"), ("momentum", "Momentum"), ("dips", "Mean reversion"), ("macro", "Macro and calendar"), ("vol", "Volatility")]

# Stage comes from the registry. These three are live since the owner's switch-on of 2026-10-20, whatever the registry says yet.
LIVE_BY_OWNER = {"btal_qqq", "core5", "etf_dv2"}
# First real-money date as the page counts it. First fills: TAA 3x, CORE5, and NDX VXN in results/scout/mr_test_audit_20261101/t5_real_fills.csv. DV2-IND: its switch-on. BTAL_QQQ: its first trade.
LIVE_SINCE = {"taa3x": "2026-07-01", "ndx_vxn": "2026-06-01", "core5": "2026-10-21", "dv2_ind": "2026-10-20", "btal_qqq": "2026-11-02"}
FILLS_ON_RECORD = {"taa3x", "ndx_vxn", "core5"}; LIVE_WORD = {"dv2_ind": "Switched on", "btal_qqq": "First trade"}  # what that date is, where no fill is on record
# Where the early history is not the traded rule: the drawer's growth chart dashes the line up to this date.
STAND_IN = {
    "taa3x": ("2012-10-02", "Proxy run"), "taa3x_1n": ("2012-10-02", "Proxy run"), "taa2x": ("2012-10-02", "Proxy run"), "btal_qqq": ("2012-10-02", "Proxy run"),
    "dv2_ind": ("2012-02-01", "Early run"), "downshock": ("2013-01-02", "Thin years"),
}
# Rules frozen: the last commit that changed the strategy's trading logic, by engine id. After this date every session is out of sample.
# Filled from the git history on 2026-11-05; a later logic change must move the date, or the line lies.
OLD_RULE = "This repository starts on Mar 11, 2026 and the rule was carried in unchanged: it is older than this date."
FROZEN_SINCE: dict[str, tuple[str, str]] = {
    "taa3x": ("2026-04-04", OLD_RULE), "taa3x_1n": ("2026-04-04", OLD_RULE), "taa2x": ("2026-04-04", OLD_RULE), "btal_qqq": ("2026-04-04", OLD_RULE),
    "ndx_vxn": ("2026-09-27", "Last logic change: ATR in unadjusted dollars and month ends on exchange sessions (corporate-action fix), after the June go-live."),
    "ndx_atr": ("2026-09-27", "Last logic change: ATR in unadjusted dollars and month ends on exchange sessions (corporate-action fix)."),
    "ndx_cap_atr": ("2026-10-04", "Design of record from the sector-cap study; later commits are live wiring."),
    "ndx_cap_natr": ("2026-10-04", "Design of record from the sector-cap study; later commits are live wiring."),
    "dv2": ("2026-03-31", "Last logic change: the 126-day return filter moved from above 0 to above 5%."),
    "dv2_g": ("2026-10-05", "Last logic change: the gate switch reads the previous pricing date. Study spec frozen Oct 3."),
    "hpi": ("2026-07-27", "Unchanged since it entered this repository; the Jul 31 membership loader rewrite claims parity."),
    "hpi_g": ("2026-10-05", "Last logic change: the gate switch reads the previous pricing date. Study spec frozen Oct 3."),
    "capsule": ("2026-10-05", "The later of its two pods' freeze dates."),
    "dv2_g_liq": ("2026-11-02", "Frozen rule in its wiring note; study spec frozen Nov 1."),
    "hpi_g_liq": ("2026-11-02", "Frozen rule in its wiring note; study spec frozen Nov 1."),
    "hpi_rsi": ("2026-07-27", "Unchanged since it entered this repository."),
    "dv2_ind": ("2026-09-27", "Unchanged since it was created."),
    "downshock": ("2026-07-27", "Unchanged since it entered this repository; later commits add timing hooks only."),
    "dispersion": ("2026-07-12", "Created in that commit, which also moved sizing to 1/N."),
    "core5": ("2026-09-27", "Last logic change: month ends on exchange sessions; parameters unchanged since Sep 1. Wiring note: rules unchanged."),
    "compass": ("2026-09-28", "Last logic change: the inflation-expectation series read as published (lag fix). Study spec frozen the same day."),
    "eom": ("2026-09-16", "Unchanged since it was created."),
    "eom_vote": ("2026-11-01", "Study spec frozen Nov 1; the module landed Nov 4 unchanged from it."),
    "mo_leg1": ("2026-10-31", "The owner closed the capsule as three legs on Oct 31; the modules of Nov 3 carry that rule."),
    "mo_leg2": ("2026-10-31", "The owner closed the capsule as three legs on Oct 31; the modules of Nov 3 carry that rule."),
    "mo_leg3": ("2026-10-31", "The owner closed the capsule as three legs on Oct 31; the modules of Nov 3 carry that rule."),
    "mo_capsule": ("2026-10-31", "The owner closed the capsule as three legs on Oct 31."),
    "vrp": ("2026-11-03", "Owner lock and frozen spec of Nov 3."),
    "t1": ("2026-11-05", "Study spec frozen and module added Nov 5; PM READY the same day."),
}
PARKED: set[str] = set()
LIVE_TIP = "Live by your note of Oct 20. The registry still says {tier}."
LIVE_NOTE = {"taa3x": "Account funded Apr to May. First fills Jul 1, June return 0."}
# Added to the alpha cell's own tip, Gross and Net. TFI sits in T-bills: its gross lag is withholding on BIL, its net lag mostly the fee, in points read from its own two alphas.
ALPHA_TIP = {
    "compass": "Lucky peak: 826 trials, honest Sharpe about 0.85.", "dispersion": "Scout: Rejected. Does not beat shuffled prices, p 0.14.",
    "eom_vote": "Plan on Sharpe 0.95: the 2023-on stretch shaped this rule.",
}
ALPHA_NET_TIP: dict[str, str] = {}

SPMO_WORD = "Idle money in SPMO, from Oct 2017"  # first SPMO fill in both saved runs: Oct 30, 2017. Before it such a row is the T-bill rule.
ENGINES = [
    # id, frame key, name, short description, family, versions, ledger registration of its research grade
    ("taa3x", "taa3x", "TAA 3x", "Monthly rotation into 3x Nasdaq", "rotation", [], "taa_3x_reaudition_20261002"),
    ("taa3x_1n", "taa3x_1n", "TAA 3x 1N", "TAA 3x's rotation, equal slots", "rotation", [], "taa_3x_1n_reaudition_20261002"),
    ("taa2x", "taa2x_1n", "TAA 2x", "Equal-slot rotation into 2x Nasdaq", "rotation", [("taa_1n_qld", "No BTAL, 2x Nasdaq"), ("taa_1n_sso", "No BTAL, 2x S&P 500")], "taa_2x_1n_qld_reaudition_20261002"),
    ("btal_qqq", "btal_qqq", "BTAL_QQQ", "Monthly rotation into plain QQQ", "rotation", [("taa_lin_qqq", "No BTAL")], "taa_lin_1n_qqq_reaudition_20261002"),
    ("ndx_vxn", "ndx_vxn", "NDX VXN", "Nasdaq-100 top 10, less in high volatility", "momentum", [], "ndx_vxn_reaudition_20261002"),
    ("ndx_atr", "ndx_atr", "NDX ATR", "NDX VXN's picks, no volatility sizing", "momentum", [], "ndx_atr_reaudition_20261002"),
    ("ndx_cap_atr", "ndx_atr_cap", "NDX cap ATR", "Max 4 names per sector", "momentum", [], "ndx_e2_sector_cap_20261004"),
    ("ndx_cap_natr", "ndx_natr_cap", "NDX cap NATR", "Picks by percent range, max 4 per sector", "momentum", [], "ndx_e2_sector_cap_20261004"),
    ("mo_leg1", "mo_leg1", "Mom leg 1", "Nasdaq-100 momentum by percent range", "momentum", [], "residual_momentum_round21_20261031"),
    ("mo_leg2", "mo_leg2", "Mom leg 2", "Nasdaq-100 momentum beyond the market's", "momentum", [], "residual_momentum_round21_20261031"),
    ("mo_leg3", "mo_leg3", "Mom leg 3", "Russell 1000 momentum beyond the market's", "momentum", [], "residual_momentum_round21_20261031"),
    ("mo_capsule", None, "Mom capsule", "The three momentum legs, a third each", "momentum", [], "residual_momentum_round21_20261031"),
    ("dv2", "dv2", "DV2", "S&P 500 dips, weak two-day close", "dips", [], "dv2_reaudition_20261002"),
    ("dv2_g", "dv2_g", "DV2-G", "S&P 500 dips, only in stressed markets", "dips", [("dv2_g_spmo", SPMO_WORD)], None),
    ("hpi", "hpi_vote", "HPI", "S&P 500 dips, unusually deep fall", "dips", [], "hpi_vote_reaudition_20261002"),
    ("hpi_g", "hpi_g", "HPI-G", "Deep S&P 500 falls, only in stressed markets", "dips", [("hpi_g_spmo", SPMO_WORD)], None),
    ("capsule", None, "MR capsule", "The two stressed-market dip rules, half each", "dips", [], None),
    ("dv2_g_liq", "dv2_g_liq", "DV2-G sized", "DV2-G, more money in more liquid stocks", "dips", [], "growth_mr_products_20261101"),
    ("hpi_g_liq", "hpi_g_liq", "HPI-G sized", "HPI-G, more money in more liquid stocks", "dips", [], "growth_mr_products_20261101"),
    ("hpi_rsi", "hpi_ibs_rsi", "HPI RSI", "Deep S&P 500 falls, sold on a bounce", "dips", [], "hpi_ibs_rsi_reaudition_20261002"),
    ("dv2_ind", "etf_dv2", "DV2-IND", "Dip-buying on 19 industry ETFs", "dips", [], "dv2_industry_etf_reaudition_20261002"),
    ("downshock", "downshock", "Downshock", "Sector ETFs after sharp falls", "dips", [], "sector_ibs_vox_iyr_reaudition_20261002"),
    ("dispersion", "disp", "Dispersion", "Volatile industry ETF dips", "dips", [("disp_xlc", "With XLC"), ("disp_xlc_sma", "With XLC, 200-day filter")], "dispersion_ibs_kie_ihi_sma200_reaudition_20261002"),
    ("core5", "core5", "CORE5", "Five assets, each on its own trend", "macro", [], "core5_reaudition_20261002"),
    ("t1", "t1", "Beyond 60/40 T1", "Stocks, long bonds, gold, each out below its trend", "macro", [], None),
    ("compass", "compass", "Compass", "Sectors by growth and inflation", "macro", [("compass_qqq", "QQQ in the growth cell")], "compass_reaudition_20261002"),
    ("eom", "eom_flow", "EOM", "Month-end stock and bond flow", "macro", [], "eom_reaudition_20261002"),
    ("eom_vote", "eom_vote", "Flow Vote", "Month-end stocks and bonds, by three votes", "macro", [], "eom_tradable_retro_20261101"),
    ("vrp", "vrp", "VRP", "Short volatility through SVXY, VIXM hedge", "vol", [], "vrp_rsi2_locked_retro_20261103"),
]
KEY_OF = {engine_id: key for engine_id, key, *_ in ENGINES if key}
# Fixed mixes of cards above: parts, weights, reset ("month" or "year"), stage, words. The MR capsule's figures on the frozen window are checked below.
BLENDS = {
    "capsule": (["dv2_g", "hpi_g"], [0.5, 0.5], "month", "WIRED", "Two S&P 500 dip rules at half each, buying only in stressed markets, idle money in T-bills."),
    "mo_capsule": (["mo_leg1", "mo_leg2", "mo_leg3"], [1 / 3, 1 / 3, 1 / 3], "year", "WIRED", "The three momentum legs at a third each, reset once a year: Nasdaq-100 by percent range, and Nasdaq-100 and Russell 1000 by momentum beyond the market's. Each decides on a month's last session and 10 sessions before it."),
}
# What an engine does, where the mapping run (numbers_factors.json) has no row, or words that do not match the rule in its module.
WORDS = {
    "taa3x_1n": "TAA 3x's rotation, but each defensive ETF gets an equal 20% slot; failed slots go to 3x Nasdaq or cash.",
    "ndx_vxn": "Monthly: hold up to ten Nasdaq-100 stocks with the best 12-month momentum per unit of daily range, only while SPY is above its 200-day average, and hold less when Nasdaq implied volatility is high.",
    "ndx_atr": "NDX VXN's Nasdaq-100 picks without the implied-volatility shrink.",
    "dv2_g": "DV2's dip-buying, new entries only while a VIX stress gate is open, idle money in T-bills.",
    "hpi_g": "HPI's vote on deep falls, new entries only while the VIX stress gate is open, idle money in T-bills.",
    "eom_vote": "Month-End Flow Vote: hold what month-end rebalancers must buy, SPY or TLT, by three votes. After a strong stock month, long SPY against short TLT for five sessions.",
    "taa3x": "Each month rank five defensive ETFs; strong slots stay defensive, weak slots go to 3x Nasdaq (TQQQ), or cash when realized volatility is above VIX.",
    "etf_dv2": "Buy an oversold industry ETF, from 19 liquid ones, still in a long-term uptrend, at most ten positions of a tenth each. Decide at the close, fill at the next open; sell on the first close above the prior day's high.",
    # module docstrings of strategies/mr_capsule/strategy_mr_*_vix_gated_liq_bil.py, in plain words
    "dv2_g_liq": "DV2-G's dip-buying, new entries only while the VIX stress gate is open, idle money in T-bills; each new entry is a tenth of the portfolio scaled by the stock's trading volume, between half and double.",
    "hpi_g_liq": "HPI-G's vote on deep falls, new entries only while the VIX stress gate is open, idle money in T-bills; each new entry is a tenth of the portfolio scaled by the stock's trading volume, between half and double.",
    # strategies/momentum/momentum_capsule_legs.py and strategies/vix_stuff/strategy_vrp_svxy_vixm_rsi2.py, in plain words
    "mo_leg1": "Nasdaq-100 stocks ranked by 12-month momentum per unit of daily percent range, only while the market is above its trend; half the book set on a month's last session, half 10 sessions before.",
    "mo_leg2": "Nasdaq-100 stocks ranked by momentum left after taking out the market's move (residual momentum), with a rank buffer; half the book set on a month's last session, half 10 sessions before.",
    "mo_leg3": "Russell 1000 stocks ranked by residual momentum, with a rank buffer and sector caps; half the book set on a month's last session, half 10 sessions before.",
    "vrp": "Hold SVXY (short VIX futures) while VIX futures are in contango, more when SPY's two-day RSI is low; buy VIXM when the curve turns. Idle money in T-bills.",
    # strategies/taa_beyond_6040/strategy_taa_beyond_6040_trend_tilt.py, in plain words
    "t1": "Each month weight VTI, TLT and GLD by inverse 63-day volatility; an asset below its 10-month average is left out, its share stays in cash. Cut the whole book when its volatility is above 8.5%, back to 8%. No leverage.",
    "disp": "Any of five industry ETFs that closes near its low on a wide-range day, above its 200-day average, gets a 20% slot.",
}
EOM_OPEN = 0.85
_PLAN_OLD = "Plan of record 0.75 took no rate off; this is over T-bills."
_PLAN_PAIR = f"The 50/50 pair, median of 21 decision days, Sep 2000 to Oct 2026, over T-bills. {_PLAN_OLD} Closed Nov 1 as three legs, plan 0.65. docs/research/RESIDUAL_MOMENTUM_20261031.md:15, :49"
# Plan Sharpe from a study: always shown, even above the Sharpe. Value, its word, and a tip that states window, basis and file. Braces read the engine's Scout card.
PLAN_FROM_STUDY = {
    "taa3x": (0.90, "plan", "Plan of record, about 0.9. Scout card, in sample to 2022, no rate taken off: {best} on the best of 16 decision days, median {med}. docs/research/SCOUT_P5_REAUDITION_20261002.md"),
    "ndx_vxn": (0.60, "median day", f"Median of 21 decision days, Sep 2000 to Oct 2026, over T-bills. {_PLAN_OLD} Scout card, its own window: median {{med}}, best {{best}}. docs/research/RESIDUAL_MOMENTUM_20261031.md:47"),
    "ndx_cap_atr": (0.66, "for the pair", _PLAN_PAIR), "ndx_cap_natr": (0.66, "for the pair", _PLAN_PAIR),
    "core5": (0.80, "plan", "Plan of record: the median of 200 random settings; its own read 0.95. Apr 2008 to 2022, no rate taken off. docs/research/SCOUT_ROBUSTNESS_20261002.md"),
    "compass": (0.85, "in-sample", "Honest in-sample figure, May 2003 to Aug 2026, no rate taken off. Holdout 1983 to 2002: 0.74. docs/research/INFLATION_COMPASS_DEEP_RESEARCH_20260928.md"),
    "mo_capsule": (0.65, "plan", "Plan of record: 11% / 0.65 / -26%. The backtest is the best of 11 day pairs, median Sharpe 0.72. docs/research/RESIDUAL_MOMENTUM_20261031.md"),
    "t1": (0.65, "plan", "Over cash, 1970 to 2022 development sample with synthetic history before 2004, idle cash earning Fed funds (0.61 in T-bills, 0.30 at 0%); the 2023 to 2026 vault 0.50. docs/research/BEYOND_6040_20261105.md"),
    "vrp": (0.70, "plan", "Plan of record about 0.70, not the backtest's 0.93: the RSI2 boost, the threshold and VIXM were chosen after 460 trials. docs/research/VRP_RSI2_BUILD_20261103.md"),
    "eom_vote": (0.95, "plan", "Plan of record: what is left if the vote fades, always TLT in the final week plus the pair. 2003 to Oct 30, 2026, idle cash at 0%, no rate taken off. docs/research/EOM_TRADABLE_20261101.md"),
}
# Next 10% in Build, and the line under it for a mix holding the engine: a lower Sharpe to say alongside the gain, only under Since 2008.
FADE = {"eom": {"sharpe": EOM_OPEN, "word": "if filled at the open", "label": "open-fill Sharpe"}, "eom_vote": {"sharpe": 0.95, "word": "at its plan Sharpe", "label": "plan Sharpe"}}
# Next 10% again: month-end flow is one idea, so its older rule is never a candidate, and a mix holding either rule takes neither.
NOT_CANDIDATE = {"eom": "Flow Vote is the newer rule of this family"}
SAME_IDEA = {"eom": "month-end flow", "eom_vote": "month-end flow",
             # the sized pods are DV2-G and HPI-G with other entry sizes: one pod per engine
             "dv2_g": "DV2-G", "dv2_g_liq": "DV2-G", "hpi_g": "HPI-G", "hpi_g_liq": "HPI-G",
             # the capsule and its legs are one engine: a mix holds the capsule or legs, never both
             "mo_leg1": "momentum capsule", "mo_leg2": "momentum capsule", "mo_leg3": "momentum capsule", "mo_capsule": "momentum capsule"}
# Each module declares its execution policy (execution_policy_str, default timing): MOC for these two, MOO for every other engine, the capsule's two legs included.
CLOSE_AUCTION = {"eom", "eom_vote"}
MENU_NOTE = "Growth weights are proposals, not decided."
PORTFOLIO_TIP = f"Menu of Nov 1. {MENU_NOTE} The products page and portfolio files still hold the older cards."
_NOT_DECIDED = "Recommended Nov 1, not yet decided."
# A preset's tip is its figures on the frozen window plus these words. Only the owner's account has no record of those figures, so its tip calls them the frame's and quotes its real record.
# Sources: the cards in records.json, the account in docs/research/PORTFOLIO_VANILLA_REFRESH_20260930.md.
PRESET_NOTE = {
    "Growth": f"{_NOT_DECIDED} Products page card: 18.3%, 1.27, −13.0%.", "Aggressive": f"{_NOT_DECIDED} Products page card: 23.4%, 1.19, −18.0%.",
    "Growth, next stage": _NOT_DECIDED, "Growth Plus, next stage": _NOT_DECIDED, "Growth, fund scale": _NOT_DECIDED,
    "TAA 3x + NDX VXN": "Its record, Oct 2012 to Sep 2026, no reset, zero-rate Sharpe: 21.1%, 1.35, −15.3%.",
}
PRESET_HEAD = {"TAA 3x + NDX VXN": "This page's series"}
_CARD_NOTE_CAP = "The pair, to Oct 2: the sector cap passed its registered rule, Max DD −24.8% to −21.5%, Sharpe within 0.05. No shuffled-price test of its own."
_CARD_NOTE_GATE = "Own test: the stress gate beats 99.8% of 2,000 placebo gates in the Growth portfolio; alone it adds no Sharpe."
GRADE_WORD = {"growth_mr_products_20261101": "Candidate, Nov 1, as the MR form of the Growth products"}
CARD_NOTE = {"ndx_cap_atr": _CARD_NOTE_CAP, "ndx_cap_natr": _CARD_NOTE_CAP, "dv2_g": _CARD_NOTE_GATE, "hpi_g": _CARD_NOTE_GATE, "capsule": _CARD_NOTE_GATE}
TRADES_NOTE = {"capsule": "Half DV2-G, half HPI-G", "mo_capsule": "A third each in Mom leg 1, Mom leg 2 and Mom leg 3"}
TRADES_OPENS = {"capsule": ["dv2_g", "hpi_g"], "mo_capsule": ["mo_leg1", "mo_leg2", "mo_leg3"]}  # engines named in the note above, each opens its own drawer
# Stage words, owner's definitions of 2026-10-10: alpha/strategy_registry.py lines 15 to 36. A tier is a claim about plumbing, not about edge.
_PLUMBING = "Says the plumbing holds, not that it makes money."
CANDIDATE_TIP = "Passed the Scout research gate; not portfolio-ready yet. Says the research held, not that it is wired."
STAGE_TIPS = {
    "CANDIDATE": CANDIDATE_TIP,
    "WIRED": f"Connected to the live route, checks passed, ready to deploy. No real money yet. {_PLUMBING}",
    "PM READY": f"A portfolio may hold it. Not connected to the live route. {_PLUMBING}",
}
STAGE_HEAD_TIP = f"LIVE: switched on by the owner. WIRED: connected to the live route, checks passed, no real money yet. PM READY: a portfolio may hold it, not connected. {_PLUMBING} CANDIDATE: {CANDIDATE_TIP}"
SCOPE_NOTE = "On this map: every rule at PM READY or above, and every Scout candidate below it with a saved run. Other research rules are not shown."
SCOUT_TIP = "The research gate's verdict, separate from the stage: Candidate, Watchlist or Rejected."
# The place a rule takes in Scout's T-bill test, on the 60/40 of TAA 3x and NDX VXN (alpha/scout/reaudit.py line 50): the rotations TAA 3x's, every other rule NDX VXN's (slot_str in scripts/research/scout_p6_reaudition_20261002/run_p6a.py and its siblings). The ledger holds no name.
SLOT_NAME = {"rotation": "TAA 3x's 60%"}
SLOT_OTHER = "NDX VXN's 40%"

# One line per caveat, at most 4, fund scale only. Braces are filled at build time from the data (see caveat_values).
_SCOUT = "Scout card, in sample to 2022, no rate off"
_PLUS5 = "Sharpe {p5} at +5 bp a side, to Aug 19."
_PROXY_THEM = "Before Oct 2012 a proxy run, with stand-in funds until they listed in 2010-11: about 0.5 to 1.3 points a year too high. {dd_them}"
_PROXY_IT = "Before Oct 2012 a proxy run, with a stand-in fund until it listed in 2011: about 0.5 to 1.3 points a year too high. {dd_it}"
_NDX_CAP_CHOSEN = "The pair was chosen after about 45 trials. Over T-bills, median of 21 days: Sharpe 0.66, not 0.73 at month-end."
_NDX_CAP_REST = [
    ("Unproven", "Pair, picks against QQQ held at the same exposure: better on 71% of paths, Sharpe +0.07."),
    ("Optimistic", "Today's sector labels are used for all history. Not measurable."),
]
# The owner closed the capsule on Nov 1, 2026: docs/research/RESIDUAL_MOMENTUM_20261031.md:15 (three legs, plan 0.65), :296 (dollar ATR book out).
_NDX_CAP_ATR = [
    ("Optimistic", f"{_NDX_CAP_CHOSEN} Closed Nov 1 as three other legs, plan 0.65; this dollar-ATR book is out."),
    *_NDX_CAP_REST,
    ("Design", "Since 2023 {y_own}% a year here, {y_vxn}% in NDX VXN, which has no cap. The 40% sector cap lags technology booms (pair, to Oct 2: 19.7% vs 24.9%) and saved {cap_saved} points Jul 1 to Oct 22, 2026."),
]
_NDX_CAP_NATR = [
    ("Optimistic", f"{_NDX_CAP_CHOSEN} Closed Nov 1 as three legs, plan 0.65; its NATR rule stays as one leg."),
    *_NDX_CAP_REST,
    ("Design", "Since 2023 {y_own}% a year here, {y_vxn}% in NDX VXN, which has no cap and ranks differently. The 40% sector cap lags technology booms (pair, to Oct 2: 19.7% vs 24.9%) and saved {cap_saved} points against the same rule uncapped, Jul 1 to Oct 22, 2026."),
]
_MR_TRIALS = "Gate and parking were chosen after about 100 variants; 410 trials counted. Deflated Sharpe (Bailey and Lopez de Prado) 90% on the whole history, 65% since 2018: it discounts the trials, it does not show an economic edge."
_MR_GATED = [
    ("Optimistic", _MR_TRIALS),
    ("Optimistic", _PLUS5),
    ("Regime", "2020-21 are {c_gain}% of the capsule's gain since 2008; without them its Sharpe falls from {c_s} to {c_ex}."),
]
# What the cap was worth on each book, Jul 1 to Oct 22, 2026: -20.5% against -27.5%, -19.8% against -33.6%. docs/research/MOMENTUM_CAPSULE_DD_REVIEW_20261023.md:172-176
CAP_SAVED = {"ndx_cap_atr": "7", "ndx_cap_natr": "14"}
# The sized capsule's caveats on record: docs/strategies/book-strategy-caveats.md, "MR capsule with liquidity sizing".
_MR_SIZED = [
    ("Optimistic", "Chosen among 13 sizing rules on history already studied: family p 0.06. Plan on half its gain over equal entries."),
    ("Optimistic", "The liquidity part of the gain is concentrated in time: t 1.7, flat without 2020 and 2024-26."),
    ("Risk", "One name reaches about 24% of the pod, against about 13% with equal entries."),
]
_MOM_LEG = [
        ("Optimistic", "Chosen over 21 research rounds and about 3,000 series, no holdout, no deflated Sharpe; kept by the owner's decision after missing the frozen bar twice."),
        ("Optimistic", "No measurable alpha since 2017: over the S&P 500, 6.0% a year, t 1.52; from 2022 t 0.82."),
        ("Live risk", "Legs share names: on 85% of swap days since 2021 two legs hold opposite orders, and IBKR cancels the later in related accounts."),
        ("Note", "Bench costs 2.5 bp a side and 0% cash; the research record used 5 bp plus half-spread and T-bill cash."),
    ]
CAVEATS = {
    "mo_leg1": [*_MOM_LEG[:3], ("Optimistic, small", "No guard against a pinned price, unlike legs 2 and 3: 0.01 to 0.035 of Sharpe in research.")],
    "mo_leg2": _MOM_LEG,
    "mo_leg3": [*_MOM_LEG[:3], ("Limit", "Leg 3's decision peaks near 7.8 GB of memory; the VPS needs 16 GB.")],
    "mo_capsule": [
        ("Optimistic", "Plan on 11% / 0.65 / -26%: the backtest is the best of 11 day pairs, median Sharpe 0.72, deepest -30.5%."),
        ("Optimistic", "Sharpe by era: 2000-07 0.54, 2008-12 0.78, 2013-21 1.07, 2022-26 0.43. No alpha since 2017, t 1.52."),
        ("Note", "Does not earn its slot in Growth: ahead on 0.33 of paired paths against the bar of 0.80."),
        ("Live risk", "Opposite orders between legs on 85% of swap days: one account for the capsule, or ask IBKR, before paper."),
    ],
    "t1": [
        ("Optimistic", "Before Nov 2004 its study history is synthetic and the 1970s lean on one gold bull; this page shows the ETF run only."),
        ("Regime", "Insurance: trails the plain rule when stocks and bonds do not fall together, about 0.7 points a year 2008 to 2022; vault 2023 to 2026 7.9% vs 11.2%."),
        ("Regime", "Loses to 60/40 in long bull decades: Sharpe 0.44 vs 0.87 in the 1990s, 0.97 vs 1.38 in the 2010s."),
        ("Design", "The edge needs idle cash in T-bills: Sharpe 0.61 in T-bills, 0.30 with cash at 0%, below 60/40 0.45. Not the same engine as CORE5 (correlation 0.61)."),
    ],
    "vrp": [
        ("Optimistic", "RSI2 boost, 0.98 threshold and VIXM chosen after 460 trials on 2011-2026: plan Sharpe about 0.70, not 0.93."),
        ("Tail", "A doubling of front VIX futures is about -50% for SVXY, about -15% for the pod at a typical 0.3 weight."),
        ("Live risk", "One missed session is expensive: a day late, Sharpe 0.93 to 0.62 and Feb 2018 -3% to -23%."),
        ("Limit", "SVXY turnover sets capacity: 15 bp a side at a $300-400K pod, about a $3-4M fund at 10 points. Fill cost not measured."),
    ],
    "dv2_g_liq": [*_MR_SIZED, ("Optimistic", "Capacity at the open: Bench suggests about $250K a pod, the route model's $5M for the book is too high.")],
    "hpi_g_liq": [*_MR_SIZED, ("Optimistic", "Capacity at the open: Bench suggests about $100K a pod, the route model's $5M for the book is too high.")],
    "taa3x": [
        ("Optimistic", _SCOUT + ": month-end is the best of 16 possible trading days, {c_best} against a median of {c_med}. Plan on Sharpe about 0.9."),
        ("Optimistic", _PROXY_THEM),
        ("Optimistic", "BTAL and the TQQQ fallback were chosen after seeing 2012-2026: 3rd of 48 sibling rules."),
        ("Conservative", "Commissions on split-adjusted shares averaged about 0.3 points a year too high over 2012-2026, near zero now."),
    ],
    "taa3x_1n": [
        ("Optimistic", _SCOUT + ": Sharpe {c_med}, the median of 16 decision days; {c_best} on the best."),
        ("Optimistic", _PROXY_THEM),
        ("Optimistic", "Same search as TAA 3x."),
        ("Conservative", "Commissions on split-adjusted shares averaged about 0.4 points a year too high over 2012-2026, near zero now."),
    ],
    "taa2x": [
        ("Optimistic", _SCOUT + ": Sharpe {c_med}, the median of 16 decision days; {c_best} on the best."),
        ("Optimistic", "Before Oct 2012 a proxy run, with a stand-in fund until it listed in 2011: its sibling rules read about 0.5 to 1.3 points a year too high. {dd_it}"),
        ("Conservative", "Commissions on split-adjusted shares run about 0.1 points a year too high."),
    ],
    "btal_qqq": [
        ("Optimistic", _SCOUT + ": Sharpe 1.02 on the day used, {c_med} the median of 16 days, {c_best} on the best."),
        ("Optimistic", _PROXY_IT),
        ("Optimistic", "QQQ as the fallback has a hindsight lead over SPY: the SPY form gives up 0.7 to 0.8 points a year in Defensive Target."),
    ],
    "ndx_vxn": [
        ("Optimistic", "Plan of record 0.75, 12%, −30% took no rate off. Over T-bills, median day: 0.60. The model has since reached {dd}."),
        ("Unproven", "Sep 2000 to Oct 2026, stock picks against QQQ held at the same exposure: Sharpe 0.80 vs 0.79, better on 54% of paths."),
        ("Optimistic", "151 settings tried; best-of-many test p 0.61."),
        ("Fading", "Alpha after QQQ and its 200-day rule: {a08}% a year (t {t08}) since 2008, {a13}% (t {t13}) since Sep 2013."),
    ],
    "ndx_atr": [
        ("Optimistic", _SCOUT + ": Sharpe {c_med}, the median of 16 decision days; {c_best} on the best. NDX VXN on the same card: {vxn_med}."),
        ("Unproven", "Same picks as NDX VXN, not proven against QQQ held at the same exposure."),
        ("Optimistic", "Same search as NDX VXN: 151 settings, best-of-many test p 0.61."),
    ],
    "ndx_cap_atr": _NDX_CAP_ATR,
    "ndx_cap_natr": _NDX_CAP_NATR,
    "dv2": [
        ("Optimistic", "Cost-sensitive: " + _PLUS5 + " " + _SCOUT + ": {c_double} at double costs plus 10 bp."),
        ("Conservative", "Commissions on split-adjusted shares run about 0.9 points a year too high."),
        ("Requirement", "Needs margin on 78% of entry days, up to 97% of the portfolio."),
    ],
    "dv2_g": _MR_GATED,
    "hpi_g": _MR_GATED,
    "capsule": [
        ("Optimistic", _MR_TRIALS),
        ("Optimistic", "Sharpe {p5} at +5 bp a side, to Aug 19, and no real fills yet."),
        ("Regime", "2020-21 are {c_gain}% of its gain since 2008; without them its Sharpe falls from {c_s} to {c_ex}."),
        ("Open", "Sizing and the real-cost test are still open."),
    ],
    "hpi": [
        ("Optimistic", _PLUS5),
        ("Weak signal", "Signal test on dates: t 0.8, placebo p 0.36."),
        ("Optimistic", "Chosen from a 20-rule sweep."),
        ("Conservative", "Commissions on split-adjusted shares run about 0.3 points a year too high."),
    ],
    "hpi_rsi": [
        ("Optimistic", _PLUS5),
        ("Weak signal", "Signal test on dates: t 0.8, placebo p 0.32."),
    ],
    "dv2_ind": [
        ("Optimistic", "Before Jan 2010 a research run, about 0.35 points a year too high, Sharpe {s_res}. Jan 2010 to Jan 2012 its own code run before its set start date, {s_mod}. Feb 2012 on {s_after}."),
        ("Optimistic", "Industry ETFs were the best of 4 ETF groups: 1.19 vs 0.87 for all 60. The rule on 29 other ETFs reads 0.78."),
        ("Optimistic", _PLUS5),
        ("Unknown", "Four ETFs listed on Cboe, 19% of order value: no real opening fill on record as of Nov 1; live fills since Oct 20 not read."),
    ],
    "downshock": [
        ("Optimistic", "Before 2013 the ETFs were thin. Max DD {dd} full, {dd12} since Oct 2012."),
        ("Optimistic", "Modelled hidden cost a year, low confidence: at $1M 2.2 points full history, 0.8 over the last 3 years; at $10M 7.9 and 3.5; at $100K 0.51, 0.27 since 2013, 0.12 over the last 3 years."),
        ("Optimistic", "504-cell grid search."),
        ("Optimistic", _PLUS5),
    ],
    "dispersion": [
        ("Failed", "Does not beat shuffled prices: p 0.14. The signal test on dates has the wrong sign: t −1.5."),
        ("Optimistic", "About 24x turnover a year in thin ETFs. Modelled hidden cost a year, low confidence: at $1M 11.3 points full history, 3.4 over the last 3 years; at $10M 37.1 and 11.9; at $100K 3.2 and 0.7."),
        ("Optimistic", "149 combinations; the 200-day gate was not pre-registered."),
        ("Optimistic", _PLUS5),
    ],
    "core5": [
        ("Optimistic", "Apr 2008 to 2022, no rate off: its settings read 0.95, in the top 1% of 200 random draws, median 0.80. Plan on Sharpe 0.80."),
        ("Optimistic", "About 441 trials. The commodity short was kept against its gates."),
        ("Requirement", "Shorts DBC: needs margin and a borrow, modelled at 1% a year."),
        ("Conservative", "Withholding: up to 0.2 points a year."),
    ],
    "compass": [
        ("Optimistic", "Settings sit at a lucky peak: 826 trials, failed the pre-2003 holdout. In sample about 0.85, holdout 0.74."),
        ("Failed", "Below the bar against T-bills in TAA 3x + NDX VXN: better on {c_slot}% of paths, bar 80%."),
        ("Conservative", "Withholding: about 0.6 points a year."),
    ],
    "eom": [
        ("Unmeasured", "Assumes close-auction fills. Sharpe by era (2003-12, 2013-22, 2023 on) at the same day's open 0.85, 0.76, 0.95. An earlier study at the next day's open: 0.80, 0.38, 0.89, eras not on record."),
        ("Optimistic", _PLUS5),
        ("Weak 2026", "Since 2023 Sharpe {s23}. In 2026 {y26}% against T-bills {t26}%."),
        ("Note", "Month-End Flow Vote is the newer rule of this family: plan on Sharpe about 0.95."),
    ],
    "eom_vote": [
        ("Optimistic", "Plan on Sharpe 0.95. The 2023-on stretch shaped the choice of the vote, so 1.26 since 2003 overstates it."),
        ("Unmeasured", "Assumes close-auction fills. Since 2003: Sharpe 0.98 at the open, 0.76 one session late."),
        ("Unmeasured", "Sharpe 1.06 at +5 bp a side, 2003 to Oct 2026."),
        ("Weak 2026", "In 2026 {y26}% against T-bills {t26}%."),
    ],
}
USED_TOKENS = [("TAA 3x 1N", "taa3x_1n"), ("(TAA 3x)", "taa3x"), ("CORE5", "core5"), ("BTAL_QQQ", "btal_qqq"), ("DV2-IND", "dv2_ind"), ("(EOM)", "eom"), ("MR capsule", "capsule"), ("NDX VXN", "ndx_vxn")]
PRODUCT_NAMES = {
    "Defensive Next": "Defensive Next", "Defensive Target (the ideal)": "Defensive Target", "Growth, recommended without momentum": "Growth",
    "Monthly card on the page, also the growth fallback if mean reversion fails its cost gate": "Monthly", "Aggressive, recommended without momentum": "Aggressive",
    "Growth after the month-end route": "Growth, next stage", "Growth Plus after the month-end route": "Growth Plus, next stage",
    "Growth at fund scale, no mean reversion": "Growth, fund scale", "The owner's own account, not a fund product": "TAA 3x + NDX VXN",
}
# Fund menu of 2026-11-01 (memory project_fund_products_no_breakout_20261101), with the numbers each must reproduce on the frozen window.
PRESETS = [
    ("Defensive Next", {"core5": 33.4, "btal_qqq": 33.3, "dv2_ind": 33.3}, (8.8, 1.26, -6.5)),
    ("Defensive Target", {"core5": 25, "btal_qqq": 25, "dv2_ind": 25, "eom": 25}, (10.2, 1.60, -5.3)),
    ("Growth", {"taa3x_1n": 50, "dv2_g": 15, "hpi_g": 15, "core5": 20}, (20.9, 1.26, -15.6)),
    ("Aggressive", {"taa3x_1n": 60, "dv2_g": 15, "hpi_g": 15, "core5": 10}, (22.9, 1.24, -18.1)),
    ("Monthly", {"taa3x_1n": 60, "core5": 40}, (19.8, 1.15, -16.2)),
    ("Growth, next stage", {"taa3x_1n": 40, "dv2_g": 15, "hpi_g": 15, "core5": 15, "eom": 15}, (19.9, 1.38, -12.9)),
    ("Growth Plus, next stage", {"taa3x_1n": 50, "dv2_g": 15, "hpi_g": 15, "eom": 20}, (22.5, 1.36, -14.5)),
    ("Growth, fund scale", {"taa3x_1n": 60, "core5": 20, "eom": 20}, (21.4, 1.26, -15.2)),
    ("TAA 3x + NDX VXN", {"taa3x": 60, "ndx_vxn": 40}, (18.5, 1.12, -17.5)),
]
ETF_COLS = ["QQQ", "IEF", "GLD", "DBC", "UUP"]
FF_COLS = ["Mkt-RF", "SMB", "HML", "MOM"]
TRADE_FIELDS = ("days", "turnover", "names", "namesMax", "hold", "shortPct", "shortDays", "end")
HOLD_ROWS = 10  # the drawer lists the biggest ten; the rest is one "others" row


def raw_holdings(engine_id: str) -> dict | None:
    """What refresh.py saved from the engine's last run; a fixed mix is its parts at their fixed weights."""
    if engine_id in BLENDS:
        parts, weights = BLENDS[engine_id][:2]
        raws = [raw_holdings(part) for part in parts]
        if any(raw is None for raw in raws):
            return None
        now: dict[str, float] = {}
        avg: dict[str, list] = {}
        for raw, weight in zip(raws, weights):
            for asset, w in raw["now"].items():
                now[asset] = now.get(asset, 0.0) + weight * w
            for asset, (w, _days) in raw["avg"].items():
                avg[asset] = [avg.get(asset, [0.0])[0] + weight * w, None]  # days held has no meaning across parts
        return {"date": min(raw["date"] for raw in raws), "start": max(raw["start"] for raw in raws), "now": now, "avg": avg, "mix": True}
    key = KEY_OF.get(engine_id)
    path = DAILY / f"{key}.holdings.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if key else None
    except (OSError, ValueError):
        return None


def asset_word(asset: str) -> str:
    """Norgate marks a delisted stock with its delisting month (DISH-202312); the page shows the ticker."""
    return re.sub(r"-\d{6}$", "", asset)


def top_rows(rows: list[list]) -> tuple[list[list], dict | None]:
    """Biggest first by size; the tail folds into one row with its count and summed weight. Idle cash is never folded: it comes last, on its own."""
    cash = [row for row in rows if row[0] == "Cash" and abs(row[1]) >= 1e-3]
    rows = sorted([row for row in rows if row[0] != "Cash"], key=lambda row: -abs(row[1]))
    head, tail = rows[:HOLD_ROWS], rows[HOLD_ROWS:]
    return head + cash, ({"n": len(tail), "w": round(sum(row[1] for row in tail), 4)} if tail else None)


def holdings_view(engine_id: str) -> dict | None:
    raw = raw_holdings(engine_id)
    if not raw:
        return None
    now_by: dict[str, float] = {}
    avg_by: dict[str, list] = {}
    for asset, w in raw["now"].items():
        now_by[asset_word(asset)] = now_by.get(asset_word(asset), 0.0) + w
    for asset, (w, days) in raw["avg"].items():  # two listings of one ticker (an old delisting and today's) are one row
        prior = avg_by.get(asset_word(asset))
        avg_by[asset_word(asset)] = [w + (prior[0] if prior else 0.0), None if days is None else max(days, prior[1] or 0.0) if prior else days]
    now, now_rest = top_rows([[a, round(w, 4)] for a, w in now_by.items() if abs(w) > 1e-4])
    avg, avg_rest = top_rows([[a, round(w, 4), d] for a, (w, d) in avg_by.items() if abs(w) > 1e-5])
    return {"date": raw["date"], "start": raw["start"], "mix": bool(raw.get("mix")), "now": now, "nowRest": now_rest, "avg": avg, "avgRest": avg_rest}


# ----------------------------------------------------------------- the return frame: frozen history plus refreshed sessions
def lab(series: pd.Series) -> pd.Series:
    return series.loc[LAB[0]:LAB[1]].dropna()


def saved_history(key: str) -> pd.Series:
    """A module with no frozen column: the newest saved research run that covers the page's start, same fair-cash add, cut to that start."""
    stem = RUNS[key][0].rsplit(".", 1)[1]
    try:
        state = saved_run(stem, covers=LAB[0])[1]
    except FileNotFoundError:  # its saved run starts after the page's start: it shows from its own first session
        state = saved_run(stem)[1]
    path = path_of(state["results"])
    returns = house_returns(path) + cash_add(path, dtb3_rate(path.index)).reindex(path.index).fillna(0.0)
    return returns.loc[LAB[0]:].dropna()


def extended(key: str) -> tuple[pd.Series, str | None]:
    """Saved history (the frozen fund frame where it has the column); later sessions from the refreshed run, with the same fair-cash add. Empty, with no end, when neither exists."""
    if key in FROZEN.columns:
        frozen = lab(FROZEN[key])
    else:
        try:
            frozen = saved_history(key)
        except Exception as exc:  # a saved run can be missing or unreadable; the night still goes on
            warn(f"{key}: no saved history, {exc!r}")
            frozen = pd.Series(dtype=float)
    series = frozen
    path_file = DAILY / f"{key}.parquet"
    if path_file.exists():
        path = pd.read_parquet(path_file)
        returns = house_returns(path) + cash_add(path, dtb3_rate(path.index)).reindex(path.index).fillna(0.0)
        later = returns.loc[returns.index > frozen.index[-1]].dropna() if len(frozen) else returns.dropna()
        series = pd.concat([frozen, later]) if len(frozen) else later
    return series, str(series.index[-1].date()) if len(series) else None


def bench_series() -> tuple[pd.Series, pd.Series]:
    frozen_q, frozen_t = lab(FROZEN["qqq_tr"]), lab(FROZEN["tbill"])
    path_file = DAILY / "_bench.parquet"
    if not path_file.exists():
        return frozen_q, frozen_t
    bench = pd.read_parquet(path_file)
    return (pd.concat([frozen_q, bench["QQQ"].loc[bench.index > frozen_q.index[-1]].dropna()]),
            pd.concat([frozen_t, bench["BIL"].loc[bench.index > frozen_t.index[-1]].dropna()]))


def blend(parts: list[pd.Series], weights: list[float], reset: str = "month") -> pd.Series:
    """A fixed mix, back to its weights at the first session of each month or year: half and half by month reproduces the recorded MR capsule."""
    frame = pd.concat(parts, axis=1).dropna()
    weights = np.array(weights, dtype=float)
    held, values, last = weights.copy(), [], None
    for date, row in frame.iterrows():
        period = date.month if reset == "month" else date.year
        if last is not None and period != last:
            held = weights * held.sum()
        held = held * (1 + row.to_numpy())
        values.append(held.sum())
        last = period
    equity = pd.Series(values, index=frame.index)
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] - 1
    return returns


def monthly_blend(a: pd.Series, b: pd.Series) -> pd.Series:
    return blend([a, b], [0.5, 0.5], "month")


# ----------------------------------------------------------------- factors, same method as the mapping run
def nw_ols(y: pd.Series, x: pd.DataFrame, lag: int, per_year: int) -> dict:
    """OLS with Newey-West errors, same arithmetic as allocator_first_look.newey_west_ols."""
    frame = pd.concat([y.rename("y"), x], axis=1).dropna()
    y_arr = frame["y"].to_numpy()
    x_arr = np.column_stack([np.ones(len(frame)), frame[x.columns].to_numpy()])
    xtx_inv = np.linalg.inv(x_arr.T @ x_arr)
    beta = xtx_inv @ x_arr.T @ y_arr
    resid = y_arr - x_arr @ beta
    score = x_arr * resid[:, None]
    s_mat = score.T @ score
    for k in range(1, lag + 1):
        g = score[k:].T @ score[:-k]
        s_mat += (1.0 - k / (lag + 1.0)) * (g + g.T)
    se = np.sqrt(np.diag(xtx_inv @ s_mat @ xtx_inv))
    spread = frame[x.columns].std()
    return {
        "alpha": float(beta[0] * per_year), "alpha_t": float(beta[0] / se[0]), "r2": float(1.0 - resid.var() / y_arr.var()),
        "beta": {c: float(b) for c, b in zip(x.columns, beta[1:])}, "t": {c: float(b / s) for c, b, s in zip(x.columns, beta[1:], se[1:])},
        "size": {c: abs(float(b * spread[c])) for c, b in zip(x.columns, beta[1:])},
    }


def weekly(series: pd.Series) -> pd.Series:
    return (1 + series.dropna()).resample("W-FRI").prod(min_count=1) - 1


def monthly(series: pd.Series) -> pd.Series:
    out = (1 + series.dropna()).resample("ME").prod(min_count=1) - 1
    out.index = out.index.to_period("M")
    return out


def net_of_fees(r: pd.Series, mgmt: float = 0.02, perf: float = 0.20) -> pd.Series:
    """2% a year and 20% over a high-water mark, crystallised each calendar year: same arithmetic as ga_lib.net_return_series."""
    keep, pre, hwm, out, years = 1 - mgmt / 252, 1.0, 1.0, [], r.index.year
    for t, x in enumerate(r.to_numpy()):
        pre *= (1 + x) * keep
        accrued = perf * max(pre - hwm, 0.0)
        out.append(pre - accrued)
        if (t == len(r) - 1 or years[t + 1] != years[t]) and accrued > 0:
            pre -= accrued
            hwm = pre
    nav = pd.Series(out, index=r.index)
    return nav.pct_change().fillna(nav.iloc[0] - 1)


# The factor file stops where it was downloaded; refresh.py's bench file carries the five funds through the last close.
bench_file = DAILY / "_bench.parquet"
later = pd.read_parquet(bench_file) if bench_file.exists() else pd.DataFrame()
later = later.loc[later.index > ETF_RAW.index[-1]] if len(later) else later
ETF = pd.concat([ETF_RAW, later[ETF_COLS]]) if set(ETF_COLS) <= set(later.columns) and len(later) else ETF_RAW.copy()
level = (1 + ETF["QQQ"].fillna(0)).cumprod()
in_qqq = (level > level.rolling(200).mean()).astype(float).shift(1)
ETF["TREND200"] = in_qqq * ETF["QQQ"] + (1 - in_qqq) * ETF["IEF"]  # 0 mismatches over 1 bp in 6,104 days against the saved column
REG = ("2016-07-01", str(ETF[ETF_COLS].dropna().index[-1].date()))
REG_OLD = ("2016-07-01", "2026-07-31")  # parity asserts stay on this window
qqq, tbill = bench_series()
TB_W = weekly(tbill)  # the frozen T-bill alone stops at Aug 21, which silently drops later weeks
FACTOR_FRAMES = {
    window: (ETF[ETF_COLS + ["TREND200"]].apply(weekly).sub(TB_W, axis=0).loc[window[0]:window[1]], FF.loc[pd.Period(window[0], "M"):pd.Period(window[1], "M")])
    for window in (REG, REG_OLD)
}


def trend_alpha(returns: pd.Series, start: str) -> dict:
    """Alpha a year after QQQ and its 200-day rule alone, weekly over T-bills, Newey-West 4."""
    y_w = (weekly(returns) - TB_W).loc[start:].dropna()
    funds = ETF[["QQQ", "TREND200"]].apply(weekly).sub(TB_W, axis=0).reindex(y_w.index)
    return nw_ols(y_w, funds, 4, 52)


def loadings(returns: pd.Series, window: tuple[str, str]) -> dict:
    etf_w, ff_m = FACTOR_FRAMES[window]
    y_w = (weekly(returns) - TB_W).loc[window[0]:window[1]].dropna()
    y_m = (monthly(returns).reindex(ff_m.index) - ff_m["RF"]).dropna()
    y_net = (weekly(net_of_fees(returns)) - TB_W).loc[window[0]:window[1]].dropna()
    etf = nw_ols(y_w, etf_w[ETF_COLS].reindex(y_w.index), 4, 52)
    trend = nw_ols(y_w, etf_w[ETF_COLS + ["TREND200"]].reindex(y_w.index), 4, 52)
    french = nw_ols(y_m, ff_m[FF_COLS].reindex(y_m.index), 3, 12)
    net = nw_ols(y_net, etf_w[ETF_COLS].reindex(y_net.index), 4, 52)
    trend_net = nw_ols(y_net, etf_w[ETF_COLS + ["TREND200"]].reindex(y_net.index), 4, 52)
    return {
        "etf": etf, "french": french, "trend_alpha": trend["alpha"], "trend_alpha_t": trend["alpha_t"], "net_alpha": net["alpha"], "net_alpha_t": net["alpha_t"],
        "trend_net_alpha": trend_net["alpha"], "trend_net_alpha_t": trend_net["alpha_t"],
    }


def registry_tier(key: str, quiet: bool = False) -> str:
    stem = ROWS[key]["code"].split(", ")[1].strip() if key in ROWS else RUNS[key][0].rsplit(".", 1)[1]
    found = re.search(rf'\.{re.escape(stem)}(?::\w+)?":\s*MaturityTier\.(\w+)', REGISTRY)
    if not found:
        if not quiet:
            warn(f"{stem} has no row in the strategy registry: the page shows it as RESEARCH")
        return "RESEARCH"
    return found.group(1)


def encode(series: pd.Series, index: pd.DatetimeIndex) -> str:
    """Daily returns as little-endian int16 in units of 0.1 bp; -32768 means no value."""
    values = series.reindex(index).to_numpy(dtype=float)
    scaled = np.where(np.isnan(values), -32768, np.clip(np.round(values * 1e5), -32767, 32767)).astype("<i2")
    return base64.b64encode(scaled.tobytes()).decode("ascii")


def frozen_stats(returns: pd.Series) -> dict:
    returns = returns.loc[LAB[0]:LAB[1]]
    excess = returns - FROZEN["tbill"].reindex(returns.index).fillna(0.0)
    equity = (1 + returns).cumprod()
    return {
        "cagr": float(((1 + returns).prod() ** (252 / len(returns)) - 1) * 100), "sharpe": float(excess.mean() / excess.std() * np.sqrt(252)),
        "dd": float((equity / equity.cummax() - 1).min() * 100),
    }


def book(weights: dict, returns_by_id: dict) -> pd.Series:
    """Fixed weights, reset at the first session of each calendar year: the fund study's book model."""
    frame = pd.DataFrame({k: returns_by_id[k] for k in weights}).loc[LAB[0]:LAB[1]].fillna(0.0)
    target = np.array([weights[k] for k in weights], dtype=float)
    target = target / target.sum()
    years = frame.index.year.to_numpy()
    held, out = target.copy(), np.zeros(len(frame))
    array = frame.to_numpy()
    for i in range(len(frame)):
        if i and years[i] != years[i - 1]:
            held = target * held.sum()
        previous = held.sum()
        held = held * (1 + array[i])
        out[i] = held.sum() / previous - 1
    return pd.Series(out, index=frame.index)


def page_data(series: pd.Series) -> pd.Series:
    """A series as the page decodes it, in units of 0.1 bp, so a caveat reads what a tile reads."""
    return (series * 1e5).round() / 1e5


def page_stats(returns: pd.Series, start: str | None = None, end: str | None = None) -> dict:
    """The page's statsOf on one window: CAGR once a year has passed, Sharpe over T-bills with its 1.5% vol floor, Max DD inside the window. Empty under 2 sessions."""
    x = page_data(returns.loc[start:end].dropna())
    if len(x) < 2:
        return {}
    excess = x - page_data(tbill.reindex(x.index).fillna(0.0))
    equity = (1 + x).cumprod()
    vol, spread = float(x.std() * np.sqrt(252)), float(excess.std())
    return {
        "n": len(x), "cagr": float((equity.iloc[-1] ** (252 / len(x)) - 1 if len(x) >= 252 else equity.iloc[-1] - 1) * 100), "vol": vol * 100,
        "sharpe": float(excess.mean() / spread * np.sqrt(252)) if spread > 1e-7 and vol > 0.015 else None, "dd": float((equity / equity.cummax() - 1).min() * 100),
    }


def minus(value: float | None, digits: int = 1) -> str:
    """The page's number format: a real minus sign. A missing value stays visibly missing."""
    return "n/a" if value is None else f"{value:.{digits}f}".replace("-", "−")


def t_value(value: float) -> float:
    """A t for data.json: four decimals, so the page's test of 2 or more reads it unrounded. The page prints one."""
    return round(value, 4)


def regression_end(last: pd.Timestamp) -> pd.Timestamp:
    """The last session of the regression's last week, never past `last`. A week enters once its Friday label has passed, so a close on Thursday still ends a week early."""
    return min(ETF[ETF_COLS].dropna().loc[:FACTOR_FRAMES[REG][0].index[-1]].index[-1], last)


def month_day(iso: str) -> str:
    return f"{pd.Timestamp(iso):%b} {pd.Timestamp(iso).day}"


def full_date(iso: str) -> str:
    return f"{month_day(iso)}, {pd.Timestamp(iso).year}"


# ----------------------------------------------------------------- Scout card, one engine's checks from the ledger and the multiple-test report
SCOUT_SEAL = "2022-12-30"  # the vault seal: Scout's S4 station measures from the grid's common start to here (alpha/scout/stations/s4_strategy.py)


def number_after(pattern: str, text) -> float | None:
    found = re.search(pattern, str(text))
    return float(found.group(1)) if found else None


def card_run(gate: dict) -> dict:
    """The saved run beside a Scout test: the Sharpe of the very settings the doubled-cost figure stresses, at normal costs, and where that window starts.
    The ledger holds neither: its "best" is the best decision day, which need not be these settings (BTAL_QQQ 1.17 against 1.02). No file, no figure."""
    source = str(gate.get("source_str") or "").replace("\\", "/")
    try:
        saved = json.loads((REPO / source).with_name("summary.json").read_text(encoding="utf-8"))
        return {"net": round(saved["live"]["sharpe_float"], 2), "window": f"{pd.Timestamp(saved['plateau']['common_start_str']):%b %Y} to {pd.Timestamp(SCOUT_SEAL):%b %Y}"}
    except (OSError, ValueError, KeyError) as exc:
        warn(f"{source}: no saved run beside this Scout test, its doubled-cost cell shows no normal-cost figure, {exc!r}")
        return {"net": None, "window": None}


def scout_card(registration: str | None) -> dict | None:
    """Last row wins. A field that is not in the ledger or the report stays null, and the page says Not tested."""
    rows = [row for row in LEDGER if registration and row.get("registration_id_str") == registration]
    tests = {row.get("test_id_str"): row for row in rows if row.get("row_type_str") == "test_result"}
    gate = tests.get("s5_mcpt_gate")
    if gate is None or gate.get("p_value_float") is None:
        return None
    stations = {row.get("station_str"): row for row in rows if row.get("row_type_str") == "station_verdict"}

    def check(station: str, prefix: str) -> list:
        return next((c for c in stations.get(station, {}).get("check_list", []) if str(c[0]).startswith(prefix)), [None, None, None])

    luck, double, slot, shuffled = check("S4", "luck band"), check("S4", "net Sharpe"), check("S6", "T-bill slot"), check("S5", "MCPT")
    fdr = FDR_ROWS.get(registration)
    grade = stations.get("grade", {})
    return {
        "p": round(gate["p_value_float"], 3), "pPass": gate["p_value_float"] <= 0.05, "pParts": shuffled[2] if ";" in str(shuffled[2]) else None,
        "q": round(fdr["by_q_float"], 4) if fdr and fdr.get("by_q_float") is not None else None, "qPass": bool(fdr["discovery_bool"]) if fdr and "discovery_bool" in fdr else None,
        "slotP": number_after(r"P (-?[\d.]+)", slot[2]), "slotPass": None if slot[1] is None else slot[1] == "PASS",
        "slotWith": number_after(r"Sharpe (-?[\d.]+) vs", slot[2]), "slotWithout": number_after(r"vs (-?[\d.]+)", slot[2]),
        "double": number_after(r"^(-?[\d.]+)$", double[2]), **card_run(gate), "median": number_after(r"median (-?[\d.]+)", luck[2]), "best": number_after(r"best (-?[\d.]+)", luck[2]),
        "verdict": grade.get("verdict_str"), "date": str(grade.get("utc_ts_str", ""))[:10] or None,
    }


def latest_grade(registration: str | None) -> str:
    """The last grade row's verdict word for a registration, as Scout wrote it: CANDIDATE, WATCHLIST, REJECTED, or empty."""
    grades = [row for row in LEDGER if registration and row.get("registration_id_str") == registration and row.get("row_type_str") == "station_verdict" and row.get("station_str") == "grade"]
    return str(grades[-1].get("verdict_str") or "").split(" ")[0] if grades else ""


def grade_without_card(registration: str | None) -> dict | None:
    """A grade row with no shuffled-price test behind it, as the ledger words it: WATCHLIST (no card) reads Watchlist, no card."""
    grades = [row for row in LEDGER if registration and row.get("registration_id_str") == registration and row.get("row_type_str") == "station_verdict" and row.get("station_str") == "grade"]
    if not grades:
        return None
    verdict = str(grades[-1].get("verdict_str") or "")
    return {"verdict": verdict.split(" ")[0], "word": GRADE_WORD.get(registration) or verdict.replace(" (", ", ").replace(")", "").capitalize()} if verdict else None


def plan_for(engine_id: str, card: dict | None, plus5: float | None, sharpe: float | None) -> dict | None:
    """One rule, first match wins. None means nothing lower than the engine's Since 2008 Sharpe to plan on."""
    if engine_id in PLAN_FROM_STUDY:
        value, word, tip = PLAN_FROM_STUDY[engine_id]
        return {"v": value, "kind": word, "src": tip.format(med=minus((card or {}).get("median"), 2), best=minus((card or {}).get("best"), 2))}
    if engine_id == "eom":
        return {"v": EOM_OPEN, "kind": "at the open", "src": "Same day's open, mean of 0.85, 0.76, 0.95 by era (2003-12, 2013-22, 2023 on), 0.83 over all of them. Idle cash at 0%, no rate taken off. Month-end report, Nov 1."}
    if sharpe is None:
        return None
    if card and card["median"] is not None and card["best"] is not None and card["median"] < card["best"] and card["median"] < sharpe:
        dated = f" of {month_day(card['date'])}" if card["date"] else ""
        return {"v": card["median"], "kind": "typical day", "src": f"Median of 16 decision days: {card['median']:.2f}, on the best {card['best']:.2f}. Scout card{dated}, in sample to 2022, no rate taken off."}
    if plus5 is not None and plus5 < sharpe:
        return {"v": plus5, "kind": "at +5 bp", "src": "At +5 bp a side, since 2008 to Aug 19, over T-bills."}
    return None


def caveat_values(engine: dict, returns: pd.Series, load: dict) -> dict:
    """What the braces in CAVEATS read, from this build's data."""
    def calendar_2026(series: pd.Series) -> str:  # a fixed year, so the sentence stays true after it ends
        return minus(float((1 + page_data(series.loc["2026-01-01":"2026-12-31"].dropna())).prod() - 1) * 100)

    def since_stand_ins(them: str) -> str:
        return f"Its worst fall, {dd}, is after the stand-in years." if dd == dd12 else f"Max DD {dd} with {them}, {dd12} after."

    dd, dd12 = minus(page_stats(returns).get("dd")) + "%", minus(page_stats(returns, "2012-10-02").get("dd")) + "%"
    card, vxn_card = engine["card"] or {}, next(e["card"] or {} for e in engines if e["id"] == "ndx_vxn")
    capsule = returns_by_id["capsule"]
    log_gain = np.log1p(page_data(capsule.dropna()))
    values = {
        "dd": dd, "dd12": dd12, "dd_them": since_stand_ins("them"), "dd_it": since_stand_ins("it"),
        "p5": minus(engine["plus5"], 2),
        "s_res": minus(page_stats(returns, None, "2010-01-12").get("sharpe"), 2), "s_mod": minus(page_stats(returns, "2010-01-13", "2012-01-31").get("sharpe"), 2), "s_after": minus(page_stats(returns, "2012-02-01").get("sharpe"), 2),
        "s23": minus(page_stats(returns, "2023-01-03").get("sharpe"), 2),
        "y26": calendar_2026(returns), "t26": calendar_2026(tbill.loc[:as_of]),
        "y_own": minus(page_stats(returns, "2023-01-03").get("cagr")), "y_vxn": minus(page_stats(returns_by_id["ndx_vxn"], "2023-01-03").get("cagr")), "cap_saved": CAP_SAVED.get(engine["id"]),
        "tfi20": minus(page_stats(returns, "2020-01-02").get("cagr")), "tb20": minus(page_stats(tbill.reindex(returns.index), "2020-01-02").get("cagr")),
        "alpha": minus(load["etf"]["alpha"] * 100), "t": minus(load["etf"]["alpha_t"]),
        "c_med": minus(card.get("median"), 2), "c_best": minus(card.get("best"), 2), "c_double": minus(card.get("double"), 2), "c_slot": minus(card["slotP"] * 100, 0) if card.get("slotP") is not None else "n/a",
        "vxn_med": minus(vxn_card.get("median"), 2),
        "c_s": minus(page_stats(capsule).get("sharpe"), 2), "c_ex": minus(page_stats(capsule.loc[~capsule.index.year.isin((2020, 2021))]).get("sharpe"), 2),
        "c_gain": minus(float(log_gain.loc[log_gain.index.year.isin((2020, 2021))].sum() / log_gain.sum() * 100), 0),
    }
    if engine["id"] == "ndx_vxn":
        from_2008, from_sep_2013 = trend_alpha(returns, LAB[0]), trend_alpha(returns, "2013-09-01")
        values.update(a08=minus(from_2008["alpha"] * 100), t08=minus(from_2008["alpha_t"]), a13=minus(from_sep_2013["alpha"] * 100), t13=minus(from_sep_2013["alpha_t"]))
    return values


# ----------------------------------------------------------------- assemble
USED: dict[str, list] = {}
for product in R["current_menu"]:
    label = PRODUCT_NAMES.get(product["product"])
    if label is None:
        continue
    for strategy in product.get("strategies", []):
        for token, engine_id in USED_TOKENS:
            if token in str(strategy.get("name")):
                USED.setdefault(engine_id, []).append({"p": label, "w": round(float(strategy["weight"]))})
USED["dv2_g"] = USED["hpi_g"] = [{"p": u["p"], "w": u["w"] / 2} for u in USED["capsule"]]

# shadow.py writes no months when a Sharpe does not beat its benchmark: such a line has nothing for the page to count against.
FORWARD = {row["registration_id_str"]: {"start": row["start_date_str"], "months": round(row["min_trl_months_float"], 1)} for row in LEDGER if row.get("row_type_str") == "shadow_start" and row.get("min_trl_months_float") is not None}
assert set(CAVEATS) == {row[0] for row in ENGINES} and max(len(lines) for lines in CAVEATS.values()) <= 4

# Registry guard: every PM_READY, WIRED or LIVE module is on this page or known to be off it.
found_tiers = re.findall(r'"(strategies\.[\w.]+(?::\w+)?)":\s*MaturityTier\.(\w+)', REGISTRY)
registry_stems = {route.split(":")[0].rsplit(".", 1)[1] for route, tier in found_tiers if tier in ("PM_READY", "WIRED", "LIVE")}
# Off the page by choice, with the words the footer prints for them.
OFF_MAP_OK = {"strategy_mr_dv2_vix_gated_cash": "DV2-G and HPI-G with idle money in cash", "strategy_mr_hpi_vote_vix_gated_cash": "DV2-G and HPI-G with idle money in cash", "strategy_passive_bil": "passive T-bills"}
PAGE_STEMS = {module.rsplit(".", 1)[1] for module, _ in RUNS.values()}
off_map = sorted(registry_stems - PAGE_STEMS - set(OFF_MAP_OK))
# What each promoted module still off the page is, in plain words; a stem not listed reads as itself
PLAIN_NAME = {
    "strategy_mo_capsule_leg1_natr_ndx": "Momentum leg 1, Nasdaq-100 by percent range",
    "strategy_mo_capsule_leg2_residual_ndx": "Momentum leg 2, Nasdaq-100 residual momentum",
    "strategy_mo_capsule_leg3_residual_r1000": "Momentum leg 3, Russell 1000 residual momentum",
    "strategy_vrp_svxy_vixm_rsi2": "VRP, short volatility through SVXY with a VIXM hedge",
}
RESEARCH_RUN_ROOT = REPO / "results/research/strategy"


def off_map_reason(stem: str) -> str:
    """Why a promoted module is not drawn: this page draws only saved research runs, it never runs a backtest."""
    has_run = any((run / f"{stem}.pkl").exists() for run in (RESEARCH_RUN_ROOT / stem / "vanilla_backtest").glob("*"))
    return "not added to this page yet" if has_run else "no saved research run with daily returns yet"


off_map_why = {stem: {"name": PLAIN_NAME.get(stem, stem.removeprefix("strategy_").replace("_", " ")), "why": off_map_reason(stem)} for stem in off_map}
# Scout candidates that are not on the page: the latest grade says CANDIDATE, but there is no strategy module and no saved run to draw.
# Book studies graded CANDIDATE are not strategies: their rules are on the page already or are portfolio weights.
CANDIDATE_OFF = {
    "hunt_a2_round4_20261030": "Intraday breakout on QQQ and three other ETFs",
    "hunt_a3_round4_20261030": "Dip-buying after outsized falls in liquid US stocks",
    "hunt_a5_round4_20261030": "QQQ dip-buying after a multi-day fall",
    "hunt_a8_round4_20261030": "Market-neutral low-volatility and momentum stocks",
}
CANDIDATE_OFF_WHY = "research script only, no strategy module or saved run"
BOOK_STUDIES = {"growth_mr_products_20261101", "mr_position_sizing_20261101", "fund_products_opt_20261102"}
left_out = list(dict.fromkeys(words for stem, words in OFF_MAP_OK.items() if stem in registry_stems))

returns_by_id: dict[str, pd.Series] = {}
loads_by_id: dict[str, dict] = {}
series_for_page: dict[str, pd.Series] = {}
page_names: dict[str, str] = {}
engines = []
for engine_id, key, name, sub, family, versions, registration in ENGINES:
    if key is None:
        parts, weights, reset, stage, words = BLENDS[engine_id]
        if any(part not in returns_by_id for part in parts):
            warn(f"{name}: a part has no history tonight, left off the page")
            continue
        returns = blend([returns_by_id[part] for part in parts], weights, reset)
        end = str(returns.index[-1].date())
        if engine_id == "capsule":
            check = frozen_stats(returns)
            expect(abs(check["cagr"] - 16.8) < 0.1 and abs(check["sharpe"] - 1.02) < 0.01 and abs(check["dd"] + 21.4) < 0.1, f"MR capsule no longer reads 16.8, 1.02, -21.4 on the frozen window: {check}")
        tier = stage
    else:
        returns, end = extended(key)
        if returns.empty:
            warn(f"{name}: no history at all, left off the page tonight")
            continue
        page_names[key] = name
        if key in FROZEN.columns:
            check, recorded = frozen_stats(returns), MEASURED[key]
            expect(abs(check["cagr"] - recorded["cagr_pct"]) < 0.11 and abs(check["dd"] - recorded["max_dd_pct"]) < 0.11 and abs(check["sharpe"] - recorded["excess_sharpe"]) < 0.011, f"{name} no longer matches its record on the frozen window: {check}, {recorded}")
        candidate = latest_grade(registration) == "CANDIDATE"
        tier = registry_tier(key, quiet=candidate)
        stage = "LIVE" if key in LIVE_BY_OWNER or tier == "LIVE" else tier
        # below PM_READY, or not in the registry at all, a rule is on the page only as a Scout candidate
        if stage not in ("LIVE", "WIRED", "PM_READY") and candidate:
            stage = "CANDIDATE"
        expect(stage in ("LIVE", "WIRED", "PM_READY", "CANDIDATE"), f"{name} is {stage} in the registry and not a Scout candidate, below the page's own rule")
        words = WORDS.get(key) or ROWS[key]["words"].rstrip(".") + "."
    returns_by_id[engine_id] = returns
    series_for_page[engine_id] = returns
    load = loadings(returns, REG)
    loads_by_id[engine_id] = load
    version_rows = []
    for version_key, version_name in versions:
        version_returns, version_end = extended(version_key)
        if version_returns.empty:
            continue
        series_for_page[version_key] = version_returns
        page_names[version_key] = f"{name}: {version_name}"
        # a version that stopped before its engine says to when, as an engine does
        version_rows.append({"key": version_key, "name": f"{version_name}, data to {full_date(version_end)}" if version_end < end else version_name, "stage": registry_tier(version_key), "end": version_end})
    if engine_id == "capsule":
        plus5 = frozen_stats(monthly_blend(lab(PLUS5["dv2_g"]), lab(PLUS5["hpi_g"])))["sharpe"]
    elif key is None:
        plus5 = None
    else:
        plus5 = frozen_stats(lab(PLUS5[key]))["sharpe"] if key in PLUS5.columns else None
    stand_in = STAND_IN.get(engine_id)
    card = scout_card(registration)
    if card:
        card["slotName"] = SLOT_NAME.get(family, SLOT_OTHER)
    trades = TRADES.get(engine_id)
    if engine_id in PARKED and trades and not trades.get("turnover"):  # nothing but zeros to print: TRADES_NOTE says it in words
        trades = None
    engines.append({
        "id": engine_id, "name": name, "sub": sub, "fam": family, "stage": "PM READY" if stage == "PM_READY" else stage, "words": words, "end": end,
        "tier": tier, "blend": key is None, "frozen": FROZEN_SINCE.get(engine_id, (None, None))[0], "frozenWhy": FROZEN_SINCE.get(engine_id, (None, None))[1], "live": LIVE_SINCE.get(engine_id), "fillsOnRecord": engine_id in FILLS_ON_RECORD, "liveWord": LIVE_WORD.get(engine_id),
        "liveTip": LIVE_TIP.format(tier=tier.replace("_", " ")) if key in LIVE_BY_OWNER and tier != "LIVE" else LIVE_NOTE.get(engine_id),
        "alphaTip": ALPHA_TIP[engine_id].format(end=full_date(end)) if engine_id in ALPHA_TIP else None, "alphaNetTip": ALPHA_NET_TIP[engine_id].format(fee=abs(load["net_alpha"] - load["etf"]["alpha"]) * 100, net=abs(load["net_alpha"]) * 100, end=full_date(end)) if engine_id in ALPHA_NET_TIP else ALPHA_TIP.get(engine_id), "standIn": {"to": stand_in[0], "word": stand_in[1]} if stand_in else None, "parked": engine_id in PARKED,
        "etf": {c: {"b": round(load["etf"]["beta"][c], 2), "t": t_value(load["etf"]["t"][c])} for c in ETF_COLS}, "etfR2": round(load["etf"]["r2"], 2),
        "ff": {c: {"b": round(load["french"]["beta"][c], 2), "t": t_value(load["french"]["t"][c])} for c in FF_COLS}, "ffR2": round(load["french"]["r2"], 2),
        "alpha": round(load["etf"]["alpha"] * 100, 1), "alphaT": t_value(load["etf"]["alpha_t"]), "alphaNet": round(load["net_alpha"] * 100, 1), "alphaNetT": t_value(load["net_alpha_t"]),
        "alphaTrend": round(load["trend_alpha"] * 100, 1), "alphaTrendT": t_value(load["trend_alpha_t"]), "alphaTrendNet": round(load["trend_net_alpha"] * 100, 1), "alphaTrendNetT": t_value(load["trend_net_alpha_t"]),
        "used": USED.get(engine_id, []), "versions": version_rows,
        "card": card, "gradeOnly": None if card else grade_without_card(registration), "cardNote": CARD_NOTE.get(engine_id), "forward": FORWARD.get(registration),
        "plus5": round(plus5, 2) if plus5 is not None else None, "fade": FADE.get(engine_id),
        "candidate": engine_id not in NOT_CANDIDATE, "candidateWhy": NOT_CANDIDATE.get(engine_id), "idea": SAME_IDEA.get(engine_id),
        "trades": {k: trades.get(k) for k in TRADE_FIELDS} if trades else None, "tradesNote": TRADES_NOTE.get(engine_id), "tradesOpens": TRADES_OPENS.get(engine_id), "holdings": holdings_view(engine_id),
        "fillWord": "Fills in the closing auction" if engine_id in CLOSE_AUCTION else "Fills at the next open",
    })

graded = {}
for row in LEDGER:
    if row.get("row_type_str") == "station_verdict" and row.get("station_str") == "grade":
        graded[row.get("registration_id_str")] = str(row.get("verdict_str") or "").split(" ")[0]
on_page = {registration for *_, registration in ENGINES if registration}
unplaced = sorted(r for r, verdict in graded.items() if verdict == "CANDIDATE" and r not in on_page | set(CANDIDATE_OFF) | BOOK_STUDIES)
expect(not unplaced, f"Scout candidates neither on the page nor listed as off it: {unplaced}")

# the fund products must come out as the fund pages quote them, or the portfolio builder cannot be trusted
for preset_name, weights, expected in PRESETS:
    got = frozen_stats(book(weights, returns_by_id))
    expect(abs(got["cagr"] - expected[0]) < 0.11 and abs(got["sharpe"] - expected[1]) < 0.015 and abs(got["dd"] - expected[2]) < 0.11, f"{preset_name} no longer reproduces its record: {got}, {expected}")

# Month-End Flow Vote has no recorded row: its saved run must read as the study does, on a fixed end date so a later close cannot break this
for start, expected in (("2008-03-04", (13.5, 1.25, -10.0)), ("2012-10-02", (10.6, 1.10, -6.7)), ("2023-01-03", (11.4, 0.96, -5.6))):
    got = page_stats(returns_by_id["eom_vote"], start, "2026-10-30") if "eom_vote" in returns_by_id else {}
    expect(bool(got) and abs(got["cagr"] - expected[0]) < 0.5 and abs(got["sharpe"] - expected[1]) < 0.05 and abs(got["dd"] - expected[2]) < 0.5, f"Flow Vote from {start} no longer reads as its study: {got}, {expected}")

# parity of the regressions with the mapping run, on the same window
for engine_id, key in (("taa3x", "taa3x"), ("eom", "eom_flow"), ("ndx_vxn", "ndx_vxn")):
    old = loadings(returns_by_id[engine_id], REG_OLD)
    expect(abs(old["etf"]["beta"]["QQQ"] - ROWS[key]["etf_loading"]["beta"]["QQQ"]) < 0.015, f"{engine_id}: QQQ beta no longer matches the mapping run")
    expect(abs(old["french"]["beta"]["MOM"] - ROWS[key]["french_loading"]["beta"]["MOM"]) < 0.015, f"{engine_id}: Momentum beta no longer matches the mapping run")
    if engine_id == "ndx_vxn":
        expect(abs(old["etf"]["alpha"] * 100 - 4.2) < 0.05 and abs(old["etf"]["alpha_t"] - 1.0) < 0.05, f"NDX VXN alpha no longer matches the mapping run: {old['etf']['alpha']}")

returns_plus5 = {engine_id: lab(PLUS5[key]) for engine_id, key, *_ in ENGINES if key in PLUS5.columns}
index = pd.DatetimeIndex(sorted(set(qqq.index) | set(tbill.index) | {d for s in series_for_page.values() for d in s.index}))
# The last close is the engines' own: QQQ or T-bills reading a later session cannot move it.
index = index[(index >= pd.Timestamp(LAB[0])) & (index <= max(returns_by_id[e["id"]].index[-1] for e in engines if not e["parked"]))]
as_of = str(index[-1].date())
reg_start, reg_end, ff_months = f"{pd.Timestamp(REG[0]):%b %Y}", regression_end(pd.Timestamp(as_of)), FACTOR_FRAMES[REG][1].index
for engine in engines:
    returns = returns_by_id[engine["id"]]
    own_end, own_month = regression_end(returns.index[-1]), monthly(returns).reindex(ff_months).dropna().index[-1]
    engine["factorWindow"] = None if own_end == reg_end and own_month == ff_months[-1] else f"{reg_start} to {full_date(str(own_end.date()))}; stock factors to {own_month.strftime('%b %Y')}"
    engine["plan"] = plan_for(engine["id"], engine["card"], engine["plus5"], page_stats(returns).get("sharpe"))
    values = caveat_values(engine, returns, loads_by_id[engine["id"]])
    engine["caveats"] = [{"d": direction, "t": text.format_map(values)} for direction, text in CAVEATS[engine["id"]]]


def qqq_falls(returns: pd.Series, floor: float = -0.15) -> list[dict]:
    """QQQ's falls of at least 15% in the window, peak to trough, oldest first."""
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    falls, i, dates = [], 0, equity.index
    values = drawdown.to_numpy()
    while i < len(values):
        if values[i] < 0:
            j = i
            while j + 1 < len(values) and values[j + 1] < 0:
                j += 1
            trough = i + int(np.argmin(values[i:j + 1]))
            if values[trough] <= floor:
                falls.append({"from": str(dates[max(i - 1, 0)].date()), "to": str(dates[trough].date()), "depth": round(float(values[trough]) * 100, 1)})
            i = j + 1
        else:
            i += 1
    return falls


def preset_row(name: str, weights: dict, page: tuple) -> dict:
    """A preset for the page, with its tip: which figures these are, and whether its weights are decided."""
    plus = {k: round(v, 2) for k, v in frozen_stats(book(weights, returns_plus5)).items()}
    record = f"{PRESET_HEAD.get(name, 'On record')}, to {full_date(LAB[1])}: CAGR {minus(page[0])}%, Sharpe {minus(page[1], 2)}, Max DD {minus(page[2])}%. Sharpe at +5 bp a side: {minus(plus['sharpe'], 2)}."
    return {"name": name, "w": weights, "page": list(page), "plus5": plus, "tip": f"{record} {PRESET_NOTE[name]}" if name in PRESET_NOTE else record}


def map_base() -> dict | None:
    """The frozen layout of the same-bets map, written once by hand. It is an input, never rewritten here."""
    try:
        return json.loads(MAP_BASE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        warn(f"map_base.json is unreadable: {exc!r}")
        return None


live_engines = [e for e in engines if e["stage"] == "LIVE"]
still_wired = sum(1 for e in live_engines if e["tier"] == "WIRED")
first_trade_ahead = [e["name"] for e in live_engines if e["live"] and e["live"] > as_of]
tally_tip = "LIVE here means switched on by the owner." + (f" The registry still says WIRED for {still_wired} of them." if still_wired else "") + (f" First trade still ahead: {', '.join(first_trade_ahead)}." if first_trade_ahead else "")
weeks = pd.Series(np.arange(len(index)), index=index).groupby(index.to_period("W-FRI")).last().to_numpy().tolist()
now = datetime.now()
data = {
    "built": f"{now:%b} {now.day}, {now:%H:%M}", "builtIso": now.isoformat(timespec="seconds"),
    "asOf": as_of, "start": str(index[0].date()),
    "days": base64.b64encode(np.array((index - index[0]).days, dtype="<u2").tobytes()).decode("ascii"),
    "weeks": weeks, "families": [{"id": i, "name": n} for i, n in FAMILIES],
    "shaded": [{"name": n, "from": s, "to": e} for n, s, e in SHADED], "falls": qqq_falls(qqq.loc[:as_of]),
    "engines": engines, "r": {k: encode(s, index) for k, s in series_for_page.items()},
    "qqq": encode(qqq, index), "tbill": encode(tbill, index),
    "presets": [preset_row(n, w, p) for n, w, p in PRESETS], "menuNote": MENU_NOTE, "portfolioTip": PORTFOLIO_TIP, "tallyTip": tally_tip,
    "stageTips": {"LIVE": tally_tip, **STAGE_TIPS}, "stageHeadTip": STAGE_HEAD_TIP, "scopeNote": SCOPE_NOTE, "scoutTip": SCOUT_TIP,
    "regWindow": f"{reg_start} to {full_date(str(reg_end.date()))}", "regWindowFF": ff_months[-1].strftime("%b %Y"), "regYears": round((reg_end - pd.Timestamp(REG[0])).days / 365.25),
    "fdrDate": FDR_FILES[-1].parent.name if FDR_FILES else None, "fdrTests": FDR.get("test_count_int"), "fdrRegs": FDR.get("registration_count_int"), "offMap": off_map, "offMapWhy": off_map_why,
    "candidatesOff": [{"name": words, "why": CANDIDATE_OFF_WHY} for words in CANDIDATE_OFF.values()],
    "leftOut": f"Left out by choice: {', '.join(left_out)}." if left_out else None, "mapBase": map_base(),
}
payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
ROOT.mkdir(parents=True, exist_ok=True)
# written beside and then moved, so Bench never serves half a file
staged = DATA_PATH.with_suffix(".json.tmp")
staged.write_text(payload, encoding="utf-8")
staged.replace(DATA_PATH)
data_ends = {e["name"]: e["end"] for e in engines if e["end"] != as_of} | {page_names[v["key"]]: v["end"] for e in engines for v in e["versions"] if v["end"] != as_of}
BUILD_SUMMARY_PATH.write_text(json.dumps({
    "as_of": as_of, "built": data["builtIso"], "engines": len(engines), "sessions": len(index),
    "data_ends": data_ends, "warnings": WARNINGS, "names": page_names,
}, indent=1), encoding="utf-8")

print("data.json", len(payload), "bytes, as of", as_of, "sessions", len(index))
print("stages", {s: sum(1 for e in engines if e["stage"] == s) for s in ("LIVE", "WIRED", "PM READY")})
print("data ends", data_ends)
print("falls", [(f["to"][:7], f["depth"]) for f in data["falls"]])
print("verdicts", {e["name"]: (e["card"] or {}).get("verdict") for e in engines})
print("factor window", data["regWindow"], "| stock factors to", data["regWindowFF"], "| fdr", data["fdrDate"], data["fdrTests"], data["fdrRegs"])
print("registry guard", len(registry_stems), "modules, off the map:", off_map)
print("warnings:", WARNINGS if WARNINGS else "none")
