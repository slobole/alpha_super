"""Station S0 of the DV2 limit-entry study (2026-10-02): one registration, written before any result of this study.

Question (owner): the size ladder (dv2_size_ladder_20261002) found DV2's gross event edge largest in small and micro caps
(Russell 2000 lower half +15.4 bp at h3, t 4.2), but a market-on-open round trip there costs about 78 bp, so only large
caps survive net. Would passive LIMIT orders (earning or saving the spread instead of paying it) change that?

The DV2 signal is FROZEN (alpha/scout/specs/dv2.py LIVE_CONFIG: DV2 < 10, Close > SMA200, 126-session return > 0.05,
NATR14 rank, 10 slots, exit Close > High(t-1)). Only the order type changes.

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_entry_20261002/register.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import Ledger
from alpha.scout.registration import Registration, register, registration_rows

REGISTRATION_ID_STR = "dv2_limit_entry_20261002"
PARENT_ID_STR = "dv2_size_ladder_20261002"
UNIVERSE_TUPLE = ("S&P 500", "S&P 100", "Russell 1000", "Russell 2000 lower half", "S&P SmallCap 600")
SMALL_CAP_UNIVERSE_TUPLE = ("Russell 2000 lower half", "S&P SmallCap 600")
# "moo" = today's market-on-open entry; a number k = day limit buy at Close_T x (1 - k x NATR14_T) for session T+1.
ENTRY_TUPLE = ("moo", 0.0, 0.25, 0.5, 1.0)
# "moo" = today's exit (Close_t > High_(t-1), then market-on-open); "limit" = sell limit at the previous close, at most
# 5 sessions, then market-on-open (exact rule in the execution text below and in limit_book.py).
EXIT_TUPLE = ("moo", "limit")
PARAM_GRID_DICT = {"universe_str": UNIVERSE_TUPLE, "entry": ENTRY_TUPLE, "exit_str": EXIT_TUPLE}
GRID_SIZE_INT = len(UNIVERSE_TUPLE) * len(ENTRY_TUPLE) * len(EXIT_TUPLE)
PRIOR_TRIALS_INT = 212 + GRID_SIZE_INT  # the size ladder's count (200 deep research + 12 universes) plus this grid
MAX_EXIT_ATTEMPT_INT = 5
TICK_FLOAT = 0.01
TRADE_THROUGH_SPREAD_FRACTION_FLOAT = 0.1

EXECUTION_STR = (
    "Signal after Close_T (frozen LIVE rule). Entry 'moo': buy at Open_(T+1), half-spread charged. Entry k: day limit buy "
    "L = Close_T x (1 - k x NATR14_T / 100) (NATR as a fraction of price), rounded DOWN to a $0.01 nominal tick. Fill on "
    "T+1: if Open <= L, at Open (marketable in the opening auction: half-spread charged); else if Low <= L x (1 - m) with "
    "m = max($0.01 / nominal Close_T, 0.1 x half-spread_T) (trade-through: a touch does not fill), at L with no spread "
    "charge; else no fill: the order expires, the slot stays empty that session, and the name can be ordered again only "
    "if its signal holds at the next close. Orders go to the top candidates by NATR up to the slots free at the decision "
    "(held names skipped, as live); unfilled orders are not backfilled intraday. Exit 'moo': today's rule. Exit 'limit': "
    "once Close_t > High_(t-1) triggers, the position is committed to exit; each following session s it rests a sell "
    "limit at Close_(s-1) rounded UP to the tick: if Open_s >= limit, filled at Open (half-spread charged); else if "
    "High_s >= limit x (1 + m_(s-1)), at the limit with no spread; else carried; after 5 unfilled sessions it sells "
    "market-on-open on the 6th (half-spread charged). A position with a resting sell keeps its slot until it fills "
    "(only a certain exit, i.e. a forced market-on-open, frees a slot at the decision). Sizing V_T / 10 / Close_T shares "
    "for every entry type. Costs: engine commissions on nominal shares ($0.005/share, $1 minimum, 1% cap); every "
    "marketable fill pays max(2.5 bp, half-spread of the session before the fill) under BOTH the per-stock Abdi-Ranaldo "
    "and the ADV-bucket pooled half-spread models (tick-floored, as the size ladder); passive fills pay commission only. "
    "The trade-through margin m uses the larger of the two half-spread models, so both cost models see the same fills. "
    "No dividends (replica convention).")


def registration() -> Registration:
    return Registration(
        registration_id_str=REGISTRATION_ID_STR,
        family_id_str="us_equity_short_term_reversal",
        hypothesis_str=("Passive limit entries below the close cut DV2's trading cost enough to make the small-cap edge "
                        "net-positive; adverse selection (fills concentrate on names that keep falling) eats part of the saving."),
        mechanism_str=("A resting limit order supplies liquidity instead of taking it, so it saves the half-spread a "
                       "market-on-open order pays, and a lower limit buys the same reversal at a better price. The cost is "
                       "adverse selection: a limit buy fills exactly when the name keeps falling (informed or persistent "
                       "selling) and misses the names that rebound at once, which carry most of the reversal premium."),
        expected_sign_and_location_str=(
            "Average cost per round trip falls with any limit variant (largest saving in the small-cap universes); fill rate "
            "falls as k rises; the h3 excess of filled signals is BELOW that of unfilled signals (adverse selection), more "
            "so for larger k; the net Sharpe gain over the market-on-open baseline is largest in Russell 2000 lower half and "
            "S&P SmallCap 600 and small in S&P 100 / S&P 500."),
        hypothesis_class_str="E",
        universe_str=("Norgate point-in-time members, exact flag, from the size-ladder superset panel (05df9965e4063fc6): "
                      "S&P 500 (the live universe), S&P 100, Russell 1000, Russell 2000 lower half (members of Russell 2000 "
                      "that are also members of Russell Micro Cap on the same session), S&P SmallCap 600."),
        horizon_str="days (event table at h3, the P7 / size-ladder horizon; pods hold until the exit rule)",
        schedule_str="daily close decision",
        execution_str=EXECUTION_STR,
        param_grid_dict=PARAM_GRID_DICT,
        primary_metric_str=("Net Sharpe of the costed fast replica (engine commissions on nominal shares, 1% cap) under BOTH the "
                            "per-stock Abdi-Ranaldo and the pooled half-spread models, per universe x (entry, exit), in sample "
                            "2004-01-01 to 2022-12-30, with the era table 2004-07 / 2008-15 / 2016-22. Secondary: fill rate, "
                            "trades per year, gross Sharpe, CAGR, max drawdown, average cost per round trip (bp), the adverse-"
                            "selection table (h3 excess over same-date regime-eligible members of filled vs unfilled vs all "
                            "signals) and capacity at 1% of ADV for the best variant. The cost model is an evaluation axis, not "
                            "a trial axis."),
        kill_criteria_str=("Hypothesis rejected if no limit variant lowers the average cost per round trip, or if no small-cap "
                           "variant (Russell 2000 lower half or S&P SmallCap 600) has net Sharpe > 0 under BOTH half-spread "
                           "models. A small-cap variant is a candidate only if net Sharpe > 0 under both models AND the per-asset "
                           "panel MCPT (A8 null, 1000 permutations) of the whole entry x exit grid with plateau selection gives "
                           "p <= 0.05. The adverse-selection part is rejected if filled signals do not have a lower h3 excess "
                           "than unfilled ones."),
        source_str=("Own follow-up of the DV2 size ladder (dv2_size_ladder_20261002, report "
                    "docs/research/SCOUT_DV2_SIZE_LADDER_20261002.md); rule = alpha/scout/specs/dv2.py LIVE_CONFIG; no signal "
                    "parameter search. Daily bars cannot see the queue: the fill model is deliberately conservative "
                    "(trade-through margin, no backfill, slot held by a resting sell)."),
        retro_bool=False,
        prior_trials_int=PRIOR_TRIALS_INT,
        parent_id_str=PARENT_ID_STR,
        universe_choice_str=(
            "Chosen after the size-ladder results: the S&P 500 (live), the two large-cap references, and the two small-cap "
            "universes where the ladder found the largest gross edge (Russell 2000 lower half, the bucket with +15.4 bp t 4.2) "
            "or a long S&P history (SmallCap 600). Marked as chosen after results. Prior trials: 212 (size ladder count) + "
            f"{GRID_SIZE_INT} (this grid: 5 universes x 5 entries x 2 exits) = {PRIOR_TRIALS_INT}. Vault sealed: data to "
            "2022-12-30 only."),
        universe_chosen_after_results_bool=True,
    )


def main() -> None:
    ledger = Ledger()
    if REGISTRATION_ID_STR in registration_rows(ledger):
        print("already registered:", REGISTRATION_ID_STR)
        return
    row_dict = register(ledger, registration())
    print("registered", REGISTRATION_ID_STR, "row", row_dict.get("row_id_int"), "ledger", ledger.ledger_path)


if __name__ == "__main__":
    main()
