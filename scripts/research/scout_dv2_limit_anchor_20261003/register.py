"""Station S0 of the DV2 limit-anchor study (2026-10-03): one registration, written before any result of this study.

Question (owner), after the limit-entry study (dv2_limit_entry_20261002: on the S&P 500 a day limit buy at
Close_T x (1 - 0.5 x NATR14_T) with a limit exit lifts net Sharpe from 0.26 / 0.77 to 0.75 / 0.86 at a 38% fill rate):
is there a better way to set the limit? (1) anchor at today's OPEN instead of yesterday's close; (2) use the stock's
last-month movement instead of NATR (is it the same thing?); (3) use its DOWNSIDE movement (how far it typically falls
below the open) rather than a symmetric volatility.

Method point: a deeper limit is not "better", only different (fewer, cheaper fills). Every variant is therefore
calibrated to the same realized fill rates (about 25%, 40% and 60%) and compared with Close - k x NATR14 at the SAME
fill rate. The DV2 signal is FROZEN (alpha/scout/specs/dv2.py LIVE_CONFIG). Only the entry limit price changes.

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_anchor_20261003/register.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import Ledger
from alpha.scout.registration import Registration, register, registration_rows

REGISTRATION_ID_STR = "dv2_limit_anchor_20261003"
PARENT_ID_STR = "dv2_limit_entry_20261002"
UNIVERSE_TUPLE = ("S&P 500", "S&P 100", "Russell 1000")
MAIN_UNIVERSE_STR = "S&P 500"
# "close": the limit is anchored at Close_T (can be sent overnight); "open": anchored at Open_(T+1), sent just after the open.
ANCHOR_TUPLE = ("close", "open")
# Offset measures, all known at Close_T:
#   natr14          NATR14_T (Wilder ATR14 / Close, the live ranking measure; the parent study's offset)
#   std21           standard deviation of the last 21 close-to-close returns (rows T-20 .. T)
#   dex21_mean      mean of the last 21 intraday downside excursions (Open_t - Low_t) / Open_t (rows T-20 .. T)
#   dex63_quantile  the stock's own empirical (1 - p) quantile of its last 63 downside excursions (rows T-62 .. T): the
#                   level it fell below with probability p; open anchor (Open_t - Low_t) / Open_t, close anchor
#                   (Close_(t-1) - Low_t) / Close_(t-1)
MEASURE_TUPLE = ("natr14", "std21", "dex21_mean", "dex63_quantile")
FILL_TARGET_TUPLE = (0.25, 0.40, 0.60)
# "limit": the parent study's limit exit (sell limit at the previous close, market-on-open after 5 sessions), run for
# every entry; "moo": today's market-on-open exit, run for the best two entries only (counted here in full).
EXIT_TUPLE = ("limit", "moo")
PARAM_GRID_DICT = {"universe_str": UNIVERSE_TUPLE, "anchor_str": ANCHOR_TUPLE, "measure_str": MEASURE_TUPLE,
                   "fill_target_float": FILL_TARGET_TUPLE, "exit_str": EXIT_TUPLE}
GRID_SIZE_INT = len(UNIVERSE_TUPLE) * len(ANCHOR_TUPLE) * len(MEASURE_TUPLE) * len(FILL_TARGET_TUPLE) * len(EXIT_TUPLE)
PRIOR_TRIALS_INT = 262 + 50 + GRID_SIZE_INT  # the parent's count (262), its grid again (50, conservative), this grid
CALIBRATION_START_STR, CALIBRATION_END_STR = "2004-01-01", "2012-12-31"
BASELINE_TUPLE = ("close", "natr14")  # the parent study's limit, at the same fill target

EXECUTION_STR = (
    "Signal after Close_T (frozen LIVE rule; slots, NATR ranking, sizing V_T / 10 / Close_T shares and the slot rules of "
    "dv2_limit_entry_20261002 unchanged). Entry: a day limit buy for session T+1. Close anchor: L = Close_T x (1 - k x "
    "x_T), rounded DOWN to the $0.01 nominal tick; fills at Open if Open_(T+1) <= L (marketable, half-spread charged), "
    "else at L if Low_(T+1) <= L x (1 - m) (trade-through, no spread), else no fill. Open anchor: the order is placed "
    "right after the opening print, L = min(Open_(T+1) x (1 - k x x_T) rounded DOWN to the tick, Open_(T+1) - one tick); "
    "only the price uses the open, the decision (which names, how many) is made at Close_T; by construction L < Open, so "
    "it fills only by trade-through, Low_(T+1) <= L x (1 - m), at L with no spread. m = max($0.01 / nominal Close_T, 0.1 x "
    "the larger half-spread estimate at T). x_T is the offset measure, known at Close_T. For natr14, std21 and dex21_mean "
    "the multiplier k is set per universe x anchor x measure x fill target by calibration: the k whose book fill rate "
    "(entries / entry orders, limit exit) on 2004-01-01..2012-12-31 is closest to the target (bisection; the calibration "
    "book sees no bar after 2012-12-31); the same k is then run on 2004-2022. For dex63_quantile the limit is L = anchor x "
    "(1 - q_T(p)), with q_T(p) the stock's own (1 - p) quantile of its last 63 excursions (floored at 0), and the target "
    "probability p is calibrated the same way (so the comparison is at equal realized fill rate; the uncalibrated p = "
    "target is reported as a diagnostic, not a trial). Exit: the limit exit of the parent study (sell limit at the "
    "previous close, market-on-open after 5 unfilled sessions) for every entry; market-on-open exit for the best two "
    "entries on the S&P 500 (mean of AR and pooled net Sharpe). Costs: engine commissions on nominal shares ($0.005/share, "
    "$1 minimum, 1% cap); marketable fills pay max(2.5 bp, half-spread of the session before the fill) under BOTH the "
    "per-stock Abdi-Ranaldo and the ADV-bucket pooled half-spread models; passive fills pay commission only. Fill stress: "
    "the trade-through margin re-run with f = 0.1 / 0.5 / 1.0 x the larger half-spread (diagnostic). No dividends.")


def registration() -> Registration:
    return Registration(
        registration_id_str=REGISTRATION_ID_STR,
        family_id_str="us_equity_short_term_reversal",
        hypothesis_str=("A limit offset scaled by the stock's own typical intraday downside excursion, anchored at the open, "
                        "gives better net results than Close - k x NATR14 at the SAME fill rate."),
        mechanism_str=(
            "The close-anchored limit fills partly through the overnight gap (opens below the limit pay the spread and buy "
            "names that gapped down on news); an open-anchored limit only buys an intraday dip below the open, which is the "
            "liquidity-provision event a resting order is paid for. NATR mixes up- and down-moves and overnight gaps; the "
            "downside excursion below the open measures how far a resting buy order has to wait, so at the same fill rate "
            "it should place the limit where the fill is a temporary intraday push rather than a persistent move."),
        expected_sign_and_location_str=(
            "At the same realized fill rate, the open-anchored dex63_quantile / dex21_mean variants have higher net Sharpe "
            "under both half-spread models than close-anchored natr14 on the S&P 500, a lower round-trip cost (no "
            "marketable open fills), and filled events continue at least as well as unfilled ones after the fill day "
            "(Close_(T+1) -> Close_(T+3)); the gain replicates in S&P 100 or Russell 1000. NATR14 and std21 correlate "
            "strongly across signal-day stocks (>= 0.8); the downside excursion less so."),
        hypothesis_class_str="E",
        universe_str=("Norgate point-in-time members, exact flag, from the size-ladder superset panel (05df9965e4063fc6): "
                      "S&P 500 (main, the live universe), S&P 100 and Russell 1000 (replication)."),
        horizon_str="days (pods hold until the exit rule; post-fill-day continuation at Close_(T+1) -> Close_(T+3))",
        schedule_str="daily close decision; the open-anchored price is set just after the open of T+1",
        execution_str=EXECUTION_STR,
        param_grid_dict=PARAM_GRID_DICT,
        primary_metric_str=(
            "Net Sharpe of the costed fast replica under BOTH the Abdi-Ranaldo and the pooled half-spread models, per universe "
            "x anchor x measure x fill target, in sample 2004-01-01 to 2022-12-30 (eras 2004-07 / 2008-15 / 2016-22), and the "
            "paired difference against close-anchored natr14 at the same fill target with a stationary-bootstrap 95% CI "
            "(mean block 20 sessions). Secondary: realized fill rate (calibration window and full), trades per year, CAGR, "
            "max drawdown, cost per round trip, post-fill-day continuation filled vs unfilled, fill-stress Sharpe "
            "(f = 0.1 / 0.5 / 1.0), and the cross-sectional correlation of NATR14 with std21 and with the downside "
            "excursion on signal days. The cost model is an evaluation axis, not a trial axis."),
        kill_criteria_str=(
            "Hypothesis rejected if no open-anchored downside-excursion variant beats close-anchored natr14 at the same fill "
            "target on the S&P 500 under BOTH half-spread models. A variant is a candidate only if it beats the baseline on "
            "the S&P 500 under both cost models AND replicates (beats the baseline under both models) in S&P 100 or Russell "
            "1000; then the per-asset panel MCPT (A8 null) over the registered S&P 500 grid with plateau choice must give "
            "p <= 0.05, and the paired bootstrap CI is reported (a CI that includes 0 means not shown better). A variant "
            "whose gain disappears under strict fills (f = 1.0) is not a candidate."),
        source_str=("Owner question after docs/research/SCOUT_DV2_LIMIT_ENTRY_20261002.md (A17); rule = "
                    "alpha/scout/specs/dv2.py LIVE_CONFIG; execution model = "
                    "scripts/research/scout_dv2_limit_entry_20261002/limit_book.py, entry limit price only."),
        retro_bool=False,
        prior_trials_int=PRIOR_TRIALS_INT,
        parent_id_str=PARENT_ID_STR,
        universe_choice_str=(
            "Chosen after the limit-entry results: the S&P 500 (live universe, where the parent found the robust gain) and "
            "the two large-cap references where it replicated (S&P 100, Russell 1000); small caps dropped because the parent "
            "found them not a pod. Marked as chosen after results. Prior trials: 262 (parent count) + 50 (the parent grid, "
            f"counted again to be conservative) + {GRID_SIZE_INT} (this grid: 3 universes x 2 anchors x 4 measures x 3 fill "
            f"targets x 2 exits; the market-on-open exit is run for 2 entries only but counted in full) = {PRIOR_TRIALS_INT}. "
            "Calibration window 2004-2012; vault sealed: data to 2022-12-30 only."),
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
