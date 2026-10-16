"""Station S0, retro: the October 2026 studies that ran outside Scout (design A19).

Ten studies were run in pakal-research between 2026-10-10 and 2026-10-16 with their own frozen specs, but none was
written to the Scout ledger. Their trials therefore did not count toward any family's N. This script registers them
retroactively (retro_bool True, prior_trials_int = configurations already evaluated on real data), so the family
counts used by the DSR and the BY-FDR report include them. No trial rows are written; nothing is re-run.

Counting rules (conservative, as in trials.py: prior trials count as independent):
- prior_trials_int is the number of books / engine runs the study itself evaluated, from its knowledge record or its
  result tables (source noted per row). Exploratory rounds disclosed in a frozen spec are added.
- A study that mixes mechanisms is split by family; an overlap is double counted rather than dropped.
- book_picture_20261014 is not registered: it combines sleeves that are already registered and states no new
  hypothesis (Scout station S6 work, not a family trial).

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_retro_registrations_20261016/register.py [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import Ledger
from alpha.scout.registration import Registration, register, registration_rows

PAKAL_REPORTS_STR = "pakal-research/reports"
KILL_RETRO_STR = "Retro registration of a closed study: its verdict stands; a forward line, if any, is killed by its S8 CUSUM / Cold Blood Index."


def registration_list() -> list[Registration]:
    return [
        Registration(
            registration_id_str="antonacci_gem_retro_20261010",
            family_id_str="tactical_asset_allocation",
            hypothesis_str="Global Equities Momentum (absolute momentum on the S&P 500, then relative US vs ex-US, else bonds) beats the S&P 500 on excess Sharpe, including after its 2014 publication.",
            mechanism_str="Cross-asset momentum persistence plus trend-following exit to bonds in bear markets.",
            expected_sign_and_location_str="GEM excess Sharpe above the S&P 500 1991-07..2026-08 and not below it in 2014-11..2026-08; Holm over HG1-HG3.",
            hypothesis_class_str="W",
            universe_str="US equity, ex-US equity (83/17 developed/EM), bonds; long proxies 1991-2026 and IVV / ACWX / AGG / BIL 2009-2026",
            horizon_str="one month",
            schedule_str="decision at the month-end close",
            execution_str="paper month-end close; executable Open_(T+1), 5 bps one-way",
            param_grid_dict={"lookback_month_int": (3, 6, 9, 12), "safe_asset_str": ("bonds", "tbills", "ust_7_10y"), "intl_proxy_str": ("blend_83_17", "developed_only")},
            primary_metric_str="Paired circular block bootstrap of monthly excess returns (block 6, 10,000 draws), Holm over HG1-HG3",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Concretum Research interview of G. Antonacci (Substack, 2026-10-10); {PAKAL_REPORTS_STR}/antonacci_models_study (tables/gem_diagnostics.csv: 24 configurations). Verdict: diagnostic; post-publication GEM trails the S&P 500 (P 0.02).",
            source_published_date_str="2026-10-10",
            retro_bool=True,
            prior_trials_int=24,
            universe_choice_str="The published GEM asset list; no choice by the owner.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="antonacci_trend_models_retro_20261010",
            family_id_str="time_series_trend_and_breakout",
            hypothesis_str="Public-ingredient reconstructions of Antonacci's SMURF (SPY/QQQ), GLTR (gold) and BADASS (blockchain ETFs + BTC/ETH) Donchian-ensemble trend models reach the claimed Sharpe (1.5-2.2).",
            mechanism_str="Per-asset trend following with Donchian channel ensembles, a stretch-based mean-reversion pause and a cash / FIRM residual.",
            expected_sign_and_location_str="Best grid cell per model within 0.3 of the claimed Sharpe 2005-2026 (BADASS 2018-2026).",
            hypothesis_class_str="W",
            universe_str="SPY, QQQ, IWF, VUG; GLD, IAU; BLOK, BKCH, BITQ, DAPP, GBTC/IBIT, ETHE/ETHA; BIL",
            horizon_str="days to months",
            schedule_str="daily close decision",
            execution_str="Open_(T+1) primary; Close_(T+1) and paper Close_T as robustness",
            param_grid_dict={"model_str": ("SMURF", "GLTR", "BADASS"), "lookback_set_str": ("63_126_252", "42_84_168", "126_252", "126", "252"), "exit_str": ("midline", "half_low"), "theta": ("off", 1.5, 2.0, 2.5)},
            primary_metric_str="Excess Sharpe of the best and of the primary cell per model versus the claim; stationary bootstrap",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"optimalmomentum.com proprietary-model descriptions (accessed 2026-10-10); {PAKAL_REPORTS_STR}/antonacci_models_study (knowledge_record n_trials = 498 engine runs, all reconstructions incl. FIRM, counted here in full). Verdict: claims not reachable (best 0.67 / 0.90 / 0.88).",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=498,
            universe_choice_str="The vendor's published instrument lists; variants declared in the frozen spec.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="antonacci_reversal_overlays_retro_20261010",
            family_id_str="etf_short_term_reversal",
            hypothesis_str="Antonacci's FIRM bond-ETF reversal overlay and Snap Back (1-3 day reversal on about 100 ETFs and mega-caps) win 70-80% of trades with average gain above average loss.",
            mechanism_str="Liquidity provision after sharp short-term ETF declines.",
            expected_sign_and_location_str="Snap Back win rate 70-80% and positive net expectancy; FIRM overlay adds return over BIL.",
            hypothesis_class_str="E",
            universe_str="FIRM: TLT, IEF, LQD, HYG, EMB, JNK, PFF, TIP, MBB, BNDX, AGG; Snap Back: about 100 ETFs and mega-caps",
            horizon_str="1-3 sessions",
            schedule_str="daily close decision",
            execution_str="buy Open_(T+1), exit Open_(T+1+h)",
            param_grid_dict={"zeta": (1.5, 2.0, 2.5), "hold_sessions_int": (1, 2, 3)},
            primary_metric_str="Win rate, net expectancy and excess Sharpe at central costs",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"optimalmomentum.com (accessed 2026-10-10); {PAKAL_REPORTS_STR}/antonacci_models_study (FIRM grid 9 + tables/rec_snapback.csv 36 rows; FIRM also inside the 498 of the trend row, double counted on purpose). Verdict: Snap Back win rate 51.2%.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=45,
            universe_choice_str="The vendor's described universes.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="bond_calendar_flow_family_retro_20261016",
            family_id_str="calendar_and_flow",
            hypothesis_str="Treasury index-extension buying makes long-duration bonds rise in the final five sessions of the month, scaling with duration, with related auction-cycle, FOMC, NFP, coupon and quarter-end effects.",
            mechanism_str="Passive bond-index rebalancing at month end (duration extension) and dealer inventory around Treasury auctions.",
            expected_sign_and_location_str="Final-5 excess return positive for every rate-sensitive ETF and proportional to duration; yields fall on the final 5 sessions in 1985-2026 but not in 1962-84; BH over 26 ETFs, Holm over eras.",
            hypothesis_class_str="E",
            universe_str="26 bond ETFs (Treasury, TIPS, IG, HY, MBS, muni, EM, international, aggregate, hybrid); FRED constant-maturity yields 1962-2026",
            horizon_str="days (calendar windows)",
            schedule_str="calendar known in advance; auction windows no earlier than the announcement close",
            execution_str="close-to-close windows (MOC)",
            param_grid_dict={"window_str": ("eom_final5", "bom_first5", "auction_pre", "auction_post", "fomc", "nfp", "mid_month_coupon", "quarter_end")},
            primary_metric_str="Per-ETF final-5 minus other-day mean excess return (BH), yield-history era tests (Holm), frozen composite gate",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"ZROZ EOM / EOMS public claims; {PAKAL_REPORTS_STR}/bond_calendar_flow_family_study (tables/deflated_sharpe_trials.csv 57 + tables/post_hoc_tlt_composites.csv 60). Verdict: effect real and index-driven, composite gate passed no window; forward hypotheses only.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=117,
            universe_choice_str="Declared ETF list frozen before results; EOM on TLT/ZROZ 2003-2026 was already known from eom_flow_family_study (disclosed in the spec).",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="core5_equity_vehicle_v2_retro_20261011",
            family_id_str="time_series_trend_and_breakout",
            parent_id_str="core5_reaudition_20261002",
            hypothesis_str="Swapping CORE5's SPY sleeve for QQQ (own or SPY signal) or a 3x vehicle (UPRO / TQQQ) improves the book, and a 3x vehicle beats levering the whole CORE5 book to the same volatility.",
            mechanism_str="Per-asset adaptive trend; a higher-Sharpe equity vehicle or concentrated leverage in the equity sleeve.",
            expected_sign_and_location_str="QQQ: Sharpe above SPY with bootstrap P >= 0.80, not lower in >= 3 of 4 eras, sleeve 2000-2026 not below SPY; 3x: beats the equal-risk whole-book twin with P >= 0.80.",
            hypothesis_class_str="W",
            universe_str="CORE5 sleeves (SPY, IEF, GLD, DBC, UUP, BIL) with the equity column swapped to QQQ / UPRO / TQQQ (synthetic before inception), SSO / QLD context",
            horizon_str="one month or state change",
            schedule_str="rebalance on a long-state change or at month end",
            execution_str="decision Close_T, fill Open_(T+1) (alpha/scout/engines/weights.simulate, historical share units)",
            param_grid_dict={"variant_str": ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "X1", "X2")},
            primary_metric_str="Paired circular block bootstrap of the Sharpe difference vs A0 (Holm over 7) and vs the equal-risk levered twin (Holm over 5)",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Owner request 2026-10-11; {PAKAL_REPORTS_STR}/core5_equity_vehicle_v2 (knowledge_record n_trials = 10, plus the post-hoc F7 leverage ladder 4 rungs x 2 sleeves = 8). Verdict: QQQ own-signal and UPRO SPY-signal forward hypotheses; TQQQ downgraded by 2000-02.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=18,
            universe_choice_str="Owner chose the vehicles after the August vehicle study had shown the direction (QQQ up a little, 3x higher CAGR).",
            universe_chosen_after_results_bool=True,
        ),
        Registration(
            registration_id_str="ltr_momentum_retro_20261014",
            family_id_str="equity_cross_sectional_momentum",
            hypothesis_str="Learning-to-rank models (LambdaMART, ListNet, ListMLE, RankNet) on momentum features beat regress-then-rank and plain 12-month momentum in US stocks, as reported by Poh et al. (2020).",
            mechanism_str="Momentum persistence; a ranking loss is claimed to extract it better than a regression loss.",
            expected_sign_and_location_str="LambdaMART Sharpe near the paper's 2.16 on Russell 3000 1996-2020 and above the MLP baseline; Holm in both eras.",
            hypothesis_class_str="X",
            universe_str="Russell 3000, Russell 1000 and a liquid Russell 3000 subset, point-in-time",
            horizon_str="one month",
            schedule_str="after the final Close_T of each month",
            execution_str="Open_(T+1) (paper Open-to-Open); executable layer with costs and T-bill financing",
            param_grid_dict={"model_str": ("Rand", "JT", "Baz", "MLP", "LM", "RankNet", "ListNet", "ListMLE", "XGB_reg_ret", "XGB_reg_dec", "XGB_ndcg", "MLP_dec", "Ridge_dec"), "universe_str": ("r3000", "r1000", "r3000_liquid")},
            primary_metric_str="Walk-forward out-of-sample Sharpe per model (6 blocks), Holm H1/H2, spanning alpha after SREV and UMD",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Poh, Lim, Zohren, Roberts (2020), arXiv 2012.07149; {PAKAL_REPORTS_STR}/ltr_momentum_study (research_state: 44 declared variants + 720 XGBoost trial books with reported OOS Sharpe; the 270 neural fits selected on validation loss are not added). Verdict: not replicated.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=764,
            universe_choice_str="The paper's universe plus two tradability variants declared in the spec.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="megacap_concentration_beta_boost_retro_20261010",
            family_id_str="time_series_trend_and_breakout",
            hypothesis_str="A concentrated biggest-N S&P 500 book (optionally gated by SMA200 and levered) adds CAGR beyond the S&P 500 levered to the same beta.",
            mechanism_str="Mega-cap concentration premium (or regime) plus an index trend gate; the gate is the expected Sharpe source.",
            expected_sign_and_location_str="At least one of 8 primary books Holm-significant vs its beta-matched control, positive in 1991-2008 and 2009-2026, bootstrap P > 0.90 in 2009-2026.",
            hypothesis_class_str="X",
            universe_str="S&P 500 point-in-time (primary), Nasdaq-100 (secondary); French size deciles 1926-2026",
            horizon_str="one month",
            schedule_str="month-end close (weekly-gate variant at week ends)",
            execution_str="Close_(t+1) MOC; Open_(t+1) robustness",
            param_grid_dict={"top_count_int": (3, 5, 10, 20), "weighting_str": ("EW", "SW"), "gate_str": ("none", "SMA200m", "SMA200w"), "stock_filter_str": ("none", "CORE5"), "leverage_str": ("1.0", "1.5", "2.0", "VT30")},
            primary_metric_str="Paired daily difference vs the beta-matched S&P control, Newey-West t, Holm over 8",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Owner goal 2026-10-10; {PAKAL_REPORTS_STR}/megacap_concentration_beta_boost (knowledge_record n_trials = 244) + pakal-research/biggest_n_trend_gate exploratory rounds disclosed in the spec (91 books + 48 SMA cells + 149 grid cells = 288). Verdict: rejected as alpha, accepted as beta.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=532,
            universe_choice_str="S&P 500 is the natural universe; two exploratory rounds on the same data were seen before the freeze (disclosed).",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="ndx_megacap_tilt_sleeve_retro_20261010",
            family_id_str="time_series_trend_and_breakout",
            parent_id_str="megacap_concentration_beta_boost_retro_20261010",
            hypothesis_str="A levered (1.5x) SMA200-gated Nasdaq-100 sleeve with half of the book in the 10 largest members (3-year turnover, CORE5 per stock) beats the levered gated index alone without hurting in 1995-2008.",
            mechanism_str="Index trend gate as the Sharpe source; mega-cap tilt as a post-2009 regime bet sized at half.",
            expected_sign_and_location_str="Tilt 0.5 vs 0 Sharpe difference positive with bootstrap P >= 0.80; 1995-2008 Sharpe not more than 0.05 below; walk-forward OOS Sharpe not below tilt 0 over 2005-2026.",
            hypothesis_class_str="X",
            universe_str="Nasdaq-100 point-in-time (primary), S&P 500 (secondary); index leg QQQ TR / $NDX",
            horizon_str="one month",
            schedule_str="month-end close",
            execution_str="Close_(t+1) MOC; Open_(t+1) sensitivity",
            param_grid_dict={"universe_str": ("ndx", "sp500"), "tilt_float": (0.0, 0.25, 0.5, 0.75, 1.0), "leverage_float": (1.0, 1.25, 1.5, 1.75, 2.0), "size_proxy_sessions_int": (756, 252)},
            primary_metric_str="Paired daily difference vs tilt 0, Newey-West t, Holm over 4; walk-forward tilt choice 2005-2026",
            kill_criteria_str="S8 CUSUM or Cold Blood Index red on the paper shadow; the walk-forward weight falling to 0 for two consecutive years",
            source_str=f"Follow-up of megacap_concentration_beta_boost; {PAKAL_REPORTS_STR}/ndx_megacap_tilt_sleeve (knowledge_record n_trials = 131 books + 2 walk-forward paths). Verdict: promoted to paper shadow by its pre-registered rule (Holm p 0.35, not significant).",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=133,
            universe_choice_str="Nasdaq-100 was chosen because the parent study showed its gated index beating the S&P gated index and the tilt helping after 2009.",
            universe_chosen_after_results_bool=True,
        ),
        Registration(
            registration_id_str="ndx_dip_ladder_tqqq_retro_20261016",
            family_id_str="index_drawdown_dip_buying",
            hypothesis_str="Holding the S&P 500 and moving 2% of the book into TQQQ when the Nasdaq-100 is 5% below its high, plus 2% per further 3% fall, until the index recovers, beats the S&P 500.",
            mechanism_str="Leveraged buying of index drawdowns, paid for bearing crash risk; the alternative is plain leverage.",
            expected_sign_and_location_str="B0 CAGR and excess Sharpe above SPY and above random-timing placebos (Holm over H1-H3), in all three eras.",
            hypothesis_class_str="E",
            universe_str="SPY with NDX -> TQQQ (synthetic before 2010); SPX -> UPRO variant; QQQ / QLD vehicles as controls",
            horizon_str="drawdown episode (months to years)",
            schedule_str="after Close_T of $NDX",
            execution_str="Close_(T+1); same-day and 5-day-delay sensitivity",
            param_grid_dict={"start_dd": (0.05, 0.1, 0.15, 0.2), "step_dd": (0.03, 0.05, 0.1), "tranche": (0.01, 0.02, 0.05)},
            primary_metric_str="Paired stationary block bootstrap of the CAGR and Sharpe differences vs SPY, random-timing placebo, Holm over H1-H3",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Social-media post supplied by the owner 2026-10-16; {PAKAL_REPORTS_STR}/ndx_dip_ladder_tqqq_study (tables/table_variants.csv: 53 variants incl. the 36-cell grid). Verdict: rejected; the excess is leverage.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=53,
            universe_choice_str="The post's rule; controls and variants declared in the frozen spec.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="rangerz_breakout_generator_retro_20261012",
            family_id_str="formulaic_alpha_mining",
            hypothesis_str="The RangerZ generator (Pardo Ranger range-breakout / fade architecture) plus a mechanical evaluation shell selects algos that keep their discovery Sharpe on a locked 2022-2026 holdout.",
            mechanism_str="A mined combination of breakout and fade entries, trend / vol filters and exits; no single mechanism (generator search).",
            expected_sign_and_location_str="Selected 11-algo portfolio holdout central Sharpe >= 0.7, above classic Donchian and the 75th percentile of random 11-job picks.",
            hypothesis_class_str="E",
            universe_str="ES, NQ, GC, BTC (H1/H4), SPY, DIA, DAX, GLD, BTC (D1)",
            horizon_str="intraday to weeks",
            schedule_str="close of bar t-1, resting stop / limit orders for bar t",
            execution_str="resting stop / limit orders",
            param_grid_dict={"entry_str": ("STOP", "LIMIT"), "trend_str": ("NONE", "NEUTRAL", "WITH_MOM", "WITH_SMA", "COUNTER_MOM", "COUNTER_SMA"), "vol_str": ("OFF", "LOW", "HIGH"), "exit_str": ("OPPOSITE", "EXIT_ORDER", "TARGET", "TRAIL"), "direction_str": ("LONG", "BOTH")},
            grid_justification_str="288 jobs per market x 13 markets = 3,744 job-market cells: the vendor's generator space is itself the object under test.",
            primary_metric_str="Locked-holdout Sharpe of the selected portfolio vs random 11-job portfolios and classic Donchian; full-pipeline reality check",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"zorro-project.com/rangerz.php and financial-hacker.com Ranger articles; {PAKAL_REPORTS_STR}/rangerz_breakout_generator_study (3,744 job-market cells). Verdict: selection process rejected (holdout Sharpe -0.22).",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=3744,
            universe_choice_str="Markets picked from the Zorro templates before the run.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="valuelytica_fx_us_hours_retro_20261016",
            family_id_str="calendar_and_flow",
            hypothesis_str="Foreign currencies rise against the dollar during 11:00-15:00 New York, and a weekday JPY rule (short Mon/Tue, long Fri) earns a Sharpe above 1.",
            mechanism_str="Dollar demand into the London 4pm WM/R fix (11:00 NY) released after it: a clock-time flow.",
            expected_sign_and_location_str="Equal-weight basket positive in 2008-2020 with NW t >= 2, same sign in 2021-2025, central-cost Sharpe >= 0.5 in both periods.",
            hypothesis_class_str="E",
            universe_str="15 currencies vs USD (G10 + ILS, MXN, KRW, HUF, PLN, ...); USDJPY / EURJPY / AUDJPY",
            horizon_str="intraday window",
            schedule_str="calendar rule known in advance (clock time and weekday)",
            execution_str="enter 11:00, exit 15:00 New York",
            param_grid_dict={"window_str": ("11-15", "10-16"), "weekday_str": ("all", "Mon", "Tue", "Wed", "Thu", "Fri")},
            primary_metric_str="Basket NW t and central-cost Sharpe by period; BH within the currency and weekday families",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"valuelytica.substack.com 'The Four-Hour FX Trade' and 'YenStrike'; {PAKAL_REPORTS_STR}/valuelytica_fx_us_hours_study (246 declared variants). Verdict: rejected; costs exceed the drift after 2020.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=246,
            universe_choice_str="The posts' currency lists; no intraday FX result had been seen locally before the freeze.",
            universe_chosen_after_results_bool=False,
        ),
    ]


def main() -> None:
    dry_run_bool = "--dry-run" in sys.argv
    ledger = Ledger()
    existing_dict = registration_rows(ledger)
    for reg in registration_list():
        reg.validate()
        if reg.registration_id_str in existing_dict:
            print(f"skip (exists)  {reg.registration_id_str}")
            continue
        if dry_run_bool:
            print(f"would register {reg.registration_id_str}  family={reg.family_id_str}  prior={reg.prior_trials_int}  grid={reg.grid_size_int}")
            continue
        row_dict = register(ledger, reg)
        existing_dict[reg.registration_id_str] = row_dict
        print(f"registered row {row_dict['row_id_int']}  {reg.registration_id_str}  prior={reg.prior_trials_int}")
    print(f"ledger chain OK: {ledger.verify()} rows")


if __name__ == "__main__":
    main()
