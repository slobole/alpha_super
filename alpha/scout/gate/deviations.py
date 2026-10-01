"""Known deviations of the real engine that Scout's parity mode reproduces (design section 7.3).

Parity mode copies the engine exactly, so the identity gate can pass; truth mode removes the deviation, so each
research card can print what the deviation is worth. Scout never silently "fixes" the engine: a material deviation
goes to the engine fix list. Fields follow QUANT_PHILOSOPHY.md ("Dangerous assumptions must fail loud").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineDeviation:
    deviation_id_str: str
    issue_description_str: str
    expected_bias_direction_str: str
    impact_level_str: str
    mitigation_str: str
    affected_strategy_list: tuple[str, ...]


ENGINE_DEVIATION_TUPLE = (
    EngineDeviation(
        deviation_id_str="split_adjusted_share_units",
        issue_description_str=(
            "Per-share commissions and whole-share rounding use split-adjusted (CAPITALSPECIAL) share counts, not the "
            "shares that were actually tradable on the day (`historical_share_units_bool` is False by default). A "
            "stock that later split 10:1 is modelled with 10x the historical share count."
        ),
        expected_bias_direction_str=(
            "A later forward split overstates historical per-share fees (conservative, readiness audit fix #15); a later "
            "reverse split or share consolidation understates them (optimistic)."
        ),
        impact_level_str=(
            "TAA 3x about +0.32 pp/yr of fees (2026-09-28 readiness audit). NDX VXN already uses historical units. "
            "Inflation Compass (both variants): 0.000 pp/yr on 2003-04..2026-09 (2026-10-01): no per-share commission, "
            "so the XLE / XLK / XLU 2:1 splits change whole-share rounding only. "
            "CORE5 -0.013 pp/yr (optimistic): BIL's 1-for-2 consolidation (UnadjClose / Close = 0.5 before 2017-11-30) halves its earlier ledger share counts "
            "(Scout truth-mode run, 2026-10-01: CAGR 7.061% adjusted vs 7.048% historical units, 2007-09 to 2026-09). "
            "Trinity (VTI 2:1 split, BIL 1:2): fees $1,465 adjusted vs $1,492 historical on $100K over 2007-2026, CAGR "
            "7.049% both (2026-10-01). TFI pays no commission, so only whole-share rounding moves (not listed)."
        ),
        mitigation_str="Parity mode reproduces it; truth mode uses raw historical share units. Engine fix #15 is paused.",
        affected_strategy_list=(
            "strategy_taa_df_btal_fallback_tqqq_vix_cash", "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
            "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "strategy_taa_df_btal_1n_fallback_qld_vix_cash",
            "strategy_taa_df_1n_fallback_qld_vix_cash", "strategy_taa_df_1n_fallback_sso_vix_cash",
            "strategy_taa_inflation_compass", "strategy_taa_inflation_compass_qqq", "strategy_taa_adaptive_macro_core5",
            "strategy_taa_trinity_vol_control_8_bil",
        ),  # every gated TAA variant (fee impact measured for TAA 3x), both Compass modules, CORE5 and Trinity
    ),
)

ENGINE_DEVIATION_TUPLE = ENGINE_DEVIATION_TUPLE + (
    EngineDeviation(
        deviation_id_str="dtb3_publication_lag",
        issue_description_str=(
            "TAA's cash hurdle uses the DTB3 observation dated T at the T close, but FRED publishes that value on "
            "T+1, after the T+1 open when the orders fill."
        ),
        expected_bias_direction_str="Look-ahead of one publication day on the hurdle; direction unsigned.",
        impact_level_str="0 of 168 TAA 3x decisions change (P3 review, 2026-09-30).",
        mitigation_str="Parity mode reproduces it; readiness-audit fix #14 lags DTB3 by one session in the engine.",
        affected_strategy_list=(
            "strategy_taa_df_btal_fallback_tqqq_vix_cash", "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
            "strategy_taa_df_btal_1n_fallback_qld_vix_cash", "strategy_taa_df_1n_fallback_qld_vix_cash",
            "strategy_taa_df_1n_fallback_sso_vix_cash",
        ),  # momentum-score TAA variants only: the linearity score has no DTB3 hurdle
    ),
    EngineDeviation(
        deviation_id_str="t5yie_evening_publication",
        issue_description_str=(
            "Inflation Compass reads, at the month-end decision T, the T5YIE value dated T-1 (published values are dated "
            "strictly before T since the 2026-09-28 fix). That assumes FRED posts it by about 17:15 ET on T, an evening "
            "decision before the T+1 open. With an earlier cutoff or a late FRED update only the value dated T-2 exists."
        ),
        expected_bias_direction_str="Timing assumption, not a leak of an unpublished value; measured as not flattering.",
        impact_level_str=(
            "Reading T5YIE one session later (spec `t5yie_extra_lag_int=1`, 2026-10-01, engine costs, 2003-04..2026-09): "
            "6 of 282 decisions change; Compass Sharpe 1.073 -> 1.078 (CAGR 20.81% -> 20.93%), QQQ variant 1.123 -> 1.128."
        ),
        mitigation_str="Parity mode reproduces it; no live rule exists for the T-2 case (the strategy is not wired).",
        affected_strategy_list=("strategy_taa_inflation_compass", "strategy_taa_inflation_compass_qqq"),
    ),
    EngineDeviation(
        deviation_id_str="t5yie_current_vintage",
        issue_description_str=(
            "Inflation Compass uses the current-vintage FRED T5YIE series, not a point-in-time ALFRED vintage archive "
            "(gap G-027)."
        ),
        expected_bias_direction_str="A revised value would be a look-ahead of unknown sign.",
        impact_level_str=(
            "0 T5YIE revisions across ALFRED vintages since 2014 (2026-09-27 leakage audit); earlier vintages do not "
            "exist, so 2003-2013 cannot be measured."
        ),
        mitigation_str="Parity mode reproduces it; not fixable before 2014 with the available archives.",
        affected_strategy_list=("strategy_taa_inflation_compass", "strategy_taa_inflation_compass_qqq"),
    ),
)

ENGINE_DEVIATION_TUPLE = ENGINE_DEVIATION_TUPLE + (
    EngineDeviation(
        deviation_id_str="tfi_current_vintage_fred",
        issue_description_str=(
            "TFI reads frozen CURRENT-vintage FRED files (DGS10, DGS3MO, DAAA, DBAA). They hold later backfills: FRED "
            "stopped Moody's DAAA/DBAA after 2016-10-07 and backfilled them in March 2017, so the frozen contract "
            "decided 2016-10..2017-02 with values nobody could see then. Before 2014-04 no ALFRED vintage exists, so "
            "those decisions cannot be verified."
        ),
        expected_bias_direction_str=(
            "Look-ahead in the data, not in the rule; measured not optimistic: the point-in-time replay is equal or better."
        ),
        impact_level_str=(
            "2026-10-01, BIL ledger, $100K, 2002-08..2026-08 (engine ALFRED mode): frozen CAGR 3.029% / Sharpe 0.694; "
            "ALFRED decision-date vintage 3.029% / 0.696 (0 of 284 targets change, 5 stale decisions hold); ALFRED "
            "previous-session vintage 3.131% / 0.711 (5 targets change, 5 hold)."
        ),
        mitigation_str=(
            "Parity mode reproduces the frozen contract (the engine default). Truth mode = the engine's "
            "fred_data_mode_str='alfred_point_in_time' (not re-implemented in the Scout spec). Readiness audit: TFI NOT "
            "READY until the owner approves a point-in-time snapshot and the stale rule for PAPER/LIVE."
        ),
        affected_strategy_list=("strategy_taa_tactical_fixed_income_ief_lqd",),
    ),
)

ENGINE_DEVIATION_TUPLE = ENGINE_DEVIATION_TUPLE + (
    EngineDeviation(
        deviation_id_str="xnys_closure_hindsight",
        issue_description_str=(
            "The month-end flow counts dtme on today's exchange_calendars XNYS list, which includes unscheduled closures "
            "announced after the month's measure date. Hurricane Sandy (closed 2012-10-29/30, announced 2012-10-28/29) "
            "moved the October 2012 measure from 10-23 to 10-19 and the final-leg entry from 10-24 to 10-22, days "
            "before anyone knew. The other closures since 2002 (2004-06-11, 2007-01-02, 2018-12-05, 2025-01-09) were "
            "announced before any schedule date they move, or move none."
        ),
        expected_bias_direction_str="Calendar hindsight on one month; measured mildly optimistic.",
        impact_level_str=(
            "2026-10-01, Scout truth mode (`EomConfig.notice_time_calendar_bool`, the October 2012 schedule built on the "
            "pre-Sandy calendar), engine costs, $100K, 2003-01..2026-09: no month's bucket changes; the October 2012 "
            "final leg enters two sessions later (Oct-Nov 2012 return +1.08% -> +0.65%); CAGR 10.974% -> 10.954%, "
            "Sharpe 1.087 -> 1.086."
        ),
        mitigation_str=(
            "Parity mode reproduces it (the engine's documented 'XNYS_historical_closures_not_notice_time_replay' "
            "policy); immaterial, no engine fix proposed. A live run needs a refreshed calendar policy for future "
            "unscheduled closures (the strategy doc says so)."
        ),
        affected_strategy_list=("strategy_taa_month_end_rebalancing_flow",),
    ),
)

# Retired deviations, kept for the record.
RETIRED_DEVIATION_DICT = {
    "membership_tail_trim": (
        "The loader dropped the last 5 member sessions of ex-members. Retired 2026-09-29 by readiness-audit fix #7: "
        "exact membership is now the engine default (the trim is opt-in for old artifacts only)."
    ),
}
