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
        expected_bias_direction_str="Overstates historical per-share fees; results are conservative (readiness audit fix #15).",
        impact_level_str=(
            "TAA 3x about +0.32 pp/yr of fees (2026-09-28 readiness audit). NDX VXN already uses historical units. "
            "Trinity (VTI 2:1 split, BIL 1:2): fees $1,465 adjusted vs $1,492 historical on $100K over 2007-2026, CAGR "
            "7.049% both (2026-10-01). TFI pays no commission, so only whole-share rounding moves (not listed)."
        ),
        mitigation_str="Parity mode reproduces it; truth mode uses raw historical share units. Engine fix #15 is paused.",
        affected_strategy_list=(
            "strategy_taa_df_btal_fallback_tqqq_vix_cash", "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
            "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "strategy_taa_df_btal_1n_fallback_qld_vix_cash",
            "strategy_taa_df_1n_fallback_qld_vix_cash", "strategy_taa_df_1n_fallback_sso_vix_cash",
            "strategy_taa_trinity_vol_control_8_bil",
        ),  # every gated TAA variant (2026-10-01); impact measured for TAA 3x and Trinity
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

# Retired deviations, kept for the record.
RETIRED_DEVIATION_DICT = {
    "membership_tail_trim": (
        "The loader dropped the last 5 member sessions of ex-members. Retired 2026-09-29 by readiness-audit fix #7: "
        "exact membership is now the engine default (the trim is opt-in for old artifacts only)."
    ),
}
