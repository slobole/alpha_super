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
            "CORE5 -0.013 pp/yr (optimistic): BIL's 1-for-2 consolidation (UnadjClose / Close = 0.5 before 2017-11-30) halves its earlier ledger share counts "
            "(Scout truth-mode run, 2026-10-01: CAGR 7.061% adjusted vs 7.048% historical units, 2007-09 to 2026-09)."
        ),
        mitigation_str="Parity mode reproduces it; truth mode uses raw historical share units. Engine fix #15 is paused.",
        affected_strategy_list=(
            "strategy_taa_df_btal_fallback_tqqq_vix_cash", "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
            "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "strategy_taa_df_btal_1n_fallback_qld_vix_cash",
            "strategy_taa_df_1n_fallback_qld_vix_cash", "strategy_taa_df_1n_fallback_sso_vix_cash",
            "strategy_taa_adaptive_macro_core5",
        ),  # every gated TAA variant (2026-10-01; impact measured for TAA 3x only) and CORE5
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

# Retired deviations, kept for the record.
RETIRED_DEVIATION_DICT = {
    "membership_tail_trim": (
        "The loader dropped the last 5 member sessions of ex-members. Retired 2026-09-29 by readiness-audit fix #7: "
        "exact membership is now the engine default (the trim is opt-in for old artifacts only)."
    ),
}
