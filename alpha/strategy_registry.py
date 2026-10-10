"""One place that says how far along each strategy is.

The repo had two allowlists with the same name — ``SUPPORTED_STRATEGY_IMPORT_TUPLE``
in ``alpha.engine.portfolio_manager`` (may join a portfolio book) and in
``alpha.live.release_manifest`` (may trade real money). Nothing tied them
together, and they drifted in both directions: three sector-dispersion variants
sat in the portfolio list only, while two HPI strategies were wired for live but
absent from the portfolio list — trusted with money yet refused by the engine
that combines books.

This module makes that expressible only one way. A strategy has one maturity
tier, each consumer asks for a floor, and "wired but not portfolio-ready" cannot
be written down.

Four tiers (owner definitions, 2026-10-10), shown in Bench as:

    LIVE       actually trading in a live account today. Set by hand when a pod
               goes live or stops.
    WIRED      connected to the live execution path, passed its checks, and
               ready to deploy; deployment itself is separate.
    CANDIDATE  (code name PM_READY) a serious strategy whose engine contract
               holds — a common run_variant, honoured capital, a truthfully
               declared benchmark — so a portfolio book may allocate to it, but
               it is not wired to live execution yet.
    RESEARCH   everything else: a file that runs. The default for anything
               absent from the table.

Consumers ask for a floor (``>=``), so LIVE counts as WIRED and as PM_READY for
every live allowlist and portfolio check.

*** CRITICAL*** A tier is a claim about plumbing, not about edge. PM_READY says
the harness will not silently misreport the strategy; it says nothing about
whether the strategy makes money. Promotion is earned by the checks in
``tests/test_strategy_registry.py`` (cheap, always on) and
``scripts/research/check_pm_readiness.py`` (expensive, run when promoting) —
never by an opinion.
"""

from __future__ import annotations

from enum import IntEnum


class MaturityTier(IntEnum):
    """How far a strategy has been taken. Ordered, so ``>=`` reads naturally."""

    RESEARCH = 1
    PM_READY = 2
    WIRED = 3
    LIVE = 4


TIER_LABEL_DICT: dict[MaturityTier, str] = {
    MaturityTier.RESEARCH: "research",
    MaturityTier.PM_READY: "candidate",
    MaturityTier.WIRED: "wired",
    MaturityTier.LIVE: "live",
}


# Only promotions are listed; everything else is RESEARCH by default, which
# keeps this table a dozen lines instead of one per strategy file.
STRATEGY_TIER_DICT: dict[str, MaturityTier] = {
    # ── wired: live account routes ──────────────────────────────────────────
    "strategies.dv2.strategy_mr_dv2:DVO2Strategy": MaturityTier.WIRED,
    # QPI (strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy)
    # was demoted from WIRED to RESEARCH on 2026-09-28 by owner decision; it
    # was not running live on any VPS. HPI covers the same role. The readiness
    # audit (docs/research/STRATEGY_READINESS_AUDIT_20260928.md, sections 3.4
    # and 10) found a held name with no bar or a NaN IBS is never exited
    # (A-QPI-05), an undocumented selection among 14 variants, and -1.8 pp/yr
    # small-account friction. Its live host route was removed with it.
    "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote": MaturityTier.WIRED,
    # ── live: trading in the owner's live account (NDX ~40% / TAA ~60%) ─────
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash": MaturityTier.LIVE,
    "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash": MaturityTier.WIRED,
    "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash": MaturityTier.WIRED,
    "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy": MaturityTier.WIRED,
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy": MaturityTier.LIVE,
    # ── pm-ready: may join a book, not connected to live ────────────────────
    # Owner demotion 2026-09-30: retain portfolio eligibility without a live route.
    "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit": MaturityTier.PM_READY,
    # Five analyzers and capital/benchmark/determinism checks passed 2026-09-13.
    # MOC and fixed TLT borrow remain research execution assumptions.
    "strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow": MaturityTier.PM_READY,
    # Crisis Trend Core and VIXM backwardation were demoted to RESEARCH on
    # 2026-09-28 by owner decision after the readiness audit
    # (docs/research/STRATEGY_READINESS_AUDIT_20260928.md, section 10): CTC no
    # longer ran at HEAD, and VIXM was not tradable as modelled.
    # The 2x fallback pair. Promoted because their fallback ETFs date to
    # 2006-06 rather than 2010, so a book built on them carries the 2008
    # crisis that no 3x variant can reach. Both passed the readiness checks:
    # capital scales, and the stored benchmark is genuinely total return.
    "strategies.taa_df.strategy_taa_df_1n_fallback_qld_vix_cash": MaturityTier.PM_READY,
    "strategies.taa_df.strategy_taa_df_1n_fallback_sso_vix_cash": MaturityTier.PM_READY,
    # 2x with BTAL: the clean isolation showed BTAL adds return AND cuts the
    # drawdown, so the best 2x book carries it — at the price of BTAL's 2011
    # inception, which keeps this variant off the 2008 record.
    "strategies.taa_df.strategy_taa_df_btal_1n_fallback_qld_vix_cash": MaturityTier.PM_READY,
    # Promoted so the client-ladder books can run fresh through the manager:
    # vox_iyr is the long-history sector-MR sleeve in rungs 1-2, and the
    # no-BTAL linearity variant carries the 2008 proxy books. Both passed the
    # readiness gate (capital scales, benchmark genuinely total return).
    "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr": MaturityTier.PM_READY,
    "strategies.taa_df.strategy_taa_df_linearity_1n_fallback_qqq_vix_cash": MaturityTier.PM_READY,
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc": MaturityTier.PM_READY,
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200": MaturityTier.PM_READY,
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200": MaturityTier.PM_READY,
    # Three-asset inverse-volatility core with a daily 8% volatility control,
    # a five-point exposure band, and BIL as the invested reserve. Portfolio
    # Manager-ready only; deliberately absent from every live release surface.
    "strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil": MaturityTier.PM_READY,
    # Five fixed macro sleeves independently gated by a drawdown-adaptive
    # moving average, with inactive sleeves in BIL and a capped DBC short.
    # Daily Close_T fixed shares, next-open MOO, dedicated USD margin account.
    # Activation requires account-bound qualification plus fresh funding/borrow
    # checks; the example remains disabled. Research borrow remains fixed at 1%.
    "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5": MaturityTier.WIRED,
    # Monthly four-regime Inflation Compass using SPY SMA200, FRED T5YIE,
    # sector-ratio confirmation, and causal next-open ETF rebalancing. PM-only:
    # current-vintage FRED data is not PAPER/LIVE release evidence. Re-checked
    # 2026-09-28 after the T5YIE publication-lag fix: capital, total-return
    # benchmark and deterministic reruns passed; plumbing, not allocation approval.
    "strategies.taa_df.strategy_taa_inflation_compass": MaturityTier.PM_READY,
    # Same Compass rule with QQQ instead of XLK in the growth-up / inflation-off
    # cell (research candidate C2). Capital, total-return benchmark and
    # deterministic reruns passed 2026-09-28; plumbing, not allocation approval.
    # The research verdict is forward-shadow only; nothing here enforces that.
    "strategies.taa_df.strategy_taa_inflation_compass_qqq": MaturityTier.PM_READY,
    # Frozen Pakal L14 tactical-yield rule: publication-safe FRED term/credit
    # spreads, IEF/LQD sleeves, a BIL cash sleeve (25% withholding, from
    # 2026-09-28; legacy DGS3MO accrual selectable), and next-open fills. The
    # research verdict remains diagnostic; PM_READY certifies plumbing only.
    "strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd": MaturityTier.PM_READY,
    # Passive BIL (T-bills) as a cash pod, so a fund book can hold a fixed cash
    # sleeve in the Portfolio Manager (owner request 2026-10-01, fund products
    # study). Buy-and-hold BIL, 25% dividend withholding, 0% on residual cash.
    # PM-only: never a live route; cash in a live account is simply unallocated.
    "strategies.portfolio_controls.strategy_passive_bil": MaturityTier.PM_READY,
    # The two books of the NDX design of record "E2 + 40% GICS-sector cap"
    # (Scout amendment A15, 2026-10-04): the live dollar-ATR rule and the NATR20
    # rule, each with at most 4 of 10 names per current-label GICS sector, held
    # 50/50 in portfolios/ndx_e2_sector_cap_5050.yaml (owner request). PM-only;
    # the live NDX pod is unchanged and these have no live route.
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap:SectorCapVxnScaledAtrNormalizedNdxStrategy": MaturityTier.PM_READY,
    "strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap:SectorCapNatr20VxnScaledNdxStrategy": MaturityTier.PM_READY,
    # MR capsule pods (docs/research/MR_CAPSULE_20261003.md, build record 2026-10-04): DV2 and the HPI 2/3/5
    # vote behind one shared VIX gate. BIL is primary; SPMO is the alternative.
    # Separate cash identities provide the gate-only stage. WIRED routes require
    # exact-session account/data state and a dedicated full-account budget.
    # Templates remain disabled. See docs/plans/MR_CAPSULE_WIRING_REVIEW_20261005.md.
    "strategies.mr_capsule.strategy_mr_dv2_vix_gated_spmo": MaturityTier.WIRED,
    "strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil": MaturityTier.WIRED,
    "strategies.mr_capsule.strategy_mr_dv2_vix_gated_cash": MaturityTier.WIRED,
    "strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_spmo": MaturityTier.WIRED,
    "strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_bil": MaturityTier.WIRED,
    "strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_cash": MaturityTier.WIRED,
    # MOSAIC was demoted to RESEARCH on 2026-09-28 by owner decision: its
    # 2026-07-31 validation was invalidated by the split-price fix and it left
    # the recommended books (readiness audit, section 10). Committed YAMLs that
    # still hold it (ladder_4_growth*, attemp6) are now refused by the manager.
}


def module_import_str(strategy_import_str: str) -> str:
    """``package.module:Class`` -> ``package.module``."""
    return str(strategy_import_str).split(":", maxsplit=1)[0]


def tier_for(strategy_import_str: str) -> MaturityTier:
    """Tier of one ``module`` or ``module:Class`` reference.

    Falls back to the module path so a caller that knows only the module — the
    Bench catalog, which discovers files — resolves an entry registered with an
    explicit class.
    """
    reference_str = str(strategy_import_str)
    if reference_str in STRATEGY_TIER_DICT:
        return STRATEGY_TIER_DICT[reference_str]
    module_str = module_import_str(reference_str)
    for registered_str, tier_obj in STRATEGY_TIER_DICT.items():
        if module_import_str(registered_str) == module_str:
            return tier_obj
    return MaturityTier.RESEARCH


def tier_label_for(strategy_import_str: str) -> str:
    return TIER_LABEL_DICT[tier_for(strategy_import_str)]


def strategy_import_tuple_at_least(minimum_tier: MaturityTier) -> tuple[str, ...]:
    """Every registered strategy at or above ``minimum_tier``, in table order.

    Table order is preserved rather than sorted so the emitted allowlists read
    the way the registry does — wired first, then the pm-ready additions.
    """
    return tuple(
        strategy_import_str
        for strategy_import_str, tier_obj in STRATEGY_TIER_DICT.items()
        if tier_obj >= minimum_tier
    )


def pm_ready_import_tuple() -> tuple[str, ...]:
    """Strategies a portfolio book may allocate to. Wired implies pm-ready."""
    return strategy_import_tuple_at_least(MaturityTier.PM_READY)


def wired_import_tuple() -> tuple[str, ...]:
    """Strategies connected to a live account route."""
    return strategy_import_tuple_at_least(MaturityTier.WIRED)
