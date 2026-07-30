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

    RESEARCH  a file that runs. The default: anything absent from the table.
    PM_READY  the engine contract holds — a common run_variant, honoured
              capital, a truthfully declared benchmark — so a portfolio book
              may allocate to it.
    WIRED     connected to a live account.

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


TIER_LABEL_DICT: dict[MaturityTier, str] = {
    MaturityTier.RESEARCH: "research",
    MaturityTier.PM_READY: "pm-ready",
    MaturityTier.WIRED: "wired",
}


# Only promotions are listed; everything else is RESEARCH by default, which
# keeps this table a dozen lines instead of one per strategy file.
STRATEGY_TIER_DICT: dict[str, MaturityTier] = {
    # ── wired: live account routes ──────────────────────────────────────────
    "strategies.dv2.strategy_mr_dv2:DVO2Strategy": MaturityTier.WIRED,
    "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy": MaturityTier.WIRED,
    "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote": MaturityTier.WIRED,
    "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit": MaturityTier.WIRED,
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash": MaturityTier.WIRED,
    "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash": MaturityTier.WIRED,
    "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash": MaturityTier.WIRED,
    "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy": MaturityTier.WIRED,
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy": MaturityTier.WIRED,
    # ── pm-ready: may join a book, not connected to live ────────────────────
    # The 2x fallback pair. Promoted because their fallback ETFs date to
    # 2006-06 rather than 2010, so a book built on them carries the 2008
    # crisis that no 3x variant can reach. Both passed the readiness checks:
    # capital scales, and the stored benchmark is genuinely total return.
    "strategies.taa_df.strategy_taa_df_1n_fallback_qld_vix_cash": MaturityTier.PM_READY,
    "strategies.taa_df.strategy_taa_df_1n_fallback_sso_vix_cash": MaturityTier.PM_READY,
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc": MaturityTier.PM_READY,
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200": MaturityTier.PM_READY,
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200": MaturityTier.PM_READY,
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
