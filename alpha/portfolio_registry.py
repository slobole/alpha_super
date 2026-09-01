"""One place that says how far along each portfolio is.

Portfolio maturity mirrors the strategy registry: anything absent from the
table is RESEARCH, while PM_READY and WIRED are explicit promotions.

*** CRITICAL*** A tier is a plumbing claim, not a performance or runtime-health
claim. WIRED says the portfolio has been connected to live account routes; it
does not say those routes are enabled, healthy, or currently trading.
"""

from __future__ import annotations

from alpha.strategy_registry import MaturityTier, TIER_LABEL_DICT


PORTFOLIO_TIER_DICT: dict[str, MaturityTier] = {
    "loren": MaturityTier.WIRED,
}


def tier_for(portfolio_name_str: str) -> MaturityTier:
    """Return the tier registered for a portfolio YAML filename stem."""
    return PORTFOLIO_TIER_DICT.get(str(portfolio_name_str), MaturityTier.RESEARCH)


def tier_label_for(portfolio_name_str: str) -> str:
    return TIER_LABEL_DICT[tier_for(portfolio_name_str)]
