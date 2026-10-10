"""One place that says how far along each portfolio is.

Portfolio maturity uses the strategy tiers (LIVE, WIRED, CANDIDATE = PM_READY,
RESEARCH). A portfolio listed in the table has that tier. Any other portfolio
takes the tier of its least mature pod, capped at WIRED: a book whose every pod
is wired is ready to deploy, one with a research pod is research. LIVE is only
ever set by hand, for the book that actually trades.

*** CRITICAL*** A tier is a plumbing claim, not a performance or runtime-health
claim. WIRED says every pod is connected to live account routes; it does not say
those routes are enabled, healthy, or currently trading.
"""

from __future__ import annotations

from collections.abc import Iterable

from alpha.strategy_registry import MaturityTier, TIER_LABEL_DICT
from alpha.strategy_registry import tier_for as strategy_tier_for


PORTFOLIO_TIER_DICT: dict[str, MaturityTier] = {
    # The owner's live account book: NDX ~40% / TAA ~60%.
    "loren": MaturityTier.LIVE,
}


def tier_for(portfolio_name_str: str, strategy_ref_list: Iterable[str] | None = None) -> MaturityTier:
    """Tier of a portfolio YAML filename stem.

    Explicit entries win. Otherwise, given the pods' strategy references, the
    least mature pod sets the tier (never above WIRED); with no pods, RESEARCH.
    """
    explicit_tier_obj = PORTFOLIO_TIER_DICT.get(str(portfolio_name_str))
    if explicit_tier_obj is not None:
        return explicit_tier_obj
    pod_tier_list = [strategy_tier_for(strategy_ref_str) for strategy_ref_str in strategy_ref_list or ()]
    if not pod_tier_list:
        return MaturityTier.RESEARCH
    return min(min(pod_tier_list), MaturityTier.WIRED)


def tier_label_for(portfolio_name_str: str, strategy_ref_list: Iterable[str] | None = None) -> str:
    return TIER_LABEL_DICT[tier_for(portfolio_name_str, strategy_ref_list)]
