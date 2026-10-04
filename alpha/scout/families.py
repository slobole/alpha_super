"""Mechanism families: the unit in which Scout counts trials.

A family is defined by WHY an idea should earn money, not by which indicator
expresses it. DV2, QPI, RSI(2) and IBS on US large-cap stocks are one family.
Every registration names exactly one family, and every trial counts toward it
(the DSR's N comes from all trials of the family, across all studies).

Adding a family is a code change reviewed like any other. Do not add a new
family to escape an existing family's trial count.
"""

from __future__ import annotations

FAMILY_DESCRIPTION_DICT: dict[str, str] = {
    "us_equity_short_term_reversal": (
        "Buying short-term losers or oversold single US stocks, paid for supplying liquidity to impatient "
        "sellers. Examples: DV2, QPI, RSI(2), IBS, HPI on S&P 500 / NDX / Russell members."
    ),
    "etf_short_term_reversal": (
        "The same liquidity-provision idea on sector, industry or country ETFs. Examples: sector IBS "
        "down-shock, industry-ETF DV2."
    ),
    "equity_cross_sectional_momentum": (
        "Holding recent relative winners among stocks, paid for under-reaction and slow information "
        "diffusion. Examples: NDX rank pods, MOSAIC, 12-1 momentum, residual momentum."
    ),
    "tactical_asset_allocation": (
        "Rotating among a small set of asset-class ETFs by relative or absolute momentum. Examples: TAA 1/N, "
        "TAA 3x, dual momentum, Zorro Z9, Trinity."
    ),
    "macro_regime_allocation": (
        "Allocating by a macro or yield-curve regime read from economic data. Examples: Inflation Compass, "
        "Tactical Fixed Income."
    ),
    "time_series_trend_and_breakout": (
        "Following an asset's own trend or breakouts, paid for by slow-moving capital and crisis convexity. "
        "Examples: SMA filters, channel breakouts, crisis trend core, CORE5 adaptive macro (per-asset trend)."
    ),
    "calendar_and_flow": (
        "Predictable flows tied to the calendar. Examples: month-end rebalancing flow, turn of the month, "
        "beginning-of-month TLT, seasonality."
    ),
    "volatility_premium_and_hedges": (
        "Earning or paying the volatility risk premium and tail insurance. Examples: VIX term-structure "
        "rules, VIXM backwardation, put-write (Zorro Z13), BTAL and tail-hedge sleeves."
    ),
    "index_and_listing_events": (
        "Price effects around index inclusion, IPOs and listing milestones. Examples: S&P 500 post-inclusion "
        "fade, IPO / all-time-high breakout, post-split drift."
    ),
    "optimized_low_risk_allocation": (
        "Allocation by estimated risk or mean-variance optimisation. Examples: minimum variance, Markowitz "
        "(Zorro Z8), risk parity sleeves."
    ),
    "formulaic_alpha_mining": (
        "Large published sets of mined formulaic signals with no single mechanism. Example: WorldQuant "
        "Alpha101. Trials here are expected to be numerous and weakly motivated."
    ),
    "merger_arbitrage": "Capturing deal spreads between announcement and completion.",
}


def validate_family_id(family_id_str: str) -> str:
    if family_id_str not in FAMILY_DESCRIPTION_DICT:
        raise ValueError(
            f"Unknown family {family_id_str!r}. Choose one of {sorted(FAMILY_DESCRIPTION_DICT)} "
            "or add a new mechanism family in alpha/scout/families.py."
        )
    return family_id_str
