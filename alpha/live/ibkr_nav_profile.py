"""Versioned expanded Flex MTM contract for the single-VPS operator workspace.

Reporting only. No source discovery, configuration writes or broker calls.
See docs/live/IBKR_NAV_FIELDS.md for classification and supported boundaries.
"""


IBKR_MTM_ECONOMIC_FIELD_TUPLE = (
    "mtm", "dividends", "withholdingTax", "changeInDividendAccruals",
    "interest", "changeInInterestAccruals", "commissions", "otherFees",
)

# These fields must be explicitly reported as zero. Their future nonzero
# meaning is NOT inferred from a sample in which they happened to be zero.
# Asset transfers need a separate counterparty/cash-in-transit contract.
IBKR_MTM_ZERO_ONLY_FIELD_TUPLE = (
    "mtmAtPaxos", "realized", "changeInUnrealized", "costAdjustments",
    "transferredPnlAdjustments", "carbonCredits", "donations", "paxosTransfers",
    "excessFundSweep", "assetTransfers", "grantActivity", "withholding871m",
    "withholdingTaxCollected", "changeInLiteSurchargeAccruals",
    "changeInCGTWithholdingAccruals", "changeInIncentiveCouponAccruals",
    "brokerFees", "changeInBrokerFeeAccruals", "advisorFees", "clientFees",
    "otherIncome", "feesReceivables", "commissionsAtPaxos", "referralFee",
    "commissionCreditsRedemption", "commissionReceivables", "forexCommissions",
    "transactionTax", "taxReceivables", "salesTax", "billableSalesTax",
    "softDollars", "netFxTrading", "fxTranslation", "other", "corporateActionProceeds",
)


def local_ibkr_reporting_profile_dict():
    """Fresh per-request defaults; explicit client registries are not changed."""
    return {
        "nav_bridge": {
            "profile_id": "ibkr_mtm_expanded_v1",
            "evidence_ref": "docs/live/IBKR_NAV_FIELDS.md#ibkr-mtm-expanded-v1",
            "reviewed_by": "Alpha Super source-contract review 2026-09-08",
            "mode": "MTM", "nonoverlap_confirmed": True,
            "economic_fields": list(IBKR_MTM_ECONOMIC_FIELD_TUPLE),
            "informational_fields": [],
            "zero_only_fields": list(IBKR_MTM_ZERO_ONLY_FIELD_TUPLE),
        },
        "client_twr": {
            "method": "daily_nav_eod_v1",
            "reviewed_by": "Alpha Super daily EOD reporting contract",
            "evidence_ref": "docs/live/CLIENT_TWR.md",
        },
    }
