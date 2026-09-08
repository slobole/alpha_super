"""Hand-calculated expanded Flex cases; synthetic data, never real accounts."""

from copy import deepcopy

import pytest

from alpha.live.client_reporting import ClientReportingError, _validate_bridge_profile
from alpha.live.ibkr_nav_profile import local_ibkr_reporting_profile_dict
from test_client_reporting import client_config_dict, nav_attributes_dict, report_dict


def expanded_nav_attributes_dict(account_str="U_TEST_A", **override_dict):
    # Independent export schema, not derived from the production profile.
    monetary_field_list = """mtm mtmAtPaxos realized changeInUnrealized costAdjustments
        transferredPnlAdjustments depositsWithdrawals carbonCredits donations
        internalCashTransfers paxosTransfers excessFundSweep assetTransfers
        debitCardActivity billPay grantActivity dividends withholdingTax
        withholding871m withholdingTaxCollected changeInDividendAccruals
        changeInLiteSurchargeAccruals changeInCGTWithholdingAccruals interest
        changeInInterestAccruals changeInIncentiveCouponAccruals brokerFees
        changeInBrokerFeeAccruals advisorFees clientFees otherFees otherIncome
        feesReceivables commissions commissionsAtPaxos referralFee
        commissionCreditsRedemption commissionReceivables forexCommissions
        transactionTax taxReceivables salesTax billableSalesTax softDollars
        netFxTrading fxTranslation linkingAdjustments other corporateActionProceeds""".split()
    attribute_dict = {field_str: "0" for field_str in monetary_field_list}
    attribute_dict.update(nav_attributes_dict(account_str, **override_dict), acctAlias="", model="")
    return attribute_dict


def expanded_config_dict(second_bool=False):
    return {**client_config_dict(second_bool=second_bool), **local_ibkr_reporting_profile_dict()}


def test_profile_is_detached_and_records_its_version_in_report_identity():
    first_dict = local_ibkr_reporting_profile_dict()
    original_dict = deepcopy(first_dict)
    first_dict["nav_bridge"]["economic_fields"].clear()
    first_dict["nav_bridge"]["zero_only_fields"].clear()
    first_dict["client_twr"]["method"] = "changed"
    assert local_ibkr_reporting_profile_dict() == original_dict
    result_dict = report_dict([expanded_nav_attributes_dict()], config_dict=expanded_config_dict())
    assert result_dict["status_str"] == "ready"
    assert result_dict["pnl_float"] == 10 and result_dict["twr_float"] == pytest.approx(.01)
    changed_dict = expanded_config_dict()
    changed_dict["nav_bridge"]["profile_id"] += "_next"
    assert report_dict([expanded_nav_attributes_dict()], config_dict=changed_dict)["report_hash_str"] != result_dict["report_hash_str"]


@pytest.mark.parametrize("field_str", sorted(set(expanded_nav_attributes_dict()) - {
    "accountId", "acctAlias", "model", "currency", "fromDate", "toDate", "startingValue", "endingValue", "twr"}))
@pytest.mark.parametrize("invalid_str", [None, "", "NaN", "Infinity"])
def test_every_monetary_field_must_be_explicit_and_finite(field_str, invalid_str):
    attribute_dict = expanded_nav_attributes_dict()
    if invalid_str is None:
        attribute_dict.pop(field_str)
    else:
        attribute_dict[field_str] = invalid_str
    result_dict = report_dict([attribute_dict], config_dict=expanded_config_dict())
    assert result_dict["pnl_float"] is None and result_dict["twr_float"] is None
    assert result_dict["closing_nav_float"] == 1010
    assert result_dict["strategy_list"][0]["twr_float"] == pytest.approx(.01)
    assert any(field_str in issue_str for issue_str in result_dict["issue_list"])


@pytest.mark.parametrize("field_str", [
    "mtmAtPaxos", "realized", "changeInUnrealized", "costAdjustments", "transferredPnlAdjustments",
    "carbonCredits", "donations", "paxosTransfers", "excessFundSweep", "assetTransfers", "grantActivity",
    "withholding871m", "withholdingTaxCollected", "changeInLiteSurchargeAccruals", "changeInCGTWithholdingAccruals",
    "changeInIncentiveCouponAccruals", "brokerFees", "changeInBrokerFeeAccruals", "advisorFees", "clientFees",
    "otherIncome", "feesReceivables", "commissionsAtPaxos", "referralFee", "commissionCreditsRedemption",
    "commissionReceivables", "forexCommissions", "transactionTax", "taxReceivables", "salesTax",
    "billableSalesTax", "softDollars", "netFxTrading", "fxTranslation", "other", "corporateActionProceeds",
    "newBrokerComponent",
])
def test_unsupported_components_cannot_silently_become_profit(field_str):
    result_dict = report_dict([expanded_nav_attributes_dict(**{field_str: "1"})], config_dict=expanded_config_dict())
    assert result_dict["pnl_float"] is None and result_dict["twr_float"] is None
    assert any(field_str in issue_str for issue_str in result_dict["issue_list"])


def test_canceling_unsupported_fields_are_not_hidden_by_a_matching_nav():
    result_dict = report_dict([expanded_nav_attributes_dict(brokerFees="-1", otherIncome="1")], config_dict=expanded_config_dict())
    assert result_dict["pnl_float"] is None


def test_accrual_recognition_and_payment_reversal_are_not_double_counted():
    row_list = [
        expanded_nav_attributes_dict(closing_str="1008.14", twr_str=".814", mtm="0",
            changeInDividendAccruals="8", changeInInterestAccruals=".14"),
        expanded_nav_attributes_dict(date_str="2026-09-02", opening_str="1008.14", closing_str="1008.14",
            twr_str="0", mtm="0", dividends="10", withholdingTax="-2", changeInDividendAccruals="-8",
            interest=".14", changeInInterestAccruals="-.14"),
    ]
    result_dict = report_dict(row_list, config_dict=expanded_config_dict(), to_str="2026-09-02")
    assert result_dict["pnl_float"] == pytest.approx(8.14)
    assert [row_dict["pnl_float"] for row_dict in result_dict["daily_book_list"]] == [8.14, 0]
    assert result_dict["twr_float"] == pytest.approx(.00814)


@pytest.mark.parametrize("field_str,flow_str,closing_str", [
    ("depositsWithdrawals", "100", "1110"), ("depositsWithdrawals", "-100", "910"),
    ("debitCardActivity", "-100", "910"), ("billPay", "-100", "910"),
])
def test_capital_is_not_profit_and_official_return_is_not_overwritten(field_str, flow_str, closing_str):
    result_dict = report_dict([expanded_nav_attributes_dict(closing_str=closing_str, twr_str=".8",
        **{field_str: flow_str})], config_dict=expanded_config_dict())
    assert result_dict["capital_movement_float"] == float(flow_str)
    assert result_dict["pnl_float"] == 10 and result_dict["twr_float"] == pytest.approx(.01)
    assert result_dict["strategy_list"][0]["twr_float"] == pytest.approx(.008)


@pytest.mark.parametrize("matched_bool", [True, False])
def test_internal_transfer_retains_existing_counterparty_gate(matched_bool):
    row_list = [expanded_nav_attributes_dict(closing_str="910", internalCashTransfers="-100"),
        expanded_nav_attributes_dict("U_TEST_B", closing_str="1110" if matched_bool else "1010",
            internalCashTransfers="100" if matched_bool else "0")]
    result_dict = report_dict(row_list, config_dict=expanded_config_dict(True))
    assert result_dict["pnl_float"] == 20
    assert result_dict["twr_float"] == pytest.approx(.01) if matched_bool else result_dict["twr_float"] is None


def test_nonzero_linking_withholds_client_return_but_is_not_profit():
    result_dict = report_dict([expanded_nav_attributes_dict(closing_str="1110", linkingAdjustments="100")], config_dict=expanded_config_dict())
    assert result_dict["pnl_float"] == 10 and result_dict["linking_adjustment_float"] == 100
    assert result_dict["twr_float"] is None


def test_expanded_profile_does_not_relax_cent_tolerance():
    result_dict = report_dict([expanded_nav_attributes_dict(closing_str="1010.02")], config_dict=expanded_config_dict())
    assert result_dict["pnl_float"] is None and result_dict["twr_float"] is None


def test_supported_other_fees_are_deducted_from_profit():
    result_dict = report_dict([expanded_nav_attributes_dict(mtm="12", otherFees="-2")], config_dict=expanded_config_dict())
    assert result_dict["pnl_float"] == 10 and result_dict["twr_float"] == pytest.approx(.01)


@pytest.mark.parametrize("field_list", [None, "assetTransfers", [""], [1], ["assetTransfers", "assetTransfers"], ["mtm"], ["startingValue"]])
def test_invalid_zero_only_profile_is_rejected(field_list):
    bridge_dict = expanded_config_dict()["nav_bridge"]
    bridge_dict["zero_only_fields"] = field_list
    with pytest.raises(ClientReportingError):
        _validate_bridge_profile(bridge_dict)


def test_explicit_legacy_registry_has_no_automatic_profile():
    config_dict = client_config_dict(bridge_bool=False)
    result_dict = report_dict([expanded_nav_attributes_dict()], config_dict=config_dict)
    assert not result_dict["client_twr_configured_bool"]
    assert result_dict["twr_float"] == pytest.approx(.01) and result_dict["pnl_float"] is None
