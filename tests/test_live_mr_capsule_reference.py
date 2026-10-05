"""Exercise real capsule dividend opt-in through the live reference context."""
from contextlib import nullcontext
import importlib
import json

import pandas as pd
import pytest

from alpha.live import reference_compare
from alpha.live.dashboard_v3.client_comparison import saved_comparison_dict
from strategies.mr_capsule.dv2_vix_gated import DV2VixGatedStrategy
from strategies.mr_capsule.hpi_vote_vix_gated import HPIVoteVixGatedStrategy
from strategies.hpi.stateful_long import ENTRY_HORIZON_VOTE_STR, TURNOVER_FIELD_STR
from test_live_reference_compare import _build_release


def _capsule_dividend_day(pod_str):
    if pod_str == "dv2_vix_gated":
        strategy_obj = DV2VixGatedStrategy(name="capsule_reference", benchmarks=[], capital_base=100000.0)
    else:
        strategy_obj = HPIVoteVixGatedStrategy(
            name="capsule_reference", benchmarks=[], capital_base=100000.0, ranking_field_str=TURNOVER_FIELD_STR,
            entry_mode_str=ENTRY_HORIZON_VOTE_STR,
        )
    calendar_idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    strategy_obj.previous_bar, strategy_obj.current_bar = calendar_idx
    strategy_obj.add_transaction(900000001, calendar_idx[0], "BIL", 100, 100.0, 10000.0, 1, 0.0)
    price_df = pd.DataFrame({
        ("BIL", "Open"): [100.0, 99.0], ("BIL", "High"): [100.0, 99.0],
        ("BIL", "Low"): [100.0, 99.0], ("BIL", "Close"): [100.0, 99.0],
        ("BIL", "Dividend"): [1.0, 0.0],  # Norgate stamps entitlement on T; cash is posted before T+1 open.
    }, index=calendar_idx)
    price_df.columns = pd.MultiIndex.from_tuples(price_df.columns)
    price_df.attrs["norgate_adjustment_by_symbol_dict"] = {"BIL": "CAPITALSPECIAL"}
    strategy_obj._credit_dividend_cash_before_open(price_df)
    return strategy_obj


@pytest.mark.parametrize("pod_str", ["dv2_vix_gated", "hpi_vote_vix_gated"])
def test_capsule_reference_disables_dividends_and_preserves_dashboard_warning(pod_str, monkeypatch, tmp_path):
    research_strategy_obj = _capsule_dividend_day(pod_str)
    assert len(research_strategy_obj.get_dividend_ledger()) == 1  # same held BIL really accrues outside the context
    assert research_strategy_obj.cash > research_strategy_obj._capital_base
    strategy_import_str = f"strategies.mr_capsule.strategy_mr_{pod_str}_bil"
    module_obj = importlib.import_module(strategy_import_str)
    captured_strategy_list = []
    captured_accounting_list = []

    def fake_run_variant(backtest_start_date_str, capital_base_float, end_date_str):
        strategy_obj = _capsule_dividend_day(pod_str)
        captured_strategy_list.append(strategy_obj)
        captured_accounting_list.append(dict(strategy_obj._accounting_policy_dict))
        return strategy_obj

    monkeypatch.setattr(module_obj, "run_variant", fake_run_variant)
    monkeypatch.setattr(reference_compare, "use_norgate_data_profile", lambda _profile_str: nullcontext())
    release_obj = _build_release(strategy_import_str=strategy_import_str)
    reference_strategy_obj = reference_compare.run_auto_reference_strategy(
        release_obj=release_obj, deployment_start_date_str="2024-01-02", reference_end_date_str="2024-01-03",
        deployment_initial_cash_float=100000.0, output_dir_path_obj=tmp_path,
    )
    assert reference_strategy_obj is captured_strategy_list[0]
    assert captured_accounting_list[0]["dividend_data_status_str"] == "disabled_by_context"
    assert reference_strategy_obj._dividend_cash_ledger_mode_str == "disabled"
    assert reference_strategy_obj._accounting_policy_dict["dividend_cash_ledger_mode_str"] == "disabled"
    assert reference_strategy_obj._accounting_policy_dict["dividend_data_status_str"] == "disabled_by_context"
    assert reference_strategy_obj.cash == reference_strategy_obj._capital_base
    assert reference_strategy_obj.get_dividend_ledger().empty
    contract_dict = reference_compare.validate_live_reference_accounting_contract(reference_strategy_obj)
    assert contract_dict["reference_accounting_contract_version_str"] == "price_return_ledger_v1"
    assert contract_dict["reference_dividend_cash_ledger_mode_str"] == "disabled"
    summary_path_obj = tmp_path / "reference_summary.json"
    summary_path_obj.write_text(json.dumps({
        **contract_dict, "pod_id_str": release_obj.pod_id_str, "account_route_str": "DU_TEST",
        "mode_str": "live", "deployment_start_date_str": "2024-01-02", "target_session_date_str": "2024-01-03",
    }), encoding="utf-8")
    comparison_dict = saved_comparison_dict({
        "pod_id": release_obj.pod_id_str, "account_route": "DU_TEST", "effective_from": "2024-01-02",
        "reference_summary_path": str(summary_path_obj),
    }, from_date_str="2024-01-02", to_date_str="2024-01-03")
    assert any("dividend cash accounting is disabled or unproven" in issue_str for issue_str in comparison_dict["issue_list"])
