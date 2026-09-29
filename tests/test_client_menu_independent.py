"""Research mandate tests that can catch wrong horizons and permissive gates."""
import numpy as np
import pandas as pd
from scripts.research.client_menu_independent_20260923.analyze import horizon_metrics, risk_gate
from scripts.research.client_menu_independent_20260923.protocol import all_candidate_dict


def test_unrecovered_terminal_loss_remains_censored():
    return_series = pd.Series([.1, -.2, 0., .05], index=pd.bdate_range("2020-01-01", periods=4))
    result_dict = horizon_metrics(return_series)
    assert result_dict["terminal_recovery_censored"]
    assert result_dict["terminal_underwater_sessions"] == 3
    assert np.isnan(result_dict["worst_3y"])


def test_horizon_does_not_invent_an_unsupported_seven_year_result():
    return_series = pd.Series(np.full(1260, .0001), index=pd.bdate_range("2010-01-01", periods=1260))
    result_dict = horizon_metrics(return_series)
    assert np.isclose(result_dict["worst_5y"], 1.0001**1260-1)
    assert np.isnan(result_dict["worst_7y"])


def test_every_loss_gate_is_required_and_missing_history_fails():
    mandate_dict = {"max_dd": .15, "worst12m_loss": .1, "max_underwater_sessions": 756, "horizon_sessions": 756}
    metric_dict = {"max_drawdown": -.14, "worst_rolling252": -.09, "max_underwater_sessions": 755, "worst_3y": .01}
    assert risk_gate(metric_dict, mandate_dict)[0]
    for field_str, value_float in [("max_drawdown", -.16), ("worst_rolling252", -.11), ("max_underwater_sessions", 757), ("worst_3y", -.01), ("worst_3y", np.nan)]:
        assert not risk_gate({**metric_dict, field_str: value_float}, mandate_dict)[0]


def test_configurations_are_solvent_without_hidden_leverage_or_removed_budget():
    candidate_dict = all_candidate_dict()
    for configuration_dict in candidate_dict.values():
        weight_dict = configuration_dict["weights"]
        assert min(weight_dict.values()) >= 0
        assert np.isclose(sum(weight_dict.values()), 1.)
    assert candidate_dict["M0"]["weights"] == {"MOSAIC": .5, "DF_BTAL_QQQ_LINEAR": .5}
    assert candidate_dict["A0_without_HPI"]["weights"]["BIL"] == 1/3
