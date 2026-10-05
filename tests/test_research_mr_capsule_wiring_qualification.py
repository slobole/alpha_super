"""Pure qualification-tool checks; no Norgate loads or snapshots are performed."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research.mr_capsule_build_20261004.qualify_wiring import (
    _norgate_mode, compare_frames, validate_output_path,
)


def _frame_df():
    return pd.DataFrame({("AAA", "Close"): [1.0, 2.0], ("AAA", "Volume"): [100, 200]},
                        index=pd.date_range("2024-01-02", periods=2))


def test_dtype_and_column_order_are_reported_without_changing_exact_values():
    direct_df = _frame_df()
    snapshot_df = direct_df.iloc[:, ::-1].astype(float)
    result_dict = compare_frames(direct_df, snapshot_df)
    assert result_dict["passed_bool"]
    assert not result_dict["column_order_equal_bool"]
    assert result_dict["dtype_difference_list"]
    assert result_dict["max_absolute_difference_float"] == 0.0


def test_decision_input_dtype_promotion_fails_even_when_stored_values_are_exact():
    direct_df = pd.DataFrame({"Close": pd.Series([100., 105.], dtype="float32")})
    snapshot_df = direct_df.astype("float64")
    result_dict = compare_frames(direct_df, snapshot_df, require_observed_dtype_parity_bool=True)
    assert not result_dict["passed_bool"]
    assert result_dict["max_absolute_difference_float"] == 0.0
    assert result_dict["column_error_list"][0]["error_str"] == "observed decision-input dtype differs"
    assert compare_frames(direct_df * np.nan, snapshot_df * np.nan, require_observed_dtype_parity_bool=True)["passed_bool"]


def test_only_explicitly_recorded_all_nan_schema_columns_are_ignored():
    direct_df = _frame_df()
    snapshot_df = direct_df.copy()
    snapshot_df[("AAA", "Turnover")] = np.nan
    assert not compare_frames(direct_df, snapshot_df)["passed_bool"]
    result_dict = compare_frames(direct_df, snapshot_df, allow_empty_schema_bool=True)
    assert result_dict["passed_bool"]
    assert result_dict["ignored_all_nan_schema_column_list"] == [{"side_str": "snapshot", "column_str": "AAA|Turnover"}]
    snapshot_df.loc[snapshot_df.index[-1], ("AAA", "Turnover")] = 0.0
    assert not compare_frames(direct_df, snapshot_df, allow_empty_schema_bool=True)["passed_bool"]


@pytest.mark.parametrize("replacement_float", [2.0 + 1e-12, np.nan, np.inf])
def test_real_value_and_null_differences_are_never_tolerated(replacement_float):
    direct_df = _frame_df()
    snapshot_df = direct_df.copy()
    snapshot_df.loc[snapshot_df.index[-1], ("AAA", "Close")] = replacement_float
    result_dict = compare_frames(direct_df, snapshot_df)
    assert not result_dict["passed_bool"]
    assert result_dict["mismatched_cell_count_int"] == 1
    assert result_dict["column_error_list"][0]["first_date_str"].startswith("2024-01-03")


def test_entirely_nonmember_columns_are_reported_but_members_are_never_dropped():
    direct_df = pd.DataFrame({"AAA": [1, 1]}, index=pd.date_range("2024-01-02", periods=2))
    snapshot_df = direct_df.assign(RETIRED=0)
    result_dict = compare_frames(direct_df, snapshot_df, allow_nonmember_columns_bool=True)
    assert result_dict["passed_bool"]
    assert result_dict["ignored_zero_membership_column_list"] == [{"side_str": "snapshot", "column_str": "RETIRED"}]
    snapshot_df.loc[snapshot_df.index[-1], "RETIRED"] = 1
    assert not compare_frames(direct_df, snapshot_df, allow_nonmember_columns_bool=True)["passed_bool"]


def test_missing_dates_and_reordered_observations_fail():
    direct_df = _frame_df()
    assert not compare_frames(direct_df, direct_df.iloc[:-1])["passed_bool"]
    assert not compare_frames(direct_df, direct_df.iloc[::-1])["passed_bool"]
    assert not compare_frames(direct_df.iloc[:0], direct_df.iloc[:0])["passed_bool"]


def test_large_integer_conversion_cannot_conceal_a_difference():
    direct_df = pd.DataFrame({"Volume": [2**53 + 1]})
    snapshot_df = pd.DataFrame({"Volume": [2**53]})
    result_dict = compare_frames(direct_df, snapshot_df)
    assert not result_dict["passed_bool"]
    assert "exact float64" in result_dict["column_error_list"][0]["error_str"]


def test_output_must_be_an_empty_child_of_research_root(tmp_path):
    research_root_path_obj = tmp_path / "results" / "research"
    empty_path_obj = research_root_path_obj / "new_qualification"
    assert validate_output_path(empty_path_obj, research_root_path_obj) == empty_path_obj.resolve()
    for invalid_path_obj in (research_root_path_obj, tmp_path / "active", research_root_path_obj / ".." / "active"):
        with pytest.raises(ValueError, match="strictly below"):
            validate_output_path(invalid_path_obj, research_root_path_obj)
    empty_path_obj.mkdir(parents=True)
    (empty_path_obj / "prior_evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="new or empty"):
        validate_output_path(empty_path_obj, research_root_path_obj)


def test_process_local_mode_restores_environment_even_after_failure(monkeypatch):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "prior")
    monkeypatch.delenv("NORGATE_SNAPSHOT_ROOT", raising=False)
    import os

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with _norgate_mode(Path("isolated_snapshots")):
            assert os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] == "true"
            assert os.environ["NORGATE_SNAPSHOT_ROOT"] == "isolated_snapshots"
            raise RuntimeError("synthetic failure")
    assert os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] == "prior"
    assert "NORGATE_SNAPSHOT_ROOT" not in os.environ
