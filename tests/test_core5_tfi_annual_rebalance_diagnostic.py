from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import run_core5_tfi_annual_rebalance_diagnostic as study


def test_annual_rebalance_resets_before_first_new_year_return() -> None:
    date_idx = pd.DatetimeIndex(
        ["2007-12-31", "2008-01-02", "2008-01-03"], name="date"
    )
    core5_value_ser = pd.Series([60.0, 66.0, 72.6], index=date_idx)
    tactical_fi_value_ser = pd.Series([40.0, 40.0, 40.0], index=date_idx)

    annual_path_df, rebalance_event_df = study.build_annual_rebalanced_path_tuple(
        core5_value_ser * 7_500.0,
        tactical_fi_value_ser * 7_500.0,
    )

    assert len(rebalance_event_df) == 1
    assert rebalance_event_df.loc[0, "decision_close_date_str"] == "2007-12-31"
    assert rebalance_event_df.loc[0, "rebalance_effective_date_str"] == "2008-01-02"
    assert np.isclose(
        annual_path_df.loc["2008-01-02", "core5_notional_float"],
        412_500.0,
    )
    assert np.isclose(
        annual_path_df.loc["2008-01-02", "tactical_fi_notional_float"],
        375_000.0,
    )


def test_annual_rebalance_does_not_reset_inside_calendar_year() -> None:
    date_idx = pd.DatetimeIndex(
        ["2007-08-31", "2007-09-04", "2007-09-05"], name="date"
    )
    core5_value_ser = pd.Series([375.0, 412.5, 453.75], index=date_idx) * 1_000.0
    tactical_fi_value_ser = pd.Series([375.0, 375.0, 375.0], index=date_idx) * 1_000.0

    annual_path_df, rebalance_event_df = study.build_annual_rebalanced_path_tuple(
        core5_value_ser,
        tactical_fi_value_ser,
    )

    assert rebalance_event_df.empty
    assert np.isclose(
        annual_path_df.iloc[-1]["core5_notional_float"], 453_750.0
    )
    assert np.isclose(
        annual_path_df.iloc[-1]["tactical_fi_notional_float"], 375_000.0
    )


def test_annual_rebalance_preserves_total_value_at_reset() -> None:
    date_idx = pd.DatetimeIndex(
        ["2007-12-31", "2008-01-02"], name="date"
    )
    core5_value_ser = pd.Series([450_000.0, 450_000.0], index=date_idx)
    tactical_fi_value_ser = pd.Series([300_000.0, 300_000.0], index=date_idx)

    annual_path_df, rebalance_event_df = study.build_annual_rebalanced_path_tuple(
        core5_value_ser,
        tactical_fi_value_ser,
    )

    assert np.isclose(
        rebalance_event_df.loc[0, "transfer_to_core5_float"], -75_000.0
    )
    assert np.isclose(annual_path_df.iloc[-1]["total_value_float"], 750_000.0)


def test_source_lineage_hashes_every_consumed_input(tmp_path) -> None:
    source_artifact_dir_path = tmp_path / "source"
    for relative_path_str in (
        "source_paths/core5_375000.csv.gz",
        "source_paths/core5_750000.csv.gz",
        "source_paths/tactical_fi_375000.csv.gz",
        "source_metadata/core5_375000.json",
        "source_metadata/core5_750000.json",
        "source_metadata/tactical_fi_375000.json",
        "norgate_database_vintage_start.json",
        "norgate_database_vintage_end.json",
    ):
        input_path = source_artifact_dir_path / relative_path_str
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(relative_path_str, encoding="utf-8")

    consumed_hash_dict = study.consumed_source_sha256_by_relative_path_dict(
        source_artifact_dir_path
    )

    assert len(consumed_hash_dict) == 8
    assert set(consumed_hash_dict) == {
        "source_paths/core5_375000.csv.gz",
        "source_paths/core5_750000.csv.gz",
        "source_paths/tactical_fi_375000.csv.gz",
        "source_metadata/core5_375000.json",
        "source_metadata/core5_750000.json",
        "source_metadata/tactical_fi_375000.json",
        "norgate_database_vintage_start.json",
        "norgate_database_vintage_end.json",
    }
