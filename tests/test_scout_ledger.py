"""Scout ledger, registration and trial counting (alpha/scout)."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha.scout.__main__ import main as scout_cli_main
from alpha.scout.ledger import GENESIS_HASH_STR, LEDGER_RELATIVE_PATH, Ledger, LedgerIntegrityError, row_hash_str
from alpha.scout.registration import Registration, register
from alpha.scout.trials import effective_trial_count, family_trial_summary, record_trial

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]


def _registration(**override_dict) -> Registration:
    field_dict = dict(
        registration_id_str="qpi_pullback_v1",
        family_id_str="us_equity_short_term_reversal",
        hypothesis_str="S&P 500 members in an uptrend with QPI < 15 beat the regime baseline over 5 days.",
        mechanism_str="Liquidity provision to impatient sellers.",
        expected_sign_and_location_str="Bottom QPI decile above the regime mean; decays over 1-10 days.",
        hypothesis_class_str="E",
        universe_str="sp500_pit",
        horizon_str="5 sessions",
        schedule_str="daily",
        execution_str="next_open",
        param_grid_dict={"threshold": (5, 10, 15), "window": (2, 3)},
        primary_metric_str="date-clustered mean excess return at 5 sessions",
        kill_criteria_str="S3 hard fail",
        source_str="own idea (Pakal notebook)",
        universe_choice_str="S&P 500 point-in-time members, fixed before looking at results",
    )
    field_dict.update(override_dict)
    return Registration(**field_dict)


# ---------------------------------------------------------------- chain
def test_append_builds_a_verifiable_chain(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    first_row = ledger.append("note", {"text_str": "first"})
    second_row = ledger.append("note", {"text_str": "second"})
    assert first_row["row_id_int"] == 1 and first_row["prev_row_hash_str"] == GENESIS_HASH_STR
    assert second_row["prev_row_hash_str"] == first_row["row_hash_str"]
    assert ledger.verify() == 2


def _rewrite_lines(ledger_path: Path, transform_fn) -> None:
    line_list = ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_path.write_text("\n".join(transform_fn(line_list)) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "tamper_name_str",
    ["edit_value", "delete_middle", "swap_rows", "insert_forged_row", "rehash_edited_row", "edit_last_row", "skip_row_id"],
)
def test_any_tamper_breaks_the_chain(tmp_path, tamper_name_str):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    for idx_int in range(4):
        ledger.append("note", {"text_str": f"row {idx_int}", "value_float": float(idx_int)})

    def edit_value(line_list):
        row_dict = json.loads(line_list[1])
        row_dict["value_float"] = 99.0
        line_list[1] = json.dumps(row_dict)
        return line_list

    def rehash_edited_row(line_list):
        row_dict = json.loads(line_list[1])
        row_dict["value_float"] = 99.0
        row_dict["row_hash_str"] = row_hash_str(row_dict)  # forger fixes this row, but the next link breaks
        line_list[1] = json.dumps(row_dict)
        return line_list

    def insert_forged_row(line_list):
        return line_list[:2] + [line_list[1]] + line_list[2:]

    def edit_last_row(line_list):
        row_dict = json.loads(line_list[-1])
        row_dict["text_str"] = "rewritten"
        line_list[-1] = json.dumps(row_dict)
        return line_list

    def skip_row_id(line_list):
        # A full re-forge that is internally consistent in hashes but skips an id.
        forged_list, prev_hash_str = [], GENESIS_HASH_STR
        for idx_int, line_str in enumerate(line_list):
            row_dict = json.loads(line_str)
            row_dict["row_id_int"] = idx_int + 1 + (1 if idx_int >= 2 else 0)
            row_dict["prev_row_hash_str"] = prev_hash_str
            row_dict["row_hash_str"] = row_hash_str(row_dict)
            prev_hash_str = row_dict["row_hash_str"]
            forged_list.append(json.dumps(row_dict))
        return forged_list

    transform_dict = {
        "edit_value": edit_value,
        "delete_middle": lambda line_list: line_list[:1] + line_list[2:],
        "swap_rows": lambda line_list: [line_list[0], line_list[2], line_list[1], line_list[3]],
        "insert_forged_row": insert_forged_row,
        "rehash_edited_row": rehash_edited_row,
        "edit_last_row": edit_last_row,
        "skip_row_id": skip_row_id,
    }
    _rewrite_lines(ledger_path, transform_dict[tamper_name_str])
    with pytest.raises(LedgerIntegrityError):
        ledger.verify()
    with pytest.raises(LedgerIntegrityError):
        ledger.append("note", {"text_str": "must not extend a broken chain"})


def test_crlf_checkout_keeps_the_chain_valid(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    ledger.append("note", {"text_str": "שלום"})
    ledger.append("note", {"text_str": "second"})
    ledger_path.write_bytes(ledger_path.read_bytes().replace(b"\n", b"\r\n"))
    assert ledger.verify() == 2


def test_append_rejects_bad_payloads(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        ledger.append("unknown_type", {})
    with pytest.raises(ValueError):
        ledger.append("note", {"row_id_int": 7})
    with pytest.raises(ValueError):
        ledger.append("note", {"metrics_dict": {"sharpe_float": float("nan")}})
    assert ledger.verify() == 0


def test_stale_lock_times_out_with_a_clear_message(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", lock_timeout_float=0.2)
    ledger.lock_path.write_text("held", encoding="utf-8")
    with pytest.raises(TimeoutError, match="delete the lock file"):
        ledger.append("note", {"text_str": "x"})


def test_partial_last_line_is_an_integrity_error(tmp_path, capsys):
    ledger_path = tmp_path / "ledger.jsonl"
    Ledger(ledger_path).append("note", {"text_str": "complete"})
    with ledger_path.open("a", encoding="utf-8") as ledger_file:
        ledger_file.write('{"row_id_int": 2, "text')  # crash mid-write
    with pytest.raises(LedgerIntegrityError, match="not valid JSON"):
        Ledger(ledger_path).verify()
    assert scout_cli_main(["verify", "--ledger", str(ledger_path)]) == 1
    assert "BROKEN" in capsys.readouterr().out


def test_concurrent_appends_keep_one_valid_chain(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")

    def worker(worker_idx_int):
        for row_idx_int in range(20):
            ledger.append("note", {"worker_int": worker_idx_int, "row_int": row_idx_int})

    thread_list = [threading.Thread(target=worker, args=(idx_int,)) for idx_int in range(4)]
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    assert ledger.verify() == 80
    assert not ledger.lock_path.exists()


def _rows_from_text(text_str: str) -> list[dict]:
    return [json.loads(line_str) for line_str in text_str.splitlines() if line_str.strip()]


def test_repo_ledger_is_intact_and_only_grows():
    """This checkout's ledger verifies, and every committed version extends the one before it.

    Tail truncation and full re-forges are invisible to the chain alone; git history anchors them.
    """
    ledger_path = REPO_ROOT_PATH / LEDGER_RELATIVE_PATH
    assert ledger_path.exists()
    current_row_list = list(Ledger(ledger_path).rows())
    assert Ledger(ledger_path).verify() == len(current_row_list)

    log_result = subprocess.run(
        ["git", "log", "--format=%H", "-n", "30", "--", LEDGER_RELATIVE_PATH.as_posix()],
        cwd=REPO_ROOT_PATH, capture_output=True, text=True,
    )
    commit_list = log_result.stdout.split() if log_result.returncode == 0 else []
    if not commit_list:
        pytest.skip("Ledger not committed yet on this branch.")
    version_list = [current_row_list]  # newest first
    for commit_str in commit_list:
        show_result = subprocess.run(
            ["git", "show", commit_str + ":" + LEDGER_RELATIVE_PATH.as_posix()],
            cwd=REPO_ROOT_PATH, capture_output=True, text=True, encoding="utf-8",
        )
        if show_result.returncode == 0:
            version_list.append(_rows_from_text(show_result.stdout))
    for newer_row_list, older_row_list in zip(version_list, version_list[1:]):
        assert newer_row_list[: len(older_row_list)] == older_row_list


def test_cli_verify_and_summary(tmp_path, capsys):
    ledger_path = tmp_path / "ledger.jsonl"
    register(Ledger(ledger_path), _registration())
    assert scout_cli_main(["verify", "--ledger", str(ledger_path)]) == 0
    assert scout_cli_main(["summary", "--ledger", str(ledger_path)]) == 0
    assert "registration" in capsys.readouterr().out


# ---------------------------------------------------------------- registration
def test_register_writes_a_frozen_hashed_row(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    registration = _registration()
    row_dict = register(ledger, registration)
    assert row_dict["registration_hash_str"] == registration.hash_str()
    assert row_dict["grid_size_int"] == 6
    assert len(registration.grid_config_list()) == 6
    with pytest.raises(ValueError, match="never edited"):
        register(ledger, registration)


@pytest.mark.parametrize(
    "override_dict, message_str",
    [
        ({"family_id_str": "made_up_family"}, "Unknown family"),
        ({"hypothesis_class_str": "Z"}, "hypothesis_class_str"),
        ({"mechanism_str": "  "}, "missing required text"),
        ({"param_grid_dict": {"a": tuple(range(15)), "b": tuple(range(15))}}, "grid_justification_str"),
        ({"retro_bool": True}, "prior_trials_int"),
        ({"source_published_date_str": "2023-13-40"}, "month"),
        ({"registration_id_str": "has space"}, "slug"),
        ({"universe_choice_str": ""}, "missing required text"),
        ({"registration_id_str": "קוד"}, "slug"),
        ({"registration_id_str": ""}, "slug"),
        ({"prior_trials_int": -1}, "prior_trials_int"),
        ({"param_grid_dict": {"a": ()}}, "no values"),
    ],
)
def test_registration_validation(override_dict, message_str):
    with pytest.raises(ValueError, match=message_str):
        _registration(**override_dict).validate()


def test_grid_size_limit_is_exactly_200():
    _registration(param_grid_dict={"a": tuple(range(20)), "b": tuple(range(10))}).validate()
    with pytest.raises(ValueError, match="grid_justification_str"):
        _registration(param_grid_dict={"a": tuple(range(201))}).validate()


def test_registration_hash_ignores_grid_insertion_order():
    first = _registration(param_grid_dict={"threshold": (5, 10), "window": (2, 3)})
    second = _registration(param_grid_dict={"window": (2, 3), "threshold": (5, 10)})
    assert first.hash_str() == second.hash_str()


def test_child_registration_needs_a_same_family_parent(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    register(ledger, _registration())
    with pytest.raises(ValueError, match="does not exist"):
        register(ledger, _registration(registration_id_str="child_a", parent_id_str="missing"))
    with pytest.raises(ValueError, match="parent's family"):
        register(
            ledger,
            _registration(registration_id_str="child_b", parent_id_str="qpi_pullback_v1", family_id_str="calendar_and_flow"),
        )
    register(ledger, _registration(registration_id_str="child_c", parent_id_str="qpi_pullback_v1"))


# ---------------------------------------------------------------- trials
def test_effective_trial_count_cases():
    rng_obj = np.random.default_rng(0)
    base_vec = rng_obj.normal(0, 0.01, 2000)
    identical_df = pd.DataFrame({f"t{i}": base_vec + rng_obj.normal(0, 1e-5, 2000) for i in range(10)})
    independent_df = pd.DataFrame(rng_obj.normal(0, 0.01, (2000, 10)))
    other_vec = rng_obj.normal(0, 0.01, 2000)
    two_cluster_df = pd.DataFrame(
        {f"c{i}": (base_vec if i < 5 else other_vec) + rng_obj.normal(0, 1e-4, 2000) for i in range(10)}
    )
    assert effective_trial_count(identical_df) == 1.0
    assert effective_trial_count(independent_df) == 10.0
    assert effective_trial_count(two_cluster_df) == 2.0
    # Lopsided grids: 80 + 5 + 5 + 5 + 5 near-identical trials are 5 independent draws, not ~1.5.
    cluster_vec_list = [rng_obj.normal(0, 0.01, 2000) for _ in range(5)]
    size_list = [80, 5, 5, 5, 5]
    lopsided_df = pd.DataFrame(
        {
            f"k{cluster_idx}_{member_idx}": cluster_vec_list[cluster_idx] + rng_obj.normal(0, 1e-3, 2000)
            for cluster_idx, size_int in enumerate(size_list)
            for member_idx in range(size_int)
        }
    )
    assert effective_trial_count(lopsided_df) == 5.0
    assert effective_trial_count(identical_df.iloc[:, :1]) == 1.0
    with pytest.raises(ValueError, match="constant"):
        effective_trial_count(pd.DataFrame({"a": base_vec, "b": np.zeros(2000)}))


def test_family_summary_uses_trial_returns_for_n_eff(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    register(ledger, _registration())
    register(ledger, _registration(registration_id_str="dv2_retro", retro_bool=True, prior_trials_int=30, source_str="code"))
    for threshold_int, sharpe_float in ((5, 0.02), (10, 0.03), (15, 0.04)):
        record_trial(ledger, "qpi_pullback_v1", {"threshold": threshold_int, "window": 3}, sharpe_float, 2000, "1998-2022", "S4")
    base_vec = np.random.default_rng(1).normal(0, 0.01, 500)
    near_identical_df = pd.DataFrame({f"t{i}": base_vec + np.random.default_rng(i).normal(0, 1e-5, 500) for i in range(3)})
    summary = family_trial_summary(ledger, "us_equity_short_term_reversal", near_identical_df)
    assert summary.effective_trial_count_float == pytest.approx(31.0)
    with pytest.raises(ValueError, match="one column per recorded trial"):
        family_trial_summary(ledger, "us_equity_short_term_reversal", near_identical_df.iloc[:, :2])


def test_family_trial_summary_counts_trials_and_prior_trials(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    register(ledger, _registration())
    register(
        ledger,
        _registration(registration_id_str="dv2_retro", retro_bool=True, prior_trials_int=30, source_str="engine code"),
    )
    with pytest.raises(ValueError, match="register before"):
        record_trial(ledger, "not_registered", {}, 0.05, 1000, "1998-2022", "S4")
    for threshold_int, sharpe_float in ((5, 0.02), (10, 0.05), (15, 0.08)):
        row_dict = record_trial(
            ledger, "qpi_pullback_v1", {"threshold": threshold_int, "window": 2}, sharpe_float, 5000, "1998-2022", "S4"
        )
    assert row_dict["family_id_str"] == "us_equity_short_term_reversal"
    assert row_dict["station_str"] == "S4" and row_dict["mode_str"] == "parity"
    assert "code_commit_str" in row_dict

    summary = family_trial_summary(ledger, "us_equity_short_term_reversal")
    assert summary.recorded_trial_count_int == 3
    assert summary.prior_trial_count_int == 30
    assert summary.effective_trial_count_float == pytest.approx(33.0)
    assert summary.trial_sharpe_variance_float == pytest.approx(np.var([0.02, 0.05, 0.08], ddof=1))
    empty_summary = family_trial_summary(ledger, "calendar_and_flow")
    assert empty_summary.effective_trial_count_float == 1.0
    assert empty_summary.trial_sharpe_variance_float is None


def test_record_trial_refuses_configs_outside_the_frozen_grid(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    register(ledger, _registration())
    for bad_config_dict in ({"threshold": 999, "window": 2}, {"threshold": 5}, {"threshold": 5, "window": 2, "x": 1}):
        with pytest.raises(ValueError, match="outside the frozen grid"):
            record_trial(ledger, "qpi_pullback_v1", bad_config_dict, 0.01, 1000, "1998-2022", "S4")
    with pytest.raises(ValueError, match="mode_str"):
        record_trial(ledger, "qpi_pullback_v1", {"threshold": 5, "window": 2}, 0.01, 1000, "1998-2022", "S4", mode_str="x")
    assert ledger.verify() == 1


def test_non_string_keys_are_refused_before_they_can_brick_the_chain(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError, match="non-string key"):
        ledger.append("note", {"metrics_dict": {"by_offset": {2: 0.1, 10: 0.2}}})
    ledger.append("note", {"metrics_dict": {"by_offset": {"2": 0.1, "10": 0.2}}})
    ledger.append("note", {"text_str": "still appendable"})
    assert ledger.verify() == 2
