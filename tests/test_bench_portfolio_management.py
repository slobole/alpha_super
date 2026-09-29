"""Portfolio lifecycle and run identity, with isolated configs and no real jobs."""

import copy
import json
import re
from types import SimpleNamespace

import pytest
import yaml
from werkzeug.datastructures import MultiDict

from alpha.bench import catalog, portfolio_builder, portfolio_config, portfolio_overview, runs
from alpha.bench.app import create_app
from alpha.engine import portfolio_manager
from test_portfolio_builder import stub_pods


class JobRecorder:
    def __init__(self):
        self.job_list = []

    def list_jobs(self):
        return self.job_list

    def active_count(self):
        return len(self.job_list)

    def submit(self, label_str, target_str, kind_str, command_list):
        self.job_list.append(SimpleNamespace(kind_str=kind_str, target_str=target_str, status_str="queued"))


@pytest.fixture
def management_env(tmp_path, monkeypatch):
    config_root_path = tmp_path / "portfolios"
    config_root_path.mkdir()
    result_root_path = tmp_path / "results"
    result_root_path.mkdir()
    monkeypatch.setattr(catalog, "PORTFOLIOS_ROOT_PATH", config_root_path)
    monkeypatch.setattr(catalog, "REPO_ROOT_PATH", tmp_path)
    monkeypatch.setattr(portfolio_builder, "PORTFOLIOS_ROOT_PATH", config_root_path)
    monkeypatch.setattr(runs, "RESEARCH_PORTFOLIO_ROOT_PATH", result_root_path)
    monkeypatch.setattr(runs, "RESULTS_ROOT_PATH", tmp_path)
    # Validate with the real PM parser but use two registry-supported modules.
    config_dict = {
        "name_str": "Original", "capital_base_float": 100000,
        "backtest_start_date_str": "2024-01-02", "end_date_str": None,
        "allocation_policy_str": "fixed", "max_workers_int": 2,
        "save_pod_artifacts_bool": False, "rebalance": None,
        "pods": [
            {"pod_id_str": "ndx", "strategy_import_str": "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy", "weight_float": .4},
            {"pod_id_str": "taa", "strategy_import_str": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", "weight_float": .6},
        ],
    }
    config_path = config_root_path / "book.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    job_obj = JobRecorder()
    app_obj = create_app(job_manager_obj=job_obj)
    return SimpleNamespace(root_path=config_root_path, result_path=result_root_path, config_path=config_path,
                           config_dict=config_dict, client_obj=app_obj.test_client(), token_str=app_obj.config["bench_token_str"], job_obj=job_obj)


def _form_dict(env_obj, **override_dict):
    form_dict = {"csrf_token": env_obj.token_str, "config": "portfolios/book.yaml",
                 "revision": portfolio_config.fingerprint_str(yaml.safe_load(env_obj.config_path.read_text())),
                 "name": "Original", "capital": "100000", "benchmark": "", "start": "2024-01-02", "end": "",
                 "allocation": "fixed", "row_id": ["0", "1"], "weight_0": ".4", "weight_1": ".6",
                 "rebalance_frequency": "annually", "rebalance_policy": "fixed"}
    form_dict.update(override_dict)
    return form_dict


def _review_token_str(response_obj):
    assert response_obj.status_code == 200, response_obj.get_data(as_text=True)
    return re.search(r'name="review_token" value="([^"]+)"', response_obj.get_data(as_text=True)).group(1)


def _save_response(env_obj, review_response):
    return env_obj.client_obj.post("/api/portfolios/save-reviewed", data={"csrf_token": env_obj.token_str, "review_token": _review_token_str(review_response)})


def _delete_form_dict(env_obj):
    return _form_dict(env_obj, revision=portfolio_config.delete_source_tuple("portfolios/book.yaml")[1])


@pytest.mark.parametrize("policy_str", ["fixed", "equal", "inverse_volatility"])
def test_edit_review_save_preserves_fields_and_rebalance(management_env, policy_str):
    env_obj = management_env
    response_obj = env_obj.client_obj.post("/portfolios/manage/edit", data=_form_dict(env_obj, rebalance_policy=policy_str, rebalance_lookback="20"))
    assert yaml.safe_load(env_obj.config_path.read_text())["rebalance"] is None
    assert _save_response(env_obj, response_obj).status_code == 302
    saved_dict = yaml.safe_load(env_obj.config_path.read_text())
    assert saved_dict["rebalance"]["policy_str"] == policy_str
    assert saved_dict["max_workers_int"] == 2
    assert saved_dict["save_pod_artifacts_bool"] is False
    assert saved_dict["pods"] == env_obj.config_dict["pods"]
    assert ("lookback_day_int" in saved_dict["rebalance"]) == (policy_str == "inverse_volatility")


def test_clone_keeps_source_and_links_comparison(management_env):
    env_obj = management_env
    response_obj = env_obj.client_obj.post("/portfolios/manage/clone", data=_form_dict(env_obj, name="Variant", filename="variant.yaml"))
    assert _save_response(env_obj, response_obj).status_code == 302
    assert yaml.safe_load(env_obj.config_path.read_text())["rebalance"] is None
    assert portfolio_config.parent_path_str(env_obj.root_path / "variant.yaml") == "portfolios/book.yaml"
    response_obj = env_obj.client_obj.get("/portfolios/detail?config=portfolios/variant.yaml")
    assert response_obj.status_code == 200
    assert "Compare with source" in response_obj.get_data(as_text=True)


def test_delete_retains_results_and_last_book_leaves_create_link(management_env):
    env_obj = management_env
    report_path = env_obj.result_path / "report.html"
    report_path.write_text("historical result")
    response_obj = env_obj.client_obj.post("/portfolios/manage/delete", data=_delete_form_dict(env_obj))
    assert _save_response(env_obj, response_obj).status_code == 302
    assert not env_obj.config_path.exists()
    assert report_path.read_text() == "historical result"
    assert "New portfolio" in env_obj.client_obj.get("/portfolios").get_data(as_text=True)


@pytest.mark.parametrize("action_str", ["edit", "delete"])
def test_stale_review_cannot_mutate_newer_config(management_env, action_str):
    env_obj = management_env
    response_obj = env_obj.client_obj.post(f"/portfolios/manage/{action_str}", data=_delete_form_dict(env_obj) if action_str == "delete" else _form_dict(env_obj))
    newer_dict = dict(env_obj.config_dict, capital_base_float=200000)
    env_obj.config_path.write_text(yaml.safe_dump(newer_dict))
    assert _save_response(env_obj, response_obj).status_code == 409
    assert yaml.safe_load(env_obj.config_path.read_text())["capital_base_float"] == 200000


def test_active_job_csrf_and_unsigned_review_refused(management_env):
    env_obj = management_env
    assert env_obj.client_obj.post("/portfolios/manage/edit", data=_form_dict(env_obj, csrf_token="bad")).status_code == 403
    assert env_obj.client_obj.post("/api/portfolios/save-reviewed", data={"csrf_token": env_obj.token_str, "review_token": "bad"}).status_code == 400
    response_obj = env_obj.client_obj.post("/portfolios/manage/delete", data=_delete_form_dict(env_obj))
    env_obj.job_obj.submit("", "book", "portfolio", [])
    assert _save_response(env_obj, response_obj).status_code == 409
    assert env_obj.config_path.exists()


@pytest.mark.parametrize("filename_str", ["../other.yaml", "C:/outside.yaml", "variant/../escape.yaml", "book.yaml"])
def test_clone_refuses_paths_and_existing_file(management_env, filename_str):
    env_obj = management_env
    response_obj = env_obj.client_obj.post("/portfolios/manage/clone", data=_form_dict(env_obj, name="Variant", filename=filename_str))
    assert response_obj.status_code == 400


def test_clone_rejects_duplicate_output_name(management_env):
    env_obj = management_env
    response_obj = env_obj.client_obj.post("/portfolios/manage/clone", data=_form_dict(env_obj, filename="different.yaml"))
    assert response_obj.status_code == 400
    assert "unique name" in response_obj.get_data(as_text=True)


def test_rename_records_old_output_name(management_env):
    env_obj = management_env
    response_obj = env_obj.client_obj.post("/portfolios/manage/edit", data=_form_dict(env_obj, name="Renamed"))
    assert _save_response(env_obj, response_obj).status_code == 302
    assert portfolio_config.lineage_dict_for(env_obj.config_path)["previous_name_list"] == ["Original"]


def test_config_status_matches_snapshot_not_current_filename(management_env):
    env_obj = management_env
    run_obj = SimpleNamespace(metadata_dict={"source_config_dict": copy.deepcopy(env_obj.config_dict)})
    assert portfolio_config.run_config_status_str(env_obj.config_dict, run_obj) == "Current config"
    run_obj.metadata_dict["source_config_dict"]["pods"][0]["weight_float"] = .5
    assert portfolio_config.run_config_status_str(env_obj.config_dict, run_obj) == "Config changed"
    assert portfolio_config.run_config_status_str(env_obj.config_dict, SimpleNamespace(metadata_dict={})) == "Config unverified"


@pytest.mark.parametrize("policy_str,expected_str", [("inverse_vol", "inverse_volatility"), ("one_over_n", "equal")])
def test_editor_normalizes_supported_aliases(management_env, policy_str, expected_str):
    config_dict = management_env.config_dict
    config_dict["rebalance"] = {"frequency_str": "MONTHLY", "policy_str": policy_str}
    context_dict = portfolio_config.editor_context_dict(config_dict)
    assert context_dict["rebalance_frequency_str"] == "monthly"
    assert context_dict["rebalance_policy_str"] == expected_str


def test_simple_editor_preserves_pinned_pickle_and_extra_fields():
    config_dict = {"name": "Simple", "capital": None, "note": "keep", "pods": [{"strategy": "A", "weight": 1, "pkl": "../results/pinned/A.pkl", "note": "pod note"}]}
    form_obj = MultiDict({"name": "Simple", "capital": "", "benchmark": "", "row_id": "0", "weight_0": "1", "rebalance_frequency": "quarterly", "rebalance_policy": "fixed"})
    result_dict = portfolio_config.edited_config_dict(config_dict, form_obj)
    assert result_dict["pods"] == config_dict["pods"]
    assert result_dict["note"] == "keep"
    assert result_dict["rebalance"] == "quarterly"


def test_snapshot_is_captured_when_manager_loads_not_when_file_changes(management_env):
    env_obj = management_env
    manager_obj = portfolio_manager.PortfolioManager.from_yaml(env_obj.config_path)
    env_obj.config_path.write_text(yaml.safe_dump(dict(env_obj.config_dict, capital_base_float=200000)))
    assert manager_obj.source_config_dict["capital_base_float"] == 100000
    assert manager_obj.config.capital_base_float == 100000


@pytest.mark.parametrize("policy_str", ["fixed", "equal", "inverse_volatility"])
def test_new_builder_review_and_save_round_trip_rebalance(management_env, stub_pods, policy_str, monkeypatch):
    monkeypatch.setattr(portfolio_manager, "SUPPORTED_STRATEGY_IMPORT_TUPLE", tuple(candidate_obj.module_import_str for candidate_obj in stub_pods[1]))
    env_obj = management_env
    form_dict = {"csrf_token": env_obj.token_str, "name": "New book", "capital": "100000",
                 "pod": ["strategy_alpha", "strategy_beta"], "weight__strategy_alpha": ".5", "weight__strategy_beta": ".5",
                 "rebalance_frequency": "quarterly", "rebalance_policy": policy_str, "rebalance_lookback": "25"}
    review_obj = env_obj.client_obj.post("/portfolios/new/review", data=form_dict)
    assert review_obj.status_code == 200
    assert f'value="{policy_str}" selected' in review_obj.get_data(as_text=True)
    form_dict["filename"] = "new_book.yaml"
    assert env_obj.client_obj.post("/api/portfolios/new", data=form_dict).status_code == 302
    rebalance_dict = yaml.safe_load((env_obj.root_path / "new_book.yaml").read_text())["rebalance"]
    assert rebalance_dict["frequency_str"] == "quarterly"
    assert rebalance_dict["policy_str"] == policy_str
    assert rebalance_dict.get("lookback_day_int") == (25 if policy_str == "inverse_volatility" else None)


def test_new_builder_cannot_overwrite_or_reuse_output_name(management_env, stub_pods, monkeypatch):
    monkeypatch.setattr(portfolio_manager, "SUPPORTED_STRATEGY_IMPORT_TUPLE", tuple(candidate_obj.module_import_str for candidate_obj in stub_pods[1]))
    env_obj = management_env
    form_dict = {"csrf_token": env_obj.token_str, "name": "Original", "capital": "100000", "filename": "different.yaml",
                 "pod": ["strategy_alpha", "strategy_beta"], "weight__strategy_alpha": ".5", "weight__strategy_beta": ".5"}
    assert env_obj.client_obj.post("/api/portfolios/new", data=form_dict).status_code == 400
    form_dict.update(name="Fresh", filename="book.yaml", overwrite="1")
    assert env_obj.client_obj.post("/api/portfolios/new", data=form_dict).status_code == 409


def test_malformed_sibling_does_not_break_catalog_or_valid_edit(management_env):
    env_obj = management_env
    (env_obj.root_path / "broken.yaml").write_text("pods: [unterminated")
    page_obj = env_obj.client_obj.get("/portfolios")
    assert page_obj.status_code == 200
    assert "YAML error" in page_obj.get_data(as_text=True)
    review_obj = env_obj.client_obj.post("/portfolios/manage/edit", data=_form_dict(env_obj))
    assert _save_response(env_obj, review_obj).status_code == 302


def test_equal_allocation_omits_weights_and_has_correct_editor_values(management_env):
    env_obj = management_env
    response_obj = env_obj.client_obj.post("/portfolios/manage/edit", data=_form_dict(env_obj, allocation="equal"))
    assert _save_response(env_obj, response_obj).status_code == 302
    config_dict = yaml.safe_load(env_obj.config_path.read_text())
    assert all("weight_float" not in pod_dict for pod_dict in config_dict["pods"])
    config_dict["allocation_policy_str"] = "EQUAL"
    context_dict = portfolio_config.editor_context_dict(config_dict)
    assert context_dict["allocation_str"] == "equal"
    assert [pod_dict["weight_float"] for pod_dict in context_dict["pod_list"]] == [.5, .5]


def test_historical_alias_reserved_and_rename_history_discovered(management_env):
    env_obj = management_env
    run_path = env_obj.result_path / "Original" / "vanilla_backtest" / "2026-09-01_120000"
    run_path.mkdir(parents=True)
    (run_path / "summary.json").write_text(json.dumps({"ann_return_pct": 5}))
    (run_path / "metadata.json").write_text(json.dumps({"source_config_dict": env_obj.config_dict, "source_config_path": str(env_obj.config_path)}))
    response_obj = env_obj.client_obj.post("/portfolios/manage/edit", data=_form_dict(env_obj, name="Renamed"))
    assert _save_response(env_obj, response_obj).status_code == 302
    overview_obj = portfolio_overview.list_portfolio_overviews()[0]
    assert overview_obj.config_status_str == "Config changed"
    assert overview_obj.ann_return_float == 5
    clone_dict = _form_dict(env_obj, name="Original", filename="other.yaml")
    assert env_obj.client_obj.post("/portfolios/manage/clone", data=clone_dict).status_code == 400


def test_recorded_source_filters_another_configs_runs(management_env):
    env_obj = management_env
    run_path = env_obj.result_path / "Original" / "vanilla_backtest" / "2026-09-01_120000"
    run_path.mkdir(parents=True)
    (run_path / "summary.json").write_text(json.dumps({"ann_return_pct": 5}))
    (run_path / "metadata.json").write_text(json.dumps({"source_config_path": "C:/old/workspace/portfolios/another.yaml"}))
    assert portfolio_overview.list_portfolio_overviews()[0].latest_metric_run is None


def test_public_csrf_token_cannot_sign_review_payload(management_env):
    from itsdangerous import URLSafeTimedSerializer

    env_obj = management_env
    fake_str = URLSafeTimedSerializer(env_obj.token_str, salt="portfolio-review-v1").dumps({"source": "portfolios/book.yaml", "mode": "delete"})
    assert env_obj.client_obj.post("/api/portfolios/save-reviewed", data={"csrf_token": env_obj.token_str, "review_token": fake_str}).status_code == 400
    assert env_obj.config_path.exists()


def test_invalid_yaml_can_be_deleted_after_review(management_env):
    env_obj = management_env
    env_obj.config_path.write_text("pods: [broken")
    response_obj = env_obj.client_obj.get("/portfolios/manage/delete?config=portfolios/book.yaml")
    assert response_obj.status_code == 200
    revision_str = re.search(r'name="revision" value="([^"]+)"', response_obj.get_data(as_text=True)).group(1)
    response_obj = env_obj.client_obj.post("/portfolios/manage/delete", data={"csrf_token": env_obj.token_str, "config": "portfolios/book.yaml", "revision": revision_str})
    assert _save_response(env_obj, response_obj).status_code == 302
    assert not env_obj.config_path.exists()


def test_remove_and_add_pod_preserves_retained_identity(management_env):
    env_obj = management_env
    form_dict = _form_dict(env_obj, remove_pod="1", add_strategy="strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", add_pod_id="new_taa", add_weight=".6")
    response_obj = env_obj.client_obj.post("/portfolios/manage/edit", data=form_dict)
    assert _save_response(env_obj, response_obj).status_code == 302
    pod_list = yaml.safe_load(env_obj.config_path.read_text())["pods"]
    assert pod_list[0] == env_obj.config_dict["pods"][0]
    assert pod_list[1]["pod_id_str"] == "new_taa"


def test_simple_schema_review_save_uses_its_runner_contract(management_env, monkeypatch):
    env_obj = management_env
    from strategies import run_portfolio
    monkeypatch.setattr(run_portfolio, "find_latest_pkl", lambda strategy_name: env_obj.root_path / f"{strategy_name}.pkl")
    config_dict = {"name": "Simple", "capital": 100000, "pods": [{"strategy": "A", "weight": .4}, {"strategy": "B", "weight": .6}], "note": "preserved"}
    env_obj.config_path.write_text(yaml.safe_dump(config_dict))
    response_obj = env_obj.client_obj.post("/portfolios/manage/edit", data=_form_dict(env_obj, name="Simple"))
    assert _save_response(env_obj, response_obj).status_code == 302
    saved_dict = yaml.safe_load(env_obj.config_path.read_text())
    assert saved_dict["rebalance"] == "annually"
    assert saved_dict["pods"] == config_dict["pods"]
    assert saved_dict["note"] == "preserved"
    assert "allocation_policy_str" not in saved_dict
