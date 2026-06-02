"""Flask application factory for Bench.

Routes fall into three groups:

  * pages — the catalog, a per-strategy detail page, portfolios, and jobs,
  * the run API — POST endpoints that validate the request against the catalog
    and then hand a subprocess command to the :class:`JobManager`,
  * artifact serving — streams report.html (and its siblings) straight out of
    the ``results/`` tree, with a path-traversal guard.

The factory accepts an injectable ``job_manager_obj`` so tests can supply a
fake instead of spawning real subprocesses.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from alpha.bench import __version__, catalog, runs
from alpha.bench.jobs import JobManager


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
RUN_ANALYSIS_SCRIPT_PATH = REPO_ROOT_PATH / "scripts" / "research" / "run_strategy_analysis.py"
RUN_PORTFOLIO_SCRIPT_PATH = REPO_ROOT_PATH / "strategies" / "run_portfolio.py"
RUN_PORTFOLIO_MANAGER_SCRIPT_PATH = REPO_ROOT_PATH / "strategies" / "run_portfolio_manager.py"

SUPPORTED_ANALYSIS_TUPLE = ("vanilla", "friction", "timing", "risk", "stress")
ANALYSIS_LABEL_DICT = {
    "vanilla": "Vanilla",
    "friction": "Friction",
    "timing": "Timing",
    "risk": "Risk",
    "stress": "Stress",
}
# Quick presets surfaced as one-click buttons on each strategy.
RUN_PRESET_DICT = {
    "standard": ("vanilla", "friction", "timing"),
    "full": ("vanilla", "friction", "timing", "risk", "stress"),
}


def create_app(job_manager_obj: JobManager | None = None) -> Flask:
    flask_app_obj = Flask(__name__)
    flask_app_obj.config["job_manager_obj"] = job_manager_obj or JobManager()

    @flask_app_obj.context_processor
    def inject_globals_fn() -> dict[str, Any]:
        job_manager = flask_app_obj.config["job_manager_obj"]
        return {
            "bench_version_str": __version__,
            "server_clock_str": datetime.now().strftime("%H:%M:%S"),
            "active_job_count_int": job_manager.active_count(),
            "analysis_label_dict": ANALYSIS_LABEL_DICT,
            "single_analysis_tuple": SUPPORTED_ANALYSIS_TUPLE,
        }

    # ── pages ────────────────────────────────────────────────────────────

    @flask_app_obj.route("/")
    def index_page_fn() -> str:
        strategy_entry_list = catalog.list_strategies()
        run_index_obj = runs.build_strategy_run_index()
        card_dict_list = [
            _build_strategy_card_dict(strategy_entry_obj, run_index_obj)
            for strategy_entry_obj in strategy_entry_list
        ]
        wired_count_int = sum(1 for entry_obj in strategy_entry_list if entry_obj.is_wired_bool)
        return render_template(
            "index.html",
            card_dict_list=card_dict_list,
            category_pair_list=catalog.list_categories(),
            strategy_count_int=len(strategy_entry_list),
            wired_count_int=wired_count_int,
            recent_run_list=runs.recent_runs(limit_int=8),
        )

    @flask_app_obj.route("/strategy/<module_import_str>")
    def strategy_page_fn(module_import_str: str) -> str:
        strategy_entry_obj = catalog.get_strategy_by_module(module_import_str)
        if strategy_entry_obj is None:
            abort(404)
        run_index_obj = runs.build_strategy_run_index()
        run_entry_list = run_index_obj.runs_for(module_import_str, strategy_entry_obj.stem_str)
        latest_report_run_obj = next(
            (run_obj for run_obj in run_entry_list if run_obj.has_report_bool), None
        )
        return render_template(
            "strategy.html",
            strategy=strategy_entry_obj,
            run_entry_list=run_entry_list,
            latest_report_run=latest_report_run_obj,
            preset_dict=RUN_PRESET_DICT,
        )

    @flask_app_obj.route("/portfolios")
    def portfolios_page_fn() -> str:
        portfolio_entry_list = catalog.list_portfolios()
        portfolio_view_list = []
        for portfolio_entry_obj in portfolio_entry_list:
            run_entry_list = runs.scan_portfolio_runs(portfolio_entry_obj.name_str)
            latest_report_run_obj = next(
                (run_obj for run_obj in run_entry_list if run_obj.has_report_bool), None
            )
            portfolio_view_list.append(
                {
                    "portfolio": portfolio_entry_obj,
                    "run_entry_list": run_entry_list,
                    "latest_report_run": latest_report_run_obj,
                }
            )
        return render_template("portfolios.html", portfolio_view_list=portfolio_view_list)

    @flask_app_obj.route("/jobs")
    def jobs_page_fn() -> str:
        job_manager = flask_app_obj.config["job_manager_obj"]
        return render_template("jobs.html", job_list=job_manager.list_jobs())

    @flask_app_obj.route("/jobs/<job_id_str>/log")
    def job_log_page_fn(job_id_str: str) -> str:
        job_manager = flask_app_obj.config["job_manager_obj"]
        job_obj = job_manager.get_job(job_id_str)
        if job_obj is None:
            abort(404)
        return render_template(
            "log.html",
            job=job_obj,
            log_text_str=job_manager.read_log_text(job_id_str),
        )

    # ── HTMX fragments ───────────────────────────────────────────────────

    @flask_app_obj.route("/fragments/jobs")
    def jobs_fragment_fn() -> str:
        job_manager = flask_app_obj.config["job_manager_obj"]
        return render_template("_jobs_table.html", job_list=job_manager.list_jobs())

    @flask_app_obj.route("/fragments/job-indicator")
    def job_indicator_fragment_fn() -> str:
        job_manager = flask_app_obj.config["job_manager_obj"]
        return render_template("_job_indicator.html", active_job_count_int=job_manager.active_count())

    # ── run API ──────────────────────────────────────────────────────────

    @flask_app_obj.route("/api/run", methods=["POST"])
    def run_api_fn() -> Response:
        module_import_str = request.form.get("module_import", "")
        strategy_entry_obj = catalog.get_strategy_by_module(module_import_str)
        if strategy_entry_obj is None:
            abort(400, description="Unknown strategy module.")
        if not strategy_entry_obj.has_run_variant_bool:
            abort(400, description="Strategy has no run_variant() hook.")

        analysis_list = [
            analysis_str
            for analysis_str in request.form.getlist("analysis")
            if analysis_str in SUPPORTED_ANALYSIS_TUPLE
        ]
        if not analysis_list:
            abort(400, description="No valid analysis selected.")

        command_list = [
            sys.executable,
            str(RUN_ANALYSIS_SCRIPT_PATH),
            strategy_entry_obj.module_import_str,
        ]
        for analysis_str in analysis_list:
            command_list += ["--analysis", analysis_str]
        if len(analysis_list) > 1:
            command_list.append("--keep-going")

        label_str = f"{strategy_entry_obj.display_name_str} · {'+'.join(analysis_list)}"
        job_manager = flask_app_obj.config["job_manager_obj"]
        job_manager.submit(label_str, strategy_entry_obj.stem_str, "analysis", command_list)
        return redirect(url_for("jobs_page_fn"))

    @flask_app_obj.route("/api/run-portfolio", methods=["POST"])
    def run_portfolio_api_fn() -> Response:
        config_rel_path_str = request.form.get("config_rel_path", "")
        portfolio_entry_obj = catalog.get_portfolio_by_rel_path(config_rel_path_str)
        if portfolio_entry_obj is None:
            abort(400, description="Unknown portfolio config.")

        # The two YAML schemas are built by two different scripts — route by schema.
        script_path = (
            RUN_PORTFOLIO_MANAGER_SCRIPT_PATH
            if portfolio_entry_obj.schema_str == catalog.SCHEMA_MANAGER_STR
            else RUN_PORTFOLIO_SCRIPT_PATH
        )
        command_list = [
            sys.executable,
            str(script_path),
            str(REPO_ROOT_PATH / portfolio_entry_obj.rel_path_str),
        ]
        label_str = f"Portfolio · {portfolio_entry_obj.config_name_str}"
        job_manager = flask_app_obj.config["job_manager_obj"]
        job_manager.submit(label_str, portfolio_entry_obj.name_str, "portfolio", command_list)
        return redirect(url_for("jobs_page_fn"))

    # ── artifacts ────────────────────────────────────────────────────────

    @flask_app_obj.route("/artifact/<path:rel_path_str>")
    def artifact_fn(rel_path_str: str):
        artifact_path = runs.resolve_artifact_path(rel_path_str)
        if artifact_path is None:
            abort(404)
        return send_file(artifact_path)

    @flask_app_obj.route("/healthz")
    def healthz_fn() -> tuple[str, int]:
        return (f"bench ok {__version__}", 200)

    return flask_app_obj


def _build_strategy_card_dict(strategy_entry_obj, run_index_obj) -> dict[str, Any]:
    latest_vanilla_run_obj = run_index_obj.latest_vanilla_for(
        strategy_entry_obj.module_import_str, strategy_entry_obj.stem_str
    )
    return {
        "strategy": strategy_entry_obj,
        "run_count_int": run_index_obj.run_count_for(
            strategy_entry_obj.module_import_str, strategy_entry_obj.stem_str
        ),
        "latest_vanilla_run": latest_vanilla_run_obj,
        "headline_chip_list": latest_vanilla_run_obj.headline_chip_list()
        if latest_vanilla_run_obj is not None
        else [],
    }
