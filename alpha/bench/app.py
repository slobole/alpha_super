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

import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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

from alpha.bench import __version__, catalog, portfolio_builder, portfolio_overview, runs
from alpha.bench.jobs import JobManager
from alpha.engine.theme import build_bench_theme_css


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
RUN_ANALYSIS_SCRIPT_PATH = REPO_ROOT_PATH / "scripts" / "research" / "run_strategy_analysis.py"
RUN_PORTFOLIO_SCRIPT_PATH = REPO_ROOT_PATH / "strategies" / "run_portfolio.py"
RUN_PORTFOLIO_MANAGER_SCRIPT_PATH = REPO_ROOT_PATH / "strategies" / "run_portfolio_manager.py"
REPORT_TOOLTIP_SCRIPT_SHA256_BASE64_STR = "4x6jPzYq7ERLrCfTtOFnnJrgm6t+NFxUP+8hnzmKgAY="

SUPPORTED_ANALYSIS_TUPLE = ("vanilla", "capacity", "timing", "risk", "stress")
# *** CRITICAL*** Only these analyses receive --strategy-kwarg. Verified against
# _analysis_command_tuple in scripts/research/run_strategy_analysis.py: capacity,
# timing and stress build their commands without forwarding the kwargs at all.
#
# The consequence is quantitative, not cosmetic. Selecting a custom window plus
# a multi-analysis preset produces a vanilla run over the requested window and a
# capacity run over full history, written side by side under one job and one
# timestamp, with nothing in either artifact saying they disagree. Bench states
# this in the UI rather than letting the operator discover it by comparing two
# reports that were never measured over the same period.
KWARG_AWARE_ANALYSIS_TUPLE = ("vanilla", "risk")
ANALYSIS_LABEL_DICT = {
    "vanilla": "Vanilla",
    "capacity": "Capacity",
    "timing": "Timing",
    "risk": "Risk",
    "stress": "Stress",
}
# Quick presets surfaced as one-click buttons on each strategy.
RUN_PRESET_DICT = {
    "standard": ("vanilla", "capacity", "timing"),
    "full": ("vanilla", "capacity", "timing", "risk", "stress"),
}
RECENT_RUN_WINDOW_DAY_INT = 30
RECENT_RUN_TABLE_LIMIT_INT = 8
# Compare is a side-by-side reading aid, not a ranking engine: below 2 there is
# nothing to compare, above 5 the columns stop being readable.
COMPARE_MIN_INT = 2
COMPARE_MAX_INT = 5

# Signature variants the console can be rendered in. Display-only: the cookie
# changes nothing on the server and never reaches a launched job.
#
# It restyles *Bench*, not the reports it embeds. Those are baked at render time
# by alpha.engine.report, so an already-generated report.html keeps whatever
# variant produced it — switching here can leave the console and an embedded
# report disagreeing until that report is re-rendered.
BENCH_VARIANT_COOKIE_STR = "bench_variant"
DEFAULT_BENCH_VARIANT_STR = "swiss"
# Keep DEFAULT_BENCH_VARIANT_STR in sync with the :root fallback in bench.css —
# tests/test_theme_no_hardcoded_colors.py pins the two together.
BENCH_VARIANT_LABEL_DICT = {
    "swiss": "Swiss",
    "blueprint": "Blueprint",
    "journal": "Journal",
}
# One year: a display preference the operator sets once, not a session.
BENCH_VARIANT_COOKIE_MAX_AGE_INT = 365 * 24 * 60 * 60


def create_app(job_manager_obj: JobManager | None = None) -> Flask:
    flask_app_obj = Flask(__name__)
    flask_app_obj.config["job_manager_obj"] = job_manager_obj or JobManager()
    # Per-process token gating every state-changing POST. Bench binds to
    # localhost, but localhost is still reachable by cross-site form POSTs from
    # any page open in the browser, so we require a token only same-origin pages
    # can read, plus an Origin check. See _csrf_failure_response_fn.
    flask_app_obj.config["bench_token_str"] = secrets.token_urlsafe(24)
    # job_id -> the run it produced. A finished job's answer never changes,
    # so this stops the polling Jobs view from rescanning results/ forever.
    flask_app_obj.config["produced_run_cache_dict"] = {}

    def _active_variant_str() -> str:
        """The requested signature variant, or the default if unrecognized.

        Never trust the cookie: it reaches ``build_bench_theme_css``, which
        raises on an unknown variant, so an edited cookie would otherwise 500
        every page in the console.
        """
        cookie_value_str = request.cookies.get(BENCH_VARIANT_COOKIE_STR, "")
        if cookie_value_str in BENCH_VARIANT_LABEL_DICT:
            return cookie_value_str
        return DEFAULT_BENCH_VARIANT_STR

    @flask_app_obj.context_processor
    def inject_globals_fn() -> dict[str, Any]:
        job_manager = flask_app_obj.config["job_manager_obj"]
        active_variant_str = _active_variant_str()
        return {
            "bench_version_str": __version__,
            "server_date_str": datetime.now().strftime("%Y-%m-%d"),
            "server_clock_str": datetime.now().strftime("%H:%M:%S"),
            "active_job_count_int": job_manager.active_count(),
            "analysis_label_dict": ANALYSIS_LABEL_DICT,
            "single_analysis_tuple": SUPPORTED_ANALYSIS_TUPLE,
            "csrf_token_str": flask_app_obj.config["bench_token_str"],
            # Colour and type tokens for the console, derived from the same
            # signature palette the embedded reports render with.
            "bench_theme_css_str": build_bench_theme_css(active_variant_str),
            "active_variant_str": active_variant_str,
            "variant_label_dict": BENCH_VARIANT_LABEL_DICT,
        }

    def _csrf_failure_response_fn():
        """Return a 403 tuple when a POST fails CSRF checks, else None.

        Two independent gates: a token the cross-origin attacker cannot read
        (same-origin policy hides the page that carries it), and an Origin host
        match for browsers that send the header on form POSTs.
        """
        submitted_token_str = request.form.get("csrf_token", "")
        expected_token_str = flask_app_obj.config["bench_token_str"]
        if not (submitted_token_str and secrets.compare_digest(submitted_token_str, expected_token_str)):
            return ("CSRF token missing or invalid.", 403)
        origin_str = request.headers.get("Origin")
        if origin_str is not None and urlparse(origin_str).netloc != request.host:
            return ("Cross-origin request rejected.", 403)
        return None

    # ── pages ────────────────────────────────────────────────────────────

    @flask_app_obj.route("/")
    def index_page_fn() -> str:
        strategy_entry_list = catalog.list_strategies()
        strategy_stem_set = {entry_obj.stem_str for entry_obj in strategy_entry_list}
        run_index_obj = runs.build_strategy_run_index(strategy_stem_set=strategy_stem_set)
        recent_cutoff_timestamp_float = datetime.now().timestamp() - (
            RECENT_RUN_WINDOW_DAY_INT * 24 * 60 * 60
        )
        recent_strategy_stem_set = _recent_strategy_stem_set(
            strategy_entry_list,
            run_index_obj,
            cutoff_timestamp_float=recent_cutoff_timestamp_float,
        )
        card_dict_list = [
            _build_strategy_card_dict(strategy_entry_obj, run_index_obj)
            for strategy_entry_obj in strategy_entry_list
        ]
        for card_dict in card_dict_list:
            card_dict["has_recent_run_bool"] = (
                card_dict["strategy"].stem_str in recent_strategy_stem_set
            )
        wired_count_int = sum(1 for entry_obj in strategy_entry_list if entry_obj.is_wired_bool)
        return render_template(
            "index.html",
            card_dict_list=card_dict_list,
            category_pair_list=[
                category_pair
                for category_pair in catalog.list_categories()
                if category_pair[0] != catalog.MOMENTUM_CATEGORY_STR
            ],
            momentum_subcategory_tuple_list=catalog.list_momentum_subcategories(),
            strategy_count_int=len(strategy_entry_list),
            wired_count_int=wired_count_int,
            recent_strategy_count_int=len(recent_strategy_stem_set),
            recent_run_window_day_int=RECENT_RUN_WINDOW_DAY_INT,
            recent_run_list=run_index_obj.recent_runs(limit_int=RECENT_RUN_TABLE_LIMIT_INT),
        )

    @flask_app_obj.route("/strategy/<module_import_str>")
    def strategy_page_fn(module_import_str: str) -> str:
        strategy_entry_obj = catalog.get_strategy_by_module(module_import_str)
        if strategy_entry_obj is None:
            abort(404)
        run_index_obj = runs.build_strategy_run_index()
        run_entry_list = run_index_obj.runs_for(module_import_str, strategy_entry_obj.stem_str)
        report_run_entry_list = [
            run_obj for run_obj in run_entry_list if run_obj.has_report_bool
        ]
        latest_report_run_obj = next(
            (
                run_obj
                for run_obj in report_run_entry_list
                if run_obj.analysis_dir_str == "capacity_analysis"
                and not run_obj.is_legacy_capacity_bool
            ),
            report_run_entry_list[0] if report_run_entry_list else None,
        )
        return render_template(
            "strategy.html",
            strategy=strategy_entry_obj,
            run_entry_list=run_entry_list,
            latest_report_run=latest_report_run_obj,
            preset_dict=RUN_PRESET_DICT,
            kwarg_aware_analysis_tuple=KWARG_AWARE_ANALYSIS_TUPLE,
            kwarg_blind_analysis_tuple=tuple(
                analysis_str
                for analysis_str in SUPPORTED_ANALYSIS_TUPLE
                if analysis_str not in KWARG_AWARE_ANALYSIS_TUPLE
            ),
        )

    @flask_app_obj.route("/compare")
    def compare_page_fn() -> str:
        """Side-by-side latest-vanilla metrics for 2–5 strategies.

        Read-only: every number is lifted from summary.json exactly as the
        strategy pages show it — no metric is recomputed here.
        """
        submitted_module_list = request.args.getlist("m")
        deduped_module_list: list[str] = []
        for module_import_str in submitted_module_list:
            if module_import_str not in deduped_module_list:
                deduped_module_list.append(module_import_str)
        if not (COMPARE_MIN_INT <= len(deduped_module_list) <= COMPARE_MAX_INT):
            abort(
                400,
                description=(
                    f"Pick {COMPARE_MIN_INT}–{COMPARE_MAX_INT} strategies to compare."
                ),
            )

        run_index_obj = runs.build_strategy_run_index()
        column_view_list = []
        for module_import_str in deduped_module_list:
            strategy_entry_obj = catalog.get_strategy_by_module(module_import_str)
            if strategy_entry_obj is None:
                abort(404)
            column_view_list.append(
                {
                    "strategy": strategy_entry_obj,
                    "vanilla_run": run_index_obj.latest_vanilla_for(
                        strategy_entry_obj.module_import_str, strategy_entry_obj.stem_str
                    ),
                }
            )

        # Metrics measured over different periods are not comparable. The page
        # still renders — the windows are data the operator may want to see —
        # but the disagreement is stated up front, never left to be noticed.
        window_str_set = {
            view_dict["vanilla_run"].backtest_window_str
            for view_dict in column_view_list
            if view_dict["vanilla_run"] is not None
        }
        return render_template(
            "compare.html",
            column_view_list=column_view_list,
            windows_match_bool=len(window_str_set) <= 1,
        )

    @flask_app_obj.route("/research")
    def research_page_fn() -> str:
        """Result folders no strategy page can reach — sweeps, comparisons,
        diagnostics written under names that match no strategy file."""
        orphan_view_list = runs.orphan_research_view_list(catalog.list_strategies())
        return render_template("research.html", orphan_view_list=orphan_view_list)

    @flask_app_obj.route("/portfolios")
    def portfolios_page_fn() -> str:
        overview_list = portfolio_overview.list_portfolio_overviews()
        return render_template(
            "portfolios.html",
            overview_list=overview_list,
            measured_count_int=sum(
                1 for overview_obj in overview_list if overview_obj.has_run_bool
            ),
            stale_count_int=sum(
                1 for overview_obj in overview_list if overview_obj.is_stale_bool
            ),
        )

    @flask_app_obj.route("/portfolios/new")
    def portfolio_new_page_fn() -> str:
        """Pick pods for a new book. Read-only: nothing is written here."""
        return render_template(
            "portfolio_new.html",
            candidate_list=portfolio_builder.list_pod_candidates(),
        )

    def _submitted_selection_tuple() -> tuple[list[tuple[str, float]], str, float, str]:
        """Read one builder submission from the form, with usable defaults."""
        stem_list = request.form.getlist("pod")
        selection_pair_list: list[tuple[str, float]] = []
        for stem_str in stem_list:
            raw_weight_str = request.form.get(f"weight__{stem_str}", "").strip()
            try:
                weight_float = float(raw_weight_str) if raw_weight_str else 0.0
            except ValueError:
                weight_float = 0.0
            selection_pair_list.append((stem_str, weight_float))

        name_str = (request.form.get("name") or "").strip() or "NewPortfolio"
        try:
            capital_float = float((request.form.get("capital") or "100000").strip())
        except ValueError:
            capital_float = 100_000.0
        benchmark_str = (request.form.get("benchmark") or "").strip()
        return selection_pair_list, name_str, capital_float, benchmark_str

    @flask_app_obj.route("/portfolios/new/review", methods=["POST"])
    def portfolio_review_page_fn() -> str:
        """Diagnose a candidate book and show the YAML it would write.

        POST because the selection is a form body, not a bookmarkable view, and
        reading pod returns to correlate them is too expensive to invite from a
        crawled URL. Still writes nothing.
        """
        csrf_failure_obj = _csrf_failure_response_fn()
        if csrf_failure_obj is not None:
            return csrf_failure_obj

        (
            selection_pair_list,
            name_str,
            capital_float,
            benchmark_str,
        ) = _submitted_selection_tuple()
        diagnostics_obj = portfolio_builder.analyze_selection(
            selection_pair_list=selection_pair_list,
            name_str=name_str,
            capital_float=capital_float,
            benchmark_override_str=benchmark_str or None,
        )
        return render_template(
            "portfolio_review.html",
            diagnostics=diagnostics_obj,
            name_str=name_str,
            capital_float=capital_float,
            benchmark_str=benchmark_str,
            selection_pair_list=selection_pair_list,
        )

    @flask_app_obj.route("/api/portfolios/new", methods=["POST"])
    def portfolio_create_api_fn() -> Response:
        """Write the reviewed config into ``portfolios/``.

        The only Bench route that writes to the repo. It re-derives the YAML
        from the submitted selection rather than trusting posted text, refuses
        to write outside ``portfolios/``, and will not clobber an existing file
        unless the operator explicitly said so.
        """
        csrf_failure_obj = _csrf_failure_response_fn()
        if csrf_failure_obj is not None:
            return csrf_failure_obj

        (
            selection_pair_list,
            name_str,
            capital_float,
            benchmark_str,
        ) = _submitted_selection_tuple()
        diagnostics_obj = portfolio_builder.analyze_selection(
            selection_pair_list=selection_pair_list,
            name_str=name_str,
            capital_float=capital_float,
            benchmark_override_str=benchmark_str or None,
        )
        if diagnostics_obj.has_block_bool:
            abort(400, description="This selection cannot be written; see the review page.")

        filename_str = (
            request.form.get("filename") or diagnostics_obj.suggested_filename_str
        ).strip()
        overwrite_bool = request.form.get("overwrite") == "1"
        try:
            portfolio_builder.write_portfolio_yaml(
                filename_str=filename_str,
                yaml_text_str=diagnostics_obj.yaml_text_str,
                overwrite_bool=overwrite_bool,
            )
        except FileExistsError as exception_obj:
            abort(409, description=f"{exception_obj} Tick overwrite to replace it.")
        except ValueError as exception_obj:
            abort(400, description=str(exception_obj))
        return redirect(url_for("portfolios_page_fn"))

    @flask_app_obj.route("/variant/<variant_name_str>")
    def set_variant_fn(variant_name_str: str) -> Response:
        """Switch the console's signature variant and return where you were.

        A GET is appropriate here: this writes one display cookie in the
        operator's own browser and touches no server state, so there is nothing
        for a cross-site request to accomplish beyond restyling their page.
        """
        if variant_name_str not in BENCH_VARIANT_LABEL_DICT:
            abort(404)
        # Only follow a referrer back to ourselves — an absolute foreign URL
        # here would turn this route into an open redirect.
        referrer_str = request.referrer or ""
        same_origin_bool = bool(referrer_str) and urlparse(referrer_str).netloc == request.host
        response_obj = redirect(referrer_str if same_origin_bool else url_for("index_page_fn"))
        response_obj.set_cookie(
            BENCH_VARIANT_COOKIE_STR,
            variant_name_str,
            max_age=BENCH_VARIANT_COOKIE_MAX_AGE_INT,
            samesite="Lax",
            httponly=True,
        )
        return response_obj

    @flask_app_obj.route("/jobs")
    def jobs_page_fn() -> str:
        job_manager = flask_app_obj.config["job_manager_obj"]
        job_list = job_manager.list_jobs()
        return render_template(
            "jobs.html",
            job_list=job_list,
            job_view_list=_job_view_dict_list(
                job_list, flask_app_obj.config["produced_run_cache_dict"]
            ),
        )

    @flask_app_obj.route("/api/jobs/<job_id_str>/cancel", methods=["POST"])
    def cancel_job_api_fn(job_id_str: str) -> Response:
        """Stop a queued or running job.

        POST + CSRF, unlike the display-only variant switch: this kills a real
        process tree, and a half-finished analysis leaves partial artifacts on
        disk.
        """
        csrf_failure_obj = _csrf_failure_response_fn()
        if csrf_failure_obj is not None:
            return csrf_failure_obj

        job_manager = flask_app_obj.config["job_manager_obj"]
        if job_manager.get_job(job_id_str) is None:
            abort(404)
        job_manager.cancel(job_id_str)
        return redirect(url_for("jobs_page_fn"))

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
        job_list = job_manager.list_jobs()
        return render_template(
            "_jobs_table.html",
            job_list=job_list,
            job_view_list=_job_view_dict_list(
                job_list, flask_app_obj.config["produced_run_cache_dict"]
            ),
        )

    @flask_app_obj.route("/fragments/job-indicator")
    def job_indicator_fragment_fn() -> str:
        job_manager = flask_app_obj.config["job_manager_obj"]
        return render_template("_job_indicator.html", active_job_count_int=job_manager.active_count())

    # ── run API ──────────────────────────────────────────────────────────

    @flask_app_obj.route("/api/run", methods=["POST"])
    def run_api_fn() -> Response:
        csrf_failure_obj = _csrf_failure_response_fn()
        if csrf_failure_obj is not None:
            return csrf_failure_obj

        module_import_str = request.form.get("module_import", "")
        strategy_entry_obj = catalog.get_strategy_by_module(module_import_str)
        if strategy_entry_obj is None:
            abort(400, description="Unknown strategy module.")
        if not strategy_entry_obj.has_run_variant_bool:
            abort(400, description="Strategy has no run_variant() hook.")

        # Reject the whole request if any submitted analysis is unrecognized,
        # rather than silently dropping it and running a different subset.
        analysis_list = request.form.getlist("analysis")
        if not analysis_list:
            abort(400, description="No analysis selected.")
        invalid_analysis_list = [a for a in analysis_list if a not in SUPPORTED_ANALYSIS_TUPLE]
        if invalid_analysis_list:
            abort(400, description=f"Unknown analysis: {', '.join(invalid_analysis_list)}")
        if (
            analysis_list == ["capacity"]
            and not strategy_entry_obj.has_capacity_analysis_bool
        ):
            abort(
                400,
                description="Capacity unavailable — missing capacity hook.",
            )

        # Only kwargs this strategy's run_variant actually declares. run_strategy.py
        # raises on an undeclared kwarg, so an unfiltered pass-through would just
        # produce a job that dies on launch.
        declared_param_name_set = {
            param_obj.name_str for param_obj in strategy_entry_obj.run_variant_param_tuple
        }
        strategy_kwarg_list: list[str] = []
        for param_name_str in sorted(declared_param_name_set):
            submitted_value_str = request.form.get(f"kwarg__{param_name_str}", "").strip()
            if submitted_value_str:
                strategy_kwarg_list.append(f"{param_name_str}={submitted_value_str}")

        unknown_kwarg_list = [
            field_name_str[len("kwarg__") :]
            for field_name_str in request.form
            if field_name_str.startswith("kwarg__")
            and field_name_str[len("kwarg__") :] not in declared_param_name_set
            and request.form.get(field_name_str, "").strip()
        ]
        if unknown_kwarg_list:
            abort(
                400,
                description=(
                    "run_variant() does not accept: " + ", ".join(sorted(unknown_kwarg_list))
                ),
            )

        command_list = [
            sys.executable,
            str(RUN_ANALYSIS_SCRIPT_PATH),
            strategy_entry_obj.module_import_str,
        ]
        for analysis_str in analysis_list:
            command_list += ["--analysis", analysis_str]
        if len(analysis_list) > 1:
            command_list.append("--keep-going")
        for strategy_kwarg_str in strategy_kwarg_list:
            command_list += ["--strategy-kwarg", strategy_kwarg_str]

        # Jobs are labeled by the file stem — the same identity the results
        # tree, the logs, and the catalog cards use. Overrides go in the label
        # so the Jobs table never shows two runs of one strategy as identical.
        label_str = f"{strategy_entry_obj.stem_str} · {'+'.join(analysis_list)}"
        if strategy_kwarg_list:
            label_str += f" · {' '.join(strategy_kwarg_list)}"
        job_manager = flask_app_obj.config["job_manager_obj"]
        job_manager.submit(label_str, strategy_entry_obj.stem_str, "analysis", command_list)
        return redirect(url_for("jobs_page_fn"))

    @flask_app_obj.route("/api/run-portfolio", methods=["POST"])
    def run_portfolio_api_fn() -> Response:
        csrf_failure_obj = _csrf_failure_response_fn()
        if csrf_failure_obj is not None:
            return csrf_failure_obj

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
        response_obj = send_file(artifact_path)
        # Reports are generated HTML. Serve them sandboxed so that even if a
        # report ever contained active content, it runs with an opaque origin and
        # cannot script the control panel — whether embedded in the iframe or
        # opened full-tab. Scripts remain blocked except for the exact hashed
        # metric-tooltip helper generated by alpha.engine.report. Inline styles,
        # images, and (CDN) fonts remain allowed so reports render correctly.
        response_obj.headers["Content-Security-Policy"] = (
            "sandbox allow-scripts; default-src 'none'; "
            f"script-src 'sha256-{REPORT_TOOLTIP_SCRIPT_SHA256_BASE64_STR}'; "
            "img-src 'self' data: https:; "
            "style-src 'self' 'unsafe-inline' https:; "
            "font-src 'self' https: data:"
        )
        response_obj.headers["X-Content-Type-Options"] = "nosniff"
        return response_obj

    @flask_app_obj.route("/healthz")
    def healthz_fn() -> tuple[str, int]:
        return (f"bench ok {__version__}", 200)

    return flask_app_obj


def _job_view_dict_list(
    job_list: list,
    produced_run_cache_dict: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Pair each finished job with the report it appears to have produced.

    Only for jobs that actually completed — a failed or cancelled run may still
    have written partial artifacts, and offering them behind a "Report" button
    would present an abandoned run as a finished result.

    One ProducedRunFinder is shared by every row: it scans the results tree
    lazily and at most once per render, which is what keeps this view usable
    with a few hundred jobs on a page that polls every two seconds.

    Resolved answers are memoised across renders. A job that has already
    finished cannot produce a different artifact later, so once its report is
    located the lookup never has to run again — after the first render the
    polling view stops touching the results tree at all.
    """
    if produced_run_cache_dict is None:
        produced_run_cache_dict = {}
    produced_run_finder_obj = runs.ProducedRunFinder()
    job_view_dict_list: list[dict[str, Any]] = []
    for job_obj in job_list:
        if job_obj.job_id_str in produced_run_cache_dict:
            job_view_dict_list.append(
                {"job": job_obj, "produced_run": produced_run_cache_dict[job_obj.job_id_str]}
            )
            continue

        produced_run_obj = None
        if job_obj.status_str == "passed" and job_obj.started_at_str:
            try:
                started_at_timestamp_float = datetime.fromisoformat(
                    job_obj.started_at_str
                ).timestamp()
            except ValueError:
                started_at_timestamp_float = None
            if started_at_timestamp_float is not None:
                produced_run_obj = produced_run_finder_obj.find_run_produced_after(
                    job_obj.target_str,
                    started_at_timestamp_float,
                    job_obj.kind_str,
                )
        # Only a job that has stopped has a final answer; an active one may not
        # have written its artifacts yet, so leave it to resolve next render.
        if not job_obj.is_active_bool:
            produced_run_cache_dict[job_obj.job_id_str] = produced_run_obj
        job_view_dict_list.append({"job": job_obj, "produced_run": produced_run_obj})
    return job_view_dict_list


def _sortable_metric_float(summary_dict: dict, key_str: str) -> float | None:
    """A summary metric as a plain float, or None when it is not a number.

    None is kept distinct from 0.0 deliberately: a strategy with no Sharpe has
    not scored zero, and sorting must sink it below every measured one rather
    than placing it among the mediocre.
    """
    value_obj = summary_dict.get(key_str)
    if isinstance(value_obj, bool) or not isinstance(value_obj, (int, float)):
        return None
    return float(value_obj)


def _build_strategy_card_dict(strategy_entry_obj, run_index_obj) -> dict[str, Any]:
    latest_vanilla_run_obj = run_index_obj.latest_vanilla_for(
        strategy_entry_obj.module_import_str, strategy_entry_obj.stem_str
    )
    latest_run_obj = run_index_obj.latest_run_for(
        strategy_entry_obj.module_import_str, strategy_entry_obj.stem_str
    )
    vanilla_summary_dict = (
        latest_vanilla_run_obj.summary_dict if latest_vanilla_run_obj is not None else {}
    )
    return {
        "strategy": strategy_entry_obj,
        "run_count_int": run_index_obj.run_count_for(
            strategy_entry_obj.module_import_str, strategy_entry_obj.stem_str
        ),
        "latest_vanilla_run": latest_vanilla_run_obj,
        "latest_run": latest_run_obj,
        "has_recent_run_bool": False,
        "headline_chip_list": latest_vanilla_run_obj.headline_chip_list()
        if latest_vanilla_run_obj is not None
        else [],
        # A strategy is "tested" when the results tree has a run mapped to it.
        # This is the catalog's most useful split: most files here have never
        # been run, and a card with no numbers is a different kind of object
        # from one carrying a measured track record.
        "is_tested_bool": latest_run_obj is not None,
        "sort_value_dict": {
            "cagr": _sortable_metric_float(vanilla_summary_dict, "ann_return_pct"),
            "sharpe": _sortable_metric_float(vanilla_summary_dict, "sharpe"),
            "maxdd": _sortable_metric_float(vanilla_summary_dict, "max_drawdown_pct"),
            "trades": _sortable_metric_float(vanilla_summary_dict, "trade_count"),
            "last_run": (
                latest_run_obj.effective_activity_timestamp_float
                if latest_run_obj is not None
                else None
            ),
        },
    }


def _recent_strategy_stem_set(
    strategy_entry_list: list[catalog.StrategyEntry],
    run_index_obj: runs.StrategyRunIndex,
    cutoff_timestamp_float: float,
) -> set[str]:
    """Catalog strategy stems whose mapped latest run is after the cutoff."""
    recent_strategy_stem_set: set[str] = set()
    for strategy_entry_obj in strategy_entry_list:
        latest_run_obj = run_index_obj.latest_run_for(
            strategy_entry_obj.module_import_str,
            strategy_entry_obj.stem_str,
        )
        if (
            latest_run_obj is not None
            and latest_run_obj.effective_activity_timestamp_float >= cutoff_timestamp_float
        ):
            recent_strategy_stem_set.add(strategy_entry_obj.stem_str)
    return recent_strategy_stem_set
