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

import re
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from flask import (
    Flask,
    Response,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)

from alpha.bench import (
    __version__,
    artifact_view,
    catalog,
    portfolio_builder,
    portfolio_compare,
    portfolio_overview,
    runs,
)
from alpha.bench.jobs import JobManager
from alpha.engine.stress_test import supported_stress_test_strategy_key_list
from alpha.engine.theme import (
    VENDORED_FONT_DIR_PATH,
    VENDORED_FONT_FACE_TUPLE,
    build_bench_theme_css,
    build_font_face_css,
)


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
RUN_ANALYSIS_SCRIPT_PATH = REPO_ROOT_PATH / "scripts" / "research" / "run_strategy_analysis.py"
RUN_PORTFOLIO_SCRIPT_PATH = REPO_ROOT_PATH / "strategies" / "run_portfolio.py"
RUN_PORTFOLIO_MANAGER_SCRIPT_PATH = REPO_ROOT_PATH / "strategies" / "run_portfolio_manager.py"
REPORT_TOOLTIP_SCRIPT_SHA256_BASE64_STR = "4x6jPzYq7ERLrCfTtOFnnJrgm6t+NFxUP+8hnzmKgAY="

SUPPORTED_ANALYSIS_TUPLE = ("vanilla", "capacity", "timing", "risk", "stress")
ANALYSIS_DIR_BY_ANALYSIS_DICT = {
    "vanilla": "vanilla_backtest",
    "capacity": "capacity_analysis",
    "timing": "execution_timing_analyzer",
    "risk": "risk_analysis",
    "stress": "stress_test",
}
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
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")

# Signature variants the console can be rendered in. Display-only: the cookie
# changes nothing on the server and never reaches a launched job.
#
# It restyles *Bench*, not the reports it embeds. Those are baked at render time
# by alpha.engine.report, so an already-generated report.html keeps whatever
# variant produced it — switching here can leave the console and an embedded
# report disagreeing until that report is re-rendered.
# The console renders in one house style. There used to be a switcher offering
# four, which is a design-search artifact rather than an operator need: a
# single-operator research console gains nothing from a menu of identities, and
# the switch could leave the console and the report embedded inside it
# disagreeing about which palette was active.
#
# Keep BENCH_VARIANT_STR in sync with the :root fallback in bench.css *and*
# with the default argument of build_bench_theme_css —
# tests/test_theme_no_hardcoded_colors.py pins all three together.
BENCH_VARIANT_STR = "desk"
# One year: a display preference the operator sets once, not a session.
BENCH_VARIANT_COOKIE_MAX_AGE_INT = 365 * 24 * 60 * 60

# Display density. Purely a type-and-spacing scale applied in CSS: the same
# markup, the same rows, the same numbers. "Present" exists because the console
# gets shown on a projector, where the working density is unreadable from the
# back of a room.
#
# *** UI*** This must never gate content. If a future change makes a column or
# a panel appear in one density and not the other, the two modes have become
# two different pages and the operator can no longer trust that what they
# rehearsed is what the room sees.
BENCH_DENSITY_COOKIE_STR = "bench_density"
DEFAULT_BENCH_DENSITY_STR = "work"
BENCH_DENSITY_LABEL_DICT = {
    "work": "Working",
    "present": "Presentation",
}


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

    def _active_density_str() -> str:
        """The requested display density, or the default if unrecognized.

        Same rule as the variant cookie: the value is echoed into an HTML
        attribute, so an edited cookie must not reach the template.
        """
        cookie_value_str = request.cookies.get(BENCH_DENSITY_COOKIE_STR, "")
        if cookie_value_str in BENCH_DENSITY_LABEL_DICT:
            return cookie_value_str
        return DEFAULT_BENCH_DENSITY_STR

    # House number format, in one place.
    #
    # The same metric used to render at one decimal on the catalog and two on
    # the strategy page, so a max drawdown read as -30.9% in one view and
    # -30.94% in another. In a console whose entire claim is precision, that
    # reads as carelessness — and worse, it makes two views of one saved value
    # look like two different measurements. The rule: percentages and ratios
    # both carry two decimals, counts carry thousands separators, and a missing
    # value is an em dash rather than a zero.
    flask_app_obj.jinja_env.filters["pct"] = lambda value_obj: _decimal_str(
        value_obj, suffix_str="%"
    )
    # Signed, for figures read as a comparison down a column rather than as a
    # standalone level — catalog CAGR is the case: the sign is the first thing
    # the eye needs, before the magnitude.
    flask_app_obj.jinja_env.filters["signed_pct"] = lambda value_obj: _decimal_str(
        value_obj, suffix_str="%", signed_bool=True
    )
    flask_app_obj.jinja_env.filters["ratio"] = _decimal_str
    flask_app_obj.jinja_env.filters["count"] = _integer_str

    @flask_app_obj.context_processor
    def inject_globals_fn() -> dict[str, Any]:
        job_manager = flask_app_obj.config["job_manager_obj"]
        market_now_datetime_obj = datetime.now(MARKET_TIMEZONE_OBJ)
        return {
            "bench_version_str": __version__,
            "server_date_str": market_now_datetime_obj.strftime("%Y-%m-%d"),
            "server_clock_str": market_now_datetime_obj.strftime("%H:%M:%S"),
            "server_timezone_str": market_now_datetime_obj.tzname() or "ET",
            "active_job_count_int": job_manager.active_count(),
            "analysis_label_dict": ANALYSIS_LABEL_DICT,
            "single_analysis_tuple": SUPPORTED_ANALYSIS_TUPLE,
            "csrf_token_str": flask_app_obj.config["bench_token_str"],
            # Colour and type tokens for the console, derived from the same
            # signature palette the embedded reports render with.
            "bench_theme_css_str": (
                build_font_face_css(
                    lambda file_name_str: url_for("font_fn", file_name_str=file_name_str)
                )
                + "\n"
                + build_bench_theme_css(BENCH_VARIANT_STR)
            ),
            "active_density_str": _active_density_str(),
            "density_label_dict": BENCH_DENSITY_LABEL_DICT,
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
        # Promoted = past the research gate, by either route. WIRED and
        # PM_READY are separate maturities but the same claim at catalog level:
        # someone decided this module is more than an experiment.
        promoted_count_int = sum(
            1
            for entry_obj in strategy_entry_list
            if entry_obj.is_wired_bool or entry_obj.is_pm_ready_bool
        )
        return render_template(
            "index.html",
            card_dict_list=card_dict_list,
            promoted_count_int=promoted_count_int,
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
        requested_analysis_str = request.args.get("analysis")
        selected_analysis_str = requested_analysis_str or "vanilla"
        if selected_analysis_str not in SUPPORTED_ANALYSIS_TUPLE:
            abort(400, description=f"Unknown analysis: {selected_analysis_str}")
        analyzer_view_list = _analyzer_view_dict_list(
            strategy_entry_obj,
            run_entry_list,
            flask_app_obj.config["job_manager_obj"],
        )
        # Two pages, one route. Without ?analysis= this is the strategy's
        # control page: what it is, which contracts it satisfies, what to run,
        # and what has run before. With ?analysis= it is the evidence itself.
        #
        # They were one page, which forced the run controls into a collapsed
        # <details> under the report — so the primary reason to open a strategy
        # was the least reachable thing on it.
        if requested_analysis_str is None:
            latest_vanilla_run_obj = next(
                (
                    view_dict["latest_run"]
                    for view_dict in analyzer_view_list
                    if view_dict["analysis_str"] == "vanilla"
                    and view_dict["latest_run"] is not None
                ),
                None,
            )
            return render_template(
                "strategy_overview.html",
                strategy=strategy_entry_obj,
                analyzer_view_list=analyzer_view_list,
                run_entry_list=run_entry_list,
                latest_vanilla_run=latest_vanilla_run_obj,
                headline_stat_list=_overview_stat_dict_list(latest_vanilla_run_obj),
                ready_count_int=sum(
                    1 for view_dict in analyzer_view_list if view_dict["available_bool"]
                ),
                preset_dict=RUN_PRESET_DICT,
                analysis_available_by_key_dict={
                    view_dict["analysis_str"]: view_dict["available_bool"]
                    for view_dict in analyzer_view_list
                },
                kwarg_aware_analysis_tuple=KWARG_AWARE_ANALYSIS_TUPLE,
                kwarg_blind_analysis_tuple=tuple(
                    analysis_str
                    for analysis_str in SUPPORTED_ANALYSIS_TUPLE
                    if analysis_str not in KWARG_AWARE_ANALYSIS_TUPLE
                ),
            )
        latest_report_run_obj = next(
            (
                view_dict["latest_run"]
                for view_dict in analyzer_view_list
                if view_dict["analysis_str"] == selected_analysis_str
                and view_dict["latest_run"] is not None
                and view_dict["latest_run"].has_report_bool
            ),
            None,
        )
        selected_analyzer_view_dict = next(
            view_dict
            for view_dict in analyzer_view_list
            if view_dict["analysis_str"] == selected_analysis_str
        )
        artifact_report_view_obj = artifact_view.build_artifact_view(
            selected_analysis_str,
            latest_report_run_obj,
            request.args.get("view"),
        )
        return render_template(
            "strategy.html",
            strategy=strategy_entry_obj,
            run_entry_list=run_entry_list,
            latest_report_run=latest_report_run_obj,
            selected_analysis_str=selected_analysis_str,
            selected_analyzer=selected_analyzer_view_dict,
            artifact_report_view=artifact_report_view_obj,
            analysis_workspace=_analysis_workspace_dict(
                selected_analysis_str,
                latest_report_run_obj,
                status_str=selected_analyzer_view_dict["status_str"],
                detail_str=selected_analyzer_view_dict["detail_str"],
            ),
            analyzer_view_list=analyzer_view_list,
            analysis_available_by_key_dict={
                view_dict["analysis_str"]: view_dict["available_bool"]
                for view_dict in analyzer_view_list
            },
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
        if len(submitted_module_list) == 0:
            strategy_entry_list = catalog.list_strategies()
            run_index_obj = runs.build_strategy_run_index(
                strategy_stem_set={entry_obj.stem_str for entry_obj in strategy_entry_list}
            )
            compare_candidate_list = []
            for entry_obj in strategy_entry_list:
                latest_vanilla_run_obj = run_index_obj.latest_vanilla_for(
                    entry_obj.module_import_str,
                    entry_obj.stem_str,
                )
                if (
                    latest_vanilla_run_obj is not None
                    and latest_vanilla_run_obj.has_report_bool
                    and latest_vanilla_run_obj.summary_dict
                ):
                    compare_candidate_list.append(
                        _build_strategy_card_dict(entry_obj, run_index_obj)
                    )
            return render_template(
                "compare_landing.html",
                compare_candidate_list=compare_candidate_list,
                compare_min_int=COMPARE_MIN_INT,
                compare_max_int=COMPARE_MAX_INT,
            )
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
        strategy_entry_list = catalog.list_strategies()
        run_index_obj = runs.build_strategy_run_index(
            strategy_stem_set={entry_obj.stem_str for entry_obj in strategy_entry_list}
        )
        readiness_row_list = [
            _readiness_row_dict(
                entry_obj,
                _analyzer_view_dict_list(
                    entry_obj,
                    run_index_obj.runs_for(entry_obj.module_import_str, entry_obj.stem_str),
                    flask_app_obj.config["job_manager_obj"],
                ),
            )
            for entry_obj in strategy_entry_list
            if entry_obj.is_pm_ready_bool
        ]
        # Attention first. A matrix sorted by name buries the two rows that
        # need work among sixteen that do not, which is the opposite of what
        # this page is for.
        attention_row_list = [row for row in readiness_row_list if not row["is_ready_bool"]]
        ready_row_list = [row for row in readiness_row_list if row["is_ready_bool"]]
        last_audit_str = max(
            (row["last_verified_str"] for row in readiness_row_list if row["last_verified_str"]),
            default="—",
        )
        orphan_view_list = runs.orphan_research_view_list(strategy_entry_list)
        return render_template(
            "research.html",
            orphan_view_list=orphan_view_list,
            readiness_row_list=readiness_row_list,
            attention_row_list=attention_row_list,
            ready_row_list=ready_row_list,
            contract_count_int=len(readiness_row_list) * len(SUPPORTED_ANALYSIS_TUPLE),
            last_audit_str=last_audit_str,
        )

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

    @flask_app_obj.route("/compare-portfolios")
    def compare_portfolios_page_fn() -> str:
        """Books side by side, every column recomputed on their shared window.

        A GET so a comparison can be bookmarked and reopened; the selection
        lives entirely in the query string, like the strategy compare page.
        """
        submitted_path_list = request.args.getlist("p")
        deduped_path_list: list[str] = []
        for rel_path_str in submitted_path_list:
            if rel_path_str not in deduped_path_list:
                deduped_path_list.append(rel_path_str)
        if not (
            portfolio_compare.COMPARE_MIN_INT
            <= len(deduped_path_list)
            <= portfolio_compare.COMPARE_MAX_INT
        ):
            abort(
                400,
                description=(
                    f"Pick {portfolio_compare.COMPARE_MIN_INT}–"
                    f"{portfolio_compare.COMPARE_MAX_INT} books to compare."
                ),
            )
        return render_template(
            "portfolio_compare.html",
            comparison=portfolio_compare.compare_books(deduped_path_list),
            metric_row_tuple=portfolio_compare.METRIC_ROW_TUPLE,
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

    @flask_app_obj.route("/fonts/<file_name_str>")
    def font_fn(file_name_str: str) -> Response:
        """Serve one vendored font face.

        The filename is checked against the declared face list rather than
        sanitised, so this route can only ever return one of three known files
        — no traversal to reason about, and a typo fails as 404 instead of
        reaching the filesystem.
        """
        allowed_file_name_set = {
            face_file_name_str for _, _, face_file_name_str in VENDORED_FONT_FACE_TUPLE
        }
        if file_name_str not in allowed_file_name_set:
            abort(404)
        return send_from_directory(
            VENDORED_FONT_DIR_PATH,
            file_name_str,
            mimetype="font/woff2",
            max_age=60 * 60 * 24 * 365,
        )

    @flask_app_obj.route("/density/<density_name_str>")
    def set_density_fn(density_name_str: str) -> Response:
        """Switch the console's display density and return where you were.

        A GET is appropriate here: this writes one display cookie in the
        operator's own browser and touches no server state, so there is nothing
        for a cross-site request to accomplish beyond resizing their type. The
        referrer is only followed back to our own host — an absolute foreign
        URL would turn this route into an open redirect.
        """
        if density_name_str not in BENCH_DENSITY_LABEL_DICT:
            abort(404)
        referrer_str = request.referrer or ""
        same_origin_bool = bool(referrer_str) and urlparse(referrer_str).netloc == request.host
        response_obj = redirect(referrer_str if same_origin_bool else url_for("index_page_fn"))
        response_obj.set_cookie(
            BENCH_DENSITY_COOKIE_STR,
            density_name_str,
            max_age=BENCH_VARIANT_COOKIE_MAX_AGE_INT,
            samesite="Lax",
            httponly=True,
        )
        return response_obj

    @flask_app_obj.route("/jobs")
    def jobs_page_fn() -> str:
        job_manager = flask_app_obj.config["job_manager_obj"]
        job_list = job_manager.list_jobs()
        # Cancelled and errored jobs count as failed here: from the ledger's
        # point of view they are all "did not produce evidence", and splitting
        # them into their own tile would spend a headline slot on a
        # distinction the detail page already makes.
        return render_template(
            "jobs.html",
            job_list=job_list,
            job_status_count_dict={
                "active": sum(1 for job_obj in job_list if job_obj.is_active_bool),
                "passed": sum(1 for job_obj in job_list if job_obj.status_str == "passed"),
                "failed": sum(
                    1
                    for job_obj in job_list
                    if job_obj.status_str in ("failed", "error", "cancelled")
                ),
                "total": len(job_list),
            },
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
        job_view_dict = _job_view_dict_list(
            [job_obj], flask_app_obj.config["produced_run_cache_dict"]
        )[0]
        log_text_str = job_manager.read_log_text(job_id_str)
        analyzer_step_list = _job_analyzer_step_list(job_obj, log_text_str)
        return render_template(
            "log.html",
            job=job_obj,
            log_text_str=log_text_str,
            produced_run=job_view_dict["produced_run"],
            analyzer_step_list=analyzer_step_list,
            job_headline_str=_job_headline_str(job_obj, analyzer_step_list),
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

        # Reject the whole request if any submitted analysis is unrecognized,
        # rather than silently dropping it and running a different subset.
        analysis_list = request.form.getlist("analysis")
        if not analysis_list:
            abort(400, description="No analysis selected.")
        invalid_analysis_list = [a for a in analysis_list if a not in SUPPORTED_ANALYSIS_TUPLE]
        if invalid_analysis_list:
            abort(400, description=f"Unknown analysis: {', '.join(invalid_analysis_list)}")
        if len(analysis_list) == 1 and not _analysis_available_bool(
            strategy_entry_obj, analysis_list[0]
        ):
            analysis_label_str = ANALYSIS_LABEL_DICT[analysis_list[0]]
            abort(
                400,
                description=f"{analysis_label_str} unavailable — missing analyzer contract.",
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


def _analysis_available_bool(
    strategy_entry_obj: catalog.StrategyEntry,
    analysis_str: str,
) -> bool:
    """Whether the current module exposes the hook required by an analyzer."""
    if analysis_str in {"vanilla", "risk"}:
        return strategy_entry_obj.has_run_variant_bool
    if analysis_str == "capacity":
        return strategy_entry_obj.has_capacity_analysis_bool
    if analysis_str == "timing":
        return strategy_entry_obj.has_timing_analysis_bool
    return (
        strategy_entry_obj.has_run_variant_bool
        and strategy_entry_obj.stem_str in supported_stress_test_strategy_key_list()
    )


def _analysis_tuple_from_command_list(command_list: list[str]) -> tuple[str, ...]:
    analysis_list: list[str] = []
    for part_index_int, part_str in enumerate(command_list[:-1]):
        if (
            part_str == "--analysis"
            and command_list[part_index_int + 1] in SUPPORTED_ANALYSIS_TUPLE
        ):
            analysis_list.append(command_list[part_index_int + 1])
    return tuple(dict.fromkeys(analysis_list))


def _summary_row_by_analysis_dict(log_text_str: str) -> dict[str, dict[str, str]]:
    """Parse the runner's final per-analysis table, never an incidental line.

    Returns status, elapsed seconds and detail per analyzer. The seconds and
    detail columns are optional in the runner's output, so a row missing them
    still parses — a status with no timing is better evidence than no row.
    """
    normalized_log_text_str = log_text_str.replace("\r\n", "\n").replace("\r", "\n")
    summary_marker_str = "Summary\nAnalysis  Status  Seconds  Detail"
    if summary_marker_str not in normalized_log_text_str:
        return {}
    summary_text_str = normalized_log_text_str.rsplit(summary_marker_str, maxsplit=1)[-1]
    row_by_analysis_dict: dict[str, dict[str, str]] = {}
    for analysis_str, status_str, seconds_str, detail_str in re.findall(
        r"^(vanilla|capacity|timing|risk|stress)\s+(PASS|SKIP|FAIL)\b"
        r"[ \t]*([0-9]+\.?[0-9]*)?[ \t]*(.*)$",
        summary_text_str,
        flags=re.MULTILINE,
    ):
        row_by_analysis_dict[analysis_str] = {
            "status_str": status_str,
            "seconds_str": seconds_str,
            "detail_str": detail_str.strip(),
        }
    return row_by_analysis_dict


def _summary_status_by_analysis_dict(log_text_str: str) -> dict[str, str]:
    """Status only, for callers that just need the verdict."""
    return {
        analysis_str: row_dict["status_str"]
        for analysis_str, row_dict in _summary_row_by_analysis_dict(log_text_str).items()
    }


def _latest_job_record_by_analysis_dict(job_manager_obj, strategy_stem_str: str) -> dict:
    """Newest BENCH job evidence for each analyzer on one strategy."""
    if job_manager_obj is None or not hasattr(job_manager_obj, "list_jobs"):
        return {}
    read_log_fn = getattr(job_manager_obj, "read_log_text", None)
    latest_record_by_analysis_dict: dict[str, dict[str, Any]] = {}
    for job_obj in job_manager_obj.list_jobs():
        if job_obj.kind_str != "analysis" or job_obj.target_str != strategy_stem_str:
            continue
        analysis_tuple = _analysis_tuple_from_command_list(list(job_obj.command_list))
        if len(analysis_tuple) == 0:
            continue
        log_text_str = ""
        if callable(read_log_fn):
            try:
                log_text_str = str(read_log_fn(job_obj.job_id_str))
            except OSError:
                log_text_str = ""
        parsed_status_dict = _summary_status_by_analysis_dict(log_text_str)
        if job_obj.is_active_bool:
            evidence_timestamp_str = (
                getattr(job_obj, "started_at_str", None) or job_obj.created_at_str
            )
        else:
            evidence_timestamp_str = (
                getattr(job_obj, "ended_at_str", None) or job_obj.created_at_str
            )
        try:
            evidence_timestamp_float = datetime.fromisoformat(
                evidence_timestamp_str
            ).timestamp()
        except (TypeError, ValueError):
            evidence_timestamp_float = 0.0
        precision_tolerance_float = (
            1.0
            if not job_obj.is_active_bool and "." not in evidence_timestamp_str
            else 0.0
        )
        for analysis_str in analysis_tuple:
            existing_record_dict = latest_record_by_analysis_dict.get(analysis_str)
            if (
                existing_record_dict is not None
                and existing_record_dict["timestamp_float"]
                >= evidence_timestamp_float
            ):
                continue
            if job_obj.is_active_bool:
                status_str = job_obj.status_str.upper()
            elif analysis_str in parsed_status_dict:
                status_str = parsed_status_dict[analysis_str]
            elif len(analysis_tuple) == 1 and job_obj.status_str in {"failed", "error"}:
                status_str = "FAIL"
            else:
                # Exit 0 also covers a runner-level SKIP. Without the summary
                # table there is no per-analyzer evidence to promote to PASS.
                status_str = "NOT RUN"
            latest_record_by_analysis_dict[analysis_str] = {
                "status_str": status_str,
                "timestamp_float": evidence_timestamp_float,
                "precision_tolerance_float": precision_tolerance_float,
                "job": job_obj,
            }
    return latest_record_by_analysis_dict


def _overview_stat_dict_list(run_obj) -> list[dict[str, str]]:
    """Headline figures for the strategy control page.

    Read straight from the saved vanilla summary — nothing is recomputed here.
    A metric the artifact does not carry renders as an em dash rather than a
    zero, because "not measured" and "measured as zero" are different claims.
    """
    if run_obj is None:
        return []
    summary_dict = run_obj.summary_dict
    # *** UI*** These are exactly the figures the saved vanilla summary
    # carries. Volatility, Sortino, turnover and capacity are NOT in the
    # artifact schema, so a strip that showed them would render permanent em
    # dashes and imply the run failed to measure something it never claimed
    # to. Add a tile here only when the artifact starts recording the value.
    spec_tuple = (
        ("Return (CAGR)", _decimal_str(summary_dict.get("ann_return_pct"), suffix_str="%")),
        ("Sharpe", _decimal_str(summary_dict.get("sharpe"))),
        ("Max drawdown", _decimal_str(summary_dict.get("max_drawdown_pct"), suffix_str="%")),
        ("Final equity", _compact_money_str(summary_dict.get("final_equity"))),
        ("Trades", _integer_str(summary_dict.get("trade_count"))),
    )
    return [
        {"label_str": label_str, "value_str": value_str} for label_str, value_str in spec_tuple
    ]


def _job_analyzer_step_list(
    job_obj, log_text_str: str
) -> list[dict[str, Any]]:
    """One step per analyzer this job was asked to run.

    The steps come from the *command*, not from the log, so an analyzer that
    has not started yet is still shown as pending rather than silently missing.
    A job that requested five analyzers and finished three has two unreported
    contracts, and that gap is the thing an operator needs to see.
    """
    requested_analysis_tuple = _analysis_tuple_from_command_list(list(job_obj.command_list))
    summary_row_dict = _summary_row_by_analysis_dict(log_text_str)
    # The runner works through the requested analyzers in order, so the first
    # one with no summary row is the one currently executing.
    running_analysis_str = ""
    if job_obj.is_active_bool:
        running_analysis_str = next(
            (
                analysis_str
                for analysis_str in requested_analysis_tuple
                if analysis_str not in summary_row_dict
            ),
            "",
        )
    step_list: list[dict[str, Any]] = []
    for analysis_str in requested_analysis_tuple:
        row_dict = summary_row_dict.get(analysis_str)
        if row_dict is not None:
            status_str = row_dict["status_str"]
            elapsed_str = f"{float(row_dict['seconds_str']):.0f}s" if row_dict["seconds_str"] else "—"
            detail_str = row_dict["detail_str"] or "—"
        elif analysis_str == running_analysis_str:
            status_str, elapsed_str, detail_str = "RUNNING", job_obj.elapsed_str, "In progress"
        elif job_obj.is_active_bool:
            status_str, elapsed_str, detail_str = "QUEUED", "—", "Not started"
        else:
            # The job ended without reporting this contract at all.
            status_str, elapsed_str, detail_str = "NOT RUN", "—", "No summary row"
        step_list.append(
            {
                "analysis_str": analysis_str,
                "label_str": ANALYSIS_LABEL_DICT[analysis_str],
                "status_str": status_str,
                "status_class_str": status_str.lower().replace(" ", "-"),
                "elapsed_str": elapsed_str,
                "detail_str": detail_str,
            }
        )
    return step_list


def _job_headline_str(job_obj, step_list: list[dict[str, Any]]) -> str:
    """One plain-language sentence about what this job actually did."""
    if not step_list:
        return "This job ran no analyzer contract."
    count_by_status_dict: dict[str, int] = {}
    for step_dict in step_list:
        count_by_status_dict[step_dict["status_str"]] = (
            count_by_status_dict.get(step_dict["status_str"], 0) + 1
        )
    clause_list: list[str] = []
    for status_str, verb_str in (
        ("PASS", "passed"),
        ("FAIL", "failed"),
        ("SKIP", "were skipped"),
    ):
        count_int = count_by_status_dict.get(status_str, 0)
        if count_int:
            clause_list.append(f"{count_int} {verb_str}")
    running_label_str = next(
        (step["label_str"] for step in step_list if step["status_str"] == "RUNNING"), ""
    )
    if running_label_str:
        clause_list.append(f"{running_label_str} is running")
    if not clause_list:
        return "No analyzer has reported yet."
    # A skip is the quiet failure mode this console exists to surface, so it is
    # called out explicitly rather than left to be inferred from the counts.
    suffix_str = (
        "" if count_by_status_dict.get("SKIP") else " No analyzer was skipped."
    )
    return f"{', '.join(clause_list).capitalize()}.{suffix_str}"


def _readiness_row_dict(
    strategy_entry_obj: catalog.StrategyEntry,
    analyzer_view_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """One strategy's readiness across all five analyzer contracts.

    *** UI*** "Ready" requires every analyzer to carry positive evidence —
    SAVED or an explicit PASS. A SKIP is a missing hook, which means the
    contract cannot even be evaluated, and treating that as ready is exactly
    the failure the readiness page exists to prevent. NOT RUN and FAIL are
    likewise not ready. This mirrors the runner's own rule that exit 0 with a
    SKIP does not satisfy readiness.
    """
    ready_status_set = {"PASS", "SAVED"}
    blocking_view_list = [
        view_dict
        for view_dict in analyzer_view_list
        if view_dict["status_str"] not in ready_status_set
    ]
    # The most recent moment any contract on this strategy was actually
    # verified. Blank when nothing has ever run.
    verified_timestamp_list = [
        view_dict["latest_run"].display_timestamp_str
        for view_dict in analyzer_view_list
        if view_dict["latest_run"] is not None
    ]
    if blocking_view_list:
        first_blocking_dict = blocking_view_list[0]
        if first_blocking_dict["status_str"] == "SKIP":
            detail_str = f"missing {first_blocking_dict['label_str'].lower()} hook"
        elif first_blocking_dict["status_str"] == "FAIL":
            detail_str = f"{first_blocking_dict['label_str'].lower()} failed"
        else:
            detail_str = f"no {first_blocking_dict['label_str'].lower()} evidence"
        if len(blocking_view_list) > 1:
            detail_str = f"{detail_str} (+{len(blocking_view_list) - 1} more)"
    else:
        detail_str = "—"
    return {
        "strategy": strategy_entry_obj,
        "analyzer_view_list": analyzer_view_list,
        "is_ready_bool": not blocking_view_list,
        "blocking_count_int": len(blocking_view_list),
        "detail_str": detail_str,
        "last_verified_str": max(verified_timestamp_list, default=""),
    }


def _analyzer_view_dict_list(
    strategy_entry_obj: catalog.StrategyEntry,
    run_entry_list: list[runs.RunEntry],
    job_manager_obj=None,
) -> list[dict[str, Any]]:
    """Five analyzer cells backed by hooks, artifacts, and BENCH job evidence."""
    latest_run_by_analysis_dict: dict[str, runs.RunEntry] = {}
    for analysis_str, analysis_dir_str in ANALYSIS_DIR_BY_ANALYSIS_DICT.items():
        matching_run_list = [
            run_obj
            for run_obj in run_entry_list
            if run_obj.analysis_dir_str == analysis_dir_str and run_obj.has_report_bool
        ]
        if analysis_str == "capacity":
            current_capacity_run_list = [
                run_obj for run_obj in matching_run_list if not run_obj.is_legacy_capacity_bool
            ]
            if current_capacity_run_list:
                matching_run_list = current_capacity_run_list
        if matching_run_list:
            latest_run_by_analysis_dict[analysis_str] = matching_run_list[0]

    latest_job_record_dict = _latest_job_record_by_analysis_dict(
        job_manager_obj,
        strategy_entry_obj.stem_str,
    )
    analyzer_view_list: list[dict[str, Any]] = []
    for analysis_str in SUPPORTED_ANALYSIS_TUPLE:
        available_bool = _analysis_available_bool(strategy_entry_obj, analysis_str)
        latest_run_obj = latest_run_by_analysis_dict.get(analysis_str)
        artifact_timestamp_float = (
            latest_run_obj.effective_activity_timestamp_float if latest_run_obj is not None else 0.0
        )
        job_record_dict = latest_job_record_dict.get(analysis_str)
        if not available_bool:
            status_str = "SKIP"
            if analysis_str in {"vanilla", "risk"}:
                missing_hook_str = "run_variant hook"
            elif analysis_str == "capacity":
                missing_hook_str = "capacity hook"
            elif analysis_str == "stress":
                missing_hook_str = (
                    "run_variant hook"
                    if not strategy_entry_obj.has_run_variant_bool
                    else "stress registry"
                )
            else:
                missing_hook_str = "timing hook"
            detail_str = (
                f"{ANALYSIS_LABEL_DICT[analysis_str]} unavailable — missing {missing_hook_str}"
            )
        elif (
            job_record_dict is not None
            and (
                job_record_dict["timestamp_float"]
                + job_record_dict.get("precision_tolerance_float", 0.0)
                >= artifact_timestamp_float
            )
            and (
                job_record_dict["status_str"] != "NOT RUN"
                or latest_run_obj is None
            )
        ):
            status_str = job_record_dict["status_str"]
            detail_str = "Latest BENCH job"
        elif latest_run_obj is not None:
            # A report proves saved historical evidence, not parity with the
            # strategy source currently on disk. Only an explicit BENCH job
            # summary may say PASS until artifacts record a code fingerprint.
            status_str = "SAVED"
            detail_str = latest_run_obj.display_timestamp_str
        else:
            status_str = "NOT RUN"
            detail_str = "No saved evidence"
        analyzer_view_list.append(
            {
                "analysis_str": analysis_str,
                "code_str": analysis_str[0].upper(),
                "label_str": ANALYSIS_LABEL_DICT[analysis_str],
                "status_str": status_str,
                "status_class_str": status_str.lower().replace(" ", "-"),
                "detail_str": detail_str,
                "available_bool": available_bool,
                # Does a readable saved report exist for this analyzer?
                #
                # This is the catalog's filled-versus-hollow mark, so the
                # definition has to be narrow: a saved artifact, or a BENCH job
                # that explicitly passed. SKIP is not evidence (the hook is
                # missing), FAIL is not evidence (the run did not produce a
                # usable report), and NOT RUN is obviously not. Widening this
                # would make an unrun strategy look identical to a tested one,
                # which is the exact claim this console exists to keep honest.
                "has_evidence_bool": latest_run_obj is not None or status_str == "PASS",
                "latest_run": latest_run_obj,
            }
        )
    return analyzer_view_list


def _number_float(value_obj: object) -> float | None:
    if isinstance(value_obj, bool) or not isinstance(value_obj, (int, float)):
        return None
    return float(value_obj)


def _percent_str(value_obj: object, *, decimal_fraction_bool: bool = False) -> str:
    value_float = _number_float(value_obj)
    if value_float is None:
        return "—"
    if decimal_fraction_bool:
        value_float *= 100.0
    return f"{value_float:,.2f}%"


def _money_str(value_obj: object) -> str:
    value_float = _number_float(value_obj)
    return "—" if value_float is None else f"${value_float:,.0f}"


def _integer_str(value_obj: object) -> str:
    value_float = _number_float(value_obj)
    return "—" if value_float is None else f"{int(value_float):,}"


def _decimal_str(
    value_obj: object, *, suffix_str: str = "", signed_bool: bool = False
) -> str:
    value_float = _number_float(value_obj)
    if value_float is None:
        return "—"
    return f"{value_float:{'+' if signed_bool else ''}.2f}{suffix_str}"


def _compact_money_str(value_obj: object) -> str:
    value_float = _number_float(value_obj)
    if value_float is None:
        return "—"
    for divisor_float, suffix_str in (
        (1_000_000_000.0, "B"),
        (1_000_000.0, "M"),
        (1_000.0, "K"),
    ):
        if abs(value_float) >= divisor_float:
            compact_float = value_float / divisor_float
            decimal_count_int = 0 if compact_float.is_integer() else 1
            return f"${compact_float:.{decimal_count_int}f}{suffix_str}"
    return f"${value_float:,.0f}"


def _nested_value_obj(source_dict: dict, *key_str_tuple: str) -> object:
    value_obj: object = source_dict
    for key_str in key_str_tuple:
        if not isinstance(value_obj, dict):
            return None
        value_obj = value_obj.get(key_str)
    return value_obj


def _analysis_workspace_dict(
    analysis_str: str,
    latest_run_obj: runs.RunEntry | None,
    *,
    status_str: str = "NOT RUN",
    detail_str: str | None = None,
) -> dict[str, Any]:
    """Mockup-faithful analyzer header, using saved evidence without recomputation."""
    outline_by_analysis_dict = {
        "vanilla": ("Equity", "Monthly returns", "Composition", "Statistics", "Trades"),
        "capacity": ("Capacity curve", "Decision bands", "Liquidity", "Orders", "Assumptions"),
        "timing": ("Return matrix", "Sharpe matrix", "Drawdown", "CVaR", "Cell detail"),
        "risk": ("Read first", "Return distribution", "Monte Carlo paths", "Metrics", "Horizon odds"),
        "stress": ("Scenario summary", "Event paths", "Drawdowns", "Entry exposure", "Recovery"),
    }
    workspace_dict: dict[str, Any] = {
        "eyebrow_str": f"{ANALYSIS_LABEL_DICT[analysis_str].upper()} ANALYSIS",
        "summary_str": detail_str or "No saved report exists for this analyzer yet.",
        "meta_list": (
            {"label_str": "EVIDENCE", "value_str": "Not saved"},
            {"label_str": "STATUS", "value_str": status_str},
            {"label_str": "ANALYZER", "value_str": ANALYSIS_LABEL_DICT[analysis_str]},
            {"label_str": "RUN", "value_str": "—"},
        ),
        "stat_list": (),
        "outline_list": outline_by_analysis_dict[analysis_str],
    }
    if latest_run_obj is None:
        return workspace_dict

    summary_dict = latest_run_obj.summary_dict
    metadata_dict = latest_run_obj.metadata_dict
    parameter_dict = latest_run_obj.parameter_dict
    run_timestamp_str = latest_run_obj.display_timestamp_str

    if analysis_str == "vanilla":
        capital_float = _number_float(parameter_dict.get("capital"))
        final_equity_float = _number_float(summary_dict.get("final_equity"))
        sharpe_str = _decimal_str(summary_dict.get("sharpe"))
        workspace_dict.update(
            {
                "eyebrow_str": "VANILLA BACKTEST",
                "summary_str": (
                    f"Compounded at {_percent_str(summary_dict.get('ann_return_pct'))} "
                    f"with a {sharpe_str} Sharpe; "
                    f"the worst drawdown was {_percent_str(summary_dict.get('max_drawdown_pct'))}."
                ),
                "meta_list": (
                    {"label_str": "PERIOD", "value_str": latest_run_obj.backtest_window_str or "Not recorded"},
                    {
                        "label_str": "CAPITAL",
                        "value_str": f"{_money_str(capital_float)} → {_money_str(final_equity_float)}",
                    },
                    {"label_str": "EXECUTION", "value_str": "Recorded in strategy implementation"},
                    {"label_str": "RUN", "value_str": run_timestamp_str},
                ),
                "stat_list": (
                    {"label_str": "CAGR", "value_str": _percent_str(summary_dict.get("ann_return_pct"))},
                    {"label_str": "SHARPE", "value_str": sharpe_str},
                    {"label_str": "MAX DD", "value_str": _percent_str(summary_dict.get("max_drawdown_pct")), "tone_str": "negative"},
                    {"label_str": "TRADES", "value_str": _integer_str(summary_dict.get("trade_count"))},
                    {"label_str": "FINAL", "value_str": _money_str(final_equity_float)},
                ),
            }
        )
    elif analysis_str == "capacity":
        if "recommended_capacity_float" not in summary_dict:
            recommended_capacity_str = "—"
        elif summary_dict["recommended_capacity_float"] is None:
            recommended_capacity_str = "NOT CLEARED"
        else:
            recommended_capacity_str = _money_str(
                summary_dict["recommended_capacity_float"]
            )
        break_even_str = str(summary_dict.get("break_even_capacity_bracket_str") or "—")
        aum_grid_obj = summary_dict.get("aum_grid_list")
        if isinstance(aum_grid_obj, list) and aum_grid_obj:
            aum_grid_str = (
                f"{_compact_money_str(aum_grid_obj[0])} → "
                f"{_compact_money_str(aum_grid_obj[-1])}"
            )
        else:
            aum_grid_str = "See saved artifact"
        workspace_dict.update(
            {
                "eyebrow_str": "CAPACITY ANALYSIS",
                "summary_str": (
                    f"Recommended capacity: {recommended_capacity_str}. "
                    f"The recent-window break-even bracket was {break_even_str}."
                ),
                "meta_list": (
                    {
                        "label_str": "PERIOD",
                        "value_str": (
                            f"{summary_dict.get('actual_start_date_str', '—')} → "
                            f"{summary_dict.get('actual_end_date_str', '—')}"
                        ),
                    },
                    {"label_str": "AUM GRID", "value_str": aum_grid_str},
                    {"label_str": "EXECUTION", "value_str": str(summary_dict.get("execution_policy_str") or "—")},
                    {"label_str": "RUN", "value_str": run_timestamp_str},
                ),
                "stat_list": (
                    {"label_str": "RECOMMENDED", "value_str": recommended_capacity_str},
                    {"label_str": "BREAK-EVEN", "value_str": break_even_str},
                    {"label_str": "ORDERS", "value_str": _integer_str(summary_dict.get("assessed_order_count_int"))},
                    {"label_str": "EXTRAPOLATED", "value_str": _percent_str(summary_dict.get("model_extrapolation_share_float"), decimal_fraction_bool=True)},
                    {"label_str": "PROFILE", "value_str": str(summary_dict.get("impact_profile_str") or "—")},
                ),
            }
        )
    elif analysis_str == "timing":
        default_entry_str = str(metadata_dict.get("default_entry_timing") or "—")
        default_exit_str = str(metadata_dict.get("default_exit_timing") or "—")
        sharpe_str = _decimal_str(summary_dict.get("sharpe"))
        entry_timing_obj = parameter_dict.get("entry_timing_labels")
        exit_timing_obj = parameter_dict.get("exit_timing_labels")
        if isinstance(entry_timing_obj, list) and isinstance(exit_timing_obj, list):
            matrix_str = (
                f"{len(entry_timing_obj)} entry timings × "
                f"{len(exit_timing_obj)} exit timings"
            )
        else:
            matrix_str = "See saved artifact"
        workspace_dict.update(
            {
                "eyebrow_str": "EXECUTION TIMING ANALYSIS",
                "summary_str": (
                    f"The default {default_entry_str} / {default_exit_str} cell produced "
                    f"a {sharpe_str} Sharpe and is labeled "
                    f"{summary_dict.get('risk_label', 'Not recorded')}."
                ),
                "meta_list": (
                    {"label_str": "MATRIX", "value_str": matrix_str},
                    {"label_str": "DEFAULT ENTRY", "value_str": default_entry_str},
                    {"label_str": "DEFAULT EXIT", "value_str": default_exit_str},
                    {"label_str": "RUN", "value_str": run_timestamp_str},
                ),
                "stat_list": (
                    {"label_str": "CAGR", "value_str": _percent_str(summary_dict.get("ann_return_pct"))},
                    {"label_str": "SHARPE", "value_str": sharpe_str},
                    {"label_str": "MAX DD", "value_str": _percent_str(summary_dict.get("max_drawdown_pct")), "tone_str": "negative"},
                    {"label_str": "CVaR 5%", "value_str": _percent_str(summary_dict.get("cvar_5_pct")), "tone_str": "negative"},
                    {"label_str": "LABEL", "value_str": str(summary_dict.get("risk_label") or "—")},
                ),
            }
        )
    elif analysis_str == "risk":
        simulation_count_obj = summary_dict.get("simulation_count_int")
        mean_block_length_obj = summary_dict.get("primary_mean_block_length_int")
        max_drawdown_p05_obj = _nested_value_obj(
            summary_dict, "primary_intervals", "max_drawdown_float", "p05_float"
        )
        sharpe_observed_obj = _nested_value_obj(
            summary_dict, "primary_intervals", "sharpe_float", "observed_value_float"
        )
        underwater_12m_obj = _nested_value_obj(
            summary_dict, "primary_time_underwater_breach_probabilities", "underwater_ge_12m"
        )
        one_year_p05_obj = _nested_value_obj(
            summary_dict,
            "investor_summary",
            "headline_metric_dict",
            "modeled_1y_terminal_p05_block_specific_float",
        )
        workspace_dict.update(
            {
                "eyebrow_str": "RISK ANALYSIS",
                "summary_str": (
                    f"{_integer_str(simulation_count_obj)} historically conditioned bootstrap paths "
                    f"calibrate uncertainty around the saved return record; they are not a forecast."
                ),
                "meta_list": (
                    {
                        "label_str": "PERIOD",
                        "value_str": f"{summary_dict.get('start_date_str', '—')} → {summary_dict.get('end_date_str', '—')}",
                    },
                    {"label_str": "BOOTSTRAP", "value_str": f"{_integer_str(simulation_count_obj)} paths · block {mean_block_length_obj or '—'}d"},
                    {"label_str": "CONFIDENCE", "value_str": _percent_str(summary_dict.get("confidence_level_float"), decimal_fraction_bool=True)},
                    {"label_str": "RUN", "value_str": run_timestamp_str},
                ),
                "stat_list": (
                    {"label_str": "OBS. SHARPE", "value_str": _decimal_str(sharpe_observed_obj)},
                    {"label_str": "DD P05", "value_str": _percent_str(max_drawdown_p05_obj, decimal_fraction_bool=True), "tone_str": "negative"},
                    {"label_str": "12M+ UNDERWATER", "value_str": _percent_str(underwater_12m_obj, decimal_fraction_bool=True), "tone_str": "negative"},
                    {"label_str": "1Y TERMINAL P05", "value_str": _percent_str(one_year_p05_obj, decimal_fraction_bool=True)},
                    {"label_str": "OBSERVATIONS", "value_str": _integer_str(summary_dict.get("return_count_int"))},
                ),
            }
        )
    else:
        capital_obj = parameter_dict.get("capital")
        launch_offset_obj = parameter_dict.get("launch_offsets")
        if not isinstance(launch_offset_obj, list):
            launch_offset_obj = metadata_dict.get("launch_offsets")
        crisis_count_obj = metadata_dict.get("configured_crisis_count")
        crisis_count_str = _integer_str(crisis_count_obj)
        offset_count_str = (
            _integer_str(len(launch_offset_obj))
            if isinstance(launch_offset_obj, list)
            else "—"
        )
        scenario_contract_str = (
            f"{crisis_count_str} crises · {offset_count_str} launch offsets"
        )
        workspace_dict.update(
            {
                "eyebrow_str": "HISTORICAL STRESS TEST",
                "summary_str": (
                    f"Evaluated {_integer_str(summary_dict.get('scenario_count_int'))} pre-crisis launch scenarios; "
                    f"the worst event return was {_percent_str(summary_dict.get('worst_event_return_pct_float'))}."
                ),
                "meta_list": (
                    {"label_str": "SCENARIOS", "value_str": scenario_contract_str},
                    {"label_str": "CAPITAL", "value_str": _money_str(capital_obj)},
                    {"label_str": "MODEL", "value_str": "Historical pre-crisis launch"},
                    {"label_str": "RUN", "value_str": run_timestamp_str},
                ),
                "stat_list": (
                    {"label_str": "SCENARIOS", "value_str": _integer_str(summary_dict.get("scenario_count_int"))},
                    {"label_str": "WORST RETURN", "value_str": _percent_str(summary_dict.get("worst_event_return_pct_float")), "tone_str": "negative"},
                    {"label_str": "WORST DD", "value_str": _percent_str(summary_dict.get("worst_event_max_drawdown_pct_float")), "tone_str": "negative"},
                    {"label_str": "UNRECOVERED", "value_str": _integer_str(summary_dict.get("unrecovered_scenario_count_int"))},
                    {"label_str": "MAX GROSS", "value_str": _decimal_str(summary_dict.get("max_entering_gross_exposure_float"), suffix_str="×")},
                ),
            }
        )
    return workspace_dict


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
        "analyzer_view_list": _analyzer_view_dict_list(
            strategy_entry_obj,
            run_index_obj.runs_for(
                strategy_entry_obj.module_import_str,
                strategy_entry_obj.stem_str,
            ),
        ),
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
