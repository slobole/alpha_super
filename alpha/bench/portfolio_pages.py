"""BENCH portfolio management routes. All mutations require a reviewed token."""

from pathlib import Path
import secrets

from flask import abort, redirect, render_template, request, url_for
from itsdangerous import BadSignature, URLSafeTimedSerializer

from alpha.bench import portfolio_builder, portfolio_config, portfolio_overview


def register_routes(app_obj, csrf_failure_fn):
    # CSRF tokens are public form values, so they cannot sign reviewed payloads.
    serializer_obj = URLSafeTimedSerializer(secrets.token_hex(32), salt="portfolio-review-v1")

    def assert_idle(rel_path_str):
        for job_obj in app_obj.config["job_manager_obj"].list_jobs():
            if job_obj.kind_str == "portfolio" and job_obj.target_str == Path(rel_path_str).stem and job_obj.status_str in {"queued", "running", "unknown"}:
                raise ValueError("This portfolio has an active or unresolved job. Resolve it before changing or deleting its config.")

    @app_obj.route("/portfolios/manage/<action_str>", methods=["GET", "POST"])
    def portfolio_manage_page_fn(action_str):
        if action_str not in {"edit", "clone", "delete"}:
            abort(404)
        rel_path_str = request.values.get("config", "")
        try:
            if action_str == "delete":
                config_path, revision_str = portfolio_config.delete_source_tuple(rel_path_str)
                config_dict = {"name": config_path.stem}
            else:
                config_path, config_dict, revision_str = portfolio_config.read_config_tuple(rel_path_str)
            if request.method == "POST":
                failure_obj = csrf_failure_fn()
                if failure_obj is not None:
                    return failure_obj
                if request.form.get("revision", "") != revision_str:
                    raise ValueError("This config changed after you opened it. Reload before saving or deleting.")
                if action_str != "clone":
                    assert_idle(rel_path_str)
                proposed_dict = config_dict if action_str == "delete" else portfolio_config.edited_config_dict(config_dict, request.form)
                filename_str = request.form.get("filename", config_path.name)
                target_path = portfolio_builder.resolve_write_path(filename_str) if action_str == "clone" else config_path
                if action_str != "delete":
                    portfolio_config.validate_config(proposed_dict, target_path)
                if action_str == "clone" and target_path.exists():
                    raise ValueError("That filename already exists. Choose a new filename.")
                payload_dict = {"source": rel_path_str, "revision": revision_str, "config": proposed_dict, "mode": action_str, "filename": target_path.name}
                return render_template(
                    "portfolio_edit_review.html", action_str=action_str,
                    config_rel_path_str=rel_path_str, filename_str=target_path.name,
                    config_dict=proposed_dict, diff_str=portfolio_config.diff_text_str(config_dict, proposed_dict),
                    review_token_str=serializer_obj.dumps(payload_dict),
                )
            context_dict = portfolio_config.editor_context_dict(config_dict)
            if action_str == "clone":
                context_dict["name_str"] += " copy"
            from alpha.engine.portfolio_manager import SUPPORTED_STRATEGY_IMPORT_TUPLE
            return render_template(
                "portfolio_edit.html", **context_dict, action_str=action_str,
                config_rel_path_str=rel_path_str, revision_str=revision_str,
                filename_str=f"{config_path.stem}_copy.yaml" if action_str == "clone" else config_path.name,
                strategy_import_tuple=SUPPORTED_STRATEGY_IMPORT_TUPLE,
            )
        except (ValueError, OSError, TypeError, KeyError, IndexError) as exception_obj:
            abort(400, description=str(exception_obj))

    @app_obj.route("/api/portfolios/save-reviewed", methods=["POST"])
    def portfolio_save_reviewed_api_fn():
        failure_obj = csrf_failure_fn()
        if failure_obj is not None:
            return failure_obj
        try:
            payload_dict = serializer_obj.loads(request.form.get("review_token", ""), max_age=3600)
            with portfolio_config.MUTATION_LOCK:
                if payload_dict["mode"] != "clone":
                    assert_idle(payload_dict["source"])
                if payload_dict["mode"] == "delete":
                    config_path, revision_str = portfolio_config.delete_source_tuple(payload_dict["source"])
                    if revision_str != payload_dict["revision"]:
                        raise ValueError("This config changed after review. Review the deletion again.")
                    config_path.unlink()
                    return redirect(url_for("portfolios_page_fn"))
                saved_path = portfolio_config.write_reviewed_config(payload_dict)
            return redirect(url_for("portfolio_detail_page_fn", config=f"portfolios/{saved_path.name}"))
        except BadSignature:
            abort(400, description="The review expired or was modified. Review the changes again.")
        except (ValueError, OSError, TypeError, KeyError) as exception_obj:
            abort(409, description=str(exception_obj))

    @app_obj.route("/portfolios/detail")
    def portfolio_detail_page_fn():
        from alpha.bench import portfolio_detail

        rel_path_str = request.args.get("config", "")
        overview_obj = next((item_obj for item_obj in portfolio_overview.list_portfolio_overviews() if item_obj.portfolio.rel_path_str == rel_path_str), None)
        if overview_obj is None:
            abort(404)
        try:
            config_path, config_dict, _revision_str = portfolio_config.read_config_tuple(rel_path_str)
        except (ValueError, OSError) as exception_obj:
            abort(400, description=str(exception_obj))
        detail_dict = portfolio_detail.detail_dict(overview_obj)
        return render_template(
            "portfolio_detail.html", overview=overview_obj, config_dict=config_dict,
            rebalance_label_str=portfolio_config.rebalance_label_str(config_dict.get("rebalance")),
            parent_path_str=portfolio_config.parent_path_str(config_path),
            **detail_dict,
        )
