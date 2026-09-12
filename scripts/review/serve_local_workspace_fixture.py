"""Offline QA only: normal CLI with production-format files in a temp folder."""

import argparse
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from flask import render_template_string

ROOT_PATH_OBJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH_OBJ))
sys.path.insert(0, str(ROOT_PATH_OBJ / "tests"))

from test_dashboard_local_workspace import build_fixture_app, file_snapshot_dict
from alpha.live.dashboard_v3 import __main__ as launcher_module
from alpha.live.dashboard_v3.client_charts import nav_chart_dict
from alpha.live.dashboard_v3.client_presentation import portfolio_allocation_dict


def main():
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--port", type=int, default=18766)
    parser_obj.add_argument("--expanded", action="store_true", help="Synthetic expanded financial source; no real accounts")
    args_obj = parser_obj.parse_args()
    with TemporaryDirectory(prefix="alpha-local-workspace-qa-", ignore_cleanup_errors=True) as directory_str, pytest.MonkeyPatch.context() as patch_obj:
        root_path_obj = Path(directory_str)
        app_obj = build_fixture_app(root_path_obj, patch_obj, new_pod_bool=not args_obj.expanded, expanded_bool=args_obj.expanded)
        initial_dict = file_snapshot_dict(root_path_obj)
        network_attempt_list = []

        def forbidden_connect_fn(*arg_tuple, **kwarg_dict):
            network_attempt_list.append(True)
            raise AssertionError("The saved-evidence QA server must not initiate network connections.")

        patch_obj.setattr(socket.socket, "connect", forbidden_connect_fn)
        patch_obj.setattr("alpha.live.dashboard_v3.data.load_recent_trace_event_dict_list", lambda **kwargs: [])

        @app_obj.get("/__fixture_integrity")
        def fixture_integrity_fn():
            return {"unchanged_bool": file_snapshot_dict(root_path_obj) == initial_dict,
                "network_attempt_count_int": len(network_attempt_list)}

        @app_obj.get("/__fixture_chart_gap")
        def fixture_chart_gap_fn():
            # Isolated observations must remain visible without joining missing days.
            chart_dict = nav_chart_dict([
                {"market_date_str": f"2026-09-0{index_int + 1}", "nav_float": value_float}
                for index_int, value_float in enumerate((100, None, 120))
            ])
            return render_template_string("""<!doctype html><html><head>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" href="/static/custom.css">
                <link rel="stylesheet" href="/static/client_terminal.css"></head>
                <body class="client-workspace"><main class="client-content">
                {% from '_client_chart.html' import financial_chart %}
                {{ financial_chart(chart_dict, 'Synthetic gap regression') }}
                </main></body></html>""", chart_dict=chart_dict)

        @app_obj.get("/__fixture_cash_difference")
        def fixture_cash_difference_fn():
            # Reproduce real-shaped EOD/Flex differences using anonymous accounts.
            account_list = [{"account_route": route_str, "pod_id": route_str, "display_name": label_str}
                for route_str, label_str in [("A", "TAA"), ("B", "NDX")]]
            date_str = "2026-09-11"
            flex_nav_list = [20603.785969295, 12111.235842935]
            report_dict = {"valuation_account_list": account_list, "closing_date_str": date_str,
                "closing_nav_float": sum(flex_nav_list)}
            snapshot_obj = SimpleNamespace(row_tuple=tuple(SimpleNamespace(account_route_str=account_dict["account_route"],
                market_date_str=date_str, closing_nav_decimal=nav_float) for account_dict, nav_float in zip(account_list, flex_nav_list)))
            cash_list = [{"account_route_str": route_str, "pod_id_str": route_str, "market_date_str": date_str,
                "equity_float": equity_float, "cash_float": cash_float}
                for route_str, equity_float, cash_float in [("A", 20608.33, 287.08), ("B", 12104.40, 1969.23)]]
            allocation_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
            return render_template_string("""<!doctype html><html><head>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" href="/static/custom.css">
                <link rel="stylesheet" href="/static/client_terminal.css"></head>
                <body class="client-workspace"><main class="client-content" style="max-width:640px">
                {% from '_client_allocation.html' import allocation_chart %}
                <section class="ops-panel"><h2>Portfolio allocation</h2>
                {{ allocation_chart(allocation_dict, 'Portfolio allocation') }}</section>
                </main></body></html>""", allocation_dict=allocation_dict)

        # Exercise the normal (non-demo) launcher, substituting only isolated
        # provider paths. No real config.env, NDU, broker or external data.
        patch_obj.setattr(launcher_module, "create_app", lambda **kwargs: app_obj)
        patch_obj.setattr(sys, "argv", ["dashboard_v3", "--skip-env-file", "--port", str(args_obj.port)])
        launcher_module.main()


if __name__ == "__main__":
    main()
